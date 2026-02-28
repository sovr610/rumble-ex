# Spike Decoders: From Spike Trains to Differentiable Outputs

## 1. Decoder Contract

Every spike decoder in the brain-ai system accepts a `SpikeBatch` tensor of shape `(B, T, N)` and an optional membrane potential tensor of the same shape. Every decoder returns a `DecoderOutput` dataclass. This contract is the decoding-side counterpart to the encoder contract: it guarantees that any decoder can be swapped into the training pipeline without modifying loss computation, metric logging, or downstream modules.

The batch-first `(B, T, N)` layout is mandatory. The existing `SpikeDecoder` in `brain_ai/core/encoding.py` (lines 196-242) uses `(T, B, D)` with `dim=0` for time reduction. All new decoders adopt `(B, T, N)` to match the canonical layout established across the spiking core and encoder suite. The migration path from the old convention is described in Section 8.

```python
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import torch
from torch import Tensor


@dataclass
class DecoderOutput:
    logits_proxy: Tensor              # (B, N) or (B, num_groups) float -- differentiable, feeds loss
    prediction: Tensor                # (B,) int64 -- argmax class or discretized regression value
    confidence: Tensor                # (B,) float -- scalar confidence per sample
    aux: Dict[str, Any] = field(default_factory=dict)
```

**Field semantics:**

| Field | Shape | Dtype | Purpose |
|---|---|---|---|
| `logits_proxy` | `(B, N)` or `(B, G)` | float32 | Differentiable signal passed to `cross_entropy` or MSE. Must retain the computational graph for backprop through surrogate gradients. |
| `prediction` | `(B,)` | int64 | Hard prediction. Detached from the graph. Used for accuracy metrics only. |
| `confidence` | `(B,)` | float32 | Per-sample confidence score in `[0, 1]`. Used for dual-process routing (System 1/2 threshold) and logging. |
| `aux` | dict | mixed | Diagnostics: `count_histogram`, `first_spike_times`, `margin`, `raw_counts`. Never consumed by the forward path. |

**Universal parameters** shared by all decoder implementations:

- `temperature` (float, default 1.0): Scale factor applied to logits_proxy before softmax. Lower temperature sharpens predictions; higher temperature smooths them.
- `eps` (float, default 1e-7): Stability constant added before division or log operations.

---

## 2. Rate / Spike-Count Decoding

Rate decoding is the simplest and most robust decoder. Sum spikes over the time axis to produce a count per output neuron, then treat counts as unnormalized logits.

**Core computation:**

```python
def rate_decode(
    spikes: Tensor,        # (B, T, N) float {0, 1}
    temperature: float = 1.0,
    eps: float = 1e-7,
    normalize_by_T: bool = False,
) -> DecoderOutput:
    B, T, N = spikes.shape

    # Accumulate counts in fp32 to prevent underflow under AMP.
    counts = spikes.float().sum(dim=1)          # (B, N) fp32

    # Optional rate normalization: divide by T for firing-rate interpretation.
    if normalize_by_T:
        rate = counts / T                       # (B, N)
        logits_proxy = rate / temperature
    else:
        logits_proxy = counts / temperature     # (B, N)

    # Hard prediction: argmax over neurons.
    prediction = counts.argmax(dim=-1)          # (B,) int64

    # Confidence: margin between top-2 softmax probabilities.
    probs = torch.softmax(logits_proxy.float(), dim=-1)  # fp32 softmax
    top2 = probs.topk(2, dim=-1).values                  # (B, 2)
    margin = (top2[:, 0] - top2[:, 1]).clamp(min=0.0)    # (B,)
    confidence = margin

    return DecoderOutput(
        logits_proxy=logits_proxy,
        prediction=prediction,
        confidence=confidence,
        aux={
            "raw_counts": counts.detach(),
            "count_histogram": counts.detach().mean(dim=0),  # (N,)
            "margin": margin.detach(),
        },
    )
```

**Classification mode:** Pass `logits_proxy` directly to `F.cross_entropy(logits_proxy, targets)`. The loss function applies its own log-softmax internally.

**Regression mode:** Treat `logits_proxy` as a weighted sum input. Compute `output = (logits_proxy * centers).sum(dim=-1)` where `centers` are the population decoding centers (see Section 4). Alternatively, use `logits_proxy.argmax(dim=-1)` mapped through a lookup table.

**AMP hardening:** The `.float()` cast on line `counts = spikes.float().sum(dim=1)` is critical. Under `torch.cuda.amp.autocast`, spike tensors may be float16. Summing T=50 binary float16 values across the time axis risks saturation for neurons with high firing rates (float16 max is 65504, which is safe for T<=50, but intermediate accumulation precision matters for gradient flow). Always accumulate in fp32.

**Migration from existing rate method:** The current `SpikeDecoder.forward` with `method="rate"` computes `spikes.sum(dim=0) / spikes.shape[0]` using `dim=0` (time-first). Replace with `spikes.sum(dim=1)` (batch-first) and wrap in the `DecoderOutput` dataclass. The normalization-by-T behavior maps to `normalize_by_T=True`.

---

## 3. First-Spike Decoding (Latency / TTFS)

First-spike (time-to-first-spike) decoding treats the earliest spike as the strongest signal. The neuron that fires first wins. This exploits temporal coding: important features are encoded by spike latency rather than spike count.

**Core computation:**

```python
def first_spike_decode(
    spikes: Tensor,        # (B, T, N) float {0, 1}
    temperature: float = 1.0,
    eps: float = 1e-7,
) -> DecoderOutput:
    B, T, N = spikes.shape

    # Vectorized first-spike extraction via cumsum + argmax.
    # cumsum along time: first nonzero position is where cumsum first becomes 1.
    cumsum = spikes.cumsum(dim=1)               # (B, T, N)
    # Mask: True at and after first spike.
    has_spiked = (cumsum >= 1.0)                # (B, T, N) bool

    # argmax on bool tensor returns index of first True.
    # For neurons that never spike, cumsum is all-zero, argmax returns 0 (wrong).
    t_first = has_spiked.float().argmax(dim=1)  # (B, N) -- first spike timestep

    # Detect no-spike neurons: if cumsum[:, -1, n] == 0, neuron n never fired.
    ever_spiked = (cumsum[:, -1, :] > 0)        # (B, N) bool
    # Assign T (latest possible time) to neurons that never spiked.
    t_first = torch.where(ever_spiked, t_first, torch.full_like(t_first, T))

    # logits_proxy: negate time so earlier spike = higher logit.
    # Scale by 1/T to normalize to [−1, 0] range, then apply temperature.
    logits_proxy = (-t_first.float() / T) / temperature   # (B, N)

    # Prediction: neuron with earliest spike (smallest t_first).
    prediction = t_first.argmin(dim=-1)         # (B,) int64

    # Confidence: margin between first and second earliest spike times.
    sorted_times = t_first.sort(dim=-1).values  # (B, N) ascending
    time_margin = (sorted_times[:, 1] - sorted_times[:, 0]).float() / T  # (B,)
    confidence = time_margin.clamp(min=0.0, max=1.0)

    return DecoderOutput(
        logits_proxy=logits_proxy,
        prediction=prediction,
        confidence=confidence,
        aux={
            "first_spike_times": t_first.detach(),
            "ever_spiked": ever_spiked.detach(),
            "margin": time_margin.detach(),
        },
    )
```

**Why cumsum + argmax:** A naive implementation would iterate over timesteps or use `torch.nonzero`, both of which are slow or produce ragged outputs. The cumsum trick is fully vectorized: `cumsum >= 1` creates a boolean mask that is `False` before the first spike and `True` afterward, and `argmax` on a boolean tensor returns the index of the first `True`.

**No-spike handling:** Neurons that never fire within the T timesteps receive `t_first = T`, the worst possible latency. This places them at the bottom of the argmin ranking and assigns the lowest logits_proxy value (`-1.0 / temperature`). Do not leave no-spike neurons at `t_first = 0` -- that would make silent neurons appear to be the fastest, inverting the entire decoding logic.

**AMP safety:** Time indices are integer-valued (stored as float for gradient flow through temperature scaling but derived from argmax which returns long). The division by T and temperature is float arithmetic. No special fp32 casting is needed beyond the `.float()` on `t_first`.

**Migration from existing first_spike method:** The current code computes `torch.argmax(spikes, dim=0)` which finds the timestep of the largest spike value along `dim=0` (time-first). For binary spikes, argmax returns the first `1` timestep. Replace `dim=0` with `dim=1` and wrap in `DecoderOutput`. Add the no-spike masking, which the existing code lacks entirely.

---

## 4. Population Decoding

Population decoding partitions output neurons into groups, where each group represents a class (classification) or a range of values (regression). The encoder's `aux["population_map"]` provides the mapping from groups to neuron indices.

**Population map format:**

```python
# population_map: dict mapping group index to neuron indices
# Example for 10-class classification with 5 neurons per class:
population_map = {
    0: [0, 1, 2, 3, 4],
    1: [5, 6, 7, 8, 9],
    # ...
    9: [45, 46, 47, 48, 49],
}
# Or equivalently, a tensor of shape (num_groups, neurons_per_group)
```

**Classification mode:**

```python
def population_decode_classify(
    spikes: Tensor,                    # (B, T, N)
    population_map: Dict[int, List[int]],
    temperature: float = 1.0,
    eps: float = 1e-7,
) -> DecoderOutput:
    B, T, N = spikes.shape
    num_groups = len(population_map)

    # Accumulate spike counts per neuron in fp32.
    counts = spikes.float().sum(dim=1)             # (B, N)

    # Aggregate counts per group.
    group_counts = torch.zeros(B, num_groups, device=spikes.device, dtype=torch.float32)
    for g, indices in population_map.items():
        group_counts[:, g] = counts[:, indices].sum(dim=-1)

    logits_proxy = group_counts / temperature      # (B, num_groups)

    prediction = group_counts.argmax(dim=-1)       # (B,)

    probs = torch.softmax(logits_proxy, dim=-1)
    top2 = probs.topk(min(2, num_groups), dim=-1).values
    margin = (top2[:, 0] - top2[:, 1]).clamp(min=0.0) if num_groups >= 2 else top2[:, 0]
    confidence = margin

    return DecoderOutput(
        logits_proxy=logits_proxy,
        prediction=prediction,
        confidence=confidence,
        aux={
            "raw_counts": counts.detach(),
            "count_histogram": group_counts.detach().mean(dim=0),
            "margin": margin.detach(),
        },
    )
```

**Scalar regression mode:**

Assign each group a center value. Compute the expected value as a weighted sum of centers by group activity.

```python
def population_decode_regression(
    spikes: Tensor,                    # (B, T, N)
    population_map: Dict[int, List[int]],
    centers: Tensor,                   # (num_groups,) -- center value per group
    temperature: float = 1.0,
    eps: float = 1e-7,
) -> DecoderOutput:
    B, T, N = spikes.shape
    num_groups = len(population_map)

    counts = spikes.float().sum(dim=1)             # (B, N)

    group_counts = torch.zeros(B, num_groups, device=spikes.device, dtype=torch.float32)
    for g, indices in population_map.items():
        group_counts[:, g] = counts[:, indices].sum(dim=-1)

    # Softmax-weighted expectation over centers.
    weights = torch.softmax(group_counts / temperature, dim=-1)   # (B, num_groups)
    predicted_value = (weights * centers.unsqueeze(0)).sum(dim=-1) # (B,)

    # logits_proxy: group_counts for loss computation.
    logits_proxy = group_counts / temperature

    # Discretize prediction to nearest group.
    prediction = group_counts.argmax(dim=-1)

    confidence = weights.max(dim=-1).values

    return DecoderOutput(
        logits_proxy=logits_proxy,
        prediction=prediction,
        confidence=confidence,
        aux={
            "raw_counts": counts.detach(),
            "predicted_value": predicted_value.detach(),
            "group_weights": weights.detach(),
            "margin": confidence.detach(),
        },
    )
```

**Weighted-sum variant:** Replace the uniform sum over group neurons with learned weights. Add a `nn.Parameter` of shape `(N,)` and compute `counts[:, indices] * weights[indices]` before summing. This lets the network learn which neurons within a population are most informative, at the cost of additional parameters.

---

## 5. Membrane Decoding

Use the membrane potential directly when spike counts are too sparse to carry a reliable signal. This is common during early training when the network has not yet learned to produce consistent spiking patterns.

**Core computation:**

```python
def membrane_decode(
    membrane: Tensor,      # (B, T, N) float -- membrane potential over time
    mode: str = "final",   # "final" or "max"
    temperature: float = 1.0,
) -> DecoderOutput:
    if mode == "final":
        # Use membrane potential at the last timestep.
        v = membrane[:, -1, :]                  # (B, N)
    elif mode == "max":
        # Use maximum membrane potential over time.
        v = membrane.max(dim=1).values          # (B, N)
    else:
        raise ValueError(f"Unknown membrane decode mode: {mode}")

    logits_proxy = v / temperature              # (B, N)
    prediction = v.argmax(dim=-1)               # (B,)

    probs = torch.softmax(logits_proxy.float(), dim=-1)
    top2 = probs.topk(2, dim=-1).values
    confidence = (top2[:, 0] - top2[:, 1]).clamp(min=0.0)

    return DecoderOutput(
        logits_proxy=logits_proxy,
        prediction=prediction,
        confidence=confidence,
        aux={"membrane_values": v.detach()},
    )
```

**When to prefer membrane decoding:** Early training epochs where mean firing rate is below 0.01 spikes/timestep. In this regime, rate decoding produces near-zero counts for most neurons, making the loss landscape flat. Membrane potentials carry sub-threshold information that provides a gradient signal even when no spikes fire. As training progresses and firing rates increase, transition to rate or population decoding.

**Final vs. max:** Use `"final"` when the SNN processes a fixed-length input and the last timestep captures accumulated evidence. Use `"max"` when peak membrane potential is more informative (e.g., a neuron that briefly approached threshold but reset before the final step).

---

## 6. Decoder Composition

Combine multiple decoders into an ensemble when no single decoder is optimal across all training stages. Weight the logits_proxy from each decoder and sum them before computing prediction and confidence.

```python
class DecoderEnsemble:
    def __init__(
        self,
        decoders: List[Callable],      # Each returns DecoderOutput
        weights: List[float],          # Per-decoder weight, must sum to 1.0
    ):
        assert len(decoders) == len(weights)
        assert abs(sum(weights) - 1.0) < 1e-6
        self.decoders = decoders
        self.weights = weights

    def __call__(self, spikes, membrane=None) -> DecoderOutput:
        outputs = [d(spikes, membrane) for d in self.decoders]
        combined_logits = sum(
            w * o.logits_proxy for w, o in zip(self.weights, outputs)
        )
        prediction = combined_logits.argmax(dim=-1)
        probs = torch.softmax(combined_logits.float(), dim=-1)
        top2 = probs.topk(2, dim=-1).values
        confidence = (top2[:, 0] - top2[:, 1]).clamp(min=0.0)

        return DecoderOutput(
            logits_proxy=combined_logits,
            prediction=prediction,
            confidence=confidence,
            aux={"sub_outputs": [o.aux for o in outputs]},
        )
```

**Typical composition:** Use `0.7 * rate_logits + 0.3 * membrane_logits` during early training when spikes are sparse. Anneal membrane weight toward zero as firing rates stabilize. This provides a smooth gradient signal from membrane potentials while encouraging the network to learn proper spiking behavior for the rate decoder.

---

## 7. AMP Hardening for Decoders

Mixed-precision training (`torch.cuda.amp.autocast`) introduces float16 intermediates that interact poorly with spike-count arithmetic. Apply these rules uniformly across all decoder implementations.

**Rule 1 -- Count accumulation in fp32:**

```python
counts = spikes.float().sum(dim=1)    # Cast BEFORE sum
# NOT: counts = spikes.sum(dim=1).float()  -- sum in fp16 loses precision
```

Summing T binary float16 values risks precision loss when T is large. The `.float()` cast before `.sum()` ensures the accumulator is fp32 throughout.

**Rule 2 -- Softmax and log-softmax in fp32:**

```python
probs = torch.softmax(logits_proxy.float(), dim=-1)
log_probs = torch.log_softmax(logits_proxy.float(), dim=-1)
```

Softmax involves exponentiation; float16 overflow at `exp(11.09)` produces `inf`, corrupting the entire probability distribution.

**Rule 3 -- Temperature scaling before softmax:**

Apply `logits / temperature` before softmax, not after. Scaling after softmax changes the distribution nonlinearly and breaks the gradient relationship.

**Rule 4 -- Clamp before division and log:**

```python
safe_counts = counts.clamp(min=eps)
rate = safe_counts / T
log_rate = torch.log(safe_counts)
```

Zero counts produce `-inf` under `log` and `nan` under division. Always clamp with `eps` before these operations.

**Verification test:** Run each decoder under autocast with `T=10, 25, 50` and verify that all output fields are finite:

```python
with torch.cuda.amp.autocast():
    for T in [10, 25, 50]:
        spikes = (torch.rand(4, T, 20, device="cuda") > 0.8).float()
        output = decoder(spikes)
        assert output.logits_proxy.isfinite().all(), f"Non-finite logits at T={T}"
        assert output.confidence.isfinite().all(), f"Non-finite confidence at T={T}"
```

---

## 8. Migration from Current Code

The existing `SpikeDecoder` class in `brain_ai/core/encoding.py` (lines 196-242) implements three methods (`rate`, `first_spike`, `membrane`) in a single class with raw tensor output and time-first `(T, B, D)` axis convention.

**Current behavior summary:**

| Method | Input Axis | Operation | Output |
|---|---|---|---|
| `rate` | `dim=0` (time-first) | `spikes.sum(dim=0) / spikes.shape[0]` | `(B, D)` raw tensor |
| `first_spike` | `dim=0` (time-first) | `1 - argmax(spikes, dim=0) / T` | `(B, D)` raw tensor |
| `membrane` | N/A | Returns `membrane` directly | `(B, D)` raw tensor |

**Migration steps:**

1. **Preserve the existing class** as `SpikeDecoderLegacy` for backward compatibility during transition. Do not delete it until all call sites are updated.

2. **Add an axis-swap wrapper** that transposes `(T, B, D)` inputs to `(B, T, D)` at entry, enabling old-convention call sites to use new decoders without immediate refactoring:

```python
def adapt_time_first(spikes: Tensor) -> Tensor:
    """Transpose (T, B, D) to (B, T, D) for batch-first decoders."""
    if spikes.dim() == 3:
        return spikes.permute(1, 0, 2)
    return spikes
```

3. **Implement separate decoder classes** (`RateDecoder`, `FirstSpikeDecoder`, `MembraneDecoder`, `PopulationDecoder`) each returning `DecoderOutput`.

4. **Update call sites** in the training scripts and `BrainAI.forward` to pass `(B, T, D)` directly. Remove the axis-swap wrapper once all call sites are migrated.

5. **Add structured output** by wrapping the decoded tensor in `DecoderOutput` with computed prediction, confidence, and aux fields.

6. **Validate equivalence** by running the existing test suite against both legacy and new decoders on identical inputs (after axis transposition) and verifying that `logits_proxy` matches the legacy raw output within floating-point tolerance.

---

## 9. Anti-Patterns

**Summing spike counts in float16:**

```python
# WRONG: accumulation in fp16 loses precision for T > ~20
counts = spikes.half().sum(dim=1)
```

Always cast to float32 before summation. Float16 has only 3-4 decimal digits of precision; accumulating 50 binary values can produce incorrect counts due to rounding.

**argmin on unmasked no-spike neurons:**

```python
# WRONG: neurons that never spike have t_first=0, which wins argmin
t_first = spikes.argmax(dim=1)
prediction = t_first.argmin(dim=-1)  # Silent neuron appears fastest
```

Always mask no-spike neurons by assigning `t_first = T` before computing argmin. Failure to mask produces predictions that systematically favor inactive neurons.

**Population decoding without population_map:**

```python
# WRONG: treating each neuron as its own group collapses to rate decoding
# but with unnecessary overhead and no group-level statistics
group_counts[:, n] = counts[:, n]  # One neuron per group -- pointless
```

Population decoding requires a meaningful grouping of neurons. Without `population_map` from the encoder, fall back to rate decoding instead.

**Mixing decoder types mid-training without warmup:**

Switching from membrane decoding to rate decoding instantaneously changes the loss landscape, loss magnitude, and gradient distribution. Anneal decoder weights over 500-1000 steps using the `DecoderEnsemble` pattern (Section 6) rather than hard-switching. This gives the optimizer state (Adam moment estimates) time to adapt to the new gradient statistics.
