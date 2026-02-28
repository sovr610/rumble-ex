# Spike Encoders: From Continuous Values to Spike Trains

## Source File

`brain_ai/core/encoding.py` (292 lines). Contains five encoders (`RateEncoder`,
`TemporalEncoder`, `LatencyEncoder`, `PopulationEncoder`, `DeltaEncoder`) and one
decoder (`SpikeDecoder`). All currently emit `(T, B, D)` time-first tensors with no
structured output type, no AMP hardening, no generator control, and no deterministic
evaluation path.

This reference specifies the upgrade contract for all spike encoders: adopt `(B, T, ...)`
batch-first layout internally, emit `SpikeBatch` dataclass outputs, add AMP-safe sampling,
deterministic evaluation, and reproducible generator control.

---

## 1. Axis Convention

### Rule

Adopt `(B, T, ...)` batch-first internally for all spike encoder output. This aligns with
the canonical layout used by the SNN core (`snn_unroll`), the encoder contract
(`EncoderOutput` with `(B, T, D)` feats), and PyTorch's native batching conventions.

Any third-party library or legacy code path that produces `(T, B, ...)` must convert at
the boundary. Never propagate time-first layout beyond the encoder's own `forward` method.

### Conversion Helpers

Provide two standalone functions in `brain_ai/core/encoding.py` (or a shared
`brain_ai/core/layout.py`) that all encoders and adapters use. Never hardcode
`permute(1, 0, 2)` or `transpose(0, 1)` inline.

```python
def time_to_batch_first(x: torch.Tensor) -> torch.Tensor:
    """Permute (T, B, ...) to (B, T, ...).

    Accept any rank >= 2. Swap only the first two dims.
    """
    assert x.ndim >= 2, f"Expected rank >= 2, got {x.ndim}"
    return x.transpose(0, 1).contiguous()


def batch_to_time_first(x: torch.Tensor) -> torch.Tensor:
    """Permute (B, T, ...) to (T, B, ...).

    Accept any rank >= 2. Swap only the first two dims.
    """
    assert x.ndim >= 2, f"Expected rank >= 2, got {x.ndim}"
    return x.transpose(0, 1).contiguous()
```

### Output Shapes

| Encoder         | Output shape (batch-first)       |
|-----------------|----------------------------------|
| RateEncoder     | `(B, T, N)`                      |
| TemporalEncoder | `(B, T, N)`                      |
| LatencyEncoder  | `(B, T, N)`                      |
| PopulationEncoder | `(B, T, N*P)`                  |
| DeltaEncoder    | `(B, T, N*2)` (ON/OFF channels)  |
| ConvSpatial     | `(B, T, C, H, W)` (future)       |

Third-party wrappers (e.g., snnTorch `spikegen.rate`) produce `(T, B, N)`. Wrap their
output with `time_to_batch_first()` before returning from any encoder's `forward` method.
Provide `time_first: bool = False` as a backward-compatibility flag; when `True`, apply
`batch_to_time_first()` before return.

---

## 2. SpikeBatch Contract

### Dataclass Definition

```python
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import torch
from torch import Tensor


@dataclass
class SpikeBatch:
    """Structured container for spike encoder output.

    spikes: Binary spike tensor. dtype is bool, uint8, or float with values
            in {0, 1}. Shape (B, T, ...) batch-first.
    mask:   Validity mask (B, T) bool. True = valid timestep, False = padding.
    aux:    Metadata dict for downstream consumers and logging.
    """
    spikes: Tensor                             # (B, T, *feature_dims)
    mask: Tensor                               # (B, T) bool
    aux: Dict[str, Any] = field(default_factory=dict)
```

### Recognized `aux` Keys

| Key                    | Type          | Set by              | Purpose                                |
|------------------------|---------------|----------------------|----------------------------------------|
| `rates`                | `Tensor`      | RateEncoder          | Per-neuron firing probability `(B, N)` |
| `spike_times`          | `Tensor`      | Temporal/Latency     | Per-neuron spike time `(B, N)` int, -1=no spike |
| `population_map`       | `Tensor`      | PopulationEncoder    | `(N, P)` mapping features to population neurons |
| `normalization_stats`  | `Dict`        | Any                  | `{"mean": ..., "std": ...}` from input normalization |

### Factory Methods

```python
@staticmethod
def from_dense(spikes: Tensor, mask: Optional[Tensor] = None,
               aux: Optional[Dict[str, Any]] = None) -> "SpikeBatch":
    """Construct from a dense (B, T, ...) spike tensor.

    If mask is None, generate an all-True mask of shape (B, T).
    Validate that spikes are binary (values in {0, 1}).
    """
    ...

@staticmethod
def from_sparse(indices: Tensor, values: Tensor,
                shape: Tuple[int, ...],
                mask: Optional[Tensor] = None) -> "SpikeBatch":
    """Construct from sparse COO representation.

    Convert to dense internally. Useful for event-based inputs.
    """
    ...
```

### Property Helpers

```python
def is_binary(self) -> bool:
    """Return True if spikes contain only 0 and 1 values."""
    return torch.all((self.spikes == 0) | (self.spikes == 1)).item()

@property
def time_steps(self) -> int:
    """Return T from the (B, T, ...) layout."""
    return self.spikes.shape[1]

@property
def feature_shape(self) -> Tuple[int, ...]:
    """Return the trailing feature dimensions after (B, T)."""
    return tuple(self.spikes.shape[2:])
```

---

## 3. Rate Coding (Bernoulli / Poisson-like)

### Principle

Normalize input `x` to a rate probability `p` in `[0, 1]`. At each of `T` timesteps,
draw an independent Bernoulli sample: `spike[b, t, n] = (U[b, t, n] < p[b, n])` where
`U ~ Uniform(0, 1)`. Over `T` steps, the spike count per neuron follows a binomial
distribution with mean `T * p`. This matches the semantics of `snnTorch.spikegen.rate`.

### Normalization Modes

Provide a `normalization` parameter with these options:

| Mode       | Transform                        | When to use                     |
|------------|----------------------------------|---------------------------------|
| `"none"`   | `p = x` (assume pre-normalized)  | Input already in [0, 1]        |
| `"minmax"` | `p = (x - min) / (max - min)`   | Arbitrary-range features        |
| `"sigmoid"`| `p = sigmoid(x)`                 | Unbounded features              |
| `"clamp"`  | `p = clamp(x * gain + bias, 0, 1)` | Default, matches current code |

### Encode Signature

```python
class RateEncoder(nn.Module):
    def __init__(
        self,
        num_steps: int = 25,
        method: Literal["bernoulli", "poisson", "deterministic"] = "bernoulli",
        gain: float = 1.0,
        bias: float = 0.0,
        normalization: Literal["none", "minmax", "sigmoid", "clamp"] = "clamp",
        deterministic_eval: bool = True,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
        time_first: bool = False,  # backward compat
    ):
        ...

    def forward(self, x: Tensor) -> SpikeBatch:
        """Encode input to spike train.

        Args:
            x: (B, *features) continuous input.

        Returns:
            SpikeBatch with spikes (B, T, *features).
        """
        ...
```

### Forward Logic Pseudocode

```python
def forward(self, x: Tensor) -> SpikeBatch:
    B = x.shape[0]
    feat_shape = x.shape[1:]

    # 1. Normalize to rate p in [0, 1]
    p = self._normalize(x)  # (B, *feat_shape)

    # 2. Sample spikes
    if self.training or not self.deterministic_eval:
        # Stochastic: Bernoulli sampling in fp32
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            p_fp32 = p.float()
            U = torch.rand(
                (B, self.num_steps, *feat_shape),
                device=x.device, dtype=torch.float32,
                generator=self._get_generator(x.device),
            )
            spikes = (U < p_fp32.unsqueeze(1)).to(torch.uint8)
    else:
        # Deterministic: threshold at 0.5 (or use accumulator for deterministic method)
        spikes = (p.unsqueeze(1).expand(B, self.num_steps, *feat_shape) > 0.5)
        spikes = spikes.to(torch.uint8)

    # 3. Build mask (all valid unless overridden)
    mask = torch.ones(B, self.num_steps, dtype=torch.bool, device=x.device)

    # 4. Axis swap if requested
    if self.time_first:
        spikes = batch_to_time_first(spikes)

    return SpikeBatch(
        spikes=spikes,
        mask=mask,
        aux={"rates": p.detach()},
    )
```

### AMP Rule

All random sampling must happen in `float32`. The `torch.rand` call and the comparison
`U < p` must not execute under `float16` or `bfloat16` autocast, because half-precision
uniform samples have only 1024 distinct values in `[0, 1]`, causing severe quantization
of firing rates. Use `torch.amp.autocast(device_type=..., enabled=False)` around the
sampling block. Cast the result to `uint8` or `bool` immediately after.

### Migration from Current Code

The existing `RateEncoder` (lines 12-70 of `encoding.py`) already implements the core
Bernoulli/Poisson/deterministic logic. Migration steps:

1. Add `normalization`, `bias`, `deterministic_eval`, `generator`, `seed` parameters.
2. Replace `torch.rand(self.num_steps, *x_norm.shape, ...)` with
   `torch.rand((B, self.num_steps, *feat_shape), ..., generator=...)` -- note the axis
   order change from `(T, B, ...)` to `(B, T, ...)`.
3. Wrap sampling in `autocast(enabled=False)`.
4. Return `SpikeBatch` instead of raw tensor.
5. Add `time_first` flag for backward compatibility.

---

## 4. Latency Coding (Multi-Spike) and TTFS

### Latency Coding Principle

Map input value `x` to a firing time distribution over the window `[0, T-1]`. Optionally
emit multiple spikes around the computed time. Higher input values produce earlier spikes
(lower latency), encoding magnitude as temporal position.

### Time-to-First-Spike (TTFS)

A strict variant: emit exactly one spike per neuron per time window, at the computed
time `t(x)`. If `allow_no_spike=True` and the computed time exceeds `T-1`, emit no spike
for that neuron (spike_times = -1).

### Mapping Functions

Provide three mapping functions from value `x` (in `[0, 1]`, higher = more salient) to
spike time `t`:

| Mapping         | Formula                                       | Properties                         |
|-----------------|-----------------------------------------------|------------------------------------|
| `"linear"`      | `t = t_max - x * (t_max - t_min)`            | Uniform resolution across range    |
| `"exponential"` | `t = tau * log(1 / clamp(x, eps, 1))`        | Higher resolution near x=1         |
| `"logarithmic"` | `t = t_max * (1 - log(1 + x) / log(2))`     | Higher resolution near x=0         |

Clamp all computed times to `[t_min, t_max]` after applying the mapping.

### Encode Signature

```python
class LatencyEncoder(nn.Module):
    def __init__(
        self,
        num_steps: int = 25,
        mapping: Literal["linear", "exponential", "logarithmic"] = "linear",
        t_min: int = 0,
        t_max: Optional[int] = None,  # defaults to num_steps - 1
        tau: float = 5.0,             # time constant for exponential mapping
        allow_no_spike: bool = False,
        jitter: float = 0.0,          # std of Gaussian jitter (training only)
        normalize: bool = True,
        deterministic_eval: bool = True,
        generator: Optional[torch.Generator] = None,
        time_first: bool = False,
    ):
        ...

    def forward(self, x: Tensor) -> SpikeBatch:
        ...
```

### Forward Logic Pseudocode

```python
def forward(self, x: Tensor) -> SpikeBatch:
    B = x.shape[0]
    feat_shape = x.shape[1:]
    t_max = self.t_max if self.t_max is not None else self.num_steps - 1

    # 1. Normalize x to [0, 1]
    x_norm = self._normalize(x)  # (B, *feat_shape)

    # 2. Compute spike times in fp32
    with torch.amp.autocast(device_type=x.device.type, enabled=False):
        x_fp32 = x_norm.float()
        if self.mapping == "linear":
            t_spike = t_max - x_fp32 * (t_max - self.t_min)
        elif self.mapping == "exponential":
            t_spike = self.tau * torch.log(1.0 / x_fp32.clamp(min=1e-6))
        elif self.mapping == "logarithmic":
            t_spike = t_max * (1.0 - torch.log(1.0 + x_fp32) / math.log(2.0))

        t_spike = t_spike.clamp(self.t_min, t_max)

        # 3. Add jitter during training
        if self.training and self.jitter > 0:
            noise = torch.randn_like(t_spike, generator=self._get_generator(x.device))
            t_spike = (t_spike + noise * self.jitter).clamp(self.t_min, t_max)

        t_int = t_spike.round().long()  # (B, *feat_shape)

    # 4. Scatter spikes into (B, T, *feat_shape)
    spikes = torch.zeros(B, self.num_steps, *feat_shape,
                         dtype=torch.uint8, device=x.device)
    # Use scatter along time dimension
    t_idx = t_int.unsqueeze(1)  # (B, 1, *feat_shape)
    spikes.scatter_(1, t_idx, 1)

    # 5. Handle no-spike
    if self.allow_no_spike:
        no_spike_mask = (t_int > t_max) | (t_int < self.t_min)
        spike_times = t_int.clone()
        spike_times[no_spike_mask] = -1
    else:
        spike_times = t_int

    mask = torch.ones(B, self.num_steps, dtype=torch.bool, device=x.device)

    return SpikeBatch(
        spikes=spikes,
        mask=mask,
        aux={"spike_times": spike_times.detach()},
    )
```

### Migration from Current Code

The existing `TemporalEncoder` (lines 73-109) implements exponential TTFS with a fixed
`tau`. The existing `LatencyEncoder` (lines 112-144) implements linear mapping with
optional normalization. Merge both into a single `LatencyEncoder` class:

1. Fold `TemporalEncoder.tau` into the `exponential` mapping branch.
2. Replace the per-timestep Python loop (`for t in range(self.num_steps)`) with
   vectorized `scatter_` along the time dimension.
3. Add `jitter`, `allow_no_spike`, `generator`, `deterministic_eval`.
4. Change output axis from `(T, B, ...)` to `(B, T, ...)`.
5. Return `SpikeBatch` with `spike_times` in `aux`.

---

## 5. Population Coding

### Principle

Expand each input dimension into `P` population neurons, each with a tuning curve
centered at a different preferred value. The activation of each population neuron encodes
how close the input is to that neuron's preferred value. Convert activations to spikes
via rate coding or latency coding.

### Tuning Curves

Use Gaussian tuning curves by default:

```
a_p(x) = exp(-(x - c_p)^2 / (2 * sigma^2))
```

where `c_p` is the center of population neuron `p` and `sigma` controls width.

### Center Spacing

| Mode        | Behavior                                              |
|-------------|-------------------------------------------------------|
| `"linear"`  | `centers = linspace(0, 1, P)` -- fixed, uniform       |
| `"learned"` | `centers = nn.Parameter(linspace(0, 1, P))`           |
| `"uniform"` | `centers = linspace(x_min, x_max, P)` per-feature     |

### Sigma Configuration

- `sigma: float` -- fixed width for all neurons (default 0.2).
- `sigma: "auto"` -- set `sigma = 1 / (2 * (P - 1))` so adjacent curves overlap at
  half-height.
- `sigma: nn.Parameter` -- learned per-neuron or per-feature width.

### Pooling Groups

Provide a `pooling_groups` mapping so downstream decoders know which population neurons
correspond to which input feature. Store as a `(N,)` integer tensor where entry `i`
gives the original feature index for population neuron `i`. Place in `aux["population_map"]`.

### Output Shape

Two layout options controlled by `flatten: bool`:

- `flatten=True` (default): `(B, T, N*P)` -- population neurons concatenated along the
  feature axis. This is the layout consumed by the SNN core and workspace.
- `flatten=False`: `(B, T, P, N)` -- explicit population dimension preserved. Useful
  for population-level pooling operations.

### Encode Signature

```python
class PopulationEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_neurons_per_dim: int = 10,
        num_steps: int = 25,
        sigma: Union[float, str] = 0.2,
        center_mode: Literal["linear", "learned", "uniform"] = "linear",
        spike_method: Literal["rate", "latency"] = "rate",
        flatten: bool = True,
        deterministic_eval: bool = True,
        generator: Optional[torch.Generator] = None,
        time_first: bool = False,
    ):
        ...

    def forward(self, x: Tensor) -> SpikeBatch:
        """
        Args:
            x: (B, N) continuous input.

        Returns:
            SpikeBatch with spikes (B, T, N*P) or (B, T, P, N).
        """
        ...
```

### Migration from Current Code

The existing `PopulationEncoder` (lines 147-193) implements Gaussian tuning curves with
fixed linear centers and rate-coded output. Migration steps:

1. Add `center_mode`, `sigma="auto"` option, `spike_method`, `flatten`,
   `deterministic_eval`, `generator` parameters.
2. Build `population_map` tensor: `torch.arange(input_dim).repeat_interleave(P)`.
3. Change random sampling from `torch.rand(self.num_steps, *activations.shape, ...)`
   to `torch.rand((B, self.num_steps, N*P), ..., generator=...)` with fp32 guard.
4. Return `SpikeBatch` with `population_map` in `aux`.
5. Swap output axis order from `(T, B, N*P)` to `(B, T, N*P)`.

---

## 6. Delta Modulation / Event-Based

### Principle

Emit spikes only when the input signal changes by more than a threshold. This mimics
dynamic vision sensors (DVS) and other event-driven hardware. Produce separate ON
(positive change) and OFF (negative change) channels, doubling the feature dimension.

### Stateful Operation

`DeltaEncoder` maintains internal state (`prev_input`) tracking the last input value.
Reset this state between sequences by calling `encoder.reset()`. Failing to reset causes
the first timestep of a new sequence to compare against the last timestep of the previous
sequence, producing spurious spikes at sequence boundaries.

### Configuration

```python
class DeltaEncoder(nn.Module):
    def __init__(
        self,
        threshold: float = 0.1,
        off_threshold: Optional[float] = None,  # defaults to -threshold
        num_steps: int = 25,
        persistence: int = 1,  # how many steps an event persists
        time_first: bool = False,
    ):
        ...

    def reset(self):
        """Clear internal state. Call between sequences."""
        self.prev_input = None

    def forward(self, x: Tensor) -> SpikeBatch:
        """
        Args:
            x: (B, *features) current input frame.

        Returns:
            SpikeBatch with spikes (B, T, *features, 2) or (B, T, features*2).
        """
        ...
```

### ON/OFF Channel Layout

Concatenate ON and OFF channels along the last feature dimension:

```python
delta = x - prev_input
on_spikes  = (delta >  threshold)    # positive change
off_spikes = (delta < off_threshold) # negative change (off_threshold < 0)
spikes = torch.cat([on_spikes, off_spikes], dim=-1)  # (..., N*2)
```

### Migration from Current Code

The existing `DeltaEncoder` (lines 244-292) implements the core ON/OFF logic with a
single threshold. Migration steps:

1. Add `off_threshold` parameter (defaults to `-self.threshold`).
2. Add `persistence` parameter to control how many timesteps an event persists
   (current code repeats for all `num_steps`, which is excessive).
3. Change output from `spikes.unsqueeze(0).repeat(T, ...)` with `(T, B, N*2)` layout
   to `(B, T, N*2)` with configurable persistence.
4. Return `SpikeBatch` instead of raw tensor.
5. Add `time_first` backward-compatibility flag.

---

## 7. Deterministic Evaluation

### Principle

During training, use stochastic sampling to provide gradient-friendly noise and explore
the spike space. During inference, use deterministic paths to produce reproducible output
for the same input.

### Toggling

Deterministic mode activates when both conditions hold:
- `model.eval()` has been called (i.e., `self.training == False`).
- `deterministic_eval=True` in the encoder config (default).

When `deterministic_eval=False`, stochastic sampling runs in both train and eval modes.
This is useful for uncertainty estimation via Monte Carlo sampling at inference time.

### Per-Encoder Deterministic Paths

| Encoder          | Training (stochastic)                    | Inference (deterministic)                       |
|------------------|------------------------------------------|-------------------------------------------------|
| RateEncoder      | `spike = (U < p)`, U ~ Uniform          | `spike = (p > 0.5)` hard threshold              |
| LatencyEncoder   | Spike time + Gaussian jitter             | Spike time only, no jitter                       |
| TemporalEncoder  | (same as LatencyEncoder exponential)     | No jitter                                        |
| PopulationEncoder| Tuning activation + rate/latency noise   | Tuning activation + deterministic spike method   |
| DeltaEncoder     | Already deterministic (threshold-based)  | Same behavior                                    |

### Rate Encoder Deterministic Detail

The simplest deterministic path thresholds at 0.5: `spike = (p > 0.5)`. For finer
resolution, use the deterministic accumulator method already present in the current code
(lines 63-70): maintain a running accumulator, emit a spike when it crosses 1.0, and
subtract 1.0. This preserves the time-averaged rate information that hard thresholding
at 0.5 discards.

Select via `deterministic_method: Literal["threshold", "accumulator"] = "accumulator"`.

---

## 8. AMP Hardening for Encoders

### Core Rule

All sampling operations must execute in `float32`. Autocast may lower computation to
`float16` or `bfloat16` for performance, but the spike generation pipeline has specific
numerical requirements that `float16` violates.

### What Breaks Under float16

- `torch.rand` in float16 has only 1024 distinct values in `[0, 1]`. Rate coding with
  fine-grained probabilities (e.g., `p = 0.73`) becomes severely quantized.
- `torch.log(x)` in float16 underflows for small `x`, producing `-inf` spike times in
  latency coding.
- `torch.exp(-x^2)` in float16 flushes to zero for moderate `x`, killing population
  coding tuning curves at the tails.

### Hardening Pattern

Wrap all sampling and transcendental operations in an autocast-disabled context:

```python
with torch.amp.autocast(device_type=device_type, enabled=False):
    # Force fp32 for sampling
    p_fp32 = p.float()
    U = torch.rand(..., dtype=torch.float32, generator=gen)
    spikes_fp32 = (U < p_fp32)

# Cast to compact dtype outside the guard
spikes = spikes_fp32.to(torch.uint8)
```

### Compact Output Dtype

After generation, cast spikes to the smallest viable dtype:
- `torch.bool` -- 1 bit per element, most compact, but does not support arithmetic.
- `torch.uint8` -- 1 byte per element, supports arithmetic, preferred default.
- `torch.float32` -- Only when downstream code requires float (e.g., direct matmul with
  weight matrices without an explicit cast).

### Testing Requirements

Verify no NaN or Inf under autocast for each encoder with `T` in `{10, 25, 50}`:

```python
@pytest.mark.parametrize("T", [10, 25, 50])
def test_rate_encoder_amp(T):
    encoder = RateEncoder(num_steps=T)
    x = torch.randn(4, 128, device="cuda")
    with torch.amp.autocast(device_type="cuda"):
        out = encoder(x)
    assert not torch.isnan(out.spikes).any()
    assert not torch.isinf(out.spikes.float()).any()
    assert out.is_binary()
```

---

## 9. Migration from Current Code

### Current State

`brain_ai/core/encoding.py` contains 292 lines with five encoders and one decoder. All
encoders:
- Emit raw `torch.Tensor` with `(T, B, D)` layout.
- Use `torch.rand(self.num_steps, *x.shape, ...)` with no generator.
- Have no deterministic path toggled by `model.eval()` (except the deterministic rate
  method which is always deterministic).
- Perform no AMP guarding.
- Return no structured metadata.

### Migration Steps

Execute in this order to maintain backward compatibility at each step:

**Step 1: Add helpers and SpikeBatch.**
Add `time_to_batch_first()`, `batch_to_time_first()`, and the `SpikeBatch` dataclass
to `encoding.py`. No existing code changes yet.

**Step 2: Add `time_first` flag to all encoders.**
Default `time_first=True` to preserve current behavior. All existing callers continue
to work unchanged.

**Step 3: Upgrade internal sampling to (B, T, ...).**
Change `torch.rand` calls from `(T, *x.shape)` to `(B, T, *feat_shape)`. Apply
`batch_to_time_first()` when `time_first=True`. Add `generator` parameter.

**Step 4: Wrap output in SpikeBatch.**
Return `SpikeBatch` from all `forward` methods. Add a `raw_tensor: bool = False`
parameter; when `True`, return the bare tensor for callers that cannot handle the
dataclass yet.

**Step 5: Add deterministic_eval and AMP hardening.**
Add `deterministic_eval` parameter. Wrap sampling in `autocast(enabled=False)`.

**Step 6: Flip defaults.**
Set `time_first=False` (batch-first default) and `raw_tensor=False` (SpikeBatch
default). Update all callers in the codebase to use the new convention.

**Step 7: Remove legacy flags.**
After all callers are updated, remove `time_first` and `raw_tensor` parameters.

### Caller Updates

Grep for import sites and update:
- `brain_ai/system.py` -- `BrainAI.forward` encoder path.
- `brain_ai/encoders/*.py` -- modality encoders that use spike encoding internally.
- `scripts/train_phase1.py` -- phase 1 training loop.
- `tests/test_snn.py` -- SNN tests.
- `examples/inference_demo.py` -- inference demo.

---

## 10. Anti-Patterns

### Hardcoding axis permutations

Never write `spikes.permute(1, 0, 2)` or `spikes.transpose(0, 1)` inline. Use
`time_to_batch_first()` or `batch_to_time_first()`. Hardcoded permutations break
silently when tensor rank changes (e.g., adding spatial dimensions for conv encoders).

### Sampling in float16

Never allow `torch.rand` to execute under autocast. The 1024-value quantization in
float16 destroys rate coding resolution and causes latency coding to produce degenerate
spike time distributions. Always guard with `autocast(enabled=False)`.

### Forgetting to reset DeltaEncoder state

`DeltaEncoder` compares the current input to `self.prev_input`. Between sequences,
call `encoder.reset()`. Failing to reset produces spurious ON/OFF spikes at the first
timestep of the new sequence. In training loops, call `reset()` at the start of each
batch or sequence.

### Population coding without pooling_groups

Downstream decoders need to know which population neurons map back to which input
features. Always provide `population_map` in `aux`. Without it, decoding requires
the caller to manually reconstruct the feature-to-neuron mapping, which is error-prone
and breaks when `num_neurons_per_dim` changes.

### Missing generator/seed for reproducibility

Always accept a `generator` or `seed` parameter. When neither is provided, sampling
is non-reproducible across runs. For testing and debugging, pass a fixed seed. For
distributed training, use per-rank generators to avoid correlated spike patterns across
workers.

### Returning float32 spikes when bool/uint8 suffice

Spikes are binary. Returning `float32` tensors wastes 4x memory compared to `uint8`
and 32x compared to `bool`. Cast to the most compact dtype that downstream code
supports. If downstream matmuls require float, let the caller cast explicitly.
