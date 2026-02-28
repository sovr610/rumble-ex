# Loss Pack: Composable SNN Loss Terms with AMP Hardening

Reference for the `spike-codec-losses` skill. Covers the modular loss composition
architecture, individual loss term specifications, AMP hardening rules, logging
contract, and migration path from the existing `brain_ai/core/losses.py`.

All tensor conventions use **(B, T, N)** — batch-first — unless explicitly stated
otherwise. The existing codebase uses (T, B, N) time-first; migration to (B, T, N)
is a prerequisite for this skill.

Source file: `brain_ai/core/losses.py` (434 lines, current implementation).

---

## 1. Loss Composition Architecture

### SNNLossComposer

Define a single `SNNLossComposer(nn.Module)` that holds an ordered dict of named
`LossTerm` instances. Each term computes independently; the composer sums weighted
results and aggregates diagnostics.

```python
class LossTerm(nn.Module):
    """Base class for all SNN loss terms."""
    def forward(self, spikes, membrane, targets, **kwargs):
        # -> (loss_value: Tensor, diagnostics: dict[str, float])
        raise NotImplementedError

class SNNLossComposer(nn.Module):
    def __init__(self, terms: dict[str, tuple[LossTerm, float]]):
        """
        Args:
            terms: {"name": (LossTerm instance, weight)} ordered dict.
        """
        super().__init__()
        self.terms = nn.ModuleDict({k: v[0] for k, v in terms.items()})
        self.weights = {k: v[1] for k, v in terms.items()}

    def forward(self, spikes, membrane, targets, **kwargs):
        total = torch.tensor(0.0, device=spikes.device, dtype=torch.float32)
        components = {}
        diagnostics = {}

        for name, term in self.terms.items():
            w = self.weights[name]
            if w == 0.0:
                continue
            raw_loss, diag = term(spikes, membrane, targets, **kwargs)
            weighted = w * raw_loss
            total = total + weighted
            components[f"loss/{name}_raw"] = raw_loss.item()
            components[f"loss/{name}_weighted"] = weighted.item()
            diagnostics.update({f"diag/{name}/{k}": v for k, v in diag.items()})

        components["loss/total"] = total.item()
        return total, {**components, **diagnostics}
```

Contract:

- All terms receive `(B, T, N)` spikes, `(B, T, N)` membrane, `(B,)` targets.
- All terms return `(scalar_loss, diagnostics_dict)` where the scalar is fp32.
- The composer logs raw and weighted magnitudes for every active term.
- Enable or disable terms by setting weight to 0.0 in config.
- Per-layer targets (e.g., different target rates per layer) pass through `**kwargs`.

---

## 2. ProbSpikes (Spike-Count Cross-Entropy)

### Definition

ProbSpikes computes cross-entropy on normalized spike counts rather than membrane
potentials. Sum spikes over the time dimension to produce a count vector per sample,
convert counts to log-probabilities, and compute negative log-likelihood against class
targets.

### Computation

```python
class ProbSpikesLoss(LossTerm):
    def __init__(self, temperature=1.0, eps=1e-7, mode="softmax"):
        super().__init__()
        self.temperature = temperature
        self.eps = eps
        self.mode = mode  # "softmax" or "normalize"

    def forward(self, spikes, membrane, targets, **kwargs):
        # spikes: (B, T, C) where C = num_classes
        # Accumulate counts in fp32 regardless of AMP context
        counts = spikes.float().sum(dim=1)  # (B, C) in fp32

        if self.mode == "softmax":
            # Temperature-scaled softmax cross-entropy
            scaled = counts / self.temperature
            log_probs = F.log_softmax(scaled, dim=-1)  # fp32
            loss = F.nll_loss(log_probs, targets)
        elif self.mode == "normalize":
            # Direct normalization: counts / (sum + eps)
            total = counts.sum(dim=-1, keepdim=True)
            probs = counts / (total + self.eps)
            probs = probs.clamp(min=self.eps)
            log_probs = probs.log()
            loss = F.nll_loss(log_probs, targets)

        with torch.no_grad():
            diag = {}
            diag["mean_count"] = counts.mean().item()
            diag["max_count"] = counts.max().item()
            diag["accuracy"] = (counts.argmax(dim=-1) == targets).float().mean().item()
            p = F.softmax(counts / self.temperature, dim=-1)
            diag["output_entropy"] = -(p * (p + 1e-8).log()).sum(-1).mean().item()

        return loss, diag
```

### Key details

- Accumulate `spikes.float().sum(dim=1)` to force fp32. Under AMP with float16
  spikes, the subsequent softmax on raw counts can overflow for large T.
- Apply temperature *before* softmax, never after. Dividing log-softmax output by
  temperature changes gradient scaling without changing probability ranking.
- Clamp with `.clamp(min=eps)` before any `.log()` call. In normalize mode, a class
  with zero spikes produces `probs=0`, and `log(0)` yields `-inf`.
- Prefer `softmax` mode for classification. Use `normalize` mode when raw spike count
  ratios have physical meaning.

### Config fields

```python
prob_spikes_temperature: float = 1.0
prob_spikes_eps: float = 1e-7
prob_spikes_mode: str = "softmax"  # "softmax" | "normalize"
```

### Migration from existing code

Replace `prob_spikes_loss()` standalone function. The existing function sums over
dim=0 (time-first convention). Change to dim=1 for (B, T, N). The existing function
does not force fp32 accumulation or clamp before log. Add both.

---

## 3. Spike-Rate Regularization

### Target-Rate Penalty

Compute the mean firing rate per neuron across time, then penalize deviation from a
target rate.

```python
class SpikeRateRegularization(LossTerm):
    def __init__(self, target_rate=0.1, min_rate=0.01, max_rate=0.3,
                 rate_type="l2", use_range_penalty=True):
        super().__init__()
        self.target_rate = target_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.rate_type = rate_type
        self.use_range_penalty = use_range_penalty

    def forward(self, spikes, membrane, targets, **kwargs):
        # spikes: (B, T, N)
        rates = spikes.float().mean(dim=1)  # (B, N), fp32

        # Target-rate loss
        if self.rate_type == "l2":
            target_loss = ((rates - self.target_rate) ** 2).mean()
        elif self.rate_type == "l1":
            target_loss = (rates - self.target_rate).abs().mean()

        loss = target_loss

        # Range penalty: penalize rates outside [min_rate, max_rate]
        if self.use_range_penalty:
            below = F.relu(self.min_rate - rates)   # (B, N)
            above = F.relu(rates - self.max_rate)   # (B, N)
            range_loss = (below + above).mean()
            loss = loss + range_loss

        # Per-layer override: kwargs may contain per_layer_target_rate
        # Caller passes layer-specific targets when composing multi-layer losses

        # Diagnostics
        with torch.no_grad():
            diag = {}
            diag["mean_rate"] = rates.mean().item()
            diag["max_rate"] = rates.max().item()
            diag["min_rate"] = rates.min().item()
            # Dead neurons: rate == 0 across entire batch
            per_neuron_rate = rates.mean(dim=0)  # (N,)
            diag["dead_neuron_frac"] = (per_neuron_rate == 0).float().mean().item()
            # Saturated neurons: rate > 0.9
            diag["saturated_frac"] = (per_neuron_rate > 0.9).float().mean().item()
            if self.use_range_penalty:
                diag["below_min_frac"] = (per_neuron_rate < self.min_rate).float().mean().item()
                diag["above_max_frac"] = (per_neuron_rate > self.max_rate).float().mean().item()

        return loss, diag
```

### Per-layer targets

Some layers naturally fire more than others. Pass per-layer target rates through
`kwargs`:

```python
# In training loop, per-layer invocation:
rate_term(layer_spikes, membrane, targets, target_rate=0.05)  # sparse layer
rate_term(layer_spikes, membrane, targets, target_rate=0.2)   # active layer
```

Override `self.target_rate` with `kwargs.get("target_rate", self.target_rate)` inside
the forward method.

### Config fields

```python
spike_rate_target: float = 0.1
spike_rate_min: float = 0.01
spike_rate_max: float = 0.3
spike_rate_type: str = "l2"       # "l1" | "l2"
spike_rate_use_range: bool = True
```

### Migration

Unify `spike_rate_regularization()` and `spike_rate_range_regularization()` into this
single term. The existing functions assume dim=0 is time. Change to dim=1. The
existing functions do not report diagnostics. Add dead/saturated neuron tracking.

---

## 4. Temporal Consistency Regularization

### Goal

Discourage "flicker" — rapid, unstable oscillation in firing patterns that carries no
useful temporal information. Encourage smooth, structured temporal dynamics.

### Three penalty modes

```python
class TemporalConsistencyLoss(LossTerm):
    def __init__(self, window_size=5, penalty_type="l2"):
        super().__init__()
        self.window_size = window_size
        self.penalty_type = penalty_type  # "l1" | "l2" | "variance"

    def forward(self, spikes, membrane, targets, **kwargs):
        # spikes: (B, T, N)
        B, T, N = spikes.shape

        if self.penalty_type in ("l1", "l2"):
            # Smoothness on firing rates: penalize rate changes between adjacent steps
            rates = spikes.float()  # (B, T, N)
            diffs = rates[:, 1:, :] - rates[:, :-1, :]  # (B, T-1, N)
            if self.penalty_type == "l1":
                loss = diffs.abs().mean()
            else:
                loss = (diffs ** 2).mean()

        elif self.penalty_type == "variance":
            # Windowed variance: split T into windows, compute variance of window means
            if T < self.window_size * 2:
                return torch.tensor(0.0, device=spikes.device), {"skipped": 1.0}
            n_windows = T // self.window_size
            trimmed = spikes[:, :n_windows * self.window_size, :].float()
            # (B, n_windows, window_size, N)
            windowed = trimmed.view(B, n_windows, self.window_size, N)
            window_means = windowed.mean(dim=2)  # (B, n_windows, N)
            loss = window_means.var(dim=1).mean()  # variance across windows

        # Membrane smoothness as optional additive penalty
        if membrane is not None and membrane.dim() == 3:
            mem_diffs = membrane[:, 1:, :].float() - membrane[:, :-1, :].float()
            mem_smooth = (mem_diffs ** 2).mean()
            loss = loss + 0.1 * mem_smooth  # fixed sub-weight

        # Diagnostics
        with torch.no_grad():
            diag = {}
            rates_f = spikes.float()
            rate_diffs = rates_f[:, 1:, :] - rates_f[:, :-1, :]
            diag["mean_temporal_variance"] = rate_diffs.var().item()
            diag["max_temporal_variance"] = rate_diffs.var(dim=1).max().item()

        return loss, diag
```

### Config fields

```python
temporal_window_size: int = 5
temporal_penalty_type: str = "l2"  # "l1" | "l2" | "variance"
```

### Migration

Merge `temporal_consistency_loss()` and `temporal_sparsity_loss()` into this single
term. The existing `temporal_consistency_loss` uses a Python for-loop to build
windows — replace with `view`-based reshaping. The existing `temporal_sparsity_loss`
targets a fraction of silent timesteps; absorb this into the spike-rate range
regularizer (section 3) rather than keeping it as a separate temporal term.

---

## 5. ISI Regularization (Inter-Spike Interval)

### Goal

Discourage unrealistic burst firing and enforce minimum refractory periods. Penalize
spikes that occur within a refractory window following a previous spike.

### Soft-ISI via Convolutional Refractory Penalty

Define a refractory kernel that models the expected suppression period after a spike.
Convolve each neuron's spike train with this kernel, then penalize spikes that fire
during the refractory period (where the convolution output is high).

**This implementation MUST be fully vectorized.** No Python for-loops over neurons or
batch elements.

```python
class ISIRegularization(LossTerm):
    def __init__(self, refractory_window=5, kernel_type="exponential",
                 penalty_weight=1.0):
        super().__init__()
        self.refractory_window = refractory_window
        self.kernel_type = kernel_type
        self.penalty_weight = penalty_weight

        # Build refractory kernel (1D, causal)
        # Kernel shape: (1, 1, refractory_window)
        # The kernel represents the refractory suppression strength at each lag.
        # Index 0 = lag 1 (the timestep immediately after a spike).
        # The kernel does NOT include lag 0 (the spike itself).
        k = torch.arange(1, refractory_window + 1, dtype=torch.float32)
        if kernel_type == "exponential":
            tau = refractory_window / 3.0
            kernel = torch.exp(-k / tau)
        elif kernel_type == "rectangular":
            kernel = torch.ones_like(k)
        # Register as buffer so it moves with device and persists in state_dict
        self.register_buffer("kernel", kernel.flip(0).unsqueeze(0).unsqueeze(0))
        # kernel shape: (1, 1, refractory_window), flipped for causal conv

    def forward(self, spikes, membrane, targets, **kwargs):
        # spikes: (B, T, N), binary
        B, T, N = spikes.shape

        # Reshape for grouped conv1d: treat each neuron independently
        # F.conv1d expects (batch, channels, length)
        # Reshape spikes to (B*N, 1, T) for per-neuron convolution
        s = spikes.float().permute(0, 2, 1).reshape(B * N, 1, T)  # (B*N, 1, T)

        # Causal convolution: pad left by refractory_window, no right pad
        # Kernel is in fp32 (buffer), ensure input is fp32
        kernel_fp32 = self.kernel  # (1, 1, refractory_window), already fp32
        padded = F.pad(s, (self.refractory_window, 0))  # (B*N, 1, T + rw)
        refractory_signal = F.conv1d(padded, kernel_fp32)  # (B*N, 1, T)

        # Penalty: spike * refractory_signal measures spikes during refractory period
        # High refractory_signal at a spike location means a burst violation
        penalty = (s * refractory_signal).mean()

        loss = self.penalty_weight * penalty

        # Diagnostics
        with torch.no_grad():
            diag = {}
            # Reshape back to (B, N, T) for analysis
            s_bn = s.view(B, N, T)
            # Mean ISI: compute per-neuron mean ISI across the batch
            # Use spike count and total time to estimate mean ISI
            spike_counts = s_bn.sum(dim=-1)  # (B, N)
            # Mean ISI ~ T / (count + 1) for neurons that fire
            firing_mask = spike_counts > 1
            if firing_mask.any():
                mean_isi = (T / spike_counts[firing_mask]).mean().item()
                diag["mean_isi"] = mean_isi
            else:
                diag["mean_isi"] = float(T)
            # Burst fraction: fraction of spikes with refractory_signal > threshold
            refr_at_spikes = (s * refractory_signal).view(B, N, T)
            total_spikes = s_bn.sum()
            if total_spikes > 0:
                burst_count = (refr_at_spikes > 0.5).float().sum()
                diag["burst_fraction"] = (burst_count / total_spikes).item()
            else:
                diag["burst_fraction"] = 0.0
            diag["mean_refractory_penalty"] = penalty.item()

        return loss, diag
```

### Vectorization strategy

The critical operation is `F.conv1d` with the spike train reshaped to `(B*N, 1, T)`.
This processes all neurons in all batch elements as a single batched convolution
call — no Python loops. The kernel is shared (same refractory profile for all
neurons). For per-neuron kernels, use `groups=N` with kernel shape `(N, 1, rw)` and
spike shape `(B, N, T)`.

### Why conv1d and not manual indexing

The existing `inter_spike_interval_loss` iterates over 4 batch elements and 100
sampled neurons in Python, calling `torch.where` per neuron. This is O(B * N * T) in
Python with poor GPU utilization. The conv1d approach runs in a single CUDA kernel,
scaling to arbitrary B, T, N with no Python loop overhead.

### Config fields

```python
isi_refractory_window: int = 5
isi_kernel_type: str = "exponential"   # "exponential" | "rectangular"
isi_penalty_weight: float = 1.0
```

### Migration

Replace `inter_spike_interval_loss()` entirely. The new implementation covers all
neurons and batch elements (no sampling), penalizes refractory violations directly
(differentiable) rather than targeting a non-differentiable CV statistic, and reports
burst fraction and mean ISI as diagnostics.

---

## 6. Membrane Potential Regularization

### Definition

Prevent membrane potential explosion by penalizing potentials that exceed a threshold.
Use a quadratic penalty on the excess above `max_membrane`.

```python
class MembraneRegularization(LossTerm):
    def __init__(self, max_membrane=1.5):
        super().__init__()
        self.max_membrane = max_membrane  # default: 1.5 * V_thresh

    def forward(self, spikes, membrane, targets, **kwargs):
        if membrane is None:
            zero = torch.tensor(0.0, device=spikes.device)
            return zero, {"skipped": 1.0}

        # membrane: (B, T, N) or (B, N)
        excess = F.relu(membrane.float().abs() - self.max_membrane)
        loss = (excess ** 2).mean()

        with torch.no_grad():
            diag = {}
            diag["mean_membrane"] = membrane.float().mean().item()
            diag["max_membrane"] = membrane.float().abs().max().item()
            diag["explosion_frac"] = (membrane.float().abs() > self.max_membrane).float().mean().item()

        return loss, diag
```

### Config fields

```python
membrane_max: float = 1.5   # multiple of V_thresh; set to 1.5 * V_thresh
```

### Migration

Replace `membrane_potential_regularization()`. The existing function returns
`excess.mean()` (linear penalty). Change to `(excess ** 2).mean()` (quadratic) to
penalize large violations more heavily. Add explosion-fraction diagnostic.

---

## 7. AMP Hardening Rules

### Mandatory fp32 Operations

Under `torch.cuda.amp.autocast`, certain operations must be forced to fp32 to prevent
silent numerical corruption. Apply these rules to every loss term.

**Rule 1: Accumulate counts in fp32.**

```python
# WRONG: accumulates in float16 under autocast
counts = spikes.sum(dim=1)

# CORRECT: explicit fp32 cast before reduction
counts = spikes.float().sum(dim=1)
```

**Rule 2: All softmax and log-softmax in fp32.**

```python
# WRONG: autocast may run softmax in float16
log_probs = F.log_softmax(counts / temp, dim=-1)

# CORRECT: counts is already fp32 from Rule 1; verify dtype assertion
assert counts.dtype == torch.float32
log_probs = F.log_softmax(counts / temp, dim=-1)
```

**Rule 3: Clamp before log.**

```python
# WRONG: direct log on probabilities that may be zero
log_p = probs.log()

# CORRECT: clamp to eps floor
log_p = probs.clamp(min=1e-7).log()
```

**Rule 4: Temperature scaling before softmax, not after.**

```python
# WRONG: dividing log_softmax output by temperature
log_probs = F.log_softmax(counts, dim=-1) / temp

# CORRECT: divide input by temperature
log_probs = F.log_softmax(counts / temp, dim=-1)
```

**Rule 5: Loss values remain fp32.**

Never cast a loss tensor to float16. All `LossTerm.forward()` returns are fp32.
The composer accumulates in fp32.

**Rule 6: Reductions over T in fp32.**

Any `sum`, `mean`, or `var` along the time dimension must operate on fp32 tensors.
Call `.float()` before the reduction.

**Rule 7: ISI convolution kernel in fp32.**

Register the refractory kernel as a buffer (inherently fp32). Cast the spike input
to fp32 before `F.conv1d`. The convolution output will be fp32.

### Testing Protocol

Run each loss term under `torch.cuda.amp.autocast` for T in {10, 25, 50}. For each
configuration, verify:

```python
with torch.cuda.amp.autocast():
    loss, diag = term(spikes_fp16, membrane_fp16, targets)

assert torch.isfinite(loss), f"Loss is not finite: {loss}"
assert loss.item() > 0 or loss.item() == 0, f"Loss is negative: {loss}"

loss.backward()
for p in model.parameters():
    if p.grad is not None:
        assert torch.isfinite(p.grad).all(), "Non-finite gradient detected"
        assert p.grad.abs().sum() > 0, "Zero gradient (dead term)"
```

### Common failure modes

- **float16 underflow in eps:** `eps=1e-8` is below float16 min normal (`6e-5`).
  Cast to fp32 first, then use any eps value safely.
- **Overflow in large-T counts:** T=1000, rate=0.5 gives counts up to 500.
  `softmax(500)` overflows float16. Always accumulate counts in fp32.
- **Grad scaler interaction:** GradScaler multiplies loss by ~1024+. Keep all loss
  computation in fp32 to prevent overflow in scaled intermediate values.

---

## 8. Loss Normalization and Scaling

### Normalize by T

Spike counts scale linearly with T. A network trained at T=25 produces counts roughly
half as large as the same network at T=50. Without normalization, the ProbSpikes loss
magnitude changes with T, requiring per-T weight tuning.

Normalize counts by dividing by T:

```python
counts = spikes.float().sum(dim=1) / T  # (B, C), normalized to [0, 1]
```

This makes counts represent firing rates (fraction of timesteps with spikes) rather
than absolute counts. Loss magnitude becomes independent of T.

### Temperature auto-scaling

When using unnormalized counts (raw spike counts, not divided by T), set temperature
proportional to sqrt(T) to stabilize softmax entropy:

```python
effective_temp = temperature * math.sqrt(T) if auto_scale_temp else temperature
```

With normalized counts (rate-based), fixed temperature works across T values.
Prefer normalized counts with fixed temperature over raw counts with auto-scaled
temperature.

### Gradient magnitude monitoring

Optionally log gradient norms per loss term using backward hooks:

```python
def register_grad_hooks(composer, log_fn):
    for name, term in composer.terms.items():
        for pname, param in term.named_parameters():
            param.register_hook(
                lambda grad, n=f"{name}/{pname}": log_fn(f"grad_norm/{n}", grad.norm().item())
            )
```

Monitor for gradient imbalance: if one term's gradient norm is 100x larger than
another, the smaller term is effectively silenced. Adjust weights to bring gradient
norms within 10x of each other.

---

## 9. Logging Contract

### Per-forward output

Every call to `SNNLossComposer.forward()` returns a flat diagnostics dict with the
following keys:

```
loss/total                          — scalar, total weighted loss
loss/{term_name}_raw                — scalar, unweighted loss for each term
loss/{term_name}_weighted           — scalar, weight * raw loss for each term
diag/{term_name}/{stat_name}        — scalar, per-term diagnostic statistics
```

### Diagnostic keys by term

```
diag/spike_rate/mean_rate, max_rate, min_rate, dead_neuron_frac, saturated_frac
diag/prob_spikes/mean_count, max_count, output_entropy, accuracy
diag/temporal/mean_temporal_variance, max_temporal_variance
diag/isi/mean_isi, burst_fraction, mean_refractory_penalty
diag/membrane/mean_membrane, max_membrane, explosion_frac
```

### Format

Use flat string keys with `/` separators for TensorBoard and WandB compatibility.
All values are Python floats (not tensors). Log the entire dict each training step:

```python
total_loss, log_dict = composer(spikes, membrane, targets)
for key, value in log_dict.items():
    writer.add_scalar(key, value, global_step)
```

Optionally register backward hooks (section 8) to add
`grad_norm/{term_name}/{param_name}` keys. Enable via `log_grad_norms: bool = False`
in config. Disable in production training due to hook overhead.

---

## 10. Anti-Patterns

**Computing counts in float16.**

```python
# WRONG
counts = spikes.sum(dim=1)  # inherits dtype from spikes; float16 under AMP
```

Under AMP, `spikes` may be float16. Summing T=100 timesteps of binary spikes in
float16 is numerically fine (max count 100, well within float16 range), but the
subsequent `softmax` and `log` operations on these counts can overflow or underflow.
Always cast to fp32 before reduction: `spikes.float().sum(dim=1)`.

**ISI with Python loops.**

```python
# WRONG: O(B * N * T) in Python with per-element torch.where calls
for b in range(batch_size):
    for n in range(num_neurons):
        spike_times = torch.where(spikes[:, b, n] > 0)[0]
```

This pattern saturates the Python interpreter and starves the GPU. Replace with
the conv1d approach (section 5) for O(1) Python overhead regardless of B, N, T.

**Forgetting to normalize by T.**

```python
# WRONG: loss magnitude scales with T
counts = spikes.float().sum(dim=1)  # counts grow with T
loss = F.cross_entropy(counts / temp, targets)
```

A model trained at T=25 with weight=0.1 needs weight=0.05 at T=50 to maintain
the same effective penalty. Normalize counts by T, or auto-scale temperature.

**Temperature too low with large T.**

```python
# WRONG: counts=250 (T=500, rate=0.5), temp=0.1 -> softmax(2500) overflows
scaled = counts / 0.1
```

With raw counts and low temperature, the softmax input can overflow even in fp32
(exp(2500) is infinite). Either normalize counts by T first or raise temperature.

**Not logging per-term magnitudes.**

When total loss stalls, there is no way to determine which term dominates without
per-term logging. One term at magnitude 10.0 with weight 0.01 (effective 0.1) and
another at magnitude 0.001 with weight 1.0 (effective 0.001) means the first term
drives all gradients. Log both raw and weighted magnitudes.

**Mixing (T, B, N) and (B, T, N) in loss inputs.**

```python
# WRONG: passing time-first spikes to a batch-first loss term
spikes_tbf = snn_output  # (T, B, N) from legacy SNN core
loss = prob_spikes(spikes_tbf, membrane, targets)  # expects (B, T, N)
```

The sum over dim=1 would sum over the batch dimension instead of time, producing
nonsensical counts. Always verify tensor layout at the loss boundary. Add a shape
assertion:

```python
assert spikes.shape[0] == targets.shape[0], (
    f"Batch dim mismatch: spikes {spikes.shape} vs targets {targets.shape}. "
    f"Expected (B, T, N) spikes with B={targets.shape[0]}."
)
```

---

## Appendix: Default Composer Configuration

```python
default_composer = SNNLossComposer(terms={
    "prob_spikes": (ProbSpikesLoss(temperature=1.0, mode="softmax"), 1.0),
    "spike_rate":  (SpikeRateRegularization(target_rate=0.1), 0.1),
    "temporal":    (TemporalConsistencyLoss(window_size=5, penalty_type="l2"), 0.01),
    "isi":         (ISIRegularization(refractory_window=5, kernel_type="exponential"), 0.01),
    "membrane":    (MembraneRegularization(max_membrane=1.5), 0.001),
})

# Usage:
total_loss, log_dict = default_composer(spikes, membrane, targets)
total_loss.backward()
```

Weights are starting points. Tune by monitoring `loss/{name}_weighted` magnitudes
in TensorBoard and adjusting until no single term dominates by more than 10x.
