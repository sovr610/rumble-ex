# DCLS-Style Learnable Delays for Spiking Neural Networks

Reference for the `delays-heterogeneous-tau` skill. Covers the theory, algorithm, and
implementation architecture needed to upgrade `AdvancedLIFNeuron` from soft-attention
delay taps to Dilated-Convolution with Learnable Spacings (DCLS) Gaussian interpolation.

Source paper: Hammouamri et al. (2023/2024), "Learning Delays in Spiking Neural Networks
using Dilated Convolutions with Learnable Spacings."

---

## 1. Theory: Axonal and Synaptic Delays in SNNs

### Why Delays Matter

Biological neural circuits rely on temporal structure. Neurons encode information not
only in firing rate but in the precise timing of spikes relative to each other and to
ongoing oscillations. Learnable delays are the mechanism through which a spiking network
can exploit this temporal dimension during training.

A delay between a pre-synaptic spike and its effect on the post-synaptic membrane
potential implements a temporal filter. Different delay values select different phases of
the input spike train. When delays vary across synapses, each post-synaptic neuron
effectively integrates a weighted superposition of time-shifted copies of its input —
a continuous-time matched filter.

Patterns that are invisible to rate-based integration become separable when delays are
tuned to align the relevant spikes at the same moment. This is the computational
mechanism behind polychronization (Izhikevich 2006) and spike-timing-dependent sequence
detection.

### Biological Motivation

In the brain, delays arise from two sources:

**Axonal conduction velocity.** Myelination thickness and axon diameter determine how
fast an action potential propagates. Thin unmyelinated axons introduce delays of tens
of milliseconds; thick myelinated axons carry spikes in under one millisecond. The
distribution across a typical cortical population spans roughly 1–50 ms, equivalent to
1–50 discrete timesteps at 1 kHz simulation resolution.

**Synaptic delay.** Even after the action potential arrives at the axon terminal, release
of neurotransmitter, diffusion across the synaptic cleft, and receptor activation take
0.5–5 ms. This is irreducible regardless of conduction speed.

The combined effect is a per-synapse delay d_ij that the network uses as a free
parameter. Learning d_ij is equivalent to learning the wiring geometry of the circuit.

### Discrete-Time Equivalence

At timestep resolution Δt, a delay of d timesteps maps a pre-synaptic spike train
S_j[t] to a contribution at time t − d. The synaptic input to neuron i is:

    I_i[t] = Σ_j  w_ij * S_j[t − d_ij]

This is precisely a 1D temporal convolution of S_j with a unit-impulse kernel positioned
at tap d_ij, scaled by weight w_ij. The DCLS framework generalizes the integer tap to a
continuous real-valued position, making d_ij differentiable and therefore learnable via
standard backpropagation.

---

## 2. DCLS-Style Learnable Delays

### Core Idea

Store each delay as a real-valued parameter d_ij in [0, Td−1]. During the forward pass,
distribute the synaptic contribution across the two (or more) nearest integer delay
bins using a Gaussian interpolation kernel. This makes the output a smooth function of
d_ij, enabling gradient flow.

During inference, round d_ij to the nearest integer for zero-overhead computation with
a pre-computed sparse kernel.

### Training Forward Pass

Define the Gaussian interpolation weight for integer bin n given continuous delay d and
bandwidth σ:

    g(n; d, σ) = exp(−(n − d)² / (2σ²))

Normalize over the active range [0, Td−1]:

    Z = Σ_{n=0}^{Td-1} g(n; d, σ)
    g_norm(n; d, σ) = g(n; d, σ) / Z

The synaptic current to neuron i at time t is:

    I_i[t] = Σ_j  w_ij * Σ_{n=0}^{Td-1}  g_norm(n; d_ij, σ) * S_j[t − n]

During training, only K bins closest to d_ij have non-negligible weight. Setting K=3
(bins floor(d)−1, floor(d), floor(d)+1) captures >99% of the Gaussian mass when
σ ≤ 1.0. Bins outside [0, Td−1] are masked to zero before normalization.

### Inference Forward Pass

At inference time, discretize:

    d_discrete_ij = clamp(round(d_ij), 0, Td−1)

Use the sparse single-tap form:

    I_i[t] = Σ_j  w_ij * S_j[t − d_discrete_ij]

This is equivalent to integer-indexed gather from the spike history ring buffer —
no floating-point interpolation cost.

### σ Annealing Schedule

σ controls the smoothness of the delay kernel. A large σ provides wide gradient support
early in training so d_ij can move freely. Decay σ toward a small value so the kernel
sharpens and d_ij converges to a stable integer-vicinity value.

Exponential schedule:

    σ(epoch) = σ_end + (σ_start − σ_end) * exp(−epoch / σ_decay_epochs)

Recommended defaults:

    σ_start        = 1.0
    σ_end          = 0.5
    σ_decay_epochs = total_epochs / 3

Call `update_sigma(epoch)` at the start of each epoch, before the forward pass.
Store σ as a module buffer (not a parameter) so it is saved and restored by
`state_dict()` but not updated by the optimizer.

### Delay Parametrization

Use an unconstrained raw parameter d_raw_ij in ℝ and map to the valid range:

    d_ij = (Td − 1) * sigmoid(d_raw_ij)

After each optimizer step, optionally hard-clamp:

    d_ij = clamp(d_ij, 0, Td − 1)

The sigmoid mapping already keeps d_ij in (0, Td−1) in the forward pass, but the
clamp ensures exact boundary enforcement after gradient accumulation moves d_raw.

### Initialization

Uniform initialization spreads delays across the full range, avoiding symmetry:

    d_raw_init = logit(uniform(0, 1))   → d_ij ~ Uniform(0, Td−1)

Alternatively, initialize near the midpoint with some spread:

    d_raw_init ~ Normal(0, 0.5)         → d_ij concentrated near (Td−1)/2

Uniform is preferred when Td ≥ 8; midpoint initialization is better for Td ≤ 4.

---

## 3. Granularity Modes

The delay parameter tensor d has shape determined by the granularity mode. All modes
broadcast into the full (out_features, in_features) space during the forward pass.

### per_synapse

    d shape: (out_features, in_features)

One delay per synapse. Most expressive. Allows complete decoupling of temporal
selectivity at each output-input pair. Parameter count scales as O(out × in), which
matches the weight matrix. Use when temporal precision is the primary objective and
parameter budget permits.

### per_output

    d shape: (out_features, 1)

One delay per output neuron, shared across all inputs. Each output neuron selects a
single lag at which to integrate all its inputs simultaneously. Reduces parameter count
to O(out). This is the recommended default for most architectures — it provides a good
balance between expressiveness and regularization.

### per_input

    d shape: (1, in_features)

One delay per input feature, shared across all output neurons. Each input channel is
uniformly shifted before contributing to any output. Useful when the input has known
temporal structure (e.g., auditory filterbank outputs with known latencies). Parameter
count O(in).

### per_block

    d shape: (out_blocks, in_blocks)

Block-wise delay sharing. Partition out_features into out_blocks groups and in_features
into in_blocks groups. Each (out_block, in_block) pair shares one delay value. The
delay tensor is broadcast into (out_features, in_features) by repeating each block
value across its member rows and columns. Parameter count O(out_blocks × in_blocks).
This mode is appropriate when layer width is very large and per_synapse is prohibitively
expensive.

### Config Field

Add `delays_granularity: str` to `SNNConfig` with values:

    "per_synapse" | "per_output" | "per_input" | "per_block"

Default: `"per_output"`.

### Ablation Guidance

Run ablation in this order when debugging temporal coding capacity:

1. Start with `per_output` as baseline.
2. Upgrade to `per_synapse` if the task requires fine-grained temporal selectivity.
3. Downgrade to `per_input` or `per_block` if memory is constrained and per_synapse
   provides no measurable benefit on validation metrics.

---

## 4. Efficient Forward Implementations

### Option A: Bin-Accumulation (Recommended for Training)

Avoid materializing the full (out × in × Td) kernel tensor. Instead, compute a set of
K sparse weight matrices, one per nearby integer bin.

**Step 1.** Expand d from its granularity shape to (out_features, in_features) by
broadcasting. Call the result d_full.

**Step 2.** For each of the K nearest integer bins n_k in {floor(d)−1, floor(d), floor(d)+1}:

    - Compute Gaussian weight: g_k = exp(−(n_k − d_full)² / (2σ²))
    - Mask bins outside [0, Td−1] to zero.
    - Build bin weight matrix: W_bin[k] = w * g_k  where w is the synapse weight matrix.

**Step 3.** Normalize across K bins (not across Td):

    Z = Σ_k g_k   (element-wise sum, shape (out, in))
    W_bin[k] = W_bin[k] / Z   (avoid double-counting; optional if K=Td)

**Step 4.** Shift the spike vector for each integer delay n and apply the corresponding
bin weight matrix:

    I[t] = Σ_k  spike_history[:, n_k, :] @ W_bin[k].T

`spike_history` has shape (B, Td, in_features). Index `n_k` directly (it is an
integer tensor). The result I[t] has shape (B, out_features).

**Memory cost.** K matrices of shape (out, in) held in float32. With K=3, this is
3 × out × in × 4 bytes. For a 512→512 layer, K=3, that is 3 MB — acceptable.

**Compute cost.** K batched matrix multiplications of size (B × out × in). Linear in
both K and Td. No extra cost from increasing Td beyond the ring buffer storage.

**Implementation note.** Compute W_bin matrices once per forward call (not per timestep
when unrolling T timesteps). When unrolling over T timesteps inside a sequence loop,
re-use the same W_bin matrices and only re-index into spike_history.

### Option B: Ring-Buffer Gather (Low-Latency Inference)

Maintain a circular ring buffer of shape (B, Td, in_features). At each new timestep:

1. Write the new spike vector into the next buffer slot (index = current_step % Td).
2. For integer delays: index directly as `buffer[:, (current_step - n) % Td, :]`.
3. For smoothed delays (K-neighbor blend): gather K neighboring slots, weight by
   Gaussian, sum.

This pattern is optimal for streaming inference on fixed-rate event streams. It avoids
materializing the full spike history as a contiguous tensor. The drawback is that
indexing with modular arithmetic is harder to vectorize across the time dimension during
training, so Option A is preferred for batch training.

Use Option B for deployment behind a real-time interface where latency per timestep
matters more than throughput.

### Memory Guards

Enforce these limits to prevent silent memory explosion:

    Td_max  = 32   (hard cap; warn when Td > 32, error when Td > 64)
    K_max   = 3    (interpolation bins; K=5 is rarely beneficial)
    param_threshold = out_features * in_features * Td_max * 4 bytes

At module construction time, check:

    if (out_features * in_features * Td) > 1e8:
        warn("Delay module may use >400 MB; consider reducing granularity or Td")

Add profiler hooks to measure:

    delay_overhead_factor = time_with_delays / time_without_delays

Log this ratio at the end of the first training epoch. Target: < 2.0. If overhead
exceeds 3.0, reduce K or switch to per_output granularity.

---

## 5. Delay Module Architecture

Define a three-class hierarchy. All three share the delay computation logic through
a common mixin or base class.

```
DelayModuleBase (nn.Module, abstract)
├── DelayLinear       — wraps nn.Linear for fully-connected SNN layers
├── DelayConv1d       — wraps nn.Conv1d for temporal convolution layers
└── DelayRecurrent    — wraps recurrent synaptic connections in AdvancedLIFNeuron
```

### DelayModuleBase

Attributes:

    d_raw        : nn.Parameter  — unconstrained delay, shape depends on granularity
    sigma        : buffer (float) — current interpolation bandwidth
    sigma_config : dict          — {sigma_start, sigma_end, sigma_decay_epochs}
    granularity  : str           — one of the four modes
    Td           : int           — max delay in timesteps
    K            : int           — number of Gaussian bins (default 3)
    use_discrete_inference : bool — if True, round delays at inference time

Methods:

    d_continuous() -> Tensor     — apply sigmoid mapping, return d in [0, Td-1]
    d_effective()  -> Tensor     — d_continuous() in train, rounded in inference
    gaussian_weights(d, n_bins) -> (Tensor, Tensor)  — bin indices and weights
    update_sigma(epoch: int)    — recompute sigma from schedule, store to buffer
    forward(spikes_history, weight) -> Tensor  — must be implemented by subclass

### DelayLinear

Wraps `nn.Linear`. The `weight` is stored in the Linear submodule. The forward
receives the full spike history tensor and computes bin-accumulation as described in
Option A. Output is the delayed synaptic current vector of shape (B, out_features).

### DelayConv1d

Wraps `nn.Conv1d` where the kernel dimension is the delay axis. The continuous delay
d shifts the effective center of the convolution kernel. During training, apply the
Gaussian-weighted superposition of K shifted kernels. During inference, shift the kernel
by the rounded integer delay using standard Conv1d with appropriate padding.

Useful for layers that already process sequences as (B, C, T) tensors (e.g., encoder
output before SNN integration).

### DelayRecurrent

Used specifically inside `AdvancedLIFNeuron` to replace the existing `apply_delays`
method. Receives `spike_history` of shape (B, Td, in_features) and the scalar coupling
strength (or a weight matrix for full recurrent connections). Returns delayed current
of shape (B, out_features).

This is the immediate upgrade target for the current codebase.

---

## 6. σ Schedule Management

### Storage

Register σ as a buffer, not a parameter:

    self.register_buffer("sigma", torch.tensor(sigma_start))

This ensures:
- σ appears in `state_dict()` and is restored on checkpoint load.
- The optimizer does not update σ.
- σ moves to the correct device with `.to(device)`.

### Update Protocol

Call `update_sigma(epoch)` once per epoch, before the first batch of that epoch. Never
call it inside the batch loop — σ should be constant within an epoch.

    def update_sigma(self, epoch: int) -> None:
        cfg = self.sigma_config
        new_sigma = cfg["sigma_end"] + (cfg["sigma_start"] - cfg["sigma_end"]) * \
                    math.exp(-epoch / cfg["sigma_decay_epochs"])
        self.sigma.fill_(new_sigma)

### Checkpoint Resume

When resuming from a checkpoint at epoch E, load the state_dict first (which restores
the stored σ value), then call `update_sigma(E)` to verify consistency. The two values
should match. If they differ by more than 1e-4, log a warning and use the computed
value (the schedule is ground truth).

### Schedule Options

    constant            — σ fixed at sigma_start; useful for ablation only
    exponential_decay   — formula above; default and recommended
    linear_decay        — σ(epoch) = sigma_start - (sigma_start - sigma_end) * (epoch / total_epochs)
                          less steep than exponential, use when delays need more time to settle

Add `sigma_schedule: str` to the delay config block, defaulting to `"exponential_decay"`.

---

## 7. Inference Discretization

### Rounding Procedure

At inference time (after calling `model.eval()`), convert continuous delays to integers:

    d_discrete = clamp(round(d_continuous()), 0, Td−1).long()

Use `d_discrete` as a direct integer index into `spike_history`:

    delayed_spikes_j = spike_history[:, d_discrete[i, j], j]

This eliminates all floating-point interpolation from the inference path, making
inference as fast as a standard SNN with fixed delays.

### Controlling Discretization

The flag `use_discrete_inference` (default `True`) switches between training and
inference computation modes. Disable for analysis (e.g., to compare continuous vs.
discrete outputs on the same batch):

    model.eval()
    module.use_discrete_inference = False   # use Gaussian interp for comparison
    out_smooth = model(x)

    module.use_discrete_inference = True    # back to integer-tap mode
    out_discrete = model(x)

### Validation Check

After training, run this sanity check on a held-out batch:

    mse = mean_squared_error(out_smooth, out_discrete)

Acceptable MSE: < 0.01 × variance(out_smooth). If MSE is large, σ_end was too large
at the end of training (delays did not sharpen enough). Reduce σ_end or extend
σ_decay_epochs and retrain.

Large degradation from continuous to discrete output is the primary indicator of a
poorly converged delay schedule.

---

## 8. Delay Logging and Diagnostics

Attach these diagnostics to the training loop, logging at the end of each epoch.

### Delay Histogram

For each delay module, compute the histogram of d_continuous values over all output-input
pairs. Log as a tensor of bin counts over B=Td bins (one bin per integer delay step).

A healthy histogram is roughly uniform or bimodal (short delays and long delays, driven
by the task). A spike at bin 0 or bin Td−1 indicates boundary pressure.

### Per-Layer Mean and Std

    d_mean = d_continuous().mean()
    d_std  = d_continuous().std()

Log these as scalars. d_std < 0.5 timestep across a layer means delays have collapsed
to a single value — the layer is not using its temporal capacity.

For per_block or per_output granularity, also log per-block or per-output-neuron
statistics.

### Delay Entropy

Bin d values into Td integer bins, compute occupancy fractions p_n, and compute:

    entropy = -Σ_n  p_n * log(p_n + 1e-8)   (nats)
    max_entropy = log(Td)

Report `entropy / max_entropy` as a normalized score in [0, 1].

Score near 1.0 → delays are diverse and spread across the range (healthy).
Score near 0.0 → all delays collapsed to one or two bins (degenerate).

If entropy < 0.3 by the midpoint of training, increase σ_start or add a mild entropy
regularization term to the loss:

    L_reg = −λ_delay * entropy(d)   (maximization regularizer, λ_delay ~ 1e-3)

### Boundary Pressure

Count the fraction of d values within 0.5 timestep of the boundaries:

    frac_low  = (d < 0.5).float().mean()
    frac_high = (d > Td − 1.5).float().mean()
    boundary_pressure = frac_low + frac_high

Boundary pressure > 0.3 means many delays have saturated the range limits. Consider
increasing Td, or inspect whether the task requires delays longer than the current cap.

### σ Trace

Log the current σ value each epoch alongside the loss curve. σ should decrease
monotonically. A σ that stops decreasing (due to a bug in `update_sigma`) will prevent
delays from sharpening.

---

## 9. Migration from the Current Implementation

### Current State

`AdvancedLIFNeuron.apply_delays` uses soft-attention over delay taps:

    delay_attn = softmax(delay_weights, dim=-1)   # (size, max_delay)
    delayed = einsum('bdn,nd->bn', recent_spikes, delay_attn)

This is a smooth weighted average over the full history window. It does not converge
to a specific delay value because softmax over Td bins distributes weight across all
taps at all times. Gradients flow through all taps equally, preventing the network from
specializing to a narrow temporal offset.

The parameters are `delay_weights: (size, max_delay)` — no separation between delay
position and weight magnitude.

### Target State

Replace with `DelayRecurrent` module holding:

    d_raw : (size, 1)  for per_output granularity   (d_raw : (size, size) for per_synapse)
    sigma  : buffer

The forward passes through Gaussian-interpolated bin-accumulation, converging d toward
a specific integer delay as σ decays.

### Migration Steps

**Step 1.** Implement `DelayModuleBase` and `DelayRecurrent` in
`brain_ai/core/delay_modules.py`. Write unit tests verifying:
- Gaussian weights sum to 1.0 after normalization.
- At σ → 0, output matches the single-tap integer-delay output.
- `update_sigma` correctly matches the schedule formula.

**Step 2.** Add `use_dcls_delays: bool = False` flag to `SNNConfig`. Wire
`DelayRecurrent` into `AdvancedLIFNeuron.__init__` when this flag is True. Both code
paths coexist:

    if config.use_dcls_delays:
        self.delay_module = DelayRecurrent(...)
    elif self.use_delays:
        self.delay_weights = nn.Parameter(...)   # legacy path

**Step 3.** Update `AdvancedLIFNeuron.apply_delays` to dispatch on the flag:

    if hasattr(self, 'delay_module'):
        delayed_input = self.delay_module(self.spike_history, coupling)
    else:
        # existing softmax-attention path
        ...

**Step 4.** Run training on a small benchmark (MNIST-DVS or SHD) with
`use_dcls_delays=True`. Compare validation accuracy and delay entropy metrics against
the softmax-attention baseline. Expect equal or better accuracy and higher delay entropy.

**Step 5.** Once `use_dcls_delays=True` is validated, deprecate the softmax-attention
path. Mark `delay_weights` parameter with a DeprecationWarning in `__init__`. Keep the
old path for one release cycle to allow checkpoint compatibility.

**Step 6.** Remove the softmax-attention path after the deprecation window. Delete
`delay_weights` parameter, `get_delay_distribution`, and `get_effective_delays` methods.
Replace `get_effective_delays` with `delay_module.d_continuous().mean(dim=-1)`.

---

## 10. Anti-Patterns

**Do not materialize the dense (out × in × Td) kernel tensor.** Computing the full
Gaussian-weighted kernel for every synapse at every delay tap consumes O(out × in × Td)
memory. For a 512→512 layer with Td=32 in float32, this is 512 × 512 × 32 × 4 = 32 MB
per layer, which accumulates rapidly in deep networks. Use the bin-accumulation approach
(K=3 sparse matrices) instead.

**Do not apply softmax over delay bins.** The existing softmax-attention implementation
distributes weight across all delay taps every forward pass. This prevents any single
bin from becoming dominant, so the effective delay never converges to a specific value.
DCLS-style delays use a normalized Gaussian that sharpens as σ decays — the distinction
is that the Gaussian is centered on a learnable continuous position d, while softmax
treats all positions as competing logits with no positional structure.

**Do not share a single σ buffer across layers.** Each layer may reach its optimal delay
resolution at different training stages. A shallow layer processing simple temporal
features may converge its delays early; a deeper layer integrating complex sequences may
need more exploration time. Store a separate σ buffer per `DelayModuleBase` instance.
A global σ annealer that overrides per-layer buffers defeats the purpose of per-layer
sigma_config.

**Do not skip σ annealing.** Running with constant σ = σ_start throughout training
leaves delays soft and diffuse at inference time, causing large discretization error
(Section 7). Running with constant σ = σ_end from the start provides insufficient
gradient support early in training and causes delays to stagnate near initialization
values. The schedule is not optional.

**Do not forget to clamp d_raw after gradient updates.** The sigmoid parametrization
ensures d_continuous stays in (0, Td−1) during the forward pass, but accumulated
floating-point rounding in d_raw can push d_continuous to 0.0 or exactly Td−1, which
creates zero-gradient boundary conditions for the sigmoid. After each optimizer step,
apply:

    with torch.no_grad():
        module.d_raw.clamp_(-6.0, 6.0)   # sigmoid(-6) ≈ 0.002, sigmoid(6) ≈ 0.998

This prevents d_raw from wandering to large magnitudes where the sigmoid saturates.

**Do not use delays without a spike history buffer in the module state.** Applying delay
logic to a zero-initialized or stale history buffer produces incorrect currents at the
start of each sequence. Call `reset_mem()` (or the equivalent `reset_state()` on the
delay module) at the beginning of each new batch or sequence episode. Forgetting this
step causes the first Td timesteps of every batch to be corrupted by residual history
from the previous batch.

---

## Appendix: Config Reference

Additions to `SNNConfig` for DCLS delays:

```python
use_dcls_delays: bool = False
delays_granularity: str = "per_output"   # per_synapse | per_output | per_input | per_block
max_delay: int = 16                       # Td; existing field, reused
dcls_sigma_start: float = 1.0
dcls_sigma_end: float = 0.5
dcls_sigma_decay_epochs: int = 30         # set to ~total_epochs / 3
dcls_K: int = 3                           # number of Gaussian interpolation bins
dcls_use_discrete_inference: bool = True
dcls_delay_entropy_reg: float = 0.0       # set > 0 to add entropy regularization
```

Delay module constructor signature (DelayRecurrent for AdvancedLIFNeuron):

```python
DelayRecurrent(
    in_features: int,
    out_features: int,
    Td: int,
    granularity: str,
    sigma_start: float,
    sigma_end: float,
    sigma_decay_epochs: int,
    K: int = 3,
    use_discrete_inference: bool = True,
)
```

Existing fields preserved without change:

```python
use_learnable_delays: bool = True    # legacy soft-attention path flag
max_delay: int = 16                  # reused as Td
```

When `use_dcls_delays=True`, `use_learnable_delays` is ignored for the AdvancedLIFNeuron
delay path. Both flags may coexist in the config for checkpoint compatibility.
