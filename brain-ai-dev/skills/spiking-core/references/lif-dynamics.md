# LIF Family Dynamics — Reference for Spiking Core Skill

This document specifies the governing equations, numerical policies, and correctness requirements for the LIF neuron family implemented in `brain_ai/core/neurons.py`. Use this as the canonical reference when auditing, extending, or debugging spiking neuron code.

---

## 1. LIFNeuron — Core Update Equations

The Leaky Integrate-and-Fire (LIF) neuron is the foundational unit of the spiking core. Each forward step executes three operations in sequence:

```
i_t = W_in @ x_t + b         # synaptic input current
v_t = β * v_{t-1} + i_t      # membrane integration
s_t = H(v_t - v_th)          # Heaviside spike decision
```

Where:
- `x_t` is the input tensor at timestep `t` with shape `(batch, in_features)`
- `i_t` is the resulting synaptic current (pre-spike)
- `v_t` is the membrane potential (post-integration, pre-reset)
- `s_t` is the binary spike output — exactly 0.0 or 1.0
- `H(·)` is the Heaviside step function, approximated in the backward pass by a surrogate gradient

Apply one of two reset rules immediately after computing `s_t`:

### 1.1 Subtractive Reset (Default)

```
v_t = v_t - s_t * v_reset
```

The membrane voltage is reduced by `v_reset` at every spiking unit. When `v_reset == v_th`, this exactly returns the neuron to threshold — but any overshoot above `v_th` is preserved in the residual. This is the correct default for gradient-based training because:

- The overshoot encodes information about input magnitude, which can propagate through subsequent timesteps.
- Gradients flow more smoothly: the membrane state is never hard-zeroed, so the Jacobian of `v_t` with respect to `v_{t-1}` remains non-zero even after a spike.
- Empirically, subtractive reset converges faster on classification tasks.

Use subtractive reset for all experiments involving surrogate-gradient backpropagation.

### 1.2 Hard Reset (Zero Reset)

```
v_t = (1 - s_t) * v_t
```

Every neuron that fired is forced to exactly zero, discarding the overshoot. This matches the behavior of most neuromorphic hardware implementations (Intel Loihi, BrainScaleS) and produces cleaner, more interpretable membrane traces. The trade-off is that gradient flow is weaker — when a neuron spikes, its membrane is zeroed, which means gradients from that timestep cannot propagate backward through the membrane state.

Use hard reset when deploying to neuromorphic hardware or when spike timing reproducibility is more important than training efficiency.

The existing code selects between these via `reset_mechanism='subtract'` (default) or `reset_mechanism='zero'`.

### 1.3 Parameter Reference

| Parameter | Symbol | Default | Learnable | Valid Range | Notes |
|---|---|---|---|---|---|
| Membrane decay | β | 0.9 | Optional | (0, 0.999] | Must be strictly less than 1.0. See stability policy. |
| Spike threshold | v_th | 1.0 | Optional | (0, ∞) | Learnable threshold requires surrogate gradient chain to include ∂s/∂v_th. |
| Reset voltage | v_reset | 1.0 | No | [0, v_th] | Only used when reset_mechanism='subtract'. |

---

## 2. AdaptiveLIFNeuron — Dynamic Threshold

The adaptive LIF neuron adds a scalar adaptation variable `a_t` that tracks recent spike history and raises the effective threshold. This models spike-frequency adaptation — a well-documented biophysical phenomenon driven by slow K+ currents and Na+ channel inactivation.

### 2.1 Update Equations

```
a_t = ρ * a_{t-1} + s_{t-1}       # adaptation variable (updated on previous spike)
v_th_eff = v_th + α * a_t          # effective threshold rises after spikes
i_t = W_in @ x_t + b
v_t = β * v_{t-1} + i_t
s_t = H(v_t - v_th_eff)            # decision uses effective threshold
```

The adaptation variable `a_t` accumulates whenever `s_{t-1} = 1` and decays at rate `ρ` otherwise. A neuron that fires repeatedly accumulates a large `a_t`, which raises `v_th_eff`, making each subsequent spike harder to trigger. Once the neuron goes silent, `a_t` decays exponentially and the threshold returns to baseline.

### 2.2 Why Spike-Frequency Adaptation Matters

Without adaptation, high-gain inputs drive neurons into saturation — they fire every timestep and convey no information about input variation. Adaptation enforces sparse, temporally structured spiking:

- Neurons respond strongly to the onset of a stimulus (burst) and then reduce their rate even if the stimulus persists.
- Population coding becomes more efficient: downstream layers see variation rather than saturated firing.
- Pathological synchrony is suppressed. Networks with adaptation are harder to drive into epileptic-style all-or-nothing firing.

In the brain-AI system, adaptive LIF neurons are preferred for the SNN core when training on sequential inputs where temporal dynamics carry information (e.g., audio, video, time-series sensors).

### 2.3 Additional Parameters

| Parameter | Symbol | Default | Notes |
|---|---|---|---|
| Adaptation decay | ρ | 0.9 | Controls how long elevated threshold persists. Higher = longer memory of past spikes. |
| Adaptation strength | α | 1.8 | Scales how much each spike raises the threshold. Higher = stronger adaptation. |

All standard LIF parameters (β, v_th, v_reset) apply in addition to the above.

---

## 3. RecurrentLIFNeuron — Lateral Connections

The recurrent LIF neuron augments the synaptic input with a learned lateral connection matrix `W_rec` applied to the previous timestep's spike output. This enables the layer to function as an attractor network or sequence generator without stacking layers.

### 3.1 Update Equations

```
i_t = W_in @ x_t + W_rec @ s_{t-1} + b   # recurrent on spikes
v_t = β * v_{t-1} + i_t
s_t = H(v_t - v_th)
```

Note the input to `W_rec` is `s_{t-1}` — the binary spike vector from the previous timestep, not the membrane potential.

### 3.2 Recurrent on Spikes vs. Recurrent on Membrane

**Recurrent on spikes** (default): The recurrent input is `W_rec @ s_{t-1}`. Because spikes are binary, the effective recurrent drive is bounded by `||W_rec||_1`. This is far easier to stabilize than recurrent connections on continuous membrane values — gradient norms are controlled, and the Lipschitz constant of the recurrent map is tractable.

**Recurrent on membrane** (not recommended with surrogate gradients): Using `W_rec @ v_{t-1}` instead of `W_rec @ s_{t-1}` creates a continuous-valued recurrent loop. This is equivalent to a standard RNN and loses most of the benefits of the spiking formulation. Surrogate gradients are designed to handle the spike discontinuity; applying them on top of continuous recurrence creates gradient signal mismatches. Avoid this pattern in the brain-AI SNN core.

### 3.3 Weight Initialization

Initialize `W_rec` with:

```python
nn.init.normal_(self.W_rec.weight, mean=0.0, std=recurrent_weight_scale)
```

The default `recurrent_weight_scale` is small (typically 0.1 or smaller). Reasons:

- Large initial recurrent weights cause oscillatory instability — the network enters a resonance mode where spikes beget spikes with no external input.
- A small initial scale keeps the effective initial recurrent gain below 1.0, ensuring the layer behaves like a standard LIF at the start of training before `W_rec` specializes.
- Diagonal initialization (identity-like W_rec) is an alternative that initializes each neuron with self-excitation; avoid this as it creates strong attractor states that resist training signal.

---

## 4. AdvancedLIFNeuron — 2025 Research Extensions

The `AdvancedLIFNeuron` class (lines 330-532) implements three extensions beyond standard LIF that correspond to active research in neural coding and neuromorphic computation.

### 4.1 Learnable Synaptic Delays

Biological synapses have heterogeneous transmission delays (0.5–20 ms). Learnable delays allow the network to discover which temporal offset of the input is most informative for each synaptic connection.

Implementation: maintain a ring buffer of the last `max_delay` timesteps of activations. A soft attention weight vector `w_delay` (size `max_delay`, constrained to sum to 1 via softmax) is learned per synapse group. The effective input at time `t` is:

```
i_delay = einsum('d,bnd->bn', softmax(w_delay), delay_buffer)
```

This differentiable soft selection is preferable to hard integer delays because it admits gradient-based optimization. During inference, the highest-weight delay tap can be used for efficient fixed-delay execution on hardware.

### 4.2 Heterogeneous Time Constants

Standard LIF uses a single shared β for all neurons in the layer. Heterogeneous time constants assign a distinct β per neuron, allowing the layer to simultaneously maintain short-memory and long-memory units — analogous to the diversity of cortical neuron types.

Implement via a logit parameterization:

```python
self.log_beta = nn.Parameter(torch.zeros(hidden_size))
# ...
beta = torch.sigmoid(self.log_beta)   # constrained to (0, 1) automatically
```

This parameterization enforces β ∈ (0, 1) for all neurons without requiring explicit clamping after optimizer steps. The sigmoid is monotone, so gradient flow is unobstructed. When initializing, set `log_beta` to a value that maps to the desired initial β:

```python
# If target initial β = 0.9, then log_beta_init = logit(0.9) ≈ 2.197
nn.init.constant_(self.log_beta, 2.197)
```

### 4.3 Optional Adaptive Threshold

The `AdvancedLIFNeuron` supports the same adaptive threshold mechanism as `AdaptiveLIFNeuron` (Section 2), optionally enabled. When combined with learnable delays and heterogeneous time constants, the full model approaches a biologically detailed single-compartment neuron model.

Combine these features carefully: each added component increases the number of stateful variables per neuron and thus the memory footprint during BPTT. Profile memory before enabling all three in a large-scale run.

---

## 5. Numerical Stability Policies

These are hard rules. Every implementation of any LIF variant must satisfy all of them. Violations produce silent training failures — the loss may decrease slowly, or training may appear stable while the model learns pathological representations.

### Rule 1: Clamp β to (0, 0.999)

β ≥ 1.0 produces unbounded membrane growth. With β = 1.0, the membrane is a pure integrator with no leak — it accumulates input current indefinitely and never forgets past states. With β > 1.0, the membrane diverges exponentially. Both conditions cause gradient explosion through time.

Enforce this constraint in one of two ways:

**Option A — Sigmoid parameterization (preferred):**
```python
self.log_beta = nn.Parameter(torch.tensor(2.197))   # sigmoid(2.197) ≈ 0.9
# In forward:
beta = torch.sigmoid(self.log_beta) * 0.999         # max value is 0.999
```

**Option B — Post-step clamp (acceptable fallback):**
```python
with torch.no_grad():
    self.beta.clamp_(0.001, 0.999)
```
Apply this clamp at the end of every optimizer step when using Option B.

Never pass `beta = 1.0` as a default or allow it as a valid configuration value.

### Rule 2: No In-Place Operations on Autograd Tensors

In-place tensor operations on any tensor that requires gradients silently corrupt the autograd graph. PyTorch's gradient tape records references to tensors at the time of computation; an in-place mutation changes the tensor's data without creating a new node, causing the backward pass to use stale or inconsistent values.

**Forbidden patterns:**
```python
v.add_(current)           # WRONG: in-place add
v += current              # WRONG: in Python, += on a tensor calls add_
v.mul_(self.beta)         # WRONG: in-place multiply
v[mask] = 0.0             # WRONG: in-place indexed assignment
v.copy_(new_v)            # WRONG: in-place copy
```

**Correct patterns:**
```python
v = v + current           # OK: creates new tensor
v = self.beta * v         # OK: creates new tensor
v = v * (1 - spike)       # OK: creates new tensor (hard reset)
v = torch.where(mask, torch.zeros_like(v), v)   # OK: functional select
```

This rule applies to `v` (membrane potential), `a` (adaptation variable in AdaptiveLIF), `s` (spike output — never modify after assignment), and any intermediate activations that feed into the loss.

### Rule 3: fp32 State Accumulation Under AMP

When training with automatic mixed precision (`torch.cuda.amp.autocast`), the default compute dtype is bf16 or fp16. Membrane potential accumulation over many timesteps in low precision causes representational drift: small current values round to zero, and the membrane state slowly diverges from the true fp32 trajectory.

For sequences longer than 20 timesteps, enforce fp32 state accumulation:

```python
def forward(self, x, v_prev):
    # Cast state to fp32 for accumulation
    v_fp32 = v_prev.float()
    x_fp32 = x.float()

    current = self.fc(x_fp32)           # linear in fp32
    v_new = self.beta * v_fp32 + current  # accumulate in fp32

    # Spike decision in fp32, then cast back to compute dtype
    spike = self.surrogate(v_new - self.threshold)    # fp32 surrogate
    spike = spike.to(x.dtype)           # cast spike to compute dtype for downstream

    return spike, v_new                 # return fp32 membrane state
```

Never cast `v` down to bf16/fp16 between timesteps when unrolling over long sequences. The spike output (`s_t`) can be cast back to the compute dtype for downstream layers since it is binary and does not accumulate.

### Rule 4: Threshold Gradient Flow

When `v_th` is a learnable parameter, ensure the surrogate gradient correctly propagates to `v_th`. The surrogate function `sigma` approximates the derivative of the Heaviside:

```
s_t = H(v_t - v_th)    (forward)
∂s/∂v_t  ≈ sigma'(v_t - v_th)
∂s/∂v_th = -sigma'(v_t - v_th)   (by chain rule, note the negative sign)
```

The negative sign is critical. If the threshold is defined as `v_th = self.base_threshold + self.threshold_offset` and the surrogate is applied to `v - v_th`, PyTorch's autograd will correctly compute the negative gradient for `v_th` as long as the expression is written as:

```python
spike = surrogate_fn(v - self.v_th)   # autograd sees ∂/∂v_th = -surrogate'(...)
```

Do not detach `v_th` from the computation graph. Do not compute `v - v_th.detach()`. Verify gradient flow by checking `self.v_th.grad` is non-None and non-zero after a backward pass.

### Rule 5: Spike Values Must Be Binary in the Forward Pass

Spikes are exactly 0.0 or 1.0 in the forward pass. The surrogate gradient trick exists precisely because the Heaviside is non-differentiable — the forward pass uses the true step function, and the backward pass substitutes a smooth approximation.

**Forbidden:**
```python
s_t = torch.sigmoid(v - v_th)          # WRONG: soft spikes in forward
s_t = torch.clamp(v - v_th, 0, 1)      # WRONG: not binary
s_t = (v > v_th).float()               # OK forward, but breaks gradient — only use inside custom Function
```

**Correct implementation using a custom autograd Function:**
```python
class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).float()         # binary: exactly 0.0 or 1.0

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # ATan surrogate derivative: 1 / (π * (1 + (πx/2)^2))
        grad = grad_output / (1.0 + (torch.pi / 2 * x) ** 2) / torch.pi
        return grad
```

The forward function must use a threshold comparison producing {0.0, 1.0}. The backward function may use any smooth function. Never use sigmoid or tanh as the forward function.

---

## 6. Extension Hooks — Define the Interface Now

The following extensions are not yet implemented but should be planned for at the interface level. Define the method signatures and state variable names now so that future implementations are consistent across neuron classes.

### 6.1 Learnable Per-Neuron Thresholds

Define the interface as:

```python
# In __init__:
if per_neuron_threshold:
    self.v_th = nn.Parameter(torch.ones(hidden_size))
else:
    self.v_th = nn.Parameter(torch.tensor(threshold))   # scalar

# In forward:
spike = SurrogateSpike.apply(v - self.v_th)   # broadcasts correctly in both cases
```

Ensure the threshold parameter is always named `v_th` (not `threshold`, `thresh`, or `theta`) for consistency across all neuron classes. This enables unified parameter sweeps and gradient inspection.

### 6.2 Synaptic Current Models

Standard LIF uses an instantaneous current model: the input current arrives and decays with the membrane in a single equation. Biologically, synaptic currents have their own dynamics (rise and fall times). Define the interface for explicit synaptic current state:

```python
# Alpha synapse: single exponential decay
# i_t = τ_syn * i_{t-1} + x_t
# v_t = β * v_{t-1} + i_t

# Double-exponential PSC: separate rise and fall
# i_rise_t = τ_rise * i_rise_{t-1} + x_t
# i_fall_t = τ_fall * i_fall_{t-1} + x_t
# i_t = i_rise_t - i_fall_t   (normalized)
# v_t = β * v_{t-1} + i_t
```

When implementing, store `i_t` as a persistent state variable alongside `v_t` in the neuron's state tuple. The state tuple interface should be:

```python
# Basic LIF: (v,)
# LIF + alpha synapse: (v, i)
# LIF + double-exp: (v, i_rise, i_fall)
```

### 6.3 Refractory Period Support

A refractory period enforces a hard silence window after each spike during which the neuron cannot fire again, regardless of input. Define the state variable and masking logic:

```python
# In __init__:
self.t_ref = refractory_steps    # integer, in timesteps

# In forward, alongside v and s:
# refractory_counter: integer tensor, shape (batch, hidden)
# decremented each step, reset to t_ref on spike

refractory_mask = (refractory_counter <= 0).float()   # 1 where neuron can fire
spike = SurrogateSpike.apply(v - self.v_th) * refractory_mask

# Update counter:
refractory_counter = torch.clamp(refractory_counter - 1, min=0)
refractory_counter = refractory_counter + spike.detach() * self.t_ref
```

The `refractory_mask` is computed from `refractory_counter.detach()` so the hard masking does not introduce a discontinuous gradient through the counter. The surrogate gradient still flows through the unmasked spike computation.

### 6.4 Mixed Precision Dtype Contract

Formalize the dtype policy for each state variable so that AMP interactions are unambiguous:

| State Variable | Production Dtype | Notes |
|---|---|---|
| Membrane potential `v` | fp32 always | Never cast to bf16/fp16 during accumulation |
| Adaptation variable `a` | fp32 always | Same accumulation concern as `v` |
| Synaptic current `i` | fp32 always | Subject to same drift if cast |
| Spike output `s` | compute dtype | Binary; safe to cast after assignment |
| Weight matrices | compute dtype | Managed by AMP autocast |
| Refractory counter | int32 | Not a floating-point accumulator |

Implement this contract by casting at the boundary of the neuron's forward method — accept inputs in compute dtype, immediately upcast state variables to fp32 for the update equations, and return spikes cast back to compute dtype.

---

## 7. Common Anti-Patterns

Avoid the following patterns. Each one causes incorrect behavior that may not manifest immediately as a training crash, making them particularly dangerous.

**Using β = 1.0 (no leak).**
A neuron with β = 1.0 is a pure integrator. It has no forgetting mechanism and cannot selectively attend to recent vs. distant past inputs. More critically, the gradient of `v_t` with respect to `v_0` is exactly 1.0 for all t — the network cannot develop temporal discrimination through the membrane state. Use β ∈ (0.8, 0.999) as a starting range.

**Mixing reset rules across layers in the same network.**
If layer 1 uses subtractive reset and layer 2 uses hard reset, the spike statistics have different information content at each layer boundary. Subtractive-reset spikes carry overshoot information; hard-reset spikes do not. Downstream layers cannot be learned weights that generalize across these semantically different inputs. Choose one reset rule and apply it uniformly across all LIF layers in a given training run.

**In-place membrane updates.**
Covered in Rule 2, but listed here because it is the most common bug. The symptom is gradients that are zero or NaN for no apparent reason. Always write `v = β * v + i` rather than `v.mul_(β).add_(i)`. Audit any `+=`, `-=`, `*=`, or `/=` operation where the left-hand side is a tensor that appears in a gradient computation graph.

**Not clamping β after optimizer steps when β is a learnable scalar parameter.**
If β is parameterized directly (not via sigmoid), the optimizer can push it above 1.0. This happens silently — AdamW does not know that β has physical constraints. Add a post-step clamp hook or switch to sigmoid parameterization. Failure to do this is the most common cause of NaN loss appearing after many training steps when SNN layers are unfrozen.

**Treating spike output as continuous.**
Downstream layers (HTM, global workspace) are calibrated for binary spike inputs. Passing soft spike values (e.g., sigmoid outputs) breaks the spike-rate coding assumptions, causes incorrect spike-count normalization, and produces inconsistent behavior between training and inference if any binarization is applied at inference time. Always use the custom autograd Function pattern (Section 5, Rule 5) to produce exactly-binary spikes in forward.

**Initializing W_rec with large weights.**
Starting recurrent weights at standard normal scale (std=1.0) for a 512-neuron layer creates a recurrent gain of approximately sqrt(512) ≈ 22.6 — far above the stability threshold of 1.0. The network enters resonance on the first forward pass and produces all-spike or all-silent outputs. Use `std = recurrent_weight_scale / sqrt(hidden_size)` or set `recurrent_weight_scale` to a small absolute value like 0.1.

**Unrolling over too many timesteps without gradient checkpointing.**
BPTT through T timesteps stores T copies of the membrane state in memory. For T=100 and batch size 32 on a 512-neuron layer, this is manageable. For T=1000, the memory footprint becomes prohibitive. Use `torch.utils.checkpoint.checkpoint` on the per-timestep forward function when T > 50. This trades compute for memory by recomputing activations during the backward pass.
