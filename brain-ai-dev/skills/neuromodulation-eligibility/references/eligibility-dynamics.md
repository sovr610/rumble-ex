# Eligibility Trace Dynamics for Three-Factor Learning

This document specifies the mathematical foundations, implementation patterns, and numerical considerations for eligibility traces in a three-factor learning framework. Eligibility traces serve as the synaptic memory that bridges the temporal gap between local neural activity (pre/post coincidences) and delayed global modulatory signals (reward, error, neuromodulator release). Every weight update in the system follows the canonical three-factor structure: local correlation establishes eligibility, and a third factor gates whether that eligibility converts into a lasting weight change.

---

## 1. Eligibility Trace Fundamentals

### Definition

An eligibility trace `e_ij(t)` is a per-synapse variable that records a decaying memory of recent correlations between presynaptic neuron `i` and postsynaptic neuron `j`. The trace does not directly modify the weight. Instead, it marks the synapse as "eligible" for modification when a third modulatory factor arrives.

Formally, the trace for synapse (i, j) at time t is a scalar value that accumulates evidence of pre/post coincidences and decays exponentially toward zero in their absence.

### General Update Rule

At each timestep, apply:

```
e_ij(t+1) = (1 - dt / tau_e) * e_ij(t) + f(pre_i(t), post_j(t))
```

where:
- `tau_e` is the eligibility time constant (in the same units as dt)
- `dt` is the simulation timestep
- `f(pre, post)` is a correlation function (STDP kernel, Hebbian product, etc.)
- The decay factor `(1 - dt / tau_e)` is a linear approximation of `exp(-dt / tau_e)`

Use the exponential form `exp(-dt / tau_e)` when dt/tau_e is not small (i.e., when dt > 0.1 * tau_e). Use the linear approximation when dt/tau_e < 0.1 for computational efficiency with negligible error.

### Three-Factor Weight Update

The weight update applies the third factor M(t) (a modulatory signal) to gate the eligibility trace:

```
delta_w_ij = eta * M(t) * e_ij(t)
```

where:
- `eta` is the learning rate
- `M(t)` is the third factor: a neuromodulatory signal (dopamine for reward, acetylcholine for attention, norepinephrine for surprise, serotonin for temporal discounting), a reward prediction error, a top-down error gradient, or any global broadcast signal
- `e_ij(t)` is the current eligibility trace value

This factorization is the core principle. The synapse locally computes what activity pattern occurred (the eligibility), and a global signal determines whether that pattern should be reinforced or suppressed. Without M(t), no weight change occurs regardless of eligibility magnitude. Without eligibility, M(t) has no target to modulate.

### Temporal Credit Assignment

Eligibility traces solve the temporal credit assignment problem. Neural activity at time t produces a correlation that decays over a window of approximately 3 * tau_e timesteps. If a modulatory signal arrives within that window, the synapse receives a weight update proportional to its remaining eligibility. Synapses active long before the signal receive smaller updates (due to decay), while those active just before the signal receive larger updates. This implements a soft temporal window for credit assignment without requiring backpropagation through time.

---

## 2. Trace Types

Three trace types are supported, each with distinct accumulation semantics. Select the type based on the activity regime and learning requirements.

### 2.1 Accumulating Traces

The standard form. New correlation evidence is added to the decayed trace.

**Mathematical form:**

```
e(t+1) = decay * e(t) + f(pre, post)
```

where `decay = exp(-dt / tau_e)` or `(1 - dt / tau_e)`.

**Properties:**
- Traces can grow arbitrarily large if pre/post correlations are sustained. Under continuous firing at rate r, the steady-state trace magnitude is approximately `f_mag * tau_e / dt`, where f_mag is the typical magnitude of f(pre, post).
- Requires clamping to a bounded range (default [-5.0, 5.0]) to prevent numerical explosion.
- Sensitive to the choice of tau_e: too large accumulates excessively, too small loses credit assignment range.

**Use case:** General-purpose eligibility in rate-coded networks and spiking networks with moderate firing rates. This is the default trace type.

**Code pattern:**

```python
trace = decay * trace + correlation
trace = torch.clamp(trace, -5.0, 5.0)
```

### 2.2 Replacing Traces

The trace is replaced by the new correlation if it exceeds the decayed value. Prevents unbounded growth by capping at the maximum single-step correlation.

**Mathematical form:**

```
e(t+1) = max(decay * e(t), f(pre, post))
```

For signed traces where f can be negative, apply replacing logic per sign:

```
e_pos(t+1) = max(decay * e_pos(t), max(f(pre, post), 0))
e_neg(t+1) = min(decay * e_neg(t), min(f(pre, post), 0))
e(t+1) = e_pos(t+1) + e_neg(t+1)
```

**Properties:**
- Bounded by the maximum single-event correlation magnitude. Never exceeds the peak of f(pre, post).
- Robust against sustained high-frequency activity. The trace reflects the most recent strong event rather than the sum of all events.
- Loses information about event frequency; a synapse active once and a synapse active continuously produce similar traces.

**Use case:** Event-driven systems with sparse spike trains where each spike event should independently mark eligibility. Preferred when firing rates vary widely across neurons and accumulating traces would create large magnitude disparities.

**Code pattern:**

```python
decayed = decay * trace
trace = torch.where(correlation.abs() > decayed.abs(), correlation, decayed)
```

### 2.3 Dutch Traces

A hybrid that interpolates between accumulating and replacing behavior. Named after the Dutch trace variant in reinforcement learning (used in TD(lambda) algorithms).

**Mathematical form:**

```
e(t+1) = (1 - alpha) * decay * e(t) + f(pre, post)
```

where `alpha` is the replacement rate parameter in [0, 1]:
- `alpha = 0` recovers the accumulating trace
- `alpha = 1` recovers a pure replacing trace (trace is fully replaced on each event, with only the new correlation preserved)
- `alpha` in (0, 1) provides graded interpolation

**Properties:**
- Controlled growth: the (1 - alpha) factor shrinks the old trace before adding new evidence, preventing runaway accumulation without hard clamping.
- Steady-state magnitude is approximately `f_mag / (1 - (1 - alpha) * decay)`, which is bounded for any alpha > 0.
- The alpha parameter provides a single knob to tune the accumulation/replacement tradeoff.

**Use case:** Reinforcement learning eligibility traces in TD(lambda)-style algorithms. Set alpha = lambda (the RL trace parameter) for direct correspondence. Also useful when the network mixes event-driven and rate-based processing.

**Code pattern:**

```python
trace = (1.0 - alpha) * decay * trace + correlation
trace = torch.clamp(trace, -5.0, 5.0)
```

---

## 3. Decay Dynamics

### Exponential Decay

The fundamental decay factor per timestep is:

```
decay = exp(-dt / tau_e)
```

This produces exact exponential decay: after time T, the remaining fraction is `exp(-T / tau_e)`. The trace drops to 1/e (~37%) of its peak after exactly tau_e time units.

The linear approximation `decay = 1 - dt / tau_e` is a first-order Taylor expansion valid when `dt / tau_e << 1`. Use it when dt < 0.1 * tau_e for a maximum approximation error below 0.5%. When dt approaches tau_e, the linear form can produce negative decay factors; always clamp to [0, 1]:

```python
decay = max(0.0, 1.0 - dt / tau_e)
```

### Time Constant Ranges

Select tau_e based on the timescale of the credit assignment problem:

| Regime | tau_e Range | Use Case |
|--------|-------------|----------|
| Fast spike timing | 5 -- 10 ms | STDP learning windows, precise temporal coding |
| Medium synaptic | 20 -- 50 ms | Local circuit plasticity, sensory processing |
| Slow behavioral | 100 -- 500 ms | Reward-based learning, action selection |
| Extended cognitive | 500 -- 1000 ms | Working memory tasks, delayed rewards |
| Ultra-long | 1000 -- 5000 ms | Episodic credit assignment, multi-step planning |

When dt represents discrete simulation steps rather than real milliseconds, interpret tau_e in the same units. For example, if dt = 1 (one step), tau_e = 20 means the trace decays to 1/e after 20 steps.

### Decay Ordering

Apply decay BEFORE adding the new correlation at each timestep. This ensures that the current timestep's correlation is recorded at full strength, while all previous contributions are attenuated. The ordering matters: reversing it (add then decay) would immediately attenuate the current event, effectively reducing the useful correlation magnitude by the decay factor.

```python
# Correct ordering:
trace = decay * trace       # Step 1: decay old trace
trace = trace + correlation  # Step 2: add new evidence

# Equivalent single line:
trace = decay * trace + correlation
```

### Clamping

After each update, clamp traces to a bounded range to prevent numerical explosion:

```python
trace = torch.clamp(trace, min=-5.0, max=5.0)
```

The default range [-5.0, 5.0] is suitable for most configurations. Adjust the range if the correlation function f(pre, post) has an unusual scale. The clamp range should be at least 5x the typical single-step correlation magnitude to avoid clipping during normal operation.

---

## 4. STDP Kernels (Spike-Based)

When the network uses spiking neurons, the correlation function f is derived from spike-timing-dependent plasticity (STDP) rules.

### 4.1 Pair-Based STDP

The classical STDP rule computes f based on the timing difference between pre and post spikes:

```
delta_t = t_post - t_pre
```

**Potentiation window (delta_t > 0, pre fires before post):**

```
f = A_plus * exp(-delta_t / tau_plus)
```

**Depression window (delta_t < 0, post fires before pre):**

```
f = -A_minus * exp(delta_t / tau_minus)
```

where:
- `A_plus` is the potentiation amplitude (typical: 0.01 -- 0.1)
- `A_minus` is the depression amplitude (typical: 0.01 -- 0.1, often A_minus > A_plus for stability)
- `tau_plus` is the potentiation time constant (typical: 10 -- 20 ms)
- `tau_minus` is the depression time constant (typical: 10 -- 20 ms, can differ from tau_plus)

The asymmetric window captures causal timing: when a presynaptic spike reliably precedes a postsynaptic spike, the synapse is potentiated, encoding a predictive relationship.

### 4.2 Symmetric STDP

Some brain regions (e.g., hippocampal CA3) exhibit symmetric STDP where the magnitude of plasticity depends on |delta_t| regardless of sign:

```
|f| = A * exp(-|delta_t| / tau_stdp)
```

The sign of f is determined by a separate rule: a threshold on postsynaptic activity, a neuromodulatory signal, or the sign of a global error. Use symmetric STDP when the network must learn temporal proximity rather than temporal ordering.

### 4.3 Discrete-Time Implementation

In discrete-time simulation, exact spike times are not available. Instead, maintain exponentially decaying trace variables for pre and post spike trains and use them to compute the STDP update.

**Pre-synaptic trace (tracks recent pre spikes):**

```
x_pre(t+1) = decay_pre * x_pre(t) + spike_pre(t)
```

where `decay_pre = exp(-dt / tau_plus)` and `spike_pre(t)` is 1 if neuron i spiked at time t, 0 otherwise.

**Post-synaptic trace (tracks recent post spikes):**

```
x_post(t+1) = decay_post * x_post(t) + spike_post(t)
```

where `decay_post = exp(-dt / tau_minus)`.

**Eligibility update on spike events:**

On a post spike at time t (spike_post(t) = 1):

```
e_ij += A_plus * x_pre_i(t)   # LTP: pre trace indicates how recently pre fired
```

On a pre spike at time t (spike_pre(t) = 1):

```
e_ij -= A_minus * x_post_j(t)  # LTD: post trace indicates how recently post fired
```

This pair-based discrete approximation converges to the continuous STDP kernel as dt approaches zero.

**Vectorized implementation:**

```python
# x_pre: (B, N_pre), x_post: (B, N_post), spikes_pre/post: (B, N)
x_pre = decay_pre * x_pre + spikes_pre
x_post = decay_post * x_post + spikes_post

# Eligibility update: (B, N_post, N_pre)
# LTP contribution: where post fires, use pre trace
ltp = A_plus * spikes_post.unsqueeze(2) * x_pre.unsqueeze(1)
# LTD contribution: where pre fires, use post trace
ltd = A_minus * x_post.unsqueeze(2) * spikes_pre.unsqueeze(1)

eligibility = decay_e * eligibility + ltp - ltd
eligibility = torch.clamp(eligibility, -5.0, 5.0)
```

---

## 5. Rate-Based Correlation

When neurons produce continuous firing rates rather than discrete spikes, use rate-based correlation functions.

### 5.1 Simple Hebbian

The outer product of pre and post activity vectors:

```
f(pre, post) = post (x) pre^T
```

For a batch of B samples with N_post postsynaptic neurons and N_pre presynaptic neurons:

```python
# pre: (B, N_pre), post: (B, N_post)
correlation = torch.bmm(post.unsqueeze(2), pre.unsqueeze(1))  # (B, N_post, N_pre)
```

This produces a full correlation matrix per batch element. The eligibility trace `e` has shape `(B, N_post, N_pre)`, matching the weight matrix shape.

### 5.2 Diagonal Variant

When N_pre equals N_post and only self-connections (or one-to-one mappings) matter, use element-wise multiplication:

```python
# pre: (B, N), post: (B, N)
correlation = pre * post  # (B, N)
```

The eligibility trace `e` has shape `(B, N)`. This is computationally efficient and appropriate for layer-norm-style lateral modulation or recurrent self-connections.

### 5.3 Anti-Hebbian Correlation

For inhibitory connections or decorrelation learning:

```
f(pre, post) = -post (x) pre^T
```

Negate the correlation. Anti-Hebbian traces drive weight decreases when the third factor is positive, implementing competitive learning and decorrelation.

### 5.4 BCM-Like Correlation

The Bienenstock-Cooper-Munro (BCM) rule introduces a sliding threshold that stabilizes learning:

```
f(pre, post) = post * (post - theta) * pre
```

where theta is a sliding threshold that tracks the time-averaged postsynaptic activity:

```
theta(t+1) = (1 - dt / tau_theta) * theta(t) + (dt / tau_theta) * post(t)^2
```

BCM correlation is positive when post > theta (potentiation) and negative when post < theta (depression). This implements automatic gain control: highly active synapses have a higher threshold and are harder to potentiate further.

**Implementation:**

```python
# post: (B, N_post), pre: (B, N_pre), theta: (B, N_post)
theta = (1 - dt / tau_theta) * theta + (dt / tau_theta) * post ** 2
bcm_factor = post * (post - theta)  # (B, N_post)
correlation = torch.bmm(bcm_factor.unsqueeze(2), pre.unsqueeze(1))  # (B, N_post, N_pre)
```

---

## 6. Reset and Carry Semantics

### Reset Operation

`reset(batch_size, device)` clears all trace state to zero and reinitializes internal buffers:

```python
def reset(self, batch_size: int, device: torch.device):
    self.eligibility.zero_()
    # Resize if batch size changed
    if self.eligibility.shape[0] != batch_size:
        self.eligibility = torch.zeros(batch_size, *self.trace_shape, device=device)
    # Reset STDP trace variables if spike-based
    if self.x_pre is not None:
        self.x_pre = torch.zeros(batch_size, self.n_pre, device=device)
    if self.x_post is not None:
        self.x_post = torch.zeros(batch_size, self.n_post, device=device)
    # Reset BCM threshold if rate-based
    if self.theta is not None:
        self.theta = torch.zeros(batch_size, self.n_post, device=device)
```

Call reset at:
- Episode boundaries in reinforcement learning
- Task switches in meta-learning (between inner-loop adaptation episodes)
- Explicit context changes (new input sequence, new sensory stream)
- After NaN detection (emergency recovery)

### Carry Mode

In carry mode, traces persist across successive forward calls without reset. This is the default behavior for streaming or online processing where the agent processes a continuous input stream without discrete episode boundaries.

When carry mode is active:
- Do not call reset between forward passes.
- Traces accumulate a running memory of recent correlations.
- The decay constant tau_e determines how far back the effective memory extends (approximately 3 * tau_e timesteps).

Toggle carry mode with a boolean flag:

```python
if not carry:
    self.reset(batch_size, device)
```

### Batch Independence

Each batch element maintains independent traces. Element `e[b]` is never influenced by element `e[b']` for b != b'. This is automatically enforced by the element-wise and batched matrix operations. Never use operations that mix across the batch dimension (e.g., batch normalization on traces, cross-batch attention on eligibility state).

When batch size changes between calls (e.g., last batch in an epoch is smaller), reallocate trace tensors rather than slicing to avoid shape mismatches:

```python
if pre.shape[0] != self.eligibility.shape[0]:
    self.reset(pre.shape[0], pre.device)
```

---

## 7. Numerical Stability

### Precision Requirements

Eligibility traces MUST be computed in fp32 (float32). Half-precision (fp16) introduces catastrophic drift because:
- Traces accumulate small increments over many timesteps. fp16 has only ~3 decimal digits of precision, causing small updates to be rounded to zero.
- The decay operation `decay * trace` loses information when decay is close to 1.0 (slow decay). In fp16, `1.0 - 1e-4` rounds to 1.0, stopping all decay.
- Clamping interacts poorly with fp16 saturation: values near the clamp boundary oscillate.

Under automatic mixed precision (AMP), explicitly disable autocast for trace computations:

```python
with torch.cuda.amp.autocast(enabled=False):
    trace_fp32 = trace.float()
    correlation_fp32 = correlation.float()
    trace_fp32 = decay * trace_fp32 + correlation_fp32
    trace_fp32 = torch.clamp(trace_fp32, -5.0, 5.0)
    trace = trace_fp32  # Keep as fp32; do NOT cast back to fp16
```

Store trace tensors as fp32 buffers, not parameters, to avoid them being cast by AMP:

```python
self.register_buffer('eligibility', torch.zeros(*shape, dtype=torch.float32))
```

### Clamp Strategy

Apply clamping after every update step:

```python
trace = torch.clamp(trace, min=self.clamp_min, max=self.clamp_max)
```

Default range: [-5.0, 5.0]. This range accommodates typical correlation magnitudes (order 0.01--1.0) with headroom for transient accumulation. Adjust if:
- The correlation function has unusually large outputs (increase range).
- The third factor M(t) is very large (decrease range to keep delta_w reasonable).
- Using very long tau_e (increase range to allow slow accumulation).

### NaN Detection and Recovery

Check for NaN after each trace update. NaN can arise from:
- Division by zero in correlation functions
- Inf * 0 in spike-based updates when both spike counts and rates are used
- Corrupted input tensors

```python
if torch.isnan(trace).any():
    warnings.warn("NaN detected in eligibility trace, resetting to zero")
    trace = torch.zeros_like(trace)
    # Also reset auxiliary state
    if self.x_pre is not None:
        self.x_pre.zero_()
    if self.x_post is not None:
        self.x_post.zero_()
```

In production, log the NaN event with context (layer name, timestep, input statistics) for debugging. Do not silently ignore NaN; it indicates a configuration or numerical issue that should be addressed.

### Monitoring

Track trace statistics for diagnostics:
- `trace.abs().mean()`: average magnitude. Should be stable, not growing unboundedly.
- `trace.abs().max()`: peak magnitude. Should stay well below clamp bounds during normal operation. Frequent clamping indicates tau_e is too large or correlation magnitude is too high.
- `(trace == 0).float().mean()`: sparsity. Very high sparsity (>99%) suggests tau_e is too short or activity is too sparse for effective credit assignment.

Log these metrics every N steps during training. Set alerts if mean magnitude exceeds 50% of the clamp range.

---

## 8. Implementation Patterns

### 8.1 EligibilityTraceModule Class

```python
import torch
import torch.nn as nn
from enum import Enum
from typing import Optional, Tuple

class TraceType(Enum):
    ACCUMULATING = "accumulating"
    REPLACING = "replacing"
    DUTCH = "dutch"

class EligibilityTraceModule(nn.Module):
    """Manages eligibility traces for a single synaptic connection matrix.

    Supports accumulating, replacing, and Dutch trace types.
    Handles spike-based (STDP) and rate-based (Hebbian) correlation modes.
    All trace computations are performed in fp32 regardless of AMP settings.
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        tau_e: float = 100.0,
        dt: float = 1.0,
        trace_type: TraceType = TraceType.ACCUMULATING,
        alpha: float = 0.0,           # Dutch trace replacement rate
        clamp_min: float = -5.0,
        clamp_max: float = 5.0,
        use_exponential_decay: bool = True,
        # STDP parameters (only used in spike mode)
        A_plus: float = 0.01,
        A_minus: float = 0.012,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
    ):
        super().__init__()
        self.n_pre = n_pre
        self.n_post = n_post
        self.tau_e = tau_e
        self.dt = dt
        self.trace_type = trace_type
        self.alpha = alpha
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        # Compute decay factors
        if use_exponential_decay:
            self.decay_e = math.exp(-dt / tau_e)
            self.decay_pre = math.exp(-dt / tau_plus)
            self.decay_post = math.exp(-dt / tau_minus)
        else:
            self.decay_e = max(0.0, 1.0 - dt / tau_e)
            self.decay_pre = max(0.0, 1.0 - dt / tau_plus)
            self.decay_post = max(0.0, 1.0 - dt / tau_minus)

        self.A_plus = A_plus
        self.A_minus = A_minus

        # Buffers allocated on first call or reset
        self.eligibility: Optional[torch.Tensor] = None
        self.x_pre: Optional[torch.Tensor] = None
        self.x_post: Optional[torch.Tensor] = None

    def reset(self, batch_size: int, device: torch.device):
        """Clear all traces to zero. Call at episode/task boundaries."""
        self.eligibility = torch.zeros(
            batch_size, self.n_post, self.n_pre, device=device, dtype=torch.float32
        )
        self.x_pre = torch.zeros(
            batch_size, self.n_pre, device=device, dtype=torch.float32
        )
        self.x_post = torch.zeros(
            batch_size, self.n_post, device=device, dtype=torch.float32
        )

    def _ensure_state(self, batch_size: int, device: torch.device):
        """Allocate state on first call or batch size change."""
        if self.eligibility is None or self.eligibility.shape[0] != batch_size:
            self.reset(batch_size, device)

    def _apply_trace_update(
        self, trace: torch.Tensor, correlation: torch.Tensor
    ) -> torch.Tensor:
        """Apply trace type semantics: accumulating, replacing, or Dutch."""
        if self.trace_type == TraceType.ACCUMULATING:
            trace = self.decay_e * trace + correlation
        elif self.trace_type == TraceType.REPLACING:
            decayed = self.decay_e * trace
            # Replace where new correlation exceeds decayed magnitude
            use_new = correlation.abs() > decayed.abs()
            trace = torch.where(use_new, correlation, decayed)
        elif self.trace_type == TraceType.DUTCH:
            trace = (1.0 - self.alpha) * self.decay_e * trace + correlation
        return torch.clamp(trace, self.clamp_min, self.clamp_max)

    @torch.no_grad()
    def step_stdp(
        self,
        spikes_pre: torch.Tensor,   # (B, N_pre), binary 0/1
        spikes_post: torch.Tensor,  # (B, N_post), binary 0/1
    ) -> torch.Tensor:
        """Update eligibility using spike-based STDP.

        Returns the current eligibility trace (B, N_post, N_pre).
        """
        with torch.cuda.amp.autocast(enabled=False):
            spikes_pre = spikes_pre.float()
            spikes_post = spikes_post.float()
            B = spikes_pre.shape[0]
            self._ensure_state(B, spikes_pre.device)

            # Decay pre/post spike traces
            self.x_pre = self.decay_pre * self.x_pre + spikes_pre
            self.x_post = self.decay_post * self.x_post + spikes_post

            # Compute STDP correlation
            # LTP: post spike uses pre trace
            ltp = self.A_plus * torch.bmm(
                spikes_post.unsqueeze(2), self.x_pre.unsqueeze(1)
            )  # (B, N_post, N_pre) -- note: unsqueeze(2) on (B,N_post) -> (B,N_post,1)
            # Correction: spikes_post is (B, N_post)
            ltp = self.A_plus * spikes_post.unsqueeze(2) * self.x_pre.unsqueeze(1)

            # LTD: pre spike uses post trace
            ltd = self.A_minus * self.x_post.unsqueeze(2) * spikes_pre.unsqueeze(1)

            correlation = ltp - ltd  # (B, N_post, N_pre)

            # Apply trace update semantics
            self.eligibility = self._apply_trace_update(self.eligibility, correlation)

        return self.eligibility

    @torch.no_grad()
    def step_hebbian(
        self,
        pre: torch.Tensor,   # (B, N_pre), continuous rates
        post: torch.Tensor,  # (B, N_post), continuous rates
    ) -> torch.Tensor:
        """Update eligibility using rate-based Hebbian correlation.

        Returns the current eligibility trace (B, N_post, N_pre).
        """
        with torch.cuda.amp.autocast(enabled=False):
            pre = pre.float()
            post = post.float()
            B = pre.shape[0]
            self._ensure_state(B, pre.device)

            # Outer product correlation
            correlation = post.unsqueeze(2) * pre.unsqueeze(1)  # (B, N_post, N_pre)

            # Apply trace update semantics
            self.eligibility = self._apply_trace_update(self.eligibility, correlation)

        return self.eligibility

    def get_trace(self) -> Optional[torch.Tensor]:
        """Return current eligibility trace without updating."""
        return self.eligibility

    def check_health(self) -> dict:
        """Return diagnostic statistics for monitoring."""
        if self.eligibility is None:
            return {"initialized": False}
        e = self.eligibility
        return {
            "initialized": True,
            "mean_abs": e.abs().mean().item(),
            "max_abs": e.abs().max().item(),
            "sparsity": (e == 0).float().mean().item(),
            "has_nan": torch.isnan(e).any().item(),
            "has_inf": torch.isinf(e).any().item(),
            "at_clamp_min": (e == self.clamp_min).float().mean().item(),
            "at_clamp_max": (e == self.clamp_max).float().mean().item(),
        }
```

### 8.2 Spike-Based STDP Update Step (Standalone)

For integration into existing spiking network modules without the full class:

```python
@torch.no_grad()
def stdp_eligibility_step(
    eligibility: torch.Tensor,   # (B, N_post, N_pre)
    x_pre: torch.Tensor,         # (B, N_pre)
    x_post: torch.Tensor,        # (B, N_post)
    spikes_pre: torch.Tensor,    # (B, N_pre)
    spikes_post: torch.Tensor,   # (B, N_post)
    decay_e: float,
    decay_pre: float,
    decay_post: float,
    A_plus: float,
    A_minus: float,
    clamp_range: Tuple[float, float] = (-5.0, 5.0),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single STDP eligibility update step.

    Returns (updated_eligibility, updated_x_pre, updated_x_post).
    """
    with torch.cuda.amp.autocast(enabled=False):
        spikes_pre = spikes_pre.float()
        spikes_post = spikes_post.float()

        # Update spike traces
        x_pre = decay_pre * x_pre + spikes_pre
        x_post = decay_post * x_post + spikes_post

        # Compute STDP correlation
        ltp = A_plus * spikes_post.unsqueeze(2) * x_pre.unsqueeze(1)
        ltd = A_minus * x_post.unsqueeze(2) * spikes_pre.unsqueeze(1)

        # Update eligibility with decay
        eligibility = decay_e * eligibility + (ltp - ltd)
        eligibility = torch.clamp(eligibility, clamp_range[0], clamp_range[1])

    return eligibility, x_pre, x_post
```

### 8.3 Rate-Based Hebbian Update Step (Standalone)

```python
@torch.no_grad()
def hebbian_eligibility_step(
    eligibility: torch.Tensor,   # (B, N_post, N_pre)
    pre: torch.Tensor,           # (B, N_pre)
    post: torch.Tensor,          # (B, N_post)
    decay: float,
    clamp_range: Tuple[float, float] = (-5.0, 5.0),
) -> torch.Tensor:
    """Single Hebbian eligibility update step.

    Returns updated eligibility trace.
    """
    with torch.cuda.amp.autocast(enabled=False):
        pre = pre.float()
        post = post.float()
        correlation = post.unsqueeze(2) * pre.unsqueeze(1)
        eligibility = decay * eligibility + correlation
        eligibility = torch.clamp(eligibility, clamp_range[0], clamp_range[1])
    return eligibility
```

### 8.4 Batched Trace Computation (Efficient Vectorized)

For large-scale networks, compute eligibility updates across all layers simultaneously:

```python
def batched_eligibility_update(
    traces: list[torch.Tensor],        # List of (B, N_post_l, N_pre_l) per layer
    pre_acts: list[torch.Tensor],      # List of (B, N_pre_l)
    post_acts: list[torch.Tensor],     # List of (B, N_post_l)
    decays: list[float],               # Per-layer decay factors
    clamp_range: Tuple[float, float] = (-5.0, 5.0),
) -> list[torch.Tensor]:
    """Update eligibility traces for all layers.

    Each layer is updated independently. This function exists to organize
    the computation and ensure consistent numerical handling.
    """
    updated = []
    with torch.cuda.amp.autocast(enabled=False):
        for trace, pre, post, decay in zip(traces, pre_acts, post_acts, decays):
            pre = pre.float()
            post = post.float()
            correlation = post.unsqueeze(2) * pre.unsqueeze(1)
            trace = decay * trace.float() + correlation
            trace = torch.clamp(trace, clamp_range[0], clamp_range[1])
            updated.append(trace)
    return updated
```

For GPU efficiency, if all layers have the same dimensions, pad and stack into a single tensor to exploit parallelism:

```python
# If all layers have same (N_post, N_pre), stack for parallel computation:
# stacked_traces: (L, B, N_post, N_pre)
# stacked_corr: (L, B, N_post, N_pre)
stacked_traces = decay_tensor * stacked_traces + stacked_corr  # Single fused op
stacked_traces = torch.clamp(stacked_traces, -5.0, 5.0)
```

### 8.5 Reset and Carry State Management

Pattern for managing trace state across training iterations:

```python
class ThreeFactorLearner:
    """Coordinates eligibility traces with three-factor weight updates."""

    def __init__(self, layers, tau_e, dt, trace_type):
        self.trace_modules = [
            EligibilityTraceModule(
                n_pre=layer.in_features,
                n_post=layer.out_features,
                tau_e=tau_e,
                dt=dt,
                trace_type=trace_type,
            )
            for layer in layers
        ]

    def begin_episode(self, batch_size: int, device: torch.device):
        """Reset all traces at episode start."""
        for tm in self.trace_modules:
            tm.reset(batch_size, device)

    def step(self, pre_acts, post_acts, mode="hebbian"):
        """Update traces for one timestep across all layers."""
        traces = []
        for tm, pre, post in zip(self.trace_modules, pre_acts, post_acts):
            if mode == "hebbian":
                t = tm.step_hebbian(pre, post)
            elif mode == "stdp":
                t = tm.step_stdp(pre, post)  # pre/post are spike tensors
            traces.append(t)
        return traces

    def apply_third_factor(self, modulation: torch.Tensor, layers, lr: float):
        """Apply weight updates: delta_w = lr * M * e."""
        with torch.no_grad():
            for tm, layer in zip(self.trace_modules, layers):
                e = tm.get_trace()
                if e is None:
                    continue
                # M is broadcast: (B,) or (B, 1) or (B, 1, 1)
                M = modulation
                while M.dim() < e.dim():
                    M = M.unsqueeze(-1)
                # Average over batch for weight update
                delta_w = lr * (M * e).mean(dim=0)  # (N_post, N_pre)
                layer.weight.data += delta_w

    def health_check(self) -> list[dict]:
        """Return diagnostics for all trace modules."""
        return [tm.check_health() for tm in self.trace_modules]
```

---

## Summary of Key Equations

| Concept | Formula |
|---------|---------|
| Accumulating trace | `e(t+1) = decay * e(t) + f(pre, post)` |
| Replacing trace | `e(t+1) = max(decay * e(t), f(pre, post))` |
| Dutch trace | `e(t+1) = (1 - alpha) * decay * e(t) + f(pre, post)` |
| Exponential decay | `decay = exp(-dt / tau_e)` |
| Linear decay approx | `decay = 1 - dt / tau_e` |
| Three-factor update | `delta_w = eta * M(t) * e(t)` |
| STDP LTP | `f = A_plus * exp(-delta_t / tau_plus)` for delta_t > 0 |
| STDP LTD | `f = -A_minus * exp(delta_t / tau_minus)` for delta_t < 0 |
| Hebbian correlation | `f = post (x) pre^T` (outer product) |
| BCM correlation | `f = post * (post - theta) * pre` |

---

## Parameter Quick Reference

| Parameter | Typical Range | Default | Notes |
|-----------|---------------|---------|-------|
| tau_e | 5 -- 5000 ms | 100.0 | Eligibility decay time constant |
| dt | 0.1 -- 10 ms | 1.0 | Simulation timestep |
| A_plus | 0.001 -- 0.1 | 0.01 | STDP potentiation amplitude |
| A_minus | 0.001 -- 0.1 | 0.012 | STDP depression amplitude |
| tau_plus | 5 -- 40 ms | 20.0 | STDP potentiation time constant |
| tau_minus | 5 -- 40 ms | 20.0 | STDP depression time constant |
| alpha | 0.0 -- 1.0 | 0.0 | Dutch trace replacement rate |
| clamp_min | -10.0 -- -1.0 | -5.0 | Lower clamp bound |
| clamp_max | 1.0 -- 10.0 | 5.0 | Upper clamp bound |
| tau_theta | 100 -- 10000 ms | 1000.0 | BCM threshold time constant |
