# Heterogeneous and Learnable Membrane Time Constants — Reference

This document specifies the theory, parametrization, initialization strategies, stability constraints, and integration patterns for heterogeneous membrane time constants in spiking neural networks. It serves as the canonical reference for implementing the `HeterogeneousTau` module and migrating the existing `AdvancedLIFNeuron` logit-beta approach to a principled τ-based parametrization.

---

## 1. Theory: Why Heterogeneous Time Constants Matter

### 1.1 Biological Reality

Cortical neurons exhibit a wide distribution of membrane time constants. Measured τ_m values range from 1 ms (fast-spiking parvalbumin interneurons) to 100 ms (layer 5 pyramidal neurons engaged in sustained working memory activity). This diversity is not noise — it is a functional resource. Different τ values enable different cells to act as specialized temporal filters:

- Short τ (1–5 ms): respond only to fast transients; effectively low-pass at the population level
- Medium τ (10–30 ms): integrate over hundreds of milliseconds; suited to rhythm entrainment
- Long τ (50–100 ms): hold membrane state across gaps in input; provide short-term integration

A network of neurons sharing a single τ operates as a single-bandwidth temporal filter. A network with diverse τ operates as a bank of filters covering multiple timescales simultaneously.

### 1.2 Computational Benefit: Multi-Timescale Temporal Integration

The discrete-time membrane update equation is:

```
v_t = beta * v_{t-1} + i_t
```

where `beta = exp(-dt / tau)`. With a shared β across all neurons in a layer, every neuron has the same effective memory horizon. With per-neuron β, each neuron independently tunes its memory horizon. This is equivalent to having a filterbank in the temporal domain:

- A neuron with τ = 2 ms (β ≈ 0.607 at dt=1) forgets its input in ~2 timesteps
- A neuron with τ = 20 ms (β ≈ 0.951 at dt=1) retains input for ~20 timesteps
- A neuron with τ = 100 ms (β ≈ 0.990 at dt=1) retains input for ~100 timesteps

Operating all three simultaneously in a single layer allows downstream layers to read off temporal features at multiple resolutions from the same spike representation, without requiring separate layers for each timescale.

### 1.3 Perez-Nieves et al.: Fixed Heterogeneous τ Improves Robustness

Perez-Nieves et al. (2021, "Neural heterogeneity promotes robust learning") demonstrated that networks with heterogeneous but fixed (non-learnable) time constants are significantly more robust to perturbations than homogeneous networks. Key findings:

- Heterogeneous networks maintain accuracy under weight noise, input noise, and partial neuron dropout
- The benefit scales with the degree of heterogeneity — wider τ distributions are more robust up to a saturation point
- The result holds across MNIST, CIFAR-10, and SHD (Spiking Heidelberg Digits)
- Homogeneous initialization that later diverges via gradient descent does not reach the same robustness as initialization-level diversity

The interpretation is that heterogeneity acts as a form of built-in ensemble diversity: different neurons are sensitive to different temporal features, so no single point of failure can silence the full representation.

### 1.4 Fang et al.: Learnable τ (Parametric LIF) Further Improves Performance

Fang et al. (2021, "Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks") extended the fixed-heterogeneous result by showing that learnable τ — initialized heterogeneously and optimized via gradient descent — further improves accuracy beyond fixed heterogeneity. Key findings:

- Test accuracy improvements of 1–3% on CIFAR-10 and DVS-CIFAR10 (dynamic vision sensor data)
- Largest gains appear on temporally structured inputs where the model must discover the right integration window
- The τ values learned by gradient descent cluster around task-relevant timescales
- Training converges faster: fewer epochs needed to reach the same loss level

The mechanism is that the network discovers, per neuron, the integration window that maximizes the temporal signal-to-noise ratio for the task. This is a form of learned temporal attention.

---

## 2. Two Modes: Fixed vs Learnable

### Mode 1: Heterogeneous Fixed

Initialize diverse τ per neuron at the start of training, then freeze τ during optimization. Only the synaptic weight matrices W_in and W_rec learn; τ provides structural diversity as a static inductive bias.

Characteristics:
- No additional gradient flow through τ; no extra backward computation
- Robustness benefits from Perez-Nieves et al. are fully achieved
- τ values do not adapt to the task — if initialization is poor for the specific input distribution, it stays poor
- Ablation baseline: compare against homogeneous fixed τ to isolate the diversity benefit

When to use fixed mode:
- Compute-constrained settings where gradient overhead matters
- When the task timescales are well-understood and initialization can be done informedly
- Ablation studies isolating the effect of diversity from the effect of learnability

### Mode 2: Heterogeneous Learnable

Initialize diverse τ per neuron, then allow gradient flow through τ during optimization. τ adapts to the task jointly with the weights.

Characteristics:
- Gradient flows through `beta = exp(-dt / tau)`, which is differentiable in τ
- τ converges toward task-relevant timescales
- Slightly more expensive: one additional parameter per neuron, gradient through the exp operation
- More expressive than fixed mode; adds the Fang et al. accuracy gain on top of the Perez-Nieves robustness gain
- Requires careful initialization to avoid τ collapsing toward a single value (gradient homogenizes unless initialized with sufficient diversity)

When to use learnable mode:
- Default for all production training runs where the training budget is not severely constrained
- Any input modality with unknown or mixed timescales (vision + audio, sensor fusion)
- Tasks where temporal precision matters (spike timing tasks, SHD, DVS datasets)

---

## 3. Parametrization: τ → β Mapping

### 3.1 Recommended: Softplus-Based τ

The preferred parametrization defines τ as the primary variable and derives β from it:

```
tau_raw:  unconstrained parameter (initialized around log(tau_0 - tau_min))
tau = tau_min + softplus(tau_raw)      # ensures tau > tau_min always
tau = tau.clamp(max=tau_max)           # hard upper cap for numerical safety
beta = exp(-dt / tau)                  # physics-grounded decay factor
```

Constants:
- `tau_min = 1.0`: prevents τ → 0 (instant decay, dead neuron). Softplus is always positive, so tau > tau_min is guaranteed by construction.
- `tau_max = 100.0`: prevents τ → ∞ (no decay, membrane explosion). The hard clamp is the final safety net.
- `dt = 1.0`: simulation timestep. One simulation step per biological millisecond is the standard convention unless the dataset specifies otherwise.

Advantages over direct β parametrization:
- τ is interpretable: measured in the same units as the simulation timestep (effectively milliseconds)
- The mapping tau → beta is monotone and smooth; gradients are well-behaved
- Constraints are enforced by construction (softplus) and hard clamp, not by optimizer tricks
- Log-scale histograms of τ are directly informative for biology comparisons

Gradient of β with respect to τ at the softplus-constrained τ:

```
d(beta)/d(tau) = d(exp(-dt/tau))/d(tau) = (dt / tau^2) * exp(-dt / tau)
```

This is always positive (larger τ → larger β, as expected), finite for τ > 0, and smooth. The gradient is largest at small τ and diminishes at large τ. When τ is large (long memory neurons), β gradients are small — this is acceptable because large-τ neurons already have stable β near 1.0.

Initialize `tau_raw` so that softplus(tau_raw) + tau_min = tau_0:

```python
import torch
import torch.nn.functional as F

def tau_to_tau_raw(tau_0: float, tau_min: float = 1.0) -> float:
    # Invert: tau_0 = tau_min + softplus(tau_raw)
    # softplus(x) = log(1 + exp(x))
    # tau_0 - tau_min = log(1 + exp(tau_raw))
    # exp(tau_raw) = exp(tau_0 - tau_min) - 1
    # tau_raw = log(exp(tau_0 - tau_min) - 1)
    val = tau_0 - tau_min
    return torch.tensor(val).expm1().log().item()
```

### 3.2 Current Implementation: Logit-Based β (Retained for Backward Compatibility)

The existing `AdvancedLIFNeuron` uses logit-space β parameterization:

```python
# In __init__:
logit_beta = torch.log(torch.tensor(beta_init) / (1 - beta_init))
self.log_beta = nn.Parameter(
    torch.full((size,), logit_beta.item()) + torch.randn(size) * 0.1
)

# In forward (via property):
@property
def beta(self) -> torch.Tensor:
    return torch.sigmoid(self.log_beta)
```

This parametrization is simpler and enforces β ∈ (0, 1) via sigmoid, but has two drawbacks:
- β has no direct physical meaning (no connection to τ in milliseconds)
- Gradient flow is through sigmoid, not through the exp(-dt/τ) physics

The logit-β approach is retained for backward compatibility in `AdvancedLIFNeuron`. New neuron classes use the softplus-τ approach.

### 3.3 Granularity Options

The granularity of τ determines how many independent time constants exist in a layer:

| Granularity | τ shape | Use case |
|---|---|---|
| `per_neuron` | `(N,)` | Default for dense SNN layers; maximum expressivity |
| `per_channel` | `(C, 1, 1)` | ConvSNN layers; each channel has its own τ, shared spatially |
| `per_layer` | scalar `()` | Baseline; uniform τ across all neurons in the layer |

Config field: `tau_granularity` in `SNNConfig` (string, one of `'per_neuron'`, `'per_channel'`, `'per_layer'`).

Broadcast rules for per-neuron in a dense layer (batch B, neurons N):

```python
beta = exp(-dt / tau)           # shape (N,)
v_new = beta.unsqueeze(0) * v + i   # beta broadcasts to (B, N)
```

Broadcast rules for per-channel in a convolutional layer (batch B, channels C, height H, width W):

```python
beta = exp(-dt / tau)           # shape (C, 1, 1)
v_new = beta.unsqueeze(0) * v + i   # beta broadcasts to (B, C, H, W)
```

---

## 4. Initialization Strategies (TauInitConfig)

Define a `TauInitConfig` dataclass with fields:

```python
@dataclass
class TauInitConfig:
    strategy: str = 'heterogeneous_gamma'   # see below
    tau_0: float = 20.0                      # target mean or homogeneous value
    tau_min: float = 1.0                     # hard lower bound
    tau_max: float = 100.0                   # hard upper bound
    dt: float = 1.0                          # simulation timestep
    # Gamma-specific
    gamma_k: float = 4.0                     # shape parameter
    gamma_theta: float = 5.0                # scale parameter (mean = k*theta = 20)
    # Preset bank specific
    preset_values: list = None               # e.g. [2.0, 5.0, 10.0, 20.0, 50.0]
    preset_mode: str = 'cyclic'              # 'cyclic' or 'random'
```

### Strategy 1: homogeneous

All neurons receive the same τ = tau_0. This is the baseline for ablation experiments — it tests what the model can do with uniform time constants alone.

```python
def init_homogeneous(n: int, cfg: TauInitConfig) -> torch.Tensor:
    tau = torch.full((n,), cfg.tau_0).clamp(cfg.tau_min, cfg.tau_max)
    return tau
```

When to use: ablation studies, debugging (easier to reason about uniform dynamics), sanity checks where τ diversity should not contribute.

Do not use homogeneous initialization when `use_heterogeneous_tau=True` — this defeats the purpose. It is retained only for controlled ablation comparison.

### Strategy 2: heterogeneous_gamma

Sample τ from a Gamma distribution, then clamp to [tau_min, tau_max]:

```python
def init_heterogeneous_gamma(n: int, cfg: TauInitConfig) -> torch.Tensor:
    # Gamma(k, theta) has mean = k * theta
    # Choose k and theta so mean ~ tau_0
    # Default: k=2, theta=tau_0/k
    theta = cfg.tau_0 / cfg.gamma_k
    dist = torch.distributions.Gamma(cfg.gamma_k, 1.0 / theta)
    tau = dist.sample((n,)).clamp(cfg.tau_min, cfg.tau_max)
    return tau
```

Properties of the Gamma distribution:
- Always produces positive values — no need to reject negative samples
- Right-skewed: most neurons have τ near the mode (= (k-1)*θ for k>1), with a long tail of high-τ neurons
- Matches biological observations: most neurons have moderate τ with a sparse population of long-τ cells
- k=2 gives a mild skew; k=1 gives an exponential distribution (more extreme skew); k=5 approaches Gaussian

This is the default strategy, following Perez-Nieves et al. who used Gamma-distributed τ in their heterogeneous networks.

### Strategy 3: heterogeneous_loguniform

Sample log(τ) uniformly, then exponentiate:

```python
def init_heterogeneous_loguniform(n: int, cfg: TauInitConfig) -> torch.Tensor:
    log_min = torch.tensor(cfg.tau_min).log()
    log_max = torch.tensor(cfg.tau_max).log()
    log_tau = torch.empty(n).uniform_(log_min.item(), log_max.item())
    tau = log_tau.exp().clamp(cfg.tau_min, cfg.tau_max)
    return tau
```

Properties:
- Uniform in log-space means equal representation per decade of τ: as many neurons in [1, 10] as in [10, 100]
- More balanced coverage of the full timescale range than Gamma (Gamma under-represents very long τ)
- The clamp is technically redundant (exp of values in [log_min, log_max] is already in [tau_min, tau_max]), but retained as a safety net against floating point edge cases
- More appropriate when τ is expected to cover multiple orders of magnitude and no prior exists on the distribution shape

Use loguniform when the task timescales are completely unknown and maximum diversity is desired. Use Gamma when prior knowledge suggests most neurons should have moderate τ with few long-τ outliers.

### Strategy 4: preset_bank

Define 3–5 discrete τ values and assign them to neurons either cyclically or randomly:

```python
def init_preset_bank(n: int, cfg: TauInitConfig) -> torch.Tensor:
    presets = cfg.preset_values or [2.0, 5.0, 10.0, 20.0, 50.0]
    preset_tensor = torch.tensor(presets).clamp(cfg.tau_min, cfg.tau_max)
    if cfg.preset_mode == 'cyclic':
        indices = torch.arange(n) % len(presets)
    else:  # 'random'
        indices = torch.randint(len(presets), (n,))
    return preset_tensor[indices]
```

Properties:
- Reduces effective τ diversity to a small number of discrete values
- Useful when interpretability of time constants is important (can name each group)
- Structured assignment (cyclic) ensures exact proportionality; random assignment is statistically proportional
- Each group behaves as a homogeneous sub-population — inter-group interaction is where multi-timescale computation occurs

Use preset bank when the task has a known small number of relevant timescales (e.g., gamma and alpha rhythms in neural signal processing), or when debugging the contribution of each timescale by selectively ablating groups.

---

## 5. Stability Constraints

These constraints are hard requirements. Violating any one of them produces silent training failures.

### Constraint 1: β ∈ (0, 1) by Construction

With the softplus-τ parametrization, this is guaranteed:
- softplus ensures τ > tau_min > 0
- τ > 0 implies exp(-dt/τ) ∈ (0, 1) for all positive dt
- The hard clamp tau.clamp(max=tau_max) further ensures β never reaches 1.0

Final safety net: apply `beta.clamp(BETA_MIN, BETA_MAX)` with `BETA_MIN=0.0` and `BETA_MAX=0.999` after computing β from τ. This catches any floating-point edge case.

### Constraint 2: τ > 0 Strictly

Softplus(x) = log(1 + exp(x)) > 0 for all real x. Adding tau_min > 0 to a positive value guarantees τ > tau_min > 0 without any clamping. This is the primary guarantee.

Do not skip the tau_min offset. Without it, tau_raw → -∞ drives τ → 0, which means β → 0 (instant decay, membrane never accumulates, neuron is permanently dead to input history).

### Constraint 3: Guard Against τ → tau_max Pressure

When learnable, τ may drift toward tau_max if the task requires long memory. When 5% or more of neurons in a layer are at the tau_max clamp, the model is constrained by the hyperparameter rather than the data. In this case, increase tau_max.

Diagnostic (run per epoch if learnable):

```python
def check_tau_clamp_pressure(tau: torch.Tensor, cfg: TauInitConfig, layer_name: str):
    frac_at_min = (tau <= cfg.tau_min * 1.01).float().mean().item()
    frac_at_max = (tau >= cfg.tau_max * 0.99).float().mean().item()
    if frac_at_min > 0.05:
        print(f"WARNING [{layer_name}]: {frac_at_min:.1%} neurons at tau_min — neurons dying")
    if frac_at_max > 0.05:
        print(f"WARNING [{layer_name}]: {frac_at_max:.1%} neurons at tau_max — increase tau_max")
```

### Constraint 4: fp32 for β Computation Under AMP

Do not compute `exp(-dt / tau)` in fp16 or bf16. At large τ, `-dt / tau` is a small negative number close to zero, and the exponential approaches 1.0 from below. In fp16, the precision around 1.0 is limited — values in (0.999, 1.0) round to 1.0, producing β=1.0 and triggering membrane explosion.

Enforce fp32 for the τ → β conversion:

```python
tau_fp32 = tau.float()
beta = torch.exp(-dt / tau_fp32)          # compute in fp32
beta = beta.clamp(BETA_MIN, BETA_MAX)     # safety clamp in fp32
# Use beta in fp32 throughout the membrane update
v_new = beta.unsqueeze(0) * v.float() + i.float()
```

Cast spike output `spk` back to the compute dtype before passing downstream.

### Constraint 5: Gradient Well-Posedness

The gradient of exp(-dt/τ) with respect to τ:

```
d(beta)/d(tau) = (dt / tau^2) * exp(-dt / tau)
```

This is:
- Positive for all τ > 0 and dt > 0 (monotone increasing relationship)
- Finite for all τ > tau_min > 0
- Bounded above by dt / tau_min^2 * 1 = dt (at τ → tau_min, exp → 0, but 1/tau^2 dominates slowly)
- Goes to 0 as τ → ∞ (β gradient vanishes for very long-memory neurons — acceptable)

No gradient explodes. No gradient vanishes for the range of τ used in practice (1–100 ms). The gradient through the softplus clamp is:

```
d(tau)/d(tau_raw) = sigmoid(tau_raw)      # always in (0, 1)
```

Combined gradient d(beta)/d(tau_raw) = d(beta)/d(tau) * d(tau)/d(tau_raw) is bounded, smooth, and always positive.

---

## 6. Integration with SpikingNeuronBase

### 6.1 HeterogeneousTau Module Interface

Define `HeterogeneousTau` as a standalone `nn.Module` that encapsulates all τ/β logic:

```python
class HeterogeneousTau(nn.Module):
    """
    Encapsulates per-neuron or per-channel membrane time constants.

    Manages the tau_raw parameter (or buffer for fixed mode), computes beta
    = exp(-dt/tau) with stability guarantees, and handles the correct
    broadcast shape for dense or convolutional layers.

    Args:
        n:            Number of independent τ values (neurons or channels)
        learnable:    If True, tau_raw is nn.Parameter; else register_buffer
        init_cfg:     TauInitConfig specifying initialization strategy
        dt:           Simulation timestep (default 1.0)
        tau_min:      Hard lower bound on τ (default 1.0)
        tau_max:      Hard upper bound on τ (default 100.0)
        beta_min:     Clamp floor on β (default 0.0)
        beta_max:     Clamp ceil on β (default 0.999)
    """

    def __init__(
        self,
        n: int,
        learnable: bool = True,
        init_cfg: TauInitConfig = None,
        dt: float = 1.0,
        tau_min: float = 1.0,
        tau_max: float = 100.0,
        beta_min: float = 0.0,
        beta_max: float = 0.999,
    ):
        super().__init__()
        cfg = init_cfg or TauInitConfig()
        self.dt = dt
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.beta_min = beta_min
        self.beta_max = beta_max

        tau_init = _init_tau(n, cfg)           # returns (n,) tensor in [tau_min, tau_max]
        tau_raw_init = tau_to_tau_raw(tau_init, tau_min)

        if learnable:
            self.tau_raw = nn.Parameter(tau_raw_init)
        else:
            self.register_buffer('tau_raw', tau_raw_init)

    @property
    def tau(self) -> torch.Tensor:
        """Constrained τ in [tau_min, tau_max]."""
        return (self.tau_min + F.softplus(self.tau_raw)).clamp(max=self.tau_max)

    @property
    def beta(self) -> torch.Tensor:
        """Constrained β in [beta_min, beta_max], computed in fp32."""
        tau_fp32 = self.tau.float()
        beta_fp32 = torch.exp(-self.dt / tau_fp32)
        return beta_fp32.clamp(self.beta_min, self.beta_max)
```

### 6.2 Two Integration Points

Every neuron class that uses `v_new = beta * v + i` gains heterogeneous time constants by adding a `HeterogeneousTau` instance:

**Point 1 — `__init__`: create the module**

```python
def __init__(self, size: int, tau_cfg: TauInitConfig = None, learnable_tau: bool = True, ...):
    super().__init__()
    self.tau_module = HeterogeneousTau(
        n=size,
        learnable=learnable_tau,
        init_cfg=tau_cfg,
    )
    # ... rest of __init__
```

**Point 2 — `forward`: retrieve β and broadcast**

Dense layer (batch B, neurons N):

```python
beta = self.tau_module.beta          # (N,), fp32
v_new = beta.unsqueeze(0) * v.float() + i.float()
```

Convolutional layer (batch B, channels C, H, W):

```python
beta = self.tau_module.beta          # (C,), fp32
v_new = beta.view(1, -1, 1, 1) * v.float() + i.float()
```

---

## 7. Diagnostics and Logging

Log the following per-layer per-epoch when `use_heterogeneous_tau=True`.

### 7.1 τ Histogram (Primary Diagnostic)

Log τ on a log scale. The histogram should:
- Span the full [tau_min, tau_max] range at initialization with the chosen strategy
- Show task-dependent clustering after training if learnable mode is active
- Show no significant drift if fixed mode is active (verify parameter was frozen)

```python
def log_tau_histogram(tau: torch.Tensor, layer_name: str, writer, step: int):
    writer.add_histogram(f'{layer_name}/tau', tau.detach().cpu(), step)
    writer.add_histogram(f'{layer_name}/log_tau', tau.log().detach().cpu(), step)
    writer.add_scalar(f'{layer_name}/tau_mean', tau.mean().item(), step)
    writer.add_scalar(f'{layer_name}/tau_min_actual', tau.min().item(), step)
    writer.add_scalar(f'{layer_name}/tau_max_actual', tau.max().item(), step)
    writer.add_scalar(f'{layer_name}/tau_std', tau.std().item(), step)
```

### 7.2 β Statistics

```python
def log_beta_stats(beta: torch.Tensor, layer_name: str, writer, step: int):
    writer.add_scalar(f'{layer_name}/beta_mean', beta.mean().item(), step)
    writer.add_scalar(f'{layer_name}/beta_min', beta.min().item(), step)
    writer.add_scalar(f'{layer_name}/beta_max', beta.max().item(), step)
    writer.add_scalar(f'{layer_name}/beta_std', beta.std().item(), step)
```

### 7.3 Correlation: τ vs Firing Rate

High-τ neurons should fire less frequently (they integrate longer, threshold is harder to reach on average). If correlation is reversed (high-τ neurons fire more), this indicates a pathological configuration — check initialization and weight scaling.

```python
def check_tau_firing_correlation(tau: torch.Tensor, mean_rates: torch.Tensor, layer_name: str):
    # tau: (N,), mean_rates: (N,) — average spike rate per neuron over an epoch
    corr = torch.corrcoef(torch.stack([tau, mean_rates]))[0, 1].item()
    # Expect corr < 0 (higher tau → lower firing rate)
    if corr > 0.2:
        print(f"WARNING [{layer_name}]: tau-rate correlation = {corr:.3f} > 0 (expected negative)")
    return corr
```

### 7.4 τ Drift (Learnable Mode Only)

Track total drift of τ values from initialization to current state:

```python
def compute_tau_drift(tau_init: torch.Tensor, tau_current: torch.Tensor) -> float:
    return ((tau_current - tau_init).abs().sum() / tau_init.numel()).item()
```

Log this scalar once per epoch. Large drift indicates τ is actively adapting (expected in learnable mode). Near-zero drift in learnable mode may indicate vanishing gradients reaching the τ_raw parameter.

### 7.5 Clamp Pressure Warning

Run `check_tau_clamp_pressure` (defined in Section 5) after each epoch when learnable mode is active. Trigger hyperparameter review if pressure exceeds 5%.

---

## 8. ConvSNN Considerations

For convolutional SNN layers, the spatial dimensions (H, W) share one τ per channel. This reduces the parameter count from C*H*W to C — a factor of H*W, which is typically in the hundreds or thousands.

### 8.1 Shape Convention

For a convolutional SNN layer with output shape (B, C, H, W):

- `tau_module = HeterogeneousTau(n=C, ...)` — one τ per output channel
- `beta = tau_module.beta` returns shape `(C,)`
- Broadcast form: `beta.view(1, C, 1, 1)` expands to `(B, C, H, W)` correctly

### 8.2 Initialization for ConvSNN

Apply the same initialization strategies (Section 4) but to n=C rather than n=N*H*W. The Gamma and loguniform strategies operate identically. The preset bank with 5 presets assigns each channel group a distinct τ — ensure C is at least 10 for this to be meaningful.

### 8.3 Memory Overhead

Memory cost of `HeterogeneousTau` for a convolutional layer:
- tau_raw parameter: C float32 values = 4C bytes
- Compared to the weight tensor W: C * C_in * k * k float32 values

For C=256, k=3, C_in=256: weight is 256 * 256 * 9 = 589,824 floats. tau_raw adds 256 floats — overhead of 0.04%. Negligible.

### 8.4 Gradient Scaling

Gradients for τ in a convolutional layer accumulate over all spatial positions (H*W positions contribute to each channel's τ gradient). This effectively scales the τ gradient by H*W relative to the dense case. Apply gradient clipping or a reduced learning rate for τ parameters in deep ConvSNN layers with large spatial dimensions.

---

## 9. Migration from Current Implementation

### 9.1 Current State

`AdvancedLIFNeuron` (neurons.py, lines 330–532) is the only neuron class with per-neuron learnable time constants. It uses logit-β parametrization:

```python
# Current pattern:
logit_beta = torch.log(torch.tensor(beta_init) / (1 - beta_init))
self.log_beta = nn.Parameter(
    torch.full((size,), logit_beta.item()) + torch.randn(size) * 0.1
)

@property
def beta(self) -> torch.Tensor:
    return torch.sigmoid(self.log_beta)

# Forward:
beta = self.beta          # (N,)
self.mem = beta.unsqueeze(0) * self.mem + x
```

`LIFNeuron`, `AdaptiveLIFNeuron`, and `RecurrentLIFNeuron` all use scalar fixed β from constructor arguments. They have no per-neuron time constants.

### 9.2 Target State

After migration:

```
HeterogeneousTau (standalone module)
    ├── Used by: AdvancedLIFNeuron (refactored, τ-based)
    ├── Used by: LIFNeuron (new optional composition)
    ├── Used by: AdaptiveLIFNeuron (new optional composition)
    └── Used by: RecurrentLIFNeuron (new optional composition)
```

### 9.3 Migration Steps

**Step 1 — Create `HeterogeneousTau` module** in a new file `brain_ai/core/hetero_tau.py`. Implement `TauInitConfig` dataclass, `_init_tau` dispatch function, `tau_to_tau_raw` helper, and the `HeterogeneousTau` nn.Module as specified in Section 6.

**Step 2 — Add to `SNNConfig`** the new fields:

```python
tau_granularity: str = 'per_neuron'   # 'per_neuron', 'per_channel', 'per_layer'
tau_min: float = 1.0
tau_max: float = 100.0
tau_init_strategy: str = 'heterogeneous_gamma'
```

**Step 3 — Refactor `AdvancedLIFNeuron`** to use `HeterogeneousTau` internally. Replace `self.log_beta` / `self.beta` property with `self.tau_module`. Preserve the existing constructor signature by converting `beta_init` to an equivalent tau_0 via `tau_0 = -dt / log(beta_init)` and passing that to `TauInitConfig`.

Backward compatibility: the `beta` property on `AdvancedLIFNeuron` can be retained as a thin wrapper over `self.tau_module.beta` so existing code reading `.beta` continues to work.

**Step 4 — Add optional `HeterogeneousTau` composition to `LIFNeuron`**:

```python
class LIFNeuron(nn.Module):
    def __init__(self, ..., use_heterogeneous_tau: bool = False, tau_cfg=None, ...):
        ...
        if use_heterogeneous_tau and size is not None:
            self.tau_module = HeterogeneousTau(n=size, init_cfg=tau_cfg, ...)
        else:
            self.tau_module = None
            self.register_buffer('beta', torch.tensor(beta))
```

In `forward`:

```python
if self.tau_module is not None:
    beta = self.tau_module.beta.unsqueeze(0)   # (1, N)
else:
    beta = self.beta                             # scalar or (1,)
self.mem = beta * self.mem + x
```

**Step 5 — Repeat Step 4 for `AdaptiveLIFNeuron` and `RecurrentLIFNeuron`**.

**Step 6 — Add `tau_granularity` dispatch** in a factory function that instantiates the correct `HeterogeneousTau` based on the config value (`'per_neuron'`, `'per_channel'`, `'per_layer'`).

---

## 10. Anti-Patterns

**Do not use raw β as a parameter.** Parameterizing β directly (not via τ) has no physical grounding. The optimizer can push β toward 1.0 (membrane explosion) or 0.0 (instant decay) without the parameter leaving a meaningful range. Use τ-based parametrization.

**Do not initialize all τ identically when heterogeneous mode is active.** If `use_heterogeneous_tau=True` but all τ start at the same value, the network begins as a homogeneous layer. Without structural asymmetry in initialization, gradient descent does not reliably break the symmetry, and diversity never emerges.

**Do not allow τ = 0.** Instant decay means the membrane resets to 0 each timestep — the neuron cannot integrate across time. Softplus + tau_min prevents this by construction. Never remove the tau_min offset.

**Do not allow τ → ∞ (unclamped).** With no upper bound, the membrane accumulates indefinitely and produces permanent spiking after the first non-zero input. This is invisible during training but produces garbage representations. Always apply the tau_max clamp.

**Do not skip clamping after gradient update.** The clamp in `self.tau` is a dynamic property computed each forward pass — it does not modify tau_raw. The forward-pass clamp is the enforcer; never rely on tau_raw staying bounded on its own.

**Do not compute β = exp(-dt/τ) in fp16.** For large τ, fp16 rounds β=0.9991 to 1.0 — producing a pure integrator with no leak. This causes gradient explosion through long sequences under AMP. Always upcast τ to fp32 before the exp (see Section 5, Constraint 4).

**Do not forget the ConvSNN spatial broadcast.** After retrieving `beta` with shape `(C,)`, apply `.view(1, C, 1, 1)` before multiplying. The dense-layer form `beta.unsqueeze(0)` does not generalize to spatial dimensions.

**Do not use τ to compensate for wrong weight initialization.** If weights cause over-firing or silence, adjusting τ post-hoc masks the root cause. Initialize weights correctly first, then configure τ independently.

**Do not log only β statistics and ignore τ.** A change from τ=50 to τ=100 shifts β from 0.980 to 0.990 — easily below statistical noise in β, yet it doubles the integration horizon. Always log τ (log-scale histograms) as the primary diagnostic.
