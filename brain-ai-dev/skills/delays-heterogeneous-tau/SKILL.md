---
name: BrainAI Learnable Delays + Heterogeneous Time Constants
description: >
  This skill should be used when the user asks to "add learnable delays",
  "implement DCLS delays", "add Gaussian interpolation delays",
  "fix delay module", "add heterogeneous tau", "add learnable time constants",
  "add per-neuron tau", "fix delay memory", "debug delay overhead",
  "add sigma annealing", "fix delay discretization", "run delay ablation",
  "add delay granularity", "fix tau initialization", "add gamma-distributed tau",
  "add loguniform tau", "add preset tau bank", "fix tau parameterization",
  "run ablation matrix", "compare delays vs baseline", "fix delay gradient flow",
  "add DelayLinear", "add DelayConv1d", "fix ring buffer", "debug delay histogram",
  "fix tau drift", "add tau logging", "fix beta explosion with heterogeneous tau",
  or mentions DelayConfig, TauConfig, DCLS, delay_granularity, sigma_schedule,
  tau_raw, bin_accumulation, delay_entropy, heterogeneous_tau, or ablation_runner
  in the BrainAI cognitive architecture.
version: 0.1.0
---

# BrainAI Learnable Delays + Heterogeneous Time Constants

## Purpose

Add two temporal-expressivity knobs to the existing LIF stack: learnable axonal/synaptic
delays (DCLS-style Gaussian interpolation) and heterogeneous membrane time constants
(per-neuron tau/beta). Both are fully optional behind flags, compose cleanly with
`core/neurons.py` + `core/snn.py`, and support clean ablations.

This skill does NOT redesign the LIF neuron contract (that is spiking-core's domain).
It standardizes the delay module, tau module, config flags, ablation framework, and
testing discipline that sit on top of the core spiking substrate.

## Key Files

| File | Role |
|------|------|
| `brain_ai/core/delays.py` | DelayLinear, DelayConv1d, SpikeHistoryBuffer |
| `brain_ai/core/heterogeneous_tau.py` | HeterogeneousTau module, TauInitConfig |
| `brain_ai/config.py` | DelayConfig, TauConfig dataclasses (integrated into SNNConfig) |
| `brain_ai/core/neurons.py` | AdvancedLIFNeuron (existing delay/tau consumer) |
| `brain_ai/core/snn.py` | SNNCore, ConvSNN (integrate delay + tau modules) |

## Learnable Delays (DCLS-Style)

### Core Idea

A delayed synapse is a 1D convolution where the kernel has a single non-zero element at
position d. Training with a discrete index has zero gradient; DCLS solves this with
Gaussian interpolation: the tap becomes a narrow Gaussian bump centered at learnable
position d, annealed from wide to narrow over training.

### Forward Pass

| Mode | Computation | When |
|------|------------|------|
| Training | `I_i[t] = sum_j w_ij * sum_n g(n; d_ij, sigma) * S_j[t-n]` | `model.train()` |
| Inference | `I_i[t] = sum_j w_ij * S_j[t - round(d_ij)]` | `model.eval()` + `discretize=True` |

Gaussian kernel: `g(n; d, sigma) = exp(-(n-d)^2/(2*sigma^2)) / Z`, normalized over K bins.

### Delay Parametrization

Store unconstrained `d_raw`; map to valid range via `d = (Td-1) * sigmoid(d_raw)`.
Final safety clamp to `[0, Td-1]`.

### Granularity Modes

| Mode | d Shape | Use Case |
|------|---------|----------|
| `per_synapse` | `(out, in)` | Most expressive, O(N^2) params |
| `per_output` | `(out, 1)` | Good default balance |
| `per_input` | `(1, in)` | Input-centric delays |
| `per_block` | `(out_blocks, in_blocks)` | Scalable to large layers |

### Efficient Implementation (Bin-Accumulation)

For each integer delay `n` in `[0, Td-1]`:
1. Extract shifted spikes: `S_n = spike_history[:, n, :]`
2. Compute Gaussian weight contribution for bins near `n`
3. Accumulate: `I += S_n @ W_effective[n].T`

Memory: O(Td * out * in) for bin matrices. Compute: O(Td * B * out * in).

### Sigma Annealing Schedule

`sigma(epoch) = sigma_end + (sigma_start - sigma_end) * exp(-epoch / sigma_decay_epochs)`

Default: sigma_start=1.0, sigma_end=0.5, decay over 50 epochs. At inference, round
delays to integers.

### Memory Guards

- Hard cap Td (default 32, warn above 64)
- Hard cap K (interpolation bins, default 3)
- Runtime warning when `out * in * Td > threshold`
- Profiler hooks log "delay overhead factor" vs baseline

## Heterogeneous Tau / Beta

### Two Modes

| Mode | Tau Behavior | Gradient | Use Case |
|------|-------------|----------|----------|
| `heterogeneous_fixed` | Diverse tau, frozen | No grad through tau | Structural diversity |
| `heterogeneous_learnable` | Diverse tau, trained | Grad through tau | Task-adaptive timescales |

### Tau-to-Beta Parametrization (Softplus-Based)

```python
tau = tau_min + softplus(tau_raw)    # ensures tau > tau_min
tau = clamp(tau, tau_min, tau_max)   # hard upper cap
beta = exp(-dt / tau)                # stable beta in (0, 1)
```

Interpretable: tau has physical meaning (membrane time constant in simulation steps).

### Initialization Strategies

| Strategy | Distribution | When |
|----------|-------------|------|
| `homogeneous` | All tau = tau_0 | Baseline/ablation control |
| `heterogeneous_gamma` | tau ~ Gamma(k, theta) clamped | Biologically plausible |
| `heterogeneous_loguniform` | log(tau) ~ Uniform | Uniform timescale coverage |
| `preset_bank` | tau in {2, 5, 10, 20, 50} | Structured, low param count |

### Granularity

| Mode | Shape | Use Case |
|------|-------|----------|
| `per_neuron` | `(N,)` | Default for FC layers |
| `per_channel` | `(C, 1, 1)` | ConvSNN (shared across spatial) |
| `per_layer` | scalar | Baseline |

## Config Flags

### DelayConfig Fields

| Field | Values | Default |
|-------|--------|---------|
| `mode` | `off` / `fixed_random` / `learnable_dcls` | `learnable_dcls` |
| `max_delay` | int | 16 |
| `num_bins` | int | 3 |
| `granularity` | `per_synapse` / `per_output` / `per_input` / `per_block` | `per_output` |
| `sigma_schedule` | `constant` / `decreasing` | `decreasing` |
| `discretize_at_inference` | bool | `True` |

### TauConfig Fields

| Field | Values | Default |
|-------|--------|---------|
| `mode` | `homogeneous_fixed` / `heterogeneous_fixed` / `heterogeneous_learnable` | `heterogeneous_learnable` |
| `granularity` | `per_neuron` / `per_channel` / `per_layer` | `per_neuron` |
| `init_strategy` | `homogeneous` / `heterogeneous_gamma` / `heterogeneous_loguniform` / `preset_bank` | `heterogeneous_loguniform` |
| `tau_min` / `tau_max` | float | 1.0 / 100.0 |

## Ablation Matrix

| Config | Delays | Tau | Purpose |
|--------|--------|-----|---------|
| `baseline` | off | homogeneous_fixed | Control |
| `delays_only` | learnable_dcls | homogeneous_fixed | Isolate delay contribution |
| `hetero_only` | off | heterogeneous_learnable | Isolate tau contribution |
| `both` | learnable_dcls | heterogeneous_learnable | Full temporal expressivity |

Run with identical seeds, optimizer, schedule, width, and dataset split.
Emit accuracy curves, delay/tau histograms, and overhead factor.

## Diagnostics

| Metric | Source | Healthy Range |
|--------|--------|--------------|
| `delay_mean` | per-layer | Spread across [0, Td-1] |
| `delay_entropy` | per-layer | > 1.0 (not collapsed) |
| `delay_boundary_pct` | per-layer | < 10% (low clamp pressure) |
| `sigma` | per-module | Decreasing per schedule |
| `tau_mean` | per-layer | Near tau_0 with reasonable drift |
| `tau_firing_rate_corr` | per-layer | Negative (high tau -> lower rate) |
| `overhead_factor` | global | < 2.0x for delays, < 1.2x for tau |

## Common Failure Modes

- "Delays all collapse to 0 or Td-1" -- sigma annealing too fast, or LR too high for d_raw
- "Memory explodes with per_synapse granularity" -- switch to per_output or per_block
- "Tau all drift to tau_min" -- LR too high for tau_raw, or task has no temporal structure
- "Inference accuracy drops sharply vs training" -- discretization mismatch, check sigma schedule
- "No gradient through delays" -- spike history not in computation graph
- "Beta = 1.0 after tau_max increase" -- tau_max too large, exp(-dt/tau) approaches 1.0

## Anti-Patterns

- Do NOT materialize dense (out x in x Td) kernels -- use bin-accumulation.
- Do NOT use softmax attention over delay taps (not DCLS-style, loses continuous semantics).
- Do NOT share sigma across layers without explicit reason.
- Do NOT use raw beta as parameter -- use tau-to-beta mapping for interpretability.
- Do NOT initialize all tau identically when heterogeneous mode is on.
- Do NOT run ablations with different random seeds across configs.
- Do NOT skip the baseline config in ablation comparisons.
- Do NOT forget to log parameter count deltas between ablation configs.

## Additional Resources

### Reference Files

- **`references/dcls-delays.md`** -- Full DCLS theory, Gaussian interpolation math, bin-accumulation vs ring-buffer, sigma schedule, parametrization, migration path
- **`references/heterogeneous-tau.md`** -- Fixed vs learnable modes, softplus parametrization, initialization strategies, stability constraints, ConvSNN integration
- **`references/ablation-framework.md`** -- Config flags, 4-config matrix, benchmark selection (SHD/SSC/sMNIST), logging contract, performance budget
- **`references/testing-matrix.md`** -- 6 test categories, concrete test specs, memory guards, checkpoint round-trip, 26-item "done when" checklist

### Scripts

- **`scripts/validate_delays_tau.py`** -- Runtime contract validation for delay and tau modules
- **`scripts/gen_delay_tau_tests.py`** -- Generate parameterized pytest test suite
- **`scripts/ablation_runner.py`** -- Run 4-config ablation with identical seeds, emit comparison report

### Assets

- **`assets/delay_module_template.py`** -- Template for `brain_ai/core/delays.py` (DelayLinear, DelayConv1d, Gaussian interp)
- **`assets/heterogeneous_tau_template.py`** -- Template for `brain_ai/core/heterogeneous_tau.py` (HeterogeneousTau, init strategies)
- **`assets/delay_tau_config_template.py`** -- Template for DelayConfig, TauConfig, AblationConfig dataclasses
