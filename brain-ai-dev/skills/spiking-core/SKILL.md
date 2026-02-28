---
name: BrainAI Spiking Core (LIF Family) + Surrogate Gradient Training
description: >
  This skill should be used when the user asks to "fix spiking neuron",
  "add a new neuron variant", "fix surrogate gradient", "debug SNN training",
  "fix SNN convergence", "add truncated BPTT", "fix state leakage",
  "implement explicit state", "add detach_state", "fix membrane explosion",
  "add refractory period", "fix gradient flow through spikes", "add learnable threshold",
  "add learnable beta", "fix mixed precision SNN", "validate spiking contracts",
  "debug firing rates", "fix dead neurons", "fix saturated neurons",
  "unify ConvSNN and MLP SNN unrolling", "add time unroll utility",
  "fix in-place ops in SNN", "clamp beta", "swap surrogate gradient",
  "add spike-frequency adaptation", "fix recurrent SNN", "debug membrane stats",
  or mentions LIFNeuron, SpikingState, surrogate gradient, snn_unroll,
  truncated BPTT, membrane potential, firing rate, or spike sparsity
  in the BrainAI cognitive architecture.
version: 0.1.0
---

# BrainAI Spiking Core + Surrogate Gradient Training

## Purpose

Enforce correctness of the neuronal physics and gradient plumbing layer. If the
spiking substrate is wrong, everything above it becomes un-debuggable: training
will not converge, states will leak across batches, and CUDA/CPU behavior will drift.

This skill does NOT design high-level architectures (SNNCore, ConvSNN wiring). It
standardizes the neuron contract, surrogate backward pass, state lifecycle, and
time-unrolling discipline that all spiking architectures depend on.

## Key Files

| File | Role |
|------|------|
| `brain_ai/core/neurons.py` | LIF family neurons, `SpikingState`, `SpikingNeuronBase` |
| `brain_ai/core/surrogates.py` | Surrogate gradient functions (ATan, FastSigmoid, STE) |
| `brain_ai/core/unroll.py` | Single time-unroll utility with truncated BPTT |
| `brain_ai/core/snn.py` | SNNCore (MLP) and ConvSNN architectures |
| `brain_ai/core/losses.py` | Spike losses, regularizers, `compute_snn_metrics` |
| `brain_ai/core/encoding.py` | Rate, temporal, latency, population, delta encoders |
| `brain_ai/config.py` | `SNNConfig` dataclass (lines 41-73) |

## The Core Contract

### A) Time Semantics

Every spiking layer supports two modes via a single convention:

| Mode | Input Shape | Output Shape | State |
|------|------------|-------------|-------|
| Step | `(B, ...)` | `(B, ...)` + next_state | Single timestep |
| Sequence | `(B, T, ...)` | `(B, T, ...)` + final_state | Via `snn_unroll` utility |

Canonical convention: **batch-first, time-second** `(B, T, ...)` everywhere.

### B) State Semantics — No Hidden Surprises

State is explicit and batch-aligned. Every neuron provides:

```python
@dataclass
class SpikingState:
    v: Tensor                              # (B, N) membrane potential
    i: Optional[Tensor] = None             # synaptic current
    a: Optional[Tensor] = None             # adaptation variable
    prev_spk: Optional[Tensor] = None      # previous spikes (recurrent)
    spike_history: Optional[Tensor] = None # delay buffer (advanced)
    ref: Optional[Tensor] = None           # refractory timer
```

Required methods on every neuron:

| Method | Purpose |
|--------|---------|
| `reset_state(B, device, dtype)` | Fresh zeros, never store on `self` |
| `detach_state(state)` | Cut computation graph, preserve values |
| `forward(x, state=, carry_state=)` | Explicit state in/out |

### C) Surrogate Gradient Contract

Forward is **always** binary Heaviside. Backward uses a smooth approximation:

| Surrogate | Gradient Formula | Max Grad (normalized) |
|-----------|-----------------|----------------------|
| ATan | `α / (2(1 + (παx)²))` | 1.0 (α=2.0) |
| FastSigmoid | `slope / (2(1 + slope|x|)²)` | 1.0 (slope=2.0) |
| STE | `1.0` (constant) | 1.0 |

All surrogates normalized to max gradient ≈ 1.0 (SpikingJelly convention).

## LIF Family Neurons

Four variants sharing `SpikingNeuronBase`:

| Variant | Key Addition | Extra State |
|---------|-------------|-------------|
| `LIFNeuron` | Baseline: `v = β·v + i`, spike, reset | `v` only |
| `AdaptiveLIFNeuron` | Dynamic threshold: `v_th_eff = v_th + α·a` | `v`, `a` |
| `RecurrentLIFNeuron` | Lateral connections: `i = W_in·x + W_rec·s_prev` | `v`, `prev_spk` |
| `AdvancedLIFNeuron` | Learnable delays, heterogeneous τ | `v`, `spike_history`, `a` |

See **`references/lif-dynamics.md`** for full update equations, reset rules, and
numerical stability policies.

## Numerical Stability (Enforced, Not Optional)

| Rule | Enforcement |
|------|-------------|
| Clamp β ∈ [0, 0.999] | Sigmoid parameterization or post-step clamp |
| No in-place ops | `v = v + current`, never `v.add_(current)` |
| fp32 state accumulation | Even under AMP, v/i stay fp32 during unrolls |
| Binary spikes | Forward outputs exactly 0.0 or 1.0 |
| Threshold gradient flow | `∂s/∂v_th = -surrogate'(v - v_th)` via autograd |

See **`references/surrogate-gradients.md`** for slope effects, normalization tables,
and gradient chain analysis.

## Time Unrolling

One utility for all architectures. ConvSNN and MLP SNN share the same unroll path:

```python
spikes, final_state, traces = snn_unroll(
    cell_fn=my_cell,     # (x_t, state) -> (spk_t, new_state)
    inputs=x,            # (B, T, ...)
    initial_state=state,
    chunk_size=10,       # truncated BPTT window
)
```

Truncated BPTT: unroll in windows, detach state between chunks. Forward output is
identical to full BPTT; only gradient scope changes.

See **`references/time-unrolling.md`** for ConvSNN time batching, recording modes,
and memory estimation.

## Reset vs Carry

| Mode | When | State Behavior |
|------|------|---------------|
| Training (default) | Each batch | Reset to zeros |
| Truncated BPTT | Between chunks | Detach (values preserved, graph cut) |
| Streaming inference | Across calls | Carry with `carry_state=True` |

See **`references/state-management.md`** for explicit state API, partial reset,
mixed precision policy, and migration from implicit `self.mem`.

## Debug Surfaces

Accessible via `return_details=True` without significant overhead:

| Probe | Shape/Type | Healthy Range |
|-------|-----------|--------------|
| `firing_rate` | per-layer float | 0.01 – 0.50 |
| `membrane_mean` | per-layer float | finite, < 10·v_th |
| `membrane_var` | per-layer float | finite, > 0 |
| `spike_sparsity` | per-layer float | 0.50 – 0.99 |
| `dead_neuron_frac` | per-layer float | < 0.05 |
| `saturated_neuron_frac` | per-layer float | < 0.05 |

## Common Failure Modes

- "Training converges then suddenly NaNs" — β ≥ 1.0 (unclamped), membrane explodes
- "Gradients are zero" — spikes detached before loss, or surrogate slope too narrow
- "Different results each run with same seed" — state leaked from previous batch
- "ConvSNN works but MLP breaks" — different unroll loops, fix with `snn_unroll`
- "Loss plateaus at step 5000" — fp16 state accumulation drift, switch to fp32
- "CPU and CUDA diverge" — accumulation order + fp precision; test with tolerance

## Anti-Patterns

- Do NOT store state on `self`. Pass state explicitly.
- Do NOT use in-place ops on tensors in the autograd graph.
- Do NOT mix reset rules across layers in the same network.
- Do NOT use soft sigmoid in forward. Forward is binary Heaviside.
- Do NOT use different unroll loops for Conv and MLP variants.
- Do NOT skip `detach_state()` between BPTT chunks.
- Do NOT use β=1.0 (no leak = no temporal gradient signal).

## Extension Hooks (Design Now, Implement Later)

| Extension | Interface Point |
|-----------|----------------|
| Learnable per-neuron thresholds | `SpikingNeuronBase.__init__(learnable_threshold=True)` |
| Synaptic current models (alpha/PSC) | `SpikingState.i` field + PSC update in `_step` |
| Refractory period | `SpikingState.ref` field + mask in spike computation |
| Mixed precision formal policy | `dtype_policy` parameter on `SpikingNeuronBase` |
| Recording modes | `return_details=True` path in forward |

## Additional Resources

### Reference Files

- **`references/lif-dynamics.md`** — Full update equations, reset rules, numerical stability, extension hooks
- **`references/surrogate-gradients.md`** — Surrogate implementations, normalization, slope effects, gradient chains
- **`references/time-unrolling.md`** — Unroll utility, truncated BPTT, ConvSNN batching, recording modes
- **`references/state-management.md`** — Explicit state API, reset/detach/carry, mixed precision, migration
- **`references/testing-matrix.md`** — 6 test categories, concrete test specs, "done when" checklist

### Scripts

- **`scripts/validate_spiking.py`** — Runtime contract validation for all neuron variants
- **`scripts/gen_spiking_tests.py`** — Generate parameterized pytest test suite
- **`scripts/spiking_debug_report.py`** — Firing rates, membrane stats, gradient norms, sparsity

### Assets

- **`assets/neurons_template.py`** — Template for `brain_ai/core/neurons.py` with explicit state API
- **`assets/surrogate_functions_template.py`** — Template for `brain_ai/core/surrogates.py`
- **`assets/snn_unroll_template.py`** — Template for `brain_ai/core/unroll.py` unroll utility
