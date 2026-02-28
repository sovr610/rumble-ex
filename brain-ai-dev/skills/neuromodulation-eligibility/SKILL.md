---
name: Neuromodulation + Eligibility Traces (Three-Factor Learning)
description: >-
  This skill should be used when the user asks to "implement eligibility traces",
  "add three-factor learning", "implement neuromodulation", "add DA/ACh/NE/5-HT modulators",
  "implement STDP eligibility", "add synaptic plasticity", "implement online plasticity",
  "add reward-modulated learning", "implement trace dynamics", "add Dutch traces",
  "implement accumulating traces", "add replacing traces", "implement neuromodulatory gate",
  "add plasticity gain", "implement delayed reward association", "add eligibility decay",
  "implement spike-based traces", "add rate-based traces", "implement three-factor weight update",
  "add fast memory adapter", "implement bioplausible learning",
  or mentions eligibility traces, three-factor learning rules, neuromodulatory signals,
  STDP-based eligibility, reward-modulated plasticity, DA/ACh/NE/5-HT computation,
  or online synaptic updates in the cognitive pipeline.
version: 0.1.0
---

# Neuromodulation + Eligibility Traces (Three-Factor Learning)

## Purpose

This skill standardizes the "biologically inspired online plasticity" path: eligibility traces
accumulate local pre/post correlations (spike-based or rate-based), then a delayed third-factor
neuromodulatory signal gates when and how weights change. This is the canonical three-factor
learning rule where synaptic changes require a modulatory signal beyond just pre/post activity.
The non-negotiable goals are correct gating semantics (zero modulator = zero update) and
deterministic trace dynamics.

## Key Files

| Target Module | Template Asset | Purpose |
|---|---|---|
| `brain_ai/meta/eligibility_traces.py` | `assets/eligibility_traces_template.py` | EligibilityTraceModule: trace dynamics, STDP kernels, decay, reset/carry |
| `brain_ai/meta/neuromodulation.py` | `assets/neuromodulation_template.py` | NeuromodulatoryGate: DA/ACh/NE/5-HT computation, plasticity gain |
| `brain_ai/meta/three_factor_update.py` | `assets/three_factor_update_template.py` | ThreeFactorUpdate: weight update rules, online/hybrid integration |
| `brain_ai/meta/plasticity_diagnostics.py` | `assets/plasticity_diagnostics_template.py` | PlasticityTrace: logging, modulator stats, trace inspection |
| `brain_ai/config.py` (extend) | `assets/neuromod_config_template.py` | EligibilityConfig, NeuromodConfig, ThreeFactorConfig, PlasticityConfig |

## Public Contract

```python
# EligibilityTraceModule
reset(batch_size, device) -> None
update(pre, post, dt=None) -> e  # eligibility tensor
apply_update(weights, mod_signal, *, lr, clamp) -> updated_weights

# NeuromodulatoryGate
forward(signals, state=None) -> modulators, global_plasticity
update_state(reward, novelty, urgency, patience) -> state
```

`pre` and `post` are activations or spike trains `(B, N_pre)` / `(B, N_post)`. `e` is the
eligibility trace `(B, N_post, N_pre)` or `(B, N)` for diagonal variants. `mod_signal` is a
scalar or `(B,)` third-factor signal gating the update.

## Core Output Contract

| Field | Shape / Type | Description |
|---|---|---|
| `e` | `(B, N_post, N_pre)` | Eligibility trace matrix |
| `modulators` | `Dict[str, Tensor]` | `{DA, ACh, NE, 5HT}` each `(B,)` or `(B,1)` |
| `global_plasticity` | `(B,)` | Combined scalar plasticity gain |
| `delta_w` | same as weights | Weight update tensor |
| `trace_log` | `Optional[PlasticityTrace]` | Per-step diagnostics when `return_details=True` |

**Hard invariants**:
- If third-factor modulator is zero, `delta_w` is exactly zero regardless of pre/post activity.
- Eligibility traces have no cross-batch leakage (batch dimension is independent).
- `reset()` clears traces to zero; `carry` mode preserves traces across calls.
- All trace computations run in fp32 for numerical stability under AMP.

## Eligibility Trace Dynamics

Three trace types with configurable STDP kernel:

| Trace Type | Update Rule | Use Case |
|---|---|---|
| Accumulating | `e += f(pre, post)` | Standard eligibility accumulation |
| Replacing | `e = max(e, f(pre, post))` | Event-driven, prevents unbounded growth |
| Dutch | `e = (1 - α)*e + f(pre, post)` | Hybrid decay + event, balanced dynamics |

All types share the decay step: `e(t+1) = (1 - dt/tau_e) * e(t) + trace_update(t)`

The pre/post correlation function `f` supports:
- **Spike-based**: pair-based STDP with timing-dependent kernel `f(Δt)`
- **Rate-based**: `f(pre, post) = pre ⊗ post` (outer product or element-wise)

See `references/eligibility-dynamics.md` for trace mathematics, STDP kernels, decay analysis, and
rate vs spike implementation details.

## Neuromodulator Computation

Four modulators computed deterministically from observable signals:

| Modulator | Input Signal | Biological Analog | Output Range |
|---|---|---|---|
| DA (Dopamine) | Reward / TD error proxy | Reward prediction error | [-1, 1] |
| ACh (Acetylcholine) | Novelty / uncertainty / entropy | Attention / learning gate | [0, 1] |
| NE (Norepinephrine) | Urgency / surprise magnitude | Arousal / exploration | [0, 1] |
| 5-HT (Serotonin) | Patience / long-horizon value | Discounting / exploitation | [0, 1] |

Global plasticity gain: `g = f(DA, ACh, NE, 5HT)` — configurable combination function
(weighted sum, gated product, or learned MLP).

See `references/neuromodulator-signals.md` for signal mapping, computation details, and bounded
output guarantees.

## Weight Update Integration

Two integration modes:

| Mode | Mechanism | Use Case |
|---|---|---|
| **Online plasticity** | `Δw = lr * mod_signal * e`; applied directly to designated layers | Bioplausible fast adaptation, streaming inference |
| **Hybrid training** | Three-factor update as auxiliary loss/regularizer alongside backprop | Few-shot improvement, nonstationary streams |

Online mode targets "fast memory" adapters or designated SNN synapses. Hybrid mode computes
eligibility as an auxiliary signal while the rest of the model trains normally via backprop.

See `references/three-factor-rules.md` for update rule mathematics, clamping, and integration patterns.

## Integration Points

| Module | Integration | Purpose |
|---|---|---|
| SNN core (`core/`) | Spike-based traces on SNN synapses | Bioplausible SNN plasticity |
| Workspace (`workspace/`) | ACh-gated attention modulation | Novelty-driven workspace competition |
| Reasoning (`reasoning/`) | DA-gated confidence updates | Reward-modulated System 2 refinement |
| Meta (`meta/`) | Eligibility-augmented inner loop | Online adaptation complement to MAML |

See `references/integration-hooks.md` for per-module integration patterns and adapter design.

## Configuration Surface

### EligibilityConfig

| Field | Default | Purpose |
|---|---|---|
| `trace_type` | `"accumulating"` | `"accumulating"`, `"replacing"`, `"dutch"` |
| `tau_e` | 20.0 | Eligibility decay time constant (ms or steps) |
| `kernel` | `"rate"` | `"rate"`, `"stdp_pair"`, `"stdp_symmetric"` |
| `stdp_tau_plus` | 20.0 | STDP potentiation time constant |
| `stdp_tau_minus` | 20.0 | STDP depression time constant |
| `dutch_alpha` | 0.1 | Dutch trace replacement rate |
| `clamp_range` | `[-5.0, 5.0]` | Trace value clamp bounds |

### NeuromodConfig

| Field | Default | Purpose |
|---|---|---|
| `da_source` | `"reward"` | DA input signal mapping |
| `ach_source` | `"novelty"` | ACh input signal mapping |
| `ne_source` | `"urgency"` | NE input signal mapping |
| `sht_source` | `"patience"` | 5-HT input signal mapping |
| `combination_fn` | `"weighted_sum"` | `"weighted_sum"`, `"gated_product"`, `"mlp"` |
| `modulator_hidden_dim` | 64 | Hidden dim for MLP combination |

### ThreeFactorConfig

| Field | Default | Purpose |
|---|---|---|
| `mode` | `"online"` | `"online"`, `"hybrid"`, `"auxiliary_loss"` |
| `lr` | 0.001 | Plasticity learning rate |
| `weight_clamp` | `[-1.0, 1.0]` | Weight update clamp range |
| `target_layers` | `"all_eligible"` | Which layers receive three-factor updates |
| `update_frequency` | 1 | Steps between weight updates |

Presets: `PlasticityFullConfig.minimal()`, `.dev()`, `.production()`.

## Done-When Gates

| Gate | Test | Threshold |
|---|---|---|
| **(a) Third-factor gating** | Set mod_signal=0; assert delta_w == 0 with strong pre/post activity; set mod_signal != 0; assert delta_w != 0 | Exact zero / non-zero |
| **(b) Deterministic traces** | Fixed pre/post sequences + dt; assert identical e(t) across 10 runs | Exact match |
| **(c) Delayed reward association** | Toy task: reward arrives after delay; eligibility traces enable learning despite gap; loss decreases | Loss decreases, accuracy improves |

## Common Failure Modes

| Symptom | Cause | Fix |
|---|---|---|
| Updates occur with zero modulator | Gating not applied correctly | Assert `delta_w = mod * e` with explicit zero check |
| Traces explode | No decay or clamp | Verify tau_e > 0, enable clamp_range |
| Cross-batch leakage | Shared trace state | Ensure batch dim is independent, reset between episodes |
| NaN under AMP | fp16 trace accumulation | Force fp32 for all trace/modulator computation |
| DA always saturated | Reward signal not normalized | Normalize reward to [-1,1] before DA computation |
| ACh always zero | Novelty source not connected | Verify HTM anomaly or entropy signal is flowing |
| No learning despite high eligibility | Modulator timing mismatch | Check delay between activity and reward signal arrival |
| Traces identical for all inputs | Kernel function collapsed | Verify pre/post differ across inputs, check kernel |

## Anti-Patterns

- **Skipping the third factor** -- the whole point is gated plasticity; without it, this is just Hebbian
- **fp16 trace computation** -- eligibility accumulation needs fp32 precision
- **Shared traces across batch items** -- each batch item has independent trace state
- **Unbounded trace accumulation** -- always apply decay and/or clamp
- **Hardcoded modulator weights** -- use NeuromodConfig, not magic numbers
- **Online updates on all layers** -- designate specific "eligible" layers via config
- **No reset between episodes** -- eligibility from previous episode contaminates current
- **Testing without delayed reward** -- the delayed association test is the core validation

## Additional Resources

### Reference Files

- **`references/eligibility-dynamics.md`** -- Trace types, STDP kernels, decay mathematics, rate vs spike, carry/reset semantics
- **`references/neuromodulator-signals.md`** -- DA/ACh/NE/5-HT computation, signal mapping, bounded outputs, combination functions
- **`references/three-factor-rules.md`** -- Weight update rules, online vs hybrid mode, clamping, convergence properties
- **`references/integration-hooks.md`** -- SNN/workspace/reasoning integration, fast memory adapters, per-module patterns
- **`references/testing-matrix.md`** -- All test cases: gating, determinism, delayed reward, stability, integration

### Asset Templates

- **`assets/eligibility_traces_template.py`** -- EligibilityTraceModule, trace types, STDP kernels, decay, self-test
- **`assets/neuromodulation_template.py`** -- NeuromodulatoryGate, modulator computation, plasticity gain, self-test
- **`assets/three_factor_update_template.py`** -- ThreeFactorUpdate, weight update rules, online/hybrid, self-test
- **`assets/plasticity_diagnostics_template.py`** -- PlasticityTrace, logging, modulator stats, inspection, self-test
- **`assets/neuromod_config_template.py`** -- All configs, presets, serialization, self-test

### Scripts

- **`scripts/validate_neuromod.py`** -- Runtime contract validation (third-factor gating, trace determinism, delayed reward)
- **`scripts/gen_neuromod_tests.py`** -- Generates `tests/test_neuromodulation.py` (~80+ test cases)
- **`scripts/plasticity_benchmark.py`** -- Benchmark trace update throughput, modulator computation, weight update speed
