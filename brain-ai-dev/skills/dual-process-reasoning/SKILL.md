---
name: Dual-Process Reasoning (System 1/2 + Metacognitive Control)
description: >-
  This skill should be used when the user asks to "implement dual-process reasoning", "add System 1/2 router",
  "implement metacognitive control", "add confidence calibration", "implement temperature scaling",
  "add iterative refinement", "implement convergence detection", "add effort budget",
  "implement reasoning trace", "add novelty scoring", "implement routing policy",
  "add System 2 early stopping", "implement calibrated confidence", "add isotonic calibration",
  "implement GRU reasoning loop", "add selective S2 execution", "implement halt criteria",
  "add reasoning budget enforcement", "implement route score computation",
  or mentions dual-process reasoning, System 1/2 routing, metacognitive control,
  confidence calibration, convergence criteria, or reasoning traces in the cognitive pipeline.
version: 0.1.0
---

# Dual-Process Reasoning (System 1/2 + Metacognitive Control)

## Purpose

This skill standardizes the "cognitive control" layer (Phase 6): given a workspace representation,
produce a fast System 1 prediction with calibrated confidence, route uncertain or novel inputs
to an iterative System 2 refinement loop with convergence detection, and return a traceable
reasoning output. Metacognition controls the routing policy deterministically via confidence,
novelty, and effort budget signals.

## Key Files

| Target Module | Template Asset | Purpose |
|---|---|---|
| `brain_ai/reasoning/system1.py` | `assets/system1_template.py` | System1Fast: single-pass predictor + confidence heads |
| `brain_ai/reasoning/calibration.py` | `assets/calibration_template.py` | TemperatureScaler, IsotonicCalibrator, calibration utilities |
| `brain_ai/reasoning/system2.py` | `assets/system2_template.py` | System2Iterative: GRU refinement loop, convergence, halting |
| `brain_ai/reasoning/metacognition.py` | `assets/metacognition_template.py` | MetacognitiveRouter: routing policy, novelty scoring, budget |
| `brain_ai/reasoning/dual_process.py` | `assets/dual_process_template.py` | DualProcessReasoner: main module wiring S1+S2+metacognition |
| `brain_ai/reasoning/trace.py` | `assets/reasoning_trace_template.py` | ReasoningTrace, StepTrace, serialization |
| `brain_ai/config.py` (extend) | `assets/reasoning_config_template.py` | System1Config, System2Config, MetacognitionConfig, etc. |

## Public Contract

```python
forward(x, *, context=None, return_details=False, state=None) -> ReasoningOutput
```

Input `x` is `(B, D)` pooled workspace or `(B, K, D)` slots. Optional `context` carries workspace
slots, WM state, or HTM anomaly scores. `state` carries recurrent state across time steps.

## ReasoningOutput Contract

| Field | Shape / Type | Description |
|---|---|---|
| `y` | `(B, output_dim)` | Final output logits/embedding/decision signal |
| `used_system2` | `(B,)` bool | Per-item flag: True if System 2 was invoked |
| `s1` | `System1Result` | `{y1, conf_raw, conf_calibrated, entropy, margin, uncertainty_metrics}` |
| `s2` | `System2Result` | `{y2, steps_used, converged, halt_reason}` (None if S2 not used) |
| `trace` | `ReasoningTrace` or None | Full trace only when `return_details=True` |
| `aux` | `Dict[str, Tensor]` | `{novelty, route_score, effort_budget, s2_fraction}` |

**Hard invariants**:
- Routing is **deterministic** for a fixed seed and deterministic settings.
- `return_details=False` produces `trace=None` with near-zero tracing overhead.
- Per-item routing is computed independently (no cross-batch sorting/reordering).

## System 1: Fast Predictor + Calibrated Confidence

Single-pass feed-forward (1-3 layer MLP or shallow transformer block). Outputs both prediction
and multiple uncertainty proxies:

| Metric | Formula | Purpose |
|---|---|---|
| `conf_raw` | `softmax(logits).max(dim=-1)` | Raw maximum probability |
| `entropy` | `-sum(p * log(p))` | Distribution spread |
| `margin` | `top1_logit - top2_logit` | Decision boundary distance |

Route on a **combined uncertainty score**, not a single scalar. Calibration utilities
(TemperatureScaler, IsotonicCalibrator) map raw confidence to calibrated probabilities.
Calibrator must be serializable, included in checkpoints, and frozen during eval.

See `references/system1-confidence.md` for architecture details, confidence head design, and calibration fitting.

## System 2: Iterative Refinement with Convergence

GRU-based iterative loop starting from S1 prediction:

1. Initialize `y0 = y1`, `h0 = summary(x, context)`
2. For step `k` in `[1..steps_budget]`: refine `yk`, update `hk`, compute delta metrics
3. Halt when convergence criterion met or budget exhausted

**Halt criteria** (two-part rule):
- Primary: `KL(p_k || p_{k-1}) < eps` for `M` consecutive steps
- Secondary: steps hit `max_steps` or budget exhausted

**Halt reasons**: `"converged"`, `"max_steps"`, `"budget_exhausted"`, `"nan_guard"`

Only the subset of items routed to S2 enters the loop (scatter/gather for efficiency).

See `references/system2-convergence.md` for loop structure, convergence criteria, and effort budget enforcement.

## Metacognitive Routing

Deterministic routing function combining multiple signals:

```
route_score = w_conf * (1 - calibrated_conf)
            + w_novelty * novelty
            + w_anomaly * anomaly
            - w_budget * remaining_budget_penalty

used_system2 = route_score >= route_threshold
steps_budget = clamp(round(base_steps + alpha * route_score), 1, max_steps)
```

No sampling, no nondeterministic GPU operations, explicit tie-breaking.

See `references/metacognitive-routing.md` for routing policy, novelty scoring options, and budget allocation.

## Reasoning Trace

JSON-serializable trace schema with configurable detail level:

- **Route section**: used_system2, route_score, threshold, confidence metrics, novelty, budget
- **System 1 section**: top-k logits (indices + values), not full tensors
- **System 2 section**: per-step StepTrace (conf_k, delta_metric, halt_check), halt_reason
- **Full tensors mode**: opt-in for debugging, off by default

See `references/reasoning-trace.md` for schema definition, serialization, and storage policy.

## Configuration Surface

### System1Config

| Field | Default | Purpose |
|---|---|---|
| `input_dim` | 4096 | Workspace representation dimension |
| `hidden_dim` | 512 | MLP hidden width |
| `output_dim` | 256 | Output logits/embedding dimension |
| `num_layers` | 2 | MLP depth |
| `confidence_head` | True | Enable dedicated confidence head |
| `dropout` | 0.1 | Dropout rate |

### System2Config

| Field | Default | Purpose |
|---|---|---|
| `hidden_dim` | 512 | GRU hidden state dimension |
| `max_steps` | 10 | Maximum refinement iterations |
| `convergence_eps` | 1e-3 | KL stability threshold |
| `convergence_patience` | 2 | Consecutive stable steps required |
| `nan_guard` | True | Halt on NaN detection |

### MetacognitionConfig

| Field | Default | Purpose |
|---|---|---|
| `route_threshold` | 0.5 | Score threshold for S2 activation |
| `w_conf` | 1.0 | Confidence weight |
| `w_novelty` | 0.5 | Novelty weight |
| `w_anomaly` | 0.3 | Anomaly weight |
| `w_budget` | 0.1 | Budget penalty weight |
| `min_conf_to_skip_s2` | 0.95 | Hard skip threshold |
| `base_steps` | 3 | Base S2 step allocation |
| `step_scale_alpha` | 5.0 | Route score to steps scaling |
| `always_run_s2` | False | Debug flag to force S2 |

### CalibrationConfig

| Field | Default | Purpose |
|---|---|---|
| `method` | `"temperature"` | `"temperature"`, `"isotonic"`, `"none"` |
| `initial_temperature` | 1.5 | Starting temperature for scaling |
| `freeze_after_fit` | True | Lock calibrator after fitting |

Presets: `DualProcessFullConfig.minimal()`, `.dev()`, `.production_1b()`, `.production_3b()`, `.production_7b()`.

## Done-When Gates

| Gate | Test | Threshold |
|---|---|---|
| **(a) Routing deterministic** | Fixed seed + input + device -> same `used_system2` bitmask across 10 runs | Exact match |
| **(b) S2 halts on convergence** | Synthetic S2 with decreasing deltas; assert halt before max_steps, halt_reason=="converged" | Early halt |
| **(c) Reasoning trace returned** | `return_details=True` -> trace not None, JSON serializable, contains route + per-step entries | Valid JSON |

## Common Failure Modes

| Symptom | Cause | Fix |
|---|---|---|
| Routing flips between runs | Nondeterministic ops in confidence | Use deterministic mode, explicit tie-breaking |
| S2 never converges | Eps too tight or refinement diverges | Loosen eps, add gradient clipping in GRU |
| All items routed to S2 | Uncalibrated confidence always low | Fit temperature scaler on validation set |
| No items reach S2 | Route threshold too high or confidence overfit | Lower threshold, check calibration |
| Trace bloats memory | Full tensors stored by default | Use top-k only, disable full tensor mode |
| S2 latency too high | Processing full batch including confident items | Enable selective execution (scatter/gather) |
| Calibration drifts over time | Online temperature retraining | Freeze calibrator after fitting |
| Novelty score always zero | Prototype bank not updated | Update prototypes during training |

## Anti-Patterns

- **Single-scalar routing** -- route on combined uncertainty score, not just max softmax
- **Nondeterministic routing** -- no sampling, no unstable GPU sorts in the routing path
- **Uncalibrated confidence** -- always apply temperature scaling or isotonic regression
- **Processing full batch in S2** -- use scatter/gather to select only uncertain items
- **Storing full tensors in trace** -- default to top-k logit snapshots only
- **Online calibrator retraining** -- fit once, freeze; online updates cause routing drift
- **Hard-coded confidence threshold** -- use configurable MetacognitionConfig, not magic numbers
- **No convergence detection** -- always check halt criteria; open-ended loops waste compute
- **Cross-batch routing dependencies** -- per-item routing must be independent

## Additional Resources

### Reference Files

- **`references/system1-confidence.md`** -- System 1 architecture, confidence heads, calibration fitting
- **`references/system2-convergence.md`** -- S2 loop structure, convergence criteria, effort budget
- **`references/metacognitive-routing.md`** -- Routing policy, novelty scoring, budget allocation
- **`references/reasoning-trace.md`** -- Trace schema, StepTrace, serialization, storage policy
- **`references/testing-matrix.md`** -- All test cases: routing determinism, convergence, trace validation

### Asset Templates

- **`assets/system1_template.py`** -- System1Fast module with confidence heads, self-test
- **`assets/calibration_template.py`** -- TemperatureScaler, IsotonicCalibrator, fitting utilities, self-test
- **`assets/system2_template.py`** -- System2Iterative, GRU refinement, convergence detection, self-test
- **`assets/metacognition_template.py`** -- MetacognitiveRouter, novelty scoring, budget policy, self-test
- **`assets/dual_process_template.py`** -- DualProcessReasoner, scatter/gather S2, integration, self-test
- **`assets/reasoning_trace_template.py`** -- ReasoningTrace, StepTrace, JSON serialization, self-test
- **`assets/reasoning_config_template.py`** -- All configs, presets, serialization, self-test

### Scripts

- **`scripts/validate_reasoning.py`** -- Runtime contract validation (routing determinism, convergence, trace)
- **`scripts/gen_reasoning_tests.py`** -- Generates `tests/test_dual_process_reasoning.py` (~80+ test cases)
- **`scripts/routing_benchmark.py`** -- Benchmark S1 latency, S2 convergence speed, routing overhead
