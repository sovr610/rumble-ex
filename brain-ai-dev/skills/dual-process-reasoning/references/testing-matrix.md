# Testing Matrix: Dual-Process Reasoning

Complete test inventory for the dual-process reasoning pipeline (Skill #9). Covers
System 1 fast predictor, confidence calibration, System 2 iterative refinement,
metacognitive routing, reasoning traces, novelty scoring, and end-to-end integration.

70 test methods across 10 test classes. Generated test file target:
`tests/test_dual_process_reasoning.py`.

Run the full suite:

```bash
pytest tests/test_dual_process_reasoning.py -v
```

Run a single class:

```bash
pytest tests/test_dual_process_reasoning.py -v -k "TestSystem1Fast"
```

---

## Table of Contents

1. [Fixtures and Factories](#1-fixtures-and-factories)
2. [TestSystem1Fast](#2-testsystem1fast)
3. [TestCalibration](#3-testcalibration)
4. [TestSystem2Iterative](#4-testsystem2iterative)
5. [TestMetacognitiveRouter](#5-testmetacognitiverouter)
6. [TestRoutingDeterminism](#6-testroutingdeterminism)
7. [TestS2Convergence](#7-tests2convergence)
8. [TestReasoningTrace](#8-testreasontingtrace)
9. [TestDualProcessReasoner](#9-testdualprocessreasoner)
10. [TestNoveltyScorer](#10-testnoveltyscorer)
11. [TestIntegration](#11-testintegration)
12. [Done-When Gate Summary](#12-done-when-gate-summary)
13. [Skip Conditions](#13-skip-conditions)
14. [Assertion Cheat Sheet](#14-assertion-cheat-sheet)
15. [Coverage Targets](#15-coverage-targets)
16. [Failure Triage Guide](#16-failure-triage-guide)

---

## 1. Fixtures and Factories

All fixtures are session- or function-scoped pytest fixtures defined at the top of the
test file. Factories produce lightweight configurations and synthetic tensors for
isolated, fast-running unit tests.

### Fixtures Table

| Fixture Name | Scope | Returns | Purpose |
|---|---|---|---|
| `device` | session | `torch.device` | Return `"cuda"` if available, else `"cpu"`. Shared across all tests. |
| `s1_config` | function | `System1Config` | Minimal System1Config: `input_dim=64, hidden_dim=32, output_dim=10, num_layers=2, dropout=0.0`. |
| `s2_config` | function | `System2Config` | Minimal System2Config: `hidden_dim=32, max_steps=6, convergence_eps=1e-3, convergence_patience=2, nan_guard=True`. |
| `meta_config` | function | `MetacognitionConfig` | Default MetacognitionConfig: `route_threshold=0.5, w_conf=1.0, w_novelty=0.5, w_anomaly=0.3, min_conf_to_skip_s2=0.95, base_steps=3, step_scale_alpha=5.0`. |
| `cal_config` | function | `CalibrationConfig` | Default CalibrationConfig: `method="temperature", initial_temperature=1.5, freeze_after_fit=True`. |
| `full_config` | function | `DualProcessFullConfig` | Aggregated config built from `s1_config + s2_config + meta_config + cal_config` using `DualProcessFullConfig.minimal()`. |
| `system1` | function | `System1Fast` | Instantiated System1Fast from `s1_config`, moved to `device`, set to `model.eval()`. |
| `system2` | function | `System2Iterative` | Instantiated System2Iterative from `s2_config`, moved to `device`, set to `model.eval()`. |
| `router` | function | `MetacognitiveRouter` | Instantiated MetacognitiveRouter from `meta_config`, moved to `device`, set to `model.eval()`. |
| `reasoner` | function | `DualProcessReasoner` | Full DualProcessReasoner wiring `system1 + system2 + router`, moved to `device`. |
| `rand_2d` | function | `Callable[[int], Tensor]` | Factory: `lambda B: torch.randn(B, 64, device=device)`. Produce `(B, input_dim)` workspace vectors. |
| `rand_3d` | function | `Callable[[int], Tensor]` | Factory: `lambda B: torch.randn(B, 4, 64, device=device)`. Produce `(B, K, input_dim)` slot tensors. |
| `synth_logits` | function | `Callable[[int, int], Tensor]` | Factory: `lambda B, C: torch.randn(B, C)`. Produce synthetic logit tensors for calibration tests. |
| `synth_labels` | function | `Callable[[int, int], Tensor]` | Factory: `lambda B, C: torch.randint(0, C, (B,))`. Produce synthetic label tensors for calibration tests. |
| `seeded` | function | context manager | Set `torch.manual_seed(42)` and `torch.use_deterministic_algorithms(True)` inside a context, restore on exit. |

### Config Factory Functions

```python
def make_s1_config(**overrides) -> System1Config:
    defaults = dict(input_dim=64, hidden_dim=32, output_dim=10,
                    num_layers=2, confidence_head=True, dropout=0.0)
    defaults.update(overrides)
    return System1Config(**defaults)

def make_s2_config(**overrides) -> System2Config:
    defaults = dict(hidden_dim=32, max_steps=6, convergence_eps=1e-3,
                    convergence_patience=2, nan_guard=True,
                    convergence_criterion="kl_stability")
    defaults.update(overrides)
    return System2Config(**defaults)

def make_meta_config(**overrides) -> MetacognitionConfig:
    defaults = dict(route_threshold=0.5, w_conf=1.0, w_novelty=0.5,
                    w_anomaly=0.3, w_budget=0.1, min_conf_to_skip_s2=0.95,
                    base_steps=3, step_scale_alpha=5.0, always_run_s2=False)
    defaults.update(overrides)
    return MetacognitionConfig(**defaults)
```

### Synthetic Data Generators

```python
def make_confident_logits(B, C=10, peak_class=0, peak_value=10.0):
    """Produce logits where peak_class dominates (high confidence)."""
    logits = torch.zeros(B, C)
    logits[:, peak_class] = peak_value
    return logits

def make_uniform_logits(B, C=10):
    """Produce near-uniform logits (low confidence)."""
    return torch.zeros(B, C)

def make_converging_s2_input(B, output_dim=10, steps=6):
    """Produce a y1 and x_summary that drive S2 toward convergence."""
    y1 = torch.randn(B, output_dim) * 0.01
    x_summary = torch.randn(B, 64)
    return y1, x_summary
```

---

## 2. TestSystem1Fast

**Purpose.** Validate the System 1 fast predictor module: output shapes for both 2D and
3D inputs, confidence metric computation, dropout behavior, deterministic reproducibility,
and gradient flow through the prediction MLP.

8 test methods.

### Test Methods

| # | Method | Description |
|---|---|---|
| 1 | `test_forward_shape_2d` | Assert output shapes when input is `(B, D)` pooled workspace vector. |
| 2 | `test_forward_shape_slots` | Assert output shapes when input is `(B, K, D)` slot tensor with mean pooling. |
| 3 | `test_confidence_metrics` | Assert `System1Result` contains `conf_raw`, `entropy`, and `margin` as `(B,)` tensors. |
| 4 | `test_confidence_range` | Assert `conf_raw` in `[1/C, 1.0]`, `entropy` in `[0, log(C)]`, and `margin >= 0`. |
| 5 | `test_confidence_head` | Assert dedicated confidence head produces `conf_learned` in `uncertainty_metrics` with shape `(B,)` and values in `[0, 1]`. |
| 6 | `test_dropout_effect` | Assert that with `dropout > 0`, train-mode outputs differ across two calls, but outputs in inference mode are identical. |
| 7 | `test_deterministic` | Assert identical outputs across two forward passes with the same seed and inference mode. |
| 8 | `test_gradient_flow` | Assert that `loss.backward()` on `y1` populates `.grad` on all `system1` parameters. |

### Key Assertions

```python
# test_forward_shape_2d
result = system1(rand_2d(4))
assert result.y1.shape == (4, 10)
assert result.conf_raw.shape == (4,)
assert result.entropy.shape == (4,)
assert result.margin.shape == (4,)

# test_confidence_range
assert (result.conf_raw >= 1.0 / 10).all()
assert (result.conf_raw <= 1.0).all()
assert (result.entropy >= 0).all()
assert (result.entropy <= math.log(10) + 1e-6).all()
assert (result.margin >= 0).all()

# test_confidence_head
assert "conf_learned" in result.uncertainty_metrics
conf_learned = result.uncertainty_metrics["conf_learned"]
assert conf_learned.shape == (4,)
assert (conf_learned >= 0).all() and (conf_learned <= 1).all()

# test_gradient_flow
result = system1(rand_2d(4))
loss = result.y1.sum()
loss.backward()
for p in system1.parameters():
    assert p.grad is not None, f"No gradient for {p.shape}"
```

---

## 3. TestCalibration

**Purpose.** Validate temperature scaling and isotonic calibration: fitting procedure,
ECE reduction, freeze/unfreeze behavior, serialization round-trip, and the calibrator
factory method.

8 test methods.

### Test Methods

| # | Method | Description |
|---|---|---|
| 1 | `test_temperature_fit` | Fit TemperatureScaler on synthetic logits and assert fitted temperature is a positive finite float. |
| 2 | `test_temperature_ece` | Assert ECE decreases after fitting the TemperatureScaler on overconfident synthetic data. |
| 3 | `test_temperature_freeze` | Assert `log_temperature.requires_grad` is `False` after calling `freeze()`. |
| 4 | `test_temperature_serialization` | Save and reload TemperatureScaler via `state_dict()` and `state_dict_extra()`; assert temperature value and `_fitted` flag round-trip correctly. |
| 5 | `test_isotonic_fit` | Fit IsotonicCalibrator on synthetic `(conf_raw, correct)` pairs and assert `_fitted` is `True`. |
| 6 | `test_isotonic_range` | Assert IsotonicCalibrator output is clamped to `[0, 1]` for inputs spanning `[0, 1]`. |
| 7 | `test_ece_computation` | Compute ECE on a perfectly calibrated synthetic dataset and assert ECE is approximately zero. |
| 8 | `test_calibrator_factory` | Assert `create_calibrator("temperature")` returns `TemperatureScaler`, `create_calibrator("isotonic")` returns `IsotonicCalibrator`, and `create_calibrator("none")` returns `None`. |

### Key Assertions

```python
# test_temperature_fit
scaler = TemperatureScaler(initial_temperature=1.5)
diag = scaler.fit(synth_logits(500, 10), synth_labels(500, 10))
assert diag["temperature"] > 0
assert math.isfinite(diag["temperature"])

# test_temperature_ece
assert diag["ece_after"] <= diag["ece_before"]

# test_temperature_freeze
scaler.freeze()
assert not scaler.log_temperature.requires_grad
assert scaler._fitted is True

# test_temperature_serialization
sd = scaler.state_dict()
extra = scaler.state_dict_extra()
scaler2 = TemperatureScaler()
scaler2.load_state_dict(sd)
assert abs(scaler2.temperature - extra["temperature"]) < 1e-6
assert extra["fitted"] is True

# test_isotonic_range
cal_out = iso_cal.calibrate(torch.linspace(0, 1, 100))
assert (cal_out >= 0).all() and (cal_out <= 1).all()
```

---

## 4. TestSystem2Iterative

**Purpose.** Validate the System 2 iterative refinement loop: output shapes, convergence
early halt, halt reason strings, max-steps termination, NaN guard, effort budget
enforcement, gradient flow through the GRU, and deterministic behavior.

8 test methods.

### Test Methods

| # | Method | Description |
|---|---|---|
| 1 | `test_forward_shape` | Assert `System2Result.y2` has shape `(B, output_dim)`, `steps_used` has shape `(B,)`, and `converged` has shape `(B,)`. |
| 2 | `test_convergence_early_halt` | Feed near-constant input so S2 converges early; assert `steps_used < max_steps` for all items. |
| 3 | `test_convergence_halt_reason` | Assert `halt_reason` is `"converged"` for items that halted before `max_steps`. |
| 4 | `test_max_steps_halt` | Feed random input with tight `convergence_eps=1e-12`; assert all items reach `max_steps` and `halt_reason == "max_steps"`. |
| 5 | `test_nan_guard` | Inject NaN into the GRU output via a forward hook; assert `halt_reason == "nan_guard"` and output reverts to last valid `y`. |
| 6 | `test_budget_enforcement` | Set per-item `steps_budget` tensor `[2, 4, 6]`; assert `steps_used[i] <= steps_budget[i]` for all items. |
| 7 | `test_gradient_flow` | Run S2 in training mode; call `loss.backward()` on `y2.sum()`; assert gradients populate on GRU and summary net parameters. |
| 8 | `test_deterministic` | Run two forward passes with the same seed in inference mode; assert `y2` tensors are bitwise identical. |

### Key Assertions

```python
# test_forward_shape
result = system2(y1=torch.randn(3, 10), x_summary=torch.randn(3, 64),
                 steps_budget=6)
assert result.y2.shape == (3, 10)
assert result.steps_used.shape == (3,)
assert result.converged.shape == (3,)
assert len(result.halt_reason) == 3

# test_convergence_early_halt
y1 = torch.zeros(2, 10)  # near-constant => minimal delta
result = system2(y1=y1, x_summary=torch.zeros(2, 64), steps_budget=6)
assert (result.steps_used < 6).all()

# test_nan_guard
# (after injecting NaN hook)
assert all(r == "nan_guard" for r in result.halt_reason)
assert torch.isfinite(result.y2).all()

# test_budget_enforcement
budget = torch.tensor([2, 4, 6])
result = system2(y1=torch.randn(3, 10), x_summary=torch.randn(3, 64),
                 steps_budget=budget)
assert (result.steps_used <= budget).all()
```

---

## 5. TestMetacognitiveRouter

**Purpose.** Validate the metacognitive routing policy: confident inputs skip S2,
uncertain inputs route to S2, novelty and anomaly triggers, forced S2 mode, per-item
step scaling, deterministic routing, and batch independence.

8 test methods. Gate **(a)** applies to `test_deterministic_routing`.

### Test Methods

| # | Method | Description |
|---|---|---|
| 1 | `test_confident_skip` | Feed high-confidence S1 result (`conf_calibrated > 0.95`); assert `used_system2` is `False` for all items. |
| 2 | `test_uncertain_route` | Feed low-confidence S1 result (`conf_calibrated < 0.3`); assert `used_system2` is `True` for all items. |
| 3 | `test_novelty_trigger` | Set `novelty=0.9` with moderate confidence; assert `used_system2` is `True` due to novelty weight. |
| 4 | `test_anomaly_trigger` | Set `anomaly=0.9` with moderate confidence; assert `used_system2` is `True` due to anomaly weight. |
| 5 | `test_always_s2` | Set `always_run_s2=True` in config; assert `used_system2` is `True` even for high-confidence inputs. |
| 6 | `test_steps_scaling` | Verify allocated `steps_budget` scales with `route_score`: higher route score yields more steps, clamped to `[1, max_steps]`. |
| 7 | `test_deterministic_routing` | **[Gate a]** Run routing 10 times on the same input with the same seed; assert `used_system2` bitmask is identical across all runs. |
| 8 | `test_batch_independence` | Route a batch of 8 items; remove item 4; re-route; assert routing decisions for the remaining 7 items are unchanged. |

### Key Assertions

```python
# test_confident_skip
s1_result = make_s1_result(conf_calibrated=torch.full((4,), 0.99))
decision = router(s1_result)
assert not decision.used_system2.any()

# test_uncertain_route
s1_result = make_s1_result(conf_calibrated=torch.full((4,), 0.1))
decision = router(s1_result)
assert decision.used_system2.all()

# test_deterministic_routing [Gate a]
masks = []
for _ in range(10):
    torch.manual_seed(42)
    decision = router(s1_result)
    masks.append(decision.used_system2.clone())
for m in masks[1:]:
    assert torch.equal(m, masks[0])

# test_batch_independence
full_decision = router(s1_result_8)
partial_result = drop_item(s1_result_8, idx=4)
partial_decision = router(partial_result)
remaining_indices = [i for i in range(8) if i != 4]
for j, orig_idx in enumerate(remaining_indices):
    assert partial_decision.used_system2[j] == full_decision.used_system2[orig_idx]
```

---

## 6. TestRoutingDeterminism

**Purpose.** Exhaustive determinism tests for the routing path. Ensure no
nondeterministic GPU operations, cross-batch stability, and explicit tie-breaking.
All tests in this class map to **Gate (a)**.

6 test methods.

### Test Methods

| # | Method | Description |
|---|---|---|
| 1 | `test_cpu_deterministic` | Run routing 20 times on CPU with fixed seed; assert exact match on `used_system2` and `route_score`. |
| 2 | `test_cuda_deterministic` | Run routing 20 times on CUDA with fixed seed and `torch.use_deterministic_algorithms(True)`; assert exact match. Skip if no CUDA. |
| 3 | `test_cross_batch_stable` | Route the same items as a batch of 8 and as two batches of 4; assert per-item decisions match across batch splits. |
| 4 | `test_seed_sensitivity` | Route with seed 42 and seed 99; assert results may differ (non-trivial randomness in initialization). |
| 5 | `test_tie_breaking` | Construct input where `route_score == threshold` exactly; assert a deterministic tie-break rule applies (default: route to S2 on tie). |
| 6 | `test_no_nondeterministic_ops` | Patch `torch.use_deterministic_algorithms(True)` and run routing; assert no `RuntimeError` about nondeterministic operations. |

### Key Assertions

```python
# test_cpu_deterministic
scores = []
for _ in range(20):
    torch.manual_seed(42)
    decision = router(s1_result)
    scores.append(decision.route_score.clone())
for s in scores[1:]:
    assert torch.equal(s, scores[0])

# test_cuda_deterministic
@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")
def test_cuda_deterministic(self, ...):
    torch.use_deterministic_algorithms(True)
    # ... same pattern as CPU ...
    torch.use_deterministic_algorithms(False)

# test_cross_batch_stable
decision_full = router(s1_result_8)
decision_a = router(s1_result_8[:4])
decision_b = router(s1_result_8[4:])
combined = torch.cat([decision_a.used_system2, decision_b.used_system2])
assert torch.equal(combined, decision_full.used_system2)

# test_tie_breaking
s1_result = make_s1_result_at_threshold(route_threshold=0.5)
decision = router(s1_result)
assert decision.used_system2.all()  # tie breaks toward S2
```

---

## 7. TestS2Convergence

**Purpose.** Focused convergence behavior tests for the System 2 refinement loop.
Validate that synthetic inputs with decreasing deltas trigger convergence,
patience accumulation works correctly, and budget exhaustion overrides patience.
All tests in this class map to **Gate (b)**.

6 test methods.

### Test Methods

| # | Method | Description |
|---|---|---|
| 1 | `test_synthetic_convergence` | **[Gate b]** Feed near-zero input to S2 so deltas shrink quickly; assert `converged=True` and `halt_reason="converged"` and `steps_used < max_steps`. |
| 2 | `test_patience_required` | Set `convergence_patience=3`; feed input that stabilizes for only 2 consecutive steps then diverges; assert loop does not halt at step 2. |
| 3 | `test_patience_met` | Set `convergence_patience=2`; feed input that stabilizes for 2 consecutive steps; assert `halt_reason="converged"`. |
| 4 | `test_no_convergence_max_steps` | Feed highly variable input with `convergence_eps=1e-12`; assert `halt_reason="max_steps"` and `steps_used == max_steps`. |
| 5 | `test_budget_exhausted` | Set per-item `steps_budget=2` with `max_steps=10`; feed input that would converge at step 5; assert `halt_reason="budget_exhausted"` and `steps_used <= 2`. |
| 6 | `test_different_criteria` | Run S2 with `convergence_criterion="logit_stability"` and `"argmax_stability"` on the same input; assert both produce a valid `halt_reason` string from the allowed set. |

### Key Assertions

```python
# test_synthetic_convergence [Gate b]
s2 = System2Iterative(**make_s2_config(max_steps=10, convergence_eps=1e-2,
                                        convergence_patience=2))
y1 = torch.zeros(2, 10)
result = s2(y1=y1, x_summary=torch.zeros(2, 64), steps_budget=10)
assert result.converged.all()
assert all(r == "converged" for r in result.halt_reason)
assert (result.steps_used < 10).all()

# test_patience_required
# (custom input producing 2 stable then 1 unstable step)
assert not result.converged.all()

# test_budget_exhausted
budget = torch.tensor([2, 2])
result = s2(y1=torch.randn(2, 10), x_summary=torch.randn(2, 64),
            steps_budget=budget)
assert (result.steps_used <= 2).all()
assert all(r in ("budget_exhausted", "converged") for r in result.halt_reason)

# test_different_criteria
VALID_HALT_REASONS = {"converged", "max_steps", "budget_exhausted", "nan_guard"}
assert all(r in VALID_HALT_REASONS for r in result.halt_reason)
```

---

## 8. TestReasoningTrace

**Purpose.** Validate the reasoning trace schema: trace creation, absence when
`return_details=False`, JSON serialization, round-trip reconstruction, route field
presence, System 1 top-k storage, System 2 step traces, and trace comparison diff.
All tests in this class map to **Gate (c)**.

8 test methods.

### Test Methods

| # | Method | Description |
|---|---|---|
| 1 | `test_trace_returned` | **[Gate c]** Call `reasoner(x, return_details=True)`; assert returned trace is not `None` and is a `ReasoningTrace` instance. |
| 2 | `test_trace_not_returned` | Call `reasoner(x, return_details=False)`; assert trace is `None`. |
| 3 | `test_trace_json_valid` | Call `trace.to_json()`; parse with `json.loads()`; assert no exception and result is a `dict`. |
| 4 | `test_trace_roundtrip` | Convert trace to dict, reconstruct with `ReasoningTrace.from_dict()`, convert back to dict; assert both dicts are equal. |
| 5 | `test_route_fields` | Assert `trace.route` contains all required fields: `used_system2`, `route_score`, `threshold`, `conf_raw`, `conf_calibrated`, `entropy`, `margin`, `steps_budget`. |
| 6 | `test_s1_top_k` | Assert `trace.system1.top_k_indices` has length equal to `trace.system1.top_k`; assert values are sorted descending; assert no duplicates. |
| 7 | `test_s2_steps` | Route a low-confidence input to S2; assert `trace.system2` is not `None`; assert `len(trace.system2.steps) == trace.system2.steps_used`; assert each step has `conf_k`, `delta_kl`, `argmax_k`. |
| 8 | `test_trace_diff` | Create two traces from the same input with the same seed; call `compare_traces()`; assert `TraceDiff.same_route` is `True` and `max_logit_delta == 0.0`. |

### Key Assertions

```python
# test_trace_returned [Gate c]
output, trace = reasoner(rand_2d(4), return_details=True)
assert trace is not None
assert isinstance(trace, list)
assert all(isinstance(t, ReasoningTrace) for t in trace)

# test_trace_json_valid
json_str = trace[0].to_json()
parsed = json.loads(json_str)
assert isinstance(parsed, dict)
assert "route" in parsed
assert "system1" in parsed

# test_trace_roundtrip
d1 = trace[0].to_dict()
reconstructed = ReasoningTrace.from_dict(d1)
d2 = reconstructed.to_dict()
assert d1 == d2

# test_route_fields
route = trace[0].route
REQUIRED = {"used_system2", "route_score", "threshold",
            "conf_raw", "conf_calibrated", "entropy", "margin",
            "steps_budget"}
assert REQUIRED.issubset(set(route.to_dict().keys()))

# test_s1_top_k
s1 = trace[0].system1
assert len(s1.top_k_indices) == s1.top_k
assert s1.top_k_values == sorted(s1.top_k_values, reverse=True)
assert len(set(s1.top_k_indices)) == len(s1.top_k_indices)

# test_s2_steps
t = trace_with_s2[0]
assert t.system2 is not None
assert len(t.system2.steps) == t.system2.steps_used
for step in t.system2.steps:
    assert hasattr(step, "conf_k")
    assert hasattr(step, "delta_kl")
    assert hasattr(step, "argmax_k")

# test_trace_diff
diff = compare_traces(trace_a[0], trace_b[0], logit_atol=0.0)
assert diff.same_route is True
assert diff.same_prediction is True
assert diff.max_logit_delta == 0.0
```

---

## 9. TestDualProcessReasoner

**Purpose.** End-to-end tests for the `DualProcessReasoner` module that wires
System 1, System 2, and the metacognitive router together. Validate the full forward
pass, S1-only paths, S2 paths, selective execution, output fields, state management,
determinism, and the overhead of `return_details`.

8 test methods.

### Test Methods

| # | Method | Description |
|---|---|---|
| 1 | `test_full_forward_shape` | Assert `ReasoningOutput.y` has shape `(B, output_dim)` and `used_system2` has shape `(B,)`. |
| 2 | `test_s1_only_path` | Feed high-confidence input; assert `used_system2.all() == False` and `s2` field is `None`. |
| 3 | `test_s2_path` | Feed low-confidence input; assert `used_system2.any() == True` and `s2.y2` is a valid tensor. |
| 4 | `test_selective_execution` | Feed a mixed batch (half confident, half uncertain); assert confident items retain S1 output exactly and uncertain items have modified output. |
| 5 | `test_reasoning_output_fields` | Assert `ReasoningOutput` has all required fields: `y`, `used_system2`, `s1`, `s2`, `trace`, `aux`. |
| 6 | `test_state_management` | Pass `state` dict into forward; assert returned state dict contains recurrent state keys from S2. |
| 7 | `test_deterministic_full` | Run two forward passes with the same seed; assert `y` tensors are bitwise identical. |
| 8 | `test_return_details_overhead` | Time two forward passes (one with `return_details=False`, one with `True`); assert the overhead is less than 50% of the base forward time. |

### Key Assertions

```python
# test_full_forward_shape
output = reasoner(rand_2d(8))
assert output.y.shape == (8, 10)
assert output.used_system2.shape == (8,)
assert output.used_system2.dtype == torch.bool

# test_s1_only_path
output = reasoner(make_confident_input(8))
assert not output.used_system2.any()
assert output.s2 is None

# test_selective_execution
x = torch.cat([make_confident_input(4), make_uncertain_input(4)])
output = reasoner(x)
s1_items = ~output.used_system2
s2_items = output.used_system2
assert s1_items[:4].all()   # first 4 confident
assert s2_items[4:].all()   # last 4 uncertain

# test_reasoning_output_fields
REQUIRED = {"y", "used_system2", "s1", "s2", "trace", "aux"}
assert REQUIRED.issubset(set(vars(output).keys()))

# test_deterministic_full
torch.manual_seed(42)
out1 = reasoner(x)
torch.manual_seed(42)
out2 = reasoner(x)
assert torch.equal(out1.y, out2.y)
assert torch.equal(out1.used_system2, out2.used_system2)
```

---

## 10. TestNoveltyScorer

**Purpose.** Validate the novelty scorer component: high novelty for out-of-distribution
prototypes, low novelty for in-distribution inputs, EMA prototype bank updates,
output range enforcement, and empty bank edge case.

5 test methods.

### Test Methods

| # | Method | Description |
|---|---|---|
| 1 | `test_prototype_high` | Register prototypes from class A; feed class B inputs; assert novelty score > 0.7. |
| 2 | `test_prototype_low` | Register prototypes from class A; feed class A inputs; assert novelty score < 0.3. |
| 3 | `test_ema_update` | Call `update_prototypes()` twice with different data; assert prototype bank values shift toward the new data. |
| 4 | `test_novelty_range` | Feed 100 random inputs; assert all novelty scores are in `[0, 1]`. |
| 5 | `test_empty_bank` | Create a NoveltyScorer with an empty prototype bank; assert it returns a default novelty of 0.5 (maximum uncertainty) without error. |

### Key Assertions

```python
# test_prototype_high
scorer = NoveltyScorer(dim=64, num_prototypes=10)
scorer.register(in_distribution_data)
novelty = scorer(out_of_distribution_data)
assert (novelty > 0.7).all()

# test_novelty_range
novelty = scorer(torch.randn(100, 64))
assert (novelty >= 0).all() and (novelty <= 1).all()

# test_empty_bank
scorer = NoveltyScorer(dim=64, num_prototypes=0)
novelty = scorer(torch.randn(4, 64))
assert (novelty == 0.5).all()
```

---

## 11. TestIntegration

**Purpose.** Integration tests verifying the dual-process reasoning module works
correctly within the broader brain-ai pipeline context. Test workspace-to-reasoning
data flow, HTM anomaly score propagation, config preset instantiation, checkpoint
save/load, and mixed routing within a single batch.

5 test methods.

### Test Methods

| # | Method | Description |
|---|---|---|
| 1 | `test_workspace_to_reasoning` | Create a mock workspace output `(B, 4096)`; feed into DualProcessReasoner with `input_dim=4096`; assert no shape errors and output is `(B, output_dim)`. |
| 2 | `test_htm_anomaly` | Create a mock HTM anomaly tensor `(B,)` in `[0, 1]`; pass as context to the router; assert anomaly influences routing decisions. |
| 3 | `test_config_presets` | Instantiate `DualProcessFullConfig.minimal()`, `.dev()`, `.production_1b()`; assert each passes validation with zero errors. |
| 4 | `test_checkpoint_roundtrip` | Save full DualProcessReasoner state via `state_dict()`, re-instantiate from config, load state; assert output for the same input matches within `atol=1e-6`. |
| 5 | `test_mixed_routing_batch` | Feed a batch of 16 items with varying confidence levels; assert at least 1 routes to S1 and at least 1 routes to S2; assert final output has no NaN. |

### Key Assertions

```python
# test_workspace_to_reasoning
workspace_out = torch.randn(4, 4096)
reasoner = DualProcessReasoner(DualProcessFullConfig.minimal())
output = reasoner(workspace_out)
assert output.y.shape[0] == 4

# test_config_presets
for preset_fn in [DualProcessFullConfig.minimal,
                  DualProcessFullConfig.dev,
                  DualProcessFullConfig.production_1b]:
    cfg = preset_fn()
    errors = cfg.validate()
    assert errors == [], f"Preset {preset_fn.__name__} failed: {errors}"

# test_checkpoint_roundtrip
sd = reasoner.state_dict()
reasoner2 = DualProcessReasoner(full_config)
reasoner2.load_state_dict(sd)
reasoner.eval(); reasoner2.eval()
torch.manual_seed(42); out1 = reasoner(x)
torch.manual_seed(42); out2 = reasoner2(x)
assert torch.allclose(out1.y, out2.y, atol=1e-6)

# test_mixed_routing_batch
output = reasoner(torch.randn(16, 64))
assert output.used_system2.any()       # at least one S2
assert (~output.used_system2).any()    # at least one S1
assert torch.isfinite(output.y).all()  # no NaN
```

---

## 12. Done-When Gate Summary

Each gate maps to specific test classes. All tests in the mapped classes must pass
for the gate to be considered satisfied.

| Gate | Criterion | Test Classes | Total Tests | Threshold |
|---|---|---|---|---|
| **(a) Routing deterministic** | Fixed seed + input + device produces same `used_system2` bitmask across 10 runs | `TestMetacognitiveRouter.test_deterministic_routing`, `TestRoutingDeterminism` (all 6) | 7 | Exact match |
| **(b) S2 halts on convergence** | Synthetic S2 with decreasing deltas halts before max_steps with `halt_reason=="converged"` | `TestS2Convergence` (all 6) | 6 | Early halt |
| **(c) Trace returned and valid** | `return_details=True` returns non-None trace, JSON-serializable, contains route + per-step entries | `TestReasoningTrace` (all 8) | 8 | Valid JSON |

### Gate Pass Requirements

```
Gate (a): PASS when all 7 tests in {TestMetacognitiveRouter.test_deterministic_routing,
          TestRoutingDeterminism.*} pass with zero failures.

Gate (b): PASS when all 6 tests in TestS2Convergence pass.
          Specifically, test_synthetic_convergence must show steps_used < max_steps
          and halt_reason == "converged".

Gate (c): PASS when all 8 tests in TestReasoningTrace pass.
          Specifically, test_trace_returned must show trace is not None,
          test_trace_json_valid must parse without exception,
          and test_trace_roundtrip must produce equal dicts.
```

---

## 13. Skip Conditions

Certain tests require optional dependencies or hardware. Use `pytest.mark.skipif` to
handle these gracefully.

| Condition | Marker | Tests Affected |
|---|---|---|
| No CUDA device | `@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")` | `TestRoutingDeterminism.test_cuda_deterministic` |
| No scipy installed | `@pytest.mark.skipif(not _HAS_SCIPY, reason="scipy not installed")` | `TestCalibration.test_isotonic_fit`, `TestCalibration.test_isotonic_range` (if using scipy backend) |
| No htm.core installed | `@pytest.mark.skipif(not _HAS_HTM, reason="htm.core not installed")` | `TestIntegration.test_htm_anomaly` (uses HTM anomaly scores) |
| Deterministic mode unavailable | `@pytest.mark.skipif(not _DETERMINISTIC_OK, reason="Deterministic algorithms not supported")` | `TestRoutingDeterminism.test_no_nondeterministic_ops` |

### Skip Detection Pattern

```python
_CUDA_AVAILABLE = torch.cuda.is_available()
_CUDA_REASON = "CUDA not available"

try:
    import scipy
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    import htm.core
    _HAS_HTM = True
except ImportError:
    _HAS_HTM = False

try:
    torch.use_deterministic_algorithms(True)
    torch.use_deterministic_algorithms(False)
    _DETERMINISTIC_OK = True
except RuntimeError:
    _DETERMINISTIC_OK = False
```

---

## 14. Assertion Cheat Sheet

Quick reference for the most common assertion patterns used across test classes.

### Shape Assertions

```python
assert tensor.shape == (B, D)
assert tensor.ndim == 2
assert tensor.size(0) == batch_size
assert tensor.size(-1) == output_dim
```

### Range Assertions

```python
assert (tensor >= lo).all(), f"Min value {tensor.min()} < {lo}"
assert (tensor <= hi).all(), f"Max value {tensor.max()} > {hi}"
assert torch.isfinite(tensor).all(), "Non-finite values detected"
```

### Equality Assertions

```python
assert torch.equal(a, b)                          # bitwise exact
assert torch.allclose(a, b, atol=1e-6, rtol=1e-5) # float tolerance
assert a.tolist() == b.tolist()                    # list comparison
```

### Type Assertions

```python
assert isinstance(obj, ReasoningTrace)
assert tensor.dtype == torch.bool
assert tensor.dtype in (torch.float32, torch.float16)
```

### Collection Assertions

```python
assert len(items) == expected_count
assert set(halt_reasons).issubset(VALID_HALT_REASONS)
assert all(isinstance(s, StepTrace) for s in steps)
```

### Gradient Assertions

```python
loss.backward()
for name, p in model.named_parameters():
    assert p.grad is not None, f"No grad: {name}"
    assert torch.isfinite(p.grad).all(), f"Non-finite grad: {name}"
```

### Determinism Assertions

```python
torch.manual_seed(seed)
out1 = model(x)
torch.manual_seed(seed)
out2 = model(x)
assert torch.equal(out1, out2), "Non-deterministic output"
```

### JSON Assertions

```python
import json
parsed = json.loads(json_str)
assert isinstance(parsed, dict)
assert "route" in parsed
# Round-trip
d1 = trace.to_dict()
d2 = ReasoningTrace.from_dict(d1).to_dict()
assert d1 == d2
```

---

## 15. Coverage Targets

Minimum line and branch coverage targets per source module.

| Source Module | Target Line Coverage | Target Branch Coverage | Critical Paths |
|---|---|---|---|
| `reasoning/system1.py` | 90% | 85% | Forward pass, slot pooling, confidence metrics |
| `reasoning/calibration.py` | 90% | 85% | `fit()`, `calibrate()`, `freeze()`, ECE computation |
| `reasoning/system2.py` | 95% | 90% | Main loop, all 4 halt reasons, convergence check, NaN guard |
| `reasoning/metacognition.py` | 95% | 90% | Route score computation, threshold comparison, budget allocation |
| `reasoning/dual_process.py` | 90% | 85% | Full forward, scatter/gather, S1-only path, S2 path |
| `reasoning/trace.py` | 90% | 85% | `to_dict()`, `from_dict()`, `to_json()`, `compare_traces()` |
| `config.py` (reasoning section) | 85% | 80% | Validation, presets, serialization |

### Aggregate Target

```
Overall line coverage for brain_ai/reasoning/*: >= 90%
Overall branch coverage for brain_ai/reasoning/*: >= 85%
```

### Coverage Commands

```bash
# Full coverage report
pytest tests/test_dual_process_reasoning.py --cov=brain_ai.reasoning \
    --cov-report=term-missing --cov-report=html

# Branch coverage
pytest tests/test_dual_process_reasoning.py --cov=brain_ai.reasoning \
    --cov-branch --cov-report=term-missing

# Coverage for a single module
pytest tests/test_dual_process_reasoning.py -k "TestSystem2" \
    --cov=brain_ai.reasoning.system2 --cov-report=term-missing
```

### Uncovered Paths (Expected)

The following paths are expected to have lower coverage and are acceptable:

- `IsotonicCalibrator._pava()` internal loop edge cases (tested indirectly via fit)
- `System1Fast` CLS token pooling path (rarely used, tested in a single method)
- Deprecated or debug-only code paths (e.g., `full_trace=True` tensor storage)
- Error message formatting in validation functions

---

## 16. Failure Triage Guide

When tests fail, use this guide to diagnose the root cause quickly.

### System 1 Failures

| Failure Pattern | Likely Cause | Fix |
|---|---|---|
| `test_forward_shape_2d` shape mismatch | `output_proj` dimension wrong | Check `System1Config.output_dim` matches test expectation |
| `test_confidence_range` out of bounds | Logit explosion or NaN | Check weight init; add gradient clipping |
| `test_confidence_head` missing key | `confidence_head=False` in config | Set `confidence_head=True` in test fixture |
| `test_gradient_flow` None grad | Detached tensor in forward path | Check for `.detach()` applied too early |
| `test_dropout_effect` same output in train mode | Dropout rate is 0.0 | Set `dropout > 0` in test config override |

### Calibration Failures

| Failure Pattern | Likely Cause | Fix |
|---|---|---|
| `test_temperature_fit` non-finite T | L-BFGS diverged | Check logit scale; reduce `lr`; increase `max_iter` |
| `test_temperature_ece` no improvement | Too few samples or trivial logits | Increase sample count; use non-trivial synthetic logits |
| `test_temperature_serialization` mismatch | Missing `state_dict_extra()` | Implement `state_dict_extra()` returning `temperature` and `fitted` |
| `test_isotonic_fit` not fitted | PAVA implementation bug | Check `_fitted` flag is set after `fit()` |
| `test_calibrator_factory` wrong type | Factory not mapping strings correctly | Check factory function switch statement |

### System 2 Failures

| Failure Pattern | Likely Cause | Fix |
|---|---|---|
| `test_convergence_early_halt` no early halt | `convergence_eps` too tight for test input | Use zero input or loosen eps in test config |
| `test_nan_guard` no NaN detected | Hook not injecting NaN correctly | Check hook registration and tensor modification |
| `test_budget_enforcement` exceeds budget | Budget not clamped or checked per-step | Add `steps_budget.clamp(min=1, max=self.max_steps)` |
| `test_gradient_flow` None grad on GRU | Training mode not set | Call `system2.train()` before forward |
| `test_deterministic` different outputs | Nondeterministic GRU op on GPU | Use `torch.use_deterministic_algorithms(True)` |

### Routing Failures

| Failure Pattern | Likely Cause | Fix |
|---|---|---|
| `test_deterministic_routing` flips | Nondeterministic `topk` or `sort` | Replace with stable sort; use `torch.use_deterministic_algorithms(True)` |
| `test_confident_skip` routes to S2 | Threshold too high or confidence too low | Check `min_conf_to_skip_s2` and synthetic confidence values |
| `test_batch_independence` different decisions | Cross-batch normalization (e.g., batch norm) | Remove batch-dependent operations from routing path |
| `test_tie_breaking` wrong direction | No tie-break rule implemented | Add explicit `>= threshold` (not `> threshold`) |

### Trace Failures

| Failure Pattern | Likely Cause | Fix |
|---|---|---|
| `test_trace_returned` is None | `return_details` not passed through | Thread `return_details` kwarg through all layers |
| `test_trace_json_valid` exception | Tensor not converted to Python scalar | Check `_to_scalar()` and `_to_list()` in trace builder |
| `test_trace_roundtrip` unequal dicts | `from_dict()` drops optional fields | Handle `None` fields in `from_dict()` |
| `test_s1_top_k` wrong length | `trace_top_k` config not respected | Pass `top_k` from config to trace builder |
| `test_trace_diff` non-zero delta | Model not in inference mode | Call `model.eval()` before both runs |

### Integration Failures

| Failure Pattern | Likely Cause | Fix |
|---|---|---|
| `test_workspace_to_reasoning` shape error | `input_dim` mismatch between workspace and S1 | Align `workspace.output_dim` with `System1Config.input_dim` |
| `test_config_presets` validation errors | Preset has invalid field combination | Fix the preset factory method |
| `test_checkpoint_roundtrip` mismatch | Missing key in state dict | Check all submodules are registered as `nn.Module` attributes |
| `test_mixed_routing_batch` all S1 or all S2 | Threshold too extreme for random input | Use a balanced threshold (0.5) and random input |

---

## Appendix: Test Class Summary Table

| # | Test Class | Tests | Gate | Module Under Test | Key Invariant |
|---|---|---|---|---|---|
| 1 | `TestSystem1Fast` | 8 | -- | `system1.py` | Shape, confidence range, gradient flow |
| 2 | `TestCalibration` | 8 | -- | `calibration.py` | ECE reduction, serialization, range |
| 3 | `TestSystem2Iterative` | 8 | -- | `system2.py` | Shape, halt reasons, gradient flow |
| 4 | `TestMetacognitiveRouter` | 8 | (a) | `metacognition.py` | Deterministic routing, batch independence |
| 5 | `TestRoutingDeterminism` | 6 | (a) | `metacognition.py` | Exact match across 20 runs, no nondeterministic ops |
| 6 | `TestS2Convergence` | 6 | (b) | `system2.py` | Early halt, patience, budget override |
| 7 | `TestReasoningTrace` | 8 | (c) | `trace.py` | JSON valid, round-trip, field presence |
| 8 | `TestDualProcessReasoner` | 8 | -- | `dual_process.py` | Full forward, selective execution, determinism |
| 9 | `TestNoveltyScorer` | 5 | -- | `metacognition.py` | Range [0,1], empty bank fallback |
| 10 | `TestIntegration` | 5 | -- | `dual_process.py` + pipeline | Config presets, checkpoint, mixed routing |
| | **Total** | **70** | | | |

---

*End of testing matrix. Target audience: another Claude instance implementing or
verifying the dual-process reasoning test suite for brain-ai-dev.*
