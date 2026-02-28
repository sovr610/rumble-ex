# Testing Matrix for Hyperparameter Search Infrastructure

## Overview

This document defines the comprehensive test scenarios for the hyperparameter search infrastructure. Every component -- search engine, search space, LR finder, trial manager, early stopping, and configuration -- must be tested for correctness, edge cases, and integration. Tests should use toy objective functions and simple models (no GPU required) to run in seconds.

---

## 1. Search Engine Tests

### 1.1 Basic Functionality

| Test ID | Scenario | Setup | Expected Result |
|---------|----------|-------|----------------|
| SE-01 | Random search converges on quadratic | `f(x) = (x-3)^2`, 50 trials | Best x within 0.5 of 3.0 |
| SE-02 | Grid search covers all points | 3 params, 3 values each (27 combos) | Exactly 27 trials evaluated |
| SE-03 | Bayesian (TPE) outperforms random | Branin function, 50 trials each | TPE best < Random best |
| SE-04 | SearchEngine.best_params() returns dict | Any search | Returns Dict[str, Any] with all param names |
| SE-05 | SearchEngine.best_value() returns float | Any search | Returns float, not NaN |
| SE-06 | Search with 0 trials | n_trials=0 | Returns empty results or raises ValueError |
| SE-07 | Search with 1 trial | n_trials=1 | Returns single result |

### 1.2 Search Quality

| Test ID | Scenario | Setup | Expected Result |
|---------|----------|-------|----------------|
| SE-08 | Quadratic 1D optimization | `f(x) = (x-2)^2`, x in [-5, 5] | Best x within 1.0 of 2.0 in 30 trials |
| SE-09 | Quadratic 2D optimization | `f(x,y) = (x-1)^2 + (y+2)^2` | Best within 1.0 of (1, -2) in 50 trials |
| SE-10 | Rosenbrock 2D | `f(x,y) = (1-x)^2 + 100(y-x^2)^2` | Best value < 10.0 in 100 trials |
| SE-11 | Categorical parameter | Minimize `f(c) = {a:1, b:0, c:2}[c]` | Best is c='b' |
| SE-12 | Mixed parameter types | float + int + categorical | All types handled correctly |
| SE-13 | Log-scale parameter | LR in [1e-5, 1e-1], log scale | Samples distributed across orders of magnitude |
| SE-14 | Maximize direction | `f(x) = -(x-3)^2`, direction=maximize | Best x near 3.0 |

### 1.3 Robustness

| Test ID | Scenario | Setup | Expected Result |
|---------|----------|-------|----------------|
| SE-15 | Objective returns NaN | 20% of trials return NaN | Search completes; NaN trials excluded from best |
| SE-16 | Objective returns inf | Some trials return inf | Search completes; inf treated as worst |
| SE-17 | Objective raises exception | Some trials raise RuntimeError | Exception caught; trial marked as failed |
| SE-18 | Constant objective | `f(x) = 5.0` always | Search completes; does not crash |
| SE-19 | Very noisy objective | `f(x) = (x-3)^2 + N(0, 10)` | Best x within 2.0 of 3.0 in 100 trials |

### 1.4 Resume and State

| Test ID | Scenario | Setup | Expected Result |
|---------|----------|-------|----------------|
| SE-20 | Resume from saved state | Save after 25 trials, resume for 25 more | Total 50 trials; results consistent |
| SE-21 | Reproducibility with seed | Same seed, same space | Identical trial sequence (random strategy) |
| SE-22 | Different seeds differ | Different seeds, same space | Different trial sequences |

---

## 2. Search Space Tests

### 2.1 Parameter Types

| Test ID | Scenario | Setup | Expected Result |
|---------|----------|-------|----------------|
| SP-01 | Float parameter | `add_float("x", 0.0, 1.0)` | Samples in [0.0, 1.0] |
| SP-02 | Float log-scale | `add_float("lr", 1e-5, 1e-1, log=True)` | Samples distributed on log scale |
| SP-03 | Integer parameter | `add_int("n", 1, 10)` | Samples are integers in [1, 10] |
| SP-04 | Categorical parameter | `add_categorical("opt", ["adam", "sgd"])` | Samples are "adam" or "sgd" |
| SP-05 | Boolean (via categorical) | `add_categorical("flag", [True, False])` | Samples are True or False |

### 2.2 Sampling

| Test ID | Scenario | Setup | Expected Result |
|---------|----------|-------|----------------|
| SP-06 | Sample returns complete dict | Space with 5 params | Dict has all 5 keys |
| SP-07 | 1000 float samples in bounds | Sample 1000 times | All values in [low, high] |
| SP-08 | 1000 int samples in bounds | Sample 1000 times | All values are ints in [low, high] |
| SP-09 | 1000 categorical samples valid | Sample 1000 times | All values in choices list |
| SP-10 | Log-scale distribution | 10000 samples, check log-median | Median close to geometric mean of bounds |
| SP-11 | Uniform distribution | 10000 samples, check mean | Mean close to arithmetic mean of bounds |

### 2.3 Conditional Parameters

| Test ID | Scenario | Setup | Expected Result |
|---------|----------|-------|----------------|
| SP-12 | Conditional active | `use_htm=True`, tune `column_count` | `column_count` present in sample |
| SP-13 | Conditional inactive | `use_htm=False` | `column_count` absent from sample |
| SP-14 | Nested conditionals | `use_htm=True` AND `use_reflex=True` | Reflex params present |
| SP-15 | Multiple conditions | Several conditional groups | Only active groups present |

### 2.4 Config Integration

| Test ID | Scenario | Setup | Expected Result |
|---------|----------|-------|----------------|
| SP-16 | from_config_class basic | `from_config_class(SNNConfig)` | Space has all SNNConfig fields |
| SP-17 | from_config_class with overrides | Override `beta` range to [0.9, 0.99] | Override applied; other fields use defaults |
| SP-18 | from_config_class ignores non-tunable | list/tuple fields skipped | No list parameters in space |
| SP-19 | Empty space | No parameters added | `sample()` returns empty dict |
| SP-20 | Large space | 50 parameters | `sample()` returns dict with 50 keys |

---

## 3. LR Finder Tests

### 3.1 Core Algorithm

| Test ID | Scenario | Setup | Expected Result |
|---------|----------|-------|----------------|
| LR-01 | Basic LR sweep | Linear model, MSE loss, synthetic data | Returns list of (lr, loss) pairs |
| LR-02 | Loss decreases then increases | Standard U-shaped curve | Curve has a minimum region |
| LR-03 | suggested_lr returns valid | After find() | Returns (min_lr, max_lr), both > 0, min < max |
| LR-04 | suggested_lr not NaN | After find() | Neither value is NaN |
| LR-05 | suggested_lr not at extremes | start_lr=1e-7, end_lr=10 | min_lr > 1e-7, max_lr < 10 |
| LR-06 | Model state restored | Check params before/after find() | Parameters identical |
| LR-07 | Optimizer state restored | Check optimizer state before/after | State identical |

### 3.2 Divergence Detection

| Test ID | Scenario | Setup | Expected Result |
|---------|----------|-------|----------------|
| LR-08 | Stops on divergence | High end_lr, simple model | Stops before end_lr |
| LR-09 | Stops on NaN loss | Model that produces NaN at high LR | Stops; no NaN in results |
| LR-10 | Custom divergence threshold | threshold=2.0 vs threshold=10.0 | Lower threshold stops earlier |

### 3.3 Smoothing

| Test ID | Scenario | Setup | Expected Result |
|---------|----------|-------|----------------|
| LR-11 | Smoothed loss is smoother | Compare raw vs smoothed | Smoothed has lower variance |
| LR-12 | Smooth factor effect | factor=0.0 vs factor=0.5 | Higher factor = smoother curve |
| LR-13 | No smoothing (factor=0) | factor=0 | Smoothed == raw |

### 3.4 Edge Cases

| Test ID | Scenario | Setup | Expected Result |
|---------|----------|-------|----------------|
| LR-14 | Very small model (1 param) | Single linear neuron | Find completes; reasonable suggestion |
| LR-15 | Already converged model | Pre-trained model | Still finds valid LR range |
| LR-16 | num_steps > data batches | 500 steps, 100 batches | Wraps data; completes all steps |
| LR-17 | Single batch | 1 batch of data | At least 1 step; no crash |
| LR-18 | Batch size 1 | Very noisy gradients | Completes; heavier smoothing needed |

### 3.5 Plotting

| Test ID | Scenario | Setup | Expected Result |
|---------|----------|-------|----------------|
| LR-19 | Plot returns figure | After find() | Returns matplotlib Figure (or data if no mpl) |
| LR-20 | Plot shows suggested LR | Inspect plot | Vertical lines at suggested values |

---

## 4. Trial Manager Tests

### 4.1 Logging and Retrieval

| Test ID | Scenario | Setup | Expected Result |
|---------|----------|-------|----------------|
| TM-01 | Log single trial | log_trial with params and metrics | get_trial returns same data |
| TM-02 | Log multiple trials | Log 50 trials | All 50 retrievable |
| TM-03 | Trial IDs unique | Log 100 trials with auto-IDs | All IDs unique |
| TM-04 | Custom trial ID | log_trial(trial_id="custom_01") | get_trial("custom_01") works |
| TM-05 | Overwrite trial | Log trial_id="x" twice | Most recent data kept |

### 4.2 Best Trial Selection

| Test ID | Scenario | Setup | Expected Result |
|---------|----------|-------|----------------|
| TM-06 | get_best_trials(n=1) | 50 trials, minimize | Returns trial with lowest metric |
| TM-07 | get_best_trials(n=5) | 50 trials, minimize | Returns 5 trials, sorted best-first |
| TM-08 | get_best_trials maximize | 50 trials, maximize | Returns highest metric first |
| TM-09 | get_best_trials(n=100) | Only 50 trials | Returns all 50 (no error) |
| TM-10 | Tied metric values | Multiple trials with same metric | Returns all tied trials (deterministic order) |

### 4.3 Export

| Test ID | Scenario | Setup | Expected Result |
|---------|----------|-------|----------------|
| TM-11 | export_csv basic | 10 trials with params + metrics | Valid CSV file, 10 data rows |
| TM-12 | CSV has all columns | Params: {a, b}, metrics: {loss, acc} | Columns: trial_id, a, b, loss, acc |
| TM-13 | CSV round-trip | Export then read back | Values match original |
| TM-14 | Export empty | 0 trials | CSV with header only (or empty file) |
| TM-15 | Export with special chars | Param values with commas, quotes | Properly escaped in CSV |

### 4.4 Parameter Importance

| Test ID | Scenario | Setup | Expected Result |
|---------|----------|-------|----------------|
| TM-16 | Importance with 1 correlated param | `y = x1 + noise`, x2 random | x1 importance >> x2 importance |
| TM-17 | Importance sums to ~1 | Multiple params | Sum of importances in [0.8, 1.2] |
| TM-18 | Importance with all identical params | Constant params | All importances near 0 or equal |
| TM-19 | Importance with 50 trials | Enough data for significance | Non-trivial importance scores |
| TM-20 | Importance with 5 trials | Very few data points | Returns something (may be unreliable) |

### 4.5 Persistence

| Test ID | Scenario | Setup | Expected Result |
|---------|----------|-------|----------------|
| TM-21 | Persist and reload | Save manager state, create new instance | Previous trials preserved |
| TM-22 | Concurrent writes | Two managers same directory | No data corruption |
| TM-23 | Large trial count | Log 1000 trials | All retrievable; no performance issue |

---

## 5. Early Stopping / Pruner Tests

### 5.1 ASHA (Asynchronous Successive Halving)

| Test ID | Scenario | Setup | Expected Result |
|---------|----------|-------|----------------|
| ES-01 | Bottom trials pruned | 9 trials at rung 1, eta=3 | 6 pruned, 3 promoted |
| ES-02 | Best trial never pruned | Best at every rung | Never pruned |
| ES-03 | Min resource respected | min_resource=3 | No pruning before step 3 |
| ES-04 | Reduction factor effect | eta=2 vs eta=3 | eta=2 keeps more trials |
| ES-05 | Multiple rungs | 27 trials, eta=3, 3 rungs | 9 survive rung 1, 3 survive rung 2, 1 survives rung 3 |

### 5.2 Hyperband

| Test ID | Scenario | Setup | Expected Result |
|---------|----------|-------|----------------|
| ES-06 | Multiple brackets created | max_resource=81, eta=3 | 5 brackets (s=0..4) |
| ES-07 | Bracket 0 has most resources | First bracket | Starts at max_resource |
| ES-08 | Bracket s_max has most trials | Last bracket | Has most initial trials |
| ES-09 | Total budget bounded | All brackets | Total trials * resources <= budget |

### 5.3 Median Pruner

| Test ID | Scenario | Setup | Expected Result |
|---------|----------|-------|----------------|
| ES-10 | Below-median pruned | Trial worse than median at step k | should_prune returns True |
| ES-11 | Above-median kept | Trial better than median at step k | should_prune returns False |
| ES-12 | Not enough data | Fewer than n_startup_trials | Never prunes |
| ES-13 | No data at step | No other trials at step k | Does not prune |

### 5.4 General Pruner Tests

| Test ID | Scenario | Setup | Expected Result |
|---------|----------|-------|----------------|
| ES-14 | should_prune interface | Any pruner | Returns bool |
| ES-15 | Pruner with single trial | Only 1 trial | Never prunes (nothing to compare) |
| ES-16 | Report intermediate value | Report loss at step k | Value stored for later comparison |
| ES-17 | Monotonic improvement not pruned | Loss decreasing every step | Never pruned |
| ES-18 | Non-monotonic reasonable not pruned | Small fluctuations | Not pruned (within tolerance) |

---

## 6. Search Config Tests

### 6.1 Validation

| Test ID | Scenario | Setup | Expected Result |
|---------|----------|-------|----------------|
| SC-01 | Valid config | Default SearchConfig | Validates successfully |
| SC-02 | Invalid strategy | strategy="invalid" | Raises ValueError |
| SC-03 | Invalid direction | direction="sideways" | Raises ValueError |
| SC-04 | n_trials <= 0 | n_trials=-1 | Raises ValueError |
| SC-05 | Invalid sampler | sampler="magic" | Raises ValueError |
| SC-06 | Invalid pruner | pruner="none_such" | Raises ValueError |
| SC-07 | Negative min_resource | min_resource=-1 | Raises ValueError |
| SC-08 | reduction_factor < 2 | reduction_factor=1 | Raises ValueError |

### 6.2 LRFinderConfig Validation

| Test ID | Scenario | Setup | Expected Result |
|---------|----------|-------|----------------|
| SC-09 | Valid LR config | Default LRFinderConfig | Validates successfully |
| SC-10 | start_lr >= end_lr | start_lr=1.0, end_lr=0.1 | Raises ValueError |
| SC-11 | start_lr <= 0 | start_lr=0 | Raises ValueError |
| SC-12 | num_steps <= 0 | num_steps=0 | Raises ValueError |
| SC-13 | smooth_factor out of range | smooth_factor=2.0 | Raises ValueError |
| SC-14 | divergence_threshold <= 1 | divergence_threshold=0.5 | Raises ValueError |

### 6.3 Phase Presets

| Test ID | Scenario | Setup | Expected Result |
|---------|----------|-------|----------------|
| SC-15 | Phase 1 preset | Phase 1 SNN search space | Contains lr, beta, surrogate |
| SC-16 | Phase 3 preset | Phase 3 HTM search space | Contains column_count, sparsity |
| SC-17 | Phase 7 preset | Phase 7 Meta search space | Contains inner_lr, outer_lr |
| SC-18 | All presets valid | Each phase preset | All validate successfully |
| SC-19 | Preset search spaces sampleable | Each preset's space | sample() returns valid dict |

---

## 7. Integration Tests

### 7.1 End-to-End Pipeline

| Test ID | Scenario | Setup | Expected Result |
|---------|----------|-------|----------------|
| INT-01 | Full search pipeline | SearchEngine + SearchSpace + TrialManager on quadratic | TrialManager has all trials; best improves over first |
| INT-02 | LR finder feeds search | LR finder -> narrow LR range -> search | Search finds better LR than wide-range search |
| INT-03 | ASHA with search engine | SearchEngine with ASHA pruner | Pruned trials marked; survivors complete |
| INT-04 | Resume after crash | Save state mid-search; restart | Completes remaining trials; results consistent |
| INT-05 | Export and analysis | Full search -> export CSV -> importance | All data round-trips correctly |

### 7.2 Done-When Gate Validation

| Test ID | Scenario | Expected Result |
|---------|----------|----------------|
| DW-01 | Gate 1: Search Execution | 10 random trials on minimal config; best_params improves over first trial |
| DW-02 | Gate 2: LR Finder | Smooth loss-vs-LR curve; suggested_lr returns valid, non-extreme bounds |
| DW-03 | Gate 3: Trial Management | CSV export valid; importance scores non-trivial for >1 parameter |

---

## 8. Performance Benchmarks

These are not pass/fail tests but benchmarks to track search efficiency:

| Benchmark | Setup | Metric | Target |
|-----------|-------|--------|--------|
| BM-01 | Random search, Branin 2D, 100 trials | Best value | < 1.0 (global min ~0.398) |
| BM-02 | TPE, Branin 2D, 50 trials | Best value | < 0.5 |
| BM-03 | Random search, Rosenbrock 2D, 200 trials | Best value | < 5.0 |
| BM-04 | TPE, Rosenbrock 2D, 100 trials | Best value | < 2.0 |
| BM-05 | ASHA efficiency | 100 trials, 3 rungs | >60% trials pruned at rung 1 |
| BM-06 | Trial logging throughput | Log 10000 trials | < 5 seconds |
| BM-07 | LR finder speed | Simple model, 100 steps | < 2 seconds (CPU) |
| BM-08 | Search space sampling | Sample 100000 points | < 1 second |

---

## 9. Test Implementation Guidelines

### Toy Objective Functions

All search engine tests should use these standard test functions:

**Quadratic (simplest):**
```python
def quadratic(params):
    return sum((v - target)**2 for v, target in zip(params.values(), targets))
```

**Branin (2D, 3 global minima):**
```python
def branin(params):
    x1, x2 = params['x1'], params['x2']
    a, b, c, r, s, t = 1, 5.1/(4*pi^2), 5/pi, 6, 10, 1/(8*pi)
    return a*(x2 - b*x1^2 + c*x1 - r)^2 + s*(1-t)*cos(x1) + s
```

**Rosenbrock (2D, narrow valley):**
```python
def rosenbrock(params):
    x, y = params['x'], params['y']
    return (1-x)**2 + 100*(y-x**2)**2
```

### Simple Models for LR Finder

```python
# Minimal model for LR finder tests
model = nn.Linear(10, 1)
data = [(torch.randn(32, 10), torch.randn(32, 1)) for _ in range(100)]
criterion = nn.MSELoss()
```

### Test Organization

- Group tests by component (search engine, search space, etc.).
- Each test should be independent (no shared state between tests).
- Each test should complete in < 5 seconds.
- Use fixed random seeds for reproducibility.
- Mark slow tests (>1 second) with a "slow" marker.
- Integration tests are separate from unit tests.
