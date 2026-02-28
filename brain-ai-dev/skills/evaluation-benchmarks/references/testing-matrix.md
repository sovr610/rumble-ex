# Testing Matrix

Test scenarios for the benchmark infrastructure: synthetic data tests, edge cases, round-trip verification, and end-to-end benchmark validation.

---

## 1. Overview

The testing matrix ensures that all benchmark components produce correct results under normal conditions and handle edge cases gracefully. Every test uses synthetic or mock data -- no real datasets are required.

Tests are organized into five categories:
1. **Metric Correctness**: Verify formulas against manual calculation.
2. **Edge Cases**: Degenerate inputs that stress boundary conditions.
3. **Round-Trip**: Serialization fidelity for reports.
4. **Benchmark End-to-End**: Full pipeline with mock model.
5. **Cross-Phase Coverage**: All 7 phases produce valid output.

---

## 2. Metric Correctness Tests

### 2.1 Accuracy

| Test ID | Input | Expected | Notes |
|---------|-------|----------|-------|
| ACC-001 | preds=[0,1,2], targets=[0,1,2] | 1.0 | Perfect accuracy |
| ACC-002 | preds=[0,0,0], targets=[1,1,1] | 0.0 | All wrong |
| ACC-003 | preds=[0,1,0], targets=[0,0,0] | 0.6667 | 2/3 correct |
| ACC-004 | preds=[], targets=[] | 0.0 | Empty batch |
| ACC-005 | preds=[0]*1000, targets=[0]*1000 | 1.0 | Large batch, all same |

### 2.2 F1 Score

| Test ID | Input | Expected Macro F1 | Notes |
|---------|-------|-------------------|-------|
| F1-001 | Perfect classification, 3 classes | 1.0 | All TP |
| F1-002 | All wrong, 3 classes | 0.0 | All FP/FN |
| F1-003 | Binary: TP=80, FP=10, FN=20 | precision=0.889, recall=0.8, F1=0.842 | Manual calc |
| F1-004 | One class never predicted | F1 for that class = 0.0 | Zero division |
| F1-005 | One class never in targets | F1 for that class = 0.0 | Zero division |
| F1-006 | 3-class with imbalance: [100, 10, 5] | Weighted != Macro | Compare modes |

### 2.3 AUROC

| Test ID | Input | Expected | Notes |
|---------|-------|----------|-------|
| AUC-001 | Perfect separation (probs match labels) | 1.0 | |
| AUC-002 | Random predictions (uniform probs) | ~0.5 | Within tolerance |
| AUC-003 | Inverse predictions | 0.0 | Worst case |
| AUC-004 | All same probability | 0.5 | No discrimination |
| AUC-005 | Only one class in targets | NaN | Undefined |
| AUC-006 | Binary with known TPR/FPR curve | Manual area | Trapezoidal |

### 2.4 Confusion Matrix

| Test ID | Input | Expected | Notes |
|---------|-------|----------|-------|
| CM-001 | Perfect predictions, 3 classes | Diagonal matrix | |
| CM-002 | All predict class 0 | First column non-zero, rest zero | |
| CM-003 | Systematic shift: pred = (target+1) % C | Off-diagonal pattern | |
| CM-004 | Single sample | 1x1 or CxC with one entry | |
| CM-005 | Row sums equal supports | Always | Invariant check |

### 2.5 NAB Score

| Test ID | Input | Expected | Notes |
|---------|-------|----------|-------|
| NAB-001 | Perfect early detection | 100.0 | Max score |
| NAB-002 | No detections | 0.0 (or negative) | All false negatives |
| NAB-003 | All false positives | Negative score | Penalty only |
| NAB-004 | Late detection (end of window) | Low positive | Reduced reward |
| NAB-005 | Detection right at window start | High positive | Max reward per window |

### 2.6 Few-Shot Metrics

| Test ID | Input | Expected | Notes |
|---------|-------|----------|-------|
| FS-001 | 5-way, random predictions over 100 episodes | ~0.2 (1/5) | Random baseline |
| FS-002 | 5-way, perfect predictions | 1.0, CI=0.0 | Zero variance |
| FS-003 | 2-way with 50/50 accuracy | 0.5, wide CI | Coin flip |

### 2.7 Continual Learning Metrics

| Test ID | Input | Expected | Notes |
|---------|-------|----------|-------|
| CL-001 | No forgetting (R[T,j] == R[j,j] for all j) | BWT=0.0 | |
| CL-002 | Complete forgetting (R[T,j] = 0 for j < T) | BWT=-avg_original | |
| CL-003 | Positive transfer (R[T,j] > R[j,j]) | BWT > 0 | |
| CL-004 | Single task | BWT=NaN, FWT=NaN | Undefined |

---

## 3. Edge Case Tests

### 3.1 Empty and Degenerate Inputs

| Test ID | Scenario | Expected Behavior |
|---------|----------|-------------------|
| EDGE-001 | Empty batch (0 samples) | All metrics return 0.0, warning logged |
| EDGE-002 | Single sample | Metrics computed (accuracy 0 or 1) |
| EDGE-003 | Single class in targets | AUROC=NaN, accuracy=valid |
| EDGE-004 | num_classes=1 | Degenerate, all predictions correct |
| EDGE-005 | num_classes=10000 | Confusion matrix too large, skip or sparse |
| EDGE-006 | Predictions contain NaN | NaN samples skipped, warning |
| EDGE-007 | Targets contain negative indices | Raise ValueError |
| EDGE-008 | Predictions and targets different lengths | Raise ValueError |

### 3.2 Numeric Stability

| Test ID | Scenario | Expected Behavior |
|---------|----------|-------------------|
| NUM-001 | Very small probabilities (1e-20) | No underflow in AUROC |
| NUM-002 | Very large logits (1e6) | Overflow handled gracefully |
| NUM-003 | All probabilities exactly 0.5 | AUROC=0.5, no NaN |
| NUM-004 | int32 overflow in confusion matrix | Use int64 |
| NUM-005 | Float32 precision loss in accumulation | Accumulate in float64 |

### 3.3 Multi-Batch Accumulation

| Test ID | Scenario | Expected Behavior |
|---------|----------|-------------------|
| BATCH-001 | Same data, 1 batch vs 10 batches | Identical metrics |
| BATCH-002 | Reset between computations | Independent results |
| BATCH-003 | Compute without any updates | All metrics 0.0 or NaN |
| BATCH-004 | Very large number of batches (10000) | No memory growth |

---

## 4. Round-Trip Tests

### 4.1 JSON Round-Trip

| Test ID | Content | Verification |
|---------|---------|-------------|
| RT-001 | Simple metrics dict | `load(dump(x)) == x` |
| RT-002 | Nested per_class dict | All class keys preserved |
| RT-003 | Confusion matrix (tensor -> list -> tensor) | Element-wise equality |
| RT-004 | NaN values | Serialized as null, loaded as None |
| RT-005 | Report with all sections populated | Full round-trip |
| RT-006 | Empty report (minimal required fields) | No crash on load |
| RT-007 | Report with unicode characters | Preserved correctly |
| RT-008 | Large report (>1MB) | No truncation |

### 4.2 CSV Round-Trip

| Test ID | Content | Verification |
|---------|---------|-------------|
| CSRT-001 | Metrics summary row | All columns preserved |
| CSRT-002 | Per-class table | Class IDs and metrics correct |
| CSRT-003 | NaN in CSV | Empty string in CSV, NaN on reload |
| CSRT-004 | Confusion matrix flat format | Reconstructs to correct matrix |
| CSRT-005 | Special characters in dataset names | Properly quoted |

### 4.3 Report Validation

| Test ID | Input | Expected |
|---------|-------|----------|
| VAL-001 | Valid complete report | No errors |
| VAL-002 | Missing metadata section | Error: "Missing required section: metadata" |
| VAL-003 | Missing report_id in metadata | Error: "Missing required metadata field: report_id" |
| VAL-004 | Non-numeric metric value | Error: "Metric X has non-numeric value" |
| VAL-005 | Extra unknown sections | No error (forward compatible) |

---

## 5. Benchmark End-to-End Tests

### 5.1 Mock Model

```python
class MockBrainAI(nn.Module):
    """Mock model for benchmark testing."""
    def __init__(self, num_classes=10, output_dim=512):
        super().__init__()
        self.linear = nn.Linear(output_dim, num_classes)

    def forward(self, inputs, **kwargs):
        x = list(inputs.values())[0]
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = x[:, :self.linear.in_features]
        return self.linear(x)
```

### 5.2 End-to-End Scenarios

| Test ID | Scenario | Verification |
|---------|----------|-------------|
| E2E-001 | BenchmarkHarness.run() with mock model + synthetic data | Returns valid BenchmarkResult |
| E2E-002 | BenchmarkHarness.run_suite() with 3 synthetic datasets | Returns 3 results |
| E2E-003 | BenchmarkHarness.compare() with 2 results | Returns ComparisonReport with deltas |
| E2E-004 | FewShotAssessor.assess() with synthetic episodes | Returns FewShotResult with CI |
| E2E-005 | AnomalyAssessor.assess() with synthetic sequences | Returns AnomalyResult with NAB score |
| E2E-006 | ReasoningAssessor.assess() with synthetic logic tasks | Returns ReasoningResult |
| E2E-007 | Full pipeline: run + save + load + compare | Reports match after round-trip |
| E2E-008 | Timing is recorded and positive | duration_seconds > 0 |

---

## 6. Cross-Phase Coverage Tests

Each phase must produce valid metrics end-to-end with synthetic data.

| Phase | Test | Synthetic Data | Expected Metrics |
|-------|------|----------------|-----------------|
| 1 SNN | Classify synthetic images | Random (B, 1, 28, 28) | accuracy, F1 |
| 2 Encoders | Encode + classify | Random per-modality tensors | accuracy per modality |
| 3 HTM | Anomaly detection | Sine + injected anomalies | NAB score, F1 |
| 4 Workspace | Multi-modal classify | Random vision + text | accuracy, FIR |
| 5 Active Inference | Control task | Random state sequences | mean reward |
| 6 Reasoning | Logic task | Synthetic premise/conclusion | exact match |
| 7 Meta-Learning | Few-shot classify | Random N-way K-shot episodes | mean accuracy + CI |

**Time constraint**: Each phase test must complete in <60 seconds on CPU.

### Synthetic Data Generators

```python
def make_classification_data(n_samples, input_shape, num_classes):
    """Generate random classification data."""
    data = torch.randn(n_samples, *input_shape)
    labels = torch.randint(0, num_classes, (n_samples,))
    return data, labels

def make_anomaly_sequences(n_sequences, seq_len, anomaly_rate=0.05):
    """Generate sequences with injected anomalies."""
    ...

def make_few_shot_dataset(n_classes, samples_per_class, feature_dim):
    """Generate few-shot classification dataset."""
    ...

def make_logic_tasks(n_tasks, n_premises=3):
    """Generate simple logic tasks with known answers."""
    ...
```

---

## 7. Test Organization

### Directory Structure

```
tests/
  test_metrics.py           # 40+ tests for MetricsSuite
  test_benchmark_harness.py # 30+ tests for BenchmarkHarness
  test_few_shot.py          # 25+ tests for FewShotAssessor
  test_anomaly.py           # 25+ tests for AnomalyAssessor
  test_reasoning.py         # 20+ tests for ReasoningAssessor
  test_reporting.py         # 30+ tests for report generation/loading
  test_config.py            # 15+ tests for configuration
  test_cross_modality.py    # 15+ tests for cross-modality
```

### Test Naming Convention

```python
def test_<component>_<scenario>_<expected>():
    """Test that <component> <scenario> produces <expected>."""
```

Examples:
```python
def test_accuracy_perfect_returns_one():
def test_f1_empty_batch_returns_zero():
def test_auroc_single_class_returns_nan():
def test_confusion_matrix_row_sums_equal_support():
def test_json_roundtrip_preserves_all_metrics():
def test_benchmark_run_produces_valid_result():
```

### Fixtures

```python
@pytest.fixture
def metrics_suite():
    return MetricsSuite(task_type="classify", num_classes=10)

@pytest.fixture
def mock_model():
    return MockBrainAI(num_classes=10)

@pytest.fixture
def bench_config():
    return BenchConfig(task_type="classify", num_classes=10, batch_size=32)

@pytest.fixture
def synthetic_data():
    return make_classification_data(100, (1, 28, 28), 10)
```

---

## 8. Continuous Integration Notes

- All tests run on CPU only (no GPU required).
- All tests use synthetic data (no downloads required).
- Total test suite should complete in <120 seconds.
- Each test is independent (no shared state, no ordering dependency).
- Random seeds fixed per test for reproducibility.
- Tests should not produce any output files (use tmpdir fixtures).

---

## 9. Coverage Targets

| Component | Line Coverage Target | Branch Coverage Target |
|-----------|---------------------|----------------------|
| MetricsSuite | >95% | >90% |
| BenchmarkHarness | >90% | >85% |
| FewShotAssessor | >90% | >85% |
| AnomalyAssessor | >90% | >85% |
| ReasoningAssessor | >85% | >80% |
| Configuration | >95% | >90% |
| Report generation | >90% | >85% |
| Report loading | >90% | >85% |
| Comparison engine | >90% | >85% |

---

## 10. Regression Test Protocol

When a bug is found in benchmark code:

1. Write a failing test that reproduces the bug.
2. Fix the bug.
3. Verify the test passes.
4. Add the test to the regression suite with a comment referencing the bug.

```python
def test_f1_zero_division_regression_bug123():
    """Regression test for bug #123: F1 crashes when a class has zero support."""
    suite = MetricsSuite("classify", num_classes=3)
    # Only classes 0 and 1 present
    suite.update(torch.tensor([0, 1, 0, 1]), torch.tensor([0, 1, 0, 1]))
    result = suite.compute()
    # Should not crash, F1 for class 2 should be 0.0
    assert result["f1_macro"] >= 0.0
```
