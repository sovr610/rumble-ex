# Reporting Format

Structured report schemas for benchmark results, CSV export format, comparison reports, and delta computation for the brain_ai infrastructure.

---

## 1. JSON Report Schema

All benchmark reports follow a standard JSON schema with the following top-level sections.

### 1.1 Complete Report Structure

```json
{
    "metadata": { ... },
    "per_metric": { ... },
    "per_class": { ... },
    "confusion_matrix": { ... },
    "timing": { ... },
    "comparison_baseline": { ... },
    "raw_data": { ... }
}
```

### 1.2 Metadata Section

```json
{
    "metadata": {
        "report_id": "bench_20260220_143052_a1b2c3",
        "timestamp": "2026-02-20T14:30:52.123456",
        "model_name": "brain_ai_v0.1",
        "model_hash": "sha256:abcdef1234567890",
        "model_params": 7000000000,
        "dataset": "mnist",
        "split": "test",
        "num_samples": 10000,
        "task_type": "classify",
        "num_classes": 10,
        "config": {
            "batch_size": 64,
            "device": "cuda",
            "metrics": ["accuracy", "f1_macro", "auroc"],
            "seed": 42
        },
        "environment": {
            "python_version": "3.11.5",
            "torch_version": "2.1.0",
            "cuda_version": "12.1",
            "gpu_name": "NVIDIA A100-SXM4-80GB",
            "hostname": "train-node-01"
        }
    }
}
```

**Required fields**: `report_id`, `timestamp`, `dataset`, `split`, `num_samples`, `task_type`, `config`.

**Optional fields**: `model_name`, `model_hash`, `model_params`, `environment`.

**Report ID format**: `bench_YYYYMMDD_HHMMSS_<6-char-hex>`

### 1.3 Per-Metric Section

Contains the aggregated metric values.

```json
{
    "per_metric": {
        "accuracy": 0.9532,
        "top_5_accuracy": 0.9987,
        "f1_macro": 0.9518,
        "f1_weighted": 0.9531,
        "f1_micro": 0.9532,
        "auroc": 0.9981,
        "precision_macro": 0.9525,
        "recall_macro": 0.9520,
        "loss": 0.1523
    }
}
```

For few-shot tasks:
```json
{
    "per_metric": {
        "mean_accuracy": 0.6234,
        "ci_95": 0.0089,
        "accuracy_low": 0.6145,
        "accuracy_high": 0.6323,
        "n_episodes": 600
    }
}
```

For anomaly tasks:
```json
{
    "per_metric": {
        "nab_score": 78.5,
        "precision": 0.82,
        "recall": 0.91,
        "f1": 0.86,
        "auroc": 0.94,
        "best_threshold": 0.73,
        "nab_profile": "standard"
    }
}
```

For reasoning tasks:
```json
{
    "per_metric": {
        "exact_match": 0.78,
        "logical_consistency": 0.85,
        "proof_accuracy": 0.72,
        "task_type": "babi"
    }
}
```

For continual learning:
```json
{
    "per_metric": {
        "average_accuracy": 0.82,
        "backward_transfer": -0.05,
        "forward_transfer": 0.12,
        "forgetting": 0.08,
        "num_tasks": 5
    }
}
```

**NaN Handling**: NaN values are serialized as `null` in JSON.

### 1.4 Per-Class Section

Breakdown of metrics for each class.

```json
{
    "per_class": {
        "0": {
            "precision": 0.98,
            "recall": 0.97,
            "f1": 0.975,
            "support": 980,
            "accuracy": 0.97
        },
        "1": {
            "precision": 0.96,
            "recall": 0.99,
            "f1": 0.975,
            "support": 1135,
            "accuracy": 0.99
        }
    }
}
```

**Keys**: String representations of class indices (JSON requires string keys).

**Support**: Number of true samples for each class.

### 1.5 Confusion Matrix Section

```json
{
    "confusion_matrix": {
        "matrix": [[970, 1, 2, 0, 0, 0, 3, 1, 2, 1], [0, 1120, 3, 2, 0, 1, 4, 1, 3, 1]],
        "labels": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "normalization": "none",
        "shape": [10, 10]
    }
}
```

**Normalization options**: `"none"` (raw counts), `"true"` (row-normalized), `"pred"` (column-normalized), `"all"` (total-normalized).

For large class counts (>100), the confusion matrix section is omitted by default to keep report size manageable. Set `save_confusion_matrix=True` in the config to force inclusion.

### 1.6 Timing Section

```json
{
    "timing": {
        "total_seconds": 45.23,
        "per_sample_ms": 4.523,
        "per_batch_ms": 289.5,
        "preprocessing_seconds": 2.1,
        "inference_seconds": 40.3,
        "metrics_seconds": 2.83,
        "peak_memory_mb": 4523.7,
        "batch_size": 64,
        "num_batches": 157
    }
}
```

### 1.7 Comparison Baseline Section

Present only when `baseline_path` is set in the config.

```json
{
    "comparison_baseline": {
        "baseline_report_id": "bench_20260219_100000_xyz789",
        "baseline_dataset": "mnist",
        "deltas": {
            "accuracy": 0.0032,
            "f1_macro": 0.0028,
            "auroc": 0.0015
        },
        "relative_deltas": {
            "accuracy": 0.0034,
            "f1_macro": 0.0029,
            "auroc": 0.0015
        },
        "improved": ["accuracy", "f1_macro", "auroc"],
        "degraded": [],
        "unchanged": []
    }
}
```

**Delta computation**: `delta = current - baseline`

**Relative delta**: `relative_delta = (current - baseline) / |baseline|` (if baseline != 0, else `Inf`).

**Classification**:
- Improved: delta > threshold (default 1e-4).
- Degraded: delta < -threshold.
- Unchanged: |delta| <= threshold.

### 1.8 Raw Data Section (Optional)

For debugging and detailed analysis. Only included when `save_raw_data=True`.

```json
{
    "raw_data": {
        "per_sample_predictions": [3, 7, 2],
        "per_sample_targets": [3, 7, 2],
        "per_sample_correct": [true, true, true],
        "per_sample_confidence": [0.95, 0.87, 0.92]
    }
}
```

---

## 2. CSV Export Format

### 2.1 Metrics Summary CSV

One row per run. Suitable for spreadsheet analysis.

```csv
report_id,timestamp,dataset,split,num_samples,accuracy,f1_macro,f1_weighted,auroc,precision_macro,recall_macro,duration_seconds
bench_20260220_143052_a1b2c3,2026-02-20T14:30:52,mnist,test,10000,0.9532,0.9518,0.9531,0.9981,0.9525,0.9520,45.23
bench_20260220_150000_d4e5f6,2026-02-20T15:00:00,cifar10,test,10000,0.8734,0.8712,0.8730,0.9823,0.8720,0.8715,120.45
```

**Column order**: metadata columns first, then metrics in alphabetical order, then timing.

**NaN Handling**: NaN values are written as empty strings in CSV.

### 2.2 Per-Class CSV

One row per class per run.

```csv
report_id,dataset,class_id,class_name,precision,recall,f1,support
bench_20260220_143052_a1b2c3,mnist,0,digit_0,0.98,0.97,0.975,980
bench_20260220_143052_a1b2c3,mnist,1,digit_1,0.96,0.99,0.975,1135
```

### 2.3 Confusion Matrix CSV

Flat format for easy import.

```csv
report_id,dataset,true_label,pred_label,count
bench_20260220_143052_a1b2c3,mnist,0,0,970
bench_20260220_143052_a1b2c3,mnist,0,1,1
bench_20260220_143052_a1b2c3,mnist,0,2,2
```

---

## 3. Comparison Report Format

### 3.1 Multi-Run Comparison

Compares multiple runs side by side.

```json
{
    "comparison": {
        "runs": [
            {"report_id": "run_a", "dataset": "mnist", "accuracy": 0.95},
            {"report_id": "run_b", "dataset": "mnist", "accuracy": 0.96}
        ],
        "best_per_metric": {
            "accuracy": {"report_id": "run_b", "value": 0.96},
            "f1_macro": {"report_id": "run_b", "value": 0.955}
        },
        "pairwise_deltas": {
            "run_a_vs_run_b": {
                "accuracy": -0.01,
                "f1_macro": -0.005
            }
        }
    }
}
```

### 3.2 Comparison CSV

```csv
metric,run_a,run_b,delta,relative_delta,winner
accuracy,0.9500,0.9600,0.0100,0.0105,run_b
f1_macro,0.9480,0.9550,0.0070,0.0074,run_b
auroc,0.9980,0.9975,-0.0005,-0.0005,run_a
```

### 3.3 Delta Computation Rules

1. **Absolute delta**: `delta = value_new - value_baseline`
2. **Relative delta**: `relative = delta / |value_baseline|` if `|value_baseline| > epsilon` else `sign(delta) * Inf`
3. **Significance threshold**: Changes smaller than `epsilon` (default 1e-4) are classified as "unchanged".
4. **Higher-is-better metrics**: accuracy, F1, AUROC, recall, precision, NAB score.
5. **Lower-is-better metrics**: loss, forgetting.
6. **Winner determination**: For higher-is-better, higher value wins. For lower-is-better, lower value wins.

```python
def compute_delta(current, baseline, higher_is_better=True, epsilon=1e-4):
    delta = current - baseline
    if abs(baseline) > epsilon:
        relative = delta / abs(baseline)
    else:
        relative = float("inf") if delta > 0 else float("-inf") if delta < 0 else 0.0

    if abs(delta) <= epsilon:
        status = "unchanged"
    elif (delta > 0 and higher_is_better) or (delta < 0 and not higher_is_better):
        status = "improved"
    else:
        status = "degraded"

    return {"delta": delta, "relative": relative, "status": status}
```

---

## 4. Report File Organization

### Directory Structure

```
runs/
  <run_id>/
    benchmark/
      summary.json           # Full JSON report
      summary.csv             # Metrics summary (one row)
      per_class.csv           # Per-class breakdown
      confusion_matrix.csv    # Confusion matrix (flat)
      comparison.json         # Comparison with baseline (if baseline_path set)
      comparison.csv          # Comparison table
```

### Naming Convention

- Report files: `{dataset}_{split}_{task_type}.{ext}`
- Comparison files: `compare_{baseline_id}_vs_{current_id}.{ext}`
- Suite reports: `suite_{timestamp}.{ext}`

### Auto-Save Behavior

```python
def save_report(result, output_dir: str):
    """Save benchmark report in all configured formats."""
    os.makedirs(output_dir, exist_ok=True)

    # Always save JSON
    json_path = os.path.join(output_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=_json_serializer)

    # Save CSV if configured
    if result.config.report_format in ("csv", "both"):
        csv_path = os.path.join(output_dir, "summary.csv")
        _write_metrics_csv(result, csv_path)

        if result.config.save_per_class:
            _write_per_class_csv(result, os.path.join(output_dir, "per_class.csv"))

        if result.config.save_confusion_matrix and result.confusion_matrix is not None:
            _write_confusion_csv(result, os.path.join(output_dir, "confusion_matrix.csv"))
```

---

## 5. JSON Serialization Details

### Custom Serializer

```python
def _json_serializer(obj):
    """Handle non-standard types in JSON serialization."""
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, set):
        return sorted(list(obj))
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
```

### Round-Trip Fidelity

Reports must survive JSON round-trip without data loss:

```python
# Save
with open(path, "w") as f:
    json.dump(report, f, default=_json_serializer)

# Load
with open(path, "r") as f:
    loaded = json.load(f)

# Verify
assert loaded["per_metric"]["accuracy"] == report["per_metric"]["accuracy"]
assert loaded["metadata"]["num_samples"] == report["metadata"]["num_samples"]
```

**Float precision**: Use Python's default JSON float representation (sufficient for 15+ significant digits).

**Tensor conversion**: Tensors are converted to nested lists via `.tolist()`.

**NaN round-trip**: NaN is serialized as `null` (JSON standard). On load, `null` is read as `None`. Consumers must handle `None` values.

---

## 6. Streaming Reports

For long-running assessments, emit partial reports periodically.

```python
class StreamingReporter:
    def __init__(self, output_path, interval_seconds=60):
        self.output_path = output_path
        self.interval = interval_seconds
        self.last_write = time.time()

    def update(self, metrics_snapshot):
        """Write partial report if interval elapsed."""
        now = time.time()
        if now - self.last_write >= self.interval:
            partial = {
                "status": "in_progress",
                "samples_processed": metrics_snapshot["count"],
                "partial_metrics": metrics_snapshot["metrics"],
                "elapsed_seconds": metrics_snapshot["elapsed"],
            }
            with open(self.output_path, "w") as f:
                json.dump(partial, f, indent=2)
            self.last_write = now
```

---

## 7. Report Validation

### Schema Validation

```python
REQUIRED_SECTIONS = ["metadata", "per_metric", "timing"]
REQUIRED_METADATA = ["report_id", "timestamp", "dataset", "split", "num_samples", "task_type"]
REQUIRED_TIMING = ["total_seconds"]

def validate_report(report: dict) -> list:
    """Validate report structure, return list of errors."""
    errors = []
    for section in REQUIRED_SECTIONS:
        if section not in report:
            errors.append(f"Missing required section: {section}")

    if "metadata" in report:
        for field_name in REQUIRED_METADATA:
            if field_name not in report["metadata"]:
                errors.append(f"Missing required metadata field: {field_name}")

    if "per_metric" in report:
        for key, value in report["per_metric"].items():
            if value is not None and not isinstance(value, (int, float)):
                errors.append(f"Metric '{key}' has non-numeric value: {value}")

    return errors
```

### Backward Compatibility

When loading old reports that may lack newer fields:

```python
def load_report(path: str) -> dict:
    with open(path) as f:
        report = json.load(f)

    # Backfill missing sections with defaults
    report.setdefault("timing", {"total_seconds": None})
    report.setdefault("comparison_baseline", None)
    report.setdefault("confusion_matrix", None)

    # Backfill metadata
    if "metadata" in report:
        report["metadata"].setdefault("config", {})
        report["metadata"].setdefault("environment", {})

    return report
```

---

## 8. Implementation Checklist

- [ ] JSON reports pass schema validation.
- [ ] CSV export handles NaN as empty string.
- [ ] Confusion matrix CSV uses flat format (true_label, pred_label, count).
- [ ] Comparison deltas use correct higher-is-better / lower-is-better logic.
- [ ] Report ID is unique (timestamp + random suffix).
- [ ] Tensor values converted via `.tolist()` before serialization.
- [ ] Round-trip JSON save/load preserves all metric values.
- [ ] Large confusion matrices (>100 classes) are optionally excluded.
- [ ] Streaming reporter writes valid JSON at each checkpoint.
- [ ] Reports stored under `runs/<run_id>/benchmark/` directory.
