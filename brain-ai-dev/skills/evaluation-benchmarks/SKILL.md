---
name: Evaluation & Benchmarks
description: >
  This skill should be used when the user asks to "evaluate the model",
  "run benchmarks", "compute metrics", "measure accuracy", "test on MNIST",
  "compute F1 score", "generate confusion matrix", "evaluate few-shot",
  "measure anomaly detection", "run cognitive benchmarks", "compare model variants",
  "create evaluation report", "set up eval harness", or needs guidance on
  evaluation protocols, metrics computation, benchmark harnesses, or performance
  reporting for the brain_ai system.
version: 0.1.0
---

# Evaluation & Benchmarks

## Overview

Guide implementation of comprehensive evaluation infrastructure for all 7 cognitive layers. Cover standard classification metrics, few-shot learning evaluation, anomaly detection scoring, reasoning task accuracy, continual learning forgetting metrics, and cross-modality evaluation protocols. Produce structured JSON reports consumable by downstream tooling.

## Public Contract

### MetricsSuite

Core metrics computation engine handling per-class and aggregated metrics.

```python
class MetricsSuite:
    def __init__(self, task_type: str, num_classes: int, device: str = "cpu"): ...
    def update(self, predictions: Tensor, targets: Tensor) -> None: ...
    def compute(self) -> Dict[str, float]: ...  # accuracy, f1_macro, f1_weighted, auroc
    def confusion_matrix(self) -> Tensor: ...
    def reset(self) -> None: ...
    def per_class_metrics(self) -> Dict[int, Dict[str, float]]: ...
```

### BenchmarkHarness

Standardized evaluation loop for any dataset + model combination.

```python
class BenchmarkHarness:
    def __init__(self, model: BrainAI, config: EvalConfig): ...
    def run(self, dataset: str, split: str = "test") -> BenchmarkResult: ...
    def run_suite(self, datasets: List[str]) -> List[BenchmarkResult]: ...
    def compare(self, results: List[BenchmarkResult]) -> ComparisonReport: ...
```

### FewShotEvaluator

N-way K-shot evaluation protocol for meta-learning assessment.

```python
class FewShotEvaluator:
    def __init__(self, model: BrainAI, n_way: int, k_shot: int, n_query: int = 15): ...
    def evaluate(self, dataset: str, n_episodes: int = 600) -> FewShotResult: ...
```

### AnomalyEvaluator

HTM anomaly detection evaluation with temporal scoring (NAB-style).

```python
class AnomalyEvaluator:
    def __init__(self, model: BrainAI, window_size: int = 100): ...
    def evaluate(self, sequences: Tensor, labels: Tensor) -> AnomalyResult: ...
```

### ReasoningEvaluator

Logical reasoning task evaluation (bAbI, ProofWriter, FOLIO).

```python
class ReasoningEvaluator:
    def __init__(self, model: BrainAI, task_type: str): ...
    def evaluate(self, dataset: str) -> ReasoningResult: ...
```

## Key Concepts

### Metric Hierarchy

- **Classification**: accuracy, top-5 accuracy, F1 (macro/weighted), AUROC, precision, recall
- **Few-shot**: N-way K-shot accuracy, 95% CI over episodes
- **Anomaly**: precision, recall, F1, NAB score, early detection bonus
- **Reasoning**: exact match, logical consistency, proof accuracy
- **Continual**: backward transfer, forward transfer, average accuracy, forgetting measure
- **Cross-modality**: per-modality accuracy, fusion improvement ratio

### Benchmark Datasets by Phase

| Phase | Dev Benchmarks | Production Benchmarks |
|-------|---------------|----------------------|
| 1 SNN | MNIST, FashionMNIST | CIFAR-10, CIFAR-100 |
| 2 Encoders | MNIST multimodal | ImageNet-1k, LibriSpeech, WikiText-103 |
| 3 HTM | Synthetic sequences | NAB, Yahoo S5 |
| 4 Workspace | Simple multimodal | VQA v2, CMU-MultimodalSDK |
| 5 Active Inference | CartPole, MountainCar | D4RL, Minari suite |
| 6 Reasoning | Mini-bAbI | bAbI full, ProofWriter, FOLIO |
| 7 Meta-Learning | Omniglot 5-way 1-shot | mini-ImageNet, tiered-ImageNet |

### Evaluation Report Format

Structured JSON with sections: `metadata`, `per_metric`, `per_class`, `confusion_matrix`, `timing`, `comparison_baseline`. Reports auto-saved to `runs/<run_id>/eval/`.

## Configuration Surface

```python
@dataclass
class EvalConfig:
    task_type: str = "classify"         # classify | few_shot | anomaly | reasoning | continual
    metrics: List[str] = ("accuracy", "f1_macro", "auroc")
    num_classes: int = 10
    batch_size: int = 64
    device: str = "auto"
    # Few-shot
    n_way: int = 5
    k_shot: int = 1
    n_episodes: int = 600
    # Anomaly
    anomaly_window: int = 100
    # Reporting
    save_confusion_matrix: bool = True
    save_per_class: bool = True
    baseline_path: Optional[str] = None
    report_format: str = "json"         # json | csv | both
```

## Done-When Gates

1. **Metric Computation** — `MetricsSuite.compute()` returns correct values on synthetic data with known ground truth; per-class metrics match manual calculation within 1e-6.
2. **Benchmark Harness** — `BenchmarkHarness.run("mnist", "test")` completes end-to-end producing valid JSON report with all required sections.
3. **Cross-Phase Coverage** — All 7 phases have at least one dev benchmark that runs end-to-end producing valid metrics in <60s.

## Failure Modes

| Mode | Symptom | Fix |
|------|---------|-----|
| OOM on large dataset | Crash during eval | Reduce batch_size, enable gradient-free eval |
| Label mismatch | Metrics near zero | Verify dataset label mapping matches output_heads |
| Few-shot collapse | 1/N accuracy | Check meta-learning adaptation actually runs |
| Anomaly threshold sensitivity | All-positive/negative | Run threshold sweep, report AUROC |
| Stale baseline | Misleading deltas | Re-run baseline with same split seed |

## Anti-Patterns

- Evaluating on training data — always use held-out splits
- Single metric fixation — report the full suite, not just accuracy
- Ignoring confidence intervals — report CI for stochastic evaluations
- Skipping per-class breakdown — class imbalance hides rare-class failures
- Non-deterministic eval — set eval seeds for reproducibility

## Resources

### Reference Files
- **`references/metrics-catalog.md`** — Complete metric definitions, formulas, edge cases
- **`references/benchmark-harnesses.md`** — Per-phase benchmark setup and evaluation loops
- **`references/cross-modality-eval.md`** — Multi-modal evaluation protocols and fusion metrics
- **`references/reporting-format.md`** — JSON report schema, CSV export, comparison tables
- **`references/testing-matrix.md`** — Test scenarios for evaluation infrastructure

### Asset Files
- **`assets/metrics_template.py`** — MetricsSuite with per-class, confusion matrix, self-tests
- **`assets/benchmark_runner_template.py`** — BenchmarkHarness with phase-specific harnesses
- **`assets/few_shot_eval_template.py`** — FewShotEvaluator with episode sampling
- **`assets/anomaly_eval_template.py`** — AnomalyEvaluator with temporal scoring
- **`assets/reasoning_eval_template.py`** — ReasoningEvaluator for logical tasks
- **`assets/eval_config_template.py`** — EvalConfig + reporting + comparison utilities

### Scripts
- **`scripts/validate_eval.py`** — Validates evaluation infrastructure against done-when gates
- **`scripts/gen_eval_tests.py`** — Generates 100+ pytest test cases for evaluation modules
- **`scripts/run_benchmarks.py`** — CLI for running benchmark suites with reporting
