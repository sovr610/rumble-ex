"""
BenchmarkHarness: Standardized assessment loop for brain_ai models.

Provides BenchmarkResult and ComparisonReport dataclasses, a full harness for
running single datasets, suites of datasets, and comparing results with delta
computation.  All assessment is gradient-free (torch.no_grad).

Dependencies: torch, numpy (standard for brain_ai)
"""

from __future__ import annotations

import copy
import json
import math
import os
import time
import uuid
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


# ======================================================================
# Data classes
# ======================================================================

@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    dataset: str
    split: str
    metrics: Dict[str, float]
    per_class: Dict[int, Dict[str, float]]
    confusion_matrix: Optional[Tensor]
    config: Any  # EvalConfig or dict
    timestamp: str = ""
    duration_seconds: float = 0.0
    num_samples: int = 0
    model_info: Dict[str, Any] = field(default_factory=dict)
    report_id: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()
        if not self.report_id:
            short_hex = uuid.uuid4().hex[:6]
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            self.report_id = f"bench_{ts}_{short_hex}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-safe dictionary."""
        d: Dict[str, Any] = {
            "metadata": {
                "report_id": self.report_id,
                "timestamp": self.timestamp,
                "dataset": self.dataset,
                "split": self.split,
                "num_samples": self.num_samples,
                "task_type": _config_field(self.config, "task_type", "classify"),
                "num_classes": _config_field(self.config, "num_classes", -1),
                "config": _config_to_dict(self.config),
                "model_info": self.model_info,
            },
            "per_metric": _nan_to_none(self.metrics),
            "per_class": {
                str(k): _nan_to_none(v) for k, v in self.per_class.items()
            },
            "timing": {
                "total_seconds": self.duration_seconds,
                "per_sample_ms": (
                    self.duration_seconds / max(self.num_samples, 1) * 1000.0
                ),
            },
        }
        if self.confusion_matrix is not None:
            cm = self.confusion_matrix
            d["confusion_matrix"] = {
                "matrix": cm.tolist() if isinstance(cm, Tensor) else cm,
                "shape": list(cm.shape) if isinstance(cm, Tensor) else None,
            }
        return d

    def save(self, path: str) -> None:
        """Save report as JSON."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=_json_default)


@dataclass
class ComparisonReport:
    """Comparison across multiple BenchmarkResults."""

    results: List[BenchmarkResult]
    deltas: Dict[str, Dict[str, float]] = field(default_factory=dict)
    baseline: Optional[BenchmarkResult] = None
    summary: str = ""
    best_per_metric: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "comparison": {
                "num_results": len(self.results),
                "deltas": self.deltas,
                "best_per_metric": self.best_per_metric,
                "summary": self.summary,
            },
        }


# ======================================================================
# MetricsSuite -- lightweight inline copy for self-contained template
# ======================================================================

class _MetricsSuite:
    """Minimal metrics accumulator (mirrors assets/metrics_template.py)."""

    def __init__(self, task_type: str = "classify", num_classes: int = 10,
                 device: str = "cpu") -> None:
        self.task_type = task_type
        self.num_classes = num_classes
        self.device = device
        self.reset()

    def reset(self) -> None:
        self._confusion = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.int64,
            device=self.device,
        )
        self._all_probs: List[Tensor] = []
        self._all_targets: List[Tensor] = []
        self._total = 0

    def update(self, predictions: Tensor, targets: Tensor) -> None:
        if predictions.numel() == 0 and targets.numel() == 0:
            return
        targets = targets.to(self.device).long()
        predictions = predictions.to(self.device)
        if predictions.dim() == 1:
            pred_labels = predictions.long()
        else:
            pred_labels = predictions.argmax(dim=-1)
            self._all_probs.append(predictions.detach().cpu().float())
            self._all_targets.append(targets.detach().cpu())
        for t, p in zip(targets, pred_labels):
            ti, pi = t.item(), p.item()
            if 0 <= ti < self.num_classes and 0 <= pi < self.num_classes:
                self._confusion[ti, pi] += 1
        self._total += targets.shape[0]

    def compute(self) -> Dict[str, float]:
        if self._total == 0:
            return {"accuracy": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0}
        tp = self._confusion.diag()
        fp = self._confusion.sum(dim=0) - tp
        fn = self._confusion.sum(dim=1) - tp
        support = self._confusion.sum(dim=1)
        accuracy = float(tp.sum()) / float(self._total)
        prec = self._sdiv(tp.float(), (tp + fp).float())
        rec = self._sdiv(tp.float(), (tp + fn).float())
        f1 = self._sdiv(2.0 * prec * rec, prec + rec)
        f1_macro = float(f1.mean())
        total_s = support.sum().float()
        f1_weighted = float((f1 * support.float() / total_s.clamp(min=1)).sum()) if total_s > 0 else 0.0
        return {"accuracy": accuracy, "f1_macro": f1_macro, "f1_weighted": f1_weighted}

    def per_class_metrics(self) -> Dict[int, Dict[str, float]]:
        tp = self._confusion.diag()
        fp = self._confusion.sum(dim=0) - tp
        fn = self._confusion.sum(dim=1) - tp
        support = self._confusion.sum(dim=1)
        prec = self._sdiv(tp.float(), (tp + fp).float())
        rec = self._sdiv(tp.float(), (tp + fn).float())
        f1 = self._sdiv(2.0 * prec * rec, prec + rec)
        out: Dict[int, Dict[str, float]] = {}
        for c in range(self.num_classes):
            out[c] = {
                "precision": float(prec[c]),
                "recall": float(rec[c]),
                "f1": float(f1[c]),
                "support": int(support[c]),
            }
        return out

    def confusion_matrix(self, normalize: Optional[str] = None) -> Tensor:
        cm = self._confusion.clone().float()
        if normalize == "true":
            cm = cm / cm.sum(dim=1, keepdim=True).clamp(min=1)
        elif normalize == "pred":
            cm = cm / cm.sum(dim=0, keepdim=True).clamp(min=1)
        elif normalize == "all":
            cm = cm / cm.sum().clamp(min=1)
        return cm

    @staticmethod
    def _sdiv(num: Tensor, den: Tensor) -> Tensor:
        out = torch.zeros_like(num)
        mask = den > 0
        out[mask] = num[mask] / den[mask]
        return out


# ======================================================================
# BenchmarkHarness
# ======================================================================

class BenchmarkHarness:
    """
    Standardised assessment loop for any dataset + model combination.

    Args:
        model: A ``torch.nn.Module`` that accepts a dict of tensors.
        config: An EvalConfig (or plain dict) with task_type, num_classes,
                batch_size, device, etc.
        metrics_cls: Optional custom MetricsSuite class.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Any,
        metrics_cls: Optional[type] = None,
    ) -> None:
        self.model = model
        self.config = config
        self._device = _config_field(config, "device", "cpu")
        num_classes = _config_field(config, "num_classes", 10)
        task_type = _config_field(config, "task_type", "classify")
        cls = metrics_cls or _MetricsSuite
        self.metrics = cls(task_type=task_type, num_classes=num_classes, device="cpu")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        dataset: str = "synthetic",
        split: str = "test",
        dataloader: Optional[DataLoader] = None,
        prepare_fn: Optional[Callable] = None,
    ) -> BenchmarkResult:
        """
        Run assessment on *dataloader* (or a synthetic fallback).

        Args:
            dataset: Name of the dataset (for metadata).
            split: Split name (for metadata).
            dataloader: Optional DataLoader.  When ``None`` a synthetic
                loader is generated.
            prepare_fn: Optional callable ``(batch) -> (inputs_dict, targets)``.
        """
        if dataloader is None:
            dataloader = self._synthetic_loader()
        if prepare_fn is None:
            prepare_fn = self._default_prepare

        self.metrics.reset()
        self.model.eval()
        num_samples = 0
        t0 = time.time()

        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = prepare_fn(batch)
                outputs = self.model(inputs)
                preds = self._extract_predictions(outputs)
                self.metrics.update(preds, targets)
                num_samples += targets.shape[0]

        duration = time.time() - t0
        cm = self.metrics.confusion_matrix()
        return BenchmarkResult(
            dataset=dataset,
            split=split,
            metrics=self.metrics.compute(),
            per_class=self.metrics.per_class_metrics(),
            confusion_matrix=cm,
            config=self.config,
            duration_seconds=duration,
            num_samples=num_samples,
            model_info=self._model_info(),
        )

    def run_suite(
        self,
        datasets: List[str],
        dataloaders: Optional[Dict[str, DataLoader]] = None,
    ) -> List[BenchmarkResult]:
        """Run assessment on multiple datasets."""
        results: List[BenchmarkResult] = []
        for ds_name in datasets:
            loader = (dataloaders or {}).get(ds_name)
            results.append(self.run(dataset=ds_name, dataloader=loader))
        return results

    def compare(
        self,
        results: List[BenchmarkResult],
        baseline: Optional[BenchmarkResult] = None,
        epsilon: float = 1e-4,
    ) -> ComparisonReport:
        """
        Compare BenchmarkResults.  If *baseline* is given, deltas are
        computed against it; otherwise the first result is the baseline.
        """
        if not results:
            return ComparisonReport(results=[])
        base = baseline or results[0]

        deltas: Dict[str, Dict[str, float]] = {}
        best_per: Dict[str, Dict[str, Any]] = {}

        metric_names = list(base.metrics.keys())
        for mn in metric_names:
            best_val = None
            best_id = None
            for r in results:
                v = r.metrics.get(mn, float("nan"))
                if not math.isnan(v):
                    if best_val is None or v > best_val:
                        best_val = v
                        best_id = r.report_id
            best_per[mn] = {"report_id": best_id, "value": best_val}

        for r in results:
            if r.report_id == base.report_id:
                continue
            pair_deltas: Dict[str, float] = {}
            for mn in metric_names:
                bv = base.metrics.get(mn, float("nan"))
                cv = r.metrics.get(mn, float("nan"))
                if math.isnan(bv) or math.isnan(cv):
                    pair_deltas[mn] = float("nan")
                else:
                    pair_deltas[mn] = cv - bv
            deltas[f"{base.report_id}_vs_{r.report_id}"] = pair_deltas

        improved: List[str] = []
        degraded: List[str] = []
        unchanged: List[str] = []
        if len(results) == 2 and results[0].report_id == base.report_id:
            for mn in metric_names:
                d = deltas.get(
                    f"{base.report_id}_vs_{results[1].report_id}", {}
                ).get(mn, float("nan"))
                if math.isnan(d):
                    unchanged.append(mn)
                elif d > epsilon:
                    improved.append(mn)
                elif d < -epsilon:
                    degraded.append(mn)
                else:
                    unchanged.append(mn)

        summary_parts = [f"{len(results)} results compared."]
        if improved:
            summary_parts.append(f"Improved: {', '.join(improved)}.")
        if degraded:
            summary_parts.append(f"Degraded: {', '.join(degraded)}.")
        if unchanged:
            summary_parts.append(f"Unchanged: {', '.join(unchanged)}.")

        return ComparisonReport(
            results=results,
            deltas=deltas,
            baseline=base,
            summary=" ".join(summary_parts),
            best_per_metric=best_per,
        )

    # ------------------------------------------------------------------
    # Aggregation utility
    # ------------------------------------------------------------------

    @staticmethod
    def aggregate_suite(results: List[BenchmarkResult]) -> Dict[str, float]:
        """Aggregate metrics across a suite of results (mean, skip NaN)."""
        buckets: Dict[str, List[float]] = defaultdict(list)
        for r in results:
            for k, v in r.metrics.items():
                if not math.isnan(v):
                    buckets[k].append(v)
        return {k: float(np.mean(v)) for k, v in buckets.items() if v}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_predictions(self, outputs: Any) -> Tensor:
        if isinstance(outputs, dict):
            for key in ("logits", "predictions", "output"):
                if key in outputs:
                    return outputs[key]
            return next(iter(outputs.values()))
        return outputs

    def _default_prepare(self, batch: Any) -> Tuple[Dict[str, Tensor], Tensor]:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            data, targets = batch
            return {"vision": data}, targets
        raise TypeError(f"Cannot auto-prepare batch of type {type(batch)}")

    def _synthetic_loader(self) -> DataLoader:
        nc = _config_field(self.config, "num_classes", 10)
        bs = _config_field(self.config, "batch_size", 32)
        n = bs * 4
        data = torch.randn(n, 1, 28, 28)
        labels = torch.randint(0, nc, (n,))
        ds = TensorDataset(data, labels)
        return DataLoader(ds, batch_size=bs, shuffle=False)

    def _model_info(self) -> Dict[str, Any]:
        total = sum(p.numel() for p in self.model.parameters())
        return {"total_params": total}


# ======================================================================
# Helpers
# ======================================================================

def _config_field(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _config_to_dict(config: Any) -> Dict[str, Any]:
    if isinstance(config, dict):
        return config
    try:
        return asdict(config)
    except Exception:
        return {}


def _nan_to_none(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, float) and math.isnan(v):
            out[k] = None
        else:
            out[k] = v
    return out


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Tensor):
        return obj.tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Not serialisable: {type(obj)}")


# ======================================================================
# MockBrainAI -- used by self-tests
# ======================================================================

class MockBrainAI(nn.Module):
    """Trivial model that maps flattened input to logits."""

    def __init__(self, num_classes: int = 10, input_dim: int = 784) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, inputs: Dict[str, Tensor], **kw: Any) -> Tensor:
        x = list(inputs.values())[0]
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        if x.shape[-1] != self.linear.in_features:
            x = x[:, : self.linear.in_features]
            if x.shape[-1] < self.linear.in_features:
                pad = torch.zeros(
                    x.shape[0],
                    self.linear.in_features - x.shape[-1],
                    device=x.device,
                )
                x = torch.cat([x, pad], dim=-1)
        return self.linear(x)


class PerfectMockModel(nn.Module):
    """Returns one-hot logits matching target labels stored at creation."""

    def __init__(self, targets: Tensor, num_classes: int = 10) -> None:
        super().__init__()
        self._targets = targets
        self._nc = num_classes
        self._idx = 0
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, inputs: Dict[str, Tensor], **kw: Any) -> Tensor:
        bs = list(inputs.values())[0].shape[0]
        out = torch.zeros(bs, self._nc)
        for i in range(bs):
            idx = (self._idx + i) % len(self._targets)
            out[i, self._targets[idx].item()] = 10.0
        self._idx = (self._idx + bs) % len(self._targets)
        return out


# ======================================================================
# Self-tests (30+)
# ======================================================================

def _run_self_tests() -> None:
    passed = 0
    failed = 0

    def check(cond: bool, name: str) -> None:
        nonlocal passed, failed
        if cond:
            passed += 1
            print(f"  PASS: {name}")
        else:
            failed += 1
            print(f"  FAIL: {name}")

    def approx(a: float, b: float, tol: float = 1e-4) -> bool:
        if math.isnan(a) and math.isnan(b):
            return True
        if math.isnan(a) or math.isnan(b):
            return False
        return abs(a - b) < tol

    print("=" * 60)
    print("BenchmarkHarness Self-Tests")
    print("=" * 60)

    # ---- BenchmarkResult construction ----
    r = BenchmarkResult(
        dataset="mnist", split="test",
        metrics={"accuracy": 0.95, "f1_macro": 0.94},
        per_class={0: {"f1": 0.9, "precision": 0.9, "recall": 0.9, "support": 100}},
        confusion_matrix=torch.eye(10, dtype=torch.int64),
        config={"task_type": "classify", "num_classes": 10},
    )
    check(r.report_id.startswith("bench_"), "T01 report_id prefix")
    check(len(r.timestamp) > 0, "T02 timestamp non-empty")

    # ---- to_dict ----
    d = r.to_dict()
    check("metadata" in d, "T03 to_dict has metadata")
    check("per_metric" in d, "T04 to_dict has per_metric")
    check(d["per_metric"]["accuracy"] == 0.95, "T05 to_dict accuracy value")
    check("confusion_matrix" in d, "T06 to_dict has confusion_matrix")

    # ---- NaN handling ----
    r2 = BenchmarkResult(
        dataset="x", split="t",
        metrics={"auroc": float("nan"), "accuracy": 0.5},
        per_class={}, confusion_matrix=None,
        config={},
    )
    d2 = r2.to_dict()
    check(d2["per_metric"]["auroc"] is None, "T07 NaN serialised as None")
    check(d2["per_metric"]["accuracy"] == 0.5, "T08 non-NaN preserved")

    # ---- ComparisonReport ----
    cr = ComparisonReport(results=[r, r2])
    check(len(cr.results) == 2, "T09 ComparisonReport holds 2 results")

    # ---- MockBrainAI forward ----
    model = MockBrainAI(num_classes=5, input_dim=100)
    out = model({"vision": torch.randn(4, 100)})
    check(out.shape == (4, 5), "T10 MockBrainAI output shape")

    # ---- MockBrainAI with images ----
    out2 = model({"vision": torch.randn(2, 1, 10, 10)})
    check(out2.shape == (2, 5), "T11 MockBrainAI flatten images")

    # ---- BenchmarkHarness instantiation ----
    cfg = {"task_type": "classify", "num_classes": 5, "batch_size": 16, "device": "cpu"}
    harness = BenchmarkHarness(model, cfg)
    check(harness.metrics.num_classes == 5, "T12 harness num_classes")

    # ---- run with synthetic ----
    res = harness.run("synthetic", "test")
    check(isinstance(res, BenchmarkResult), "T13 run returns BenchmarkResult")
    check(res.num_samples > 0, "T14 run num_samples > 0")
    check(res.duration_seconds >= 0.0, "T15 run duration >= 0")
    check("accuracy" in res.metrics, "T16 run has accuracy")

    # ---- run_suite ----
    suite = harness.run_suite(["ds1", "ds2", "ds3"])
    check(len(suite) == 3, "T17 run_suite returns 3 results")
    check(suite[0].dataset == "ds1", "T18 suite[0] dataset name")
    check(suite[2].dataset == "ds3", "T19 suite[2] dataset name")

    # ---- compare ----
    comp = harness.compare(suite)
    check(isinstance(comp, ComparisonReport), "T20 compare returns ComparisonReport")
    check(comp.baseline is not None, "T21 compare has baseline")
    check(len(comp.deltas) > 0, "T22 compare has deltas")
    check(len(comp.summary) > 0, "T23 compare has summary")

    # ---- compare with explicit baseline ----
    comp2 = harness.compare(suite, baseline=suite[1])
    check(comp2.baseline.dataset == "ds2", "T24 explicit baseline dataset")

    # ---- PerfectMockModel ----
    targets = torch.tensor([0, 1, 2, 3, 4])
    pmodel = PerfectMockModel(targets, num_classes=5)
    po = pmodel({"vision": torch.randn(5, 10)})
    pred_labels = po.argmax(dim=-1)
    check(torch.equal(pred_labels, targets), "T25 PerfectMockModel correct")

    # ---- Run with perfect model ----
    n_samples = 64
    labels = torch.randint(0, 5, (n_samples,))
    pmodel2 = PerfectMockModel(labels, num_classes=5)
    data = torch.randn(n_samples, 1, 28, 28)
    ds = TensorDataset(data, labels)
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    h2 = BenchmarkHarness(pmodel2, cfg)
    res_perf = h2.run("perfect_test", "test", dataloader=loader)
    check(approx(res_perf.metrics["accuracy"], 1.0), "T26 perfect model accuracy=1.0")
    check(approx(res_perf.metrics["f1_macro"], 1.0), "T27 perfect model f1_macro=1.0")

    # ---- aggregate_suite ----
    r_a = BenchmarkResult(
        dataset="a", split="t", metrics={"accuracy": 0.8, "f1_macro": 0.7},
        per_class={}, confusion_matrix=None, config={},
    )
    r_b = BenchmarkResult(
        dataset="b", split="t", metrics={"accuracy": 0.9, "f1_macro": 0.85},
        per_class={}, confusion_matrix=None, config={},
    )
    agg = BenchmarkHarness.aggregate_suite([r_a, r_b])
    check(approx(agg["accuracy"], 0.85), "T28 aggregate accuracy mean")
    check(approx(agg["f1_macro"], 0.775), "T29 aggregate f1 mean")

    # ---- aggregate with NaN ----
    r_c = BenchmarkResult(
        dataset="c", split="t",
        metrics={"accuracy": float("nan"), "f1_macro": 0.6},
        per_class={}, confusion_matrix=None, config={},
    )
    agg2 = BenchmarkHarness.aggregate_suite([r_a, r_c])
    check(approx(agg2["accuracy"], 0.8), "T30 aggregate skips NaN")
    check(approx(agg2["f1_macro"], 0.65), "T31 aggregate mean with partial NaN")

    # ---- compare deltas sign ----
    r_lo = BenchmarkResult(
        dataset="d", split="t", metrics={"accuracy": 0.5},
        per_class={}, confusion_matrix=None, config={},
    )
    r_hi = BenchmarkResult(
        dataset="d", split="t", metrics={"accuracy": 0.9},
        per_class={}, confusion_matrix=None, config={},
    )
    comp3 = harness.compare([r_lo, r_hi])
    key = list(comp3.deltas.keys())[0]
    check(comp3.deltas[key]["accuracy"] > 0, "T32 delta positive when hi > lo")

    # ---- best_per_metric ----
    check(comp3.best_per_metric["accuracy"]["value"] == 0.9, "T33 best_per_metric value")

    # ---- save / load round-trip ----
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
        tmp_path = tmp.name
    try:
        res.save(tmp_path)
        with open(tmp_path) as fh:
            loaded = json.load(fh)
        check(loaded["per_metric"]["accuracy"] == 0.95, "T34 JSON round-trip accuracy")
        check(loaded["metadata"]["dataset"] == "mnist", "T35 JSON round-trip dataset")
    finally:
        os.unlink(tmp_path)

    # ---- custom prepare_fn ----
    def custom_prepare(batch):
        data, labels = batch
        return {"sensor": data}, labels

    h3 = BenchmarkHarness(model, cfg)
    res_custom = h3.run("custom", "test", dataloader=loader, prepare_fn=custom_prepare)
    check(res_custom.num_samples == n_samples, "T36 custom prepare_fn sample count")

    # ---- model_info has total_params ----
    check("total_params" in res.model_info, "T37 model_info has total_params")
    check(res.model_info["total_params"] > 0, "T38 total_params > 0")

    # ---- empty compare ----
    comp_empty = harness.compare([])
    check(len(comp_empty.results) == 0, "T39 empty compare returns empty")

    # ---- config as dataclass-like ----
    class _FakeCfg:
        task_type = "classify"
        num_classes = 3
        batch_size = 8
        device = "cpu"

    m3 = MockBrainAI(num_classes=3, input_dim=784)
    h4 = BenchmarkHarness(m3, _FakeCfg())
    r4 = h4.run("dc_test")
    check(r4.metrics["accuracy"] >= 0.0, "T40 dataclass-like config works")

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")
    print("=" * 60)
    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    _run_self_tests()
