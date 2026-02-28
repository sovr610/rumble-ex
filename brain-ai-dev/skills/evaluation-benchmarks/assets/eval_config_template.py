"""
EvalConfig: Central configuration, report generation, and comparison engine
for the brain_ai benchmark infrastructure.

Provides the EvalConfig dataclass, JSON/CSV report generators, comparison
engine with delta computation, and report validation.

Dependencies: torch, numpy (standard for brain_ai)
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


# ======================================================================
# EvalConfig dataclass
# ======================================================================

@dataclass
class EvalConfig:
    """
    Central configuration for all assessment modes.

    Fields cover classification, few-shot, anomaly, reasoning, continual
    learning, and reporting options.
    """

    # General
    task_type: str = "classify"  # classify | few_shot | anomaly | reasoning | continual
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1_macro", "auroc"])
    num_classes: int = 10
    batch_size: int = 64
    device: str = "auto"
    seed: int = 42
    num_workers: int = 4

    # Few-shot
    n_way: int = 5
    k_shot: int = 1
    n_query: int = 15
    n_episodes: int = 600

    # Anomaly
    anomaly_window: int = 100
    n_thresholds: int = 50
    nab_profile: str = "standard"

    # Reasoning
    reasoning_task: str = "babi"

    # Reporting
    save_confusion_matrix: bool = True
    save_per_class: bool = True
    save_raw_data: bool = False
    baseline_path: Optional[str] = None
    report_format: str = "json"  # json | csv | both
    output_dir: str = "runs"

    # Thresholds for comparison
    delta_threshold: float = 1e-4

    def resolve_device(self) -> str:
        """Resolve 'auto' to actual device string."""
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvalConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    @classmethod
    def for_phase(cls, phase: int, mode: str = "dev") -> "EvalConfig":
        """Return a sensible default config for a training phase."""
        configs = {
            1: {"task_type": "classify", "num_classes": 10, "batch_size": 128},
            2: {"task_type": "classify", "num_classes": 10, "batch_size": 64},
            3: {"task_type": "anomaly", "anomaly_window": 50, "batch_size": 1},
            4: {"task_type": "classify", "num_classes": 10, "batch_size": 64},
            5: {"task_type": "classify", "num_classes": 2, "batch_size": 32},
            6: {"task_type": "reasoning", "reasoning_task": "babi"},
            7: {"task_type": "few_shot", "n_way": 5, "k_shot": 1, "n_episodes": 100},
        }
        base = configs.get(phase, {})
        if mode == "production":
            base["device"] = "cuda"
            if phase == 1:
                base["num_classes"] = 100
            if phase == 7:
                base["n_episodes"] = 600
        else:
            base["device"] = "cpu"
        return cls(**base)


# ======================================================================
# Higher-is-better / lower-is-better registry
# ======================================================================

HIGHER_IS_BETTER = {
    "accuracy", "f1_macro", "f1_weighted", "f1_micro", "auroc",
    "precision_macro", "recall_macro", "mean_accuracy", "nab_score",
    "exact_match", "logical_consistency", "proof_accuracy",
    "average_accuracy", "forward_transfer", "top_5_accuracy",
}

LOWER_IS_BETTER = {
    "loss", "forgetting", "perplexity",
}


def is_higher_better(metric_name: str) -> bool:
    """Return True if higher values are better for *metric_name*."""
    if metric_name in LOWER_IS_BETTER:
        return False
    return True  # default assumption


# ======================================================================
# Delta computation
# ======================================================================

@dataclass
class MetricDelta:
    """Result of comparing a single metric between two runs."""
    metric: str
    current: float
    baseline: float
    delta: float
    relative: float
    status: str  # "improved" | "degraded" | "unchanged"


def compute_delta(
    current: float,
    baseline: float,
    metric_name: str = "",
    epsilon: float = 1e-4,
) -> MetricDelta:
    """Compute absolute and relative delta between two metric values."""
    delta = current - baseline
    if abs(baseline) > epsilon:
        relative = delta / abs(baseline)
    else:
        if delta > 0:
            relative = float("inf")
        elif delta < 0:
            relative = float("-inf")
        else:
            relative = 0.0

    higher = is_higher_better(metric_name)
    if abs(delta) <= epsilon:
        status = "unchanged"
    elif (delta > 0 and higher) or (delta < 0 and not higher):
        status = "improved"
    else:
        status = "degraded"

    return MetricDelta(
        metric=metric_name,
        current=current,
        baseline=baseline,
        delta=delta,
        relative=relative,
        status=status,
    )


def compare_reports(
    current: Dict[str, float],
    baseline: Dict[str, float],
    epsilon: float = 1e-4,
) -> Dict[str, MetricDelta]:
    """Compare all shared metrics between two reports."""
    shared = set(current.keys()) & set(baseline.keys())
    results: Dict[str, MetricDelta] = {}
    for mn in sorted(shared):
        cv = current[mn]
        bv = baseline[mn]
        if (isinstance(cv, float) and math.isnan(cv)) or (isinstance(bv, float) and math.isnan(bv)):
            continue
        results[mn] = compute_delta(cv, bv, metric_name=mn, epsilon=epsilon)
    return results


# ======================================================================
# Report generation
# ======================================================================

def generate_report_id() -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    short_hex = uuid.uuid4().hex[:6]
    return f"bench_{ts}_{short_hex}"


def build_json_report(
    metrics: Dict[str, float],
    config: EvalConfig,
    dataset: str = "",
    split: str = "test",
    num_samples: int = 0,
    per_class: Optional[Dict[int, Dict[str, float]]] = None,
    confusion_matrix: Optional[Any] = None,
    timing: Optional[Dict[str, float]] = None,
    model_info: Optional[Dict[str, Any]] = None,
    baseline_metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Build a complete JSON report dict following the standard schema."""
    report_id = generate_report_id()
    report: Dict[str, Any] = {
        "metadata": {
            "report_id": report_id,
            "timestamp": datetime.utcnow().isoformat(),
            "dataset": dataset,
            "split": split,
            "num_samples": num_samples,
            "task_type": config.task_type,
            "num_classes": config.num_classes,
            "config": config.to_dict(),
        },
        "per_metric": _nan_to_none(metrics),
        "timing": timing or {"total_seconds": 0.0},
    }

    if model_info:
        report["metadata"]["model_info"] = model_info

    if per_class is not None and config.save_per_class:
        report["per_class"] = {
            str(k): _nan_to_none(v) for k, v in per_class.items()
        }

    if confusion_matrix is not None and config.save_confusion_matrix:
        if isinstance(confusion_matrix, Tensor):
            cm_list = confusion_matrix.tolist()
            shape = list(confusion_matrix.shape)
        elif isinstance(confusion_matrix, np.ndarray):
            cm_list = confusion_matrix.tolist()
            shape = list(confusion_matrix.shape)
        else:
            cm_list = confusion_matrix
            shape = None
        report["confusion_matrix"] = {"matrix": cm_list, "shape": shape}

    if baseline_metrics is not None:
        deltas = compare_reports(metrics, baseline_metrics, config.delta_threshold)
        report["comparison_baseline"] = {
            "deltas": {k: d.delta for k, d in deltas.items()},
            "relative_deltas": {k: d.relative for k, d in deltas.items()},
            "improved": [k for k, d in deltas.items() if d.status == "improved"],
            "degraded": [k for k, d in deltas.items() if d.status == "degraded"],
            "unchanged": [k for k, d in deltas.items() if d.status == "unchanged"],
        }

    return report


def save_json_report(report: Dict[str, Any], path: str) -> None:
    """Write report dict to JSON file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2, default=_json_default)


def load_json_report(path: str) -> Dict[str, Any]:
    """Load report from JSON file with backfill for missing sections."""
    with open(path) as fh:
        report = json.load(fh)
    report.setdefault("timing", {"total_seconds": None})
    report.setdefault("comparison_baseline", None)
    report.setdefault("confusion_matrix", None)
    if "metadata" in report:
        report["metadata"].setdefault("config", {})
        report["metadata"].setdefault("model_info", {})
    return report


# ======================================================================
# CSV export
# ======================================================================

def metrics_to_csv_row(
    report: Dict[str, Any],
) -> Dict[str, Any]:
    """Flatten a report into a single CSV-friendly row."""
    meta = report.get("metadata", {})
    metrics = report.get("per_metric", {})
    timing = report.get("timing", {})

    row: Dict[str, Any] = {
        "report_id": meta.get("report_id", ""),
        "timestamp": meta.get("timestamp", ""),
        "dataset": meta.get("dataset", ""),
        "split": meta.get("split", ""),
        "num_samples": meta.get("num_samples", 0),
    }
    for k in sorted(metrics.keys()):
        v = metrics[k]
        row[k] = "" if v is None else v
    row["total_seconds"] = timing.get("total_seconds", "")
    return row


def write_metrics_csv(reports: List[Dict[str, Any]], path: str) -> None:
    """Write a list of reports as CSV rows."""
    if not reports:
        return
    rows = [metrics_to_csv_row(r) for r in reports]
    all_keys = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)


def write_per_class_csv(
    per_class: Dict[int, Dict[str, float]],
    report_id: str,
    dataset: str,
    path: str,
) -> None:
    """Write per-class metrics to CSV."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rows = []
    for cls_id in sorted(per_class.keys()):
        row = {"report_id": report_id, "dataset": dataset, "class_id": cls_id}
        row.update(per_class[cls_id])
        rows.append(row)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_comparison_csv(
    deltas: Dict[str, MetricDelta],
    path: str,
) -> None:
    """Write comparison results to CSV."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rows = []
    for mn, d in sorted(deltas.items()):
        rows.append({
            "metric": mn,
            "current": d.current,
            "baseline": d.baseline,
            "delta": d.delta,
            "relative_delta": d.relative,
            "status": d.status,
        })
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


# ======================================================================
# Report validation
# ======================================================================

REQUIRED_SECTIONS = ["metadata", "per_metric", "timing"]
REQUIRED_METADATA = ["report_id", "timestamp", "dataset", "split", "num_samples", "task_type"]


def validate_report(report: Dict[str, Any]) -> List[str]:
    """Validate report structure, return list of errors."""
    errors: List[str] = []
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

    if "confusion_matrix" in report and report["confusion_matrix"] is not None:
        cm = report["confusion_matrix"]
        if not isinstance(cm, dict) or "matrix" not in cm:
            errors.append("confusion_matrix must be a dict with 'matrix' key")

    return errors


# ======================================================================
# Helpers
# ======================================================================

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
    if isinstance(obj, set):
        return sorted(list(obj))
    raise TypeError(f"Not serialisable: {type(obj)}")


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
    print("EvalConfig & Reporting Self-Tests")
    print("=" * 60)

    # ---- EvalConfig construction ----
    cfg = EvalConfig()
    check(cfg.task_type == "classify", "T01 default task_type")
    check(cfg.num_classes == 10, "T02 default num_classes")
    check(cfg.batch_size == 64, "T03 default batch_size")
    check(cfg.n_way == 5, "T04 default n_way")
    check(cfg.anomaly_window == 100, "T05 default anomaly_window")

    # ---- resolve_device ----
    cfg_cpu = EvalConfig(device="cpu")
    check(cfg_cpu.resolve_device() == "cpu", "T06 resolve cpu")
    cfg_auto = EvalConfig(device="auto")
    dev = cfg_auto.resolve_device()
    check(dev in ("cpu", "cuda"), "T07 resolve auto")

    # ---- to_dict / from_dict ----
    d = cfg.to_dict()
    check(isinstance(d, dict), "T08 to_dict returns dict")
    check(d["task_type"] == "classify", "T09 to_dict task_type")
    cfg2 = EvalConfig.from_dict(d)
    check(cfg2.task_type == cfg.task_type, "T10 from_dict round-trip")
    check(cfg2.num_classes == cfg.num_classes, "T11 from_dict num_classes")

    # ---- from_dict ignores unknown keys ----
    cfg3 = EvalConfig.from_dict({"task_type": "anomaly", "unknown_key": 123})
    check(cfg3.task_type == "anomaly", "T12 from_dict unknown key ignored")

    # ---- for_phase ----
    cfg_p1 = EvalConfig.for_phase(1, "dev")
    check(cfg_p1.task_type == "classify", "T13 phase 1 task_type")
    cfg_p3 = EvalConfig.for_phase(3, "dev")
    check(cfg_p3.task_type == "anomaly", "T14 phase 3 task_type")
    cfg_p7 = EvalConfig.for_phase(7, "production")
    check(cfg_p7.n_episodes == 600, "T15 phase 7 prod n_episodes")

    # ---- is_higher_better ----
    check(is_higher_better("accuracy") is True, "T16 accuracy higher is better")
    check(is_higher_better("loss") is False, "T17 loss lower is better")
    check(is_higher_better("f1_macro") is True, "T18 f1_macro higher")
    check(is_higher_better("forgetting") is False, "T19 forgetting lower")

    # ---- compute_delta improved ----
    md = compute_delta(0.95, 0.90, "accuracy")
    check(md.status == "improved", "T20 delta improved")
    check(approx(md.delta, 0.05), "T21 delta value")
    check(approx(md.relative, 0.05 / 0.90), "T22 relative delta")

    # ---- compute_delta degraded ----
    md2 = compute_delta(0.85, 0.90, "accuracy")
    check(md2.status == "degraded", "T23 delta degraded")

    # ---- compute_delta unchanged ----
    md3 = compute_delta(0.90, 0.90, "accuracy")
    check(md3.status == "unchanged", "T24 delta unchanged")

    # ---- compute_delta lower-is-better ----
    md4 = compute_delta(0.10, 0.20, "loss")
    check(md4.status == "improved", "T25 loss decrease = improved")
    md5 = compute_delta(0.30, 0.20, "loss")
    check(md5.status == "degraded", "T26 loss increase = degraded")

    # ---- compare_reports ----
    curr = {"accuracy": 0.95, "f1_macro": 0.93}
    base = {"accuracy": 0.90, "f1_macro": 0.94}
    cmp = compare_reports(curr, base)
    check("accuracy" in cmp, "T27 compare has accuracy")
    check(cmp["accuracy"].status == "improved", "T28 accuracy improved")
    check(cmp["f1_macro"].status == "degraded", "T29 f1_macro degraded")

    # ---- build_json_report ----
    report = build_json_report(
        metrics={"accuracy": 0.95, "auroc": float("nan")},
        config=cfg,
        dataset="mnist",
        split="test",
        num_samples=10000,
    )
    check("metadata" in report, "T30 report has metadata")
    check("per_metric" in report, "T31 report has per_metric")
    check(report["per_metric"]["accuracy"] == 0.95, "T32 report accuracy")
    check(report["per_metric"]["auroc"] is None, "T33 NaN -> None")
    check(report["metadata"]["dataset"] == "mnist", "T34 report dataset")

    # ---- build_json_report with baseline ----
    report_cmp = build_json_report(
        metrics={"accuracy": 0.95},
        config=cfg,
        baseline_metrics={"accuracy": 0.90},
    )
    check("comparison_baseline" in report_cmp, "T35 report has comparison")
    check(len(report_cmp["comparison_baseline"]["improved"]) > 0, "T36 improved list")

    # ---- validate_report ----
    errors = validate_report(report)
    check(len(errors) == 0, "T37 valid report no errors")

    bad_report: Dict[str, Any] = {"per_metric": {"accuracy": 0.5}}
    errors_bad = validate_report(bad_report)
    check(len(errors_bad) > 0, "T38 missing sections detected")

    bad_report2: Dict[str, Any] = {
        "metadata": {"report_id": "x"},
        "per_metric": {"accuracy": "not_a_number"},
        "timing": {},
    }
    errors_bad2 = validate_report(bad_report2)
    check(any("non-numeric" in e for e in errors_bad2), "T39 non-numeric metric detected")

    # ---- JSON round-trip ----
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
        tmp_path = tmp.name
    try:
        save_json_report(report, tmp_path)
        loaded = load_json_report(tmp_path)
        check(loaded["per_metric"]["accuracy"] == 0.95, "T40 JSON round-trip accuracy")
        check(loaded["metadata"]["dataset"] == "mnist", "T41 JSON round-trip dataset")
    finally:
        os.unlink(tmp_path)

    # ---- CSV export ----
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as tmp:
        csv_path = tmp.name
    try:
        write_metrics_csv([report], csv_path)
        with open(csv_path) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        check(len(rows) == 1, "T42 CSV has 1 row")
        check(rows[0]["dataset"] == "mnist", "T43 CSV dataset field")
    finally:
        os.unlink(csv_path)

    # ---- per-class CSV ----
    per_class = {0: {"precision": 0.9, "recall": 0.8}, 1: {"precision": 0.85, "recall": 0.9}}
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as tmp:
        pc_path = tmp.name
    try:
        write_per_class_csv(per_class, "test_id", "mnist", pc_path)
        with open(pc_path) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        check(len(rows) == 2, "T44 per-class CSV has 2 rows")
    finally:
        os.unlink(pc_path)

    # ---- comparison CSV ----
    deltas = compare_reports({"accuracy": 0.95}, {"accuracy": 0.90})
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as tmp:
        cmp_path = tmp.name
    try:
        write_comparison_csv(deltas, cmp_path)
        with open(cmp_path) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        check(len(rows) == 1, "T45 comparison CSV has 1 row")
        check(rows[0]["status"] == "improved", "T46 comparison status")
    finally:
        os.unlink(cmp_path)

    # ---- generate_report_id format ----
    rid = generate_report_id()
    check(rid.startswith("bench_"), "T47 report_id prefix")
    check(len(rid) > 20, "T48 report_id length")

    # ---- metrics_to_csv_row NaN handling ----
    report_nan = build_json_report(
        metrics={"accuracy": 0.5, "auroc": float("nan")},
        config=cfg,
    )
    row = metrics_to_csv_row(report_nan)
    check(row["auroc"] == "", "T49 NaN -> empty in CSV row")

    # ---- EvalConfig for unknown phase ----
    cfg_unk = EvalConfig.for_phase(99, "dev")
    check(cfg_unk.task_type == "classify", "T50 unknown phase defaults")

    # ---- Multiple reports in CSV ----
    r1 = build_json_report(metrics={"accuracy": 0.8}, config=cfg, dataset="ds1")
    r2 = build_json_report(metrics={"accuracy": 0.9}, config=cfg, dataset="ds2")
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as tmp:
        multi_path = tmp.name
    try:
        write_metrics_csv([r1, r2], multi_path)
        with open(multi_path) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        check(len(rows) == 2, "T51 multi-report CSV")
    finally:
        os.unlink(multi_path)

    # ---- compare_reports skips NaN ----
    cmp_nan = compare_reports(
        {"accuracy": 0.9, "auroc": float("nan")},
        {"accuracy": 0.8, "auroc": 0.95},
    )
    check("auroc" not in cmp_nan, "T52 compare skips NaN")
    check("accuracy" in cmp_nan, "T53 compare keeps valid")

    # ---- validate report with confusion_matrix ----
    report_cm = build_json_report(
        metrics={"accuracy": 0.9},
        config=cfg,
        confusion_matrix=torch.eye(3),
    )
    errs_cm = validate_report(report_cm)
    check(len(errs_cm) == 0, "T54 report with CM valid")

    # ---- load_json_report backfill ----
    minimal = {"metadata": {"report_id": "x", "timestamp": "t", "dataset": "d",
                             "split": "s", "num_samples": 0, "task_type": "classify"},
               "per_metric": {"accuracy": 0.5},
               "timing": {"total_seconds": 1.0}}
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
        min_path = tmp.name
    try:
        with open(min_path, "w") as fh:
            json.dump(minimal, fh)
        loaded = load_json_report(min_path)
        check("comparison_baseline" in loaded, "T55 backfill comparison")
        check("config" in loaded["metadata"], "T56 backfill config")
    finally:
        os.unlink(min_path)

    # ---- delta with zero baseline ----
    md_zero = compute_delta(0.5, 0.0, "accuracy")
    check(md_zero.relative == float("inf"), "T57 zero baseline -> inf relative")

    # ---- delta with negative baseline for lower-is-better ----
    md_neg = compute_delta(0.05, 0.1, "forgetting")
    check(md_neg.status == "improved", "T58 forgetting decrease improved")

    # ---- EvalConfig seed ----
    check(cfg.seed == 42, "T59 default seed")

    # ---- Report with per_class and confusion_matrix disabled ----
    cfg_no = EvalConfig(save_per_class=False, save_confusion_matrix=False)
    report_no = build_json_report(
        metrics={"accuracy": 0.5}, config=cfg_no,
        per_class={0: {"f1": 0.5}}, confusion_matrix=torch.eye(2),
    )
    check("per_class" not in report_no, "T60 per_class excluded when disabled")
    check("confusion_matrix" not in report_no, "T61 CM excluded when disabled")

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")
    print("=" * 60)
    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    _run_self_tests()
