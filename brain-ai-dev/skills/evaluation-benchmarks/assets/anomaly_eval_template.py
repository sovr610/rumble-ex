"""
AnomalyEvaluator: Temporal anomaly detection assessment with NAB-style scoring.

Provides sliding-window anomaly detection, threshold sweep, NAB score
computation, and standard precision / recall / F1 metrics.

Dependencies: torch, numpy (standard for brain_ai)
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


# ======================================================================
# Data classes
# ======================================================================

@dataclass
class AnomalyResult:
    """Result of anomaly detection assessment."""

    precision: float
    recall: float
    f1: float
    nab_score: float
    auroc: float
    best_threshold: float
    num_true_anomalies: int
    num_predicted_anomalies: int
    num_samples: int
    threshold_curve: List[Dict[str, float]] = field(default_factory=list)
    duration_seconds: float = 0.0
    nab_profile: str = "standard"

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "nab_score": self.nab_score,
            "auroc": self.auroc,
            "best_threshold": self.best_threshold,
            "num_true_anomalies": self.num_true_anomalies,
            "num_predicted_anomalies": self.num_predicted_anomalies,
            "num_samples": self.num_samples,
            "nab_profile": self.nab_profile,
            "duration_seconds": self.duration_seconds,
        }
        for k, v in list(out.items()):
            if isinstance(v, float) and math.isnan(v):
                out[k] = None
        return out


# ======================================================================
# NAB scoring profiles
# ======================================================================

NAB_PROFILES: Dict[str, Dict[str, float]] = {
    "standard": {"fp_weight": -0.11, "fn_weight": -1.0},
    "low_fp": {"fp_weight": -0.22, "fn_weight": -1.0},
    "low_fn": {"fp_weight": -0.11, "fn_weight": -2.0},
}


# ======================================================================
# Core scoring functions
# ======================================================================

def compute_binary_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """Compute precision, recall, F1 for binary anomaly predictions."""
    tp = int(((predictions == 1) & (labels == 1)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def compute_nab_score(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    window_size: int = 100,
    profile: str = "standard",
) -> float:
    """
    Compute NAB-style anomaly score.

    For each true anomaly window the first detection inside the window
    receives a position-dependent reward; undetected windows get a penalty;
    false positives outside any window get a smaller penalty.

    Returns score normalised to [0, 100] (can go negative for poor detectors).
    """
    cfg = NAB_PROFILES.get(profile, NAB_PROFILES["standard"])
    fp_w = cfg["fp_weight"]
    fn_w = cfg["fn_weight"]

    # Identify anomaly windows (contiguous regions of label==1, expanded by window_size)
    windows = _find_anomaly_windows(labels, window_size)
    if len(windows) == 0:
        return float("nan")  # No anomalies -> NAB undefined

    predictions = (scores >= threshold).astype(np.int64)
    total_score = 0.0
    max_possible = 0.0

    detected_in_window = [False] * len(windows)
    detection_positions: List[Optional[int]] = [None] * len(windows)

    for t in range(len(predictions)):
        if predictions[t] == 1:
            in_window = False
            for wi, (ws, we) in enumerate(windows):
                if ws <= t <= we:
                    in_window = True
                    if not detected_in_window[wi]:
                        detected_in_window[wi] = True
                        detection_positions[wi] = t
                    break
            if not in_window:
                total_score += fp_w  # false positive penalty

    # Score each window
    for wi, (ws, we) in enumerate(windows):
        wlen = max(we - ws + 1, 1)
        max_possible += 1.0  # best possible per window
        if detected_in_window[wi] and detection_positions[wi] is not None:
            pos = detection_positions[wi]
            relative = (pos - ws) / max(wlen, 1)
            reward = 2.0 * _sigmoid(-5.0 * relative) - 1.0
            total_score += max(reward, 0.0)
        else:
            total_score += fn_w

    if max_possible == 0:
        return 0.0
    return float(total_score / max_possible * 100.0)


def _sigmoid(x: float) -> float:
    if x < -500:
        return 0.0
    if x > 500:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def _find_anomaly_windows(
    labels: np.ndarray, window_size: int
) -> List[Tuple[int, int]]:
    """
    Find contiguous anomaly regions in *labels* and expand each by
    *window_size* for early-detection credit.
    """
    windows: List[Tuple[int, int]] = []
    n = len(labels)
    i = 0
    while i < n:
        if labels[i] == 1:
            start = i
            while i < n and labels[i] == 1:
                i += 1
            end = i - 1
            # Expand window for early-detection credit
            w_start = max(0, start - window_size // 2)
            w_end = min(n - 1, end + window_size // 2)
            windows.append((w_start, w_end))
        else:
            i += 1
    return _merge_windows(windows)


def _merge_windows(
    windows: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Merge overlapping windows."""
    if not windows:
        return []
    windows = sorted(windows)
    merged = [windows[0]]
    for s, e in windows[1:]:
        if s <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUROC for anomaly scores vs binary labels."""
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    tp_cum = np.cumsum(sorted_labels)
    fp_cum = np.cumsum(1 - sorted_labels)
    tpr = np.concatenate([[0.0], tp_cum / n_pos])
    fpr = np.concatenate([[0.0], fp_cum / n_neg])
    return float(np.trapz(tpr, fpr))


def threshold_sweep(
    scores: np.ndarray,
    labels: np.ndarray,
    n_thresholds: int = 50,
    window_size: int = 100,
    profile: str = "standard",
) -> Tuple[List[Dict[str, float]], float]:
    """
    Sweep thresholds and return curve + best threshold (by F1).

    Returns:
        curve: List of dicts with threshold, precision, recall, f1, nab_score.
        best_threshold: Threshold that maximises F1.
    """
    lo = float(scores.min())
    hi = float(scores.max())
    if lo == hi:
        thresholds = [lo]
    else:
        thresholds = np.linspace(lo, hi, n_thresholds).tolist()

    curve: List[Dict[str, float]] = []
    best_f1 = -1.0
    best_thr = thresholds[0]

    for thr in thresholds:
        preds = (scores >= thr).astype(np.int64)
        m = compute_binary_metrics(preds, labels)
        nab = compute_nab_score(scores, labels, thr, window_size, profile)
        entry = {
            "threshold": thr,
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "nab_score": nab if not math.isnan(nab) else 0.0,
        }
        curve.append(entry)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thr = thr

    return curve, best_thr


# ======================================================================
# AnomalyEvaluator
# ======================================================================

class AnomalyEvaluator:
    """
    Assesses anomaly detection models on temporal sequences.

    Args:
        model: Module that maps input dict -> anomaly scores (B, T).
        window_size: NAB window expansion size.
        n_thresholds: Number of thresholds in sweep.
        profile: NAB profile name.
        score_fn: Optional callable (model, inputs) -> anomaly_scores.
    """

    def __init__(
        self,
        model: nn.Module,
        window_size: int = 100,
        n_thresholds: int = 50,
        profile: str = "standard",
        score_fn: Optional[Callable] = None,
    ) -> None:
        self.model = model
        self.window_size = window_size
        self.n_thresholds = n_thresholds
        self.profile = profile
        self.score_fn = score_fn or self._default_score_fn

    def run_assessment(
        self,
        sequences: Tensor,
        labels: Tensor,
    ) -> AnomalyResult:
        """
        Run anomaly detection assessment.

        Args:
            sequences: (B, T) or (B, T, D) input sequences.
            labels:    (B, T) binary anomaly labels.
        """
        t0 = time.time()
        self.model.eval()
        with torch.no_grad():
            scores = self.score_fn(self.model, sequences)

        # Flatten to 1-D for scoring
        scores_np = scores.cpu().numpy().ravel()
        labels_np = labels.cpu().numpy().ravel().astype(np.int64)

        # Threshold sweep
        curve, best_thr = threshold_sweep(
            scores_np, labels_np,
            n_thresholds=self.n_thresholds,
            window_size=self.window_size,
            profile=self.profile,
        )

        # Compute final metrics at best threshold
        best_preds = (scores_np >= best_thr).astype(np.int64)
        bm = compute_binary_metrics(best_preds, labels_np)
        nab = compute_nab_score(
            scores_np, labels_np, best_thr,
            self.window_size, self.profile,
        )
        auroc = compute_auroc(scores_np, labels_np)
        duration = time.time() - t0

        return AnomalyResult(
            precision=bm["precision"],
            recall=bm["recall"],
            f1=bm["f1"],
            nab_score=nab,
            auroc=auroc,
            best_threshold=best_thr,
            num_true_anomalies=int(labels_np.sum()),
            num_predicted_anomalies=int(best_preds.sum()),
            num_samples=len(labels_np),
            threshold_curve=curve,
            duration_seconds=duration,
            nab_profile=self.profile,
        )

    # Alias for backward compat with SKILL.md contract
    evaluate = run_assessment

    @staticmethod
    def _default_score_fn(model: nn.Module, sequences: Tensor) -> Tensor:
        """Default: pass sequences through model, expect scalar output."""
        out = model({"sequences": sequences})
        if isinstance(out, dict):
            for k in ("anomaly_scores", "scores", "output"):
                if k in out:
                    return out[k]
            return next(iter(out.values()))
        return out


# ======================================================================
# Synthetic data generators
# ======================================================================

def generate_anomaly_data(
    n_sequences: int = 5,
    seq_len: int = 200,
    anomaly_rate: float = 0.05,
    seed: int = 42,
) -> Tuple[Tensor, Tensor]:
    """
    Generate sine-wave sequences with injected point anomalies.

    Returns:
        data:   (n_sequences, seq_len)
        labels: (n_sequences, seq_len) binary
    """
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 20 * np.pi, seq_len)
    data = np.sin(t)[None, :].repeat(n_sequences, axis=0)
    data += rng.randn(n_sequences, seq_len) * 0.1
    labels = np.zeros((n_sequences, seq_len), dtype=np.int64)

    for i in range(n_sequences):
        n_anom = max(1, int(seq_len * anomaly_rate))
        positions = rng.choice(seq_len, size=n_anom, replace=False)
        data[i, positions] += rng.randn(n_anom) * 5.0
        labels[i, positions] = 1

    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


# ======================================================================
# Mock model
# ======================================================================

class _MockAnomalyModel(nn.Module):
    """Returns reconstruction error as anomaly score."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, inputs: Dict[str, Tensor], **kw: Any) -> Tensor:
        x = inputs.get("sequences", list(inputs.values())[0])
        # Anomaly score: abs deviation from running mean
        if x.dim() == 3:
            x = x.mean(dim=-1)
        # Simple: use absolute value as proxy for anomaly score
        mean_val = x.mean(dim=-1, keepdim=True)
        scores = (x - mean_val).abs()
        return scores


class _PerfectAnomalyModel(nn.Module):
    """Returns high scores at anomaly positions (cheating model)."""

    def __init__(self, labels: Tensor) -> None:
        super().__init__()
        self._labels = labels.float()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, inputs: Dict[str, Tensor], **kw: Any) -> Tensor:
        return self._labels * 10.0 + 0.01


# ======================================================================
# Self-tests (25+)
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

    def approx(a: float, b: float, tol: float = 0.05) -> bool:
        if math.isnan(a) and math.isnan(b):
            return True
        if math.isnan(a) or math.isnan(b):
            return False
        return abs(a - b) < tol

    print("=" * 60)
    print("AnomalyEvaluator Self-Tests")
    print("=" * 60)

    # ---- AnomalyResult construction ----
    ar = AnomalyResult(
        precision=0.8, recall=0.9, f1=0.85, nab_score=70.0,
        auroc=0.92, best_threshold=0.5,
        num_true_anomalies=50, num_predicted_anomalies=60, num_samples=1000,
    )
    check(ar.f1 == 0.85, "T01 AnomalyResult field")
    d = ar.to_dict()
    check("nab_score" in d, "T02 to_dict has nab_score")
    check(d["precision"] == 0.8, "T03 to_dict precision")

    # ---- NaN in to_dict ----
    ar2 = AnomalyResult(
        precision=0.0, recall=0.0, f1=0.0, nab_score=float("nan"),
        auroc=float("nan"), best_threshold=0.5,
        num_true_anomalies=0, num_predicted_anomalies=0, num_samples=100,
    )
    d2 = ar2.to_dict()
    check(d2["nab_score"] is None, "T04 NaN -> None in to_dict")

    # ---- compute_binary_metrics perfect ----
    preds = np.array([1, 1, 0, 0])
    labs = np.array([1, 1, 0, 0])
    bm = compute_binary_metrics(preds, labs)
    check(approx(bm["precision"], 1.0), "T05 perfect precision")
    check(approx(bm["recall"], 1.0), "T06 perfect recall")
    check(approx(bm["f1"], 1.0), "T07 perfect F1")

    # ---- compute_binary_metrics all wrong ----
    bm2 = compute_binary_metrics(np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0]))
    check(approx(bm2["precision"], 0.0), "T08 all wrong precision")
    check(approx(bm2["recall"], 0.0), "T09 all wrong recall")

    # ---- compute_binary_metrics partial ----
    bm3 = compute_binary_metrics(np.array([1, 0, 1, 0]), np.array([1, 1, 0, 0]))
    check(approx(bm3["precision"], 0.5), "T10 partial precision=0.5")
    check(approx(bm3["recall"], 0.5), "T11 partial recall=0.5")

    # ---- compute_auroc perfect separation ----
    scores_perf = np.array([0.9, 0.8, 0.2, 0.1])
    labs_perf = np.array([1, 1, 0, 0])
    auc = compute_auroc(scores_perf, labs_perf)
    check(approx(auc, 1.0), "T12 perfect AUROC")

    # ---- compute_auroc random ----
    rng = np.random.RandomState(42)
    n = 1000
    scores_rand = rng.rand(n)
    labs_rand = rng.randint(0, 2, n)
    auc_rand = compute_auroc(scores_rand, labs_rand)
    check(0.35 < auc_rand < 0.65, "T13 random AUROC ~0.5")

    # ---- compute_auroc single class -> NaN ----
    auc_nan = compute_auroc(np.array([0.5, 0.6]), np.array([0, 0]))
    check(math.isnan(auc_nan), "T14 single class AUROC NaN")

    # ---- _find_anomaly_windows ----
    labels_w = np.array([0, 0, 1, 1, 1, 0, 0, 0, 1, 0])
    windows = _find_anomaly_windows(labels_w, window_size=2)
    check(len(windows) >= 1, "T15 found anomaly windows")

    # ---- _merge_windows ----
    merged = _merge_windows([(0, 3), (2, 5), (7, 9)])
    check(merged == [(0, 5), (7, 9)], "T16 merge overlapping windows")

    # ---- NAB score: no anomalies -> NaN ----
    nab_nan = compute_nab_score(
        np.array([0.5, 0.6, 0.3]), np.array([0, 0, 0]), 0.5, window_size=2,
    )
    check(math.isnan(nab_nan), "T17 NAB no anomalies -> NaN")

    # ---- threshold_sweep ----
    scores_ts = np.array([0.1, 0.2, 0.9, 0.8, 0.3])
    labels_ts = np.array([0, 0, 1, 1, 0])
    curve, best_thr = threshold_sweep(scores_ts, labels_ts, n_thresholds=10, window_size=2)
    check(len(curve) > 0, "T18 threshold_sweep returns curve")
    check(isinstance(best_thr, float), "T19 best_threshold is float")

    # ---- generate_anomaly_data ----
    data, labs = generate_anomaly_data(n_sequences=3, seq_len=100, anomaly_rate=0.1)
    check(data.shape == (3, 100), "T20 synthetic data shape")
    check(labs.shape == (3, 100), "T21 synthetic labels shape")
    check(labs.sum().item() > 0, "T22 synthetic has anomalies")

    # ---- AnomalyEvaluator with mock model ----
    mock_model = _MockAnomalyModel()
    assessor = AnomalyEvaluator(mock_model, window_size=10, n_thresholds=20)
    data_run, labs_run = generate_anomaly_data(n_sequences=2, seq_len=200)
    result = assessor.run_assessment(data_run, labs_run)
    check(isinstance(result, AnomalyResult), "T23 returns AnomalyResult")
    check(0.0 <= result.precision <= 1.0, "T24 precision in [0,1]")
    check(0.0 <= result.recall <= 1.0, "T25 recall in [0,1]")
    check(result.duration_seconds >= 0.0, "T26 duration >= 0")
    check(result.num_samples > 0, "T27 num_samples > 0")

    # ---- Perfect model should get high scores ----
    perf_model = _PerfectAnomalyModel(labs_run)
    perf_assessor = AnomalyEvaluator(perf_model, window_size=10, n_thresholds=20)
    res_perf = perf_assessor.run_assessment(data_run, labs_run)
    check(res_perf.f1 > 0.8, "T28 perfect model F1 > 0.8")
    check(res_perf.auroc > 0.9, "T29 perfect model AUROC > 0.9")

    # ---- NAB profiles ----
    for profile_name in ("standard", "low_fp", "low_fn"):
        ev = AnomalyEvaluator(mock_model, window_size=10, profile=profile_name)
        r = ev.run_assessment(data_run, labs_run)
        check(r.nab_profile == profile_name, f"T30_{profile_name} profile name")

    # ---- threshold_curve populated ----
    check(len(result.threshold_curve) > 0, "T31 threshold_curve non-empty")
    check("threshold" in result.threshold_curve[0], "T32 curve entry has threshold")
    check("f1" in result.threshold_curve[0], "T33 curve entry has f1")

    # ---- Constant scores -> threshold sweep still works ----
    const_scores = np.ones(100) * 0.5
    const_labels = np.zeros(100, dtype=np.int64)
    const_labels[50:55] = 1
    curve_c, _ = threshold_sweep(const_scores, const_labels, n_thresholds=5, window_size=5)
    check(len(curve_c) >= 1, "T34 constant scores threshold sweep")

    # ---- Custom score_fn ----
    def custom_score(model, sequences):
        return torch.rand_like(sequences.float())

    ev_custom = AnomalyEvaluator(mock_model, score_fn=custom_score)
    r_custom = ev_custom.run_assessment(data_run, labs_run)
    check(isinstance(r_custom, AnomalyResult), "T35 custom score_fn works")

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")
    print("=" * 60)
    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    _run_self_tests()
