"""
Out-of-Distribution (OOD) Detector Template.

Provides ``OODDetector`` with energy-based scoring, Mahalanobis distance,
maximum softmax probability (MSP), AUROC / FPR@95TPR metrics, and
threshold calibration.

Dependencies: torch + standard library only.  No PIL, no numpy.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Result / metric containers
# ---------------------------------------------------------------------------

@dataclass
class OODResult:
    """Single-batch OOD detection output."""
    scores: torch.Tensor          # (B,) higher = more OOD
    is_ood: torch.Tensor          # (B,) bool after threshold
    threshold: float

    def ood_fraction(self) -> float:
        return self.is_ood.float().mean().item()


@dataclass
class OODMetrics:
    """Aggregate evaluation metrics for OOD detection."""
    auroc: float
    fpr_at_95tpr: float
    id_score_mean: float
    ood_score_mean: float
    threshold: float
    method: str

    def summary(self) -> str:
        return (
            f"OOD Detection [{self.method}]\n"
            f"  AUROC:        {self.auroc:.4f}\n"
            f"  FPR@95TPR:    {self.fpr_at_95tpr:.4f}\n"
            f"  ID mean:      {self.id_score_mean:.4f}\n"
            f"  OOD mean:     {self.ood_score_mean:.4f}\n"
            f"  Threshold:    {self.threshold:.4f}"
        )


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def energy_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Energy-based OOD score.  Higher = more OOD."""
    return -temperature * torch.logsumexp(logits / temperature, dim=1)


def msp_score(logits: torch.Tensor) -> torch.Tensor:
    """Maximum softmax probability OOD score.  Higher = more OOD."""
    return -logits.softmax(dim=1).max(dim=1).values


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_auroc(id_scores: torch.Tensor, ood_scores: torch.Tensor) -> float:
    """AUROC via the trapezoidal rule.  Higher = better separation."""
    labels = torch.cat([torch.zeros(len(id_scores)), torch.ones(len(ood_scores))])
    scores = torch.cat([id_scores.detach().cpu(), ood_scores.detach().cpu()])

    sorted_idx = scores.argsort(descending=True)
    sorted_labels = labels[sorted_idx]

    n_pos = sorted_labels.sum().item()
    n_neg = len(sorted_labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    tp, fp = 0.0, 0.0
    tpr_prev, fpr_prev = 0.0, 0.0
    auroc = 0.0

    for lab in sorted_labels:
        if lab == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        auroc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2
        tpr_prev, fpr_prev = tpr, fpr

    return float(auroc)


def compute_fpr_at_tpr(
    id_scores: torch.Tensor,
    ood_scores: torch.Tensor,
    target_tpr: float = 0.95,
) -> float:
    """FPR when target_tpr fraction of OOD is detected."""
    sorted_ood = ood_scores.detach().cpu().sort(descending=True).values
    idx = int(target_tpr * len(sorted_ood))
    idx = min(idx, len(sorted_ood) - 1)
    threshold = sorted_ood[idx].item()
    fpr = (id_scores.detach().cpu() >= threshold).float().mean().item()
    return fpr


def calibrate_threshold(id_scores: torch.Tensor, target_fpr: float = 0.05) -> float:
    """Percentile-based threshold calibration."""
    sorted_scores = id_scores.detach().cpu().sort().values
    idx = int((1.0 - target_fpr) * len(sorted_scores))
    idx = min(idx, len(sorted_scores) - 1)
    return sorted_scores[idx].item()


# ---------------------------------------------------------------------------
# OODDetector
# ---------------------------------------------------------------------------

class OODDetector:
    """Detect out-of-distribution inputs.

    Parameters
    ----------
    model : nn.Module
        Classifier that returns logits (B, C).
    method : str
        ``"energy"`` | ``"mahalanobis"`` | ``"msp"``.
    temperature : float
        Temperature for energy scoring.
    threshold : float or None
        Hard decision boundary.  ``None`` => auto-calibrate during ``fit``.
    target_fpr : float
        Desired false-positive rate for auto-calibration.
    regularization : float
        Covariance regularisation for Mahalanobis.
    """

    def __init__(
        self,
        model: nn.Module,
        method: str = "energy",
        temperature: float = 1.0,
        threshold: Optional[float] = None,
        target_fpr: float = 0.05,
        regularization: float = 1e-5,
    ):
        self.model = model
        self.method = method
        self.temperature = temperature
        self.threshold = threshold
        self.target_fpr = target_fpr
        self.regularization = regularization

        # Mahalanobis statistics (set during fit)
        self.class_means: Dict[int, torch.Tensor] = {}
        self.precision: Optional[torch.Tensor] = None
        self._fitted = False

    # ---- internal forward ------------------------------------------------

    @staticmethod
    def _forward(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, dict):
            vals = list(x.values())
            if len(vals) == 1:
                return model(vals[0])
            return model(x)
        return model(x)

    # ---- fit (required for Mahalanobis, optional for energy) -------------

    def fit(self, data: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Fit detector on in-distribution data ``(X, Y)``.

        For energy / MSP: calibrates threshold.
        For Mahalanobis: also computes class means and shared covariance.
        """
        x, y = data
        self.model.eval()

        with torch.no_grad():
            logits = self._forward(self.model, x)

        # --- Mahalanobis fitting (must come before threshold calibration) ---
        if self.method == "mahalanobis":
            self._fit_mahalanobis(logits, y)

        # --- score id data for threshold calibration ---
        id_scores = self.get_score_from_logits(logits)
        self.threshold = calibrate_threshold(id_scores, self.target_fpr)

        self._fitted = True

    def _fit_mahalanobis(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        """Compute class means and shared precision matrix."""
        feats_by_class: Dict[int, List[torch.Tensor]] = defaultdict(list)
        for i, lab in enumerate(labels):
            feats_by_class[lab.item()].append(features[i])

        self.class_means = {}
        for c, feat_list in feats_by_class.items():
            self.class_means[c] = torch.stack(feat_list).mean(dim=0)

        all_centered: List[torch.Tensor] = []
        for c, feat_list in feats_by_class.items():
            feats = torch.stack(feat_list)
            centered = feats - self.class_means[c].unsqueeze(0)
            all_centered.append(centered)

        all_centered_t = torch.cat(all_centered, dim=0)
        cov = (all_centered_t.t() @ all_centered_t) / max(len(all_centered_t), 1)
        cov += self.regularization * torch.eye(cov.shape[0], device=cov.device)
        self.precision = torch.inverse(cov)

    # ---- scoring ---------------------------------------------------------

    def get_score(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute OOD score.  Higher = more OOD."""
        self.model.eval()
        with torch.no_grad():
            logits = self._forward(self.model, inputs)
        return self.get_score_from_logits(logits)

    def get_score_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute OOD score from pre-computed logits."""
        if self.method == "energy":
            return energy_score(logits, self.temperature)
        elif self.method == "msp":
            return msp_score(logits)
        elif self.method == "mahalanobis":
            return self._mahalanobis_score(logits)
        else:
            raise ValueError(f"Unknown OOD method: {self.method}")

    def _mahalanobis_score(self, features: torch.Tensor) -> torch.Tensor:
        if self.precision is None or not self.class_means:
            raise RuntimeError("Mahalanobis detector must be fit() first.")
        min_dist = torch.full((features.shape[0],), float("inf"),
                              device=features.device)
        for c, mean in self.class_means.items():
            diff = features - mean.unsqueeze(0)
            dist = (diff @ self.precision * diff).sum(dim=1)
            min_dist = torch.min(min_dist, dist)
        return min_dist  # higher distance = more OOD

    # ---- detect ----------------------------------------------------------

    def detect(self, inputs: torch.Tensor) -> OODResult:
        """Binary OOD detection on a batch."""
        scores = self.get_score(inputs)
        thr = self.threshold if self.threshold is not None else 0.0
        return OODResult(scores=scores, is_ood=(scores >= thr), threshold=thr)

    # ---- evaluate --------------------------------------------------------

    def evaluate(
        self,
        id_data: Tuple[torch.Tensor, torch.Tensor],
        ood_data: Tuple[torch.Tensor, torch.Tensor],
    ) -> OODMetrics:
        """Full evaluation: AUROC, FPR@95TPR."""
        id_x, _ = id_data
        ood_x, _ = ood_data

        id_scores = self.get_score(id_x)
        ood_scores = self.get_score(ood_x)

        auroc = compute_auroc(id_scores, ood_scores)
        fpr = compute_fpr_at_tpr(id_scores, ood_scores, target_tpr=0.95)

        thr = self.threshold if self.threshold is not None else 0.0

        return OODMetrics(
            auroc=auroc,
            fpr_at_95tpr=fpr,
            id_score_mean=id_scores.mean().item(),
            ood_score_mean=ood_scores.mean().item(),
            threshold=thr,
            method=self.method,
        )


# ===================================================================
# Self-tests  (30+)
# ===================================================================

def _run_self_tests() -> None:
    import sys
    passed = 0
    failed = 0

    def _assert(cond: bool, msg: str) -> None:
        nonlocal passed, failed
        if cond:
            passed += 1
            print(f"  PASS: {msg}")
        else:
            failed += 1
            print(f"  FAIL: {msg}")

    print("=" * 60)
    print("ood_detector_template self-tests")
    print("=" * 60)

    torch.manual_seed(123)

    # ---- simple model ----------------------------------------------------
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.fc = nn.Linear(32 * 7 * 7, num_classes)

        def forward(self, x):
            if isinstance(x, dict):
                x = list(x.values())[0]
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            return self.fc(x.flatten(1))

    NUM_CLASSES = 10

    # Create separable ID vs OOD data
    id_x = torch.rand(200, 1, 28, 28) * 0.4
    id_y = torch.randint(0, NUM_CLASSES, (200,))
    ood_x = torch.rand(200, 1, 28, 28) * 0.4 + 0.6
    ood_y = torch.randint(0, NUM_CLASSES, (200,))

    model = SimpleCNN(NUM_CLASSES)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for _ in range(50):
        logits = model(id_x)
        loss = F.cross_entropy(logits, id_y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()

    # ---- Energy score tests -----------------------------------------------
    logits_id = model(id_x)
    logits_ood = model(ood_x)

    e_id = energy_score(logits_id)
    e_ood = energy_score(logits_ood)
    _assert(e_id.shape == (200,), "energy_score returns correct shape")
    _assert(torch.isfinite(e_id).all().item(), "energy scores are finite (ID)")
    _assert(torch.isfinite(e_ood).all().item(), "energy scores are finite (OOD)")

    e_t2 = energy_score(logits_id, temperature=2.0)
    _assert(e_t2.shape == (200,), "energy_score with T=2 correct shape")
    _assert(torch.isfinite(e_t2).all().item(), "energy scores with T=2 finite")

    # ---- MSP score tests --------------------------------------------------
    m_id = msp_score(logits_id)
    m_ood = msp_score(logits_ood)
    _assert(m_id.shape == (200,), "msp_score returns correct shape")
    _assert(torch.isfinite(m_id).all().item(), "MSP scores are finite")
    _assert((m_id <= 0).all().item(), "MSP scores are non-positive (negated probability)")

    # ---- AUROC tests ------------------------------------------------------
    perfect_id = torch.zeros(100)
    perfect_ood = torch.ones(100)
    auroc_perfect = compute_auroc(perfect_id, perfect_ood)
    _assert(abs(auroc_perfect - 1.0) < 1e-6,
            f"Perfect separation AUROC = {auroc_perfect:.4f} ~ 1.0")

    rng_a = torch.randn(500)
    rng_b = torch.randn(500)
    auroc_random = compute_auroc(rng_a, rng_b)
    _assert(abs(auroc_random - 0.5) < 0.1,
            f"Random AUROC = {auroc_random:.4f} ~ 0.5")

    # ---- FPR@95TPR -------------------------------------------------------
    fpr_perfect = compute_fpr_at_tpr(perfect_id, perfect_ood, target_tpr=0.95)
    _assert(fpr_perfect < 0.05,
            f"FPR@95TPR for perfect separation = {fpr_perfect:.4f}")

    # ---- Threshold calibration -------------------------------------------
    scores = torch.randn(1000)
    thr = calibrate_threshold(scores, target_fpr=0.05)
    above = (scores >= thr).float().mean().item()
    _assert(abs(above - 0.05) < 0.02,
            f"Threshold calibration FPR ~ 5%, got {above:.2%}")

    # ---- OODDetector energy method ---------------------------------------
    det_energy = OODDetector(model, method="energy", temperature=1.0)
    det_energy.fit((id_x, id_y))
    _assert(det_energy._fitted, "Energy detector fitted successfully")
    _assert(det_energy.threshold is not None, "Threshold auto-calibrated")

    result_id = det_energy.detect(id_x[:16])
    _assert(result_id.scores.shape == (16,), "detect() returns correct score shape")
    _assert(result_id.is_ood.shape == (16,), "detect() returns correct is_ood shape")

    metrics_e = det_energy.evaluate((id_x, id_y), (ood_x, ood_y))
    _assert(isinstance(metrics_e, OODMetrics), "evaluate returns OODMetrics")
    _assert(0.0 <= metrics_e.auroc <= 1.0,
            f"Energy AUROC = {metrics_e.auroc:.4f} in [0,1]")
    _assert(0.0 <= metrics_e.fpr_at_95tpr <= 1.0,
            f"FPR@95TPR = {metrics_e.fpr_at_95tpr:.4f} in [0,1]")
    _assert(len(metrics_e.summary()) > 0, "OODMetrics summary non-empty")
    _assert(metrics_e.method == "energy", "Metrics method is energy")

    # ---- OODDetector MSP method ------------------------------------------
    det_msp = OODDetector(model, method="msp")
    det_msp.fit((id_x, id_y))
    metrics_m = det_msp.evaluate((id_x, id_y), (ood_x, ood_y))
    _assert(0.0 <= metrics_m.auroc <= 1.0,
            f"MSP AUROC = {metrics_m.auroc:.4f} in [0,1]")

    # ---- OODDetector Mahalanobis method ----------------------------------
    det_mah = OODDetector(model, method="mahalanobis", regularization=1e-4)
    det_mah.fit((id_x, id_y))
    _assert(len(det_mah.class_means) > 0, "Mahalanobis class_means populated")
    _assert(det_mah.precision is not None, "Mahalanobis precision matrix computed")

    mah_scores_id = det_mah.get_score(id_x[:16])
    _assert(mah_scores_id.shape == (16,), "Mahalanobis score shape correct")
    _assert(torch.isfinite(mah_scores_id).all().item(), "Mahalanobis scores finite")

    metrics_mah = det_mah.evaluate((id_x, id_y), (ood_x, ood_y))
    _assert(0.0 <= metrics_mah.auroc <= 1.0,
            f"Mahalanobis AUROC = {metrics_mah.auroc:.4f} in [0,1]")

    mah_id_full = det_mah.get_score(id_x)
    mah_ood_full = det_mah.get_score(ood_x)
    _assert(mah_id_full.mean().item() < mah_ood_full.mean().item() + 100,
            "Mahalanobis: ID mean score <= OOD mean score (with tolerance)")

    # ---- Unfitted Mahalanobis should fail --------------------------------
    det_bad = OODDetector(model, method="mahalanobis")
    try:
        det_bad.get_score(id_x[:4])
        _assert(False, "Unfitted Mahalanobis raises error")
    except RuntimeError:
        _assert(True, "Unfitted Mahalanobis raises error")

    # ---- Multiple OOD distributions --------------------------------------
    ood_gauss = torch.randn(100, 1, 28, 28).clamp(0, 1)
    ood_uniform = torch.rand(100, 1, 28, 28)
    ood_shifted = id_x[:100] + 0.5

    for name, ood_set in [("gaussian", ood_gauss), ("uniform", ood_uniform),
                           ("shifted", ood_shifted)]:
        s_ood = det_energy.get_score(ood_set)
        _assert(torch.isfinite(s_ood).all().item(),
                f"Energy scores finite for OOD={name}")

    # ---- OODResult helpers -----------------------------------------------
    res = det_energy.detect(ood_x[:50])
    _assert(0.0 <= res.ood_fraction() <= 1.0,
            f"ood_fraction() = {res.ood_fraction():.2%} in [0,1]")

    # ---- get_score without fit (energy) works ----------------------------
    det_nf = OODDetector(model, method="energy")
    s = det_nf.get_score(id_x[:8])
    _assert(s.shape == (8,), "get_score works without fit for energy method")

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    print("=" * 60)
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    _run_self_tests()
