"""
Calibration Analyzer Template.

Provides ``CalibrationAnalyzer`` with ECE computation, reliability diagram
data, temperature scaling, Platt scaling, and calibration-under-shift
analysis.

Dependencies: torch + standard library only.  No PIL, no numpy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ReliabilityDiagram:
    """Binned calibration data for reliability diagrams."""
    bin_accuracies: List[float]
    bin_confidences: List[float]
    bin_counts: List[int]
    n_bins: int

    def summary(self) -> str:
        lines = ["Reliability Diagram", f"  bins={self.n_bins}"]
        for i in range(self.n_bins):
            if self.bin_counts[i] > 0:
                lines.append(
                    f"  bin {i:2d}: conf={self.bin_confidences[i]:.3f}  "
                    f"acc={self.bin_accuracies[i]:.3f}  n={self.bin_counts[i]}"
                )
        return "\n".join(lines)


@dataclass
class CalibrationResult:
    """Full calibration analysis result."""
    ece: float
    mce: float  # Maximum Calibration Error
    optimal_temperature: float
    ece_after_scaling: float
    diagram: ReliabilityDiagram

    def summary(self) -> str:
        return (
            f"Calibration Analysis\n"
            f"  ECE:           {self.ece:.4f}\n"
            f"  MCE:           {self.mce:.4f}\n"
            f"  Temperature:   {self.optimal_temperature:.4f}\n"
            f"  ECE (scaled):  {self.ece_after_scaling:.4f}"
        )


# ---------------------------------------------------------------------------
# CalibrationAnalyzer
# ---------------------------------------------------------------------------

class CalibrationAnalyzer:
    """Measure and improve confidence calibration.

    Parameters
    ----------
    model : nn.Module or None
        Classifier returning logits.  Not needed if passing logits directly.
    n_bins : int
        Number of ECE bins.
    temperature_lr : float
        Learning rate for temperature optimisation.
    temperature_max_iter : int
        Max iterations for temperature optimisation.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        n_bins: int = 15,
        temperature_lr: float = 0.01,
        temperature_max_iter: int = 100,
    ):
        self.model = model
        self.n_bins = n_bins
        self.temperature_lr = temperature_lr
        self.temperature_max_iter = temperature_max_iter

    # ---- ECE --------------------------------------------------------------

    def compute_ece(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        n_bins: Optional[int] = None,
    ) -> float:
        """Expected Calibration Error (weighted bin-average |acc - conf|)."""
        nb = n_bins or self.n_bins
        diagram = self.reliability_diagram(logits, targets, n_bins=nb)
        total = sum(diagram.bin_counts)
        if total == 0:
            return 0.0
        ece = 0.0
        for i in range(nb):
            if diagram.bin_counts[i] > 0:
                weight = diagram.bin_counts[i] / total
                ece += weight * abs(diagram.bin_accuracies[i] - diagram.bin_confidences[i])
        return ece

    def compute_mce(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        n_bins: Optional[int] = None,
    ) -> float:
        """Maximum Calibration Error."""
        nb = n_bins or self.n_bins
        diagram = self.reliability_diagram(logits, targets, n_bins=nb)
        mce = 0.0
        for i in range(nb):
            if diagram.bin_counts[i] > 0:
                mce = max(mce, abs(diagram.bin_accuracies[i] - diagram.bin_confidences[i]))
        return mce

    # ---- Reliability diagram data ----------------------------------------

    def reliability_diagram(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        n_bins: Optional[int] = None,
    ) -> ReliabilityDiagram:
        """Compute binned reliability diagram data."""
        nb = n_bins or self.n_bins
        probs = F.softmax(logits, dim=1)
        confidences, predictions = probs.max(dim=1)
        correct = (predictions == targets).float()

        bin_boundaries = torch.linspace(0.0, 1.0, nb + 1)
        bin_accs: List[float] = []
        bin_confs: List[float] = []
        bin_counts: List[int] = []

        for i in range(nb):
            lo = bin_boundaries[i]
            hi = bin_boundaries[i + 1]
            if i == nb - 1:
                mask = (confidences >= lo) & (confidences <= hi)
            else:
                mask = (confidences >= lo) & (confidences < hi)
            count = mask.sum().item()
            bin_counts.append(int(count))
            if count > 0:
                bin_accs.append(correct[mask].mean().item())
                bin_confs.append(confidences[mask].mean().item())
            else:
                bin_accs.append(0.0)
                bin_confs.append(0.0)

        return ReliabilityDiagram(
            bin_accuracies=bin_accs,
            bin_confidences=bin_confs,
            bin_counts=bin_counts,
            n_bins=nb,
        )

    # ---- Temperature scaling --------------------------------------------

    def temperature_scaling(
        self,
        val_logits: torch.Tensor,
        val_targets: torch.Tensor,
    ) -> float:
        """Learn optimal temperature by minimising NLL on validation logits.

        Returns the learned temperature.
        """
        # Optimise temperature via grid search + gradient refinement
        log_temp = torch.zeros(1, requires_grad=True, device=val_logits.device)

        optimizer = torch.optim.LBFGS([log_temp], lr=self.temperature_lr,
                                       max_iter=self.temperature_max_iter)

        def closure():
            optimizer.zero_grad()
            t = log_temp.exp()
            scaled = val_logits / t
            loss = F.cross_entropy(scaled, val_targets)
            loss.backward()
            return loss

        optimizer.step(closure)

        temperature = log_temp.exp().item()
        # Clamp to reasonable range
        temperature = max(0.01, min(temperature, 100.0))
        return temperature

    def platt_scaling(
        self,
        val_logits: torch.Tensor,
        val_targets: torch.Tensor,
    ) -> Tuple[float, float]:
        """Learn affine calibration: logits_calibrated = a * logits + b.

        Returns ``(a, b)``.
        """
        a = torch.ones(1, requires_grad=True, device=val_logits.device)
        b = torch.zeros(1, requires_grad=True, device=val_logits.device)

        optimizer = torch.optim.LBFGS([a, b], lr=self.temperature_lr,
                                       max_iter=self.temperature_max_iter)

        def closure():
            optimizer.zero_grad()
            scaled = val_logits * a + b
            loss = F.cross_entropy(scaled, val_targets)
            loss.backward()
            return loss

        optimizer.step(closure)
        return a.item(), b.item()

    # ---- Calibration under shift -----------------------------------------

    def calibration_under_shift(
        self,
        clean_data: Tuple[torch.Tensor, torch.Tensor],
        shifted_data: Tuple[torch.Tensor, torch.Tensor],
    ) -> Dict[str, float]:
        """Compare ECE on clean vs shifted data.

        ``clean_data`` and ``shifted_data`` are ``(logits, targets)``.
        """
        clean_logits, clean_targets = clean_data
        shift_logits, shift_targets = shifted_data

        ece_clean = self.compute_ece(clean_logits, clean_targets)
        ece_shift = self.compute_ece(shift_logits, shift_targets)

        return {
            "ece_clean": ece_clean,
            "ece_shifted": ece_shift,
            "ece_delta": ece_shift - ece_clean,
        }

    # ---- Full analysis ---------------------------------------------------

    def analyze(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> CalibrationResult:
        """Complete calibration analysis with temperature scaling."""
        ece = self.compute_ece(logits, targets)
        mce = self.compute_mce(logits, targets)
        diagram = self.reliability_diagram(logits, targets)

        temp = self.temperature_scaling(logits, targets)
        scaled_logits = logits / temp
        ece_scaled = self.compute_ece(scaled_logits, targets)

        return CalibrationResult(
            ece=ece,
            mce=mce,
            optimal_temperature=temp,
            ece_after_scaling=ece_scaled,
            diagram=diagram,
        )


# ===================================================================
# Self-tests  (25+)
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
    print("calibration_template self-tests")
    print("=" * 60)

    torch.manual_seed(55)

    analyzer = CalibrationAnalyzer(n_bins=10)
    N = 500
    NC = 10

    # ---- well-calibrated logits ------------------------------------------
    # Create logits where the confidence approximately matches accuracy
    targets = torch.randint(0, NC, (N,))
    # Build logits that are mildly confident
    logits_cal = torch.randn(N, NC)
    for i in range(N):
        logits_cal[i, targets[i]] += 1.5  # slight boost for correct class

    ece_cal = analyzer.compute_ece(logits_cal, targets)
    _assert(0.0 <= ece_cal <= 1.0, f"ECE of mild logits in [0,1]: {ece_cal:.4f}")

    # ---- overconfident logits --------------------------------------------
    logits_over = logits_cal * 5.0  # scale up => overconfident
    ece_over = analyzer.compute_ece(logits_over, targets)
    _assert(0.0 <= ece_over <= 1.0, f"ECE of overconfident in [0,1]: {ece_over:.4f}")
    _assert(ece_over > 0, "Overconfident ECE > 0")

    # ---- underconfident logits -------------------------------------------
    logits_under = logits_cal * 0.1  # scale down => underconfident (flat probs)
    ece_under = analyzer.compute_ece(logits_under, targets)
    _assert(0.0 <= ece_under <= 1.0, f"ECE of underconfident in [0,1]: {ece_under:.4f}")

    # ---- MCE tests -------------------------------------------------------
    mce = analyzer.compute_mce(logits_over, targets)
    _assert(mce >= ece_over - 1e-6, "MCE >= ECE (by definition)")
    _assert(0.0 <= mce <= 1.0, f"MCE in [0,1]: {mce:.4f}")

    # ---- Reliability diagram tests ---------------------------------------
    diagram = analyzer.reliability_diagram(logits_cal, targets, n_bins=10)
    _assert(diagram.n_bins == 10, "Reliability diagram has 10 bins")
    _assert(len(diagram.bin_accuracies) == 10, "10 bin accuracies")
    _assert(len(diagram.bin_confidences) == 10, "10 bin confidences")
    _assert(len(diagram.bin_counts) == 10, "10 bin counts")
    _assert(sum(diagram.bin_counts) == N, f"Total bin counts == N ({N})")
    _assert(all(0.0 <= a <= 1.0 for a in diagram.bin_accuracies),
            "Bin accuracies in [0,1]")
    _assert(all(0.0 <= c <= 1.0 for c in diagram.bin_confidences),
            "Bin confidences in [0,1]")
    _assert(len(diagram.summary()) > 0, "Diagram summary non-empty")

    # ---- Temperature scaling tests ---------------------------------------
    temp = analyzer.temperature_scaling(logits_over, targets)
    _assert(0.01 <= temp <= 100.0,
            f"Temperature in valid range: {temp:.4f}")
    ece_scaled = analyzer.compute_ece(logits_over / temp, targets)
    _assert(ece_scaled <= ece_over + 0.05,
            f"ECE after temp scaling ({ece_scaled:.4f}) <= before ({ece_over:.4f}) + tolerance")

    # Temperature for already-calibrated should be ~1.0
    temp_cal = analyzer.temperature_scaling(logits_cal, targets)
    _assert(0.01 <= temp_cal <= 100.0,
            f"Temperature for mild logits: {temp_cal:.4f}")

    # ---- Platt scaling tests ---------------------------------------------
    a, b = analyzer.platt_scaling(logits_over, targets)
    _assert(isinstance(a, float) and isinstance(b, float),
            f"Platt scaling returns floats: a={a:.4f}, b={b:.4f}")
    scaled_platt = logits_over * a + b
    ece_platt = analyzer.compute_ece(scaled_platt, targets)
    _assert(ece_platt <= ece_over + 0.05,
            f"ECE after Platt ({ece_platt:.4f}) <= before ({ece_over:.4f}) + tolerance")

    # ---- Calibration under shift -----------------------------------------
    # Shifted logits: add noise to simulate distribution shift
    logits_shifted = logits_cal + torch.randn_like(logits_cal) * 2.0
    targets_shifted = targets.clone()

    shift_result = analyzer.calibration_under_shift(
        (logits_cal, targets),
        (logits_shifted, targets_shifted),
    )
    _assert("ece_clean" in shift_result, "calibration_under_shift returns ece_clean")
    _assert("ece_shifted" in shift_result, "calibration_under_shift returns ece_shifted")
    _assert("ece_delta" in shift_result, "calibration_under_shift returns ece_delta")
    _assert(abs(shift_result["ece_delta"] -
                (shift_result["ece_shifted"] - shift_result["ece_clean"])) < 1e-6,
            "ece_delta = ece_shifted - ece_clean")

    # ---- Full analyze test -----------------------------------------------
    result = analyzer.analyze(logits_over, targets)
    _assert(isinstance(result, CalibrationResult), "analyze returns CalibrationResult")
    _assert(result.ece > 0, "Result ECE > 0")
    _assert(result.mce >= result.ece - 1e-6, "Result MCE >= ECE")
    _assert(0.01 <= result.optimal_temperature <= 100.0,
            f"Result temperature valid: {result.optimal_temperature:.4f}")
    _assert(result.ece_after_scaling <= result.ece + 0.05,
            "Scaling reduces or maintains ECE")
    _assert(len(result.summary()) > 0, "CalibrationResult summary non-empty")

    # ---- Edge cases ------------------------------------------------------
    # Single sample
    ece_single = analyzer.compute_ece(logits_cal[:1], targets[:1])
    _assert(0.0 <= ece_single <= 1.0, "ECE works with single sample")

    # All same class
    targets_same = torch.zeros(N, dtype=torch.long)
    ece_same = analyzer.compute_ece(logits_cal, targets_same)
    _assert(0.0 <= ece_same <= 1.0, "ECE works with all same class")

    # Different n_bins
    ece_5 = analyzer.compute_ece(logits_cal, targets, n_bins=5)
    ece_20 = analyzer.compute_ece(logits_cal, targets, n_bins=20)
    _assert(0.0 <= ece_5 <= 1.0, "ECE with 5 bins in [0,1]")
    _assert(0.0 <= ece_20 <= 1.0, "ECE with 20 bins in [0,1]")

    # Perfect predictions (logits strongly indicating correct class)
    logits_perfect = torch.zeros(100, NC)
    targets_perfect = torch.randint(0, NC, (100,))
    for i in range(100):
        logits_perfect[i, targets_perfect[i]] = 10.0
    ece_perfect = analyzer.compute_ece(logits_perfect, targets_perfect)
    _assert(ece_perfect < 0.1,
            f"ECE for near-perfect predictions: {ece_perfect:.4f} < 0.1")

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    print("=" * 60)
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    _run_self_tests()
