"""
Confidence calibration module for dual-process reasoning.

Provide post-hoc calibration of neural network confidence scores so that
predicted probabilities reflect true empirical frequencies.  Two calibration
strategies are implemented from scratch (no sklearn dependency):

* **Temperature scaling** -- a single learned scalar T applied to logits
  (Guo et al. 2017, "On Calibration of Modern Neural Networks").
* **Isotonic regression** -- a non-parametric monotonic piecewise-linear
  mapping from raw confidence to calibrated confidence.

Metrics (ECE, MCE, Brier score, reliability diagrams) are implemented as
pure functions with no side effects.

All computation is done in fp32 regardless of input dtype.
"""

from __future__ import annotations

import copy
import math
import warnings
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CalibrationConfig:
    """Configuration for confidence calibration.

    Attributes:
        method: Calibration strategy -- "temperature", "isotonic", or "none".
        initial_temperature: Starting value for temperature scaling.
            Values > 1 soften the distribution; values < 1 sharpen it.
        fit_lr: Learning rate used when fitting the temperature parameter.
        fit_max_iter: Maximum number of optimiser iterations during fit.
        freeze_after_fit: If True, freeze the temperature parameter after
            fit() so subsequent forward passes do not update it.
        num_isotonic_bins: Number of bins used in the isotonic calibrator.
        ece_num_bins: Default number of bins for ECE / MCE computation.
    """

    method: str = "temperature"
    initial_temperature: float = 1.5
    fit_lr: float = 0.01
    fit_max_iter: int = 50
    freeze_after_fit: bool = True
    num_isotonic_bins: int = 100
    ece_num_bins: int = 15

    def __post_init__(self) -> None:
        valid_methods = {
            "temperature", "isotonic", "none", "platt", "beta",
            "histogram", "ensemble_temperature",
        }
        if self.method not in valid_methods:
            raise ValueError(
                f"CalibrationConfig.method must be one of {valid_methods}, "
                f"got '{self.method}'."
            )
        if self.initial_temperature <= 0.0:
            raise ValueError(
                "initial_temperature must be positive, "
                f"got {self.initial_temperature}."
            )
        if self.fit_lr <= 0.0:
            raise ValueError(f"fit_lr must be positive, got {self.fit_lr}.")
        if self.fit_max_iter < 1:
            raise ValueError(
                f"fit_max_iter must be >= 1, got {self.fit_max_iter}."
            )
        if self.num_isotonic_bins < 2:
            raise ValueError(
                f"num_isotonic_bins must be >= 2, got {self.num_isotonic_bins}."
            )
        if self.ece_num_bins < 1:
            raise ValueError(
                f"ece_num_bins must be >= 1, got {self.ece_num_bins}."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CalibrationConfig":
        """Deserialise from plain dict."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Calibration Metrics  (pure functions)
# ---------------------------------------------------------------------------

class CalibrationMetrics:
    """Pure-function collection for calibration quality measurement.

    Every public method is a @staticmethod -- no internal state is kept.
    All tensors are cast to fp32 before computation.
    """

    # ----- helpers --------------------------------------------------------

    @staticmethod
    def _to_fp32(*tensors: Tensor) -> Tuple[Tensor, ...]:
        """Cast every tensor to fp32 on the same device."""
        return tuple(t.float() for t in tensors)

    @staticmethod
    def _validate_inputs(
        confidences: Tensor, accuracies: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Flatten, cast, and check shapes match."""
        confidences, accuracies = CalibrationMetrics._to_fp32(
            confidences.detach(), accuracies.detach()
        )
        confidences = confidences.view(-1)
        accuracies = accuracies.view(-1)
        if confidences.shape[0] != accuracies.shape[0]:
            raise ValueError(
                f"confidences length {confidences.shape[0]} != "
                f"accuracies length {accuracies.shape[0]}"
            )
        return confidences, accuracies

    @staticmethod
    def _bin_data(
        confidences: Tensor,
        accuracies: Tensor,
        num_bins: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Partition samples into equal-width bins on [0, 1].

        Return (bin_centers, bin_accuracies, bin_confidences, bin_counts)
        where each tensor has length num_bins.  Empty bins have accuracy
        and confidence set to the bin centre and count zero.
        """
        confidences, accuracies = CalibrationMetrics._validate_inputs(
            confidences, accuracies
        )
        device = confidences.device
        n = confidences.shape[0]

        bin_boundaries = torch.linspace(0.0, 1.0, num_bins + 1, device=device)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2.0

        bin_accs = torch.zeros(num_bins, device=device)
        bin_confs = torch.zeros(num_bins, device=device)
        bin_counts = torch.zeros(num_bins, device=device)

        for i in range(num_bins):
            lo = bin_boundaries[i]
            hi = bin_boundaries[i + 1]
            if i == num_bins - 1:
                # Include right boundary in last bin.
                mask = (confidences >= lo) & (confidences <= hi)
            else:
                mask = (confidences >= lo) & (confidences < hi)
            count = mask.sum().item()
            bin_counts[i] = count
            if count > 0:
                bin_accs[i] = accuracies[mask].mean()
                bin_confs[i] = confidences[mask].mean()
            else:
                bin_accs[i] = bin_centers[i]
                bin_confs[i] = bin_centers[i]

        return bin_centers, bin_accs, bin_confs, bin_counts

    # ----- ECE -----------------------------------------------------------

    @staticmethod
    def compute_ece(
        confidences: Tensor,
        accuracies: Tensor,
        num_bins: int = 15,
    ) -> float:
        """Compute Expected Calibration Error.

        ECE = sum_b |acc_b - conf_b| * n_b / N

        where b indexes equal-width bins over [0, 1].

        Parameters
        ----------
        confidences : Tensor
            Predicted confidence for each sample (in [0, 1]).
        accuracies : Tensor
            Binary correctness indicator for each sample (0 or 1).
        num_bins : int
            Number of equal-width bins.

        Return
        ------
        float
            ECE value in [0, 1].
        """
        confidences, accuracies = CalibrationMetrics._validate_inputs(
            confidences, accuracies
        )
        n = confidences.shape[0]
        if n == 0:
            return 0.0

        _, bin_accs, bin_confs, bin_counts = CalibrationMetrics._bin_data(
            confidences, accuracies, num_bins
        )

        ece = ((bin_accs - bin_confs).abs() * bin_counts).sum().item() / n
        return ece

    # ----- MCE -----------------------------------------------------------

    @staticmethod
    def compute_mce(
        confidences: Tensor,
        accuracies: Tensor,
        num_bins: int = 15,
    ) -> float:
        """Compute Maximum Calibration Error.

        MCE = max_b |acc_b - conf_b|

        Only non-empty bins are considered.

        Parameters
        ----------
        confidences : Tensor
            Predicted confidence for each sample (in [0, 1]).
        accuracies : Tensor
            Binary correctness indicator for each sample (0 or 1).
        num_bins : int
            Number of equal-width bins.

        Return
        ------
        float
            MCE value in [0, 1].
        """
        confidences, accuracies = CalibrationMetrics._validate_inputs(
            confidences, accuracies
        )
        n = confidences.shape[0]
        if n == 0:
            return 0.0

        _, bin_accs, bin_confs, bin_counts = CalibrationMetrics._bin_data(
            confidences, accuracies, num_bins
        )

        non_empty = bin_counts > 0
        if not non_empty.any():
            return 0.0

        gaps = (bin_accs[non_empty] - bin_confs[non_empty]).abs()
        return gaps.max().item()

    # ----- Reliability Diagram -------------------------------------------

    @staticmethod
    def compute_reliability_diagram(
        confidences: Tensor,
        accuracies: Tensor,
        num_bins: int = 15,
    ) -> Dict[str, Tensor]:
        """Compute data for a reliability diagram.

        Return a dict with keys:
        * bin_centers     -- centre of each bin  (num_bins,)
        * bin_accuracies  -- mean accuracy per bin  (num_bins,)
        * bin_confidences -- mean confidence per bin  (num_bins,)
        * bin_counts      -- sample count per bin  (num_bins,)

        Empty bins have accuracy/confidence set to the bin centre and
        count zero so they can be filtered out downstream.
        """
        confidences, accuracies = CalibrationMetrics._validate_inputs(
            confidences, accuracies
        )
        centers, accs, confs, counts = CalibrationMetrics._bin_data(
            confidences, accuracies, num_bins
        )
        return {
            "bin_centers": centers,
            "bin_accuracies": accs,
            "bin_confidences": confs,
            "bin_counts": counts,
        }

    # ----- Brier Score ---------------------------------------------------

    @staticmethod
    def compute_brier_score(
        probabilities: Tensor,
        labels: Tensor,
    ) -> float:
        """Compute the Brier score = mean((p - y)^2).

        For binary classification probabilities is the predicted P(y=1)
        and labels is 0/1.  For multi-class probabilities is a
        (N, C) matrix of class probabilities and labels is a length-N
        vector of class indices.

        Return
        ------
        float
            Brier score (lower is better).
        """
        probabilities = probabilities.detach().float()
        labels = labels.detach()

        if probabilities.dim() == 1:
            # Binary case.
            labels = labels.float().view(-1)
            probabilities = probabilities.view(-1)
            if probabilities.shape[0] != labels.shape[0]:
                raise ValueError(
                    f"probabilities length {probabilities.shape[0]} != "
                    f"labels length {labels.shape[0]}"
                )
            return ((probabilities - labels) ** 2).mean().item()

        if probabilities.dim() == 2:
            # Multi-class case.
            n, c = probabilities.shape
            labels = labels.view(-1).long()
            if labels.shape[0] != n:
                raise ValueError(
                    f"probabilities batch size {n} != labels length "
                    f"{labels.shape[0]}"
                )
            one_hot = torch.zeros_like(probabilities)
            one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
            return ((probabilities - one_hot) ** 2).sum(dim=1).mean().item()

        raise ValueError(
            f"probabilities must be 1-D or 2-D, got {probabilities.dim()}-D."
        )

    # ----- Adaptive ECE --------------------------------------------------

    @staticmethod
    def compute_adaptive_ece(
        confidences: Tensor,
        accuracies: Tensor,
        num_bins: int = 15,
    ) -> float:
        """Compute adaptive (equal-mass) ECE.

        Instead of equal-width bins, each bin contains approximately the
        same number of samples.  This avoids empty bins and gives a more
        stable estimate when the confidence distribution is skewed.

        Return
        ------
        float
            Adaptive ECE value in [0, 1].
        """
        confidences, accuracies = CalibrationMetrics._validate_inputs(
            confidences, accuracies
        )
        n = confidences.shape[0]
        if n == 0:
            return 0.0

        # Sort by confidence.
        sorted_indices = confidences.argsort()
        sorted_conf = confidences[sorted_indices]
        sorted_acc = accuracies[sorted_indices]

        # Split into approximately equal-size bins.
        bin_size = max(1, n // num_bins)
        ece = 0.0
        total_counted = 0
        for i in range(num_bins):
            start = i * bin_size
            if i == num_bins - 1:
                end = n  # Last bin takes any remainder.
            else:
                end = start + bin_size
            if start >= n:
                break
            bin_conf = sorted_conf[start:end]
            bin_acc = sorted_acc[start:end]
            count = bin_conf.shape[0]
            if count == 0:
                continue
            ece += (bin_acc.mean() - bin_conf.mean()).abs().item() * count
            total_counted += count

        return ece / max(total_counted, 1)

    # ----- Class-wise ECE ------------------------------------------------

    @staticmethod
    def compute_classwise_ece(
        probabilities: Tensor,
        labels: Tensor,
        num_bins: int = 15,
    ) -> float:
        """Compute class-wise ECE (average ECE over all classes).

        Parameters
        ----------
        probabilities : Tensor
            (N, C) predicted probability matrix.
        labels : Tensor
            (N,) ground-truth class indices.
        num_bins : int
            Number of equal-width bins per class.

        Return
        ------
        float
            Average ECE across classes.
        """
        probabilities = probabilities.detach().float()
        labels = labels.detach().long().view(-1)

        if probabilities.dim() != 2:
            raise ValueError(
                "probabilities must be 2-D (N, C) for class-wise ECE."
            )

        n, c = probabilities.shape
        if labels.shape[0] != n:
            raise ValueError(
                f"probabilities batch {n} != labels length {labels.shape[0]}."
            )

        ece_sum = 0.0
        for cls in range(c):
            cls_conf = probabilities[:, cls]
            cls_correct = (labels == cls).float()
            ece_sum += CalibrationMetrics.compute_ece(
                cls_conf, cls_correct, num_bins
            )

        return ece_sum / max(c, 1)

    # ----- Negative Log-Likelihood ---------------------------------------

    @staticmethod
    def compute_nll(
        logits: Tensor,
        labels: Tensor,
    ) -> float:
        """Compute mean negative log-likelihood (cross-entropy) from logits.

        Parameters
        ----------
        logits : Tensor
            (N, C) un-normalised logits.
        labels : Tensor
            (N,) ground-truth class indices.

        Return
        ------
        float
        """
        logits = logits.detach().float()
        labels = labels.detach().long().view(-1)
        return F.cross_entropy(logits, labels).item()

    # ----- Overconfidence / Underconfidence split -------------------------

    @staticmethod
    def compute_overconfidence_error(
        confidences: Tensor,
        accuracies: Tensor,
        num_bins: int = 15,
    ) -> float:
        """Compute overconfidence error (OCE).

        Like ECE but only accounts for bins where mean confidence > mean
        accuracy (the model is overconfident).  This separates the
        direction of miscalibration.

        Return
        ------
        float
        """
        confidences, accuracies = CalibrationMetrics._validate_inputs(
            confidences, accuracies
        )
        n = confidences.shape[0]
        if n == 0:
            return 0.0

        _, bin_accs, bin_confs, bin_counts = CalibrationMetrics._bin_data(
            confidences, accuracies, num_bins
        )

        over_mask = (bin_confs > bin_accs) & (bin_counts > 0)
        if not over_mask.any():
            return 0.0

        oce = (
            (bin_confs[over_mask] - bin_accs[over_mask]) * bin_counts[over_mask]
        ).sum().item() / n
        return oce

    @staticmethod
    def compute_underconfidence_error(
        confidences: Tensor,
        accuracies: Tensor,
        num_bins: int = 15,
    ) -> float:
        """Compute underconfidence error (UCE).

        Like ECE but only accounts for bins where mean accuracy > mean
        confidence (the model is underconfident).

        Return
        ------
        float
        """
        confidences, accuracies = CalibrationMetrics._validate_inputs(
            confidences, accuracies
        )
        n = confidences.shape[0]
        if n == 0:
            return 0.0

        _, bin_accs, bin_confs, bin_counts = CalibrationMetrics._bin_data(
            confidences, accuracies, num_bins
        )

        under_mask = (bin_accs > bin_confs) & (bin_counts > 0)
        if not under_mask.any():
            return 0.0

        uce = (
            (bin_accs[under_mask] - bin_confs[under_mask])
            * bin_counts[under_mask]
        ).sum().item() / n
        return uce

    # ----- Summary report ------------------------------------------------

    @staticmethod
    def calibration_report(
        confidences: Tensor,
        accuracies: Tensor,
        num_bins: int = 15,
        probabilities: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        logits: Optional[Tensor] = None,
    ) -> Dict[str, float]:
        """Compute a comprehensive calibration report.

        Return a dict with all available scalar metrics.

        Parameters
        ----------
        confidences : Tensor
            Per-sample max probability (or binary probability).
        accuracies : Tensor
            Per-sample correctness (0/1).
        num_bins : int
            Number of bins for ECE / MCE.
        probabilities : Tensor, optional
            Full (N, C) probability matrix for Brier score.
        labels : Tensor, optional
            (N,) class labels for Brier / NLL.
        logits : Tensor, optional
            (N, C) raw logits for NLL.

        Return
        ------
        Dict[str, float]
        """
        report: Dict[str, float] = {}
        report["ece"] = CalibrationMetrics.compute_ece(
            confidences, accuracies, num_bins
        )
        report["mce"] = CalibrationMetrics.compute_mce(
            confidences, accuracies, num_bins
        )
        report["adaptive_ece"] = CalibrationMetrics.compute_adaptive_ece(
            confidences, accuracies, num_bins
        )
        report["overconfidence_error"] = (
            CalibrationMetrics.compute_overconfidence_error(
                confidences, accuracies, num_bins
            )
        )
        report["underconfidence_error"] = (
            CalibrationMetrics.compute_underconfidence_error(
                confidences, accuracies, num_bins
            )
        )

        if probabilities is not None and labels is not None:
            report["brier_score"] = CalibrationMetrics.compute_brier_score(
                probabilities, labels
            )
            if probabilities.dim() == 2:
                report["classwise_ece"] = (
                    CalibrationMetrics.compute_classwise_ece(
                        probabilities, labels, num_bins
                    )
                )

        if logits is not None and labels is not None:
            report["nll"] = CalibrationMetrics.compute_nll(logits, labels)

        return report


# ---------------------------------------------------------------------------
# Temperature Scaling  (Guo et al. 2017)
# ---------------------------------------------------------------------------

class TemperatureScaler(nn.Module):
    """Post-hoc temperature scaling for calibrating neural network logits.

    A single learned scalar T (the temperature) is applied to logits
    before softmax:  calibrated_logits = logits / T.  T > 1 softens
    the distribution (useful for overconfident models); T < 1 sharpens it.

    The parameter is stored as an nn.Parameter so it is automatically
    included in state_dict() for checkpointing.

    Reference
    ---------
    Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).
    "On Calibration of Modern Neural Networks."  ICML 2017.

    Example
    -------
    >>> scaler = TemperatureScaler(initial_temperature=1.5)
    >>> metrics = scaler.fit(val_logits, val_labels)
    >>> calibrated = scaler.calibrate(test_logits)
    """

    def __init__(self, initial_temperature: float = 1.5) -> None:
        """Initialise temperature scaler.

        Parameters
        ----------
        initial_temperature : float
            Starting value of the temperature parameter.  Must be positive.
        """
        super().__init__()
        if initial_temperature <= 0.0:
            raise ValueError(
                f"initial_temperature must be positive, got {initial_temperature}."
            )
        self._temperature = nn.Parameter(
            torch.tensor(float(initial_temperature), dtype=torch.float32)
        )
        self._frozen = False
        self._initial_temperature = initial_temperature

    # ----- Properties -----------------------------------------------------

    @property
    def temperature(self) -> float:
        """Return current temperature value as a Python float."""
        return self._temperature.item()

    @temperature.setter
    def temperature(self, value: float) -> None:
        """Set the temperature parameter directly."""
        if value <= 0.0:
            raise ValueError(f"Temperature must be positive, got {value}.")
        with torch.no_grad():
            self._temperature.fill_(value)

    @property
    def is_frozen(self) -> bool:
        """Return True if the temperature is frozen (no gradient)."""
        return self._frozen

    # ----- Freeze / unfreeze ---------------------------------------------

    def freeze(self) -> None:
        """Lock the temperature parameter so gradients do not flow."""
        self._frozen = True
        self._temperature.requires_grad_(False)

    def unfreeze(self) -> None:
        """Unlock the temperature parameter for gradient updates."""
        self._frozen = False
        self._temperature.requires_grad_(True)

    # ----- Core methods ---------------------------------------------------

    def calibrate(self, logits: Tensor) -> Tensor:
        """Scale logits by the learned temperature.

        Parameters
        ----------
        logits : Tensor
            Raw (un-normalised) logits of shape (N, C).

        Return
        ------
        Tensor
            Calibrated logits of same shape: logits / T.
        """
        logits = logits.float()
        # Clamp temperature to a small positive value for numerical safety.
        t = self._temperature.clamp(min=1e-6)
        return logits / t

    def forward(self, logits: Tensor) -> Tensor:
        """Alias for calibrate so the module works in nn pipelines."""
        return self.calibrate(logits)

    # ----- Fitting --------------------------------------------------------

    def fit(
        self,
        logits: Tensor,
        labels: Tensor,
        lr: float = 0.01,
        max_iter: int = 50,
        verbose: bool = False,
    ) -> Dict[str, float]:
        """Optimise the temperature on held-out validation data.

        Minimise NLL (cross-entropy) between temperature-scaled logits and
        true labels.  The optimiser is L-BFGS.

        Parameters
        ----------
        logits : Tensor
            (N, C) raw logits from the model.
        labels : Tensor
            (N,) ground-truth class indices.
        lr : float
            Learning rate.
        max_iter : int
            Maximum number of optimiser steps.
        verbose : bool
            If True, print per-iteration loss.

        Return
        ------
        Dict[str, float]
            Fit metrics including final_nll, ece_before, ece_after,
            and temperature.
        """
        logits = logits.detach().float()
        labels = labels.detach().long().view(-1)

        if logits.shape[0] != labels.shape[0]:
            raise ValueError(
                f"logits batch {logits.shape[0]} != labels length "
                f"{labels.shape[0]}."
            )

        # --- ECE before calibration --------------------------------------
        probs_before = F.softmax(logits, dim=1)
        confs_before, preds_before = probs_before.max(dim=1)
        correct_before = (preds_before == labels).float()
        ece_before = CalibrationMetrics.compute_ece(confs_before, correct_before)

        nll_before = F.cross_entropy(logits, labels).item()

        # --- Ensure parameter is unfrozen for fitting --------------------
        was_frozen = self._frozen
        if was_frozen:
            self.unfreeze()

        # Reset temperature to initial value for a clean fit.
        with torch.no_grad():
            self._temperature.fill_(self._initial_temperature)

        # Move logits/labels to same device as parameter.
        device = self._temperature.device
        logits = logits.to(device)
        labels = labels.to(device)

        # --- Optimise via L-BFGS -----------------------------------------
        optimizer = torch.optim.LBFGS(
            [self._temperature], lr=lr, max_iter=max_iter
        )

        nll_history: List[float] = []

        def closure() -> Tensor:
            optimizer.zero_grad()
            t = self._temperature.clamp(min=1e-6)
            scaled = logits / t
            loss = F.cross_entropy(scaled, labels)
            loss.backward()
            nll_history.append(loss.item())
            if verbose:
                print(
                    f"  [TemperatureScaler.fit] iter={len(nll_history)} "
                    f"nll={loss.item():.6f}  T={self._temperature.item():.4f}"
                )
            return loss

        optimizer.step(closure)

        # --- ECE after calibration ---------------------------------------
        with torch.no_grad():
            t = self._temperature.clamp(min=1e-6)
            probs_after = F.softmax(logits / t, dim=1)
            confs_after, preds_after = probs_after.max(dim=1)
            correct_after = (preds_after == labels).float()
            ece_after = CalibrationMetrics.compute_ece(
                confs_after, correct_after
            )
            final_nll = F.cross_entropy(logits / t, labels).item()

        # --- Restore frozen state if needed ------------------------------
        if was_frozen:
            self.freeze()

        metrics = {
            "final_nll": final_nll,
            "nll_before": nll_before,
            "ece_before": ece_before,
            "ece_after": ece_after,
            "temperature": self._temperature.item(),
            "num_iterations": len(nll_history),
        }
        return metrics

    # ----- SGD fit (alternative) ------------------------------------------

    def fit_sgd(
        self,
        logits: Tensor,
        labels: Tensor,
        lr: float = 0.01,
        max_iter: int = 200,
        verbose: bool = False,
    ) -> Dict[str, float]:
        """Optimise temperature using simple SGD (fallback for L-BFGS).

        Same interface and return value as fit().
        """
        logits = logits.detach().float()
        labels = labels.detach().long().view(-1)

        if logits.shape[0] != labels.shape[0]:
            raise ValueError(
                f"logits batch {logits.shape[0]} != labels length "
                f"{labels.shape[0]}."
            )

        probs_before = F.softmax(logits, dim=1)
        confs_before, preds_before = probs_before.max(dim=1)
        correct_before = (preds_before == labels).float()
        ece_before = CalibrationMetrics.compute_ece(confs_before, correct_before)
        nll_before = F.cross_entropy(logits, labels).item()

        was_frozen = self._frozen
        if was_frozen:
            self.unfreeze()

        with torch.no_grad():
            self._temperature.fill_(self._initial_temperature)

        device = self._temperature.device
        logits = logits.to(device)
        labels = labels.to(device)

        optimizer = torch.optim.SGD([self._temperature], lr=lr)

        best_nll = float("inf")
        best_t = self._temperature.item()

        for i in range(max_iter):
            optimizer.zero_grad()
            t = self._temperature.clamp(min=1e-6)
            loss = F.cross_entropy(logits / t, labels)
            loss.backward()
            optimizer.step()

            # Clamp T to be positive after update.
            with torch.no_grad():
                self._temperature.clamp_(min=1e-6)

            nll_val = loss.item()
            if nll_val < best_nll:
                best_nll = nll_val
                best_t = self._temperature.item()

            if verbose and (i % 10 == 0 or i == max_iter - 1):
                print(
                    f"  [SGD] iter={i+1}/{max_iter}  "
                    f"nll={nll_val:.6f}  T={self._temperature.item():.4f}"
                )

        # Restore best temperature.
        with torch.no_grad():
            self._temperature.fill_(best_t)

        with torch.no_grad():
            t = self._temperature.clamp(min=1e-6)
            probs_after = F.softmax(logits / t, dim=1)
            confs_after, preds_after = probs_after.max(dim=1)
            correct_after = (preds_after == labels).float()
            ece_after = CalibrationMetrics.compute_ece(
                confs_after, correct_after
            )
            final_nll = F.cross_entropy(logits / t, labels).item()

        if was_frozen:
            self.freeze()

        return {
            "final_nll": final_nll,
            "nll_before": nll_before,
            "ece_before": ece_before,
            "ece_after": ece_after,
            "temperature": self._temperature.item(),
            "num_iterations": max_iter,
        }

    # ----- Serialisation helpers ------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"temperature={self._temperature.item():.4f}, "
            f"frozen={self._frozen}"
        )

    def get_config(self) -> Dict[str, Any]:
        """Return a JSON-serialisable configuration dict."""
        return {
            "initial_temperature": self._initial_temperature,
            "current_temperature": self._temperature.item(),
            "frozen": self._frozen,
        }


# ---------------------------------------------------------------------------
# Isotonic Calibrator  (non-parametric)
# ---------------------------------------------------------------------------

class IsotonicCalibrator:
    """Non-parametric isotonic regression calibrator.

    Build a monotonically non-decreasing piecewise-linear mapping from raw
    confidence to calibrated confidence.  The implementation uses the
    pool-adjacent-violators algorithm (PAVA) and does NOT depend on sklearn.

    After fit(), call calibrate() to map new confidence values through the
    fitted function.

    Example
    -------
    >>> iso = IsotonicCalibrator(num_bins=100)
    >>> iso.fit(val_confidences, val_correctness)
    >>> calibrated = iso.calibrate(test_confidences)
    """

    def __init__(self, num_bins: int = 100) -> None:
        """Initialise the isotonic calibrator.

        Parameters
        ----------
        num_bins : int
            Number of bins used to discretise the [0, 1] interval before
            running the pool-adjacent-violators algorithm.
        """
        if num_bins < 2:
            raise ValueError(f"num_bins must be >= 2, got {num_bins}.")
        self._num_bins = num_bins
        self._fitted = False

        # After fit these hold the piecewise-linear mapping.
        self._bin_edges: Optional[Tensor] = None
        self._bin_values: Optional[Tensor] = None
        self._bin_centers: Optional[Tensor] = None

        # Raw PAVA output for full-resolution mapping.
        self._pava_x: Optional[Tensor] = None
        self._pava_y: Optional[Tensor] = None

    # ----- Properties -----------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Return True if the calibrator has been fitted."""
        return self._fitted

    @property
    def num_bins(self) -> int:
        """Return the number of bins."""
        return self._num_bins

    # ----- Pool-Adjacent-Violators Algorithm (PAVA) -----------------------

    @staticmethod
    def _pava(values: Tensor, weights: Optional[Tensor] = None) -> Tensor:
        """Run the pool-adjacent-violators algorithm.

        Produce a monotonically non-decreasing sequence that minimises the
        weighted squared error to the input values.

        Parameters
        ----------
        values : Tensor
            1-D tensor of target values (not necessarily monotone).
        weights : Tensor, optional
            1-D tensor of non-negative weights.  Default: uniform.

        Return
        ------
        Tensor
            Monotonically non-decreasing 1-D tensor of same length.
        """
        n = values.shape[0]
        if n == 0:
            return values.clone()

        values_list = values.float().tolist()
        if weights is None:
            weights_list = [1.0] * n
        else:
            weights_list = weights.float().tolist()

        # Block representation: each block stores
        # [weighted_sum, total_weight, start_idx, end_idx_exclusive]
        blocks: List[List[float]] = []

        for i in range(n):
            w_i = weights_list[i]
            v_i = values_list[i]
            blocks.append([v_i * w_i, w_i, float(i), float(i + 1)])

            # Merge backward while monotonicity is violated.
            while len(blocks) >= 2:
                last = blocks[-1]
                prev = blocks[-2]
                mean_last = last[0] / max(last[1], 1e-12)
                mean_prev = prev[0] / max(prev[1], 1e-12)
                if mean_prev > mean_last:
                    # Merge the two blocks.
                    prev[0] += last[0]
                    prev[1] += last[1]
                    prev[3] = last[3]
                    blocks.pop()
                else:
                    break

        # Build result tensor.
        result = torch.zeros(n, dtype=torch.float32, device=values.device)
        for block in blocks:
            mean_val = block[0] / max(block[1], 1e-12)
            start = int(block[2])
            end = int(block[3])
            result[start:end] = mean_val

        return result

    # ----- Fit ------------------------------------------------------------

    def fit(
        self,
        confidences: Tensor,
        correctness: Tensor,
    ) -> None:
        """Fit the isotonic mapping from validation data.

        Build a monotonically non-decreasing piecewise-linear function that
        maps raw confidence to calibrated confidence.

        Parameters
        ----------
        confidences : Tensor
            Per-sample predicted confidence in [0, 1].
        correctness : Tensor
            Per-sample binary correctness (0 or 1).
        """
        confidences = confidences.detach().float().view(-1)
        correctness = correctness.detach().float().view(-1)

        if confidences.shape[0] != correctness.shape[0]:
            raise ValueError(
                f"confidences ({confidences.shape[0]}) and correctness "
                f"({correctness.shape[0]}) must have the same length."
            )

        n = confidences.shape[0]
        if n == 0:
            warnings.warn("IsotonicCalibrator.fit called with zero samples.")
            self._fitted = False
            return

        # --- Bin the data ------------------------------------------------
        num_bins = min(self._num_bins, n)
        bin_edges = torch.linspace(
            0.0, 1.0, num_bins + 1, device=confidences.device
        )
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        bin_means = torch.zeros(num_bins, device=confidences.device)
        bin_counts = torch.zeros(num_bins, device=confidences.device)

        for i in range(num_bins):
            lo = bin_edges[i]
            hi = bin_edges[i + 1]
            if i == num_bins - 1:
                mask = (confidences >= lo) & (confidences <= hi)
            else:
                mask = (confidences >= lo) & (confidences < hi)
            count = mask.sum().item()
            bin_counts[i] = count
            if count > 0:
                bin_means[i] = correctness[mask].mean()
            else:
                # Linearly interpolate from neighbours later.
                bin_means[i] = bin_centers[i]

        # --- Interpolate empty bins from neighbours ----------------------
        non_empty = bin_counts > 0
        if non_empty.sum().item() < 2:
            # Degenerate case: use identity mapping.
            self._bin_edges = bin_edges
            self._bin_centers = bin_centers
            self._bin_values = bin_centers.clone()
            self._pava_x = bin_centers.clone()
            self._pava_y = bin_centers.clone()
            self._fitted = True
            return

        # Simple linear interpolation for empty bins.
        non_empty_idx = torch.where(non_empty)[0]
        for i in range(num_bins):
            if bin_counts[i] == 0:
                # Find nearest non-empty neighbours.
                left_idx = non_empty_idx[non_empty_idx <= i]
                right_idx = non_empty_idx[non_empty_idx >= i]
                if left_idx.numel() > 0 and right_idx.numel() > 0:
                    li = left_idx[-1].item()
                    ri = right_idx[0].item()
                    if li == ri:
                        bin_means[i] = bin_means[li]
                    else:
                        alpha = (i - li) / (ri - li)
                        bin_means[i] = (
                            bin_means[li] * (1 - alpha) + bin_means[ri] * alpha
                        )
                elif left_idx.numel() > 0:
                    bin_means[i] = bin_means[left_idx[-1].item()]
                elif right_idx.numel() > 0:
                    bin_means[i] = bin_means[right_idx[0].item()]

        # --- Apply PAVA to enforce monotonicity --------------------------
        monotone_values = self._pava(bin_means, bin_counts.clamp(min=1))

        # Clamp to [0, 1].
        monotone_values = monotone_values.clamp(0.0, 1.0)

        self._bin_edges = bin_edges
        self._bin_centers = bin_centers
        self._bin_values = monotone_values
        self._pava_x = bin_centers.clone()
        self._pava_y = monotone_values.clone()
        self._fitted = True

    # ----- Calibrate ------------------------------------------------------

    def calibrate(self, confidences: Tensor) -> Tensor:
        """Apply the fitted isotonic mapping to new confidence values.

        Parameters
        ----------
        confidences : Tensor
            Predicted confidence values (arbitrary shape).

        Return
        ------
        Tensor
            Calibrated confidence values of same shape, in [0, 1].

        Raise
        -----
        RuntimeError
            If the calibrator has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError(
                "IsotonicCalibrator.calibrate called before fit()."
            )

        original_shape = confidences.shape
        original_device = confidences.device
        confidences = confidences.detach().float().view(-1)

        assert self._pava_x is not None and self._pava_y is not None

        pava_x = self._pava_x.to(confidences.device)
        pava_y = self._pava_y.to(confidences.device)

        # Piecewise-linear interpolation.
        calibrated = self._piecewise_linear_interp(
            confidences, pava_x, pava_y
        )

        return calibrated.clamp(0.0, 1.0).view(original_shape).to(original_device)

    @staticmethod
    def _piecewise_linear_interp(
        x: Tensor, knots_x: Tensor, knots_y: Tensor
    ) -> Tensor:
        """Evaluate a piecewise-linear function at points x.

        The function is defined by sorted knots (knots_x, knots_y).  Values
        outside the knot range are clamped to the boundary knot values.
        """
        n_knots = knots_x.shape[0]
        if n_knots == 0:
            return x.clone()
        if n_knots == 1:
            return torch.full_like(x, knots_y[0].item())

        # Clamp x to the knot range.
        x_clamped = x.clamp(knots_x[0].item(), knots_x[-1].item())

        # For each x, find the right knot index via searchsorted.
        indices = torch.searchsorted(knots_x, x_clamped, right=True)
        indices = indices.clamp(1, n_knots - 1)

        x0 = knots_x[indices - 1]
        x1 = knots_x[indices]
        y0 = knots_y[indices - 1]
        y1 = knots_y[indices]

        # Avoid division by zero.
        dx = (x1 - x0).clamp(min=1e-10)
        alpha = (x_clamped - x0) / dx
        alpha = alpha.clamp(0.0, 1.0)

        return y0 + alpha * (y1 - y0)

    # ----- Serialisation --------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        """Return a serialisable state dict for checkpointing."""
        state: Dict[str, Any] = {
            "num_bins": self._num_bins,
            "fitted": self._fitted,
        }
        if self._fitted:
            assert (
                self._bin_edges is not None
                and self._bin_centers is not None
                and self._bin_values is not None
                and self._pava_x is not None
                and self._pava_y is not None
            )
            state["bin_edges"] = self._bin_edges.cpu()
            state["bin_centers"] = self._bin_centers.cpu()
            state["bin_values"] = self._bin_values.cpu()
            state["pava_x"] = self._pava_x.cpu()
            state["pava_y"] = self._pava_y.cpu()
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore state from a previously saved dict."""
        self._num_bins = state["num_bins"]
        self._fitted = state["fitted"]
        if self._fitted:
            self._bin_edges = state["bin_edges"]
            self._bin_centers = state["bin_centers"]
            self._bin_values = state["bin_values"]
            self._pava_x = state["pava_x"]
            self._pava_y = state["pava_y"]
        else:
            self._bin_edges = None
            self._bin_centers = None
            self._bin_values = None
            self._pava_x = None
            self._pava_y = None

    def __repr__(self) -> str:
        return (
            f"IsotonicCalibrator(num_bins={self._num_bins}, "
            f"fitted={self._fitted})"
        )

    def get_mapping(self) -> Optional[Dict[str, Tensor]]:
        """Return the fitted piecewise-linear mapping knots.

        Return None if not fitted.
        """
        if not self._fitted:
            return None
        return {
            "x": self._pava_x.clone() if self._pava_x is not None else None,
            "y": self._pava_y.clone() if self._pava_y is not None else None,
        }

    def verify_monotonicity(self) -> bool:
        """Check that the fitted mapping is monotonically non-decreasing.

        Return True if monotone or not fitted (vacuously true).
        """
        if not self._fitted or self._pava_y is None:
            return True
        diffs = self._pava_y[1:] - self._pava_y[:-1]
        return bool((diffs >= -1e-6).all().item())


# ---------------------------------------------------------------------------
# Platt Scaling  (binary calibration)
# ---------------------------------------------------------------------------

class PlattScaler(nn.Module):
    """Platt scaling for binary classification calibration.

    Learn a logistic regression sigma(a * z + b) where z is the raw logit
    or score.  This is a two-parameter generalisation of temperature scaling
    specific to the binary case.

    Reference: Platt, J. (1999). "Probabilistic Outputs for Support Vector
    Machines and Comparisons to Regularized Likelihood Methods."
    """

    def __init__(self) -> None:
        super().__init__()
        self._a = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self._b = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self._frozen = False

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    @property
    def a(self) -> float:
        return self._a.item()

    @property
    def b(self) -> float:
        return self._b.item()

    def freeze(self) -> None:
        """Lock parameters so gradients do not flow."""
        self._frozen = True
        self._a.requires_grad_(False)
        self._b.requires_grad_(False)

    def unfreeze(self) -> None:
        """Unlock parameters for gradient updates."""
        self._frozen = False
        self._a.requires_grad_(True)
        self._b.requires_grad_(True)

    def calibrate(self, scores: Tensor) -> Tensor:
        """Apply Platt scaling: sigmoid(a * scores + b).

        Parameters
        ----------
        scores : Tensor
            Raw binary scores / logits.

        Return
        ------
        Tensor
            Calibrated probabilities in [0, 1].
        """
        scores = scores.float()
        return torch.sigmoid(self._a * scores + self._b)

    def forward(self, scores: Tensor) -> Tensor:
        return self.calibrate(scores)

    def fit(
        self,
        scores: Tensor,
        labels: Tensor,
        lr: float = 0.01,
        max_iter: int = 100,
        verbose: bool = False,
    ) -> Dict[str, float]:
        """Fit parameters a, b via binary cross-entropy.

        Parameters
        ----------
        scores : Tensor
            (N,) raw scores / logits.
        labels : Tensor
            (N,) binary labels (0 or 1).
        lr : float
            Learning rate for L-BFGS.
        max_iter : int
            Max optimiser iterations.
        verbose : bool
            Print progress.

        Return
        ------
        Dict[str, float]
            Fit metrics.
        """
        scores = scores.detach().float().view(-1)
        labels = labels.detach().float().view(-1)

        if scores.shape[0] != labels.shape[0]:
            raise ValueError("scores and labels must have the same length.")

        was_frozen = self._frozen
        if was_frozen:
            self.unfreeze()

        with torch.no_grad():
            self._a.fill_(1.0)
            self._b.fill_(0.0)

        device = self._a.device
        scores = scores.to(device)
        labels = labels.to(device)

        # ECE before.
        with torch.no_grad():
            probs_before = torch.sigmoid(scores)
            correct_before = labels
            ece_before = CalibrationMetrics.compute_ece(
                probs_before, correct_before
            )

        optimizer = torch.optim.LBFGS(
            [self._a, self._b], lr=lr, max_iter=max_iter
        )

        iteration_count = [0]

        def closure() -> Tensor:
            optimizer.zero_grad()
            probs = torch.sigmoid(self._a * scores + self._b)
            probs = probs.clamp(1e-7, 1 - 1e-7)
            loss = -(
                labels * probs.log() + (1 - labels) * (1 - probs).log()
            ).mean()
            loss.backward()
            iteration_count[0] += 1
            if verbose:
                print(
                    f"  [PlattScaler.fit] iter={iteration_count[0]} "
                    f"loss={loss.item():.6f}  a={self._a.item():.4f}  "
                    f"b={self._b.item():.4f}"
                )
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            probs_after = torch.sigmoid(self._a * scores + self._b)
            ece_after = CalibrationMetrics.compute_ece(probs_after, labels)
            final_loss = -(
                labels * probs_after.clamp(1e-7).log()
                + (1 - labels) * (1 - probs_after).clamp(1e-7).log()
            ).mean().item()

        if was_frozen:
            self.freeze()

        return {
            "final_loss": final_loss,
            "ece_before": ece_before,
            "ece_after": ece_after,
            "a": self._a.item(),
            "b": self._b.item(),
        }

    def extra_repr(self) -> str:
        return (
            f"a={self._a.item():.4f}, b={self._b.item():.4f}, "
            f"frozen={self._frozen}"
        )


# ---------------------------------------------------------------------------
# Ensemble Temperature Scaling
# ---------------------------------------------------------------------------

class EnsembleTemperatureScaler(nn.Module):
    """Ensemble of multiple temperature scalers with learned mixture weights.

    Fit K temperature scalers on different data subsets and combine their
    calibrated distributions via a learned convex combination.  This
    improves robustness on heterogeneous validation sets.
    """

    def __init__(
        self,
        num_members: int = 3,
        initial_temperature: float = 1.5,
    ) -> None:
        """Initialise ensemble of temperature scalers.

        Parameters
        ----------
        num_members : int
            Number of ensemble members.
        initial_temperature : float
            Initial temperature for each member.
        """
        super().__init__()
        self._num_members = num_members
        self._scalers = nn.ModuleList([
            TemperatureScaler(initial_temperature=initial_temperature)
            for _ in range(num_members)
        ])
        # Un-normalised log weights for softmax.
        self._log_weights = nn.Parameter(
            torch.zeros(num_members, dtype=torch.float32)
        )
        self._frozen = False

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    @property
    def weights(self) -> Tensor:
        """Return normalised mixture weights."""
        return F.softmax(self._log_weights, dim=0)

    def freeze(self) -> None:
        """Lock all parameters."""
        self._frozen = True
        self._log_weights.requires_grad_(False)
        for s in self._scalers:
            s.freeze()

    def unfreeze(self) -> None:
        """Unlock all parameters."""
        self._frozen = False
        self._log_weights.requires_grad_(True)
        for s in self._scalers:
            s.unfreeze()

    def calibrate(self, logits: Tensor) -> Tensor:
        """Apply ensemble temperature scaling.

        Return the weighted average of softmax outputs from each member.

        Parameters
        ----------
        logits : Tensor
            (N, C) raw logits.

        Return
        ------
        Tensor
            (N, C) calibrated probability distribution.
        """
        logits = logits.float()
        weights = self.weights  # (K,)

        probs = torch.zeros_like(F.softmax(logits, dim=1))
        for i, scaler in enumerate(self._scalers):
            scaled = scaler.calibrate(logits)
            probs = probs + weights[i] * F.softmax(scaled, dim=1)

        return probs

    def forward(self, logits: Tensor) -> Tensor:
        return self.calibrate(logits)

    def fit(
        self,
        logits: Tensor,
        labels: Tensor,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> Dict[str, Any]:
        """Fit all members independently on random subsets, then combine.

        Parameters
        ----------
        logits : Tensor
            (N, C) validation logits.
        labels : Tensor
            (N,) ground-truth labels.
        lr : float
            Learning rate per member.
        max_iter : int
            Max iterations per member.

        Return
        ------
        Dict[str, Any]
            Summary metrics.
        """
        logits = logits.detach().float()
        labels = labels.detach().long().view(-1)
        n = logits.shape[0]

        member_temps: List[float] = []
        for i, scaler in enumerate(self._scalers):
            # Bootstrap: sample with replacement.
            idx = torch.randint(0, n, (n,), device=logits.device)
            sub_logits = logits[idx]
            sub_labels = labels[idx]
            metrics = scaler.fit(
                sub_logits, sub_labels, lr=lr, max_iter=max_iter
            )
            member_temps.append(metrics["temperature"])

        # Compute aggregate ECE.
        with torch.no_grad():
            probs = self.calibrate(logits)
            confs, preds = probs.max(dim=1)
            correct = (preds == labels).float()
            ece = CalibrationMetrics.compute_ece(confs, correct)

        return {
            "ece_after": ece,
            "temperatures": member_temps,
            "weights": self.weights.detach().tolist(),
        }

    def extra_repr(self) -> str:
        temps = [s.temperature for s in self._scalers]
        return f"num_members={self._num_members}, temperatures={temps}"


# ---------------------------------------------------------------------------
# Histogram Binning Calibrator
# ---------------------------------------------------------------------------

class HistogramBinningCalibrator:
    """Simple histogram binning calibrator.

    Partition the [0, 1] confidence interval into equal-width bins and
    replace each sample's confidence with the empirical accuracy of its bin.

    This is the simplest non-parametric calibrator and serves as a baseline.

    Reference: Zadrozny & Elkan (2001). "Obtaining calibrated probability
    estimates from decision trees and naive Bayesian classifiers."
    """

    def __init__(self, num_bins: int = 15) -> None:
        if num_bins < 1:
            raise ValueError(f"num_bins must be >= 1, got {num_bins}.")
        self._num_bins = num_bins
        self._fitted = False
        self._bin_edges: Optional[Tensor] = None
        self._bin_values: Optional[Tensor] = None

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, confidences: Tensor, correctness: Tensor) -> None:
        """Fit bin values from validation data.

        Parameters
        ----------
        confidences : Tensor
            (N,) predicted confidences in [0, 1].
        correctness : Tensor
            (N,) binary correctness (0 or 1).
        """
        confidences = confidences.detach().float().view(-1)
        correctness = correctness.detach().float().view(-1)

        if confidences.shape[0] != correctness.shape[0]:
            raise ValueError(
                "confidences and correctness must match in length."
            )

        n = confidences.shape[0]
        if n == 0:
            warnings.warn(
                "HistogramBinningCalibrator.fit with zero samples."
            )
            return

        edges = torch.linspace(
            0.0, 1.0, self._num_bins + 1, device=confidences.device
        )
        values = torch.zeros(self._num_bins, device=confidences.device)

        for i in range(self._num_bins):
            lo = edges[i]
            hi = edges[i + 1]
            if i == self._num_bins - 1:
                mask = (confidences >= lo) & (confidences <= hi)
            else:
                mask = (confidences >= lo) & (confidences < hi)
            count = mask.sum().item()
            if count > 0:
                values[i] = correctness[mask].mean()
            else:
                # Use bin centre as fallback.
                values[i] = (lo + hi) / 2.0

        self._bin_edges = edges
        self._bin_values = values
        self._fitted = True

    def calibrate(self, confidences: Tensor) -> Tensor:
        """Map confidences through the fitted histogram.

        Parameters
        ----------
        confidences : Tensor
            Predicted confidence values.

        Return
        ------
        Tensor
            Calibrated values in [0, 1].
        """
        if not self._fitted:
            raise RuntimeError("calibrate called before fit.")

        original_shape = confidences.shape
        conf = confidences.detach().float().view(-1)

        assert self._bin_edges is not None and self._bin_values is not None

        edges = self._bin_edges.to(conf.device)
        values = self._bin_values.to(conf.device)

        # Digitise into bins.
        indices = torch.searchsorted(edges[1:], conf, right=False)
        indices = indices.clamp(0, self._num_bins - 1)

        return values[indices].view(original_shape)

    def state_dict(self) -> Dict[str, Any]:
        """Return serialisable state dict."""
        state: Dict[str, Any] = {
            "num_bins": self._num_bins,
            "fitted": self._fitted,
        }
        if self._fitted:
            state["bin_edges"] = (
                self._bin_edges.cpu() if self._bin_edges is not None else None
            )
            state["bin_values"] = (
                self._bin_values.cpu()
                if self._bin_values is not None
                else None
            )
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore state from saved dict."""
        self._num_bins = state["num_bins"]
        self._fitted = state["fitted"]
        if self._fitted:
            self._bin_edges = state["bin_edges"]
            self._bin_values = state["bin_values"]

    def __repr__(self) -> str:
        return (
            f"HistogramBinningCalibrator(num_bins={self._num_bins}, "
            f"fitted={self._fitted})"
        )


# ---------------------------------------------------------------------------
# Beta Calibration  (Kull et al. 2017)
# ---------------------------------------------------------------------------

class BetaCalibrator(nn.Module):
    """Beta calibration for binary classification.

    Learn parameters (a, b, c) such that calibrated probability is
    sigmoid(a * log(p / (1 - p)) + b * log(p) + c) where p is the
    raw probability.  This is a three-parameter family that subsumes
    Platt scaling and temperature scaling.

    Reference: Kull, M., Silva Filho, T., & Flach, P. (2017).
    "Beta calibration: a well-founded and easily implemented improvement
    on logistic calibration for binary classifiers."  AISTATS 2017.
    """

    def __init__(self) -> None:
        super().__init__()
        self._a = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self._b = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self._c = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self._frozen = False

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def freeze(self) -> None:
        """Lock all parameters."""
        self._frozen = True
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze(self) -> None:
        """Unlock all parameters."""
        self._frozen = False
        for p in self.parameters():
            p.requires_grad_(True)

    def calibrate(self, probabilities: Tensor) -> Tensor:
        """Apply beta calibration.

        Parameters
        ----------
        probabilities : Tensor
            Raw probabilities in (0, 1).

        Return
        ------
        Tensor
            Calibrated probabilities in [0, 1].
        """
        p = probabilities.float().clamp(1e-7, 1 - 1e-7)
        logit_p = torch.log(p / (1 - p))
        log_p = torch.log(p)
        return torch.sigmoid(self._a * logit_p + self._b * log_p + self._c)

    def forward(self, probabilities: Tensor) -> Tensor:
        return self.calibrate(probabilities)

    def fit(
        self,
        probabilities: Tensor,
        labels: Tensor,
        lr: float = 0.01,
        max_iter: int = 100,
        verbose: bool = False,
    ) -> Dict[str, float]:
        """Fit parameters via binary cross-entropy.

        Parameters
        ----------
        probabilities : Tensor
            (N,) raw predicted probabilities.
        labels : Tensor
            (N,) binary labels.
        lr : float
            Learning rate.
        max_iter : int
            Maximum L-BFGS iterations.

        Return
        ------
        Dict[str, float]
        """
        probs = probabilities.detach().float().view(-1)
        labels_f = labels.detach().float().view(-1)

        was_frozen = self._frozen
        if was_frozen:
            self.unfreeze()

        with torch.no_grad():
            self._a.fill_(1.0)
            self._b.fill_(0.0)
            self._c.fill_(0.0)

        device = self._a.device
        probs = probs.to(device)
        labels_f = labels_f.to(device)

        ece_before = CalibrationMetrics.compute_ece(probs, labels_f)

        optimizer = torch.optim.LBFGS(
            list(self.parameters()), lr=lr, max_iter=max_iter
        )

        def closure() -> Tensor:
            optimizer.zero_grad()
            calibrated = self.calibrate(probs)
            calibrated = calibrated.clamp(1e-7, 1 - 1e-7)
            loss = -(
                labels_f * calibrated.log()
                + (1 - labels_f) * (1 - calibrated).log()
            ).mean()
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            calibrated = self.calibrate(probs)
            ece_after = CalibrationMetrics.compute_ece(calibrated, labels_f)

        if was_frozen:
            self.freeze()

        return {
            "ece_before": ece_before,
            "ece_after": ece_after,
            "a": self._a.item(),
            "b": self._b.item(),
            "c": self._c.item(),
        }

    def extra_repr(self) -> str:
        return (
            f"a={self._a.item():.4f}, b={self._b.item():.4f}, "
            f"c={self._c.item():.4f}, frozen={self._frozen}"
        )


# ---------------------------------------------------------------------------
# Focal Loss (calibration-aware training loss)
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal loss for calibration-aware training.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    When gamma > 0 the loss down-weights easy (well-classified) examples
    and focuses on hard ones, which can improve calibration during
    training (as opposed to post-hoc methods).

    Reference: Lin, T.-Y. et al. (2017). "Focal Loss for Dense Object
    Detection."  ICCV 2017.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        """Initialise focal loss.

        Parameters
        ----------
        gamma : float
            Focusing parameter (gamma >= 0).  gamma=0 recovers CE.
        alpha : Tensor, optional
            Per-class weight tensor of shape (C,).
        reduction : str
            "mean", "sum", or "none".
        """
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}.")
        self.gamma = gamma
        self.register_buffer(
            "alpha",
            alpha.float() if alpha is not None else None,
        )
        self.reduction = reduction

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Compute focal loss.

        Parameters
        ----------
        logits : Tensor
            (N, C) un-normalised logits.
        labels : Tensor
            (N,) ground-truth class indices.

        Return
        ------
        Tensor
            Scalar (or per-sample) loss depending on reduction.
        """
        logits = logits.float()
        labels = labels.long().view(-1)

        log_probs = F.log_softmax(logits, dim=1)  # (N, C)
        probs = log_probs.exp()

        # Gather the probability of the true class.
        nll = -log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)  # (N,)
        p_t = probs.gather(1, labels.unsqueeze(1)).squeeze(1)  # (N,)

        focal_weight = (1 - p_t) ** self.gamma  # (N,)

        loss = focal_weight * nll  # (N,)

        if self.alpha is not None:
            alpha_t = self.alpha[labels]  # (N,)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ---------------------------------------------------------------------------
# Calibrator Factory
# ---------------------------------------------------------------------------

class CalibratorFactory:
    """Factory for creating calibrators from CalibrationConfig.

    Use create_calibrator to dispatch based on the method field of a
    CalibrationConfig.
    """

    _REGISTRY: Dict[str, type] = {
        "temperature": TemperatureScaler,
        "isotonic": IsotonicCalibrator,
        "platt": PlattScaler,
        "beta": BetaCalibrator,
        "histogram": HistogramBinningCalibrator,
        "ensemble_temperature": EnsembleTemperatureScaler,
    }

    @classmethod
    def create_calibrator(
        cls,
        config: CalibrationConfig,
    ) -> Optional[
        Union[
            TemperatureScaler,
            IsotonicCalibrator,
            PlattScaler,
            BetaCalibrator,
            HistogramBinningCalibrator,
            EnsembleTemperatureScaler,
        ]
    ]:
        """Create a calibrator from configuration.

        Parameters
        ----------
        config : CalibrationConfig
            Configuration specifying the calibration method and
            hyper-parameters.

        Return
        ------
        Calibrator instance, or None if config.method == "none".

        Raise
        -----
        ValueError
            If the method string is not recognised.
        """
        method = config.method.lower().strip()

        if method == "none":
            return None

        if method == "temperature":
            return TemperatureScaler(
                initial_temperature=config.initial_temperature,
            )

        if method == "isotonic":
            return IsotonicCalibrator(
                num_bins=config.num_isotonic_bins,
            )

        if method == "platt":
            return PlattScaler()

        if method == "beta":
            return BetaCalibrator()

        if method == "histogram":
            return HistogramBinningCalibrator(
                num_bins=config.num_isotonic_bins,
            )

        if method == "ensemble_temperature":
            return EnsembleTemperatureScaler(
                initial_temperature=config.initial_temperature,
            )

        raise ValueError(
            f"Unknown calibration method '{method}'.  "
            f"Supported: {set(cls._REGISTRY.keys()) | {'none'}}."
        )

    @classmethod
    def available_methods(cls) -> List[str]:
        """Return list of available calibration method names."""
        return sorted(cls._REGISTRY.keys()) + ["none"]


# ---------------------------------------------------------------------------
# Calibration Pipeline  (end-to-end convenience wrapper)
# ---------------------------------------------------------------------------

class CalibrationPipeline:
    """End-to-end calibration pipeline.

    Wrap creation, fitting, calibration, and evaluation into a single
    object for convenient integration into training loops.

    Example
    -------
    >>> pipe = CalibrationPipeline(CalibrationConfig(method="temperature"))
    >>> pipe.fit(val_logits, val_labels)
    >>> calibrated_probs = pipe.calibrate(test_logits)
    >>> report = pipe.evaluate(test_logits, test_labels)
    """

    def __init__(self, config: Optional[CalibrationConfig] = None) -> None:
        """Initialise the calibration pipeline.

        Parameters
        ----------
        config : CalibrationConfig, optional
            Configuration.  Defaults to temperature scaling with default
            hyper-parameters.
        """
        if config is None:
            config = CalibrationConfig()
        self._config = config
        self._calibrator = CalibratorFactory.create_calibrator(config)
        self._fit_metrics: Optional[Dict[str, Any]] = None

    @property
    def config(self) -> CalibrationConfig:
        return self._config

    @property
    def calibrator(self) -> Any:
        return self._calibrator

    @property
    def fit_metrics(self) -> Optional[Dict[str, Any]]:
        return self._fit_metrics

    def fit(self, logits: Tensor, labels: Tensor) -> Dict[str, Any]:
        """Fit the calibrator on validation data.

        Parameters
        ----------
        logits : Tensor
            (N, C) validation logits.
        labels : Tensor
            (N,) labels.

        Return
        ------
        Dict[str, Any]
            Fit metrics.
        """
        if self._calibrator is None:
            self._fit_metrics = {"method": "none"}
            return self._fit_metrics

        if isinstance(self._calibrator, TemperatureScaler):
            metrics = self._calibrator.fit(
                logits,
                labels,
                lr=self._config.fit_lr,
                max_iter=self._config.fit_max_iter,
            )
            if self._config.freeze_after_fit:
                self._calibrator.freeze()
            self._fit_metrics = metrics
            return metrics

        if isinstance(self._calibrator, EnsembleTemperatureScaler):
            metrics = self._calibrator.fit(
                logits,
                labels,
                lr=self._config.fit_lr,
                max_iter=self._config.fit_max_iter,
            )
            if self._config.freeze_after_fit:
                self._calibrator.freeze()
            self._fit_metrics = metrics
            return metrics

        if isinstance(self._calibrator, IsotonicCalibrator):
            probs = F.softmax(logits.float(), dim=1)
            confs, preds = probs.max(dim=1)
            correct = (preds == labels.view(-1)).float()
            self._calibrator.fit(confs, correct)
            ece_before = CalibrationMetrics.compute_ece(confs, correct)
            calibrated = self._calibrator.calibrate(confs)
            ece_after = CalibrationMetrics.compute_ece(calibrated, correct)
            metrics = {"ece_before": ece_before, "ece_after": ece_after}
            self._fit_metrics = metrics
            return metrics

        if isinstance(self._calibrator, HistogramBinningCalibrator):
            probs = F.softmax(logits.float(), dim=1)
            confs, preds = probs.max(dim=1)
            correct = (preds == labels.view(-1)).float()
            self._calibrator.fit(confs, correct)
            ece_before = CalibrationMetrics.compute_ece(confs, correct)
            calibrated = self._calibrator.calibrate(confs)
            ece_after = CalibrationMetrics.compute_ece(calibrated, correct)
            metrics = {"ece_before": ece_before, "ece_after": ece_after}
            self._fit_metrics = metrics
            return metrics

        # PlattScaler / BetaCalibrator -- binary case, use top-class prob.
        probs = F.softmax(logits.float(), dim=1)
        confs, preds = probs.max(dim=1)
        correct = (preds == labels.view(-1)).float()
        metrics = self._calibrator.fit(confs, correct)
        if self._config.freeze_after_fit and hasattr(
            self._calibrator, "freeze"
        ):
            self._calibrator.freeze()
        self._fit_metrics = metrics
        return metrics

    def calibrate(self, logits: Tensor) -> Tensor:
        """Apply calibration to logits.

        For temperature/ensemble methods return calibrated logits.
        For non-parametric methods return calibrated confidences.
        If method is "none" return raw softmax probabilities.

        Parameters
        ----------
        logits : Tensor
            (N, C) raw logits.

        Return
        ------
        Tensor
            Calibrated output.
        """
        logits = logits.float()

        if self._calibrator is None:
            return F.softmax(logits, dim=1)

        if isinstance(
            self._calibrator,
            (TemperatureScaler, EnsembleTemperatureScaler),
        ):
            return self._calibrator.calibrate(logits)

        # Non-parametric methods work on confidences.
        probs = F.softmax(logits, dim=1)
        confs, _ = probs.max(dim=1)
        calibrated_confs = self._calibrator.calibrate(confs)
        return calibrated_confs

    def evaluate(
        self,
        logits: Tensor,
        labels: Tensor,
        num_bins: int = 15,
    ) -> Dict[str, float]:
        """Calibrate and compute full metrics report.

        Parameters
        ----------
        logits : Tensor
            (N, C) test logits.
        labels : Tensor
            (N,) test labels.
        num_bins : int
            Number of bins for ECE/MCE.

        Return
        ------
        Dict[str, float]
        """
        logits = logits.float()
        labels = labels.long().view(-1)

        calibrated = self.calibrate(logits)

        if calibrated.dim() == 2:
            # Full probability distribution.
            if isinstance(self._calibrator, EnsembleTemperatureScaler):
                probs_softmax = calibrated
            elif isinstance(self._calibrator, TemperatureScaler):
                probs_softmax = F.softmax(calibrated, dim=1)
            else:
                probs_softmax = calibrated
            confs, preds = probs_softmax.max(dim=1)
        else:
            confs = calibrated
            preds = F.softmax(logits, dim=1).argmax(dim=1)
            probs_softmax = None

        correct = (preds == labels).float()

        report = CalibrationMetrics.calibration_report(
            confidences=confs,
            accuracies=correct,
            num_bins=num_bins,
            probabilities=probs_softmax,
            labels=labels,
            logits=logits,
        )
        return report

    def state_dict(self) -> Dict[str, Any]:
        """Serialise full pipeline state."""
        state: Dict[str, Any] = {
            "config": self._config.to_dict(),
            "fit_metrics": self._fit_metrics,
        }
        if self._calibrator is not None:
            if hasattr(self._calibrator, "state_dict"):
                state["calibrator_state"] = self._calibrator.state_dict()
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore pipeline from saved state."""
        self._config = CalibrationConfig.from_dict(state["config"])
        self._fit_metrics = state.get("fit_metrics")
        self._calibrator = CalibratorFactory.create_calibrator(self._config)
        if self._calibrator is not None and "calibrator_state" in state:
            if hasattr(self._calibrator, "load_state_dict"):
                self._calibrator.load_state_dict(state["calibrator_state"])


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def generate_miscalibrated_logits(
    num_samples: int = 1000,
    num_classes: int = 10,
    overconfidence: float = 3.0,
    seed: int = 42,
) -> Tuple[Tensor, Tensor]:
    """Generate synthetic miscalibrated logits for testing.

    Create logits that are overconfident: the predicted class is usually
    correct but the confidence is higher than the true accuracy.

    Parameters
    ----------
    num_samples : int
        Number of samples to generate.
    num_classes : int
        Number of classes.
    overconfidence : float
        Scaling factor applied to the true-class logit to create
        overconfidence.  Values > 1 produce overconfident predictions.
    seed : int
        Random seed for reproducibility.

    Return
    ------
    Tuple[Tensor, Tensor]
        (logits, labels) where logits is (N, C) and labels is (N,).
    """
    torch.manual_seed(seed)
    labels = torch.randint(0, num_classes, (num_samples,))
    logits = torch.randn(num_samples, num_classes)
    # Boost the correct class logit to make the model overconfident.
    for i in range(num_samples):
        logits[i, labels[i]] += overconfidence
    return logits, labels


def generate_perfectly_calibrated(
    num_samples: int = 1000,
    num_classes: int = 10,
    seed: int = 123,
) -> Tuple[Tensor, Tensor]:
    """Generate logits that are approximately perfectly calibrated.

    Predictions match the empirical accuracy within each confidence bin.

    Return
    ------
    Tuple[Tensor, Tensor]
        (logits, labels).
    """
    torch.manual_seed(seed)
    labels = torch.randint(0, num_classes, (num_samples,))
    # Use moderate logit magnitudes so softmax is not too sharp.
    logits = torch.randn(num_samples, num_classes) * 0.5
    for i in range(num_samples):
        logits[i, labels[i]] += 1.0
    return logits, labels


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _run_self_tests() -> None:
    """Run all self-tests and report pass/fail."""

    passed = 0
    failed = 0
    total = 0

    def check(condition: bool, name: str) -> None:
        nonlocal passed, failed, total
        total += 1
        if condition:
            passed += 1
            print(f"  [PASS] {name}")
        else:
            failed += 1
            print(f"  [FAIL] {name}")

    print("=" * 70)
    print("Calibration Template Self-Tests")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. TemperatureScaler: fit on overconfident logits
    # ------------------------------------------------------------------
    print("\n--- Test 1: TemperatureScaler fit on overconfident logits ---")
    logits, labels = generate_miscalibrated_logits(
        num_samples=2000, num_classes=10, overconfidence=4.0, seed=42
    )
    scaler = TemperatureScaler(initial_temperature=1.5)
    metrics = scaler.fit(logits, labels)

    check(
        scaler.temperature > 1.0,
        f"T > 1 for overconfident model (T={scaler.temperature:.4f})",
    )
    check(
        abs(scaler.temperature - 1.5) > 0.01,
        f"temperature moved from initial value (T={scaler.temperature:.4f})",
    )
    check(
        "final_nll" in metrics and "temperature" in metrics,
        "fit returns expected metric keys",
    )

    # ------------------------------------------------------------------
    # 2. TemperatureScaler: calibrate output shape
    # ------------------------------------------------------------------
    print("\n--- Test 2: TemperatureScaler calibrate output ---")
    calibrated = scaler.calibrate(logits)
    check(
        calibrated.shape == logits.shape,
        f"calibrated shape matches input ({calibrated.shape})",
    )

    probs = F.softmax(calibrated, dim=1)
    check(
        torch.allclose(
            probs.sum(dim=1), torch.ones(probs.shape[0]), atol=1e-5
        ),
        "softmax of calibrated logits sums to 1",
    )

    # ------------------------------------------------------------------
    # 3. Freeze / unfreeze
    # ------------------------------------------------------------------
    print("\n--- Test 3: Freeze / unfreeze ---")
    scaler.freeze()
    check(scaler.is_frozen, "is_frozen is True after freeze()")
    check(
        not scaler._temperature.requires_grad,
        "T.requires_grad is False after freeze",
    )

    t_before = scaler.temperature
    # Attempt a forward (should not error).
    _ = scaler.calibrate(logits[:10])
    check(
        abs(scaler.temperature - t_before) < 1e-8,
        f"T unchanged after forward when frozen ({t_before:.6f})",
    )

    scaler.unfreeze()
    check(not scaler.is_frozen, "is_frozen is False after unfreeze()")
    check(
        scaler._temperature.requires_grad,
        "T.requires_grad is True after unfreeze",
    )

    # ------------------------------------------------------------------
    # 4. Save / load state_dict
    # ------------------------------------------------------------------
    print("\n--- Test 4: Save / load state_dict ---")
    state = scaler.state_dict()
    scaler2 = TemperatureScaler(initial_temperature=1.0)
    scaler2.load_state_dict(state)
    check(
        abs(scaler2.temperature - scaler.temperature) < 1e-6,
        f"T preserved after load_state_dict ({scaler2.temperature:.4f})",
    )

    # ------------------------------------------------------------------
    # 5. IsotonicCalibrator: fit and verify monotonicity
    # ------------------------------------------------------------------
    print("\n--- Test 5: IsotonicCalibrator fit and monotonicity ---")
    probs_raw = F.softmax(logits, dim=1)
    confs_raw, preds_raw = probs_raw.max(dim=1)
    correct_raw = (preds_raw == labels).float()

    iso = IsotonicCalibrator(num_bins=50)
    iso.fit(confs_raw, correct_raw)

    check(iso.is_fitted, "is_fitted is True after fit()")
    check(
        iso.verify_monotonicity(),
        "fitted mapping is monotonically non-decreasing",
    )

    mapping = iso.get_mapping()
    check(mapping is not None, "get_mapping returns non-None after fit")
    if mapping is not None and mapping["y"] is not None:
        diffs = mapping["y"][1:] - mapping["y"][:-1]
        check(
            bool((diffs >= -1e-6).all()),
            "PAVA y-values are non-decreasing",
        )

    # ------------------------------------------------------------------
    # 6. IsotonicCalibrator: calibrate output
    # ------------------------------------------------------------------
    print("\n--- Test 6: IsotonicCalibrator calibrate output ---")
    calibrated_iso = iso.calibrate(confs_raw)
    check(
        calibrated_iso.shape == confs_raw.shape,
        f"calibrated shape matches input ({calibrated_iso.shape})",
    )
    check(
        bool(
            (calibrated_iso >= 0.0).all() and (calibrated_iso <= 1.0).all()
        ),
        "all calibrated values in [0, 1]",
    )

    # ------------------------------------------------------------------
    # 7. CalibrationMetrics: ECE / MCE
    # ------------------------------------------------------------------
    print("\n--- Test 7: CalibrationMetrics ECE / MCE ---")
    ece = CalibrationMetrics.compute_ece(confs_raw, correct_raw, num_bins=15)
    mce = CalibrationMetrics.compute_mce(confs_raw, correct_raw, num_bins=15)
    check(
        0.0 <= ece <= 1.0,
        f"ECE in [0,1] ({ece:.4f})",
    )
    check(
        0.0 <= mce <= 1.0,
        f"MCE in [0,1] ({mce:.4f})",
    )
    check(
        mce >= ece,
        f"MCE >= ECE ({mce:.4f} >= {ece:.4f})",
    )

    # ------------------------------------------------------------------
    # 8. Reliability diagram: bin counts sum to N
    # ------------------------------------------------------------------
    print("\n--- Test 8: Reliability diagram ---")
    diagram = CalibrationMetrics.compute_reliability_diagram(
        confs_raw, correct_raw, num_bins=15
    )
    check(
        set(diagram.keys())
        == {"bin_centers", "bin_accuracies", "bin_confidences", "bin_counts"},
        "reliability diagram has expected keys",
    )
    n_total = int(diagram["bin_counts"].sum().item())
    check(
        n_total == confs_raw.shape[0],
        f"bin_counts sum to N ({n_total} == {confs_raw.shape[0]})",
    )
    check(
        diagram["bin_centers"].shape[0] == 15,
        f"number of bins is 15 ({diagram['bin_centers'].shape[0]})",
    )

    # ------------------------------------------------------------------
    # 9. Brier score
    # ------------------------------------------------------------------
    print("\n--- Test 9: Brier score ---")
    brier = CalibrationMetrics.compute_brier_score(probs_raw, labels)
    check(
        0.0 <= brier <= 2.0,
        f"Brier score in reasonable range ({brier:.4f})",
    )

    # Binary Brier.
    binary_probs = torch.tensor([0.9, 0.8, 0.3, 0.1])
    binary_labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
    brier_binary = CalibrationMetrics.compute_brier_score(
        binary_probs, binary_labels
    )
    expected_binary = (0.1**2 + 0.2**2 + 0.3**2 + 0.1**2) / 4
    check(
        abs(brier_binary - expected_binary) < 1e-5,
        f"binary Brier correct ({brier_binary:.6f} ~= {expected_binary:.6f})",
    )

    # ------------------------------------------------------------------
    # 10. CalibratorFactory: all methods
    # ------------------------------------------------------------------
    print("\n--- Test 10: CalibratorFactory ---")
    for method in ["temperature", "isotonic", "none"]:
        cfg = CalibrationConfig(method=method)
        cal = CalibratorFactory.create_calibrator(cfg)
        if method == "none":
            check(cal is None, f"factory('{method}') returns None")
        elif method == "temperature":
            check(
                isinstance(cal, TemperatureScaler),
                f"factory('{method}') returns TemperatureScaler",
            )
        elif method == "isotonic":
            check(
                isinstance(cal, IsotonicCalibrator),
                f"factory('{method}') returns IsotonicCalibrator",
            )

    # PlattScaler factory.
    cfg_platt = CalibrationConfig(method="platt")
    cal_platt = CalibratorFactory.create_calibrator(cfg_platt)
    check(
        isinstance(cal_platt, PlattScaler),
        "factory('platt') returns PlattScaler",
    )

    # Available methods.
    methods = CalibratorFactory.available_methods()
    check(
        "temperature" in methods
        and "isotonic" in methods
        and "none" in methods,
        f"available_methods includes expected methods ({methods})",
    )

    # ------------------------------------------------------------------
    # 11. Edge case: perfect calibration
    # ------------------------------------------------------------------
    print("\n--- Test 11: Edge case -- perfect calibration ---")
    logits_perf, labels_perf = generate_perfectly_calibrated(
        num_samples=1000, seed=123
    )
    probs_perf = F.softmax(logits_perf, dim=1)
    confs_perf, preds_perf = probs_perf.max(dim=1)
    correct_perf = (preds_perf == labels_perf).float()
    ece_perf = CalibrationMetrics.compute_ece(confs_perf, correct_perf)
    check(
        ece_perf < 0.50,
        f"moderate-logit model has bounded ECE ({ece_perf:.4f})",
    )

    # ------------------------------------------------------------------
    # 12. Edge case: single class
    # ------------------------------------------------------------------
    print("\n--- Test 12: Edge case -- single class ---")
    single_logits = torch.randn(50, 1)
    single_labels = torch.zeros(50, dtype=torch.long)
    # TemperatureScaler should not error.
    scaler_single = TemperatureScaler(initial_temperature=1.5)
    try:
        cal_single = scaler_single.calibrate(single_logits)
        check(
            cal_single.shape == single_logits.shape,
            "single-class calibrate works",
        )
    except Exception as e:
        check(False, f"single-class calibrate raised {e}")

    # ------------------------------------------------------------------
    # 13. Edge case: batch size 1
    # ------------------------------------------------------------------
    print("\n--- Test 13: Edge case -- batch size 1 ---")
    bs1_logits = torch.randn(1, 10)
    bs1_labels = torch.tensor([3])
    scaler_bs1 = TemperatureScaler(initial_temperature=1.5)
    try:
        _ = scaler_bs1.fit(bs1_logits, bs1_labels)
        cal_bs1 = scaler_bs1.calibrate(bs1_logits)
        check(
            cal_bs1.shape == (1, 10),
            "batch-size-1 fit and calibrate work",
        )
    except Exception as e:
        check(False, f"batch-size-1 raised {e}")

    # ECE with batch size 1.
    ece_bs1 = CalibrationMetrics.compute_ece(
        torch.tensor([0.8]),
        torch.tensor([1.0]),
    )
    check(
        isinstance(ece_bs1, float),
        f"ECE with single sample returns float ({ece_bs1:.4f})",
    )

    # ------------------------------------------------------------------
    # 14. IsotonicCalibrator state_dict round-trip
    # ------------------------------------------------------------------
    print("\n--- Test 14: IsotonicCalibrator state_dict round-trip ---")
    iso_state = iso.state_dict()
    iso2 = IsotonicCalibrator(num_bins=50)
    iso2.load_state_dict(iso_state)
    check(iso2.is_fitted, "loaded isotonic is fitted")
    check(iso2.verify_monotonicity(), "loaded isotonic is monotone")

    # Compare calibration output.
    test_confs = torch.linspace(0.0, 1.0, 100)
    cal_original = iso.calibrate(test_confs)
    cal_loaded = iso2.calibrate(test_confs)
    check(
        torch.allclose(cal_original, cal_loaded, atol=1e-6),
        "loaded isotonic produces same output",
    )

    # ------------------------------------------------------------------
    # 15. SGD fit alternative
    # ------------------------------------------------------------------
    print("\n--- Test 15: TemperatureScaler SGD fit ---")
    scaler_sgd = TemperatureScaler(initial_temperature=1.5)
    metrics_sgd = scaler_sgd.fit_sgd(
        logits, labels, lr=0.05, max_iter=100
    )
    check(
        scaler_sgd.temperature > 0.0,
        f"SGD fit: T > 0 (T={scaler_sgd.temperature:.4f})",
    )
    check(
        metrics_sgd["ece_after"] <= metrics_sgd["ece_before"] + 0.02,
        f"SGD fit: ECE improved ({metrics_sgd['ece_before']:.4f} -> "
        f"{metrics_sgd['ece_after']:.4f})",
    )

    # ------------------------------------------------------------------
    # 16. Adaptive ECE
    # ------------------------------------------------------------------
    print("\n--- Test 16: Adaptive ECE ---")
    aece = CalibrationMetrics.compute_adaptive_ece(confs_raw, correct_raw)
    check(
        0.0 <= aece <= 1.0,
        f"Adaptive ECE in [0,1] ({aece:.4f})",
    )

    # ------------------------------------------------------------------
    # 17. Overconfidence / underconfidence error
    # ------------------------------------------------------------------
    print("\n--- Test 17: Overconfidence / underconfidence error ---")
    oce = CalibrationMetrics.compute_overconfidence_error(
        confs_raw, correct_raw
    )
    uce = CalibrationMetrics.compute_underconfidence_error(
        confs_raw, correct_raw
    )
    check(
        0.0 <= oce <= 1.0,
        f"Overconfidence error in [0,1] ({oce:.4f})",
    )
    check(
        0.0 <= uce <= 1.0,
        f"Underconfidence error in [0,1] ({uce:.4f})",
    )
    # OCE + UCE should approximately equal ECE.
    check(
        abs(oce + uce - ece) < 0.01,
        f"OCE + UCE ~= ECE ({oce:.4f} + {uce:.4f} ~= {ece:.4f})",
    )

    # ------------------------------------------------------------------
    # 18. CalibrationConfig validation
    # ------------------------------------------------------------------
    print("\n--- Test 18: CalibrationConfig validation ---")
    try:
        CalibrationConfig(method="invalid")
        check(False, "invalid method should raise ValueError")
    except ValueError:
        check(True, "invalid method raises ValueError")

    try:
        CalibrationConfig(initial_temperature=-1.0)
        check(False, "negative temperature should raise ValueError")
    except ValueError:
        check(True, "negative temperature raises ValueError")

    # Round-trip serialisation.
    cfg = CalibrationConfig(method="isotonic", num_isotonic_bins=50)
    d = cfg.to_dict()
    cfg2 = CalibrationConfig.from_dict(d)
    check(cfg2.method == "isotonic", "config round-trip preserves method")
    check(
        cfg2.num_isotonic_bins == 50,
        "config round-trip preserves num_isotonic_bins",
    )

    # ------------------------------------------------------------------
    # 19. Calibration report
    # ------------------------------------------------------------------
    print("\n--- Test 19: Calibration report ---")
    report = CalibrationMetrics.calibration_report(
        confidences=confs_raw,
        accuracies=correct_raw,
        num_bins=15,
        probabilities=probs_raw,
        labels=labels,
        logits=logits,
    )
    expected_keys = {
        "ece",
        "mce",
        "adaptive_ece",
        "overconfidence_error",
        "underconfidence_error",
        "brier_score",
        "classwise_ece",
        "nll",
    }
    check(
        expected_keys.issubset(set(report.keys())),
        f"report contains all expected keys ({sorted(report.keys())})",
    )

    # ------------------------------------------------------------------
    # 20. NLL computation
    # ------------------------------------------------------------------
    print("\n--- Test 20: NLL computation ---")
    nll = CalibrationMetrics.compute_nll(logits, labels)
    check(
        nll > 0.0,
        f"NLL is positive ({nll:.4f})",
    )
    # Compare with torch directly.
    nll_torch = F.cross_entropy(logits.float(), labels.long()).item()
    check(
        abs(nll - nll_torch) < 1e-4,
        f"NLL matches torch cross_entropy ({nll:.6f} ~= {nll_torch:.6f})",
    )

    # ------------------------------------------------------------------
    # 21. Classwise ECE
    # ------------------------------------------------------------------
    print("\n--- Test 21: Classwise ECE ---")
    cw_ece = CalibrationMetrics.compute_classwise_ece(
        probs_raw, labels, num_bins=15
    )
    check(
        0.0 <= cw_ece <= 1.0,
        f"Classwise ECE in [0,1] ({cw_ece:.4f})",
    )

    # ------------------------------------------------------------------
    # 22. FocalLoss
    # ------------------------------------------------------------------
    print("\n--- Test 22: FocalLoss ---")
    fl = FocalLoss(gamma=2.0)
    fl_logits = logits[:100].clone().detach().requires_grad_(True)
    focal_loss = fl(fl_logits, labels[:100])
    ce_loss = F.cross_entropy(fl_logits, labels[:100].long())
    check(
        focal_loss.item() <= ce_loss.item() + 0.1,
        f"focal loss <= CE + margin "
        f"({focal_loss.item():.4f} vs {ce_loss.item():.4f})",
    )
    check(
        focal_loss.requires_grad,
        "focal loss has gradient",
    )

    # Gamma=0 should recover CE.
    fl_zero = FocalLoss(gamma=0.0)
    fl_zero_val = fl_zero(logits[:100], labels[:100])
    check(
        abs(fl_zero_val.item() - ce_loss.item()) < 1e-4,
        f"gamma=0 focal loss ~= CE "
        f"({fl_zero_val.item():.4f} ~= {ce_loss.item():.4f})",
    )

    # ------------------------------------------------------------------
    # 23. PlattScaler
    # ------------------------------------------------------------------
    print("\n--- Test 23: PlattScaler ---")
    platt = PlattScaler()
    torch.manual_seed(77)
    binary_scores = torch.randn(500)
    binary_labels_t = (binary_scores > 0).float()
    # Add noise so it is not perfect.
    flip_mask = torch.rand(500) < 0.2
    binary_labels_t[flip_mask] = 1.0 - binary_labels_t[flip_mask]
    platt_metrics = platt.fit(binary_scores, binary_labels_t)
    check(
        "ece_before" in platt_metrics and "ece_after" in platt_metrics,
        "PlattScaler.fit returns expected keys",
    )
    platt_probs = platt.calibrate(binary_scores)
    check(
        bool(
            (platt_probs >= 0.0).all() and (platt_probs <= 1.0).all()
        ),
        "PlattScaler output in [0,1]",
    )

    # ------------------------------------------------------------------
    # 24. BetaCalibrator
    # ------------------------------------------------------------------
    print("\n--- Test 24: BetaCalibrator ---")
    beta_cal = BetaCalibrator()
    raw_probs_for_beta = torch.sigmoid(binary_scores)
    beta_metrics = beta_cal.fit(raw_probs_for_beta, binary_labels_t)
    check(
        "ece_before" in beta_metrics and "ece_after" in beta_metrics,
        "BetaCalibrator.fit returns expected keys",
    )
    beta_probs = beta_cal.calibrate(raw_probs_for_beta)
    check(
        bool(
            (beta_probs >= 0.0).all() and (beta_probs <= 1.0).all()
        ),
        "BetaCalibrator output in [0,1]",
    )

    # ------------------------------------------------------------------
    # 25. HistogramBinningCalibrator
    # ------------------------------------------------------------------
    print("\n--- Test 25: HistogramBinningCalibrator ---")
    hist_cal = HistogramBinningCalibrator(num_bins=15)
    hist_cal.fit(confs_raw, correct_raw)
    check(hist_cal.is_fitted, "HistogramBinningCalibrator is fitted")
    hist_out = hist_cal.calibrate(confs_raw)
    check(
        bool((hist_out >= 0.0).all() and (hist_out <= 1.0).all()),
        "HistogramBinning output in [0,1]",
    )
    # State dict round-trip.
    hist_state = hist_cal.state_dict()
    hist_cal2 = HistogramBinningCalibrator(num_bins=15)
    hist_cal2.load_state_dict(hist_state)
    check(hist_cal2.is_fitted, "loaded HistogramBinning is fitted")

    # ------------------------------------------------------------------
    # 26. EnsembleTemperatureScaler
    # ------------------------------------------------------------------
    print("\n--- Test 26: EnsembleTemperatureScaler ---")
    ens = EnsembleTemperatureScaler(
        num_members=3, initial_temperature=1.5
    )
    ens_metrics = ens.fit(logits, labels)
    check(
        "ece_after" in ens_metrics,
        "EnsembleTemperatureScaler.fit returns ece_after",
    )
    ens_probs = ens.calibrate(logits)
    check(
        ens_probs.shape == (logits.shape[0], logits.shape[1]),
        f"ensemble output shape correct ({ens_probs.shape})",
    )
    check(
        torch.allclose(
            ens_probs.sum(dim=1),
            torch.ones(ens_probs.shape[0]),
            atol=1e-4,
        ),
        "ensemble output sums to 1",
    )

    # ------------------------------------------------------------------
    # 27. CalibrationPipeline
    # ------------------------------------------------------------------
    print("\n--- Test 27: CalibrationPipeline ---")
    pipe = CalibrationPipeline(
        CalibrationConfig(method="temperature")
    )
    pipe_metrics = pipe.fit(logits, labels)
    check(
        "ece_after" in pipe_metrics,
        "pipeline fit returns ece_after",
    )
    pipe_out = pipe.calibrate(logits)
    check(
        pipe_out.shape == logits.shape,
        f"pipeline calibrate output shape ({pipe_out.shape})",
    )
    pipe_eval = pipe.evaluate(logits, labels)
    check(
        "ece" in pipe_eval,
        "pipeline evaluate returns ece",
    )

    # Pipeline state dict round-trip.
    pipe_state = pipe.state_dict()
    pipe2 = CalibrationPipeline()
    pipe2.load_state_dict(pipe_state)
    check(
        pipe2.config.method == "temperature",
        "pipeline state_dict round-trip preserves method",
    )

    # ------------------------------------------------------------------
    # 28. Edge case: empty tensor
    # ------------------------------------------------------------------
    print("\n--- Test 28: Edge case -- empty tensor ---")
    empty_conf = torch.tensor([])
    empty_acc = torch.tensor([])
    ece_empty = CalibrationMetrics.compute_ece(empty_conf, empty_acc)
    check(ece_empty == 0.0, f"ECE of empty input is 0.0 ({ece_empty})")
    mce_empty = CalibrationMetrics.compute_mce(empty_conf, empty_acc)
    check(mce_empty == 0.0, f"MCE of empty input is 0.0 ({mce_empty})")

    # ------------------------------------------------------------------
    # 29. Edge case: all same confidence
    # ------------------------------------------------------------------
    print("\n--- Test 29: Edge case -- uniform confidence ---")
    torch.manual_seed(999)
    uniform_conf = torch.full((100,), 0.5)
    uniform_acc = (torch.rand(100) > 0.5).float()
    ece_uniform = CalibrationMetrics.compute_ece(uniform_conf, uniform_acc)
    check(
        isinstance(ece_uniform, float),
        f"ECE with uniform confidence works ({ece_uniform:.4f})",
    )

    # ------------------------------------------------------------------
    # 30. Miscalibrated model: high ECE reduced by temperature scaling
    # ------------------------------------------------------------------
    print("\n--- Test 30: Miscalibrated model has high ECE ---")
    logits_bad, labels_bad = generate_miscalibrated_logits(
        num_samples=2000, num_classes=10, overconfidence=6.0, seed=99
    )
    probs_bad = F.softmax(logits_bad, dim=1)
    confs_bad, preds_bad = probs_bad.max(dim=1)
    correct_bad = (preds_bad == labels_bad).float()
    ece_bad = CalibrationMetrics.compute_ece(confs_bad, correct_bad)
    check(
        ece_bad > 0.01,
        f"strongly overconfident model has noticeable ECE ({ece_bad:.4f})",
    )

    # Temperature scaling should find a reasonable temperature.
    scaler_bad = TemperatureScaler(initial_temperature=1.5)
    metrics_bad = scaler_bad.fit(logits_bad, labels_bad)
    check(
        scaler_bad.temperature > 0.0 and "temperature" in metrics_bad,
        f"temperature scaling converged (T={scaler_bad.temperature:.4f})",
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"{passed}/{total} self-tests passed")
    if failed > 0:
        print(f"{failed} test(s) FAILED")
    else:
        print("All tests passed.")
    print("=" * 70)


if __name__ == "__main__":
    _run_self_tests()
