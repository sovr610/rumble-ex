"""
SNN-Specific Loss Pack with Batch-First Convention and AMP Hardening
=====================================================================

Template for ``brain_ai/core/losses.py``

This module provides a composable, production-grade set of loss functions and
regularisation terms designed for training Spiking Neural Networks (SNNs).  All
tensor operations follow the **batch-first** convention:

    spikes  : (B, T, N)   -- batch, time-steps, neurons / output-channels
    membrane: (B, T, N)   -- membrane potentials (same layout)
    targets : (B,)        -- integer class labels for classification

Key design choices
------------------
* **FP32 accumulation everywhere** -- every spike reduction (.sum(), .mean())
  is preceded by an explicit ``.float()`` cast so the computation is safe under
  ``torch.cuda.amp.autocast``.
* **Vectorised ISI** -- inter-spike-interval regularisation is implemented
  entirely with ``F.conv1d`` and tensor ops; there are *no* Python-level loops
  over batches or neurons.
* **Diagnostics by default** -- every ``LossTerm`` returns a ``(loss, diag)``
  pair.  The ``SNNLossComposer`` collects them into a single nested dict that
  is easy to feed into TensorBoard / W&B / plain-text logging.
* **Per-layer targets** -- optionally pass ``layer_name`` to any loss term and
  the ``LossConfig.per_layer_targets`` dict is consulted for that layer's
  specific target firing-rate.

References
----------
* Shrestha & Bhatt (2022) -- ProbSpikes loss for SNN classifiers.
* Yu et al. (2025) -- "Beyond Rate Coding: Surrogate Gradients Enable Spike
  Timing Learning".
* Neftci et al. (2019) -- Surrogate gradient learning in SNNs.
* Zenke & Ganguli (2018) -- SuperSpike and spike-rate regularisation.
"""

from __future__ import annotations

import abc
import math
import sys
import textwrap
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# CONSOLIDATION NOTE: The canonical LossConfig is defined in
# codec_config_template.py (target: brain_ai/config.py). This local copy
# exists so losses_template.py can be self-tested standalone. When integrating
# into brain_ai, import LossConfig from config.py instead of duplicating.
# Validation rules here (temporal_window >= 2) are authoritative.

@dataclass
class LossConfig:
    """Central configuration for every loss term used by ``SNNLossComposer``.

    All weights use a simple linear combination::

        total = w_probspikes * probspikes
              + w_rate       * rate_reg
              + w_temporal   * temporal
              + w_isi        * isi_reg
              + w_membrane   * membrane_reg

    Set any weight to 0.0 to disable that term entirely (the corresponding
    ``LossTerm`` will not even be instantiated by the composer).

    Attributes
    ----------
    w_probspikes : float
        Weight for the primary classification loss (ProbSpikes / cross-entropy
        on spike counts).  Typically the dominant term.
    w_rate : float
        Weight for spike-rate regularisation (target-rate + range penalty).
    w_temporal : float
        Weight for temporal-consistency / smoothness regularisation.
    w_isi : float
        Weight for the inter-spike-interval (refractory) penalty.
    w_membrane : float
        Weight for membrane-potential explosion penalty.
    target_rate : float
        Desired mean firing rate (fraction of time-steps).  0.1 corresponds
        to roughly 10 Hz in a 100-step simulation.
    min_rate : float
        Minimum acceptable firing rate -- below this a neuron is considered
        "dead" and receives an additional penalty.
    max_rate : float
        Maximum acceptable firing rate -- above this a neuron is considered
        "saturated".
    probspikes_temperature : float
        Temperature divisor applied to spike counts *before* softmax.  Lower
        values sharpen the distribution, higher values flatten it.  Must be
        > 0.
    probspikes_eps : float
        Epsilon for numerical stability inside ProbSpikes (log / division).
    probspikes_mode : str
        ``"softmax"`` -- divide counts by temperature, apply log-softmax,
        then NLL.  This is equivalent to ``F.cross_entropy``.
        ``"normalize"`` -- explicitly normalise counts to a simplex and take
        ``F.nll_loss(log(probs), targets)``.
    temporal_window : int
        Window size for the temporal-consistency loss (number of time-steps
        per block).
    temporal_penalty : str
        ``"l1"`` -- mean absolute difference between consecutive time-steps.
        ``"l2"`` -- mean squared difference.
        ``"variance"`` -- variance of windowed spike rates.
    isi_refractory_window : int
        Length (in time-steps) of the refractory kernel convolved against the
        spike train.  Larger values penalise longer-range bursts.
    isi_kernel_type : str
        ``"exponential"`` -- exponentially decaying kernel
        ``exp(-t / tau)`` with ``tau = refractory_window / 2``.
        ``"rectangular"`` -- flat kernel of ones.
    membrane_max : float
        Membrane potentials whose absolute value exceeds this threshold
        receive a squared excess penalty.
    per_layer_targets : dict or None
        Optional mapping ``{layer_name: target_rate}`` that overrides
        ``target_rate`` for specific named layers.
    log_diagnostics : bool
        When True, every ``LossTerm.compute()`` populates the diagnostics
        dict.  Set to False in tight inner loops where the overhead of
        ``.item()`` calls is unwanted.
    """

    # ---- primary weights ------------------------------------------------
    w_probspikes: float = 1.0
    w_rate: float = 0.1
    w_temporal: float = 0.01
    w_isi: float = 0.01
    w_membrane: float = 0.001

    # ---- spike-rate parameters ------------------------------------------
    target_rate: float = 0.1
    min_rate: float = 0.01
    max_rate: float = 0.3

    # ---- ProbSpikes parameters ------------------------------------------
    probspikes_temperature: float = 1.0
    probspikes_eps: float = 1e-7
    probspikes_mode: str = "softmax"

    # ---- temporal consistency -------------------------------------------
    temporal_window: int = 5
    temporal_penalty: str = "l2"

    # ---- ISI / refractory -----------------------------------------------
    isi_refractory_window: int = 3
    isi_kernel_type: str = "exponential"

    # ---- membrane -------------------------------------------------------
    membrane_max: float = 1.5

    # ---- per-layer overrides --------------------------------------------
    per_layer_targets: Optional[Dict[str, float]] = None

    # ---- diagnostics ----------------------------------------------------
    log_diagnostics: bool = True

    # -- validation -------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate parameters on construction."""
        if self.probspikes_temperature <= 0:
            raise ValueError(
                f"probspikes_temperature must be > 0, got {self.probspikes_temperature}"
            )
        if self.probspikes_mode not in ("softmax", "normalize"):
            raise ValueError(
                f"probspikes_mode must be 'softmax' or 'normalize', "
                f"got {self.probspikes_mode!r}"
            )
        if self.temporal_penalty not in ("l1", "l2", "variance"):
            raise ValueError(
                f"temporal_penalty must be 'l1', 'l2', or 'variance', "
                f"got {self.temporal_penalty!r}"
            )
        if self.isi_kernel_type not in ("exponential", "rectangular"):
            raise ValueError(
                f"isi_kernel_type must be 'exponential' or 'rectangular', "
                f"got {self.isi_kernel_type!r}"
            )
        if self.min_rate >= self.max_rate:
            raise ValueError(
                f"min_rate ({self.min_rate}) must be < max_rate ({self.max_rate})"
            )
        if self.isi_refractory_window < 1:
            raise ValueError(
                f"isi_refractory_window must be >= 1, got {self.isi_refractory_window}"
            )
        if self.temporal_window < 2:
            raise ValueError(
                f"temporal_window must be >= 2, got {self.temporal_window}"
            )
        if self.membrane_max <= 0:
            raise ValueError(
                f"membrane_max must be > 0, got {self.membrane_max}"
            )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class LossTerm(nn.Module, abc.ABC):
    """Abstract base for every individual loss / regularisation term.

    Inherits from ``nn.Module`` so that:
    - Learnable parameters (if any) are tracked by ``model.parameters()``.
    - Buffers (e.g. ISI kernel) participate in ``.to(device)`` calls.
    - Terms can be stored in ``nn.ModuleList`` / ``nn.ModuleDict``.

    Subclasses must implement :meth:`compute` which returns a
    ``(scalar_loss, diagnostics_dict)`` pair.  The diagnostics dict should
    contain human-readable floats (already detached from the graph).

    Convention: **all intermediate accumulations happen in fp32** regardless
    of the autocast context.  This means every subclass must call
    ``.float()`` on spike / membrane tensors before reductions.

    Parameters
    ----------
    config : LossConfig
        Shared configuration.
    """

    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        self.config = config

    # -- public interface --------------------------------------------------

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short, unique, snake_case identifier for this term."""
        ...

    @abc.abstractmethod
    def compute(
        self,
        spikes: Tensor,
        membrane: Optional[Tensor],
        targets: Optional[Tensor],
        **kwargs: Any,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute the loss value and optional diagnostics.

        Parameters
        ----------
        spikes : Tensor
            Spike tensor of shape ``(B, T, N)`` with values in {0, 1} (or
            soft surrogates in [0, 1]).
        membrane : Tensor or None
            Membrane potentials ``(B, T, N)``.  May be ``None`` if not
            available.
        targets : Tensor or None
            Class labels ``(B,)`` for classification tasks, or ``None``
            for unsupervised regularisation terms.
        **kwargs
            Arbitrary additional context, e.g. ``layer_name``.

        Returns
        -------
        loss : Tensor
            A scalar (0-dim) loss tensor that participates in the
            computation graph.
        diagnostics : dict[str, float]
            Human-readable scalars; empty ``{}`` when
            ``config.log_diagnostics`` is False.
        """
        ...

    # -- helpers -----------------------------------------------------------

    def _empty_diag(self) -> Dict[str, float]:
        """Return an empty dict -- shortcut when diagnostics are disabled."""
        return {}

    def _resolve_target_rate(self, **kwargs: Any) -> float:
        """Look up per-layer target rate, falling back to global default."""
        layer_name: Optional[str] = kwargs.get("layer_name")
        if (
            layer_name is not None
            and self.config.per_layer_targets is not None
            and layer_name in self.config.per_layer_targets
        ):
            return self.config.per_layer_targets[layer_name]
        return self.config.target_rate

    def __repr__(self) -> str:  # pragma: no cover
        return f"{type(self).__name__}(name={self.name!r})"


# ---------------------------------------------------------------------------
# 1. ProbSpikes classification loss
# ---------------------------------------------------------------------------

class ProbSpikesLoss(LossTerm):
    """Cross-entropy loss on spike-count logits / normalised spike counts.

    The idea (Shrestha et al., 2022) is to sum spikes across time for each
    output neuron, interpret the resulting counts as unnormalised logits (or
    explicitly normalise them), and compute standard cross-entropy against
    class labels.

    Two modes are supported:

    ``"softmax"``
        ``logits = counts / temperature`` fed into ``F.cross_entropy``.
        This is the numerically safest path and is recommended for most
        training runs.

    ``"normalize"``
        ``probs = (counts + eps) / (counts.sum(-1, keepdim=True) + eps*C)``
        followed by ``F.nll_loss(probs.log(), targets)``.
        This path interprets the counts as (near-)probabilities directly.

    AMP hardening
    ~~~~~~~~~~~~~
    * Spike counts are accumulated in fp32 (``spikes.float().sum(dim=1)``).
    * Temperature scaling is applied *before* the softmax / normalisation.
    * No in-place operations on autocast tensors.

    Diagnostics (when enabled)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    * ``mean_counts``  -- average spike count across batch and classes.
    * ``max_counts``   -- maximum spike count in the batch.
    * ``count_entropy`` -- entropy of the count distribution (higher = more
      uniform).
    * ``correct_class_count_mean`` -- mean count for the correct class.
    """

    @property
    def name(self) -> str:
        return "probspikes"

    def compute(
        self,
        spikes: Tensor,
        membrane: Optional[Tensor],
        targets: Optional[Tensor],
        **kwargs: Any,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute ProbSpikes loss.

        Parameters
        ----------
        spikes : Tensor
            ``(B, T, C)`` where C is the number of output classes.
        membrane : Tensor or None
            Not used by this term.
        targets : Tensor
            ``(B,)`` integer class labels.

        Returns
        -------
        loss : Tensor
            Scalar classification loss.
        diag : dict
            Diagnostic scalars.

        Raises
        ------
        ValueError
            If ``targets`` is None -- classification requires labels.
        """
        if targets is None:
            raise ValueError("ProbSpikesLoss requires targets (class labels).")

        # -- accumulate spike counts in fp32 --------------------------------
        # spikes: (B, T, C) -> counts: (B, C)
        counts: Tensor = spikes.float().sum(dim=1)  # always fp32

        B, C = counts.shape
        temperature: float = self.config.probspikes_temperature
        eps: float = self.config.probspikes_eps
        mode: str = self.config.probspikes_mode

        if mode == "softmax":
            # Scale by temperature BEFORE softmax for numerical safety.
            logits = counts / temperature  # (B, C), fp32
            loss = F.cross_entropy(logits, targets)

        elif mode == "normalize":
            # Explicit normalisation to a probability simplex.
            counts_safe = counts + eps  # avoid zero counts
            total = counts_safe.sum(dim=-1, keepdim=True) + eps * C
            probs = counts_safe / total  # (B, C)
            # Clamp before log for absolute safety.
            log_probs = torch.clamp(probs, min=eps).log()
            loss = F.nll_loss(log_probs, targets)

        else:
            raise ValueError(f"Unknown probspikes_mode: {mode!r}")

        # -- diagnostics ----------------------------------------------------
        diag: Dict[str, float] = {}
        if self.config.log_diagnostics:
            with torch.no_grad():
                diag["mean_counts"] = counts.mean().item()
                diag["max_counts"] = counts.max().item()

                # Entropy of the *batch-mean* count distribution.
                mean_counts = counts.mean(dim=0)  # (C,)
                p = F.softmax(mean_counts, dim=0)
                entropy = -(p * torch.clamp(p, min=eps).log()).sum()
                diag["count_entropy"] = entropy.item()

                # Mean count in the correct class.
                correct_counts = counts[
                    torch.arange(B, device=counts.device), targets
                ]
                diag["correct_class_count_mean"] = correct_counts.mean().item()

        return loss, diag


# ---------------------------------------------------------------------------
# 2. Spike-rate regularisation
# ---------------------------------------------------------------------------

class SpikeRateRegularization(LossTerm):
    """Penalise firing rates that deviate from a biologically plausible target.

    This term combines two complementary penalties:

    **Target-rate loss**
        ``((rate - target_rate) ** 2).mean()``
        Encourages every neuron to fire at roughly ``target_rate``.

    **Range loss**
        ``(ReLU(min_rate - rate) + ReLU(rate - max_rate)) ** 2 .mean()``
        Specifically penalises dead (< min_rate) and saturated (> max_rate)
        neurons with a one-sided squared hinge.

    The two are summed to form the final loss.

    Per-layer targets
    ~~~~~~~~~~~~~~~~~
    Pass ``layer_name="some_layer"`` in ``**kwargs`` and the term will look
    up ``config.per_layer_targets["some_layer"]`` for an override of the
    global ``config.target_rate``.

    Diagnostics
    ~~~~~~~~~~~
    * ``mean_rate`` -- average firing rate across batch and neurons.
    * ``max_rate``  -- maximum per-neuron rate.
    * ``min_rate``  -- minimum per-neuron rate.
    * ``dead_fraction``      -- fraction of neurons with rate < min_rate.
    * ``saturated_fraction``  -- fraction of neurons with rate > max_rate.
    """

    @property
    def name(self) -> str:
        return "rate_reg"

    def compute(
        self,
        spikes: Tensor,
        membrane: Optional[Tensor],
        targets: Optional[Tensor],
        **kwargs: Any,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute spike-rate regularisation.

        Parameters
        ----------
        spikes : Tensor
            ``(B, T, N)`` spike tensor.
        membrane : Tensor or None
            Not used.
        targets : Tensor or None
            Not used.

        Returns
        -------
        loss : Tensor
            Scalar regularisation loss.
        diag : dict
            Diagnostic scalars.
        """
        # -- fp32 rate computation -----------------------------------------
        # spikes: (B, T, N) -> rate: (B, N)
        rate: Tensor = spikes.float().mean(dim=1)  # always fp32

        target: float = self._resolve_target_rate(**kwargs)
        min_rate: float = self.config.min_rate
        max_rate: float = self.config.max_rate

        # Target-rate L2 penalty.
        target_loss: Tensor = ((rate - target) ** 2).mean()

        # Range penalty: one-sided squared hinge for dead / saturated.
        below_min: Tensor = F.relu(min_rate - rate)
        above_max: Tensor = F.relu(rate - max_rate)
        range_loss: Tensor = ((below_min ** 2) + (above_max ** 2)).mean()

        loss: Tensor = target_loss + range_loss

        # -- diagnostics ---------------------------------------------------
        diag: Dict[str, float] = {}
        if self.config.log_diagnostics:
            with torch.no_grad():
                flat_rate = rate.detach()  # (B, N)
                diag["mean_rate"] = flat_rate.mean().item()
                diag["max_rate"] = flat_rate.max().item()
                diag["min_rate"] = flat_rate.min().item()
                diag["dead_fraction"] = (
                    (flat_rate < min_rate).float().mean().item()
                )
                diag["saturated_fraction"] = (
                    (flat_rate > max_rate).float().mean().item()
                )

        return loss, diag


# ---------------------------------------------------------------------------
# 3. Temporal-consistency loss
# ---------------------------------------------------------------------------

class TemporalConsistencyLoss(LossTerm):
    """Encourage stable (or structured) temporal firing patterns.

    Three penalty modes are offered via ``config.temporal_penalty``:

    ``"l1"``
        Mean absolute difference between consecutive time-steps::

            diff = spikes[:, 1:, :] - spikes[:, :-1, :]
            loss = diff.abs().mean()

    ``"l2"``
        Mean squared difference (penalises large flickers more)::

            loss = (diff ** 2).mean()

    ``"variance"``
        Splits the time axis into non-overlapping windows of
        ``config.temporal_window`` steps, computes the mean rate in each
        window, then takes the variance across windows.  High variance means
        the network's firing pattern is unstable at a coarse time-scale.

    Diagnostics
    ~~~~~~~~~~~
    * ``mean_temporal_var`` -- variance of windowed rates averaged over batch
      and neurons.
    * ``max_temporal_var``  -- maximum per-neuron windowed-rate variance.
    """

    @property
    def name(self) -> str:
        return "temporal"

    def compute(
        self,
        spikes: Tensor,
        membrane: Optional[Tensor],
        targets: Optional[Tensor],
        **kwargs: Any,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute temporal-consistency loss.

        Parameters
        ----------
        spikes : Tensor
            ``(B, T, N)`` spike tensor.
        membrane, targets
            Not used.

        Returns
        -------
        loss : Tensor
            Scalar temporal loss.
        diag : dict
            Diagnostic scalars.
        """
        spikes_f: Tensor = spikes.float()  # fp32
        B, T, N = spikes_f.shape
        penalty: str = self.config.temporal_penalty
        window: int = self.config.temporal_window

        # -- l1 / l2: consecutive-step differences -------------------------
        if penalty in ("l1", "l2"):
            if T < 2:
                loss = torch.tensor(0.0, device=spikes.device, dtype=torch.float32)
                return loss, self._empty_diag()

            diff: Tensor = spikes_f[:, 1:, :] - spikes_f[:, :-1, :]
            if penalty == "l1":
                loss = diff.abs().mean()
            else:  # l2
                loss = (diff ** 2).mean()

            # Diagnostics: also compute windowed variance for monitoring.
            diag = self._windowed_variance_diag(spikes_f, window)
            return loss, diag

        # -- variance: windowed-rate variance ------------------------------
        if penalty == "variance":
            if T < window:
                loss = torch.tensor(0.0, device=spikes.device, dtype=torch.float32)
                return loss, self._empty_diag()

            # Trim to a multiple of window for clean reshaping.
            n_windows = T // window
            trimmed = spikes_f[:, : n_windows * window, :]  # (B, n_w*W, N)
            windowed = trimmed.reshape(B, n_windows, window, N)
            window_means: Tensor = windowed.mean(dim=2)  # (B, n_windows, N)
            var_across_windows: Tensor = window_means.var(dim=1)  # (B, N)
            loss = var_across_windows.mean()

            diag: Dict[str, float] = {}
            if self.config.log_diagnostics:
                with torch.no_grad():
                    diag["mean_temporal_var"] = var_across_windows.mean().item()
                    diag["max_temporal_var"] = var_across_windows.max().item()
            return loss, diag

        raise ValueError(f"Unknown temporal_penalty: {penalty!r}")

    # -- internal helpers --------------------------------------------------

    def _windowed_variance_diag(
        self, spikes_f: Tensor, window: int
    ) -> Dict[str, float]:
        """Compute windowed-variance diagnostics even when loss mode is l1/l2."""
        if not self.config.log_diagnostics:
            return {}

        B, T, N = spikes_f.shape
        if T < window:
            return {"mean_temporal_var": 0.0, "max_temporal_var": 0.0}

        with torch.no_grad():
            n_windows = T // window
            trimmed = spikes_f[:, : n_windows * window, :]
            windowed = trimmed.reshape(B, n_windows, window, N)
            window_means = windowed.mean(dim=2)
            var_across = window_means.var(dim=1)  # (B, N)
            return {
                "mean_temporal_var": var_across.mean().item(),
                "max_temporal_var": var_across.max().item(),
            }


# ---------------------------------------------------------------------------
# 4. ISI (inter-spike-interval) regularisation -- VECTORISED
# ---------------------------------------------------------------------------

class ISIRegularization(LossTerm):
    """Penalise refractory violations (bursts) using a 1-D convolution.

    Classical ISI analysis iterates over individual neurons in Python and
    extracts spike times -- this is prohibitively slow for large networks
    under gradient-based training.  Instead we construct a small 1-D
    "refractory kernel" and convolve it against every neuron's spike train
    in parallel using ``F.conv1d``.

    The kernel encodes the idea that *if a neuron fired at time t, there
    should be minimal firing in the next few time-steps*.  Two kernel shapes
    are supported:

    ``"exponential"``
        ``kernel[k] = exp(-k / tau)`` for ``k = 1 .. W`` with
        ``tau = max(1, W / 2)``.  Strongly penalises immediate re-firing
        and decays over the refractory window.

    ``"rectangular"``
        ``kernel[k] = 1`` for ``k = 1 .. W``.  Uniform penalty for any
        spike within the refractory window.

    The penalty is then the dot product of the original spikes with the
    convolution output::

        penalty = (spikes * conv_out).mean()

    This is zero when no neuron fires within another neuron's refractory
    window and large for bursty activity.

    AMP hardening
    ~~~~~~~~~~~~~
    * The kernel is constructed and stored in fp32.
    * Input spikes are cast to fp32 before the convolution.
    * No in-place operations.

    Diagnostics
    ~~~~~~~~~~~
    * ``mean_penalty`` -- average per-element penalty.
    * ``burst_fraction`` -- fraction of (B, T, N) entries where
      ``spikes * conv_out > 0``, i.e. spikes that occurred within the
      refractory window of a preceding spike.
    """

    def __init__(self, config: LossConfig) -> None:
        super().__init__(config)
        # Pre-build and register the refractory kernel as a buffer so it
        # participates in .to(device) and .state_dict().
        self.register_buffer("_kernel", self._make_kernel())

    @property
    def name(self) -> str:
        return "isi_reg"

    # -- kernel construction -----------------------------------------------

    def _make_kernel(self) -> Tensor:
        """Construct the refractory penalty kernel of shape ``(1, 1, W+1)``.

        The kernel intentionally has a zero at offset 0 (so the spike does
        not penalise itself).  Offsets 1..W carry the refractory penalty.
        """
        W: int = self.config.isi_refractory_window
        kernel_type: str = self.config.isi_kernel_type

        if kernel_type == "exponential":
            tau: float = max(1.0, W / 2.0)
            offsets = torch.arange(1, W + 1, dtype=torch.float32)
            values = torch.exp(-offsets / tau)
        elif kernel_type == "rectangular":
            values = torch.ones(W, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown isi_kernel_type: {kernel_type!r}")

        kernel = torch.zeros(W + 1, dtype=torch.float32)
        kernel[1:] = values
        return kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, W+1)

    def compute(
        self,
        spikes: Tensor,
        membrane: Optional[Tensor],
        targets: Optional[Tensor],
        **kwargs: Any,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute vectorised ISI regularisation via conv1d.

        Parameters
        ----------
        spikes : Tensor
            ``(B, T, N)`` spike tensor.
        membrane, targets
            Not used.

        Returns
        -------
        loss : Tensor
            Scalar ISI penalty.
        diag : dict
            Diagnostic scalars.
        """
        B, T, N = spikes.shape
        device = spikes.device
        W: int = self.config.isi_refractory_window

        if T < 2:
            zero = torch.tensor(0.0, device=device, dtype=torch.float32)
            return zero, self._empty_diag()

        # -- cast to fp32 and reshape for grouped conv1d -------------------
        # We need shape (B*N, 1, T) for F.conv1d.
        spikes_f: Tensor = spikes.float()  # (B, T, N), fp32
        # Permute to (B, N, T) then reshape.
        spikes_bn: Tensor = spikes_f.permute(0, 2, 1).reshape(B * N, 1, T)

        # -- use registered kernel buffer ------------------------------------
        kernel: Tensor = self._kernel  # (1, 1, W+1), tracked by .to(device)
        K: int = kernel.shape[2]

        # -- conv1d with causal (left) padding -----------------------------
        # Pad on the left so that conv_out[..., t] only sees spikes at
        # times <= t (causal).  Padding = K-1 on the left, 0 on the right.
        padded: Tensor = F.pad(spikes_bn, (K - 1, 0))  # (B*N, 1, T+K-1)
        conv_out: Tensor = F.conv1d(padded, kernel)  # (B*N, 1, T)
        # conv_out[..., t] = sum over k of kernel[k] * spikes[t-k]
        # At lag 0 the kernel is zero, so it only includes prior spikes.

        # -- penalty: spike * refractory field -----------------------------
        penalty_map: Tensor = spikes_bn * conv_out  # (B*N, 1, T)
        loss: Tensor = penalty_map.mean()

        # -- diagnostics ---------------------------------------------------
        diag: Dict[str, float] = {}
        if self.config.log_diagnostics:
            with torch.no_grad():
                diag["mean_penalty"] = penalty_map.mean().item()
                diag["burst_fraction"] = (
                    (penalty_map > 0).float().mean().item()
                )

        return loss, diag


# ---------------------------------------------------------------------------
# 5. Membrane-potential regularisation
# ---------------------------------------------------------------------------

class MembranePotentialRegularization(LossTerm):
    """Penalise membrane potentials that exceed a safe magnitude.

    Membrane explosion is a common failure mode in SNN training -- the
    potentials grow without bound, leading to gradient blow-up and NaN
    losses.  This term applies a *squared hinge* on the excess::

        excess = ReLU(|membrane| - max_membrane)
        loss   = (excess ** 2).mean()

    The squared form ensures the gradient near the boundary is gentle while
    strongly penalising large overshoots.

    Diagnostics
    ~~~~~~~~~~~
    * ``mean_membrane``     -- mean absolute membrane potential.
    * ``max_membrane``      -- maximum absolute value.
    * ``explosion_fraction`` -- fraction of entries where ``|v| > max``.
    """

    @property
    def name(self) -> str:
        return "membrane_reg"

    def compute(
        self,
        spikes: Tensor,
        membrane: Optional[Tensor],
        targets: Optional[Tensor],
        **kwargs: Any,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute membrane-potential regularisation.

        Parameters
        ----------
        spikes : Tensor
            Not used.
        membrane : Tensor
            ``(B, T, N)`` or ``(B, N)`` membrane potentials.
        targets : Tensor or None
            Not used.

        Returns
        -------
        loss : Tensor
            Scalar penalty.
        diag : dict
            Diagnostic scalars.

        Notes
        -----
        If ``membrane`` is ``None``, returns zero loss and empty diagnostics.
        """
        if membrane is None:
            zero = torch.tensor(0.0, device=spikes.device, dtype=torch.float32)
            return zero, self._empty_diag()

        membrane_f: Tensor = membrane.float()  # fp32
        max_v: float = self.config.membrane_max

        excess: Tensor = F.relu(membrane_f.abs() - max_v)
        loss: Tensor = (excess ** 2).mean()

        # -- diagnostics ---------------------------------------------------
        diag: Dict[str, float] = {}
        if self.config.log_diagnostics:
            with torch.no_grad():
                abs_v = membrane_f.abs()
                diag["mean_membrane"] = abs_v.mean().item()
                diag["max_membrane"] = abs_v.max().item()
                diag["explosion_fraction"] = (
                    (abs_v > max_v).float().mean().item()
                )

        return loss, diag


# ---------------------------------------------------------------------------
# Composer -- wires all terms into a single nn.Module
# ---------------------------------------------------------------------------

class SNNLossComposer(nn.Module):
    """Composable SNN loss module that aggregates multiple ``LossTerm`` s.

    The composer reads a ``LossConfig``, instantiates only those terms whose
    weight is non-zero, and sums the weighted losses in ``forward()``.

    Usage
    -----
    ::

        cfg = LossConfig(w_probspikes=1.0, w_rate=0.1, w_temporal=0.01)
        loss_fn = SNNLossComposer(cfg)

        total_loss, info = loss_fn(spikes, membrane, targets)
        # info["total"]        -- float, total weighted loss
        # info["components"]   -- {term_name: unweighted_loss}
        # info["diagnostics"]  -- {term_name: {diag_key: diag_val, ...}}
        # info["weights"]      -- {term_name: weight}

    Parameters
    ----------
    config : LossConfig
        Full loss configuration.

    Attributes
    ----------
    terms : list of (weight, LossTerm)
        Active loss terms with their weights.
    """

    def __init__(self, config: Optional[LossConfig] = None) -> None:
        super().__init__()
        if config is None:
            config = LossConfig()
        self.config: LossConfig = config

        # Build list of active terms using nn.ModuleList so submodule
        # parameters/buffers (e.g. ISI kernel) participate in .to(device).
        self._term_modules = nn.ModuleList()
        self._term_weights: List[float] = []

        _registry: List[Tuple[float, type]] = [
            (config.w_probspikes, ProbSpikesLoss),
            (config.w_rate, SpikeRateRegularization),
            (config.w_temporal, TemporalConsistencyLoss),
            (config.w_isi, ISIRegularization),
            (config.w_membrane, MembranePotentialRegularization),
        ]

        for weight, cls in _registry:
            if weight > 0.0:
                self._term_modules.append(cls(config))
                self._term_weights.append(weight)

        # Store last forward info for debugging.
        self._last_info: Dict[str, Any] = {}

    # -- forward -----------------------------------------------------------

    def forward(
        self,
        spikes: Tensor,
        membrane: Optional[Tensor],
        targets: Optional[Tensor],
        **kwargs: Any,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Compute weighted sum of all active loss terms.

        Parameters
        ----------
        spikes : Tensor
            ``(B, T, N)`` or ``(B, T, C)`` spike tensor.
        membrane : Tensor or None
            ``(B, T, N)`` membrane potentials.
        targets : Tensor or None
            ``(B,)`` class labels (required if ProbSpikes is active).
        **kwargs
            Forwarded to each ``LossTerm.compute()``.

        Returns
        -------
        total_loss : Tensor
            Scalar loss for ``.backward()``.
        info : dict
            Structured information dict with keys:
            ``"total"``       -- float, total loss value.
            ``"components"``  -- dict mapping term name to *unweighted* loss
            (float).
            ``"diagnostics"`` -- dict mapping term name to its diagnostics
            dict.
            ``"weights"``     -- dict mapping term name to its weight.
        """
        device = spikes.device
        total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

        components: Dict[str, float] = {}
        diagnostics: Dict[str, Dict[str, float]] = {}
        weights: Dict[str, float] = {}

        for weight, term in zip(self._term_weights, self._term_modules):
            raw_loss, diag = term.compute(
                spikes=spikes,
                membrane=membrane,
                targets=targets,
                **kwargs,
            )

            # Defensive: ensure raw_loss is at least fp32 scalar.
            raw_loss = raw_loss.float()

            total_loss = total_loss + weight * raw_loss

            term_name: str = term.name
            components[term_name] = raw_loss.detach().item()
            diagnostics[term_name] = diag
            weights[term_name] = weight

        info: Dict[str, Any] = {
            "total": total_loss.detach().item(),
            "components": components,
            "diagnostics": diagnostics,
            "weights": weights,
        }
        self._last_info = info

        return total_loss, info

    # -- utilities ---------------------------------------------------------

    def get_effective_magnitudes(self) -> Dict[str, float]:
        """Return ``weight * raw_loss`` for each term from the last forward.

        Useful for diagnosing loss-term balance: if one magnitude dominates,
        its weight probably needs tuning.

        Returns
        -------
        magnitudes : dict[str, float]
            ``{term_name: weight * unweighted_loss}``.
        """
        magnitudes: Dict[str, float] = {}
        components = self._last_info.get("components", {})
        weights = self._last_info.get("weights", {})
        for name in components:
            magnitudes[name] = weights.get(name, 0.0) * components[name]
        return magnitudes

    def log_summary(self, prefix: str = "") -> str:
        """Format a human-readable summary of the last forward pass.

        Parameters
        ----------
        prefix : str
            Optional prefix prepended to every line (useful for indentation
            or log-level markers).

        Returns
        -------
        summary : str
            Multi-line string.
        """
        info = self._last_info
        if not info:
            return f"{prefix}(no forward pass recorded yet)"

        lines: List[str] = []
        lines.append(f"{prefix}=== SNN Loss Summary ===")
        lines.append(f"{prefix}Total loss: {info['total']:.6f}")
        lines.append(f"{prefix}---")

        components = info.get("components", {})
        weights = info.get("weights", {})
        diagnostics = info.get("diagnostics", {})

        for name in components:
            w = weights.get(name, 0.0)
            raw = components[name]
            eff = w * raw
            lines.append(
                f"{prefix}  {name:20s}  weight={w:.4f}  raw={raw:.6f}  "
                f"effective={eff:.6f}"
            )
            diag = diagnostics.get(name, {})
            for dk, dv in diag.items():
                lines.append(f"{prefix}    {dk:30s}: {dv:.6f}")

        magnitudes = self.get_effective_magnitudes()
        total_mag = sum(magnitudes.values()) or 1.0
        lines.append(f"{prefix}---")
        lines.append(f"{prefix}Effective magnitude fractions:")
        for name, mag in magnitudes.items():
            frac = mag / total_mag if total_mag > 0 else 0.0
            lines.append(f"{prefix}  {name:20s}: {frac:.1%}")

        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover
        term_strs = ", ".join(
            f"{w:.3f}*{t.name}" for w, t in zip(self._term_weights, self._term_modules)
        )
        return f"SNNLossComposer([{term_strs}])"


# ---------------------------------------------------------------------------
# Standalone metrics function
# ---------------------------------------------------------------------------

def compute_snn_metrics(
    spikes: Tensor,
    membrane: Optional[Tensor] = None,
    targets: Optional[Tensor] = None,
    eps: float = 1e-8,
    max_per_class: int = 20,
) -> Dict[str, float]:
    """Compute comprehensive monitoring metrics for SNN outputs.

    All reductions are performed in fp32 for AMP safety.  This function is
    designed to be called inside ``torch.no_grad()`` but will wrap itself if
    needed.

    Parameters
    ----------
    spikes : Tensor
        ``(B, T, N)`` spike tensor (batch-first).
    membrane : Tensor or None
        ``(B, T, N)`` membrane potentials.  Omit if not tracked.
    targets : Tensor or None
        ``(B,)`` integer class labels.  Omit for unsupervised metrics.
    eps : float
        Epsilon for log / division.
    max_per_class : int
        Maximum number of classes for per-class spike reports.

    Returns
    -------
    metrics : dict[str, float]
        Flat dictionary of named scalar metrics.  All values are plain Python
        floats (already detached).
    """
    metrics: Dict[str, float] = {}

    with torch.no_grad():
        spikes_f: Tensor = spikes.float()  # fp32
        B, T, N = spikes_f.shape

        # -- 1. Classification accuracy ------------------------------------
        spike_counts: Tensor = spikes_f.sum(dim=1)  # (B, N)
        if targets is not None:
            predictions = spike_counts.argmax(dim=-1)  # (B,)
            metrics["accuracy"] = (predictions == targets).float().mean().item()
        else:
            metrics["accuracy"] = float("nan")

        # -- 2. Spike-rate statistics --------------------------------------
        per_neuron_rate: Tensor = spikes_f.mean(dim=1)  # (B, N)
        global_rate: Tensor = spikes_f.mean()
        metrics["spike_rate"] = global_rate.item()
        metrics["spike_rate_std"] = per_neuron_rate.mean(dim=0).std().item()

        # -- 3. Temporal sparsity ------------------------------------------
        # Fraction of (B, T) time-steps with zero total spikes.
        spikes_per_step: Tensor = spikes_f.sum(dim=2)  # (B, T)
        silent_steps: Tensor = (spikes_per_step == 0).float()
        metrics["temporal_sparsity"] = silent_steps.mean().item()

        # -- 4. Output entropy ---------------------------------------------
        probs: Tensor = F.softmax(spike_counts, dim=-1)
        log_probs: Tensor = torch.clamp(probs, min=eps).log()
        entropy: Tensor = -(probs * log_probs).sum(dim=-1).mean()
        metrics["output_entropy"] = entropy.item()

        # -- 5. Confidence -------------------------------------------------
        confidence: Tensor = probs.max(dim=-1).values.mean()
        metrics["confidence"] = confidence.item()

        # -- 6. Dead / saturated neurons -----------------------------------
        total_per_neuron: Tensor = spikes_f.sum(dim=(0, 1))  # (N,)
        max_possible: float = float(B * T)
        dead: Tensor = (total_per_neuron == 0).float()
        saturated: Tensor = (
            total_per_neuron >= max_possible * 0.9
        ).float()
        metrics["dead_neuron_fraction"] = dead.mean().item()
        metrics["saturated_neuron_fraction"] = saturated.mean().item()

        # -- 7. Membrane statistics ----------------------------------------
        if membrane is not None:
            mem_f: Tensor = membrane.float()
            metrics["membrane_mean"] = mem_f.mean().item()
            metrics["membrane_std"] = mem_f.std().item()
            metrics["membrane_max"] = mem_f.abs().max().item()
        else:
            metrics["membrane_mean"] = float("nan")
            metrics["membrane_std"] = float("nan")
            metrics["membrane_max"] = float("nan")

        # -- 8. Per-class spike counts ------------------------------------
        if targets is not None:
            num_classes: int = min(N, max_per_class)
            per_class: Dict[str, float] = {}
            for c in range(num_classes):
                mask = targets == c
                if mask.any():
                    class_counts = spike_counts[mask, c]
                    per_class[f"class_{c}_mean_spikes"] = (
                        class_counts.mean().item()
                    )
                else:
                    per_class[f"class_{c}_mean_spikes"] = 0.0
            metrics["per_class_spikes"] = per_class  # type: ignore[assignment]
        else:
            metrics["per_class_spikes"] = {}  # type: ignore[assignment]

    return metrics


# ===========================================================================
# Self-test suite
# ===========================================================================

def _section(title: str) -> None:
    """Print a bold section header."""
    width = 72
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _check(
    condition: bool,
    description: str,
    results: Dict[str, List[str]],
    group: str,
) -> None:
    """Record a single check result."""
    status = "PASS" if condition else "FAIL"
    tag = f"[{status}]"
    line = f"  {tag:8s} {description}"
    print(line)
    results.setdefault(group, []).append(f"{status}: {description}")


def _run_self_tests() -> None:
    """Comprehensive self-test exercising every loss term and the composer.

    Exit code 0 if all tests pass, 1 otherwise.
    """
    # -- path setup --------------------------------------------------------
    # Insert the project root (4 directories up from this file) so that
    # ``import brain_ai`` works when running standalone.
    this_file = Path(__file__).resolve()
    project_root = this_file.parents[4]  # assets -> skill -> skills -> brain-ai-dev -> project
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    print(f"Project root: {project_root}")
    print(f"PyTorch version: {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    has_cuda = device.type == "cuda"

    results: Dict[str, List[str]] = {}
    all_passed = True

    # Helper to set all_passed = False on failure.
    def check(cond: bool, desc: str, group: str) -> None:
        nonlocal all_passed
        _check(cond, desc, results, group)
        if not cond:
            all_passed = False

    # -- synthetic data factory --------------------------------------------
    def make_spikes(
        B: int = 4,
        T: int = 25,
        N: int = 10,
        rate: float = 0.1,
        device: torch.device = device,
        requires_grad: bool = False,
    ) -> Tensor:
        """Generate random binary spike tensor."""
        s = (torch.rand(B, T, N, device=device) < rate).float()
        if requires_grad:
            s = s.detach().requires_grad_(True)
        return s

    def make_membrane(
        B: int = 4,
        T: int = 25,
        N: int = 10,
        device: torch.device = device,
    ) -> Tensor:
        return torch.randn(B, T, N, device=device) * 0.5

    def make_targets(B: int = 4, C: int = 10, device: torch.device = device) -> Tensor:
        return torch.randint(0, C, (B,), device=device)

    config = LossConfig()

    # ===================================================================
    # Group 1: ProbSpikesLoss
    # ===================================================================
    _section("ProbSpikesLoss")
    group = "ProbSpikesLoss"

    try:
        ps = ProbSpikesLoss(config)

        # 1a: Basic forward (softmax mode).
        spikes = make_spikes()
        targets = make_targets()
        loss, diag = ps.compute(spikes, None, targets)
        check(loss.dim() == 0, "Loss is scalar", group)
        check(torch.isfinite(loss).item(), "Loss is finite", group)
        check("mean_counts" in diag, "Diagnostics contain mean_counts", group)
        check("max_counts" in diag, "Diagnostics contain max_counts", group)
        check("count_entropy" in diag, "Diagnostics contain count_entropy", group)
        check(
            "correct_class_count_mean" in diag,
            "Diagnostics contain correct_class_count_mean",
            group,
        )

        # 1b: Normalize mode.
        cfg_norm = LossConfig(probspikes_mode="normalize")
        ps_norm = ProbSpikesLoss(cfg_norm)
        loss_norm, diag_norm = ps_norm.compute(spikes, None, targets)
        check(torch.isfinite(loss_norm).item(), "Normalize mode: finite loss", group)
        check(loss_norm.item() >= 0, "Normalize mode: non-negative loss", group)

        # 1c: Known counts -- class 0 dominates.
        B_test, T_test, C_test = 2, 10, 5
        spikes_known = torch.zeros(B_test, T_test, C_test, device=device)
        spikes_known[:, :, 0] = 1.0  # all spikes in class 0
        targets_known = torch.zeros(B_test, dtype=torch.long, device=device)
        loss_known, _ = ps.compute(spikes_known, None, targets_known)
        check(
            loss_known.item() < 0.1,
            f"Known correct counts: low loss ({loss_known.item():.4f} < 0.1)",
            group,
        )

        # 1d: Known counts -- wrong class dominant.
        targets_wrong = torch.ones(B_test, dtype=torch.long, device=device)  # class 1
        loss_wrong, _ = ps.compute(spikes_known, None, targets_wrong)
        check(
            loss_wrong.item() > loss_known.item(),
            f"Wrong class: higher loss ({loss_wrong.item():.4f} > {loss_known.item():.4f})",
            group,
        )

        # 1e: Temperature effect.
        cfg_hot = LossConfig(probspikes_temperature=10.0)
        ps_hot = ProbSpikesLoss(cfg_hot)
        loss_hot, _ = ps_hot.compute(spikes, None, targets)
        check(
            torch.isfinite(loss_hot).item(),
            "High temperature: finite loss",
            group,
        )

        # 1f: Zero spikes -- should not crash.
        spikes_zero = torch.zeros(4, 25, 10, device=device)
        targets_zero = make_targets()
        loss_zero, diag_zero = ps.compute(spikes_zero, None, targets_zero)
        check(
            torch.isfinite(loss_zero).item(),
            "Zero spikes: finite loss (no NaN)",
            group,
        )

        # 1g: Requires targets.
        try:
            ps.compute(spikes, None, None)
            check(False, "Should raise ValueError for None targets", group)
        except ValueError:
            check(True, "Raises ValueError for None targets", group)

    except Exception as exc:
        check(False, f"Unexpected exception: {exc}", group)
        traceback.print_exc()

    # ===================================================================
    # Group 2: SpikeRateRegularization
    # ===================================================================
    _section("SpikeRateRegularization")
    group = "SpikeRateRegularization"

    try:
        sr = SpikeRateRegularization(config)

        # 2a: Basic forward.
        spikes = make_spikes(rate=0.1)
        loss, diag = sr.compute(spikes, None, None)
        check(loss.dim() == 0, "Loss is scalar", group)
        check(torch.isfinite(loss).item(), "Loss is finite", group)
        check("mean_rate" in diag, "Diagnostics contain mean_rate", group)
        check("dead_fraction" in diag, "Diagnostics contain dead_fraction", group)
        check("saturated_fraction" in diag, "Diagnostics contain saturated_fraction", group)

        # 2b: Perfect rate -> near-zero loss.
        perfect = torch.full((4, 100, 50), config.target_rate, device=device)
        loss_perfect, diag_perfect = sr.compute(perfect, None, None)
        check(
            loss_perfect.item() < 0.001,
            f"Perfect rate: near-zero loss ({loss_perfect.item():.6f})",
            group,
        )

        # 2c: Dead neurons -> high loss.
        dead_spikes = torch.zeros(4, 100, 50, device=device)
        loss_dead, diag_dead = sr.compute(dead_spikes, None, None)
        check(
            loss_dead.item() > loss_perfect.item(),
            f"Dead neurons: higher loss ({loss_dead.item():.6f} > {loss_perfect.item():.6f})",
            group,
        )
        check(
            diag_dead.get("dead_fraction", 0) > 0.9,
            f"Dead fraction high ({diag_dead.get('dead_fraction', 0):.2f})",
            group,
        )

        # 2d: Saturated neurons -> high loss.
        saturated_spikes = torch.ones(4, 100, 50, device=device)
        loss_sat, diag_sat = sr.compute(saturated_spikes, None, None)
        check(
            loss_sat.item() > loss_perfect.item(),
            f"Saturated: higher loss ({loss_sat.item():.6f})",
            group,
        )
        check(
            diag_sat.get("saturated_fraction", 0) > 0.9,
            f"Saturated fraction high ({diag_sat.get('saturated_fraction', 0):.2f})",
            group,
        )

        # 2e: Per-layer target override.
        # Set min/max range wide enough to encompass the per-layer target of 0.5.
        cfg_pl = LossConfig(
            target_rate=0.1,
            min_rate=0.01,
            max_rate=0.9,
            per_layer_targets={"layer_x": 0.5},
        )
        sr_pl = SpikeRateRegularization(cfg_pl)
        half_spikes = torch.full((4, 100, 50), 0.5, device=device)
        loss_pl, _ = sr_pl.compute(half_spikes, None, None, layer_name="layer_x")
        check(
            loss_pl.item() < 0.01,
            f"Per-layer target=0.5 with rate=0.5: near-zero ({loss_pl.item():.6f})",
            group,
        )

    except Exception as exc:
        check(False, f"Unexpected exception: {exc}", group)
        traceback.print_exc()

    # ===================================================================
    # Group 3: TemporalConsistencyLoss
    # ===================================================================
    _section("TemporalConsistencyLoss")
    group = "TemporalConsistencyLoss"

    try:
        # 3a: L2 mode (default).
        tc_l2 = TemporalConsistencyLoss(LossConfig(temporal_penalty="l2"))
        spikes = make_spikes(T=50)
        loss_l2, diag_l2 = tc_l2.compute(spikes, None, None)
        check(loss_l2.dim() == 0, "L2: scalar loss", group)
        check(torch.isfinite(loss_l2).item(), "L2: finite loss", group)
        check("mean_temporal_var" in diag_l2, "L2: has mean_temporal_var", group)

        # 3b: L1 mode.
        tc_l1 = TemporalConsistencyLoss(LossConfig(temporal_penalty="l1"))
        loss_l1, _ = tc_l1.compute(spikes, None, None)
        check(torch.isfinite(loss_l1).item(), "L1: finite loss", group)

        # 3c: Variance mode.
        tc_var = TemporalConsistencyLoss(
            LossConfig(temporal_penalty="variance", temporal_window=5)
        )
        loss_var, diag_var = tc_var.compute(spikes, None, None)
        check(torch.isfinite(loss_var).item(), "Variance: finite loss", group)
        check("mean_temporal_var" in diag_var, "Variance: has mean_temporal_var", group)

        # 3d: Flicker pattern -> high L2 loss.
        B_f, T_f, N_f = 4, 50, 10
        flicker = torch.zeros(B_f, T_f, N_f, device=device)
        flicker[:, 0::2, :] = 1.0  # every other step fires
        loss_flicker_l2, _ = tc_l2.compute(flicker, None, None)

        # Constant pattern -> low loss.
        constant = torch.full((B_f, T_f, N_f), 0.0, device=device)
        loss_const_l2, _ = tc_l2.compute(constant, None, None)
        check(
            loss_flicker_l2.item() > loss_const_l2.item(),
            f"Flicker > constant ({loss_flicker_l2.item():.6f} > {loss_const_l2.item():.6f})",
            group,
        )

        # 3e: Short T (< window) -> zero loss for variance mode.
        short_spikes = make_spikes(T=3)
        loss_short, _ = tc_var.compute(short_spikes, None, None)
        check(
            loss_short.item() == 0.0,
            "Short T < window: zero variance loss",
            group,
        )

        # 3f: T=1 for l1/l2 -> zero.
        tiny = make_spikes(T=1)
        loss_tiny, _ = tc_l2.compute(tiny, None, None)
        check(loss_tiny.item() == 0.0, "T=1: zero l2 loss", group)

    except Exception as exc:
        check(False, f"Unexpected exception: {exc}", group)
        traceback.print_exc()

    # ===================================================================
    # Group 4: ISIRegularization
    # ===================================================================
    _section("ISIRegularization")
    group = "ISIRegularization"

    try:
        isi = ISIRegularization(config)

        # 4a: Basic forward.
        spikes = make_spikes(T=50)
        loss, diag = isi.compute(spikes, None, None)
        check(loss.dim() == 0, "Loss is scalar", group)
        check(torch.isfinite(loss).item(), "Loss is finite", group)
        check("mean_penalty" in diag, "Has mean_penalty", group)
        check("burst_fraction" in diag, "Has burst_fraction", group)

        # 4b: Burst pattern -> high penalty.
        B_b, T_b, N_b = 4, 50, 20
        burst = torch.zeros(B_b, T_b, N_b, device=device)
        # Neurons fire in tight bursts: consecutive spikes.
        burst[:, 0:5, :] = 1.0
        burst[:, 20:25, :] = 1.0
        burst[:, 40:45, :] = 1.0
        loss_burst, diag_burst = isi.compute(burst, None, None)

        # 4c: Regular pattern -> low penalty.
        regular = torch.zeros(B_b, T_b, N_b, device=device)
        # Neurons fire every 10 steps (well-spaced).
        regular[:, 0::10, :] = 1.0
        loss_regular, diag_regular = isi.compute(regular, None, None)

        check(
            loss_burst.item() > loss_regular.item(),
            f"Burst > regular penalty ({loss_burst.item():.6f} > {loss_regular.item():.6f})",
            group,
        )

        # 4d: No spikes -> zero penalty.
        no_spikes = torch.zeros(B_b, T_b, N_b, device=device)
        loss_no, _ = isi.compute(no_spikes, None, None)
        check(
            loss_no.item() == 0.0,
            "No spikes: zero penalty",
            group,
        )

        # 4e: Rectangular kernel.
        cfg_rect = LossConfig(isi_kernel_type="rectangular", isi_refractory_window=3)
        isi_rect = ISIRegularization(cfg_rect)
        loss_rect, _ = isi_rect.compute(burst, None, None)
        check(
            torch.isfinite(loss_rect).item(),
            "Rectangular kernel: finite loss",
            group,
        )
        check(
            loss_rect.item() > 0,
            f"Rectangular kernel on bursts: positive ({loss_rect.item():.6f})",
            group,
        )

        # 4f: Exponential kernel with larger window.
        cfg_wide = LossConfig(isi_refractory_window=10, isi_kernel_type="exponential")
        isi_wide = ISIRegularization(cfg_wide)
        loss_wide, _ = isi_wide.compute(burst, None, None)
        check(
            torch.isfinite(loss_wide).item(),
            "Wide window: finite loss",
            group,
        )

        # 4g: Short T.
        short = make_spikes(T=1)
        loss_short, _ = isi.compute(short, None, None)
        check(
            loss_short.item() == 0.0,
            "T=1: zero loss",
            group,
        )

    except Exception as exc:
        check(False, f"Unexpected exception: {exc}", group)
        traceback.print_exc()

    # ===================================================================
    # Group 5: MembranePotentialRegularization
    # ===================================================================
    _section("MembranePotentialRegularization")
    group = "MembranePotentialRegularization"

    try:
        mr = MembranePotentialRegularization(config)

        # 5a: Normal membrane -> low loss.
        spikes = make_spikes()
        membrane_normal = make_membrane()  # std=0.5, mostly within 1.5
        loss_normal, diag_normal = mr.compute(spikes, membrane_normal, None)
        check(loss_normal.dim() == 0, "Scalar loss", group)
        check(torch.isfinite(loss_normal).item(), "Finite loss", group)
        check("mean_membrane" in diag_normal, "Has mean_membrane", group)
        check("max_membrane" in diag_normal, "Has max_membrane", group)
        check("explosion_fraction" in diag_normal, "Has explosion_fraction", group)

        # 5b: Exploding membrane -> high loss.
        membrane_explode = torch.randn(4, 25, 10, device=device) * 10.0
        loss_explode, diag_explode = mr.compute(spikes, membrane_explode, None)
        check(
            loss_explode.item() > loss_normal.item(),
            f"Exploding > normal ({loss_explode.item():.4f} > {loss_normal.item():.4f})",
            group,
        )
        check(
            diag_explode.get("explosion_fraction", 0) > 0.5,
            f"Explosion fraction high ({diag_explode.get('explosion_fraction', 0):.2f})",
            group,
        )

        # 5c: Exactly at threshold -> zero loss.
        membrane_exact = torch.full(
            (4, 25, 10), config.membrane_max, device=device
        )
        loss_exact, _ = mr.compute(spikes, membrane_exact, None)
        check(
            abs(loss_exact.item()) < 1e-6,
            f"At threshold: ~zero loss ({loss_exact.item():.8f})",
            group,
        )

        # 5d: None membrane -> zero loss.
        loss_none, diag_none = mr.compute(spikes, None, None)
        check(loss_none.item() == 0.0, "None membrane: zero loss", group)
        check(len(diag_none) == 0, "None membrane: empty diag", group)

    except Exception as exc:
        check(False, f"Unexpected exception: {exc}", group)
        traceback.print_exc()

    # ===================================================================
    # Group 6: SNNLossComposer
    # ===================================================================
    _section("SNNLossComposer")
    group = "SNNLossComposer"

    try:
        # 6a: Default config -- all terms active.
        composer = SNNLossComposer(config)
        spikes = make_spikes(T=25, N=10)
        membrane = make_membrane(T=25, N=10)
        targets = make_targets(C=10)
        total_loss, info = composer(spikes, membrane, targets)

        check(total_loss.dim() == 0, "Total loss is scalar", group)
        check(torch.isfinite(total_loss).item(), "Total loss is finite", group)
        check("total" in info, "Info has 'total'", group)
        check("components" in info, "Info has 'components'", group)
        check("diagnostics" in info, "Info has 'diagnostics'", group)
        check("weights" in info, "Info has 'weights'", group)

        # Check that all expected terms appear.
        expected_terms = {"probspikes", "rate_reg", "temporal", "isi_reg", "membrane_reg"}
        actual_terms = set(info["components"].keys())
        check(
            expected_terms == actual_terms,
            f"All terms present: {actual_terms}",
            group,
        )

        # 6b: Disabled terms.
        cfg_minimal = LossConfig(
            w_probspikes=1.0,
            w_rate=0.0,
            w_temporal=0.0,
            w_isi=0.0,
            w_membrane=0.0,
        )
        composer_min = SNNLossComposer(cfg_minimal)
        loss_min, info_min = composer_min(spikes, membrane, targets)
        check(
            len(info_min["components"]) == 1,
            "Only ProbSpikes when others are 0",
            group,
        )
        check(
            "probspikes" in info_min["components"],
            "ProbSpikes present in minimal config",
            group,
        )

        # 6c: get_effective_magnitudes.
        magnitudes = composer.get_effective_magnitudes()
        check(
            len(magnitudes) == len(expected_terms),
            f"Magnitudes for all terms ({len(magnitudes)})",
            group,
        )
        for name, mag in magnitudes.items():
            check(
                isinstance(mag, float) and math.isfinite(mag),
                f"Magnitude '{name}' is finite float ({mag:.6f})",
                group,
            )

        # 6d: log_summary doesn't crash.
        summary = composer.log_summary(prefix="  ")
        check(
            len(summary) > 50,
            f"Summary is non-trivial ({len(summary)} chars)",
            group,
        )
        print(summary)

        # 6e: repr.
        r = repr(composer)
        check("SNNLossComposer" in r, f"repr: {r}", group)

    except Exception as exc:
        check(False, f"Unexpected exception: {exc}", group)
        traceback.print_exc()

    # ===================================================================
    # Group 7: AMP (autocast) safety
    # ===================================================================
    _section("AMP Safety")
    group = "AMP"

    try:
        amp_available = has_cuda or hasattr(torch, "autocast")
        if not amp_available:
            print("  (skipping AMP tests -- no CUDA and no torch.autocast)")
        else:
            amp_device = "cuda" if has_cuda else "cpu"
            terms_to_test: List[Tuple[str, LossTerm]] = [
                ("ProbSpikes", ProbSpikesLoss(config)),
                ("RateReg", SpikeRateRegularization(config)),
                ("Temporal", TemporalConsistencyLoss(config)),
                ("ISI", ISIRegularization(config)),
                ("Membrane", MembranePotentialRegularization(config)),
            ]

            spikes_amp = make_spikes(B=4, T=25, N=10).to(device)
            membrane_amp = make_membrane(B=4, T=25, N=10).to(device)
            targets_amp = make_targets(B=4, C=10).to(device)

            for term_label, term in terms_to_test:
                try:
                    with torch.autocast(device_type=amp_device, dtype=torch.float16):
                        loss_amp, diag_amp = term.compute(
                            spikes_amp, membrane_amp, targets_amp
                        )
                    check(
                        torch.isfinite(loss_amp).item(),
                        f"{term_label}: finite under autocast",
                        group,
                    )
                except Exception as e:
                    check(False, f"{term_label}: autocast failed -- {e}", group)

            # Full composer under autocast.
            try:
                composer_amp = SNNLossComposer(config)
                with torch.autocast(device_type=amp_device, dtype=torch.float16):
                    total_amp, info_amp = composer_amp(
                        spikes_amp, membrane_amp, targets_amp
                    )
                check(
                    torch.isfinite(total_amp).item(),
                    "Composer: finite under autocast",
                    group,
                )
            except Exception as e:
                check(False, f"Composer autocast failed -- {e}", group)

    except Exception as exc:
        check(False, f"Unexpected exception: {exc}", group)
        traceback.print_exc()

    # ===================================================================
    # Group 8: Different T values (stability)
    # ===================================================================
    _section("Varying T (stability)")
    group = "T_stability"

    try:
        composer_t = SNNLossComposer(config)

        for T_val in [10, 25, 50, 100]:
            spikes_t = make_spikes(B=4, T=T_val, N=10)
            membrane_t = make_membrane(B=4, T=T_val, N=10)
            targets_t = make_targets(B=4, C=10)
            loss_t, info_t = composer_t(spikes_t, membrane_t, targets_t)
            finite = torch.isfinite(loss_t).item()
            not_huge = abs(loss_t.item()) < 1e6
            check(
                finite and not_huge,
                f"T={T_val:4d}: finite and bounded (loss={loss_t.item():.4f})",
                group,
            )

    except Exception as exc:
        check(False, f"Unexpected exception: {exc}", group)
        traceback.print_exc()

    # ===================================================================
    # Group 9: Gradient flow
    # ===================================================================
    _section("Gradient Flow")
    group = "Gradients"

    try:
        composer_grad = SNNLossComposer(config)

        # Spikes with requires_grad=True (leaf tensor).
        spikes_grad = make_spikes(B=4, T=25, N=10, requires_grad=True)
        # Create membrane as a leaf tensor with values that exceed the
        # membrane_max threshold (1.5) so the membrane regularisation
        # term produces a non-zero gradient.
        membrane_grad = (
            torch.randn(4, 25, 10, device=device) * 3.0
        ).detach().requires_grad_(True)
        targets_grad = make_targets(B=4, C=10)

        total_grad, _ = composer_grad(spikes_grad, membrane_grad, targets_grad)
        total_grad.backward()

        spike_grad_ok = (
            spikes_grad.grad is not None
            and torch.isfinite(spikes_grad.grad).all().item()
            and spikes_grad.grad.abs().sum().item() > 0
        )
        check(spike_grad_ok, "Spike gradients: present, finite, non-zero", group)

        membrane_grad_ok = (
            membrane_grad.grad is not None
            and torch.isfinite(membrane_grad.grad).all().item()
            and membrane_grad.grad.abs().sum().item() > 0
        )
        check(
            membrane_grad_ok,
            "Membrane gradients: present, finite, non-zero",
            group,
        )

        # Verify no NaN in gradients.
        if spikes_grad.grad is not None:
            check(
                not torch.isnan(spikes_grad.grad).any().item(),
                "No NaN in spike gradients",
                group,
            )
        else:
            check(False, "No NaN in spike gradients (grad is None)", group)

        if membrane_grad.grad is not None:
            check(
                not torch.isnan(membrane_grad.grad).any().item(),
                "No NaN in membrane gradients",
                group,
            )
        else:
            check(False, "No NaN in membrane gradients (grad is None)", group)

    except Exception as exc:
        check(False, f"Unexpected exception: {exc}", group)
        traceback.print_exc()

    # ===================================================================
    # Group 10: compute_snn_metrics
    # ===================================================================
    _section("compute_snn_metrics")
    group = "Metrics"

    try:
        spikes_m = make_spikes(B=8, T=25, N=10)
        membrane_m = make_membrane(B=8, T=25, N=10)
        targets_m = make_targets(B=8, C=10)

        metrics = compute_snn_metrics(spikes_m, membrane_m, targets_m)

        expected_keys = [
            "accuracy",
            "spike_rate",
            "spike_rate_std",
            "temporal_sparsity",
            "output_entropy",
            "confidence",
            "dead_neuron_fraction",
            "saturated_neuron_fraction",
            "membrane_mean",
            "membrane_std",
            "membrane_max",
            "per_class_spikes",
        ]
        for key in expected_keys:
            check(key in metrics, f"Metric '{key}' present", group)

        # Sanity ranges.
        check(
            0.0 <= metrics.get("spike_rate", -1) <= 1.0,
            f"spike_rate in [0,1]: {metrics.get('spike_rate', -1):.4f}",
            group,
        )
        check(
            0.0 <= metrics.get("temporal_sparsity", -1) <= 1.0,
            f"temporal_sparsity in [0,1]: {metrics.get('temporal_sparsity', -1):.4f}",
            group,
        )
        check(
            metrics.get("output_entropy", -1) >= 0,
            f"output_entropy >= 0: {metrics.get('output_entropy', -1):.4f}",
            group,
        )

        # Without targets.
        metrics_no_tgt = compute_snn_metrics(spikes_m, membrane_m, None)
        check(
            math.isnan(metrics_no_tgt.get("accuracy", 0)),
            "No targets: accuracy is NaN",
            group,
        )

        # Without membrane.
        metrics_no_mem = compute_snn_metrics(spikes_m, None, targets_m)
        check(
            math.isnan(metrics_no_mem.get("membrane_mean", 0)),
            "No membrane: membrane_mean is NaN",
            group,
        )

    except Exception as exc:
        check(False, f"Unexpected exception: {exc}", group)
        traceback.print_exc()

    # ===================================================================
    # Group 11: Diagnostics completeness
    # ===================================================================
    _section("Diagnostics Completeness")
    group = "Diagnostics"

    try:
        composer_diag = SNNLossComposer(config)
        spikes_d = make_spikes(B=4, T=30, N=10)
        membrane_d = make_membrane(B=4, T=30, N=10)
        targets_d = make_targets(B=4, C=10)
        _, info_d = composer_diag(spikes_d, membrane_d, targets_d)

        diags = info_d["diagnostics"]

        # ProbSpikes diagnostics.
        ps_diag = diags.get("probspikes", {})
        ps_expected = {
            "mean_counts",
            "max_counts",
            "count_entropy",
            "correct_class_count_mean",
        }
        check(
            ps_expected.issubset(set(ps_diag.keys())),
            f"ProbSpikes diag keys: {set(ps_diag.keys())}",
            group,
        )

        # Rate reg diagnostics.
        rr_diag = diags.get("rate_reg", {})
        rr_expected = {
            "mean_rate",
            "max_rate",
            "min_rate",
            "dead_fraction",
            "saturated_fraction",
        }
        check(
            rr_expected.issubset(set(rr_diag.keys())),
            f"RateReg diag keys: {set(rr_diag.keys())}",
            group,
        )

        # Temporal diagnostics.
        tc_diag = diags.get("temporal", {})
        tc_expected = {"mean_temporal_var", "max_temporal_var"}
        check(
            tc_expected.issubset(set(tc_diag.keys())),
            f"Temporal diag keys: {set(tc_diag.keys())}",
            group,
        )

        # ISI diagnostics.
        isi_diag = diags.get("isi_reg", {})
        isi_expected = {"mean_penalty", "burst_fraction"}
        check(
            isi_expected.issubset(set(isi_diag.keys())),
            f"ISI diag keys: {set(isi_diag.keys())}",
            group,
        )

        # Membrane diagnostics.
        mem_diag = diags.get("membrane_reg", {})
        mem_expected = {"mean_membrane", "max_membrane", "explosion_fraction"}
        check(
            mem_expected.issubset(set(mem_diag.keys())),
            f"Membrane diag keys: {set(mem_diag.keys())}",
            group,
        )

        # Diagnostics disabled.
        cfg_no_diag = LossConfig(log_diagnostics=False)
        composer_nodiag = SNNLossComposer(cfg_no_diag)
        _, info_nodiag = composer_nodiag(spikes_d, membrane_d, targets_d)
        for name, d in info_nodiag["diagnostics"].items():
            check(
                len(d) == 0,
                f"No diagnostics for '{name}' when disabled",
                group,
            )

    except Exception as exc:
        check(False, f"Unexpected exception: {exc}", group)
        traceback.print_exc()

    # ===================================================================
    # Group 12: LossConfig validation
    # ===================================================================
    _section("LossConfig Validation")
    group = "ConfigValidation"

    try:
        # Invalid temperature.
        try:
            LossConfig(probspikes_temperature=0.0)
            check(False, "Should reject temperature=0", group)
        except ValueError:
            check(True, "Rejects temperature=0", group)

        try:
            LossConfig(probspikes_temperature=-1.0)
            check(False, "Should reject negative temperature", group)
        except ValueError:
            check(True, "Rejects negative temperature", group)

        # Invalid mode.
        try:
            LossConfig(probspikes_mode="invalid")
            check(False, "Should reject invalid probspikes_mode", group)
        except ValueError:
            check(True, "Rejects invalid probspikes_mode", group)

        # Invalid penalty.
        try:
            LossConfig(temporal_penalty="cosine")
            check(False, "Should reject invalid temporal_penalty", group)
        except ValueError:
            check(True, "Rejects invalid temporal_penalty", group)

        # Invalid kernel type.
        try:
            LossConfig(isi_kernel_type="gaussian")
            check(False, "Should reject invalid isi_kernel_type", group)
        except ValueError:
            check(True, "Rejects invalid isi_kernel_type", group)

        # min_rate >= max_rate.
        try:
            LossConfig(min_rate=0.5, max_rate=0.3)
            check(False, "Should reject min_rate >= max_rate", group)
        except ValueError:
            check(True, "Rejects min_rate >= max_rate", group)

        # refractory_window < 1.
        try:
            LossConfig(isi_refractory_window=0)
            check(False, "Should reject refractory_window=0", group)
        except ValueError:
            check(True, "Rejects refractory_window=0", group)

        # temporal_window < 2.
        try:
            LossConfig(temporal_window=1)
            check(False, "Should reject temporal_window=1", group)
        except ValueError:
            check(True, "Rejects temporal_window=1", group)

        # membrane_max <= 0.
        try:
            LossConfig(membrane_max=0.0)
            check(False, "Should reject membrane_max=0", group)
        except ValueError:
            check(True, "Rejects membrane_max=0", group)

        # Valid config -- no exception.
        try:
            LossConfig(
                w_probspikes=2.0,
                w_rate=0.5,
                target_rate=0.2,
                probspikes_mode="normalize",
                temporal_penalty="variance",
                isi_kernel_type="rectangular",
                per_layer_targets={"enc": 0.05, "dec": 0.15},
            )
            check(True, "Valid custom config accepted", group)
        except Exception as e:
            check(False, f"Valid config rejected: {e}", group)

    except Exception as exc:
        check(False, f"Unexpected exception: {exc}", group)
        traceback.print_exc()

    # ===================================================================
    # Group 13: Edge cases and robustness
    # ===================================================================
    _section("Edge Cases & Robustness")
    group = "EdgeCases"

    try:
        composer_edge = SNNLossComposer(config)

        # 13a: Very large B.
        spikes_big_b = make_spikes(B=64, T=10, N=10)
        membrane_big_b = make_membrane(B=64, T=10, N=10)
        targets_big_b = make_targets(B=64, C=10)
        loss_bb, _ = composer_edge(spikes_big_b, membrane_big_b, targets_big_b)
        check(
            torch.isfinite(loss_bb).item(),
            f"Large B=64: finite (loss={loss_bb.item():.4f})",
            group,
        )

        # 13b: Very large N.
        spikes_big_n = make_spikes(B=4, T=10, N=512)
        membrane_big_n = make_membrane(B=4, T=10, N=512)
        targets_big_n = make_targets(B=4, C=512)
        loss_bn, _ = composer_edge(spikes_big_n, membrane_big_n, targets_big_n)
        check(
            torch.isfinite(loss_bn).item(),
            f"Large N=512: finite (loss={loss_bn.item():.4f})",
            group,
        )

        # 13c: Single element batch.
        spikes_1 = make_spikes(B=1, T=25, N=10)
        membrane_1 = make_membrane(B=1, T=25, N=10)
        targets_1 = make_targets(B=1, C=10)
        loss_1, _ = composer_edge(spikes_1, membrane_1, targets_1)
        check(
            torch.isfinite(loss_1).item(),
            f"B=1: finite (loss={loss_1.item():.4f})",
            group,
        )

        # 13d: All ones (saturated).
        spikes_ones = torch.ones(4, 25, 10, device=device)
        membrane_ones = torch.ones(4, 25, 10, device=device) * 5.0
        targets_ones = make_targets(B=4, C=10)
        loss_ones, _ = composer_edge(spikes_ones, membrane_ones, targets_ones)
        check(
            torch.isfinite(loss_ones).item(),
            f"All-ones: finite (loss={loss_ones.item():.4f})",
            group,
        )

        # 13e: All zeros.
        spikes_zeros = torch.zeros(4, 25, 10, device=device)
        membrane_zeros = torch.zeros(4, 25, 10, device=device)
        targets_zeros = make_targets(B=4, C=10)
        loss_zeros, _ = composer_edge(spikes_zeros, membrane_zeros, targets_zeros)
        check(
            torch.isfinite(loss_zeros).item(),
            f"All-zeros: finite (loss={loss_zeros.item():.4f})",
            group,
        )

        # 13f: Half-precision spikes (should still work due to .float()).
        if has_cuda:
            spikes_fp16 = make_spikes(B=4, T=25, N=10).half()
            membrane_fp16 = make_membrane(B=4, T=25, N=10).half()
            targets_fp16 = make_targets(B=4, C=10)
            loss_fp16, _ = composer_edge(spikes_fp16, membrane_fp16, targets_fp16)
            check(
                torch.isfinite(loss_fp16).item(),
                f"FP16 input: finite (loss={loss_fp16.item():.4f})",
                group,
            )
        else:
            print("  (skipping FP16 input test -- no CUDA)")

        # 13g: Negative membrane values (large negative).
        membrane_neg = torch.full((4, 25, 10), -5.0, device=device)
        mr_edge = MembranePotentialRegularization(config)
        loss_neg, diag_neg = mr_edge.compute(spikes_zeros, membrane_neg, None)
        check(
            loss_neg.item() > 0,
            f"Large negative membrane: positive loss ({loss_neg.item():.4f})",
            group,
        )

    except Exception as exc:
        check(False, f"Unexpected exception: {exc}", group)
        traceback.print_exc()

    # ===================================================================
    # Group 14: Performance sanity (timing)
    # ===================================================================
    _section("Performance Sanity")
    group = "Performance"

    try:
        composer_perf = SNNLossComposer(config)
        # Moderately large tensor.
        B_p, T_p, N_p = 32, 100, 256
        spikes_p = make_spikes(B=B_p, T=T_p, N=N_p)
        membrane_p = make_membrane(B=B_p, T=T_p, N=N_p)
        targets_p = make_targets(B=B_p, C=N_p)

        # Warm-up.
        _ = composer_perf(spikes_p, membrane_p, targets_p)

        # Time 10 iterations.
        n_iters = 10
        t0 = time.perf_counter()
        for _ in range(n_iters):
            loss_p, _ = composer_perf(spikes_p, membrane_p, targets_p)
        t1 = time.perf_counter()
        ms_per_iter = (t1 - t0) / n_iters * 1000

        check(
            ms_per_iter < 5000,
            f"Perf ({B_p}x{T_p}x{N_p}): {ms_per_iter:.1f} ms/iter (< 5000 ms)",
            group,
        )

        # Compare ISI conv1d vs hypothetical loop (we just check it is fast).
        isi_perf = ISIRegularization(config)
        t2 = time.perf_counter()
        for _ in range(n_iters):
            _ = isi_perf.compute(spikes_p, None, None)
        t3 = time.perf_counter()
        ms_isi = (t3 - t2) / n_iters * 1000
        check(
            ms_isi < 2000,
            f"ISI conv1d ({B_p}x{T_p}x{N_p}): {ms_isi:.1f} ms/iter (< 2000 ms)",
            group,
        )

    except Exception as exc:
        check(False, f"Unexpected exception: {exc}", group)
        traceback.print_exc()

    # ===================================================================
    # Final report
    # ===================================================================
    _section("FINAL REPORT")
    total_tests = 0
    total_passed = 0
    total_failed = 0

    for grp, items in results.items():
        passed = sum(1 for i in items if i.startswith("PASS"))
        failed = sum(1 for i in items if i.startswith("FAIL"))
        total_tests += len(items)
        total_passed += passed
        total_failed += failed
        status = "PASS" if failed == 0 else "FAIL"
        print(f"  [{status}] {grp:30s}  {passed}/{len(items)} passed")

    print()
    print(f"  Total: {total_passed}/{total_tests} passed, {total_failed} failed")
    print()

    if all_passed:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")

    sys.exit(0 if all_passed else 1)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    _run_self_tests()
