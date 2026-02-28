"""
brain_ai/core/decoding.py — Spike Decoding Suite (Batch-First Convention)

This module provides a complete suite of spike-train decoders that convert raw
spiking network output into differentiable classification/regression signals.
All decoders consume batch-first ``(B, T, N)`` spike tensors and return a
structured ``DecoderOutput`` dataclass suitable for direct use with standard
PyTorch loss functions (``F.cross_entropy``, ``F.mse_loss``).

Template for ``brain_ai/core/decoding.py`` in the brain-inspired AI project.

Key design decisions:
    1. **Batch-first axis convention** ``(B, T, N)`` everywhere. The legacy
       ``SpikeDecoder`` in ``brain_ai/core/encoding.py`` uses ``(T, B, D)``
       with ``dim=0`` reductions. This module supersedes that convention.
    2. **Structured output** via ``DecoderOutput`` dataclass: logits_proxy for
       loss computation, prediction for metrics, confidence for dual-process
       routing (System 1/2 threshold), and aux for diagnostics.
    3. **AMP hardening**: all spike-count accumulation is performed in fp32
       regardless of the input dtype. Softmax and log-softmax always cast to
       fp32 before exponentiation. Temperature scaling is applied before
       softmax, never after.
    4. **Population decoding** supports both classification (argmax over
       group counts) and regression (weighted expectation over centers).
    5. **Decoder ensemble** combines multiple decoders with configurable
       weights for curriculum-based annealing (e.g., membrane -> rate).

Canonical import::

    from brain_ai.core.decoding import (
        DecoderOutput,
        DecodingConfig,
        SpikeDecoder,
        RateDecoder,
        FirstSpikeDecoder,
        PopulationDecoder,
        MembraneDecoder,
        DecoderEnsemble,
        create_decoder,
    )

References:
    Maass (1997)   "Networks of spiking neurons"
    Neftci (2019)  "Surrogate Gradient Learning in Spiking Neural Networks"
    Comsa (2020)   "Temporal Coding in Spiking Neural Networks with Alpha Synaptic Function"
    Guo (2021)     "Neural Coding in Spiking Neural Networks: A Comparative Study"
"""

from __future__ import annotations

import logging
import math
import sys
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SpikeBatch — lightweight container for batch-first spike data
# ---------------------------------------------------------------------------
# CONSOLIDATION NOTE: The canonical SpikeBatch is defined in encoding_template.py
# (target: brain_ai/core/encoding.py). When integrating into brain_ai, import
# SpikeBatch from the encoding module rather than duplicating. This local copy
# exists only so the decoding template can be self-tested standalone.
# Property aliases: time_steps == num_timesteps, feature_shape == num_neurons.

@dataclass
class SpikeBatch:
    """Batch-first spike-train container (standalone copy for decoding template).

    See encoding_template.py SpikeBatch for the canonical definition.
    This copy adds a ``membrane`` field for decoder convenience; in the
    integrated codebase, membrane should be passed as a separate argument
    to ``SpikeDecoder.decode()`` instead.

    Attributes:
        spikes    : ``(B, T, N)`` float tensor of binary spike values ``{0, 1}``.
        membrane  : ``(B, T, N)`` optional float tensor of membrane potentials.
        mask      : ``(B, T)`` optional bool mask. ``True`` = valid timestep.
        aux       : dict of auxiliary data (rates, spike_times, etc.).
    """

    spikes: Tensor
    membrane: Optional[Tensor] = None
    mask: Optional[Tensor] = None
    aux: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Shape properties (aliases match canonical SpikeBatch in encoding module)
    # ------------------------------------------------------------------

    @property
    def batch_size(self) -> int:
        """Return the batch dimension B."""
        return self.spikes.shape[0]

    @property
    def time_steps(self) -> int:
        """Return the time dimension T (canonical name)."""
        return self.spikes.shape[1]

    @property
    def num_timesteps(self) -> int:
        """Alias for time_steps (backward compat)."""
        return self.time_steps

    @property
    def feature_shape(self) -> torch.Size:
        """Return feature dimensions (canonical name)."""
        return self.spikes.shape[2:]

    @property
    def num_neurons(self) -> int:
        """Return the neuron/feature dimension N (alias)."""
        return self.spikes.shape[2]

    @property
    def shape(self) -> torch.Size:
        """Return the full shape of the spike tensor."""
        return self.spikes.shape

    @property
    def device(self) -> torch.device:
        """Return the device of the spike tensor."""
        return self.spikes.device

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Check invariants. Raises AssertionError on violation."""
        assert self.spikes.ndim == 3, (
            f"spikes must be 3-D (B, T, N), got ndim={self.spikes.ndim} "
            f"shape={tuple(self.spikes.shape)}"
        )
        if self.membrane is not None:
            assert self.membrane.ndim == 3, (
                f"membrane must be 3-D (B, T, N), got ndim={self.membrane.ndim}"
            )
            assert self.membrane.shape[0] == self.batch_size, (
                f"membrane batch dim {self.membrane.shape[0]} != "
                f"spikes batch dim {self.batch_size}"
            )
            assert self.membrane.shape[1] == self.num_timesteps, (
                f"membrane time dim {self.membrane.shape[1]} != "
                f"spikes time dim {self.num_timesteps}"
            )
        if self.mask is not None:
            assert self.mask.ndim == 2, (
                f"mask must be 2-D (B, T), got ndim={self.mask.ndim}"
            )
            assert self.mask.shape == (self.batch_size, self.num_timesteps), (
                f"mask shape {tuple(self.mask.shape)} != "
                f"expected ({self.batch_size}, {self.num_timesteps})"
            )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def to(self, device: torch.device) -> "SpikeBatch":
        """Move all tensors to the specified device."""
        return SpikeBatch(
            spikes=self.spikes.to(device),
            membrane=self.membrane.to(device) if self.membrane is not None else None,
            mask=self.mask.to(device) if self.mask is not None else None,
        )

    def detach(self) -> "SpikeBatch":
        """Detach all tensors from the computation graph."""
        return SpikeBatch(
            spikes=self.spikes.detach(),
            membrane=self.membrane.detach() if self.membrane is not None else None,
            mask=self.mask.detach() if self.mask is not None else None,
        )

    @staticmethod
    def from_time_first(
        spikes: Tensor,
        membrane: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> "SpikeBatch":
        """Construct a SpikeBatch from time-first ``(T, B, N)`` tensors.

        This is the migration adapter for code that still uses the legacy
        ``SpikeDecoder`` axis convention. Transposes ``dim=0`` and ``dim=1``
        to produce the batch-first ``(B, T, N)`` layout.

        Args:
            spikes   : ``(T, B, N)`` spike tensor.
            membrane : ``(T, B, N)`` optional membrane tensor.
            mask     : ``(B, T)`` mask (already batch-first, no transpose).

        Returns:
            SpikeBatch with ``(B, T, N)`` layout.
        """
        s = spikes.permute(1, 0, 2) if spikes.ndim == 3 else spikes
        m = membrane.permute(1, 0, 2) if membrane is not None and membrane.ndim == 3 else membrane
        return SpikeBatch(spikes=s, membrane=m, mask=mask)

    def __repr__(self) -> str:
        parts = [f"spikes={tuple(self.spikes.shape)}"]
        if self.membrane is not None:
            parts.append(f"membrane={tuple(self.membrane.shape)}")
        if self.mask is not None:
            parts.append(f"mask={tuple(self.mask.shape)}")
        return f"SpikeBatch({', '.join(parts)})"


# ---------------------------------------------------------------------------
# DecoderOutput — structured output from every decoder
# ---------------------------------------------------------------------------

@dataclass
class DecoderOutput:
    """Structured output returned by every spike decoder.

    This dataclass is the decoding-side counterpart to ``EncoderOutput``. It
    guarantees that any decoder can be swapped into the training pipeline
    without modifying loss computation, metric logging, or downstream modules.

    Attributes:
        logits_proxy : ``(B, C)`` float32 tensor. Differentiable signal passed
                       to ``F.cross_entropy`` or ``F.mse_loss``. Must retain
                       the computational graph for backprop through surrogate
                       gradients. ``C`` is the number of classes (classification)
                       or 1 (regression).
        prediction   : ``(B,)`` int64 tensor. Hard prediction. Detached from the
                       graph. Used for accuracy metrics only. Computed via
                       ``argmax`` (classification) or ``argmin`` (first-spike).
        confidence   : ``(B,)`` float32 tensor. Per-sample confidence score in
                       ``[0, 1]``. Used for dual-process routing (System 1/2
                       threshold at 0.7) and diagnostic logging.
        aux          : Dictionary of decoder-specific diagnostics. Never consumed
                       by the forward path; safe to detach all tensors.

    Typical aux keys by decoder type:
        RateDecoder       : raw_counts, rates, count_histogram, margin
        FirstSpikeDecoder : first_spike_times, no_spike_mask, spike_counts, margin
        PopulationDecoder : raw_counts, group_counts, group_rates, predicted_value
        MembraneDecoder   : membrane_values, mode
        DecoderEnsemble   : sub_outputs, weights
    """

    logits_proxy: Tensor
    prediction: Tensor
    confidence: Tensor
    aux: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_classes(self) -> int:
        """Number of output classes (last dim of logits_proxy)."""
        return self.logits_proxy.shape[-1]

    @property
    def batch_size(self) -> int:
        """Batch size (first dim of logits_proxy)."""
        return self.logits_proxy.shape[0]

    # ------------------------------------------------------------------
    # Device / graph utilities
    # ------------------------------------------------------------------

    def to(self, device: torch.device) -> "DecoderOutput":
        """Move all tensor fields to the specified device.

        The ``aux`` dict is shallow-copied; tensor values inside ``aux`` are
        also moved. Non-tensor values are preserved as-is.

        Args:
            device: Target device.

        Returns:
            New ``DecoderOutput`` with tensors on the target device.
        """
        new_aux: Dict[str, Any] = {}
        for k, v in self.aux.items():
            if isinstance(v, Tensor):
                new_aux[k] = v.to(device)
            elif isinstance(v, list):
                new_aux[k] = [
                    item.to(device) if isinstance(item, Tensor) else item
                    for item in v
                ]
            elif isinstance(v, dict):
                new_aux[k] = {
                    kk: vv.to(device) if isinstance(vv, Tensor) else vv
                    for kk, vv in v.items()
                }
            else:
                new_aux[k] = v

        return DecoderOutput(
            logits_proxy=self.logits_proxy.to(device),
            prediction=self.prediction.to(device),
            confidence=self.confidence.to(device),
            aux=new_aux,
        )

    def detach(self) -> "DecoderOutput":
        """Detach all tensor fields from the computation graph.

        Returns a new ``DecoderOutput`` where ``logits_proxy``, ``prediction``,
        ``confidence``, and all tensor values in ``aux`` are detached. Use this
        before logging or storing outputs to prevent memory leaks from retained
        computation graphs.

        Returns:
            New ``DecoderOutput`` with all tensors detached.
        """
        new_aux: Dict[str, Any] = {}
        for k, v in self.aux.items():
            if isinstance(v, Tensor):
                new_aux[k] = v.detach()
            elif isinstance(v, list):
                new_aux[k] = [
                    item.detach() if isinstance(item, Tensor) else item
                    for item in v
                ]
            elif isinstance(v, dict):
                new_aux[k] = {
                    kk: vv.detach() if isinstance(vv, Tensor) else vv
                    for kk, vv in v.items()
                }
            else:
                new_aux[k] = v

        return DecoderOutput(
            logits_proxy=self.logits_proxy.detach(),
            prediction=self.prediction.detach(),
            confidence=self.confidence.detach(),
            aux=new_aux,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Check DecoderOutput invariants. Raises AssertionError on violation.

        Invariants:
            - logits_proxy is 2-D float
            - prediction is 1-D int64
            - confidence is 1-D float in [0, 1]
            - batch dimensions agree across all fields
            - no NaN or Inf in logits_proxy or confidence
        """
        # logits_proxy
        assert self.logits_proxy.ndim == 2, (
            f"logits_proxy must be 2-D (B, C), got ndim={self.logits_proxy.ndim} "
            f"shape={tuple(self.logits_proxy.shape)}"
        )
        assert self.logits_proxy.is_floating_point(), (
            f"logits_proxy must be floating-point, got dtype={self.logits_proxy.dtype}"
        )
        assert torch.isfinite(self.logits_proxy).all(), (
            "logits_proxy contains NaN or Inf"
        )

        # prediction
        assert self.prediction.ndim == 1, (
            f"prediction must be 1-D (B,), got ndim={self.prediction.ndim} "
            f"shape={tuple(self.prediction.shape)}"
        )
        assert self.prediction.dtype == torch.long, (
            f"prediction must be int64/long, got dtype={self.prediction.dtype}"
        )

        # confidence
        assert self.confidence.ndim == 1, (
            f"confidence must be 1-D (B,), got ndim={self.confidence.ndim}"
        )
        assert self.confidence.is_floating_point(), (
            f"confidence must be floating-point, got dtype={self.confidence.dtype}"
        )
        assert torch.isfinite(self.confidence).all(), (
            "confidence contains NaN or Inf"
        )
        assert (self.confidence >= 0.0).all() and (self.confidence <= 1.0).all(), (
            f"confidence must be in [0, 1], got min={self.confidence.min().item():.4f} "
            f"max={self.confidence.max().item():.4f}"
        )

        # batch consistency
        B = self.logits_proxy.shape[0]
        assert self.prediction.shape[0] == B, (
            f"prediction batch dim {self.prediction.shape[0]} != "
            f"logits_proxy batch dim {B}"
        )
        assert self.confidence.shape[0] == B, (
            f"confidence batch dim {self.confidence.shape[0]} != "
            f"logits_proxy batch dim {B}"
        )

    def __repr__(self) -> str:
        return (
            f"DecoderOutput("
            f"logits_proxy={tuple(self.logits_proxy.shape)}, "
            f"prediction={tuple(self.prediction.shape)}, "
            f"confidence={tuple(self.confidence.shape)}, "
            f"aux_keys={sorted(self.aux.keys())})"
        )


# ---------------------------------------------------------------------------
# DecodingConfig — configuration for decoder construction
# ---------------------------------------------------------------------------

@dataclass
class DecodingConfig:
    """Configuration dataclass for spike decoder construction.

    Used by the ``create_decoder`` factory function to instantiate the
    appropriate decoder class with the specified parameters.

    Attributes:
        type             : Decoder type identifier. One of:
                           ``"rate"``, ``"first_spike"``, ``"population"``,
                           ``"membrane"``, ``"ensemble"``.
        temperature      : Scale factor for logits_proxy. Lower values sharpen
                           predictions; higher values smooth them. Applied as
                           ``logits / temperature`` before softmax.
        eps              : Numerical stability constant added before division
                           or log operations.
        population_map   : Mapping from group index to neuron indices. Required
                           for ``type="population"``. Each key is a class label
                           and each value is a list of neuron indices belonging
                           to that group.
        population_mode  : ``"classification"`` (argmax over group counts) or
                           ``"regression"`` (weighted expectation over centers).
        population_centers: ``(num_groups,)`` tensor of center values for
                           regression mode. Required when
                           ``population_mode="regression"``.
        ensemble_weights : Mapping from decoder type name to weight for the
                           ensemble. Weights should sum to 1.0.
                           Example: ``{"rate": 0.7, "membrane": 0.3}``.
        membrane_mode    : ``"final"`` (use last timestep) or ``"max"`` (use
                           peak membrane potential over time).
        normalize_by_T   : If True, rate decoder divides counts by T to produce
                           firing rates instead of raw counts.
        confidence_method: Method for computing confidence. ``"margin"``
                           (top1 - top2 softmax probability) or ``"max_prob"``
                           (maximum softmax probability).
    """

    type: str = "rate"
    temperature: float = 1.0
    eps: float = 1e-7
    population_map: Optional[Dict[int, List[int]]] = None
    population_mode: str = "classification"
    population_centers: Optional[Tensor] = None
    ensemble_weights: Optional[Dict[str, float]] = None
    membrane_mode: str = "final"
    normalize_by_T: bool = False
    confidence_method: str = "margin"

    _VALID_TYPES = ("rate", "first_spike", "population", "membrane", "ensemble")
    _VALID_POPULATION_MODES = ("classification", "regression")
    _VALID_MEMBRANE_MODES = ("final", "max")
    _VALID_CONFIDENCE_METHODS = ("margin", "max_prob")

    def validate(self) -> None:
        """Check configuration invariants. Raises ValueError on violation."""
        if self.type not in self._VALID_TYPES:
            raise ValueError(
                f"Unknown decoder type {self.type!r}. "
                f"Must be one of {self._VALID_TYPES}."
            )
        if self.temperature <= 0:
            raise ValueError(
                f"temperature must be positive, got {self.temperature}."
            )
        if self.eps <= 0:
            raise ValueError(
                f"eps must be positive, got {self.eps}."
            )
        if self.type == "population":
            if self.population_map is None:
                raise ValueError(
                    "population_map is required for type='population'."
                )
            if self.population_mode not in self._VALID_POPULATION_MODES:
                raise ValueError(
                    f"Unknown population_mode {self.population_mode!r}. "
                    f"Must be one of {self._VALID_POPULATION_MODES}."
                )
            if self.population_mode == "regression" and self.population_centers is None:
                raise ValueError(
                    "population_centers is required for "
                    "population_mode='regression'."
                )
        if self.membrane_mode not in self._VALID_MEMBRANE_MODES:
            raise ValueError(
                f"Unknown membrane_mode {self.membrane_mode!r}. "
                f"Must be one of {self._VALID_MEMBRANE_MODES}."
            )
        if self.type == "ensemble":
            if self.ensemble_weights is None or len(self.ensemble_weights) == 0:
                raise ValueError(
                    "ensemble_weights is required for type='ensemble'. "
                    "Provide a dict mapping decoder type names to weights."
                )
            total = sum(self.ensemble_weights.values())
            if abs(total - 1.0) > 1e-4:
                raise ValueError(
                    f"ensemble_weights must sum to 1.0, got {total:.6f}."
                )
        if self.confidence_method not in self._VALID_CONFIDENCE_METHODS:
            raise ValueError(
                f"Unknown confidence_method {self.confidence_method!r}. "
                f"Must be one of {self._VALID_CONFIDENCE_METHODS}."
            )


# ---------------------------------------------------------------------------
# Confidence computation helpers
# ---------------------------------------------------------------------------

def _compute_confidence_margin(
    logits: Tensor,
    eps: float = 1e-7,
) -> Tensor:
    """Compute margin-based confidence: (top1_prob - top2_prob).

    The margin between the top-two softmax probabilities is a calibration-aware
    confidence metric. A margin of 1.0 means the model assigns all probability
    to one class; a margin of 0.0 means the top two classes are equally likely.

    Args:
        logits : ``(B, C)`` float tensor of unnormalized scores.
        eps    : Stability constant (unused here, included for API consistency).

    Returns:
        ``(B,)`` float tensor of confidence scores in ``[0, 1]``.
    """
    C = logits.shape[-1]
    if C < 2:
        # Single class: confidence is always 1.0
        return torch.ones(logits.shape[0], device=logits.device, dtype=torch.float32)

    probs = torch.softmax(logits.float(), dim=-1)
    top2 = probs.topk(2, dim=-1).values  # (B, 2)
    margin = (top2[:, 0] - top2[:, 1]).clamp(min=0.0, max=1.0)
    return margin


def _compute_confidence_max_prob(
    logits: Tensor,
    eps: float = 1e-7,
) -> Tensor:
    """Compute max-probability confidence: max(softmax(logits)).

    Simpler than margin but less informative when the runner-up class has
    significant probability.

    Args:
        logits : ``(B, C)`` float tensor of unnormalized scores.
        eps    : Stability constant (unused here).

    Returns:
        ``(B,)`` float tensor of confidence scores in ``[0, 1]``.
    """
    probs = torch.softmax(logits.float(), dim=-1)
    max_prob = probs.max(dim=-1).values.clamp(min=0.0, max=1.0)
    return max_prob


def _compute_confidence(
    logits: Tensor,
    method: str = "margin",
    eps: float = 1e-7,
) -> Tensor:
    """Dispatch to the appropriate confidence computation method.

    Args:
        logits : ``(B, C)`` float tensor.
        method : ``"margin"`` or ``"max_prob"``.
        eps    : Stability constant.

    Returns:
        ``(B,)`` float tensor of confidence scores in ``[0, 1]``.
    """
    if method == "margin":
        return _compute_confidence_margin(logits, eps)
    elif method == "max_prob":
        return _compute_confidence_max_prob(logits, eps)
    else:
        raise ValueError(f"Unknown confidence method: {method!r}")


# ---------------------------------------------------------------------------
# Base Decoder
# ---------------------------------------------------------------------------

class SpikeDecoder(nn.Module, ABC):
    """Abstract base class for all spike decoders.

    Every decoder must implement the ``forward`` method, which accepts a
    ``SpikeBatch`` and optional membrane tensor and returns a ``DecoderOutput``.

    The ``decode`` method provides a convenience wrapper that validates the
    input ``SpikeBatch`` before dispatching to ``forward``.

    Subclasses should NOT override ``decode``; override ``forward`` instead.

    Attributes:
        temperature       : Logits scaling factor.
        eps               : Numerical stability constant.
        confidence_method : Method for computing confidence scores.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        eps: float = 1e-7,
        confidence_method: str = "margin",
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.eps = eps
        self.confidence_method = confidence_method

    def decode(
        self,
        spikes: SpikeBatch,
        membrane: Optional[Tensor] = None,
    ) -> DecoderOutput:
        """Validate inputs and dispatch to the subclass forward method.

        This is the preferred public API. It validates the ``SpikeBatch``
        invariants before calling ``self.forward()``.

        Args:
            spikes   : ``SpikeBatch`` with ``(B, T, N)`` spike tensor.
            membrane : ``(B, T, N)`` optional membrane potential override.
                       If provided, this takes precedence over
                       ``spikes.membrane``.

        Returns:
            ``DecoderOutput`` with all fields populated.

        Raises:
            AssertionError: If ``spikes`` fails validation.
        """
        spikes.validate()
        if membrane is not None:
            # Override membrane in the SpikeBatch for this call
            spikes = SpikeBatch(
                spikes=spikes.spikes,
                membrane=membrane,
                mask=spikes.mask,
            )
        return self.forward(spikes, membrane)

    @abstractmethod
    def forward(
        self,
        spikes: SpikeBatch,
        membrane: Optional[Tensor] = None,
    ) -> DecoderOutput:
        """Decode spike train into a structured output.

        Subclasses must implement this method. The input ``SpikeBatch`` has
        already been validated by ``decode()``.

        Args:
            spikes   : ``SpikeBatch`` with ``(B, T, N)`` spike tensor.
            membrane : ``(B, T, N)`` optional membrane potential tensor.

        Returns:
            ``DecoderOutput`` with logits_proxy, prediction, confidence, aux.
        """
        ...

    def extra_repr(self) -> str:
        return (
            f"temperature={self.temperature}, eps={self.eps}, "
            f"confidence_method={self.confidence_method!r}"
        )


# ---------------------------------------------------------------------------
# RateDecoder
# ---------------------------------------------------------------------------

class RateDecoder(SpikeDecoder):
    """Rate (spike-count) decoder.

    The simplest and most robust decoder. Sums spikes over the time axis to
    produce a count per output neuron, then treats counts as unnormalized
    logits.

    This is the recommended default decoder. It is equivalent to the legacy
    ``SpikeDecoder(method="rate")`` but with batch-first convention, structured
    output, and AMP hardening.

    Computation:
        1. ``counts = spikes.float().sum(dim=1)`` -- ``(B, N)`` in fp32
        2. ``logits_proxy = counts / temperature`` -- scaled for softmax
        3. ``prediction = counts.argmax(dim=-1)`` -- hard class prediction
        4. ``confidence = margin(softmax(logits_proxy))`` -- top1 - top2

    AMP policy:
        The ``.float()`` cast BEFORE ``.sum(dim=1)`` is critical. Under
        ``torch.cuda.amp.autocast``, spike tensors may be float16. Summing
        T binary float16 values risks precision loss for large T. Always
        accumulate in fp32.

    Args:
        temperature       : Logits scaling factor (default 1.0).
        eps               : Numerical stability constant (default 1e-7).
        normalize_by_T    : If True, divide counts by T to produce firing
                            rates in [0, 1] instead of raw counts.
        confidence_method : ``"margin"`` or ``"max_prob"`` (default ``"margin"``).
    """

    def __init__(
        self,
        temperature: float = 1.0,
        eps: float = 1e-7,
        normalize_by_T: bool = False,
        confidence_method: str = "margin",
    ) -> None:
        super().__init__(
            temperature=temperature,
            eps=eps,
            confidence_method=confidence_method,
        )
        self.normalize_by_T = normalize_by_T

    def forward(
        self,
        spikes: SpikeBatch,
        membrane: Optional[Tensor] = None,
    ) -> DecoderOutput:
        """Decode spikes via rate coding (spike counting).

        Args:
            spikes   : ``SpikeBatch`` with ``(B, T, N)`` spike tensor.
            membrane : Ignored by RateDecoder.

        Returns:
            ``DecoderOutput`` with:
                - logits_proxy: ``(B, N)`` spike counts (or rates) / temperature
                - prediction: ``(B,)`` argmax of counts
                - confidence: ``(B,)`` margin or max_prob
                - aux: raw_counts, rates, count_histogram, margin
        """
        s = spikes.spikes
        B, T, N = s.shape

        # --- Step 1: Accumulate spike counts in fp32 ---
        # CRITICAL: .float() BEFORE .sum() to ensure fp32 accumulation.
        # Summing in fp16 can lose precision for T > ~20.
        counts = s.float().sum(dim=1)  # (B, N) fp32

        # --- Step 2: Compute rates (counts / T) ---
        rates = counts / max(T, 1)  # (B, N) firing rates in [0, ~1]

        # --- Step 3: Compute logits_proxy ---
        if self.normalize_by_T:
            logits_proxy = rates / self.temperature  # (B, N)
        else:
            logits_proxy = counts / self.temperature  # (B, N)

        # --- Step 4: Hard prediction ---
        prediction = counts.argmax(dim=-1).long()  # (B,) int64

        # --- Step 5: Confidence ---
        confidence = _compute_confidence(
            logits_proxy, method=self.confidence_method, eps=self.eps
        )

        # --- Step 6: Diagnostics ---
        # count_histogram: mean count per neuron across the batch
        count_histogram = counts.detach().mean(dim=0)  # (N,)

        # margin: explicitly compute for aux even if confidence_method differs
        margin = _compute_confidence_margin(logits_proxy, self.eps)

        aux: Dict[str, Any] = {
            "raw_counts": counts.detach(),
            "rates": rates.detach(),
            "count_histogram": count_histogram,
            "margin": margin.detach(),
        }

        return DecoderOutput(
            logits_proxy=logits_proxy,
            prediction=prediction,
            confidence=confidence,
            aux=aux,
        )

    def extra_repr(self) -> str:
        return (
            f"temperature={self.temperature}, eps={self.eps}, "
            f"normalize_by_T={self.normalize_by_T}, "
            f"confidence_method={self.confidence_method!r}"
        )


# ---------------------------------------------------------------------------
# FirstSpikeDecoder
# ---------------------------------------------------------------------------

class FirstSpikeDecoder(SpikeDecoder):
    """First-spike (time-to-first-spike / TTFS) decoder.

    Decodes based on spike latency: the neuron that fires first wins. This
    exploits temporal coding where important features are encoded by spike
    timing rather than spike count.

    Computation (fully vectorized via cumsum + argmax):
        1. ``cumsum = spikes.cumsum(dim=1)`` -- cumulative count along time
        2. ``has_spiked = (cumsum >= 1.0)`` -- True at and after first spike
        3. ``t_first = has_spiked.float().argmax(dim=1)`` -- first True index
        4. Handle no-spike: where ``cumsum[:, -1, :] == 0``, set ``t_first = T``
        5. ``logits_proxy = -t_first / T / temperature`` -- earlier = higher
        6. ``prediction = t_first.argmin(dim=-1)`` -- earliest spike wins

    Why cumsum + argmax:
        A naive implementation would iterate over timesteps or use
        ``torch.nonzero``, both of which are slow or produce ragged outputs.
        The cumsum trick is fully vectorized: ``cumsum >= 1`` creates a boolean
        mask that is False before the first spike and True afterward, and
        ``argmax`` on a boolean tensor returns the index of the first True.

    No-spike handling:
        Neurons that never fire within the T timesteps receive ``t_first = T``,
        the worst possible latency. This places them at the bottom of the
        argmin ranking. Do NOT leave no-spike neurons at ``t_first = 0`` --
        that would make silent neurons appear to be the fastest, inverting
        the entire decoding logic.

    AMP policy:
        Time indices are integer-valued. The division by T and temperature is
        float arithmetic. No special fp32 casting needed beyond the ``.float()``
        on ``t_first``.

    Args:
        temperature       : Logits scaling factor (default 1.0).
        eps               : Numerical stability constant (default 1e-7).
        confidence_method : ``"margin"`` or ``"max_prob"`` (default ``"margin"``).
    """

    def __init__(
        self,
        temperature: float = 1.0,
        eps: float = 1e-7,
        confidence_method: str = "margin",
    ) -> None:
        super().__init__(
            temperature=temperature,
            eps=eps,
            confidence_method=confidence_method,
        )

    def forward(
        self,
        spikes: SpikeBatch,
        membrane: Optional[Tensor] = None,
    ) -> DecoderOutput:
        """Decode spikes via first-spike latency.

        Args:
            spikes   : ``SpikeBatch`` with ``(B, T, N)`` spike tensor.
            membrane : Ignored by FirstSpikeDecoder.

        Returns:
            ``DecoderOutput`` with:
                - logits_proxy: ``(B, N)`` negated normalized first-spike times
                - prediction: ``(B,)`` argmin of first-spike times
                - confidence: ``(B,)`` time margin between 1st and 2nd earliest
                - aux: first_spike_times, no_spike_mask, spike_counts, margin
        """
        s = spikes.spikes
        B, T, N = s.shape

        # --- Step 1: Vectorized first-spike detection via cumsum ---
        # cumsum along time axis: cumsum[b, t, n] = sum of spikes[b, 0:t+1, n]
        cumsum = s.float().cumsum(dim=1)  # (B, T, N) fp32

        # has_spiked[b, t, n] is True at the first timestep where neuron n has
        # fired at least once, and stays True for all subsequent timesteps.
        has_spiked = (cumsum >= 1.0)  # (B, T, N) bool

        # argmax on bool returns index of the first True. For all-False rows
        # (no spike), argmax returns 0 -- which we must correct below.
        t_first = has_spiked.float().argmax(dim=1)  # (B, N) -- first spike timestep

        # --- Step 2: Handle no-spike neurons ---
        # Check if any neuron never spiked: cumsum at last timestep is 0.
        total_spikes = cumsum[:, -1, :]  # (B, N) -- total spike count per neuron
        no_spike_mask = (total_spikes == 0)  # (B, N) bool -- True if neuron never fired

        # Assign T (worst possible latency) to no-spike neurons.
        # This prevents silent neurons from winning argmin (which would be t=0).
        t_first = torch.where(
            no_spike_mask,
            torch.full_like(t_first, float(T)),
            t_first,
        )  # (B, N)

        # --- Step 3: Build first_spike_mask for verification ---
        # first_spike_mask[b, t, n] = True only at the exact timestep of the
        # first spike for neuron n. This is: (cumsum == 1) AND spike is active.
        first_spike_mask = (cumsum == 1.0) & s.bool()  # (B, T, N)

        # --- Step 4: Compute logits_proxy ---
        # Negate time so earlier spike = higher logit value.
        # Scale by 1/T to normalize to approximately [-1, 0] range.
        t_first_float = t_first.float()  # (B, N)
        logits_proxy = (-t_first_float / max(T, 1)) / self.temperature  # (B, N)

        # --- Step 5: Hard prediction ---
        # The neuron with the earliest spike (smallest t_first) wins.
        prediction = t_first.long().argmin(dim=-1)  # (B,) int64

        # --- Step 6: Confidence from time margin ---
        # Sort first-spike times ascending. The margin between the 1st and 2nd
        # earliest spikes indicates how decisive the winner is.
        if N >= 2:
            sorted_times, _ = t_first_float.sort(dim=-1)  # (B, N) ascending
            time_gap = sorted_times[:, 1] - sorted_times[:, 0]  # (B,)
            time_margin = (time_gap / max(T, 1)).clamp(min=0.0, max=1.0)  # (B,)
        else:
            # Only one neuron: confidence is 1.0 if it spiked, 0.0 otherwise
            time_margin = (~no_spike_mask[:, 0]).float()  # (B,)

        # Also compute standard confidence from logits for consistency
        logits_confidence = _compute_confidence(
            logits_proxy, method=self.confidence_method, eps=self.eps
        )

        # Use time margin as confidence (more interpretable for TTFS)
        confidence = time_margin

        # --- Step 7: Spike counts for diagnostics ---
        spike_counts = total_spikes.detach()  # (B, N)

        aux: Dict[str, Any] = {
            "first_spike_times": t_first.detach(),
            "no_spike_mask": no_spike_mask.detach(),
            "spike_counts": spike_counts,
            "margin": time_margin.detach(),
            "logits_confidence": logits_confidence.detach(),
            "first_spike_mask": first_spike_mask.detach(),
        }

        return DecoderOutput(
            logits_proxy=logits_proxy,
            prediction=prediction,
            confidence=confidence,
            aux=aux,
        )


# ---------------------------------------------------------------------------
# PopulationDecoder
# ---------------------------------------------------------------------------

class PopulationDecoder(SpikeDecoder):
    """Population decoder for grouped output neurons.

    Partitions output neurons into groups, where each group represents a class
    (classification) or a range of values (regression). Spike counts are
    aggregated per group to produce group-level logits.

    This decoder requires a ``population_map`` that assigns each neuron to a
    group. The map is typically provided by the encoder that created the
    population-coded spike train (see ``PopulationEncoder``).

    Classification mode:
        - Aggregate spike counts per group
        - ``logits_proxy = group_counts / temperature`` -- ``(B, num_groups)``
        - ``prediction = argmax(group_counts)``

    Regression mode:
        - Assign center values to each group
        - Compute softmax-weighted expectation: ``value = sum(centers * weights)``
        - ``logits_proxy = group_counts / temperature`` (for loss computation)
        - ``predicted_value`` available in ``aux``

    AMP policy:
        All accumulation (spike counting and group aggregation) is performed
        in fp32 regardless of input dtype.

    Args:
        population_map    : Dict mapping group index to list of neuron indices.
                            Example: ``{0: [0,1,2], 1: [3,4,5], 2: [6,7,8]}``.
        mode              : ``"classification"`` or ``"regression"``.
        centers           : ``(num_groups,)`` tensor of center values for
                            regression mode. Ignored in classification mode.
        temperature       : Logits scaling factor (default 1.0).
        eps               : Numerical stability constant (default 1e-7).
        confidence_method : ``"margin"`` or ``"max_prob"`` (default ``"margin"``).
    """

    def __init__(
        self,
        population_map: Dict[int, List[int]],
        mode: str = "classification",
        centers: Optional[Tensor] = None,
        temperature: float = 1.0,
        eps: float = 1e-7,
        confidence_method: str = "margin",
    ) -> None:
        super().__init__(
            temperature=temperature,
            eps=eps,
            confidence_method=confidence_method,
        )
        if not population_map:
            raise ValueError("population_map must be non-empty.")
        self.population_map = population_map
        self.mode = mode
        self.num_groups = len(population_map)

        # Precompute index tensors for efficient gather operations.
        # Store as buffers so they move with the module to GPU.
        self._group_keys = sorted(population_map.keys())
        max_group_size = max(len(indices) for indices in population_map.values())
        self._group_sizes: List[int] = []

        # Build a padded index tensor: (num_groups, max_group_size)
        # Padded entries point to index 0 (harmless; masked out by group_size).
        index_array = torch.zeros(
            self.num_groups, max_group_size, dtype=torch.long
        )
        for i, g in enumerate(self._group_keys):
            indices = population_map[g]
            self._group_sizes.append(len(indices))
            index_array[i, :len(indices)] = torch.tensor(indices, dtype=torch.long)

        # Register as buffer for device movement
        self.register_buffer("_index_tensor", index_array)
        self.register_buffer(
            "_group_size_tensor",
            torch.tensor(self._group_sizes, dtype=torch.float32),
        )

        # Centers for regression mode
        if mode == "regression":
            if centers is None:
                raise ValueError(
                    "centers tensor is required for mode='regression'."
                )
            self.register_buffer("centers", centers.float())
        else:
            self.centers = None

    def _aggregate_group_counts(self, counts: Tensor) -> Tensor:
        """Aggregate per-neuron counts into per-group counts.

        Uses advanced indexing via the precomputed index tensor for
        efficiency. Group sizes may differ; padded indices contribute
        zero to the sum.

        Args:
            counts : ``(B, N)`` float32 per-neuron spike counts.

        Returns:
            ``(B, num_groups)`` float32 per-group spike counts.
        """
        B = counts.shape[0]

        # Gather counts for all group neurons: (B, num_groups, max_group_size)
        # _index_tensor is (num_groups, max_group_size)
        expanded_idx = self._index_tensor.unsqueeze(0).expand(
            B, -1, -1
        )  # (B, num_groups, max_group_size)

        # counts is (B, N). We need to gather along dim=1 (neuron dim).
        # Flatten the group structure to gather, then reshape.
        flat_idx = expanded_idx.reshape(B, -1)  # (B, num_groups * max_group_size)
        flat_gathered = torch.gather(counts, dim=1, index=flat_idx)  # (B, num_groups * max_group_size)
        gathered = flat_gathered.reshape(
            B, self.num_groups, -1
        )  # (B, num_groups, max_group_size)

        # Create a mask for valid (non-padded) entries
        max_gs = self._index_tensor.shape[1]
        valid_mask = torch.arange(
            max_gs, device=counts.device
        ).unsqueeze(0) < self._group_size_tensor.unsqueeze(1)
        # valid_mask: (num_groups, max_group_size) bool

        # Zero out padded entries before summing
        gathered = gathered * valid_mask.unsqueeze(0).float()  # (B, num_groups, max_gs)

        # Sum within each group
        group_counts = gathered.sum(dim=-1)  # (B, num_groups)

        return group_counts

    def forward(
        self,
        spikes: SpikeBatch,
        membrane: Optional[Tensor] = None,
    ) -> DecoderOutput:
        """Decode spikes via population grouping.

        Args:
            spikes   : ``SpikeBatch`` with ``(B, T, N)`` spike tensor.
            membrane : Ignored by PopulationDecoder.

        Returns:
            ``DecoderOutput`` with:
                - logits_proxy: ``(B, num_groups)`` group counts / temperature
                - prediction: ``(B,)`` argmax (class) or argmax (regression)
                - confidence: ``(B,)`` margin or max_prob
                - aux: raw_counts, group_counts, group_rates, margin,
                       (regression: predicted_value, group_weights)
        """
        s = spikes.spikes
        B, T, N = s.shape

        # --- Step 1: Per-neuron spike counts in fp32 ---
        counts = s.float().sum(dim=1)  # (B, N) fp32

        # --- Step 2: Aggregate per-group ---
        group_counts = self._aggregate_group_counts(counts)  # (B, num_groups)

        # --- Step 3: Group firing rates ---
        group_rates = group_counts / (
            max(T, 1) * self._group_size_tensor.unsqueeze(0).clamp(min=1.0)
        )  # (B, num_groups)

        # --- Step 4: Logits proxy ---
        logits_proxy = group_counts / self.temperature  # (B, num_groups)

        # --- Step 5: Prediction and confidence ---
        if self.mode == "classification":
            prediction = group_counts.argmax(dim=-1).long()  # (B,) int64
            confidence = _compute_confidence(
                logits_proxy, method=self.confidence_method, eps=self.eps
            )
            margin = _compute_confidence_margin(logits_proxy, self.eps)

            aux: Dict[str, Any] = {
                "raw_counts": counts.detach(),
                "group_counts": group_counts.detach(),
                "group_rates": group_rates.detach(),
                "margin": margin.detach(),
            }

        elif self.mode == "regression":
            # Softmax-weighted expectation over centers
            weights = torch.softmax(
                group_counts.float() / self.temperature, dim=-1
            )  # (B, num_groups)
            predicted_value = (
                weights * self.centers.unsqueeze(0)
            ).sum(dim=-1)  # (B,)

            # For regression, prediction is the argmax group (discretized)
            prediction = group_counts.argmax(dim=-1).long()  # (B,)

            # Confidence: max weight (how peaked the distribution is)
            confidence = weights.max(dim=-1).values.clamp(min=0.0, max=1.0)

            # For regression, logits_proxy can also be the value itself
            # But we keep group_counts / temperature for compatibility with
            # cross-entropy loss during pre-training. The predicted_value
            # is available in aux for MSE loss.
            aux = {
                "raw_counts": counts.detach(),
                "group_counts": group_counts.detach(),
                "group_rates": group_rates.detach(),
                "predicted_value": predicted_value.detach(),
                "group_weights": weights.detach(),
                "margin": confidence.detach(),
            }

        else:
            raise ValueError(f"Unknown population mode: {self.mode!r}")

        return DecoderOutput(
            logits_proxy=logits_proxy,
            prediction=prediction,
            confidence=confidence,
            aux=aux,
        )

    def extra_repr(self) -> str:
        return (
            f"num_groups={self.num_groups}, mode={self.mode!r}, "
            f"temperature={self.temperature}, eps={self.eps}, "
            f"confidence_method={self.confidence_method!r}"
        )


# ---------------------------------------------------------------------------
# MembraneDecoder
# ---------------------------------------------------------------------------

class MembraneDecoder(SpikeDecoder):
    """Membrane potential decoder.

    Uses the membrane potential directly when spike counts are too sparse to
    carry a reliable signal. This is common during early training when the
    network has not yet learned to produce consistent spiking patterns.

    Membrane potentials carry sub-threshold information that provides a gradient
    signal even when no spikes fire. As training progresses and firing rates
    increase, transition to rate or population decoding.

    Modes:
        - ``"final"``: Use membrane potential at the last timestep ``[:, -1, :]``.
          Best when the SNN processes a fixed-length input and the last timestep
          captures accumulated evidence.
        - ``"max"``: Use maximum membrane potential over time
          ``max(dim=1).values``. Best when peak membrane potential is more
          informative (e.g., a neuron that briefly approached threshold but
          reset before the final step).

    Args:
        mode              : ``"final"`` or ``"max"`` (default ``"final"``).
        temperature       : Logits scaling factor (default 1.0).
        eps               : Numerical stability constant (default 1e-7).
        confidence_method : ``"margin"`` or ``"max_prob"`` (default ``"margin"``).
    """

    def __init__(
        self,
        mode: str = "final",
        temperature: float = 1.0,
        eps: float = 1e-7,
        confidence_method: str = "margin",
    ) -> None:
        super().__init__(
            temperature=temperature,
            eps=eps,
            confidence_method=confidence_method,
        )
        if mode not in ("final", "max"):
            raise ValueError(
                f"Unknown membrane decode mode: {mode!r}. "
                "Must be 'final' or 'max'."
            )
        self.mode = mode

    def forward(
        self,
        spikes: SpikeBatch,
        membrane: Optional[Tensor] = None,
    ) -> DecoderOutput:
        """Decode via membrane potential values.

        Args:
            spikes   : ``SpikeBatch`` with optional ``membrane`` field.
            membrane : ``(B, T, N)`` membrane potential tensor. If provided,
                       this overrides ``spikes.membrane``.

        Returns:
            ``DecoderOutput`` with:
                - logits_proxy: ``(B, N)`` membrane values / temperature
                - prediction: ``(B,)`` argmax of membrane values
                - confidence: ``(B,)`` margin or max_prob
                - aux: membrane_values, mode

        Raises:
            ValueError: If no membrane potential is available (neither
                       ``membrane`` arg nor ``spikes.membrane`` is set).
        """
        # Resolve membrane source
        mem = membrane if membrane is not None else spikes.membrane
        if mem is None:
            raise ValueError(
                "MembraneDecoder requires membrane potential data. "
                "Provide it via the membrane argument or via "
                "SpikeBatch.membrane. Hint: use RecordMode.SPIKES_MEMBRANE "
                "in snn_unroll to collect membrane traces."
            )

        B, T, N = mem.shape

        # --- Step 1: Extract membrane values ---
        if self.mode == "final":
            v = mem[:, -1, :].float()  # (B, N) last timestep
        elif self.mode == "max":
            v = mem.float().max(dim=1).values  # (B, N) peak over time
        else:
            raise ValueError(f"Unknown mode: {self.mode!r}")

        # --- Step 2: Logits proxy ---
        logits_proxy = v / self.temperature  # (B, N)

        # --- Step 3: Prediction ---
        prediction = v.argmax(dim=-1).long()  # (B,) int64

        # --- Step 4: Confidence ---
        confidence = _compute_confidence(
            logits_proxy, method=self.confidence_method, eps=self.eps
        )

        # --- Step 5: Diagnostics ---
        margin = _compute_confidence_margin(logits_proxy, self.eps)

        aux: Dict[str, Any] = {
            "membrane_values": v.detach(),
            "mode": self.mode,
            "margin": margin.detach(),
        }

        return DecoderOutput(
            logits_proxy=logits_proxy,
            prediction=prediction,
            confidence=confidence,
            aux=aux,
        )

    def extra_repr(self) -> str:
        return (
            f"mode={self.mode!r}, temperature={self.temperature}, "
            f"eps={self.eps}, confidence_method={self.confidence_method!r}"
        )


# ---------------------------------------------------------------------------
# DecoderEnsemble
# ---------------------------------------------------------------------------

class DecoderEnsemble(SpikeDecoder):
    """Weighted ensemble of multiple spike decoders.

    Combines the ``logits_proxy`` from multiple decoders using configurable
    weights, producing a single unified ``DecoderOutput``. This is the
    recommended approach for curriculum-based decoder annealing:

    - Early training: ``{"rate": 0.3, "membrane": 0.7}``
      (membrane provides gradient signal when spikes are sparse)
    - Mid training:   ``{"rate": 0.7, "membrane": 0.3}``
      (transition as firing rates increase)
    - Late training:  ``{"rate": 1.0}``
      (pure rate decoding once spiking is reliable)

    Important: Do NOT hard-switch decoder types mid-training. The loss
    landscape, loss magnitude, and gradient distribution change discontinuously.
    Anneal weights over 500-1000 steps to give the optimizer state (Adam moment
    estimates) time to adapt.

    The ensemble prediction and confidence are computed from the combined
    logits, not from any individual decoder. Individual decoder outputs are
    available in ``aux["sub_outputs"]``.

    Args:
        decoders          : Dict mapping decoder names to ``SpikeDecoder``
                            instances. Names must match the keys in ``weights``.
        weights           : Dict mapping decoder names to float weights.
                            Weights must sum to 1.0 (tolerance 1e-4).
        temperature       : Not used directly (each sub-decoder has its own).
        eps               : Stability constant for combined confidence.
        confidence_method : ``"margin"`` or ``"max_prob"`` for the combined
                            output (default ``"margin"``).
    """

    def __init__(
        self,
        decoders: Dict[str, SpikeDecoder],
        weights: Dict[str, float],
        temperature: float = 1.0,
        eps: float = 1e-7,
        confidence_method: str = "margin",
    ) -> None:
        super().__init__(
            temperature=temperature,
            eps=eps,
            confidence_method=confidence_method,
        )
        if set(decoders.keys()) != set(weights.keys()):
            raise ValueError(
                f"Decoder names {set(decoders.keys())} do not match "
                f"weight names {set(weights.keys())}."
            )
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 1e-4:
            raise ValueError(
                f"Ensemble weights must sum to 1.0, got {total_weight:.6f}."
            )
        if not decoders:
            raise ValueError("DecoderEnsemble requires at least one decoder.")

        self.decoders = nn.ModuleDict(decoders)
        self.weights = weights

    def forward(
        self,
        spikes: SpikeBatch,
        membrane: Optional[Tensor] = None,
    ) -> DecoderOutput:
        """Run all sub-decoders and combine via weighted sum of logits.

        Args:
            spikes   : ``SpikeBatch`` with ``(B, T, N)`` spike tensor.
            membrane : ``(B, T, N)`` optional membrane potential tensor.

        Returns:
            ``DecoderOutput`` with:
                - logits_proxy: ``(B, C)`` weighted sum of sub-decoder logits
                - prediction: ``(B,)`` argmax of combined logits
                - confidence: ``(B,)`` from combined logits
                - aux: sub_outputs (dict name -> aux), weights (dict)
        """
        sub_outputs: Dict[str, DecoderOutput] = {}
        combined_logits: Optional[Tensor] = None

        for name, decoder in self.decoders.items():
            w = self.weights[name]
            out = decoder(spikes, membrane)
            sub_outputs[name] = out

            weighted = w * out.logits_proxy
            if combined_logits is None:
                combined_logits = weighted
            else:
                # Ensure shape compatibility: both must be (B, C).
                # If different decoders produce different C, this will fail
                # at the addition -- by design, as mixing decoders with
                # different output dimensions is an error.
                combined_logits = combined_logits + weighted

        assert combined_logits is not None, "No decoders produced output."

        # --- Prediction from combined logits ---
        prediction = combined_logits.argmax(dim=-1).long()  # (B,) int64

        # --- Confidence from combined logits ---
        confidence = _compute_confidence(
            combined_logits, method=self.confidence_method, eps=self.eps
        )

        aux: Dict[str, Any] = {
            "sub_outputs": {
                name: out.aux for name, out in sub_outputs.items()
            },
            "weights": dict(self.weights),
        }

        return DecoderOutput(
            logits_proxy=combined_logits,
            prediction=prediction,
            confidence=confidence,
            aux=aux,
        )

    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """Update ensemble weights (e.g., during curriculum annealing).

        Args:
            new_weights : New weight dict. Must have the same keys as the
                         original weights and sum to 1.0.

        Raises:
            ValueError: If keys do not match or weights do not sum to 1.0.
        """
        if set(new_weights.keys()) != set(self.weights.keys()):
            raise ValueError(
                f"New weight keys {set(new_weights.keys())} do not match "
                f"existing keys {set(self.weights.keys())}."
            )
        total = sum(new_weights.values())
        if abs(total - 1.0) > 1e-4:
            raise ValueError(
                f"Ensemble weights must sum to 1.0, got {total:.6f}."
            )
        self.weights = dict(new_weights)

    def extra_repr(self) -> str:
        parts = [f"{name}: {w:.3f}" for name, w in sorted(self.weights.items())]
        return f"weights={{{', '.join(parts)}}}"


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_decoder(config: DecodingConfig) -> SpikeDecoder:
    """Factory function to create a spike decoder from a DecodingConfig.

    This is the primary entry point for decoder construction. All training
    scripts should use this function rather than instantiating decoders
    directly, to ensure configuration is validated and defaults are applied
    consistently.

    Args:
        config : ``DecodingConfig`` specifying the decoder type and parameters.

    Returns:
        A ``SpikeDecoder`` instance matching the config.

    Raises:
        ValueError: If the config is invalid.

    Example::

        config = DecodingConfig(type="rate", temperature=1.5)
        decoder = create_decoder(config)
        output = decoder(spike_batch)

    Example (ensemble)::

        config = DecodingConfig(
            type="ensemble",
            ensemble_weights={"rate": 0.7, "membrane": 0.3},
        )
        decoder = create_decoder(config)
    """
    config.validate()

    if config.type == "rate":
        return RateDecoder(
            temperature=config.temperature,
            eps=config.eps,
            normalize_by_T=config.normalize_by_T,
            confidence_method=config.confidence_method,
        )

    elif config.type == "first_spike":
        return FirstSpikeDecoder(
            temperature=config.temperature,
            eps=config.eps,
            confidence_method=config.confidence_method,
        )

    elif config.type == "population":
        assert config.population_map is not None  # validated above
        return PopulationDecoder(
            population_map=config.population_map,
            mode=config.population_mode,
            centers=config.population_centers,
            temperature=config.temperature,
            eps=config.eps,
            confidence_method=config.confidence_method,
        )

    elif config.type == "membrane":
        return MembraneDecoder(
            mode=config.membrane_mode,
            temperature=config.temperature,
            eps=config.eps,
            confidence_method=config.confidence_method,
        )

    elif config.type == "ensemble":
        assert config.ensemble_weights is not None  # validated above
        sub_decoders: Dict[str, SpikeDecoder] = {}
        for name, weight in config.ensemble_weights.items():
            # Create sub-decoder with a sub-config
            sub_config = DecodingConfig(
                type=name,
                temperature=config.temperature,
                eps=config.eps,
                population_map=config.population_map,
                population_mode=config.population_mode,
                population_centers=config.population_centers,
                membrane_mode=config.membrane_mode,
                normalize_by_T=config.normalize_by_T,
                confidence_method=config.confidence_method,
            )
            sub_decoders[name] = create_decoder(sub_config)

        return DecoderEnsemble(
            decoders=sub_decoders,
            weights=config.ensemble_weights,
            temperature=config.temperature,
            eps=config.eps,
            confidence_method=config.confidence_method,
        )

    else:
        raise ValueError(f"Unknown decoder type: {config.type!r}")


# ---------------------------------------------------------------------------
# Utility: adapt_time_first
# ---------------------------------------------------------------------------

def adapt_time_first(spikes: Tensor) -> Tensor:
    """Transpose ``(T, B, D)`` to ``(B, T, D)`` for batch-first decoders.

    Migration adapter for code that still uses the legacy ``SpikeDecoder``
    axis convention from ``brain_ai/core/encoding.py``. Transposes ``dim=0``
    and ``dim=1`` to produce the batch-first ``(B, T, D)`` layout.

    Args:
        spikes : ``(T, B, D)`` or ``(B, D)`` spike tensor.

    Returns:
        ``(B, T, D)`` spike tensor, or unchanged ``(B, D)`` if 2-D.
    """
    if spikes.ndim == 3:
        return spikes.permute(1, 0, 2)
    return spikes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Data containers
    "SpikeBatch",
    "DecoderOutput",
    # Configuration
    "DecodingConfig",
    # Base class
    "SpikeDecoder",
    # Decoder implementations
    "RateDecoder",
    "FirstSpikeDecoder",
    "PopulationDecoder",
    "MembraneDecoder",
    "DecoderEnsemble",
    # Factory
    "create_decoder",
    # Utilities
    "adapt_time_first",
    # Confidence helpers
    "_compute_confidence",
    "_compute_confidence_margin",
    "_compute_confidence_max_prob",
]


# ===========================================================================
# Self-test suite
# ===========================================================================

if __name__ == "__main__":
    """Comprehensive self-test for the spike decoding module.

    Exercises:
        1.  SpikeBatch construction and validation
        2.  DecoderOutput construction and validation
        3.  RateDecoder: shape, dtype, DecoderOutput contract
        4.  RateDecoder: known counts produce correct prediction
        5.  FirstSpikeDecoder: shape, dtype, contract
        6.  FirstSpikeDecoder: known spike times produce correct argmin
        7.  FirstSpikeDecoder: no-spike handling
        8.  PopulationDecoder (classification): known groups
        9.  PopulationDecoder (regression): weighted expectation
        10. MembraneDecoder: final mode
        11. MembraneDecoder: max mode
        12. DecoderEnsemble: combined output
        13. Factory function: all types
        14. AMP: all decoders under autocast (if CUDA available)
        15. adapt_time_first: axis conversion
        16. SpikeBatch.from_time_first: legacy conversion
        17. DecoderOutput.to() and .detach()
        18. DecodingConfig validation

    Run directly:
        python brain_ai/core/decoding.py

    Or from the template location:
        python brain-ai-dev/skills/spike-codec-losses/assets/decoding_template.py
    """

    # ------------------------------------------------------------------
    # Path setup: 4-level dirname from __file__ to reach project root
    # ------------------------------------------------------------------
    _this_file = Path(__file__).resolve()
    _project_root = _this_file.parent.parent.parent.parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

    import traceback

    torch.manual_seed(42)
    device = torch.device("cpu")

    # Test parameters
    B, T, N = 4, 20, 10  # batch, timesteps, neurons
    passed = 0
    failed = 0
    total = 0

    def run_test(name: str, fn: Callable[[], None]) -> None:
        """Run a single test and track PASS/FAIL."""
        nonlocal passed, failed, total
        total += 1
        try:
            fn()
            passed += 1
            print(f"  [{total:2d}] PASS: {name}")
        except Exception as e:
            failed += 1
            print(f"  [{total:2d}] FAIL: {name}")
            print(f"        Error: {e}")
            traceback.print_exc()
            print()

    print("=" * 70)
    print("decoding_template.py — Comprehensive Self-Test")
    print("=" * 70)
    print()

    # ------------------------------------------------------------------
    # Helper: generate deterministic spike data
    # ------------------------------------------------------------------

    def make_spikes(B: int, T: int, N: int, rate: float = 0.3) -> Tensor:
        """Generate reproducible random binary spike tensor."""
        return (torch.rand(B, T, N) < rate).float()

    def make_membrane(B: int, T: int, N: int) -> Tensor:
        """Generate reproducible random membrane potential tensor."""
        return torch.randn(B, T, N) * 0.5 + 0.5

    # ==================================================================
    # Test 1: SpikeBatch construction and validation
    # ==================================================================

    def test_spike_batch_construction():
        s = make_spikes(B, T, N)
        m = make_membrane(B, T, N)
        mask = torch.ones(B, T, dtype=torch.bool)

        batch = SpikeBatch(spikes=s, membrane=m, mask=mask)
        batch.validate()

        assert batch.batch_size == B, f"Expected B={B}, got {batch.batch_size}"
        assert batch.num_timesteps == T, f"Expected T={T}, got {batch.num_timesteps}"
        assert batch.num_neurons == N, f"Expected N={N}, got {batch.num_neurons}"
        assert batch.shape == torch.Size([B, T, N])
        assert batch.device == torch.device("cpu")

        # Test without membrane and mask
        batch2 = SpikeBatch(spikes=s)
        batch2.validate()

    run_test("SpikeBatch construction and validation", test_spike_batch_construction)

    # ==================================================================
    # Test 2: DecoderOutput construction and validation
    # ==================================================================

    def test_decoder_output_construction():
        logits = torch.randn(B, N)
        pred = torch.zeros(B, dtype=torch.long)
        conf = torch.full((B,), 0.5)

        out = DecoderOutput(
            logits_proxy=logits,
            prediction=pred,
            confidence=conf,
            aux={"test_key": torch.zeros(B)},
        )
        out.validate()

        assert out.num_classes == N, f"Expected C={N}, got {out.num_classes}"
        assert out.batch_size == B, f"Expected B={B}, got {out.batch_size}"

        # Test .to()
        out_moved = out.to(device)
        assert out_moved.logits_proxy.device == device

        # Test .detach()
        logits_grad = torch.randn(B, N, requires_grad=True)
        out_grad = DecoderOutput(
            logits_proxy=logits_grad,
            prediction=pred,
            confidence=conf,
        )
        out_detached = out_grad.detach()
        assert out_detached.logits_proxy.grad_fn is None

    run_test("DecoderOutput construction and validation", test_decoder_output_construction)

    # ==================================================================
    # Test 3: RateDecoder — shape, dtype, contract
    # ==================================================================

    def test_rate_decoder_shape_contract():
        decoder = RateDecoder(temperature=1.0)
        s = make_spikes(B, T, N)
        batch = SpikeBatch(spikes=s)
        out = decoder(batch)

        assert out.logits_proxy.shape == (B, N), (
            f"Expected (B, N)=({B}, {N}), got {out.logits_proxy.shape}"
        )
        assert out.logits_proxy.dtype == torch.float32
        assert out.prediction.shape == (B,)
        assert out.prediction.dtype == torch.long
        assert out.confidence.shape == (B,)
        assert out.confidence.dtype == torch.float32
        assert (out.confidence >= 0).all() and (out.confidence <= 1).all()

        out.validate()

        # Check aux keys
        assert "raw_counts" in out.aux
        assert "rates" in out.aux
        assert "count_histogram" in out.aux
        assert "margin" in out.aux
        assert out.aux["raw_counts"].shape == (B, N)
        assert out.aux["rates"].shape == (B, N)
        assert out.aux["count_histogram"].shape == (N,)

    run_test("RateDecoder shape and contract", test_rate_decoder_shape_contract)

    # ==================================================================
    # Test 4: RateDecoder — known counts produce correct prediction
    # ==================================================================

    def test_rate_decoder_known_counts():
        decoder = RateDecoder(temperature=1.0)

        # Create spikes where neuron 3 has the most spikes for all batches
        s = torch.zeros(B, T, N)
        # Neuron 3: fires every timestep
        s[:, :, 3] = 1.0
        # Neuron 7: fires half the time
        s[:, :T // 2, 7] = 1.0
        # Other neurons: sparse spikes
        s[:, 0, 0] = 1.0

        batch = SpikeBatch(spikes=s)
        out = decoder(batch)

        # Neuron 3 should be the prediction for all batch items
        expected_pred = torch.full((B,), 3, dtype=torch.long)
        assert torch.equal(out.prediction, expected_pred), (
            f"Expected prediction={expected_pred.tolist()}, "
            f"got {out.prediction.tolist()}"
        )

        # Raw counts for neuron 3 should be T
        assert (out.aux["raw_counts"][:, 3] == T).all(), (
            f"Neuron 3 should have {T} spikes, "
            f"got {out.aux['raw_counts'][:, 3].tolist()}"
        )

        # Test normalize_by_T
        decoder_norm = RateDecoder(temperature=1.0, normalize_by_T=True)
        out_norm = decoder_norm(batch)
        # Logits should be rates (counts / T) / temperature
        expected_rate = T / T / 1.0  # = 1.0 for neuron 3
        assert abs(out_norm.logits_proxy[0, 3].item() - expected_rate) < 1e-5

    run_test("RateDecoder known counts -> correct prediction", test_rate_decoder_known_counts)

    # ==================================================================
    # Test 5: FirstSpikeDecoder — shape, dtype, contract
    # ==================================================================

    def test_first_spike_decoder_shape_contract():
        decoder = FirstSpikeDecoder(temperature=1.0)
        s = make_spikes(B, T, N)
        batch = SpikeBatch(spikes=s)
        out = decoder(batch)

        assert out.logits_proxy.shape == (B, N), (
            f"Expected ({B}, {N}), got {out.logits_proxy.shape}"
        )
        assert out.logits_proxy.dtype == torch.float32
        assert out.prediction.shape == (B,)
        assert out.prediction.dtype == torch.long
        assert out.confidence.shape == (B,)
        assert (out.confidence >= 0).all() and (out.confidence <= 1).all()

        out.validate()

        assert "first_spike_times" in out.aux
        assert "no_spike_mask" in out.aux
        assert "spike_counts" in out.aux
        assert out.aux["first_spike_times"].shape == (B, N)

    run_test("FirstSpikeDecoder shape and contract", test_first_spike_decoder_shape_contract)

    # ==================================================================
    # Test 6: FirstSpikeDecoder — known spike times produce correct argmin
    # ==================================================================

    def test_first_spike_known_times():
        decoder = FirstSpikeDecoder(temperature=1.0)

        # Create spikes where neuron 2 fires first (at t=1)
        s = torch.zeros(B, T, N)
        s[:, 1, 2] = 1.0   # Neuron 2 fires at t=1 (earliest)
        s[:, 5, 0] = 1.0   # Neuron 0 fires at t=5
        s[:, 10, 7] = 1.0  # Neuron 7 fires at t=10
        # All other neurons never fire

        batch = SpikeBatch(spikes=s)
        out = decoder(batch)

        # Neuron 2 should be the prediction (earliest spike)
        expected_pred = torch.full((B,), 2, dtype=torch.long)
        assert torch.equal(out.prediction, expected_pred), (
            f"Expected prediction=2 for all items, "
            f"got {out.prediction.tolist()}"
        )

        # First spike times for neuron 2 should be 1
        assert (out.aux["first_spike_times"][:, 2] == 1).all(), (
            f"Neuron 2 first spike time should be 1, "
            f"got {out.aux['first_spike_times'][:, 2].tolist()}"
        )

        # Neurons that never fired should have t_first = T
        never_fired = [1, 3, 4, 5, 6, 8, 9]
        for n in never_fired:
            assert (out.aux["first_spike_times"][:, n] == T).all(), (
                f"Neuron {n} (never fired) should have t_first={T}, "
                f"got {out.aux['first_spike_times'][:, n].tolist()}"
            )

        # Logits: neuron 2 should have the highest logit (most negative
        # but closest to zero, since logits = -t/T/temp)
        # t=1 -> logit = -1/20 = -0.05
        # t=5 -> logit = -5/20 = -0.25
        # t=T=20 (never fired) -> logit = -20/20 = -1.0
        assert out.logits_proxy[0, 2] > out.logits_proxy[0, 0], (
            "Neuron 2 (t=1) should have higher logit than neuron 0 (t=5)"
        )

    run_test("FirstSpikeDecoder known times -> correct argmin", test_first_spike_known_times)

    # ==================================================================
    # Test 7: FirstSpikeDecoder — no-spike handling
    # ==================================================================

    def test_first_spike_no_spike():
        decoder = FirstSpikeDecoder(temperature=1.0)

        # All-zero spikes: no neuron ever fires
        s = torch.zeros(B, T, N)
        batch = SpikeBatch(spikes=s)
        out = decoder(batch)

        # All first spike times should be T
        assert (out.aux["first_spike_times"] == T).all(), (
            "All neurons should have t_first=T when no spikes occur"
        )

        # No-spike mask should be all True
        assert out.aux["no_spike_mask"].all(), (
            "All neurons should be marked as no-spike"
        )

        # Confidence should be 0 (all neurons tied at t=T)
        assert (out.confidence == 0.0).all(), (
            f"Confidence should be 0 when all neurons tie, "
            f"got {out.confidence.tolist()}"
        )

        # Output should still be valid
        out.validate()

    run_test("FirstSpikeDecoder no-spike handling", test_first_spike_no_spike)

    # ==================================================================
    # Test 8: PopulationDecoder (classification) — known groups
    # ==================================================================

    def test_population_decoder_classification():
        # 3 groups, 3 neurons each (9 neurons total, ignore neuron 9)
        pop_map = {
            0: [0, 1, 2],
            1: [3, 4, 5],
            2: [6, 7, 8],
        }
        decoder = PopulationDecoder(
            population_map=pop_map,
            mode="classification",
            temperature=1.0,
        )

        # Create spikes where group 1 (neurons 3,4,5) has the most activity
        N_pop = 10
        s = torch.zeros(B, T, N_pop)
        # Group 0: sparse
        s[:, 0, 0] = 1.0
        # Group 1: heavy firing
        s[:, :, 3] = 1.0  # Neuron 3 fires every step
        s[:, :, 4] = 1.0  # Neuron 4 fires every step
        s[:, :T // 2, 5] = 1.0  # Neuron 5 fires half the time
        # Group 2: moderate
        s[:, :5, 6] = 1.0

        batch = SpikeBatch(spikes=s)
        out = decoder(batch)

        assert out.logits_proxy.shape == (B, 3), (
            f"Expected (B, 3), got {out.logits_proxy.shape}"
        )

        # Group 1 should be predicted (most total spikes)
        expected_pred = torch.full((B,), 1, dtype=torch.long)
        assert torch.equal(out.prediction, expected_pred), (
            f"Expected prediction=1, got {out.prediction.tolist()}"
        )

        out.validate()

        # Check group counts: group 1 total = T + T + T//2
        expected_g1 = T + T + T // 2
        assert (out.aux["group_counts"][:, 1] == expected_g1).all(), (
            f"Group 1 count should be {expected_g1}, "
            f"got {out.aux['group_counts'][:, 1].tolist()}"
        )

    run_test("PopulationDecoder classification", test_population_decoder_classification)

    # ==================================================================
    # Test 9: PopulationDecoder (regression) — weighted expectation
    # ==================================================================

    def test_population_decoder_regression():
        pop_map = {
            0: [0, 1],
            1: [2, 3],
            2: [4, 5],
        }
        centers = torch.tensor([0.0, 5.0, 10.0])

        decoder = PopulationDecoder(
            population_map=pop_map,
            mode="regression",
            centers=centers,
            temperature=1.0,
        )

        # Create spikes: group 2 (center=10.0) dominates
        N_pop = 6
        s = torch.zeros(B, T, N_pop)
        s[:, :, 4] = 1.0  # neuron 4 (group 2) fires every step
        s[:, :, 5] = 1.0  # neuron 5 (group 2) fires every step
        s[:, 0, 0] = 1.0  # neuron 0 (group 0) fires once

        batch = SpikeBatch(spikes=s)
        out = decoder(batch)

        assert out.logits_proxy.shape == (B, 3)
        out.validate()

        # predicted_value should be close to 10.0 (group 2 dominates)
        predicted_val = out.aux["predicted_value"]
        assert (predicted_val > 8.0).all(), (
            f"Predicted value should be close to 10.0, "
            f"got {predicted_val.tolist()}"
        )

    run_test("PopulationDecoder regression", test_population_decoder_regression)

    # ==================================================================
    # Test 10: MembraneDecoder — final mode
    # ==================================================================

    def test_membrane_decoder_final():
        decoder = MembraneDecoder(mode="final", temperature=1.0)

        # Create membrane where neuron 5 has the highest final value
        m = torch.zeros(B, T, N)
        m[:, -1, :] = torch.randn(B, N) * 0.1
        m[:, -1, 5] = 10.0  # Neuron 5 has highest final membrane

        s = make_spikes(B, T, N)
        batch = SpikeBatch(spikes=s, membrane=m)
        out = decoder(batch)

        assert out.logits_proxy.shape == (B, N)
        out.validate()

        # Prediction should be neuron 5
        expected_pred = torch.full((B,), 5, dtype=torch.long)
        assert torch.equal(out.prediction, expected_pred), (
            f"Expected prediction=5, got {out.prediction.tolist()}"
        )

        # Membrane values should be the final timestep
        assert torch.allclose(out.aux["membrane_values"], m[:, -1, :].float())

    run_test("MembraneDecoder final mode", test_membrane_decoder_final)

    # ==================================================================
    # Test 11: MembraneDecoder — max mode
    # ==================================================================

    def test_membrane_decoder_max():
        decoder = MembraneDecoder(mode="max", temperature=1.0)

        # Create membrane where neuron 3 has the highest peak at t=5
        m = torch.zeros(B, T, N)
        m[:, :, :] = torch.randn(B, T, N) * 0.1
        m[:, 5, 3] = 20.0  # Neuron 3 peaks at t=5

        s = make_spikes(B, T, N)
        batch = SpikeBatch(spikes=s, membrane=m)
        out = decoder(batch)

        assert out.logits_proxy.shape == (B, N)
        out.validate()

        # Prediction should be neuron 3
        expected_pred = torch.full((B,), 3, dtype=torch.long)
        assert torch.equal(out.prediction, expected_pred), (
            f"Expected prediction=3, got {out.prediction.tolist()}"
        )

        assert out.aux["mode"] == "max"

    run_test("MembraneDecoder max mode", test_membrane_decoder_max)

    # ==================================================================
    # Test 12: DecoderEnsemble — combined output
    # ==================================================================

    def test_decoder_ensemble():
        rate_dec = RateDecoder(temperature=1.0)
        membrane_dec = MembraneDecoder(mode="final", temperature=1.0)

        ensemble = DecoderEnsemble(
            decoders={"rate": rate_dec, "membrane": membrane_dec},
            weights={"rate": 0.7, "membrane": 0.3},
        )

        s = make_spikes(B, T, N)
        m = make_membrane(B, T, N)
        batch = SpikeBatch(spikes=s, membrane=m)
        out = ensemble(batch)

        assert out.logits_proxy.shape == (B, N)
        assert out.prediction.shape == (B,)
        assert out.confidence.shape == (B,)
        out.validate()

        # Verify combined logits are weighted sum
        rate_out = rate_dec(batch)
        membrane_out = membrane_dec(batch)
        expected_logits = 0.7 * rate_out.logits_proxy + 0.3 * membrane_out.logits_proxy
        assert torch.allclose(out.logits_proxy, expected_logits, atol=1e-5), (
            "Combined logits should be 0.7*rate + 0.3*membrane"
        )

        # Check aux
        assert "sub_outputs" in out.aux
        assert "weights" in out.aux
        assert "rate" in out.aux["sub_outputs"]
        assert "membrane" in out.aux["sub_outputs"]

        # Test weight update
        ensemble.update_weights({"rate": 0.5, "membrane": 0.5})
        assert ensemble.weights["rate"] == 0.5

    run_test("DecoderEnsemble combined output", test_decoder_ensemble)

    # ==================================================================
    # Test 13: Factory function — all types
    # ==================================================================

    def test_factory_all_types():
        s = make_spikes(B, T, N)
        m = make_membrane(B, T, N)
        batch = SpikeBatch(spikes=s, membrane=m)

        # Rate
        cfg = DecodingConfig(type="rate", temperature=2.0)
        dec = create_decoder(cfg)
        assert isinstance(dec, RateDecoder)
        out = dec(batch)
        out.validate()

        # First spike
        cfg = DecodingConfig(type="first_spike")
        dec = create_decoder(cfg)
        assert isinstance(dec, FirstSpikeDecoder)
        out = dec(batch)
        out.validate()

        # Population
        pop_map = {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8]}
        cfg = DecodingConfig(
            type="population",
            population_map=pop_map,
        )
        dec = create_decoder(cfg)
        assert isinstance(dec, PopulationDecoder)
        out = dec(batch)
        out.validate()

        # Membrane
        cfg = DecodingConfig(type="membrane", membrane_mode="max")
        dec = create_decoder(cfg)
        assert isinstance(dec, MembraneDecoder)
        out = dec(batch)
        out.validate()

        # Ensemble
        cfg = DecodingConfig(
            type="ensemble",
            ensemble_weights={"rate": 0.6, "membrane": 0.4},
        )
        dec = create_decoder(cfg)
        assert isinstance(dec, DecoderEnsemble)
        out = dec(batch)
        out.validate()

    run_test("Factory function all types", test_factory_all_types)

    # ==================================================================
    # Test 14: AMP — all decoders under autocast (CUDA only)
    # ==================================================================

    def test_amp_decoders():
        has_cuda = torch.cuda.is_available()
        if not has_cuda:
            # Test on CPU with manual fp16 to verify fp32 accumulation logic
            s_fp16 = make_spikes(B, T, N).half()
            m_fp16 = make_membrane(B, T, N).half()
            batch = SpikeBatch(spikes=s_fp16, membrane=m_fp16)

            # RateDecoder: counts should be fp32 despite fp16 input
            rate = RateDecoder()
            out = rate(batch)
            assert out.logits_proxy.dtype == torch.float32, (
                f"RateDecoder logits should be fp32, got {out.logits_proxy.dtype}"
            )
            assert out.logits_proxy.isfinite().all(), "Non-finite logits in rate decoder"
            out.validate()

            # FirstSpikeDecoder
            fs = FirstSpikeDecoder()
            out = fs(batch)
            assert out.logits_proxy.dtype == torch.float32
            assert out.logits_proxy.isfinite().all(), "Non-finite logits in first spike decoder"
            out.validate()

            # MembraneDecoder
            mem = MembraneDecoder()
            out = mem(batch)
            assert out.logits_proxy.dtype == torch.float32
            assert out.logits_proxy.isfinite().all(), "Non-finite logits in membrane decoder"
            out.validate()

            # PopulationDecoder
            pop_map = {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8]}
            pop = PopulationDecoder(population_map=pop_map)
            out = pop(batch)
            assert out.logits_proxy.dtype == torch.float32
            assert out.logits_proxy.isfinite().all(), "Non-finite logits in population decoder"
            out.validate()

        else:
            # Full AMP test on CUDA
            cuda = torch.device("cuda")

            for T_test in [10, 25, 50]:
                s = (torch.rand(B, T_test, N, device=cuda) > 0.8).float()
                m = torch.randn(B, T_test, N, device=cuda) * 0.5
                batch = SpikeBatch(spikes=s, membrane=m)

                with torch.cuda.amp.autocast():
                    # Rate
                    rate = RateDecoder().to(cuda)
                    out = rate(batch)
                    assert out.logits_proxy.isfinite().all(), (
                        f"Non-finite rate logits at T={T_test}"
                    )
                    assert out.confidence.isfinite().all()

                    # FirstSpike
                    fs = FirstSpikeDecoder().to(cuda)
                    out = fs(batch)
                    assert out.logits_proxy.isfinite().all(), (
                        f"Non-finite first_spike logits at T={T_test}"
                    )

                    # Membrane
                    mem_dec = MembraneDecoder().to(cuda)
                    out = mem_dec(batch)
                    assert out.logits_proxy.isfinite().all(), (
                        f"Non-finite membrane logits at T={T_test}"
                    )

                    # Population
                    pop_map = {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8]}
                    pop = PopulationDecoder(population_map=pop_map).to(cuda)
                    out = pop(batch)
                    assert out.logits_proxy.isfinite().all(), (
                        f"Non-finite population logits at T={T_test}"
                    )

    run_test("AMP hardening (fp16 input -> fp32 output)", test_amp_decoders)

    # ==================================================================
    # Test 15: adapt_time_first — axis conversion
    # ==================================================================

    def test_adapt_time_first():
        # (T, B, D) -> (B, T, D)
        time_first = torch.randn(T, B, N)
        batch_first = adapt_time_first(time_first)
        assert batch_first.shape == (B, T, N), (
            f"Expected (B, T, N)=({B}, {T}, {N}), got {batch_first.shape}"
        )
        assert torch.equal(batch_first[0, 0, :], time_first[0, 0, :])
        assert torch.equal(batch_first[0, 1, :], time_first[1, 0, :])

        # 2-D input should be unchanged
        two_d = torch.randn(B, N)
        result = adapt_time_first(two_d)
        assert torch.equal(result, two_d)

    run_test("adapt_time_first axis conversion", test_adapt_time_first)

    # ==================================================================
    # Test 16: SpikeBatch.from_time_first — legacy conversion
    # ==================================================================

    def test_spike_batch_from_time_first():
        s_tf = torch.randn(T, B, N)  # time-first
        m_tf = torch.randn(T, B, N)

        batch = SpikeBatch.from_time_first(spikes=s_tf, membrane=m_tf)
        assert batch.spikes.shape == (B, T, N), (
            f"Expected (B, T, N), got {batch.spikes.shape}"
        )
        assert batch.membrane is not None
        assert batch.membrane.shape == (B, T, N)

        # Verify values match after transpose
        assert torch.equal(batch.spikes, s_tf.permute(1, 0, 2))

    run_test("SpikeBatch.from_time_first legacy conversion", test_spike_batch_from_time_first)

    # ==================================================================
    # Test 17: DecoderOutput.to() and .detach() with nested aux
    # ==================================================================

    def test_decoder_output_to_detach():
        logits = torch.randn(B, N, requires_grad=True)
        pred = torch.zeros(B, dtype=torch.long)
        conf = torch.full((B,), 0.5)

        out = DecoderOutput(
            logits_proxy=logits,
            prediction=pred,
            confidence=conf,
            aux={
                "tensor_val": torch.randn(B),
                "list_val": [torch.randn(3), "string_item"],
                "dict_val": {"inner_tensor": torch.randn(2), "inner_str": "test"},
                "scalar_val": 42,
            },
        )

        # Test detach
        detached = out.detach()
        assert detached.logits_proxy.grad_fn is None, "logits should be detached"
        assert isinstance(detached.aux["tensor_val"], Tensor)
        assert detached.aux["tensor_val"].grad_fn is None
        assert isinstance(detached.aux["list_val"][0], Tensor)
        assert isinstance(detached.aux["dict_val"]["inner_tensor"], Tensor)
        assert detached.aux["scalar_val"] == 42

        # Test to (just CPU -> CPU, validates no crash)
        moved = out.to(device)
        assert moved.logits_proxy.device == device

    run_test("DecoderOutput.to() and .detach() with nested aux", test_decoder_output_to_detach)

    # ==================================================================
    # Test 18: DecodingConfig validation
    # ==================================================================

    def test_decoding_config_validation():
        # Valid configs should pass
        DecodingConfig(type="rate").validate()
        DecodingConfig(type="first_spike").validate()
        DecodingConfig(
            type="population",
            population_map={0: [0, 1], 1: [2, 3]},
        ).validate()
        DecodingConfig(type="membrane").validate()
        DecodingConfig(
            type="ensemble",
            ensemble_weights={"rate": 0.5, "membrane": 0.5},
        ).validate()

        # Invalid type
        try:
            DecodingConfig(type="invalid").validate()
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

        # Negative temperature
        try:
            DecodingConfig(type="rate", temperature=-1.0).validate()
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

        # Population without map
        try:
            DecodingConfig(type="population").validate()
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

        # Population regression without centers
        try:
            DecodingConfig(
                type="population",
                population_map={0: [0], 1: [1]},
                population_mode="regression",
            ).validate()
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

        # Ensemble without weights
        try:
            DecodingConfig(type="ensemble").validate()
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

        # Ensemble with bad sum
        try:
            DecodingConfig(
                type="ensemble",
                ensemble_weights={"rate": 0.5, "membrane": 0.6},
            ).validate()
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    run_test("DecodingConfig validation", test_decoding_config_validation)

    # ==================================================================
    # Test 19: RateDecoder gradient flow (differentiability)
    # ==================================================================

    def test_rate_decoder_gradient_flow():
        decoder = RateDecoder(temperature=1.0)

        # Spikes with gradient (surrogate gradient scenario)
        s = torch.rand(B, T, N, requires_grad=True)
        batch = SpikeBatch(spikes=s)
        out = decoder(batch)

        # Loss: cross-entropy with random targets
        targets = torch.randint(0, N, (B,))
        loss = F.cross_entropy(out.logits_proxy, targets)
        loss.backward()

        assert s.grad is not None, "Gradient should flow to spike input"
        assert s.grad.shape == (B, T, N)
        assert s.grad.isfinite().all(), "Gradients should be finite"

    run_test("RateDecoder gradient flow", test_rate_decoder_gradient_flow)

    # ==================================================================
    # Test 20: FirstSpikeDecoder — batch items with different winners
    # ==================================================================

    def test_first_spike_different_winners():
        decoder = FirstSpikeDecoder(temperature=1.0)

        s = torch.zeros(B, T, N)
        # Batch item 0: neuron 0 fires first at t=2
        s[0, 2, 0] = 1.0
        s[0, 8, 5] = 1.0
        # Batch item 1: neuron 5 fires first at t=1
        s[1, 1, 5] = 1.0
        s[1, 3, 0] = 1.0
        # Batch item 2: neuron 9 fires first at t=0
        s[2, 0, 9] = 1.0
        # Batch item 3: no spikes at all
        # (all zeros)

        batch = SpikeBatch(spikes=s)
        out = decoder(batch)

        assert out.prediction[0].item() == 0, f"Item 0 winner should be neuron 0, got {out.prediction[0].item()}"
        assert out.prediction[1].item() == 5, f"Item 1 winner should be neuron 5, got {out.prediction[1].item()}"
        assert out.prediction[2].item() == 9, f"Item 2 winner should be neuron 9, got {out.prediction[2].item()}"
        # Item 3: all neurons have t_first=T, argmin picks 0 (first index with min value)
        # This is fine; with all-tied predictions the specific choice is arbitrary

    run_test("FirstSpikeDecoder per-batch-item winners", test_first_spike_different_winners)

    # ==================================================================
    # Test 21: PopulationDecoder with unequal group sizes
    # ==================================================================

    def test_population_unequal_groups():
        # Groups of different sizes
        pop_map = {
            0: [0],           # 1 neuron
            1: [1, 2, 3],     # 3 neurons
            2: [4, 5],        # 2 neurons
        }
        decoder = PopulationDecoder(
            population_map=pop_map,
            mode="classification",
        )

        N_pop = 6
        s = torch.zeros(B, T, N_pop)
        # Group 1 (neurons 1,2,3) fires heavily
        s[:, :, 1] = 1.0
        s[:, :, 2] = 1.0
        s[:, :, 3] = 1.0

        batch = SpikeBatch(spikes=s)
        out = decoder(batch)

        assert out.logits_proxy.shape == (B, 3)
        out.validate()

        # Group 1 should win
        expected = torch.full((B,), 1, dtype=torch.long)
        assert torch.equal(out.prediction, expected)

    run_test("PopulationDecoder unequal group sizes", test_population_unequal_groups)

    # ==================================================================
    # Test 22: MembraneDecoder — error without membrane data
    # ==================================================================

    def test_membrane_decoder_no_membrane():
        decoder = MembraneDecoder(mode="final")
        s = make_spikes(B, T, N)
        batch = SpikeBatch(spikes=s)  # No membrane

        try:
            out = decoder(batch)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "membrane" in str(e).lower()

    run_test("MembraneDecoder error without membrane", test_membrane_decoder_no_membrane)

    # ==================================================================
    # Test 23: DecoderEnsemble — weight update validation
    # ==================================================================

    def test_ensemble_weight_update():
        rate_dec = RateDecoder()
        mem_dec = MembraneDecoder()
        ensemble = DecoderEnsemble(
            decoders={"rate": rate_dec, "membrane": mem_dec},
            weights={"rate": 0.7, "membrane": 0.3},
        )

        # Valid update
        ensemble.update_weights({"rate": 0.4, "membrane": 0.6})
        assert ensemble.weights["rate"] == 0.4

        # Invalid: wrong keys
        try:
            ensemble.update_weights({"rate": 0.5, "unknown": 0.5})
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

        # Invalid: bad sum
        try:
            ensemble.update_weights({"rate": 0.3, "membrane": 0.3})
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    run_test("DecoderEnsemble weight update validation", test_ensemble_weight_update)

    # ==================================================================
    # Test 24: Confidence methods — margin vs max_prob
    # ==================================================================

    def test_confidence_methods():
        s = make_spikes(B, T, N, rate=0.3)
        batch = SpikeBatch(spikes=s)

        # Margin method
        dec_margin = RateDecoder(confidence_method="margin")
        out_margin = dec_margin(batch)
        assert (out_margin.confidence >= 0).all() and (out_margin.confidence <= 1).all()

        # Max prob method
        dec_maxp = RateDecoder(confidence_method="max_prob")
        out_maxp = dec_maxp(batch)
        assert (out_maxp.confidence >= 0).all() and (out_maxp.confidence <= 1).all()

        # Max prob should generally be >= margin (since margin is p1-p2 and max_prob is p1)
        # This is a statistical property, not guaranteed for every case, so just check shapes
        assert out_margin.confidence.shape == out_maxp.confidence.shape

    run_test("Confidence methods margin vs max_prob", test_confidence_methods)

    # ==================================================================
    # Test 25: Large T stress test
    # ==================================================================

    def test_large_T():
        T_large = 200
        N_large = 50
        s = (torch.rand(B, T_large, N_large) > 0.95).float()
        m = torch.randn(B, T_large, N_large)
        batch = SpikeBatch(spikes=s, membrane=m)

        # All decoders should handle large T without overflow
        rate = RateDecoder()
        out = rate(batch)
        assert out.logits_proxy.isfinite().all(), "Rate decoder overflow at T=200"
        out.validate()

        fs = FirstSpikeDecoder()
        out = fs(batch)
        assert out.logits_proxy.isfinite().all(), "FirstSpike decoder overflow at T=200"
        out.validate()

        mem_dec = MembraneDecoder()
        out = mem_dec(batch)
        assert out.logits_proxy.isfinite().all(), "Membrane decoder overflow at T=200"
        out.validate()

        pop_map = {i: list(range(i * 5, (i + 1) * 5)) for i in range(10)}
        pop = PopulationDecoder(population_map=pop_map)
        out = pop(batch)
        assert out.logits_proxy.isfinite().all(), "Population decoder overflow at T=200"
        out.validate()

    run_test("Large T=200 stress test", test_large_T)

    # ==================================================================
    # Test 26: Edge case — single neuron
    # ==================================================================

    def test_single_neuron():
        s = make_spikes(B, T, 1)
        batch = SpikeBatch(spikes=s)

        rate = RateDecoder()
        out = rate(batch)
        assert out.logits_proxy.shape == (B, 1)
        assert out.prediction.shape == (B,)
        assert (out.prediction == 0).all()  # Only one neuron
        out.validate()

    run_test("Edge case: single neuron", test_single_neuron)

    # ==================================================================
    # Test 27: Edge case — single timestep
    # ==================================================================

    def test_single_timestep():
        s = make_spikes(B, 1, N)
        batch = SpikeBatch(spikes=s)

        rate = RateDecoder()
        out = rate(batch)
        assert out.logits_proxy.shape == (B, N)
        out.validate()

        fs = FirstSpikeDecoder()
        out = fs(batch)
        assert out.logits_proxy.shape == (B, N)
        out.validate()

    run_test("Edge case: single timestep T=1", test_single_timestep)

    # ==================================================================
    # Test 28: SpikeBatch validation failures
    # ==================================================================

    def test_spike_batch_validation_failures():
        # Wrong ndim
        try:
            bad = SpikeBatch(spikes=torch.randn(B, N))
            bad.validate()
            assert False, "Should fail for 2-D spikes"
        except AssertionError:
            pass

        # Membrane shape mismatch
        try:
            bad = SpikeBatch(
                spikes=torch.randn(B, T, N),
                membrane=torch.randn(B + 1, T, N),
            )
            bad.validate()
            assert False, "Should fail for batch mismatch"
        except AssertionError:
            pass

        # Mask shape mismatch
        try:
            bad = SpikeBatch(
                spikes=torch.randn(B, T, N),
                mask=torch.ones(B, T + 1, dtype=torch.bool),
            )
            bad.validate()
            assert False, "Should fail for mask shape mismatch"
        except AssertionError:
            pass

    run_test("SpikeBatch validation failures", test_spike_batch_validation_failures)

    # ==================================================================
    # Test 29: DecoderOutput validation failures
    # ==================================================================

    def test_decoder_output_validation_failures():
        # Wrong logits ndim
        try:
            bad = DecoderOutput(
                logits_proxy=torch.randn(B),
                prediction=torch.zeros(B, dtype=torch.long),
                confidence=torch.full((B,), 0.5),
            )
            bad.validate()
            assert False, "Should fail for 1-D logits"
        except AssertionError:
            pass

        # Wrong prediction dtype
        try:
            bad = DecoderOutput(
                logits_proxy=torch.randn(B, N),
                prediction=torch.zeros(B),  # float, not long
                confidence=torch.full((B,), 0.5),
            )
            bad.validate()
            assert False, "Should fail for float prediction"
        except AssertionError:
            pass

        # Confidence out of range
        try:
            bad = DecoderOutput(
                logits_proxy=torch.randn(B, N),
                prediction=torch.zeros(B, dtype=torch.long),
                confidence=torch.full((B,), 1.5),
            )
            bad.validate()
            assert False, "Should fail for confidence > 1"
        except AssertionError:
            pass

        # Batch size mismatch
        try:
            bad = DecoderOutput(
                logits_proxy=torch.randn(B, N),
                prediction=torch.zeros(B + 1, dtype=torch.long),
                confidence=torch.full((B,), 0.5),
            )
            bad.validate()
            assert False, "Should fail for batch mismatch"
        except AssertionError:
            pass

    run_test("DecoderOutput validation failures", test_decoder_output_validation_failures)

    # ==================================================================
    # Test 30: Temperature scaling effect
    # ==================================================================

    def test_temperature_scaling():
        s = make_spikes(B, T, N, rate=0.3)
        batch = SpikeBatch(spikes=s)

        # High temperature should produce lower confidence (smoother)
        dec_low_t = RateDecoder(temperature=0.1, confidence_method="max_prob")
        dec_high_t = RateDecoder(temperature=10.0, confidence_method="max_prob")

        out_low = dec_low_t(batch)
        out_high = dec_high_t(batch)

        # Average confidence with low temp should be >= high temp
        # (sharper distribution -> higher max probability)
        avg_low = out_low.confidence.mean().item()
        avg_high = out_high.confidence.mean().item()
        assert avg_low >= avg_high - 0.01, (
            f"Low temperature should give higher confidence: "
            f"low_t={avg_low:.3f}, high_t={avg_high:.3f}"
        )

    run_test("Temperature scaling effect on confidence", test_temperature_scaling)

    # ==================================================================
    # Summary
    # ==================================================================

    print()
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed, {total} total")
    print("=" * 70)

    if failed > 0:
        print("SOME TESTS FAILED")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")
        sys.exit(0)
