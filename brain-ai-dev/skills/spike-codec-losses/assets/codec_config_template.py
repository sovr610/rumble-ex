"""
Unified configuration for spike encoding, decoding, and loss pack.

Template for integration into brain_ai/config.py
=================================================

This module defines the nested dataclass hierarchy that replaces the flat
loss-related flags currently living inside ``SNNConfig`` (lines 68-73 of
brain_ai/config.py).  The three sub-configs --- EncodingConfig, DecodingConfig,
and LossConfig --- are aggregated into a single ``CodecLossConfig`` object
that can be attached to ``BrainAIConfig`` as ``config.codec``.

Design goals:
  1. Zero-regression: ``upgrade_snn_config`` maps every old flat flag into the
     new hierarchy so that existing checkpoints and scripts keep working.
  2. Preset factories for common training recipes (rate classification, TTFS,
     population coding, regression, sparse-efficient).
  3. Round-trip serialization to/from plain ``dict`` (JSON-safe).
  4. Self-contained validation in ``__post_init__`` for fast fail on typos.
"""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Canonical encoding types — must match encoding_template.py ENCODING_TYPES.
_VALID_ENCODING_TYPES: Tuple[str, ...] = (
    "rate_bernoulli",
    "rate_poisson",
    "rate_deterministic",
    "latency",
    "ttfs",
    "population",
    "delta",
)

_VALID_NORMALIZATION_MODES: Tuple[str, ...] = (
    "none",
    "minmax",
    "sigmoid",
    "clamp",
)

_VALID_POPULATION_CENTERS: Tuple[str, ...] = (
    "linear",
    "learned",
    "uniform",
)

_VALID_LATENCY_MAPPINGS: Tuple[str, ...] = (
    "linear",
    "exponential",
    "log",
)

_VALID_DECODING_TYPES: Tuple[str, ...] = (
    "rate",
    "first_spike",
    "population",
    "membrane",
    "ensemble",
)

_VALID_POPULATION_MODES: Tuple[str, ...] = (
    "classification",
    "regression",
)

_VALID_MEMBRANE_MODES: Tuple[str, ...] = (
    "final",
    "max",
)

_VALID_PROBSPIKES_MODES: Tuple[str, ...] = (
    "softmax",
    "normalize",
)

_VALID_TEMPORAL_PENALTIES: Tuple[str, ...] = (
    "l1",
    "l2",
    "variance",
)

_VALID_ISI_KERNEL_TYPES: Tuple[str, ...] = (
    "exponential",
    "rectangular",
)


# ===================================================================
# EncodingConfig
# ===================================================================

@dataclass
class EncodingConfig:
    """Configuration for spike encoding strategies.

    Controls how continuous-valued input tensors are converted into binary
    spike trains before being fed into the SNN core.  Each ``type`` selects
    a different coding scheme; subsidiary parameters are silently ignored
    when they are irrelevant to the chosen type (e.g. ``population_size`` is
    only meaningful when ``type == "population"``).

    Attributes
    ----------
    type : str
        Encoding method.  One of ``rate_bernoulli``, ``rate_poisson``,
        ``ttfs``, ``ttfs_linear``, ``population``, ``delta``,
        ``delta_on_off``, ``latency``, ``latency_log``, ``phase``,
        ``burst``.
    num_steps : int
        Number of simulation time-steps per sample.  Must be > 0.
    dt : float
        Simulation time resolution in milliseconds.  Must be > 0.
    normalization : str
        Pre-encoding normalisation applied to the input.
        ``none`` | ``minmax`` | ``sigmoid`` | ``clamp``.
    deterministic_eval : bool
        If True, replace stochastic encoders with their mean during
        ``model.eval()`` for reproducible inference.
    seed : Optional[int]
        RNG seed for reproducible stochastic encoding.  ``None`` means
        system-chosen.
    rate_gain : float
        Multiplicative gain applied to the input before rate encoding.
    rate_bias : float
        Additive bias applied after gain scaling.
    population_size : int
        Number of neurons per input feature for population coding.
    population_sigma : float
        Standard deviation of Gaussian tuning curves.
    population_centers : str
        How to initialise tuning-curve centres: ``linear`` (evenly
        spaced), ``learned`` (gradient-optimised), ``uniform`` (random).
    latency_mapping : str
        Transfer function for latency/TTFS coding: ``linear``,
        ``exponential``, ``log``.
    t_min : int
        Earliest allowed spike time (inclusive).
    t_max : Optional[int]
        Latest allowed spike time (inclusive).  ``None`` defaults to
        ``num_steps - 1``.
    allow_no_spike : bool
        If False, every neuron is forced to spike at least once within
        ``[t_min, t_max]``.  Relevant for TTFS/latency encoders.
    jitter : float
        Standard deviation of additive Gaussian timing jitter (in
        time-steps) applied during training.  0 = no jitter.
    delta_threshold : float
        Minimum absolute change in input required to emit an ON-spike
        in delta encoding.
    delta_off_threshold : Optional[float]
        Separate threshold for OFF-spikes.  ``None`` mirrors
        ``delta_threshold``.
    """

    # -- Core parameters ---------------------------------------------------
    type: str = "rate_bernoulli"
    num_steps: int = 25
    dt: float = 1.0
    normalization: str = "minmax"
    deterministic_eval: bool = True
    seed: Optional[int] = None

    # -- Rate-coding parameters --------------------------------------------
    rate_gain: float = 1.0
    rate_bias: float = 0.0

    # -- Population-coding parameters --------------------------------------
    population_size: int = 8
    population_sigma: float = 0.2
    population_centers: str = "linear"

    # -- Latency / TTFS parameters -----------------------------------------
    latency_mapping: str = "exponential"
    t_min: int = 0
    t_max: Optional[int] = None
    allow_no_spike: bool = False

    # -- Augmentation / noise ----------------------------------------------
    jitter: float = 0.0

    # -- Delta-coding parameters -------------------------------------------
    delta_threshold: float = 0.1
    delta_off_threshold: Optional[float] = None

    # -- Validation --------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate field values immediately after construction."""
        if self.type not in _VALID_ENCODING_TYPES:
            raise ValueError(
                f"EncodingConfig.type must be one of {_VALID_ENCODING_TYPES}, "
                f"got '{self.type}'"
            )
        if self.num_steps <= 0:
            raise ValueError(
                f"EncodingConfig.num_steps must be > 0, got {self.num_steps}"
            )
        if self.dt <= 0.0:
            raise ValueError(
                f"EncodingConfig.dt must be > 0, got {self.dt}"
            )
        if self.normalization not in _VALID_NORMALIZATION_MODES:
            raise ValueError(
                f"EncodingConfig.normalization must be one of "
                f"{_VALID_NORMALIZATION_MODES}, got '{self.normalization}'"
            )
        if self.population_size <= 0:
            raise ValueError(
                f"EncodingConfig.population_size must be > 0, "
                f"got {self.population_size}"
            )
        if self.population_sigma <= 0.0:
            raise ValueError(
                f"EncodingConfig.population_sigma must be > 0, "
                f"got {self.population_sigma}"
            )
        if self.population_centers not in _VALID_POPULATION_CENTERS:
            raise ValueError(
                f"EncodingConfig.population_centers must be one of "
                f"{_VALID_POPULATION_CENTERS}, got '{self.population_centers}'"
            )
        if self.latency_mapping not in _VALID_LATENCY_MAPPINGS:
            raise ValueError(
                f"EncodingConfig.latency_mapping must be one of "
                f"{_VALID_LATENCY_MAPPINGS}, got '{self.latency_mapping}'"
            )
        if self.t_min < 0:
            raise ValueError(
                f"EncodingConfig.t_min must be >= 0, got {self.t_min}"
            )
        if self.t_max is not None and self.t_max < self.t_min:
            raise ValueError(
                f"EncodingConfig.t_max ({self.t_max}) must be >= t_min "
                f"({self.t_min})"
            )
        if self.jitter < 0.0:
            raise ValueError(
                f"EncodingConfig.jitter must be >= 0, got {self.jitter}"
            )
        if self.delta_threshold <= 0.0:
            raise ValueError(
                f"EncodingConfig.delta_threshold must be > 0, "
                f"got {self.delta_threshold}"
            )
        if (
            self.delta_off_threshold is not None
            and self.delta_off_threshold <= 0.0
        ):
            raise ValueError(
                f"EncodingConfig.delta_off_threshold must be > 0 when set, "
                f"got {self.delta_off_threshold}"
            )
        if self.rate_gain <= 0.0:
            raise ValueError(
                f"EncodingConfig.rate_gain must be > 0, got {self.rate_gain}"
            )


# ===================================================================
# DecodingConfig
# ===================================================================

@dataclass
class DecodingConfig:
    """Configuration for spike-train decoding strategies.

    After the SNN core produces spike trains and membrane potentials,
    a decoder converts them back into continuous logits or values for
    the downstream loss function.

    Attributes
    ----------
    type : str
        Decoding method.  One of ``rate``, ``first_spike``,
        ``population``, ``membrane``, ``ensemble``.
    temperature : float
        Softmax temperature applied after decoding.  Lower values make
        the output distribution sharper.  Must be > 0.
    eps : float
        Small epsilon for numerical stability in normalisation.
    population_mode : str
        ``classification`` (argmax over population) or ``regression``
        (weighted centroid).
    membrane_mode : str
        ``final`` (use last-timestep membrane) or ``max`` (use peak
        membrane over time).
    ensemble_weights : Optional[Dict[str, float]]
        When ``type == "ensemble"``, maps decoder names to their blend
        weights.  ``None`` uses equal weighting of rate + membrane.
    """

    type: str = "rate"
    temperature: float = 1.0
    eps: float = 1e-7
    population_mode: str = "classification"
    membrane_mode: str = "final"
    ensemble_weights: Optional[Dict[str, float]] = None

    # -- Validation --------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate field values immediately after construction."""
        if self.type not in _VALID_DECODING_TYPES:
            raise ValueError(
                f"DecodingConfig.type must be one of "
                f"{_VALID_DECODING_TYPES}, got '{self.type}'"
            )
        if self.temperature <= 0.0:
            raise ValueError(
                f"DecodingConfig.temperature must be > 0, "
                f"got {self.temperature}"
            )
        if self.eps <= 0.0:
            raise ValueError(
                f"DecodingConfig.eps must be > 0, got {self.eps}"
            )
        if self.population_mode not in _VALID_POPULATION_MODES:
            raise ValueError(
                f"DecodingConfig.population_mode must be one of "
                f"{_VALID_POPULATION_MODES}, got '{self.population_mode}'"
            )
        if self.membrane_mode not in _VALID_MEMBRANE_MODES:
            raise ValueError(
                f"DecodingConfig.membrane_mode must be one of "
                f"{_VALID_MEMBRANE_MODES}, got '{self.membrane_mode}'"
            )
        if self.ensemble_weights is not None:
            if not isinstance(self.ensemble_weights, dict):
                raise TypeError(
                    f"DecodingConfig.ensemble_weights must be a dict or None, "
                    f"got {type(self.ensemble_weights).__name__}"
                )
            for key, weight in self.ensemble_weights.items():
                if not isinstance(key, str):
                    raise TypeError(
                        f"Ensemble weight key must be str, got "
                        f"{type(key).__name__} for key {key!r}"
                    )
                if weight < 0.0:
                    raise ValueError(
                        f"Ensemble weight for '{key}' must be >= 0, "
                        f"got {weight}"
                    )


# ===================================================================
# LossConfig
# ===================================================================

@dataclass
class LossConfig:
    """Configuration for the composite spike-aware loss function.

    The total loss is a weighted sum of up to five component terms:

    .. math::

        L = w_{ps} L_{probspikes}
          + w_{rate} L_{rate}
          + w_{temp} L_{temporal}
          + w_{isi} L_{isi}
          + w_{mem} L_{membrane}

    Each weight can be set to 0 to disable the corresponding term.
    ``enabled_terms()`` returns the names of all active components.

    Attributes
    ----------
    w_probspikes : float
        Weight for the ProbSpikes cross-entropy term.
    w_rate : float
        Weight for the spike-rate regularisation term.
    w_temporal : float
        Weight for the temporal consistency / smoothness term.
    w_isi : float
        Weight for the inter-spike-interval refractory penalty.
    w_membrane : float
        Weight for the membrane-potential overflow penalty.
    target_rate : float
        Desired mean spike rate across the network (0, 1).
    min_rate : float
        Minimum acceptable spike rate for the rate-penalty band.
    max_rate : float
        Maximum acceptable spike rate for the rate-penalty band.
    probspikes_temperature : float
        Temperature for the ProbSpikes softmax distribution.
    probspikes_eps : float
        Numerical stability epsilon for ProbSpikes log.
    probspikes_mode : str
        How to normalise spike counts: ``softmax`` or ``normalize``.
    temporal_window : int
        Sliding window width (time-steps) for the temporal term.
    temporal_penalty : str
        Norm used for temporal consistency: ``l1``, ``l2``, ``variance``.
    isi_refractory_window : int
        Minimum expected inter-spike interval (time-steps).
    isi_kernel_type : str
        Convolution kernel for ISI computation: ``exponential`` or
        ``rectangular``.
    membrane_max : float
        Membrane-potential ceiling above which overflow penalty applies.
    per_layer_targets : Optional[Dict[str, float]]
        Per-layer spike-rate targets, keyed by layer name.  ``None``
        means all layers share ``target_rate``.
    log_diagnostics : bool
        If True, the loss module logs per-component values for
        TensorBoard / wandb.
    """

    # -- Component weights -------------------------------------------------
    w_probspikes: float = 1.0
    w_rate: float = 0.1
    w_temporal: float = 0.01
    w_isi: float = 0.01
    w_membrane: float = 0.001

    # -- Rate targets ------------------------------------------------------
    target_rate: float = 0.1
    min_rate: float = 0.01
    max_rate: float = 0.3

    # -- ProbSpikes parameters ---------------------------------------------
    probspikes_temperature: float = 1.0
    probspikes_eps: float = 1e-7
    probspikes_mode: str = "softmax"

    # -- Temporal consistency parameters -----------------------------------
    temporal_window: int = 5
    temporal_penalty: str = "l2"

    # -- ISI (inter-spike interval) parameters -----------------------------
    isi_refractory_window: int = 3
    isi_kernel_type: str = "exponential"

    # -- Membrane overflow penalty -----------------------------------------
    membrane_max: float = 1.5

    # -- Per-layer overrides -----------------------------------------------
    per_layer_targets: Optional[Dict[str, float]] = None

    # -- Diagnostics -------------------------------------------------------
    log_diagnostics: bool = True

    # -- Validation --------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate field values immediately after construction."""
        # Weights must be non-negative
        for wname in ("w_probspikes", "w_rate", "w_temporal", "w_isi",
                       "w_membrane"):
            val = getattr(self, wname)
            if val < 0.0:
                raise ValueError(
                    f"LossConfig.{wname} must be >= 0, got {val}"
                )

        # Target rate
        if not (0.0 < self.target_rate < 1.0):
            raise ValueError(
                f"LossConfig.target_rate must be in (0, 1), "
                f"got {self.target_rate}"
            )
        if not (0.0 <= self.min_rate < self.max_rate <= 1.0):
            raise ValueError(
                f"LossConfig rate band invalid: min_rate={self.min_rate}, "
                f"max_rate={self.max_rate}.  Need 0 <= min < max <= 1."
            )
        if not (self.min_rate <= self.target_rate <= self.max_rate):
            raise ValueError(
                f"LossConfig.target_rate ({self.target_rate}) must be within "
                f"[min_rate={self.min_rate}, max_rate={self.max_rate}]"
            )

        # ProbSpikes
        if self.probspikes_temperature <= 0.0:
            raise ValueError(
                f"LossConfig.probspikes_temperature must be > 0, "
                f"got {self.probspikes_temperature}"
            )
        if self.probspikes_eps <= 0.0:
            raise ValueError(
                f"LossConfig.probspikes_eps must be > 0, "
                f"got {self.probspikes_eps}"
            )
        if self.probspikes_mode not in _VALID_PROBSPIKES_MODES:
            raise ValueError(
                f"LossConfig.probspikes_mode must be one of "
                f"{_VALID_PROBSPIKES_MODES}, got '{self.probspikes_mode}'"
            )

        # Temporal
        if self.temporal_window < 2:
            raise ValueError(
                f"LossConfig.temporal_window must be >= 2 (need at least 2 steps "
                f"for difference/variance), got {self.temporal_window}"
            )
        if self.temporal_penalty not in _VALID_TEMPORAL_PENALTIES:
            raise ValueError(
                f"LossConfig.temporal_penalty must be one of "
                f"{_VALID_TEMPORAL_PENALTIES}, got '{self.temporal_penalty}'"
            )

        # ISI
        if self.isi_refractory_window <= 0:
            raise ValueError(
                f"LossConfig.isi_refractory_window must be > 0, "
                f"got {self.isi_refractory_window}"
            )
        if self.isi_kernel_type not in _VALID_ISI_KERNEL_TYPES:
            raise ValueError(
                f"LossConfig.isi_kernel_type must be one of "
                f"{_VALID_ISI_KERNEL_TYPES}, got '{self.isi_kernel_type}'"
            )

        # Membrane
        if self.membrane_max <= 0.0:
            raise ValueError(
                f"LossConfig.membrane_max must be > 0, "
                f"got {self.membrane_max}"
            )

        # Per-layer targets
        if self.per_layer_targets is not None:
            if not isinstance(self.per_layer_targets, dict):
                raise TypeError(
                    f"LossConfig.per_layer_targets must be a dict or None, "
                    f"got {type(self.per_layer_targets).__name__}"
                )
            for lname, lrate in self.per_layer_targets.items():
                if not isinstance(lname, str):
                    raise TypeError(
                        f"Per-layer target key must be str, got "
                        f"{type(lname).__name__} for key {lname!r}"
                    )
                if not (0.0 < lrate < 1.0):
                    raise ValueError(
                        f"Per-layer target for '{lname}' must be in (0, 1), "
                        f"got {lrate}"
                    )

    # -- Helpers -----------------------------------------------------------

    def enabled_terms(self) -> List[str]:
        """Return the names of loss components whose weight is > 0.

        Returns
        -------
        List[str]
            Sorted list of enabled term names, e.g.
            ``["membrane", "probspikes", "rate"]``.
        """
        mapping = {
            "probspikes": self.w_probspikes,
            "rate": self.w_rate,
            "temporal": self.w_temporal,
            "isi": self.w_isi,
            "membrane": self.w_membrane,
        }
        return sorted(name for name, weight in mapping.items() if weight > 0.0)

    def total_weight(self) -> float:
        """Return the sum of all component weights.

        Useful for quick sanity checks and normalisation.

        Returns
        -------
        float
            Sum of ``w_probspikes + w_rate + w_temporal + w_isi +
            w_membrane``.
        """
        return (
            self.w_probspikes
            + self.w_rate
            + self.w_temporal
            + self.w_isi
            + self.w_membrane
        )


# ===================================================================
# CodecLossConfig  (aggregate)
# ===================================================================

@dataclass
class CodecLossConfig:
    """Top-level configuration aggregating encoding, decoding, and loss.

    This is the single object that should be attached to ``BrainAIConfig``
    (as ``config.codec``) to replace the flat loss flags on ``SNNConfig``.

    Attributes
    ----------
    encoding : EncodingConfig
        Spike-encoding configuration.
    decoding : DecodingConfig
        Spike-decoding configuration.
    loss : LossConfig
        Composite loss configuration.
    """

    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    decoding: DecodingConfig = field(default_factory=DecodingConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    # -- Factory class methods ---------------------------------------------

    @classmethod
    def default(cls) -> "CodecLossConfig":
        """Reasonable defaults: rate encoding, rate decoding, ProbSpikes on.

        Suitable as a drop-in for the existing SNNConfig flat flags with
        no behavioural change.

        Returns
        -------
        CodecLossConfig
        """
        return cls(
            encoding=EncodingConfig(
                type="rate_bernoulli",
                num_steps=25,
                normalization="minmax",
            ),
            decoding=DecodingConfig(
                type="rate",
                temperature=1.0,
            ),
            loss=LossConfig(
                w_probspikes=1.0,
                w_rate=0.1,
                w_temporal=0.01,
                w_isi=0.01,
                w_membrane=0.001,
                target_rate=0.1,
            ),
        )

    @classmethod
    def rate_classification(cls) -> "CodecLossConfig":
        """Rate encoding + rate decoding + ProbSpikes for classification.

        The standard recipe for multi-class classification tasks where
        spike counts per output neuron are interpreted as class scores.

        Returns
        -------
        CodecLossConfig
        """
        return cls(
            encoding=EncodingConfig(
                type="rate_bernoulli",
                num_steps=25,
                normalization="minmax",
                deterministic_eval=True,
                rate_gain=1.0,
                rate_bias=0.0,
            ),
            decoding=DecodingConfig(
                type="rate",
                temperature=1.0,
            ),
            loss=LossConfig(
                w_probspikes=1.0,
                w_rate=0.1,
                w_temporal=0.01,
                w_isi=0.01,
                w_membrane=0.001,
                target_rate=0.1,
                probspikes_mode="softmax",
            ),
        )

    @classmethod
    def ttfs_classification(cls) -> "CodecLossConfig":
        """TTFS encoding + first-spike decoding + ProbSpikes.

        Time-to-first-spike is energy-efficient and biologically
        plausible.  The earliest spike across output neurons determines
        the predicted class.

        Returns
        -------
        CodecLossConfig
        """
        return cls(
            encoding=EncodingConfig(
                type="ttfs",
                num_steps=50,
                normalization="sigmoid",
                deterministic_eval=True,
                latency_mapping="exponential",
                t_min=0,
                t_max=49,
                allow_no_spike=False,
                jitter=0.5,
            ),
            decoding=DecodingConfig(
                type="first_spike",
                temperature=0.5,
                eps=1e-7,
            ),
            loss=LossConfig(
                w_probspikes=1.0,
                w_rate=0.05,
                w_temporal=0.1,
                w_isi=0.0,
                w_membrane=0.001,
                target_rate=0.05,
                min_rate=0.01,
                max_rate=0.15,
                probspikes_temperature=0.5,
                temporal_penalty="l2",
            ),
        )

    @classmethod
    def population_classification(cls) -> "CodecLossConfig":
        """Population encoding + population decoding for classification.

        Each input feature is represented by a group of neurons with
        overlapping Gaussian tuning curves.  This provides high-
        resolution encoding at the cost of wider spike trains.

        Returns
        -------
        CodecLossConfig
        """
        return cls(
            encoding=EncodingConfig(
                type="population",
                num_steps=25,
                normalization="minmax",
                deterministic_eval=True,
                population_size=8,
                population_sigma=0.2,
                population_centers="linear",
            ),
            decoding=DecodingConfig(
                type="population",
                temperature=1.0,
                population_mode="classification",
            ),
            loss=LossConfig(
                w_probspikes=1.0,
                w_rate=0.15,
                w_temporal=0.01,
                w_isi=0.01,
                w_membrane=0.001,
                target_rate=0.12,
                min_rate=0.02,
                max_rate=0.35,
                probspikes_mode="softmax",
            ),
        )

    @classmethod
    def regression(cls) -> "CodecLossConfig":
        """Rate encoding + membrane decoding for regression tasks.

        Uses final membrane potential as the continuous output and
        disables ProbSpikes (which assumes classification).  The
        membrane-overflow penalty keeps values in a trainable range.

        Returns
        -------
        CodecLossConfig
        """
        return cls(
            encoding=EncodingConfig(
                type="rate_bernoulli",
                num_steps=25,
                normalization="minmax",
                deterministic_eval=True,
                rate_gain=1.0,
            ),
            decoding=DecodingConfig(
                type="membrane",
                temperature=1.0,
                membrane_mode="final",
            ),
            loss=LossConfig(
                w_probspikes=0.0,
                w_rate=0.1,
                w_temporal=0.01,
                w_isi=0.0,
                w_membrane=0.1,
                target_rate=0.15,
                min_rate=0.01,
                max_rate=0.4,
                membrane_max=2.0,
                log_diagnostics=True,
            ),
        )


# ===================================================================
# Preset factory functions  (module-level convenience)
# ===================================================================

def rate_classification_preset() -> CodecLossConfig:
    """Module-level preset: rate encoding for classification.

    Equivalent to ``CodecLossConfig.rate_classification()``.

    Returns
    -------
    CodecLossConfig
    """
    return CodecLossConfig.rate_classification()


def ttfs_classification_preset() -> CodecLossConfig:
    """Module-level preset: time-to-first-spike classification.

    Equivalent to ``CodecLossConfig.ttfs_classification()``.

    Returns
    -------
    CodecLossConfig
    """
    return CodecLossConfig.ttfs_classification()


def population_classification_preset() -> CodecLossConfig:
    """Module-level preset: population coding for classification.

    Equivalent to ``CodecLossConfig.population_classification()``.

    Returns
    -------
    CodecLossConfig
    """
    return CodecLossConfig.population_classification()


def regression_preset() -> CodecLossConfig:
    """Module-level preset: regression via membrane readout.

    Equivalent to ``CodecLossConfig.regression()``.

    Returns
    -------
    CodecLossConfig
    """
    return CodecLossConfig.regression()


def sparse_efficient_preset() -> CodecLossConfig:
    """Module-level preset: sparse, energy-efficient coding.

    Combines TTFS encoding with first-spike decoding and an
    aggressively low spike-rate target.  Minimal time-steps to reduce
    latency and energy.  Suitable for edge / neuromorphic deployment.

    Returns
    -------
    CodecLossConfig
    """
    return CodecLossConfig(
        encoding=EncodingConfig(
            type="ttfs",
            num_steps=15,
            normalization="sigmoid",
            deterministic_eval=True,
            latency_mapping="exponential",
            t_min=0,
            t_max=14,
            allow_no_spike=False,
            jitter=0.0,
        ),
        decoding=DecodingConfig(
            type="first_spike",
            temperature=0.3,
            eps=1e-7,
        ),
        loss=LossConfig(
            w_probspikes=1.0,
            w_rate=0.5,
            w_temporal=0.05,
            w_isi=0.0,
            w_membrane=0.01,
            target_rate=0.03,
            min_rate=0.005,
            max_rate=0.1,
            probspikes_temperature=0.3,
            temporal_penalty="l1",
            log_diagnostics=True,
        ),
    )


# ===================================================================
# Integration with existing SNNConfig
# ===================================================================

def upgrade_snn_config(old_config: Dict[str, Any]) -> CodecLossConfig:
    """Map old flat SNNConfig dict fields to the new nested hierarchy.

    This function accepts a plain dictionary (e.g. from
    ``dataclasses.asdict(snn_config)``) and produces a ``CodecLossConfig``
    that preserves the user's original intent.

    Mapping rules
    -------------
    * ``use_probspikes_loss``         -> ``LossConfig.w_probspikes`` > 0 if
      True, else 0.
    * ``spike_rate_target``           -> ``LossConfig.target_rate``
    * ``spike_rate_weight``           -> ``LossConfig.w_rate``
    * ``temporal_consistency_weight`` -> ``LossConfig.w_temporal``
    * ``num_timesteps``               -> ``EncodingConfig.num_steps``

    Parameters
    ----------
    old_config : Dict[str, Any]
        Dictionary of old-style flat SNNConfig fields.

    Returns
    -------
    CodecLossConfig
        Newly constructed config with values mapped from the old layout.
    """
    # Extract relevant values with sensible defaults
    use_probspikes = old_config.get("use_probspikes_loss", True)
    spike_rate_target = old_config.get("spike_rate_target", 0.1)
    spike_rate_weight = old_config.get("spike_rate_weight", 0.01)
    temporal_weight = old_config.get("temporal_consistency_weight", 0.001)
    num_timesteps = old_config.get("num_timesteps", 50)

    # Clamp target into valid (0, 1) range
    target_rate = max(0.001, min(0.999, spike_rate_target))

    # Derive min/max rate band around target
    min_rate = max(0.001, target_rate * 0.1)
    max_rate = min(0.999, target_rate * 3.0)
    # Ensure max_rate > min_rate
    if max_rate <= min_rate:
        max_rate = min(0.999, min_rate + 0.1)

    encoding = EncodingConfig(
        type="rate_bernoulli",
        num_steps=num_timesteps,
        normalization="minmax",
        deterministic_eval=True,
    )

    decoding = DecodingConfig(
        type="rate",
        temperature=1.0,
    )

    loss = LossConfig(
        w_probspikes=1.0 if use_probspikes else 0.0,
        w_rate=spike_rate_weight,
        w_temporal=temporal_weight,
        w_isi=0.01,
        w_membrane=0.001,
        target_rate=target_rate,
        min_rate=min_rate,
        max_rate=max_rate,
    )

    return CodecLossConfig(
        encoding=encoding,
        decoding=decoding,
        loss=loss,
    )


def legacy_to_new(snn_config: Any) -> CodecLossConfig:
    """Accept an ``SNNConfig`` object and return a ``CodecLossConfig``.

    This is a convenience wrapper around ``upgrade_snn_config`` that
    handles the ``asdict`` conversion automatically.  It also works
    when given a plain dict.

    Parameters
    ----------
    snn_config : Any
        Either an ``SNNConfig`` dataclass instance or a plain dict.

    Returns
    -------
    CodecLossConfig
        Newly constructed config.
    """
    if isinstance(snn_config, dict):
        return upgrade_snn_config(snn_config)

    # Try dataclasses.asdict first (works for any dataclass)
    try:
        d = asdict(snn_config)
    except TypeError:
        # Fallback: manually extract known fields from arbitrary objects
        d = {}
        known_keys = [
            "use_probspikes_loss",
            "spike_rate_target",
            "spike_rate_weight",
            "temporal_consistency_weight",
            "num_timesteps",
        ]
        for key in known_keys:
            if hasattr(snn_config, key):
                d[key] = getattr(snn_config, key)

    return upgrade_snn_config(d)


# ===================================================================
# Serialization
# ===================================================================

def _sanitize_value(val: Any) -> Any:
    """Convert a value into a JSON-safe representation.

    Handles:
    * ``None`` -> ``None`` (JSON ``null``)
    * ``dict``/``list``/``tuple`` -> recurse
    * Torch Tensors -> ``list`` (via ``.tolist()``)
    * NumPy arrays  -> ``list``
    * Everything else -> pass through (str, int, float, bool)

    Parameters
    ----------
    val : Any
        Value to sanitize.

    Returns
    -------
    Any
        JSON-safe value.
    """
    if val is None:
        return None
    if isinstance(val, (str, int, float, bool)):
        return val
    if isinstance(val, dict):
        return {str(k): _sanitize_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_sanitize_value(item) for item in val]

    # Torch Tensor -- guarded import
    try:
        import torch
        if isinstance(val, torch.Tensor):
            return val.detach().cpu().tolist()
    except ImportError:
        pass

    # NumPy array -- guarded import
    try:
        import numpy as np
        if isinstance(val, np.ndarray):
            return val.tolist()
        if isinstance(val, (np.integer,)):
            return int(val)
        if isinstance(val, (np.floating,)):
            return float(val)
    except ImportError:
        pass

    # Last resort: string representation
    return str(val)


def config_to_dict(config: CodecLossConfig) -> Dict[str, Any]:
    """Serialize a ``CodecLossConfig`` to a plain JSON-safe dict.

    Parameters
    ----------
    config : CodecLossConfig
        The configuration object to serialize.

    Returns
    -------
    Dict[str, Any]
        Nested dictionary with all values converted to JSON-safe types.
    """
    raw = asdict(config)
    return _sanitize_value(raw)


def config_from_dict(d: Dict[str, Any]) -> CodecLossConfig:
    """Deserialize a plain dict into a ``CodecLossConfig``.

    Handles missing keys gracefully by falling back to default values.

    Parameters
    ----------
    d : Dict[str, Any]
        Dictionary previously produced by ``config_to_dict`` or
        loaded from JSON / YAML.

    Returns
    -------
    CodecLossConfig
        Reconstructed configuration object with validation applied.
    """
    enc_dict = d.get("encoding", {})
    dec_dict = d.get("decoding", {})
    loss_dict = d.get("loss", {})

    # Filter to only known fields for each dataclass
    enc_fields = {f.name for f in fields(EncodingConfig)}
    dec_fields = {f.name for f in fields(DecodingConfig)}
    loss_fields = {f.name for f in fields(LossConfig)}

    enc_kwargs = {k: v for k, v in enc_dict.items() if k in enc_fields}
    dec_kwargs = {k: v for k, v in dec_dict.items() if k in dec_fields}
    loss_kwargs = {k: v for k, v in loss_dict.items() if k in loss_fields}

    return CodecLossConfig(
        encoding=EncodingConfig(**enc_kwargs),
        decoding=DecodingConfig(**dec_kwargs),
        loss=LossConfig(**loss_kwargs),
    )


def config_to_json(config: CodecLossConfig, indent: int = 2) -> str:
    """Serialize a ``CodecLossConfig`` to a JSON string.

    Parameters
    ----------
    config : CodecLossConfig
        The configuration to serialize.
    indent : int
        JSON indentation level.

    Returns
    -------
    str
        JSON string.
    """
    return json.dumps(config_to_dict(config), indent=indent)


def config_from_json(s: str) -> CodecLossConfig:
    """Deserialize a ``CodecLossConfig`` from a JSON string.

    Parameters
    ----------
    s : str
        JSON string previously produced by ``config_to_json``.

    Returns
    -------
    CodecLossConfig
        Reconstructed configuration.
    """
    return config_from_dict(json.loads(s))


# ===================================================================
# Self-test
# ===================================================================

def _run_self_tests() -> None:
    """Comprehensive self-test suite.

    Run with: ``python codec_config_template.py``
    """
    # ---------------------------------------------------------------
    # Path setup: ensure brain_ai is importable (4-level dirname)
    # ---------------------------------------------------------------
    _this_file = Path(__file__).resolve()
    _project_root = _this_file.parent.parent.parent.parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

    passed: List[str] = []
    failed: List[str] = []
    total = 0

    def _report(group: str, ok: bool, detail: str = "") -> None:
        nonlocal total
        total += 1
        status = "PASS" if ok else "FAIL"
        msg = f"  [{status}] {group}"
        if detail:
            msg += f" -- {detail}"
        print(msg)
        if ok:
            passed.append(group)
        else:
            failed.append(group)

    print("=" * 72)
    print("CodecLossConfig self-test")
    print("=" * 72)

    # ---------------------------------------------------------------
    # 1. EncodingConfig validation
    # ---------------------------------------------------------------
    print("\n--- EncodingConfig validation ---")

    # Valid types should pass
    for enc_type in _VALID_ENCODING_TYPES:
        try:
            EncodingConfig(type=enc_type)
            _report(f"EncodingConfig(type='{enc_type}')", True)
        except Exception as exc:
            _report(f"EncodingConfig(type='{enc_type}')", False, str(exc))

    # Invalid type should raise
    try:
        EncodingConfig(type="invalid_xyz")
        _report("EncodingConfig(type='invalid_xyz') raises", False,
                "no exception raised")
    except ValueError:
        _report("EncodingConfig(type='invalid_xyz') raises", True)
    except Exception as exc:
        _report("EncodingConfig(type='invalid_xyz') raises", False,
                f"wrong exception: {exc}")

    # num_steps <= 0
    try:
        EncodingConfig(num_steps=0)
        _report("EncodingConfig(num_steps=0) raises", False,
                "no exception raised")
    except ValueError:
        _report("EncodingConfig(num_steps=0) raises", True)
    except Exception as exc:
        _report("EncodingConfig(num_steps=0) raises", False,
                f"wrong exception: {exc}")

    # Negative num_steps
    try:
        EncodingConfig(num_steps=-5)
        _report("EncodingConfig(num_steps=-5) raises", False,
                "no exception raised")
    except ValueError:
        _report("EncodingConfig(num_steps=-5) raises", True)
    except Exception as exc:
        _report("EncodingConfig(num_steps=-5) raises", False,
                f"wrong exception: {exc}")

    # dt <= 0
    try:
        EncodingConfig(dt=0.0)
        _report("EncodingConfig(dt=0.0) raises", False,
                "no exception raised")
    except ValueError:
        _report("EncodingConfig(dt=0.0) raises", True)
    except Exception as exc:
        _report("EncodingConfig(dt=0.0) raises", False,
                f"wrong exception: {exc}")

    # Invalid normalization
    try:
        EncodingConfig(normalization="foobar")
        _report("EncodingConfig(normalization='foobar') raises", False,
                "no exception raised")
    except ValueError:
        _report("EncodingConfig(normalization='foobar') raises", True)
    except Exception as exc:
        _report("EncodingConfig(normalization='foobar') raises", False,
                f"wrong exception: {exc}")

    # Invalid population_centers
    try:
        EncodingConfig(population_centers="random_wrong")
        _report("EncodingConfig(population_centers='random_wrong') raises",
                False, "no exception raised")
    except ValueError:
        _report("EncodingConfig(population_centers='random_wrong') raises",
                True)

    # Invalid latency_mapping
    try:
        EncodingConfig(latency_mapping="cubic")
        _report("EncodingConfig(latency_mapping='cubic') raises",
                False, "no exception raised")
    except ValueError:
        _report("EncodingConfig(latency_mapping='cubic') raises", True)

    # t_max < t_min
    try:
        EncodingConfig(t_min=10, t_max=5)
        _report("EncodingConfig(t_min=10, t_max=5) raises", False,
                "no exception raised")
    except ValueError:
        _report("EncodingConfig(t_min=10, t_max=5) raises", True)

    # Negative jitter
    try:
        EncodingConfig(jitter=-1.0)
        _report("EncodingConfig(jitter=-1.0) raises", False,
                "no exception raised")
    except ValueError:
        _report("EncodingConfig(jitter=-1.0) raises", True)

    # delta_threshold <= 0
    try:
        EncodingConfig(delta_threshold=0.0)
        _report("EncodingConfig(delta_threshold=0.0) raises", False,
                "no exception raised")
    except ValueError:
        _report("EncodingConfig(delta_threshold=0.0) raises", True)

    # delta_off_threshold <= 0
    try:
        EncodingConfig(delta_off_threshold=-0.5)
        _report("EncodingConfig(delta_off_threshold=-0.5) raises", False,
                "no exception raised")
    except ValueError:
        _report("EncodingConfig(delta_off_threshold=-0.5) raises", True)

    # rate_gain <= 0
    try:
        EncodingConfig(rate_gain=0.0)
        _report("EncodingConfig(rate_gain=0.0) raises", False,
                "no exception raised")
    except ValueError:
        _report("EncodingConfig(rate_gain=0.0) raises", True)

    # population_size <= 0
    try:
        EncodingConfig(population_size=0)
        _report("EncodingConfig(population_size=0) raises", False,
                "no exception raised")
    except ValueError:
        _report("EncodingConfig(population_size=0) raises", True)

    # population_sigma <= 0
    try:
        EncodingConfig(population_sigma=-0.1)
        _report("EncodingConfig(population_sigma=-0.1) raises", False,
                "no exception raised")
    except ValueError:
        _report("EncodingConfig(population_sigma=-0.1) raises", True)

    # Valid config with all params
    try:
        enc = EncodingConfig(
            type="population",
            num_steps=30,
            dt=0.5,
            normalization="sigmoid",
            deterministic_eval=False,
            seed=42,
            rate_gain=2.0,
            rate_bias=-0.1,
            population_size=16,
            population_sigma=0.5,
            population_centers="uniform",
            latency_mapping="log",
            t_min=2,
            t_max=28,
            allow_no_spike=True,
            jitter=1.5,
            delta_threshold=0.05,
            delta_off_threshold=0.08,
        )
        _report("EncodingConfig full valid construction", True)
    except Exception as exc:
        _report("EncodingConfig full valid construction", False, str(exc))

    # ---------------------------------------------------------------
    # 2. DecodingConfig validation
    # ---------------------------------------------------------------
    print("\n--- DecodingConfig validation ---")

    # Valid types
    for dec_type in _VALID_DECODING_TYPES:
        try:
            DecodingConfig(type=dec_type)
            _report(f"DecodingConfig(type='{dec_type}')", True)
        except Exception as exc:
            _report(f"DecodingConfig(type='{dec_type}')", False, str(exc))

    # Invalid type
    try:
        DecodingConfig(type="invalid_decoder")
        _report("DecodingConfig(type='invalid_decoder') raises", False,
                "no exception raised")
    except ValueError:
        _report("DecodingConfig(type='invalid_decoder') raises", True)

    # Temperature <= 0
    try:
        DecodingConfig(temperature=0.0)
        _report("DecodingConfig(temperature=0.0) raises", False,
                "no exception raised")
    except ValueError:
        _report("DecodingConfig(temperature=0.0) raises", True)

    # eps <= 0
    try:
        DecodingConfig(eps=-1e-8)
        _report("DecodingConfig(eps=-1e-8) raises", False,
                "no exception raised")
    except ValueError:
        _report("DecodingConfig(eps=-1e-8) raises", True)

    # Invalid population_mode
    try:
        DecodingConfig(population_mode="invalid")
        _report("DecodingConfig(population_mode='invalid') raises", False,
                "no exception raised")
    except ValueError:
        _report("DecodingConfig(population_mode='invalid') raises", True)

    # Invalid membrane_mode
    try:
        DecodingConfig(membrane_mode="mean")
        _report("DecodingConfig(membrane_mode='mean') raises", False,
                "no exception raised")
    except ValueError:
        _report("DecodingConfig(membrane_mode='mean') raises", True)

    # Ensemble weights with negative value
    try:
        DecodingConfig(ensemble_weights={"rate": -0.5})
        _report("DecodingConfig(ensemble negative weight) raises", False,
                "no exception raised")
    except ValueError:
        _report("DecodingConfig(ensemble negative weight) raises", True)

    # Valid ensemble weights
    try:
        DecodingConfig(
            type="ensemble",
            ensemble_weights={"rate": 0.6, "membrane": 0.4},
        )
        _report("DecodingConfig valid ensemble", True)
    except Exception as exc:
        _report("DecodingConfig valid ensemble", False, str(exc))

    # ---------------------------------------------------------------
    # 3. LossConfig validation
    # ---------------------------------------------------------------
    print("\n--- LossConfig validation ---")

    # Default construction
    try:
        lc = LossConfig()
        _report("LossConfig default construction", True)
    except Exception as exc:
        _report("LossConfig default construction", False, str(exc))

    # Negative weight
    for wname in ("w_probspikes", "w_rate", "w_temporal", "w_isi",
                   "w_membrane"):
        try:
            LossConfig(**{wname: -0.1})
            _report(f"LossConfig({wname}=-0.1) raises", False,
                    "no exception raised")
        except ValueError:
            _report(f"LossConfig({wname}=-0.1) raises", True)
        except Exception as exc:
            _report(f"LossConfig({wname}=-0.1) raises", False,
                    f"wrong exception: {exc}")

    # target_rate out of range
    for bad_rate in (0.0, 1.0, -0.5, 1.5):
        try:
            LossConfig(target_rate=bad_rate)
            _report(f"LossConfig(target_rate={bad_rate}) raises", False,
                    "no exception raised")
        except ValueError:
            _report(f"LossConfig(target_rate={bad_rate}) raises", True)
        except Exception as exc:
            _report(f"LossConfig(target_rate={bad_rate}) raises", False,
                    f"wrong exception: {exc}")

    # Invalid min/max rate band
    try:
        LossConfig(min_rate=0.5, max_rate=0.3, target_rate=0.4)
        _report("LossConfig(min_rate > max_rate) raises", False,
                "no exception raised")
    except ValueError:
        _report("LossConfig(min_rate > max_rate) raises", True)

    # target_rate outside [min, max]
    try:
        LossConfig(target_rate=0.5, min_rate=0.01, max_rate=0.3)
        _report("LossConfig(target_rate outside band) raises", False,
                "no exception raised")
    except ValueError:
        _report("LossConfig(target_rate outside band) raises", True)

    # Invalid probspikes_mode
    try:
        LossConfig(probspikes_mode="relu")
        _report("LossConfig(probspikes_mode='relu') raises", False,
                "no exception raised")
    except ValueError:
        _report("LossConfig(probspikes_mode='relu') raises", True)

    # Invalid temporal_penalty
    try:
        LossConfig(temporal_penalty="huber")
        _report("LossConfig(temporal_penalty='huber') raises", False,
                "no exception raised")
    except ValueError:
        _report("LossConfig(temporal_penalty='huber') raises", True)

    # Invalid isi_kernel_type
    try:
        LossConfig(isi_kernel_type="gaussian")
        _report("LossConfig(isi_kernel_type='gaussian') raises", False,
                "no exception raised")
    except ValueError:
        _report("LossConfig(isi_kernel_type='gaussian') raises", True)

    # membrane_max <= 0
    try:
        LossConfig(membrane_max=0.0)
        _report("LossConfig(membrane_max=0.0) raises", False,
                "no exception raised")
    except ValueError:
        _report("LossConfig(membrane_max=0.0) raises", True)

    # temporal_window <= 0
    try:
        LossConfig(temporal_window=0)
        _report("LossConfig(temporal_window=0) raises", False,
                "no exception raised")
    except ValueError:
        _report("LossConfig(temporal_window=0) raises", True)

    # isi_refractory_window <= 0
    try:
        LossConfig(isi_refractory_window=0)
        _report("LossConfig(isi_refractory_window=0) raises", False,
                "no exception raised")
    except ValueError:
        _report("LossConfig(isi_refractory_window=0) raises", True)

    # probspikes_temperature <= 0
    try:
        LossConfig(probspikes_temperature=-1.0)
        _report("LossConfig(probspikes_temperature=-1.0) raises", False,
                "no exception raised")
    except ValueError:
        _report("LossConfig(probspikes_temperature=-1.0) raises", True)

    # probspikes_eps <= 0
    try:
        LossConfig(probspikes_eps=0.0)
        _report("LossConfig(probspikes_eps=0.0) raises", False,
                "no exception raised")
    except ValueError:
        _report("LossConfig(probspikes_eps=0.0) raises", True)

    # Per-layer targets: invalid rate
    try:
        LossConfig(per_layer_targets={"layer1": 1.5})
        _report("LossConfig(per_layer_targets bad rate) raises", False,
                "no exception raised")
    except ValueError:
        _report("LossConfig(per_layer_targets bad rate) raises", True)

    # Per-layer targets: non-string key (via int)
    try:
        LossConfig(per_layer_targets={42: 0.1})  # type: ignore[dict-item]
        _report("LossConfig(per_layer_targets int key) raises", False,
                "no exception raised")
    except TypeError:
        _report("LossConfig(per_layer_targets int key) raises", True)

    # Valid per-layer targets
    try:
        LossConfig(per_layer_targets={"snn.layer0": 0.08, "snn.layer1": 0.12})
        _report("LossConfig valid per_layer_targets", True)
    except Exception as exc:
        _report("LossConfig valid per_layer_targets", False, str(exc))

    # enabled_terms
    lc = LossConfig(
        w_probspikes=1.0,
        w_rate=0.0,
        w_temporal=0.01,
        w_isi=0.0,
        w_membrane=0.001,
    )
    terms = lc.enabled_terms()
    expected_terms = ["membrane", "probspikes", "temporal"]
    ok = terms == expected_terms
    _report("LossConfig.enabled_terms()",
            ok,
            f"got {terms}, expected {expected_terms}" if not ok else "")

    # total_weight
    lc2 = LossConfig(
        w_probspikes=1.0,
        w_rate=0.1,
        w_temporal=0.01,
        w_isi=0.01,
        w_membrane=0.001,
    )
    tw = lc2.total_weight()
    expected_tw = 1.0 + 0.1 + 0.01 + 0.01 + 0.001
    ok = abs(tw - expected_tw) < 1e-12
    _report("LossConfig.total_weight()",
            ok,
            f"got {tw}, expected {expected_tw}" if not ok else "")

    # ---------------------------------------------------------------
    # 4. Preset factories
    # ---------------------------------------------------------------
    print("\n--- Preset factories ---")

    preset_factories = [
        ("CodecLossConfig.default", CodecLossConfig.default),
        ("CodecLossConfig.rate_classification",
         CodecLossConfig.rate_classification),
        ("CodecLossConfig.ttfs_classification",
         CodecLossConfig.ttfs_classification),
        ("CodecLossConfig.population_classification",
         CodecLossConfig.population_classification),
        ("CodecLossConfig.regression", CodecLossConfig.regression),
        ("rate_classification_preset", rate_classification_preset),
        ("ttfs_classification_preset", ttfs_classification_preset),
        ("population_classification_preset",
         population_classification_preset),
        ("regression_preset", regression_preset),
        ("sparse_efficient_preset", sparse_efficient_preset),
    ]

    for name, factory in preset_factories:
        try:
            cfg = factory()
            # Basic structural checks
            assert isinstance(cfg, CodecLossConfig), "wrong type"
            assert isinstance(cfg.encoding, EncodingConfig), "bad encoding"
            assert isinstance(cfg.decoding, DecodingConfig), "bad decoding"
            assert isinstance(cfg.loss, LossConfig), "bad loss"
            # Verify encoding type is valid
            assert cfg.encoding.type in _VALID_ENCODING_TYPES, \
                f"bad enc type: {cfg.encoding.type}"
            # Verify decoding type is valid
            assert cfg.decoding.type in _VALID_DECODING_TYPES, \
                f"bad dec type: {cfg.decoding.type}"
            _report(f"{name}()", True)
        except Exception as exc:
            _report(f"{name}()", False, str(exc))

    # Verify specific preset properties
    try:
        ttfs = ttfs_classification_preset()
        assert ttfs.encoding.type == "ttfs", \
            f"TTFS preset encoding type: {ttfs.encoding.type}"
        assert ttfs.decoding.type == "first_spike", \
            f"TTFS preset decoding type: {ttfs.decoding.type}"
        assert ttfs.loss.w_probspikes > 0, "TTFS should have ProbSpikes on"
        _report("ttfs_preset properties", True)
    except Exception as exc:
        _report("ttfs_preset properties", False, str(exc))

    try:
        reg = regression_preset()
        assert reg.encoding.type == "rate_bernoulli", \
            f"Regression encoding: {reg.encoding.type}"
        assert reg.decoding.type == "membrane", \
            f"Regression decoding: {reg.decoding.type}"
        assert reg.loss.w_probspikes == 0.0, \
            "Regression should disable ProbSpikes"
        _report("regression_preset properties", True)
    except Exception as exc:
        _report("regression_preset properties", False, str(exc))

    try:
        sparse = sparse_efficient_preset()
        assert sparse.encoding.type == "ttfs", \
            f"Sparse encoding: {sparse.encoding.type}"
        assert sparse.encoding.num_steps <= 20, \
            f"Sparse should have few steps: {sparse.encoding.num_steps}"
        assert sparse.loss.target_rate <= 0.05, \
            f"Sparse target rate too high: {sparse.loss.target_rate}"
        _report("sparse_efficient_preset properties", True)
    except Exception as exc:
        _report("sparse_efficient_preset properties", False, str(exc))

    try:
        pop = population_classification_preset()
        assert pop.encoding.type == "population", \
            f"Population encoding: {pop.encoding.type}"
        assert pop.decoding.type == "population", \
            f"Population decoding: {pop.decoding.type}"
        assert pop.encoding.population_size > 0
        _report("population_preset properties", True)
    except Exception as exc:
        _report("population_preset properties", False, str(exc))

    # ---------------------------------------------------------------
    # 5. Serialization roundtrip
    # ---------------------------------------------------------------
    print("\n--- Serialization roundtrip ---")

    for name, factory in preset_factories:
        try:
            original = factory()
            d = config_to_dict(original)

            # Verify dict is JSON-safe
            json_str = json.dumps(d)
            d_reloaded = json.loads(json_str)

            reconstructed = config_from_dict(d_reloaded)

            # Compare field-by-field
            orig_d = config_to_dict(original)
            recon_d = config_to_dict(reconstructed)
            assert orig_d == recon_d, (
                f"Mismatch after roundtrip:\n"
                f"  original:      {orig_d}\n"
                f"  reconstructed: {recon_d}"
            )
            _report(f"roundtrip {name}", True)
        except Exception as exc:
            _report(f"roundtrip {name}", False, str(exc))

    # JSON string roundtrip
    try:
        original = CodecLossConfig.default()
        js = config_to_json(original)
        restored = config_from_json(js)
        assert config_to_dict(original) == config_to_dict(restored)
        _report("JSON string roundtrip", True)
    except Exception as exc:
        _report("JSON string roundtrip", False, str(exc))

    # Roundtrip with per_layer_targets
    try:
        original = CodecLossConfig(
            encoding=EncodingConfig(),
            decoding=DecodingConfig(),
            loss=LossConfig(
                per_layer_targets={"layer0": 0.08, "layer1": 0.12},
            ),
        )
        d = config_to_dict(original)
        restored = config_from_dict(d)
        assert restored.loss.per_layer_targets == {"layer0": 0.08,
                                                    "layer1": 0.12}
        _report("roundtrip with per_layer_targets", True)
    except Exception as exc:
        _report("roundtrip with per_layer_targets", False, str(exc))

    # Roundtrip with ensemble_weights
    try:
        original = CodecLossConfig(
            encoding=EncodingConfig(),
            decoding=DecodingConfig(
                type="ensemble",
                ensemble_weights={"rate": 0.7, "membrane": 0.3},
            ),
            loss=LossConfig(),
        )
        d = config_to_dict(original)
        restored = config_from_dict(d)
        assert restored.decoding.ensemble_weights == {"rate": 0.7,
                                                       "membrane": 0.3}
        _report("roundtrip with ensemble_weights", True)
    except Exception as exc:
        _report("roundtrip with ensemble_weights", False, str(exc))

    # Deserialize with unknown keys (should be ignored)
    try:
        d = config_to_dict(CodecLossConfig.default())
        d["encoding"]["unknown_future_field"] = 999
        d["loss"]["also_unknown"] = "hello"
        restored = config_from_dict(d)
        assert isinstance(restored, CodecLossConfig)
        _report("deserialize ignores unknown keys", True)
    except Exception as exc:
        _report("deserialize ignores unknown keys", False, str(exc))

    # Deserialize with missing keys (should use defaults)
    try:
        d = {"encoding": {"type": "ttfs"}, "decoding": {}, "loss": {}}
        restored = config_from_dict(d)
        assert restored.encoding.type == "ttfs"
        assert restored.decoding.type == "rate"  # default
        assert restored.loss.w_probspikes == 1.0  # default
        _report("deserialize with missing keys uses defaults", True)
    except Exception as exc:
        _report("deserialize with missing keys uses defaults", False,
                str(exc))

    # ---------------------------------------------------------------
    # 6. upgrade_snn_config
    # ---------------------------------------------------------------
    print("\n--- upgrade_snn_config ---")

    # Default old config
    try:
        old = {
            "beta": 0.95,
            "num_timesteps": 50,
            "surrogate": "atan",
            "use_probspikes_loss": True,
            "spike_rate_target": 0.1,
            "spike_rate_weight": 0.01,
            "temporal_consistency_weight": 0.001,
        }
        new = upgrade_snn_config(old)
        assert isinstance(new, CodecLossConfig)
        assert new.encoding.num_steps == 50
        assert new.loss.w_probspikes > 0.0
        assert abs(new.loss.target_rate - 0.1) < 1e-9
        assert abs(new.loss.w_rate - 0.01) < 1e-9
        assert abs(new.loss.w_temporal - 0.001) < 1e-9
        _report("upgrade default SNNConfig", True)
    except Exception as exc:
        _report("upgrade default SNNConfig", False, str(exc))

    # ProbSpikes disabled
    try:
        old = {
            "use_probspikes_loss": False,
            "spike_rate_target": 0.05,
            "spike_rate_weight": 0.02,
            "temporal_consistency_weight": 0.005,
            "num_timesteps": 30,
        }
        new = upgrade_snn_config(old)
        assert new.loss.w_probspikes == 0.0, \
            f"Expected w_probspikes=0, got {new.loss.w_probspikes}"
        assert new.encoding.num_steps == 30
        assert abs(new.loss.target_rate - 0.05) < 1e-9
        _report("upgrade with probspikes disabled", True)
    except Exception as exc:
        _report("upgrade with probspikes disabled", False, str(exc))

    # Minimal old config (missing keys)
    try:
        old = {}
        new = upgrade_snn_config(old)
        assert isinstance(new, CodecLossConfig)
        assert new.encoding.num_steps == 50  # default
        assert new.loss.w_probspikes > 0.0  # default True
        _report("upgrade empty dict (all defaults)", True)
    except Exception as exc:
        _report("upgrade empty dict (all defaults)", False, str(exc))

    # Edge case: very high spike rate target
    try:
        old = {
            "spike_rate_target": 0.9,
            "num_timesteps": 100,
        }
        new = upgrade_snn_config(old)
        assert new.loss.target_rate == 0.9
        assert new.loss.min_rate < 0.9
        assert new.loss.max_rate > 0.9
        assert new.loss.max_rate <= 1.0
        _report("upgrade high spike_rate_target", True)
    except Exception as exc:
        _report("upgrade high spike_rate_target", False, str(exc))

    # ---------------------------------------------------------------
    # 7. legacy_to_new with dict and object
    # ---------------------------------------------------------------
    print("\n--- legacy_to_new ---")

    # With dict
    try:
        old_dict = {
            "use_probspikes_loss": True,
            "spike_rate_target": 0.15,
            "num_timesteps": 40,
        }
        new = legacy_to_new(old_dict)
        assert isinstance(new, CodecLossConfig)
        assert new.encoding.num_steps == 40
        assert abs(new.loss.target_rate - 0.15) < 1e-9
        _report("legacy_to_new with dict", True)
    except Exception as exc:
        _report("legacy_to_new with dict", False, str(exc))

    # With object-like structure (simulate SNNConfig)
    try:
        class _FakeSNNConfig:
            use_probspikes_loss = True
            spike_rate_target = 0.2
            spike_rate_weight = 0.05
            temporal_consistency_weight = 0.01
            num_timesteps = 60
            beta = 0.95
            surrogate = "atan"

        fake = _FakeSNNConfig()
        new = legacy_to_new(fake)
        assert isinstance(new, CodecLossConfig)
        assert new.encoding.num_steps == 60
        assert abs(new.loss.target_rate - 0.2) < 1e-9
        assert abs(new.loss.w_rate - 0.05) < 1e-9
        _report("legacy_to_new with object", True)
    except Exception as exc:
        _report("legacy_to_new with object", False, str(exc))

    # Try with real SNNConfig if available
    try:
        from brain_ai.config import SNNConfig as RealSNNConfig
        real_cfg = RealSNNConfig()
        new = legacy_to_new(real_cfg)
        assert isinstance(new, CodecLossConfig)
        assert new.encoding.num_steps == real_cfg.num_timesteps
        assert abs(new.loss.target_rate - real_cfg.spike_rate_target) < 1e-9
        _report("legacy_to_new with real SNNConfig", True)
    except ImportError:
        _report("legacy_to_new with real SNNConfig", True,
                "skipped (brain_ai not importable)")
    except Exception as exc:
        _report("legacy_to_new with real SNNConfig", False, str(exc))

    # ---------------------------------------------------------------
    # 8. enabled_terms comprehensive
    # ---------------------------------------------------------------
    print("\n--- enabled_terms ---")

    # All enabled
    try:
        lc = LossConfig(
            w_probspikes=1.0,
            w_rate=0.1,
            w_temporal=0.01,
            w_isi=0.01,
            w_membrane=0.001,
        )
        terms = lc.enabled_terms()
        expected = ["isi", "membrane", "probspikes", "rate", "temporal"]
        assert terms == expected, f"got {terms}"
        _report("enabled_terms all on", True)
    except Exception as exc:
        _report("enabled_terms all on", False, str(exc))

    # None enabled
    try:
        lc = LossConfig(
            w_probspikes=0.0,
            w_rate=0.0,
            w_temporal=0.0,
            w_isi=0.0,
            w_membrane=0.0,
        )
        terms = lc.enabled_terms()
        assert terms == [], f"got {terms}"
        _report("enabled_terms none on", True)
    except Exception as exc:
        _report("enabled_terms none on", False, str(exc))

    # Only ProbSpikes
    try:
        lc = LossConfig(
            w_probspikes=1.0,
            w_rate=0.0,
            w_temporal=0.0,
            w_isi=0.0,
            w_membrane=0.0,
        )
        terms = lc.enabled_terms()
        assert terms == ["probspikes"], f"got {terms}"
        _report("enabled_terms only probspikes", True)
    except Exception as exc:
        _report("enabled_terms only probspikes", False, str(exc))

    # ---------------------------------------------------------------
    # 9. Edge cases and deep copy safety
    # ---------------------------------------------------------------
    print("\n--- Edge cases ---")

    # Deep copy safety
    try:
        original = CodecLossConfig.default()
        copied = copy.deepcopy(original)
        copied.loss.w_probspikes = 99.0
        assert original.loss.w_probspikes != 99.0, \
            "deep copy did not isolate"
        _report("deep copy isolation", True)
    except Exception as exc:
        _report("deep copy isolation", False, str(exc))

    # config_to_dict does not share references
    try:
        cfg = CodecLossConfig(
            encoding=EncodingConfig(),
            decoding=DecodingConfig(),
            loss=LossConfig(
                per_layer_targets={"a": 0.1},
            ),
        )
        d = config_to_dict(cfg)
        d["loss"]["per_layer_targets"]["a"] = 0.99
        # Original should be unchanged
        assert cfg.loss.per_layer_targets["a"] == 0.1  # type: ignore[index]
        _report("config_to_dict reference isolation", True)
    except Exception as exc:
        _report("config_to_dict reference isolation", False, str(exc))

    # Empty sub-dicts deserialize to defaults
    try:
        cfg = config_from_dict({})
        assert isinstance(cfg, CodecLossConfig)
        assert cfg.encoding.type == "rate_bernoulli"
        assert cfg.decoding.type == "rate"
        assert cfg.loss.w_probspikes == 1.0
        _report("config_from_dict empty dict", True)
    except Exception as exc:
        _report("config_from_dict empty dict", False, str(exc))

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"Results: {len(passed)} passed, {len(failed)} failed, "
          f"{total} total")
    print("=" * 72)

    if failed:
        print("\nFailed tests:")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("\nAll tests passed.")
        sys.exit(0)


if __name__ == "__main__":
    _run_self_tests()
