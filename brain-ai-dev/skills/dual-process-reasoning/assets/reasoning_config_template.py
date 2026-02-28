"""
Configuration dataclasses for the dual-process reasoning system.

Provide complete configuration for System 1 (fast, parallel), System 2
(slow, iterative), metacognitive routing, confidence calibration, reasoning
trace logging, and training hyperparameters.  All presets map to the
brain-ai cognitive pipeline described in the project CLAUDE.md.

Pure Python -- no torch or framework dependency.  Optional pyyaml for YAML
serialization.

Typical usage
-------------
>>> cfg = DualProcessFullConfig.dev()
>>> cfg.validate()
[]
>>> cfg.estimate_params()
{'system1': ..., 'system2': ..., 'metacognition': ..., 'total': ...}
>>> cfg.to_json()
'{ ... }'
"""

from __future__ import annotations

import copy
import json
import math
import sys
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

# ---------------------------------------------------------------------------
# Optional YAML support
# ---------------------------------------------------------------------------
try:
    import yaml  # type: ignore[import-untyped]

    _HAS_YAML = True
except ImportError:  # pragma: no cover
    _HAS_YAML = False

# ---------------------------------------------------------------------------
# Sentinel used only inside helpers
# ---------------------------------------------------------------------------
_MISSING = object()

T = TypeVar("T")

# ===================================================================
# Valid enum-like constants
# ===================================================================

VALID_POOL_MODES: Tuple[str, ...] = ("mean", "attention", "cls")
VALID_ACTIVATIONS: Tuple[str, ...] = ("gelu", "relu", "silu")
VALID_CONVERGENCE_CRITERIA: Tuple[str, ...] = ("kl", "logit", "argmax")
VALID_NOVELTY_METHODS: Tuple[str, ...] = (
    "prototype",
    "htm_anomaly",
    "engram_miss",
    "none",
)
VALID_CALIBRATION_METHODS: Tuple[str, ...] = ("temperature", "isotonic", "none")


# ===================================================================
# Helper utilities
# ===================================================================


def _positive_int(name: str, value: int) -> Optional[str]:
    """Return an error string if *value* is not a positive integer."""
    if not isinstance(value, int) or value <= 0:
        return f"{name} must be a positive integer, got {value!r}"
    return None


def _nonneg_int(name: str, value: int) -> Optional[str]:
    """Return an error string if *value* is negative."""
    if not isinstance(value, int) or value < 0:
        return f"{name} must be a non-negative integer, got {value!r}"
    return None


def _positive_float(name: str, value: float) -> Optional[str]:
    """Return an error string if *value* is not positive."""
    if not isinstance(value, (int, float)) or value <= 0:
        return f"{name} must be a positive float, got {value!r}"
    return None


def _nonneg_float(name: str, value: float) -> Optional[str]:
    """Return an error string if *value* is negative."""
    if not isinstance(value, (int, float)) or value < 0:
        return f"{name} must be a non-negative float, got {value!r}"
    return None


def _in_set(name: str, value: str, valid: Tuple[str, ...]) -> Optional[str]:
    """Return an error string if *value* is not in *valid*."""
    if value not in valid:
        return f"{name} must be one of {valid}, got {value!r}"
    return None


def _prob_range(name: str, value: float) -> Optional[str]:
    """Return an error string if *value* is not in [0, 1]."""
    if not isinstance(value, (int, float)) or not (0.0 <= value <= 1.0):
        return f"{name} must be in [0, 1], got {value!r}"
    return None


# ===================================================================
# 1. System1Config
# ===================================================================


@dataclass
class System1Config:
    """System 1 fast predictor configuration.

    System 1 operates as a feed-forward pathway that produces a fast answer
    together with a scalar confidence score.  Its confidence drives the
    metacognitive router: when high enough the slow System 2 path is skipped.
    """

    # --- dimensions ---
    input_dim: int = 4096
    hidden_dim: int = 512
    output_dim: int = 256
    num_layers: int = 2

    # --- pooling ---
    pool_mode: str = "mean"  # "mean", "attention", "cls"
    num_slots: Optional[int] = None  # required when pool_mode == "attention"

    # --- head ---
    confidence_head: bool = True

    # --- activation / regularisation ---
    activation: str = "gelu"  # "gelu", "relu", "silu"
    dropout: float = 0.1

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        """Validate System1Config fields. Return list of errors/warnings."""
        errors: List[str] = []

        msg = _positive_int("System1Config.input_dim", self.input_dim)
        if msg:
            errors.append(msg)

        msg = _positive_int("System1Config.hidden_dim", self.hidden_dim)
        if msg:
            errors.append(msg)

        msg = _positive_int("System1Config.output_dim", self.output_dim)
        if msg:
            errors.append(msg)

        msg = _positive_int("System1Config.num_layers", self.num_layers)
        if msg:
            errors.append(msg)

        msg = _in_set("System1Config.pool_mode", self.pool_mode, VALID_POOL_MODES)
        if msg:
            errors.append(msg)

        msg = _in_set("System1Config.activation", self.activation, VALID_ACTIVATIONS)
        if msg:
            errors.append(msg)

        msg = _nonneg_float("System1Config.dropout", self.dropout)
        if msg:
            errors.append(msg)
        elif self.dropout > 0.9:
            errors.append(
                f"System1Config.dropout={self.dropout} is very high; "
                "consider a value <= 0.5"
            )

        if self.pool_mode == "attention" and (
            self.num_slots is None or self.num_slots <= 0
        ):
            errors.append(
                "System1Config.num_slots must be a positive integer when "
                f"pool_mode='attention', got {self.num_slots!r}"
            )

        return errors

    # ------------------------------------------------------------------
    # Parameter estimation
    # ------------------------------------------------------------------

    def estimate_params(self) -> int:
        """Estimate learnable parameter count for System 1.

        Architecture assumed:
          - Linear(input_dim, hidden_dim) + bias
          - (num_layers - 1) x Linear(hidden_dim, hidden_dim) + bias
          - Linear(hidden_dim, output_dim) + bias
          - Optional confidence head: Linear(hidden_dim, 1) + bias
          - Optional attention pool: Linear(hidden_dim, num_slots) + bias
        """
        total = 0

        # First projection
        total += self.input_dim * self.hidden_dim + self.hidden_dim

        # Intermediate layers
        for _ in range(max(0, self.num_layers - 1)):
            total += self.hidden_dim * self.hidden_dim + self.hidden_dim

        # Output projection
        total += self.hidden_dim * self.output_dim + self.output_dim

        # Confidence head
        if self.confidence_head:
            total += self.hidden_dim + 1  # Linear(hidden_dim, 1) + bias

        # Attention pooling
        if self.pool_mode == "attention" and self.num_slots is not None:
            total += self.hidden_dim * self.num_slots + self.num_slots

        return total


# ===================================================================
# 2. System2Config
# ===================================================================


@dataclass
class System2Config:
    """System 2 iterative refinement configuration.

    System 2 is a recurrent reasoning loop (GRU-based by default) that
    iteratively refines its hidden state until a convergence criterion is
    met or *max_steps* is reached.
    """

    # --- dimensions ---
    hidden_dim: int = 512
    output_dim: int = 256
    input_summary_dim: int = 512

    # --- iteration control ---
    max_steps: int = 10
    convergence_eps: float = 1e-3
    convergence_patience: int = 2
    convergence_criterion: str = "kl"  # "kl", "logit", "argmax"

    # --- numerical safety ---
    nan_guard: bool = True
    gradient_clip_per_step: float = 1.0

    # --- deep supervision ---
    deep_supervision: bool = False
    deep_supervision_discount: float = 0.9

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        """Validate System2Config fields. Return list of errors/warnings."""
        errors: List[str] = []

        msg = _positive_int("System2Config.hidden_dim", self.hidden_dim)
        if msg:
            errors.append(msg)

        msg = _positive_int("System2Config.output_dim", self.output_dim)
        if msg:
            errors.append(msg)

        msg = _positive_int("System2Config.input_summary_dim", self.input_summary_dim)
        if msg:
            errors.append(msg)

        msg = _positive_int("System2Config.max_steps", self.max_steps)
        if msg:
            errors.append(msg)

        msg = _positive_float("System2Config.convergence_eps", self.convergence_eps)
        if msg:
            errors.append(msg)

        msg = _positive_int(
            "System2Config.convergence_patience", self.convergence_patience
        )
        if msg:
            errors.append(msg)

        msg = _in_set(
            "System2Config.convergence_criterion",
            self.convergence_criterion,
            VALID_CONVERGENCE_CRITERIA,
        )
        if msg:
            errors.append(msg)

        msg = _positive_float(
            "System2Config.gradient_clip_per_step", self.gradient_clip_per_step
        )
        if msg:
            errors.append(msg)

        if self.deep_supervision:
            if not (0.0 < self.deep_supervision_discount <= 1.0):
                errors.append(
                    f"System2Config.deep_supervision_discount must be in (0, 1], "
                    f"got {self.deep_supervision_discount}"
                )

        if self.max_steps < self.convergence_patience:
            errors.append(
                f"System2Config.max_steps ({self.max_steps}) must be >= "
                f"convergence_patience ({self.convergence_patience})"
            )

        return errors

    # ------------------------------------------------------------------
    # Parameter estimation
    # ------------------------------------------------------------------

    def estimate_params(self) -> int:
        """Estimate learnable parameter count for System 2.

        Architecture assumed:
          - GRUCell(input_summary_dim, hidden_dim):
              3 * (input_summary_dim * hidden_dim + hidden_dim)  # input weights
            + 3 * (hidden_dim * hidden_dim + hidden_dim)         # hidden weights
          - Linear(hidden_dim, output_dim) + bias
          - Optional deep supervision head: same as output Linear
        """
        total = 0

        # GRU cell weights (standard 3-gate formulation)
        # input-to-hidden
        total += 3 * (self.input_summary_dim * self.hidden_dim + self.hidden_dim)
        # hidden-to-hidden
        total += 3 * (self.hidden_dim * self.hidden_dim + self.hidden_dim)

        # Output projection
        total += self.hidden_dim * self.output_dim + self.output_dim

        # Deep supervision re-uses the output head (no extra params by default)
        # but we model an auxiliary linear for each step for the discount loss
        if self.deep_supervision:
            # Auxiliary linear (shared across steps)
            total += self.hidden_dim * self.output_dim + self.output_dim

        return total


# ===================================================================
# 3. MetacognitionConfig
# ===================================================================


@dataclass
class MetacognitionConfig:
    """Metacognitive routing configuration.

    Decide whether to invoke System 2 based on a linear combination of
    confidence, novelty, anomaly, and remaining compute budget.
    """

    # --- routing threshold ---
    route_threshold: float = 0.5

    # --- score weights ---
    w_conf: float = 1.0
    w_novelty: float = 0.5
    w_anomaly: float = 0.3
    w_budget: float = 0.1

    # --- shortcut ---
    min_conf_to_skip_s2: float = 0.95

    # --- adaptive step budgeting ---
    base_steps: int = 3
    step_scale_alpha: float = 5.0
    max_steps: int = 10

    # --- override ---
    always_run_s2: bool = False

    # --- novelty detection ---
    novelty_method: str = "prototype"  # "prototype", "htm_anomaly", "engram_miss", "none"
    num_prototypes: int = 64
    prototype_dim: int = 512
    prototype_ema_decay: float = 0.99

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        """Validate MetacognitionConfig fields. Return list of errors/warnings."""
        errors: List[str] = []

        msg = _prob_range("MetacognitionConfig.route_threshold", self.route_threshold)
        if msg:
            errors.append(msg)

        msg = _prob_range(
            "MetacognitionConfig.min_conf_to_skip_s2", self.min_conf_to_skip_s2
        )
        if msg:
            errors.append(msg)

        for attr in ("w_conf", "w_novelty", "w_anomaly", "w_budget"):
            msg = _nonneg_float(f"MetacognitionConfig.{attr}", getattr(self, attr))
            if msg:
                errors.append(msg)

        msg = _positive_int("MetacognitionConfig.base_steps", self.base_steps)
        if msg:
            errors.append(msg)

        msg = _nonneg_float(
            "MetacognitionConfig.step_scale_alpha", self.step_scale_alpha
        )
        if msg:
            errors.append(msg)

        msg = _positive_int("MetacognitionConfig.max_steps", self.max_steps)
        if msg:
            errors.append(msg)

        msg = _in_set(
            "MetacognitionConfig.novelty_method",
            self.novelty_method,
            VALID_NOVELTY_METHODS,
        )
        if msg:
            errors.append(msg)

        msg = _positive_int(
            "MetacognitionConfig.num_prototypes", self.num_prototypes
        )
        if msg:
            errors.append(msg)

        msg = _positive_int(
            "MetacognitionConfig.prototype_dim", self.prototype_dim
        )
        if msg:
            errors.append(msg)

        if not (0.0 < self.prototype_ema_decay <= 1.0):
            errors.append(
                f"MetacognitionConfig.prototype_ema_decay must be in (0, 1], "
                f"got {self.prototype_ema_decay}"
            )

        # Budget feasibility check
        effective_max = self.step_scale_alpha * 1.0 + self.base_steps
        if effective_max > self.max_steps:
            errors.append(
                f"MetacognitionConfig budget overflow: step_scale_alpha * 1.0 + "
                f"base_steps = {effective_max:.1f} > max_steps = {self.max_steps}. "
                f"Consider increasing max_steps or decreasing step_scale_alpha."
            )

        return errors

    # ------------------------------------------------------------------
    # Parameter estimation
    # ------------------------------------------------------------------

    def estimate_params(self) -> int:
        """Estimate learnable parameter count for metacognition.

        Architecture assumed:
          - Prototype memory: num_prototypes x prototype_dim  (not a learned
            parameter per se, but an EMA buffer that is part of the state)
          - Route MLP: Linear(4, 16) + Linear(16, 1)
          - We count the prototypes as "parameters" because they occupy GPU
            memory and are updated during training.
        """
        total = 0

        # Prototype buffer
        if self.novelty_method == "prototype":
            total += self.num_prototypes * self.prototype_dim

        # Route MLP (small)
        route_hidden = 16
        # 4 input features: conf, novelty, anomaly, budget
        total += 4 * route_hidden + route_hidden
        total += route_hidden * 1 + 1

        return total


# ===================================================================
# 4. CalibrationConfig
# ===================================================================


@dataclass
class CalibrationConfig:
    """Confidence calibration configuration.

    Apply post-hoc calibration to the confidence scores emitted by
    System 1 so that p(correct | confidence = c) approximately equals c.
    """

    method: str = "temperature"  # "temperature", "isotonic", "none"

    # --- temperature scaling ---
    initial_temperature: float = 1.5
    fit_lr: float = 0.01
    fit_max_iter: int = 50
    freeze_after_fit: bool = True

    # --- isotonic regression ---
    num_isotonic_bins: int = 100

    # --- evaluation ---
    ece_num_bins: int = 15

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        """Validate CalibrationConfig fields. Return list of errors/warnings."""
        errors: List[str] = []

        msg = _in_set(
            "CalibrationConfig.method", self.method, VALID_CALIBRATION_METHODS
        )
        if msg:
            errors.append(msg)

        msg = _positive_float(
            "CalibrationConfig.initial_temperature", self.initial_temperature
        )
        if msg:
            errors.append(msg)

        msg = _positive_float("CalibrationConfig.fit_lr", self.fit_lr)
        if msg:
            errors.append(msg)

        msg = _positive_int("CalibrationConfig.fit_max_iter", self.fit_max_iter)
        if msg:
            errors.append(msg)

        msg = _positive_int(
            "CalibrationConfig.num_isotonic_bins", self.num_isotonic_bins
        )
        if msg:
            errors.append(msg)

        msg = _positive_int("CalibrationConfig.ece_num_bins", self.ece_num_bins)
        if msg:
            errors.append(msg)

        return errors

    # ------------------------------------------------------------------
    # Parameter estimation
    # ------------------------------------------------------------------

    def estimate_params(self) -> int:
        """Estimate learnable parameter count for calibration.

        Temperature scaling has exactly 1 learnable parameter.
        Isotonic regression stores bin boundaries (num_isotonic_bins).
        """
        if self.method == "temperature":
            return 1
        elif self.method == "isotonic":
            return self.num_isotonic_bins
        return 0


# ===================================================================
# 5. TraceConfig
# ===================================================================


@dataclass
class TraceConfig:
    """Reasoning trace configuration.

    Control how much introspection data the dual-process system records
    during inference and training.
    """

    top_k: int = 5
    full_mode: bool = False
    log_to_file: bool = False
    log_path: Optional[str] = None
    max_log_traces: int = 1000

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        """Validate TraceConfig fields. Return list of errors/warnings."""
        errors: List[str] = []

        msg = _positive_int("TraceConfig.top_k", self.top_k)
        if msg:
            errors.append(msg)

        msg = _positive_int("TraceConfig.max_log_traces", self.max_log_traces)
        if msg:
            errors.append(msg)

        if self.log_to_file and not self.log_path:
            errors.append(
                "TraceConfig.log_path must be set when log_to_file is True"
            )

        return errors

    # ------------------------------------------------------------------
    # Parameter estimation
    # ------------------------------------------------------------------

    def estimate_params(self) -> int:
        """Trace has no learnable parameters."""
        return 0


# ===================================================================
# 6. TrainingConfig
# ===================================================================


@dataclass
class TrainingConfig:
    """Training-specific configuration for dual-process reasoning.

    Collect learning rates, loss weights, and scheduling knobs that only
    matter during the training loop.
    """

    s1_lr: float = 1e-3
    s2_lr: float = 1e-4
    calibration_lr: float = 0.01
    s2_loss_weight: float = 1.0
    deep_supervision_weight: float = 0.1
    novelty_update_freq: int = 1  # update prototypes every N batches
    freeze_calibrator_epoch: int = 5  # freeze calibrator after this epoch

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        """Validate TrainingConfig fields. Return list of errors/warnings."""
        errors: List[str] = []

        msg = _positive_float("TrainingConfig.s1_lr", self.s1_lr)
        if msg:
            errors.append(msg)

        msg = _positive_float("TrainingConfig.s2_lr", self.s2_lr)
        if msg:
            errors.append(msg)

        msg = _positive_float("TrainingConfig.calibration_lr", self.calibration_lr)
        if msg:
            errors.append(msg)

        msg = _positive_float("TrainingConfig.s2_loss_weight", self.s2_loss_weight)
        if msg:
            errors.append(msg)

        msg = _nonneg_float(
            "TrainingConfig.deep_supervision_weight", self.deep_supervision_weight
        )
        if msg:
            errors.append(msg)

        msg = _positive_int(
            "TrainingConfig.novelty_update_freq", self.novelty_update_freq
        )
        if msg:
            errors.append(msg)

        msg = _nonneg_int(
            "TrainingConfig.freeze_calibrator_epoch", self.freeze_calibrator_epoch
        )
        if msg:
            errors.append(msg)

        return errors

    # ------------------------------------------------------------------
    # Parameter estimation
    # ------------------------------------------------------------------

    def estimate_params(self) -> int:
        """Training config has no learnable parameters of its own."""
        return 0


# ===================================================================
# 7. DualProcessFullConfig  --  aggregate root
# ===================================================================


@dataclass
class DualProcessFullConfig:
    """Complete configuration for the dual-process reasoning system.

    Aggregate all sub-configs and provide presets, validation,
    serialization, and parameter estimation.
    """

    system1: System1Config = field(default_factory=System1Config)
    system2: System2Config = field(default_factory=System2Config)
    metacognition: MetacognitionConfig = field(default_factory=MetacognitionConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    trace: TraceConfig = field(default_factory=TraceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # ==================================================================
    # Presets
    # ==================================================================

    @classmethod
    def minimal(cls) -> "DualProcessFullConfig":
        """Create a minimal configuration for unit tests (~10K params).

        Use the smallest feasible dimensions so that tests run instantly
        on CPU with negligible memory.
        """
        return cls(
            system1=System1Config(
                input_dim=64,
                hidden_dim=32,
                output_dim=16,
                num_layers=1,
                pool_mode="mean",
                confidence_head=True,
                activation="relu",
                dropout=0.0,
                num_slots=None,
            ),
            system2=System2Config(
                hidden_dim=32,
                output_dim=16,
                input_summary_dim=32,
                max_steps=3,
                convergence_eps=1e-2,
                convergence_patience=1,
                convergence_criterion="kl",
                nan_guard=True,
                gradient_clip_per_step=1.0,
                deep_supervision=False,
                deep_supervision_discount=0.9,
            ),
            metacognition=MetacognitionConfig(
                route_threshold=0.5,
                w_conf=1.0,
                w_novelty=0.5,
                w_anomaly=0.3,
                w_budget=0.1,
                min_conf_to_skip_s2=0.95,
                base_steps=1,
                step_scale_alpha=1.0,
                max_steps=3,
                always_run_s2=False,
                novelty_method="prototype",
                num_prototypes=8,
                prototype_dim=32,
                prototype_ema_decay=0.99,
            ),
            calibration=CalibrationConfig(
                method="temperature",
                initial_temperature=1.5,
                fit_lr=0.01,
                fit_max_iter=20,
                freeze_after_fit=True,
                num_isotonic_bins=20,
                ece_num_bins=10,
            ),
            trace=TraceConfig(
                top_k=3,
                full_mode=False,
                log_to_file=False,
                log_path=None,
                max_log_traces=100,
            ),
            training=TrainingConfig(
                s1_lr=1e-3,
                s2_lr=1e-4,
                calibration_lr=0.01,
                s2_loss_weight=1.0,
                deep_supervision_weight=0.1,
                novelty_update_freq=1,
                freeze_calibrator_epoch=2,
            ),
        )

    @classmethod
    def dev(cls) -> "DualProcessFullConfig":
        """Create a development configuration (~1M params).

        Suitable for rapid iteration on a single GPU with small datasets
        such as MNIST or CIFAR-10.
        """
        return cls(
            system1=System1Config(
                input_dim=256,
                hidden_dim=128,
                output_dim=64,
                num_layers=2,
                pool_mode="mean",
                confidence_head=True,
                activation="gelu",
                dropout=0.1,
                num_slots=None,
            ),
            system2=System2Config(
                hidden_dim=128,
                output_dim=64,
                input_summary_dim=128,
                max_steps=5,
                convergence_eps=1e-3,
                convergence_patience=2,
                convergence_criterion="kl",
                nan_guard=True,
                gradient_clip_per_step=1.0,
                deep_supervision=False,
                deep_supervision_discount=0.9,
            ),
            metacognition=MetacognitionConfig(
                route_threshold=0.5,
                w_conf=1.0,
                w_novelty=0.5,
                w_anomaly=0.3,
                w_budget=0.1,
                min_conf_to_skip_s2=0.95,
                base_steps=2,
                step_scale_alpha=2.0,
                max_steps=5,
                always_run_s2=False,
                novelty_method="prototype",
                num_prototypes=32,
                prototype_dim=128,
                prototype_ema_decay=0.99,
            ),
            calibration=CalibrationConfig(
                method="temperature",
                initial_temperature=1.5,
                fit_lr=0.01,
                fit_max_iter=50,
                freeze_after_fit=True,
                num_isotonic_bins=50,
                ece_num_bins=15,
            ),
            trace=TraceConfig(
                top_k=5,
                full_mode=False,
                log_to_file=False,
                log_path=None,
                max_log_traces=500,
            ),
            training=TrainingConfig(
                s1_lr=1e-3,
                s2_lr=1e-4,
                calibration_lr=0.01,
                s2_loss_weight=1.0,
                deep_supervision_weight=0.1,
                novelty_update_freq=1,
                freeze_calibrator_epoch=5,
            ),
        )

    @classmethod
    def production_1b(cls) -> "DualProcessFullConfig":
        """Create a production configuration for ~1B param models (~50M params).

        Suitable for single-node multi-GPU training with moderate datasets.
        """
        return cls(
            system1=System1Config(
                input_dim=2048,
                hidden_dim=1024,
                output_dim=512,
                num_layers=3,
                pool_mode="mean",
                confidence_head=True,
                activation="gelu",
                dropout=0.1,
                num_slots=None,
            ),
            system2=System2Config(
                hidden_dim=1024,
                output_dim=512,
                input_summary_dim=1024,
                max_steps=8,
                convergence_eps=1e-3,
                convergence_patience=2,
                convergence_criterion="kl",
                nan_guard=True,
                gradient_clip_per_step=1.0,
                deep_supervision=True,
                deep_supervision_discount=0.9,
            ),
            metacognition=MetacognitionConfig(
                route_threshold=0.5,
                w_conf=1.0,
                w_novelty=0.5,
                w_anomaly=0.3,
                w_budget=0.1,
                min_conf_to_skip_s2=0.95,
                base_steps=3,
                step_scale_alpha=4.0,
                max_steps=8,
                always_run_s2=False,
                novelty_method="prototype",
                num_prototypes=128,
                prototype_dim=1024,
                prototype_ema_decay=0.995,
            ),
            calibration=CalibrationConfig(
                method="temperature",
                initial_temperature=1.5,
                fit_lr=0.01,
                fit_max_iter=50,
                freeze_after_fit=True,
                num_isotonic_bins=100,
                ece_num_bins=15,
            ),
            trace=TraceConfig(
                top_k=5,
                full_mode=False,
                log_to_file=False,
                log_path=None,
                max_log_traces=1000,
            ),
            training=TrainingConfig(
                s1_lr=5e-4,
                s2_lr=5e-5,
                calibration_lr=0.01,
                s2_loss_weight=1.0,
                deep_supervision_weight=0.1,
                novelty_update_freq=2,
                freeze_calibrator_epoch=5,
            ),
        )

    @classmethod
    def production_3b(cls) -> "DualProcessFullConfig":
        """Create a production configuration for ~3B param models (~200M params).

        Suitable for distributed multi-node training with large datasets.
        """
        return cls(
            system1=System1Config(
                input_dim=4096,
                hidden_dim=2048,
                output_dim=1024,
                num_layers=3,
                pool_mode="attention",
                confidence_head=True,
                activation="gelu",
                dropout=0.1,
                num_slots=16,
            ),
            system2=System2Config(
                hidden_dim=2048,
                output_dim=1024,
                input_summary_dim=2048,
                max_steps=10,
                convergence_eps=1e-3,
                convergence_patience=2,
                convergence_criterion="kl",
                nan_guard=True,
                gradient_clip_per_step=1.0,
                deep_supervision=True,
                deep_supervision_discount=0.9,
            ),
            metacognition=MetacognitionConfig(
                route_threshold=0.5,
                w_conf=1.0,
                w_novelty=0.5,
                w_anomaly=0.3,
                w_budget=0.1,
                min_conf_to_skip_s2=0.95,
                base_steps=3,
                step_scale_alpha=5.0,
                max_steps=10,
                always_run_s2=False,
                novelty_method="prototype",
                num_prototypes=256,
                prototype_dim=2048,
                prototype_ema_decay=0.995,
            ),
            calibration=CalibrationConfig(
                method="temperature",
                initial_temperature=1.5,
                fit_lr=0.005,
                fit_max_iter=100,
                freeze_after_fit=True,
                num_isotonic_bins=100,
                ece_num_bins=15,
            ),
            trace=TraceConfig(
                top_k=5,
                full_mode=False,
                log_to_file=False,
                log_path=None,
                max_log_traces=1000,
            ),
            training=TrainingConfig(
                s1_lr=3e-4,
                s2_lr=3e-5,
                calibration_lr=0.005,
                s2_loss_weight=1.0,
                deep_supervision_weight=0.1,
                novelty_update_freq=4,
                freeze_calibrator_epoch=8,
            ),
        )

    @classmethod
    def production_7b(cls) -> "DualProcessFullConfig":
        """Create a production configuration for ~7B param models (~500M params).

        Full production scale with maximum capacity and longest reasoning
        chains. Designed for multi-node distributed training.
        """
        return cls(
            system1=System1Config(
                input_dim=4096,
                hidden_dim=4096,
                output_dim=2048,
                num_layers=4,
                pool_mode="attention",
                confidence_head=True,
                activation="gelu",
                dropout=0.05,
                num_slots=32,
            ),
            system2=System2Config(
                hidden_dim=4096,
                output_dim=2048,
                input_summary_dim=4096,
                max_steps=15,
                convergence_eps=5e-4,
                convergence_patience=3,
                convergence_criterion="kl",
                nan_guard=True,
                gradient_clip_per_step=0.5,
                deep_supervision=True,
                deep_supervision_discount=0.95,
            ),
            metacognition=MetacognitionConfig(
                route_threshold=0.5,
                w_conf=1.0,
                w_novelty=0.5,
                w_anomaly=0.3,
                w_budget=0.1,
                min_conf_to_skip_s2=0.95,
                base_steps=4,
                step_scale_alpha=8.0,
                max_steps=15,
                always_run_s2=False,
                novelty_method="prototype",
                num_prototypes=512,
                prototype_dim=4096,
                prototype_ema_decay=0.998,
            ),
            calibration=CalibrationConfig(
                method="temperature",
                initial_temperature=1.5,
                fit_lr=0.001,
                fit_max_iter=200,
                freeze_after_fit=True,
                num_isotonic_bins=200,
                ece_num_bins=20,
            ),
            trace=TraceConfig(
                top_k=10,
                full_mode=False,
                log_to_file=False,
                log_path=None,
                max_log_traces=2000,
            ),
            training=TrainingConfig(
                s1_lr=1e-4,
                s2_lr=1e-5,
                calibration_lr=0.001,
                s2_loss_weight=1.0,
                deep_supervision_weight=0.05,
                novelty_update_freq=8,
                freeze_calibrator_epoch=10,
            ),
        )

    # ==================================================================
    # Validation
    # ==================================================================

    def validate(self) -> List[str]:
        """Validate configuration consistency across all sub-configs.

        Return a list of error/warning strings.  An empty list means the
        configuration is valid.
        """
        errors: List[str] = []

        # Validate each sub-config
        errors.extend(self.system1.validate())
        errors.extend(self.system2.validate())
        errors.extend(self.metacognition.validate())
        errors.extend(self.calibration.validate())
        errors.extend(self.trace.validate())
        errors.extend(self.training.validate())

        # ----- Cross-config consistency checks -----

        # S1 and S2 output_dim must match
        if self.system1.output_dim != self.system2.output_dim:
            errors.append(
                f"Dimension mismatch: system1.output_dim ({self.system1.output_dim}) "
                f"!= system2.output_dim ({self.system2.output_dim}). "
                f"Both systems must produce same-dimensional outputs for the "
                f"metacognitive router to blend them."
            )

        # S1 hidden_dim should align with S2 input_summary_dim for
        # efficient feature sharing (warning, not error)
        if self.system1.hidden_dim != self.system2.input_summary_dim:
            errors.append(
                f"Warning: system1.hidden_dim ({self.system1.hidden_dim}) != "
                f"system2.input_summary_dim ({self.system2.input_summary_dim}). "
                f"This requires an extra projection layer and may be intentional."
            )

        # S2 max_steps should be consistent with metacognition max_steps
        if self.system2.max_steps != self.metacognition.max_steps:
            errors.append(
                f"Warning: system2.max_steps ({self.system2.max_steps}) != "
                f"metacognition.max_steps ({self.metacognition.max_steps}). "
                f"These should typically match."
            )

        # Prototype dim should match S1 hidden_dim when novelty_method is
        # prototype
        if self.metacognition.novelty_method == "prototype":
            if self.metacognition.prototype_dim != self.system1.hidden_dim:
                errors.append(
                    f"Warning: metacognition.prototype_dim "
                    f"({self.metacognition.prototype_dim}) != "
                    f"system1.hidden_dim ({self.system1.hidden_dim}). "
                    f"Prototypes should live in the same space as S1 features."
                )

        # Deep supervision weight check
        if (
            self.system2.deep_supervision
            and self.training.deep_supervision_weight <= 0
        ):
            errors.append(
                "system2.deep_supervision is True but "
                "training.deep_supervision_weight is <= 0."
            )

        # Calibration LR check
        if (
            self.calibration.method != "none"
            and self.training.calibration_lr <= 0
        ):
            errors.append(
                f"calibration.method is '{self.calibration.method}' but "
                f"training.calibration_lr is <= 0."
            )

        return errors

    # ==================================================================
    # Parameter estimation
    # ==================================================================

    def estimate_params(self) -> Dict[str, int]:
        """Estimate parameter count for each component and total.

        Return a dict with keys: system1, system2, metacognition,
        calibration, trace, training, total.
        """
        s1 = self.system1.estimate_params()
        s2 = self.system2.estimate_params()
        mc = self.metacognition.estimate_params()
        cal = self.calibration.estimate_params()
        tr = self.trace.estimate_params()
        trn = self.training.estimate_params()
        total = s1 + s2 + mc + cal + tr + trn

        return {
            "system1": s1,
            "system2": s2,
            "metacognition": mc,
            "calibration": cal,
            "trace": tr,
            "training": trn,
            "total": total,
        }

    # ==================================================================
    # Serialization
    # ==================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full config to a nested dictionary.

        Use dataclasses.asdict for a complete recursive conversion.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DualProcessFullConfig":
        """Reconstruct a DualProcessFullConfig from a nested dictionary.

        Handle missing keys gracefully by using defaults.
        """
        system1_d = d.get("system1", {})
        system2_d = d.get("system2", {})
        metacognition_d = d.get("metacognition", {})
        calibration_d = d.get("calibration", {})
        trace_d = d.get("trace", {})
        training_d = d.get("training", {})

        return cls(
            system1=_dataclass_from_dict(System1Config, system1_d),
            system2=_dataclass_from_dict(System2Config, system2_d),
            metacognition=_dataclass_from_dict(
                MetacognitionConfig, metacognition_d
            ),
            calibration=_dataclass_from_dict(CalibrationConfig, calibration_d),
            trace=_dataclass_from_dict(TraceConfig, trace_d),
            training=_dataclass_from_dict(TrainingConfig, training_d),
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialize the full config to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_json(cls, s: str) -> "DualProcessFullConfig":
        """Reconstruct a DualProcessFullConfig from a JSON string."""
        d = json.loads(s)
        return cls.from_dict(d)

    def to_yaml(self) -> str:
        """Serialize the full config to a YAML string.

        Fall back to JSON if pyyaml is not installed.
        """
        if _HAS_YAML:
            return yaml.dump(
                self.to_dict(),
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )
        # Fallback
        return self.to_json(indent=2)

    @classmethod
    def from_yaml(cls, s: str) -> "DualProcessFullConfig":
        """Reconstruct a DualProcessFullConfig from a YAML string.

        Fall back to JSON parsing if pyyaml is not installed.
        """
        if _HAS_YAML:
            d = yaml.safe_load(s)
        else:
            d = json.loads(s)
        return cls.from_dict(d)

    # ==================================================================
    # Human-readable summary
    # ==================================================================

    def summary(self) -> str:
        """Return a human-readable multi-line summary of the config."""
        params = self.estimate_params()
        lines = [
            "=== DualProcessFullConfig Summary ===",
            "",
            "  System 1:",
            f"    input_dim     = {self.system1.input_dim}",
            f"    hidden_dim    = {self.system1.hidden_dim}",
            f"    output_dim    = {self.system1.output_dim}",
            f"    num_layers    = {self.system1.num_layers}",
            f"    pool_mode     = {self.system1.pool_mode}",
            f"    activation    = {self.system1.activation}",
            f"    dropout       = {self.system1.dropout}",
            f"    params        ~ {params['system1']:,}",
            "",
            "  System 2:",
            f"    hidden_dim           = {self.system2.hidden_dim}",
            f"    output_dim           = {self.system2.output_dim}",
            f"    input_summary_dim    = {self.system2.input_summary_dim}",
            f"    max_steps            = {self.system2.max_steps}",
            f"    convergence_eps      = {self.system2.convergence_eps}",
            f"    convergence_patience = {self.system2.convergence_patience}",
            f"    convergence_criterion= {self.system2.convergence_criterion}",
            f"    deep_supervision     = {self.system2.deep_supervision}",
            f"    params               ~ {params['system2']:,}",
            "",
            "  Metacognition:",
            f"    route_threshold     = {self.metacognition.route_threshold}",
            f"    weights (c/n/a/b)   = {self.metacognition.w_conf}/"
            f"{self.metacognition.w_novelty}/"
            f"{self.metacognition.w_anomaly}/"
            f"{self.metacognition.w_budget}",
            f"    min_conf_to_skip_s2 = {self.metacognition.min_conf_to_skip_s2}",
            f"    base_steps          = {self.metacognition.base_steps}",
            f"    step_scale_alpha    = {self.metacognition.step_scale_alpha}",
            f"    max_steps           = {self.metacognition.max_steps}",
            f"    always_run_s2       = {self.metacognition.always_run_s2}",
            f"    novelty_method      = {self.metacognition.novelty_method}",
            f"    num_prototypes      = {self.metacognition.num_prototypes}",
            f"    prototype_dim       = {self.metacognition.prototype_dim}",
            f"    params              ~ {params['metacognition']:,}",
            "",
            "  Calibration:",
            f"    method              = {self.calibration.method}",
            f"    initial_temperature = {self.calibration.initial_temperature}",
            f"    params              ~ {params['calibration']:,}",
            "",
            "  Trace:",
            f"    top_k          = {self.trace.top_k}",
            f"    full_mode      = {self.trace.full_mode}",
            f"    log_to_file    = {self.trace.log_to_file}",
            "",
            "  Training:",
            f"    s1_lr                   = {self.training.s1_lr}",
            f"    s2_lr                   = {self.training.s2_lr}",
            f"    calibration_lr          = {self.training.calibration_lr}",
            f"    s2_loss_weight          = {self.training.s2_loss_weight}",
            f"    deep_supervision_weight = {self.training.deep_supervision_weight}",
            f"    novelty_update_freq     = {self.training.novelty_update_freq}",
            f"    freeze_calibrator_epoch = {self.training.freeze_calibrator_epoch}",
            "",
            f"  Total estimated params: {params['total']:,}",
            "=" * 40,
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return a compact repr showing key dimensions and param count."""
        params = self.estimate_params()
        return (
            f"DualProcessFullConfig("
            f"s1=[{self.system1.input_dim}->{self.system1.hidden_dim}"
            f"->{self.system1.output_dim}], "
            f"s2=[{self.system2.input_summary_dim}->{self.system2.hidden_dim}"
            f"->{self.system2.output_dim}, steps={self.system2.max_steps}], "
            f"total_params~{params['total']:,})"
        )


# ===================================================================
# Helper: reconstruct a dataclass from a dict (handles unknown keys)
# ===================================================================


def _dataclass_from_dict(klass: Type[T], d: Dict[str, Any]) -> T:
    """Instantiate a dataclass from a dict, ignoring unknown keys.

    Only pass keys that are actual fields of *klass*.
    """
    if not d:
        return klass()  # type: ignore[call-arg]
    valid_fields = {f.name for f in fields(klass)}
    filtered = {k: v for k, v in d.items() if k in valid_fields}
    return klass(**filtered)  # type: ignore[call-arg]


# ===================================================================
# Convenience: merge two configs (overlay style)
# ===================================================================


def merge_configs(
    base: DualProcessFullConfig,
    overrides: Dict[str, Any],
) -> DualProcessFullConfig:
    """Merge a dict of overrides into a base config, returning a new config.

    The *overrides* dict should have the same nested structure as
    ``DualProcessFullConfig.to_dict()``.  Only keys present in *overrides*
    are changed; everything else keeps the *base* value.

    Example::

        cfg = merge_configs(
            DualProcessFullConfig.dev(),
            {"system1": {"input_dim": 512}, "system2": {"max_steps": 7}},
        )
    """
    base_d = base.to_dict()
    _deep_update(base_d, overrides)
    return DualProcessFullConfig.from_dict(base_d)


def _deep_update(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    """Recursively update *target* dict with values from *source*."""
    for key, value in source.items():
        if (
            key in target
            and isinstance(target[key], dict)
            and isinstance(value, dict)
        ):
            _deep_update(target[key], value)
        else:
            target[key] = value


# ===================================================================
# Convenience: diff two configs
# ===================================================================


def diff_configs(
    a: DualProcessFullConfig,
    b: DualProcessFullConfig,
) -> Dict[str, Any]:
    """Return a nested dict showing only fields that differ between *a* and *b*.

    Useful for logging which settings changed between experiments.
    """
    a_d = a.to_dict()
    b_d = b.to_dict()
    return _dict_diff(a_d, b_d)


def _dict_diff(
    a: Dict[str, Any], b: Dict[str, Any]
) -> Dict[str, Any]:
    """Recursively compute difference between two dicts."""
    result: Dict[str, Any] = {}
    all_keys = set(a.keys()) | set(b.keys())
    for key in sorted(all_keys):
        va = a.get(key, _MISSING)
        vb = b.get(key, _MISSING)
        if va is _MISSING:
            result[key] = {"added": vb}
        elif vb is _MISSING:
            result[key] = {"removed": va}
        elif isinstance(va, dict) and isinstance(vb, dict):
            inner = _dict_diff(va, vb)
            if inner:
                result[key] = inner
        elif va != vb:
            result[key] = {"from": va, "to": vb}
    return result


# ===================================================================
# Convenience: config from environment variables
# ===================================================================


def config_from_env(
    prefix: str = "DUALPROCESS_",
    base: Optional[DualProcessFullConfig] = None,
) -> DualProcessFullConfig:
    """Build or override a config from environment variables.

    Environment variables are expected to be named like:
        DUALPROCESS_SYSTEM1_INPUT_DIM=256
        DUALPROCESS_SYSTEM2_MAX_STEPS=8
        DUALPROCESS_METACOGNITION_NUM_PROTOTYPES=32

    Numeric values are automatically cast to int or float as appropriate.
    Boolean values accept "true"/"false"/"1"/"0".
    """
    import os

    if base is None:
        base = DualProcessFullConfig()

    base_d = base.to_dict()

    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        parts = env_key[len(prefix):].lower().split("_", 1)
        if len(parts) != 2:
            continue
        section, field_name = parts[0], parts[1]
        if section not in base_d:
            continue
        if field_name not in base_d[section]:
            continue

        current = base_d[section][field_name]
        base_d[section][field_name] = _cast_env_value(env_val, current)

    return DualProcessFullConfig.from_dict(base_d)


def _cast_env_value(raw: str, current: Any) -> Any:
    """Cast a raw environment variable string to match the type of *current*."""
    if current is None:
        # Guess: try int, then float, then keep string
        for caster in (int, float):
            try:
                return caster(raw)
            except (ValueError, TypeError):
                continue
        if raw.lower() in ("true", "false"):
            return raw.lower() == "true"
        return raw

    if isinstance(current, bool):
        return raw.lower() in ("true", "1", "yes")
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw


# ===================================================================
# Convenience: quick param count table
# ===================================================================


def param_table() -> str:
    """Return a formatted table comparing parameter counts across presets.

    Useful for documentation and README generation.
    """
    presets = [
        ("minimal", DualProcessFullConfig.minimal()),
        ("dev", DualProcessFullConfig.dev()),
        ("production_1b", DualProcessFullConfig.production_1b()),
        ("production_3b", DualProcessFullConfig.production_3b()),
        ("production_7b", DualProcessFullConfig.production_7b()),
    ]

    header = (
        f"{'Preset':<16} {'System1':>12} {'System2':>12} "
        f"{'Metacog':>12} {'Total':>14}"
    )
    sep = "-" * len(header)
    lines = [header, sep]

    for name, cfg in presets:
        p = cfg.estimate_params()
        lines.append(
            f"{name:<16} {p['system1']:>12,} {p['system2']:>12,} "
            f"{p['metacognition']:>12,} {p['total']:>14,}"
        )

    return "\n".join(lines)


# ===================================================================
# Convenience: register custom presets
# ===================================================================

_CUSTOM_PRESETS: Dict[str, DualProcessFullConfig] = {}


def register_preset(name: str, config: DualProcessFullConfig) -> None:
    """Register a custom preset by name for later retrieval."""
    _CUSTOM_PRESETS[name] = config


def get_preset(name: str) -> DualProcessFullConfig:
    """Retrieve a preset by name.

    Check built-in presets first, then custom registered presets.
    Raise ValueError if not found.
    """
    builtins = {
        "minimal": DualProcessFullConfig.minimal,
        "dev": DualProcessFullConfig.dev,
        "production_1b": DualProcessFullConfig.production_1b,
        "production_3b": DualProcessFullConfig.production_3b,
        "production_7b": DualProcessFullConfig.production_7b,
    }
    if name in builtins:
        return builtins[name]()
    if name in _CUSTOM_PRESETS:
        return copy.deepcopy(_CUSTOM_PRESETS[name])
    available = list(builtins.keys()) + list(_CUSTOM_PRESETS.keys())
    raise ValueError(f"Unknown preset {name!r}. Available: {available}")


# ===================================================================
# Self-test block
# ===================================================================


def _run_self_tests() -> None:
    """Run self-tests and print results."""
    passed = 0
    failed = 0

    def _test(name: str, fn: Any) -> None:
        nonlocal passed, failed
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as exc:
            print(f"  [FAIL] {name}: {exc}")
            failed += 1

    print("Running reasoning_config_template self-tests...\n")

    # ------------------------------------------------------------------
    # Test 1: Default config instantiation
    # ------------------------------------------------------------------
    def test_default_instantiation() -> None:
        cfg = DualProcessFullConfig()
        assert cfg.system1.input_dim == 4096
        assert cfg.system2.max_steps == 10
        assert cfg.metacognition.route_threshold == 0.5
        assert cfg.calibration.method == "temperature"
        assert cfg.trace.top_k == 5
        assert cfg.training.s1_lr == 1e-3

    _test("1. Default config instantiation", test_default_instantiation)

    # ------------------------------------------------------------------
    # Test 2: All 5 presets instantiate without error
    # ------------------------------------------------------------------
    def test_all_presets() -> None:
        presets = [
            DualProcessFullConfig.minimal(),
            DualProcessFullConfig.dev(),
            DualProcessFullConfig.production_1b(),
            DualProcessFullConfig.production_3b(),
            DualProcessFullConfig.production_7b(),
        ]
        for p in presets:
            assert isinstance(p, DualProcessFullConfig)

    _test("2. All 5 presets instantiate", test_all_presets)

    # ------------------------------------------------------------------
    # Test 3: Validation catches invalid configs
    # ------------------------------------------------------------------
    def test_validation_catches_invalid() -> None:
        cfg = DualProcessFullConfig.minimal()

        # Make several things invalid
        cfg.system1.pool_mode = "invalid_mode"
        cfg.system2.convergence_eps = -1.0
        cfg.calibration.initial_temperature = 0.0

        errors = cfg.validate()
        assert len(errors) > 0, "Expected validation errors for invalid config"

        # Check specific errors are caught
        has_pool = any("pool_mode" in e for e in errors)
        has_eps = any("convergence_eps" in e for e in errors)
        has_temp = any("initial_temperature" in e for e in errors)
        assert has_pool, f"Expected pool_mode error, got: {errors}"
        assert has_eps, f"Expected convergence_eps error, got: {errors}"
        assert has_temp, f"Expected initial_temperature error, got: {errors}"

    _test("3. Validation catches invalid configs", test_validation_catches_invalid)

    # ------------------------------------------------------------------
    # Test 4: Serialization roundtrip (to_dict -> from_dict)
    # ------------------------------------------------------------------
    def test_dict_roundtrip() -> None:
        preset_names = [
            "minimal", "dev", "production_1b",
            "production_3b", "production_7b",
        ]
        for preset_name in preset_names:
            original = get_preset(preset_name)
            d = original.to_dict()
            restored = DualProcessFullConfig.from_dict(d)
            assert d == restored.to_dict(), (
                f"Dict roundtrip failed for {preset_name}: "
                f"diff = {diff_configs(original, restored)}"
            )

    _test(
        "4. Serialization roundtrip (to_dict -> from_dict)",
        test_dict_roundtrip,
    )

    # ------------------------------------------------------------------
    # Test 5: JSON serialization
    # ------------------------------------------------------------------
    def test_json_serialization() -> None:
        original = DualProcessFullConfig.dev()
        json_str = original.to_json()
        restored = DualProcessFullConfig.from_json(json_str)
        assert original.to_dict() == restored.to_dict(), "JSON roundtrip failed"

        # Verify it is valid JSON
        parsed = json.loads(json_str)
        assert "system1" in parsed
        assert "system2" in parsed
        assert parsed["system1"]["input_dim"] == 256

    _test("5. JSON serialization", test_json_serialization)

    # ------------------------------------------------------------------
    # Test 6: YAML serialization (skip if no pyyaml)
    # ------------------------------------------------------------------
    def test_yaml_serialization() -> None:
        original = DualProcessFullConfig.production_1b()
        yaml_str = original.to_yaml()
        assert len(yaml_str) > 0, "YAML output is empty"

        if _HAS_YAML:
            restored = DualProcessFullConfig.from_yaml(yaml_str)
            assert original.to_dict() == restored.to_dict(), (
                "YAML roundtrip failed"
            )
            print("    (pyyaml available, full roundtrip tested)")
        else:
            # Fallback to JSON
            restored = DualProcessFullConfig.from_json(yaml_str)
            assert original.to_dict() == restored.to_dict(), (
                "YAML fallback (JSON) roundtrip failed"
            )
            print("    (pyyaml not available, tested JSON fallback)")

    _test("6. YAML serialization", test_yaml_serialization)

    # ------------------------------------------------------------------
    # Test 7: Parameter estimation for each preset
    # ------------------------------------------------------------------
    def test_param_estimation() -> None:
        preset_names = [
            "minimal", "dev", "production_1b",
            "production_3b", "production_7b",
        ]
        for preset_name in preset_names:
            cfg = get_preset(preset_name)
            params = cfg.estimate_params()
            assert "system1" in params
            assert "system2" in params
            assert "metacognition" in params
            assert "total" in params
            assert params["total"] > 0, f"{preset_name} total params is 0"
            assert params["total"] == sum(
                v for k, v in params.items() if k != "total"
            ), f"{preset_name} total != sum of parts"
            print(f"    {preset_name}: {params['total']:,} params")

    _test("7. Parameter estimation for each preset", test_param_estimation)

    # ------------------------------------------------------------------
    # Test 8: Minimal preset has smallest params
    # ------------------------------------------------------------------
    def test_minimal_smallest() -> None:
        minimal_params = (
            DualProcessFullConfig.minimal().estimate_params()["total"]
        )
        for preset_name in [
            "dev", "production_1b", "production_3b", "production_7b"
        ]:
            other_params = get_preset(preset_name).estimate_params()["total"]
            assert minimal_params < other_params, (
                f"minimal ({minimal_params:,}) should be < "
                f"{preset_name} ({other_params:,})"
            )

    _test("8. Minimal preset has smallest params", test_minimal_smallest)

    # ------------------------------------------------------------------
    # Test 9: Production_7b has largest params
    # ------------------------------------------------------------------
    def test_production_7b_largest() -> None:
        p7b_params = (
            DualProcessFullConfig.production_7b().estimate_params()["total"]
        )
        for preset_name in [
            "minimal", "dev", "production_1b", "production_3b"
        ]:
            other_params = get_preset(preset_name).estimate_params()["total"]
            assert p7b_params > other_params, (
                f"production_7b ({p7b_params:,}) should be > "
                f"{preset_name} ({other_params:,})"
            )

    _test("9. Production_7b has largest params", test_production_7b_largest)

    # ------------------------------------------------------------------
    # Test 10: Field override
    # ------------------------------------------------------------------
    def test_field_override() -> None:
        cfg = DualProcessFullConfig(
            system1=System1Config(input_dim=128)
        )
        assert cfg.system1.input_dim == 128
        # Other fields keep defaults
        assert cfg.system1.hidden_dim == 512
        assert cfg.system2.max_steps == 10

        # Override via merge
        merged = merge_configs(
            DualProcessFullConfig.dev(),
            {"system1": {"input_dim": 999}, "system2": {"max_steps": 42}},
        )
        assert merged.system1.input_dim == 999
        assert merged.system2.max_steps == 42
        # Unchanged fields stay at the dev default
        assert merged.system1.hidden_dim == 128  # dev default

    _test("10. Field override", test_field_override)

    # ------------------------------------------------------------------
    # Test 11: Validation warnings for mismatched dims
    # ------------------------------------------------------------------
    def test_validation_dim_mismatch() -> None:
        cfg = DualProcessFullConfig(
            system1=System1Config(output_dim=256),
            system2=System2Config(output_dim=512),
        )
        errors = cfg.validate()
        has_dim_mismatch = any(
            "Dimension mismatch" in e and "output_dim" in e for e in errors
        )
        assert has_dim_mismatch, (
            f"Expected dimension mismatch warning, got: {errors}"
        )

    _test(
        "11. Validation warnings for mismatched dims",
        test_validation_dim_mismatch,
    )

    # ------------------------------------------------------------------
    # Test 12: Diff configs
    # ------------------------------------------------------------------
    def test_diff_configs() -> None:
        a = DualProcessFullConfig.minimal()
        b = DualProcessFullConfig.dev()
        d = diff_configs(a, b)
        # There should be differences in system1.input_dim at least
        assert "system1" in d, f"Expected system1 in diff, got: {d}"
        assert "input_dim" in d["system1"], (
            f"Expected input_dim in system1 diff, got: {d['system1']}"
        )

    _test("12. Diff configs", test_diff_configs)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{passed}/{passed + failed} self-tests passed")
    if failed > 0:
        print(f"WARNING: {failed} test(s) failed!")
        sys.exit(1)
    else:
        print("All tests passed.")

    # Bonus: print param table
    print("\n" + param_table())


if __name__ == "__main__":
    _run_self_tests()
