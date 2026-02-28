"""
Neuromodulation + Eligibility Trace Configuration Template.

Complete, self-contained configuration dataclasses for the three-factor
learning pipeline: eligibility traces, neuromodulatory gating, weight
update rules, and plasticity diagnostics.

All configs support:
  - Field-level validation via __post_init__
  - Serialization: to_dict() / from_dict() with JSON round-trip
  - Preset factories: minimal(), dev(), production()
  - Deep copy safety (no shared mutable state)

These dataclasses are designed to extend brain_ai/config.py and plug into
PlasticityFullConfig which aggregates them alongside feature flags.

Usage::

    from neuromod_config_template import PlasticityFullConfig

    # Quick start with presets
    cfg = PlasticityFullConfig.minimal()     # unit tests
    cfg = PlasticityFullConfig.dev()         # development
    cfg = PlasticityFullConfig.production()  # full training

    # Serialize / deserialize
    d = cfg.to_dict()
    cfg2 = PlasticityFullConfig.from_dict(d)

    # JSON round-trip
    import json
    s = json.dumps(cfg.to_dict())
    cfg3 = PlasticityFullConfig.from_dict(json.loads(s))
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field, fields, asdict
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_TRACE_TYPES: Tuple[str, ...] = ("accumulating", "replacing", "dutch")
VALID_KERNELS: Tuple[str, ...] = ("rate", "stdp_pair", "stdp_symmetric")
VALID_COMBINATION_FNS: Tuple[str, ...] = ("weighted_sum", "gated_product", "mlp")
VALID_THREE_FACTOR_MODES: Tuple[str, ...] = ("online", "hybrid", "auxiliary_loss")

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Enumerations (informational -- configs use plain strings for JSON compat)
# ---------------------------------------------------------------------------

class TraceType(str, Enum):
    """Eligibility trace accumulation strategy.

    Attributes:
        ACCUMULATING: Standard additive accumulation.  ``e += f(pre, post)``
        REPLACING:    Event-driven maximum.             ``e = max(e, f(pre, post))``
        DUTCH:        Hybrid decay + event.             ``e = (1-a)*e + f(pre, post)``
    """

    ACCUMULATING = "accumulating"
    REPLACING = "replacing"
    DUTCH = "dutch"


class KernelType(str, Enum):
    """Pre/post correlation kernel for eligibility traces.

    Attributes:
        RATE:           Outer product of firing rates.
        STDP_PAIR:      Pair-based STDP with asymmetric time constants.
        STDP_SYMMETRIC: Symmetric STDP window.
    """

    RATE = "rate"
    STDP_PAIR = "stdp_pair"
    STDP_SYMMETRIC = "stdp_symmetric"


class CombinationFunction(str, Enum):
    """Strategy for combining neuromodulatory signals into a single gain.

    Attributes:
        WEIGHTED_SUM:   Linear weighted combination.
        GATED_PRODUCT:  Multiplicative gating (sigmoid-based).
        MLP:            Learned nonlinear combination.
    """

    WEIGHTED_SUM = "weighted_sum"
    GATED_PRODUCT = "gated_product"
    MLP = "mlp"


class ThreeFactorMode(str, Enum):
    """Integration mode for the three-factor weight update.

    Attributes:
        ONLINE:         Direct in-place weight update per step.
        HYBRID:         Online updates + auxiliary loss for backprop.
        AUXILIARY_LOSS: Three-factor signal used only as auxiliary loss.
    """

    ONLINE = "online"
    HYBRID = "hybrid"
    AUXILIARY_LOSS = "auxiliary_loss"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_positive(value: float, name: str) -> None:
    """Raise ``ValueError`` if *value* is not strictly positive."""
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def _validate_non_negative(value: float, name: str) -> None:
    """Raise ``ValueError`` if *value* is negative."""
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")


def _validate_in_range(
    value: float, lo: float, hi: float, name: str, *, inclusive: bool = True
) -> None:
    """Raise ``ValueError`` if *value* is outside ``[lo, hi]``."""
    if inclusive:
        if not (lo <= value <= hi):
            raise ValueError(f"{name} must be in [{lo}, {hi}], got {value}")
    else:
        if not (lo < value < hi):
            raise ValueError(f"{name} must be in ({lo}, {hi}), got {value}")


def _validate_choice(value: str, choices: Tuple[str, ...], name: str) -> None:
    """Raise ``ValueError`` if *value* is not in *choices*."""
    if value not in choices:
        raise ValueError(
            f"{name} must be one of {choices}, got '{value}'"
        )


def _validate_clamp_range(
    clamp: Tuple[float, float], name: str
) -> None:
    """Raise ``ValueError`` if clamp min >= clamp max."""
    if clamp[0] >= clamp[1]:
        raise ValueError(
            f"{name} lower bound must be < upper bound, got {clamp}"
        )


def _validate_positive_int(value: int, name: str) -> None:
    """Raise ``ValueError`` if *value* is not a positive integer."""
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}")


def _validate_non_negative_int(value: int, name: str) -> None:
    """Raise ``ValueError`` if *value* is not a non-negative integer."""
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value}")


# ---------------------------------------------------------------------------
# Mixin: serialization helpers
# ---------------------------------------------------------------------------

class _SerializableMixin:
    """Adds ``to_dict`` / ``from_dict`` to any dataclass.

    Supports nested dataclasses that also inherit from this mixin.
    Tuple fields are serialised as lists and reconstituted on load.
    """

    def to_dict(self) -> Dict[str, Any]:
        """Recursively convert this config to a plain dict.

        Returns:
            A JSON-compatible dictionary with all fields.
        """
        result: Dict[str, Any] = {}
        for f in fields(self):  # type: ignore[arg-type]
            value = getattr(self, f.name)
            if isinstance(value, _SerializableMixin):
                result[f.name] = value.to_dict()
            elif isinstance(value, tuple):
                result[f.name] = list(value)
            elif isinstance(value, Enum):
                result[f.name] = value.value
            else:
                result[f.name] = value
        return result

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Construct a config from a dict, recursively handling nested configs.

        Args:
            data: Dictionary with field names as keys.

        Returns:
            An instance of this config class.

        Raises:
            TypeError: If unknown keys are present.
        """
        field_map = {f.name: f for f in fields(cls)}  # type: ignore[arg-type]
        kwargs: Dict[str, Any] = {}
        for key, value in data.items():
            if key not in field_map:
                logger.warning("Unknown config key '%s' for %s -- skipping", key, cls.__name__)
                continue
            f = field_map[key]
            # Detect nested serialisable dataclass
            if (
                isinstance(f.type, type)
                and issubclass(f.type, _SerializableMixin)
                and isinstance(value, dict)
            ):
                kwargs[key] = f.type.from_dict(value)
            else:
                # Convert lists back to tuples where the field type annotation
                # indicates a tuple.  We inspect the *actual* default on the
                # class to determine the intended Python type since string-based
                # annotations are common.
                default_value = getattr(cls, key, None) if hasattr(cls, key) else None
                if isinstance(value, list) and isinstance(default_value, tuple):
                    kwargs[key] = tuple(value)
                elif isinstance(value, list):
                    # Also handle fields where the type hint is Tuple but no
                    # class-level default exists.  We peek at the field's
                    # default / default_factory.
                    _default = f.default if f.default is not f.default_factory else None  # type: ignore[comparison-overlap]
                    if isinstance(_default, tuple):
                        kwargs[key] = tuple(value)
                    else:
                        kwargs[key] = value
                else:
                    kwargs[key] = value
        return cls(**kwargs)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# 1. EligibilityConfig
# ---------------------------------------------------------------------------

@dataclass
class EligibilityConfig(_SerializableMixin):
    """Configuration for eligibility trace dynamics.

    Eligibility traces accumulate local pre/post correlations so that a
    delayed third-factor signal can gate weight changes.  Three trace types
    are supported (accumulating, replacing, Dutch) with configurable STDP
    or rate-based correlation kernels.

    Attributes:
        trace_type:     Trace accumulation strategy.
        tau_e:          Eligibility decay time constant (ms or steps).
        kernel:         Pre/post correlation function type.
        stdp_tau_plus:  STDP potentiation time constant.
        stdp_tau_minus: STDP depression time constant.
        a_plus:         STDP potentiation amplitude scaling.
        a_minus:        STDP depression amplitude scaling.
        dutch_alpha:    Replacement rate for Dutch traces (in [0, 1]).
        clamp_range:    Hard clamp applied to trace values.
        diagonal:       If True, use diagonal (element-wise) traces instead
                        of a full (N_post, N_pre) matrix.  Reduces memory
                        from O(N^2) to O(N) per batch item.
    """

    # -- Trace dynamics --
    trace_type: str = "accumulating"
    tau_e: float = 20.0
    kernel: str = "rate"

    # -- STDP parameters --
    stdp_tau_plus: float = 20.0
    stdp_tau_minus: float = 20.0
    a_plus: float = 1.0
    a_minus: float = 1.0

    # -- Dutch trace --
    dutch_alpha: float = 0.1

    # -- Clamp / memory --
    clamp_range: Tuple[float, float] = (-5.0, 5.0)
    diagonal: bool = False

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate all fields on construction.

        Raises:
            ValueError: If any field value is out of range or invalid.
        """
        _validate_choice(self.trace_type, VALID_TRACE_TYPES, "trace_type")
        _validate_positive(self.tau_e, "tau_e")
        _validate_choice(self.kernel, VALID_KERNELS, "kernel")
        _validate_positive(self.stdp_tau_plus, "stdp_tau_plus")
        _validate_positive(self.stdp_tau_minus, "stdp_tau_minus")
        _validate_positive(self.a_plus, "a_plus")
        _validate_positive(self.a_minus, "a_minus")
        _validate_in_range(self.dutch_alpha, 0.0, 1.0, "dutch_alpha")
        _validate_clamp_range(self.clamp_range, "clamp_range")

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def decay_factor(self) -> float:
        """Per-step multiplicative decay derived from ``tau_e``.

        Computed as ``1 - 1/tau_e`` which gives exponential decay such that
        the trace reaches ~37 % of its peak after ``tau_e`` steps.

        Returns:
            float in (0, 1).
        """
        return 1.0 - 1.0 / self.tau_e

    @property
    def is_stdp(self) -> bool:
        """Return True if the kernel is any STDP variant."""
        return self.kernel.startswith("stdp")

    def __repr__(self) -> str:
        return (
            f"EligibilityConfig(trace_type='{self.trace_type}', tau_e={self.tau_e}, "
            f"kernel='{self.kernel}', diagonal={self.diagonal}, "
            f"clamp_range={self.clamp_range})"
        )


# ---------------------------------------------------------------------------
# 2. NeuromodConfig
# ---------------------------------------------------------------------------

@dataclass
class NeuromodConfig(_SerializableMixin):
    """Configuration for neuromodulatory signal computation.

    Four biologically inspired modulators (DA, ACh, NE, 5-HT) are computed
    from observable signals and combined into a scalar ``global_plasticity``
    gain that gates eligibility-based weight updates.

    Signal sources:
        - DA  (Dopamine):       reward prediction error  -> ``[-1, 1]``
        - ACh (Acetylcholine):  novelty / uncertainty     -> ``[0, 1]``
        - NE  (Norepinephrine): urgency / surprise        -> ``[0, 1]``
        - 5-HT (Serotonin):    patience / long-horizon    -> ``[0, 1]``

    The combination function merges these into a single gain:
        - ``weighted_sum``:   ``g = w_da*DA + w_ach*ACh + w_ne*NE + w_sht*5HT``
        - ``gated_product``:  ``g = sigmoid(DA) * sigmoid(ACh) * ...``
        - ``mlp``:            ``g = MLP([DA, ACh, NE, 5HT])``

    Attributes:
        da_source:            Signal name mapped to DA input.
        ach_source:           Signal name mapped to ACh input.
        ne_source:            Signal name mapped to NE input.
        sht_source:           Signal name mapped to 5-HT input.
        combination_fn:       Strategy for merging modulators.
        modulator_hidden_dim: Hidden layer size when ``combination_fn='mlp'``.
        da_ema_alpha:         EMA smoothing rate for reward baseline.
        ne_smoothing_beta:    Temporal smoothing for NE signal.
        w_da:                 Weight for DA in weighted_sum mode.
        w_ach:                Weight for ACh in weighted_sum mode.
        w_ne:                 Weight for NE in weighted_sum mode.
        w_sht:                Weight for 5-HT in weighted_sum mode.
        learnable_weights:    If True, modulator combination weights are
                              registered as learnable parameters.
    """

    # -- Source mappings --
    da_source: str = "reward"
    ach_source: str = "novelty"
    ne_source: str = "urgency"
    sht_source: str = "patience"

    # -- Combination --
    combination_fn: str = "weighted_sum"
    modulator_hidden_dim: int = 64

    # -- Temporal smoothing --
    da_ema_alpha: float = 0.01
    ne_smoothing_beta: float = 0.1

    # -- Combination weights --
    w_da: float = 1.0
    w_ach: float = 0.5
    w_ne: float = 0.3
    w_sht: float = 0.2

    # -- Learnable flag --
    learnable_weights: bool = False

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate all fields on construction.

        Raises:
            ValueError: If any field value is out of range or invalid.
        """
        _validate_choice(self.combination_fn, VALID_COMBINATION_FNS, "combination_fn")
        _validate_positive_int(self.modulator_hidden_dim, "modulator_hidden_dim")
        _validate_in_range(self.da_ema_alpha, 0.0, 1.0, "da_ema_alpha")
        _validate_in_range(self.ne_smoothing_beta, 0.0, 1.0, "ne_smoothing_beta")
        _validate_non_negative(self.w_da, "w_da")
        _validate_non_negative(self.w_ach, "w_ach")
        _validate_non_negative(self.w_ne, "w_ne")
        _validate_non_negative(self.w_sht, "w_sht")

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def weights_vector(self) -> Tuple[float, float, float, float]:
        """Return combination weights as a tuple ``(DA, ACh, NE, 5-HT)``."""
        return (self.w_da, self.w_ach, self.w_ne, self.w_sht)

    @property
    def source_map(self) -> Dict[str, str]:
        """Return a dict mapping modulator name to source signal name."""
        return {
            "DA": self.da_source,
            "ACh": self.ach_source,
            "NE": self.ne_source,
            "5HT": self.sht_source,
        }

    @property
    def num_modulators(self) -> int:
        """Number of neuromodulatory signals (always 4)."""
        return 4

    def __repr__(self) -> str:
        return (
            f"NeuromodConfig(combination_fn='{self.combination_fn}', "
            f"learnable={self.learnable_weights}, "
            f"weights=({self.w_da}, {self.w_ach}, {self.w_ne}, {self.w_sht}))"
        )


# ---------------------------------------------------------------------------
# 3. ThreeFactorConfig
# ---------------------------------------------------------------------------

@dataclass
class ThreeFactorConfig(_SerializableMixin):
    """Configuration for the three-factor weight update rule.

    The canonical three-factor rule is::

        delta_w = lr * mod_signal * eligibility_trace

    ``mod_signal`` is the global plasticity gain from :class:`NeuromodConfig`.
    ``eligibility_trace`` is computed per :class:`EligibilityConfig`.

    Three integration modes control how delta_w is applied:

    - **online**: Direct in-place weight modification each step.  Targets
      designated "eligible" layers (fast memory adapters, SNN synapses).
    - **hybrid**: Online updates *plus* an auxiliary loss term that flows
      through the standard backprop graph for additional gradient signal.
    - **auxiliary_loss**: Three-factor signal contributes only as a loss
      term; no direct weight modification.

    Attributes:
        mode:             Integration mode.
        lr:               Plasticity learning rate for direct updates.
        weight_clamp:     Hard clamp applied to weights after update.
        delta_clamp:      Hard clamp applied to delta_w before application.
        target_layers:    Specification of which layers are eligible.
                          ``"all_eligible"`` targets every layer that has
                          been registered for three-factor updates.
        update_frequency: Number of forward steps between weight updates.
                          Setting > 1 amortises the update cost.
        auxiliary_weight: Scalar weight for the auxiliary loss term in
                          hybrid or auxiliary_loss mode.
    """

    # -- Mode --
    mode: str = "online"

    # -- Learning rate --
    lr: float = 0.001

    # -- Clamp ranges --
    weight_clamp: Tuple[float, float] = (-1.0, 1.0)
    delta_clamp: Tuple[float, float] = (-0.1, 0.1)

    # -- Layer targeting --
    target_layers: str = "all_eligible"

    # -- Scheduling --
    update_frequency: int = 1

    # -- Auxiliary loss --
    auxiliary_weight: float = 0.1

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate all fields on construction.

        Raises:
            ValueError: If any field value is out of range or invalid.
        """
        _validate_choice(self.mode, VALID_THREE_FACTOR_MODES, "mode")
        _validate_positive(self.lr, "lr")
        _validate_clamp_range(self.weight_clamp, "weight_clamp")
        _validate_clamp_range(self.delta_clamp, "delta_clamp")
        _validate_positive_int(self.update_frequency, "update_frequency")
        _validate_non_negative(self.auxiliary_weight, "auxiliary_weight")

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def is_online(self) -> bool:
        """True if direct weight updates are applied."""
        return self.mode in ("online", "hybrid")

    @property
    def has_auxiliary_loss(self) -> bool:
        """True if an auxiliary loss term is produced."""
        return self.mode in ("hybrid", "auxiliary_loss")

    def __repr__(self) -> str:
        return (
            f"ThreeFactorConfig(mode='{self.mode}', lr={self.lr}, "
            f"update_freq={self.update_frequency}, "
            f"aux_weight={self.auxiliary_weight})"
        )


# ---------------------------------------------------------------------------
# 4. PlasticityDiagnosticsConfig
# ---------------------------------------------------------------------------

@dataclass
class PlasticityDiagnosticsConfig(_SerializableMixin):
    """Configuration for plasticity diagnostic logging.

    Controls what metrics are recorded during three-factor learning and
    how frequently they are emitted.  Diagnostics include:

    - Neuromodulator statistics (mean, std, min, max per modulator)
    - Trace norms (Frobenius norm of eligibility matrices)
    - Update norms (magnitude of delta_w)
    - Clamp hit rates (fraction of updates hitting clamp bounds)

    These are accumulated into a ring buffer of length ``max_history``
    and optionally saved alongside model checkpoints.

    Attributes:
        log_every_n_steps:     Emit diagnostics every N forward steps.
        max_history:           Maximum entries retained in ring buffer.
        detailed:              If True, log per-layer breakdowns in
                               addition to aggregate statistics.
        save_with_checkpoint:  If True, include diagnostic history when
                               the model checkpoint is saved.
        log_modulator_stats:   Log DA/ACh/NE/5-HT statistics.
        log_trace_norms:       Log eligibility trace Frobenius norms.
        log_update_norms:      Log delta_w magnitudes.
        log_clamp_hits:        Log fraction of updates at clamp bounds.
    """

    log_every_n_steps: int = 10
    max_history: int = 10000
    detailed: bool = False
    save_with_checkpoint: bool = True
    log_modulator_stats: bool = True
    log_trace_norms: bool = True
    log_update_norms: bool = True
    log_clamp_hits: bool = True

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate all fields on construction.

        Raises:
            ValueError: If any field value is out of range or invalid.
        """
        _validate_positive_int(self.log_every_n_steps, "log_every_n_steps")
        _validate_positive_int(self.max_history, "max_history")

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def any_logging_enabled(self) -> bool:
        """True if at least one logging category is active."""
        return any([
            self.log_modulator_stats,
            self.log_trace_norms,
            self.log_update_norms,
            self.log_clamp_hits,
        ])

    @property
    def num_enabled_categories(self) -> int:
        """Count of enabled logging categories."""
        return sum([
            self.log_modulator_stats,
            self.log_trace_norms,
            self.log_update_norms,
            self.log_clamp_hits,
        ])

    def __repr__(self) -> str:
        return (
            f"PlasticityDiagnosticsConfig(every={self.log_every_n_steps}, "
            f"history={self.max_history}, detailed={self.detailed}, "
            f"categories={self.num_enabled_categories})"
        )


# ---------------------------------------------------------------------------
# 5. PlasticityFullConfig (aggregate)
# ---------------------------------------------------------------------------

@dataclass
class PlasticityFullConfig(_SerializableMixin):
    """Aggregate configuration for the full three-factor learning pipeline.

    Combines all sub-configs and adds feature flags to enable/disable
    major subsystems at the top level.

    The typical usage pattern is one of the class-method presets::

        cfg = PlasticityFullConfig.minimal()      # unit tests
        cfg = PlasticityFullConfig.dev()           # development
        cfg = PlasticityFullConfig.production()    # full training

    Or construct with overrides::

        cfg = PlasticityFullConfig(
            eligibility=EligibilityConfig(tau_e=50.0, kernel="stdp_pair"),
            neuromod=NeuromodConfig(combination_fn="mlp"),
        )

    Attributes:
        eligibility:          Eligibility trace dynamics configuration.
        neuromod:             Neuromodulatory signal computation config.
        three_factor:         Weight update rule configuration.
        diagnostics:          Diagnostic logging configuration.
        use_neuromodulation:  Master flag -- if False, modulators are
                              bypassed and a constant gain of 1.0 is used.
        use_eligibility:      Master flag -- if False, eligibility traces
                              are not computed (weight updates use raw
                              pre/post correlations directly).
    """

    # -- Sub-configs --
    eligibility: EligibilityConfig = field(default_factory=EligibilityConfig)
    neuromod: NeuromodConfig = field(default_factory=NeuromodConfig)
    three_factor: ThreeFactorConfig = field(default_factory=ThreeFactorConfig)
    diagnostics: PlasticityDiagnosticsConfig = field(
        default_factory=PlasticityDiagnosticsConfig
    )

    # -- Feature flags --
    use_neuromodulation: bool = True
    use_eligibility: bool = True

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Cross-validate sub-configs after construction.

        Raises:
            ValueError: If inconsistent settings are detected.
        """
        # Sub-configs validate themselves in their own __post_init__.
        # Here we check cross-config consistency.
        if self.three_factor.mode == "auxiliary_loss" and not self.use_eligibility:
            logger.warning(
                "auxiliary_loss mode with use_eligibility=False: auxiliary "
                "loss will have no eligibility signal and may be trivial."
            )

        if (
            self.neuromod.combination_fn == "mlp"
            and not self.use_neuromodulation
        ):
            logger.warning(
                "MLP combination function configured but use_neuromodulation "
                "is False -- MLP parameters will not be used."
            )

    # ------------------------------------------------------------------
    # Serialization (override for nested dataclass handling)
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Recursively convert to a plain dictionary.

        Returns:
            JSON-compatible dictionary.
        """
        return {
            "eligibility": self.eligibility.to_dict(),
            "neuromod": self.neuromod.to_dict(),
            "three_factor": self.three_factor.to_dict(),
            "diagnostics": self.diagnostics.to_dict(),
            "use_neuromodulation": self.use_neuromodulation,
            "use_eligibility": self.use_eligibility,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlasticityFullConfig":
        """Construct from a dictionary, recursively rebuilding sub-configs.

        Args:
            data: Dictionary with keys matching field names.

        Returns:
            A fully validated ``PlasticityFullConfig`` instance.
        """
        eligibility_data = data.get("eligibility", {})
        neuromod_data = data.get("neuromod", {})
        three_factor_data = data.get("three_factor", {})
        diagnostics_data = data.get("diagnostics", {})

        # Convert list clamp_range fields back to tuples
        for key in ("clamp_range",):
            if key in eligibility_data and isinstance(eligibility_data[key], list):
                eligibility_data[key] = tuple(eligibility_data[key])

        for key in ("weight_clamp", "delta_clamp"):
            if key in three_factor_data and isinstance(three_factor_data[key], list):
                three_factor_data[key] = tuple(three_factor_data[key])

        return cls(
            eligibility=EligibilityConfig(**eligibility_data),
            neuromod=NeuromodConfig(**neuromod_data),
            three_factor=ThreeFactorConfig(**three_factor_data),
            diagnostics=PlasticityDiagnosticsConfig(**diagnostics_data),
            use_neuromodulation=data.get("use_neuromodulation", True),
            use_eligibility=data.get("use_eligibility", True),
        )

    # ------------------------------------------------------------------
    # JSON helpers
    # ------------------------------------------------------------------

    def to_json(self, indent: int = 2) -> str:
        """Serialize to a JSON string.

        Args:
            indent: JSON indentation level.

        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "PlasticityFullConfig":
        """Deserialize from a JSON string.

        Args:
            json_str: JSON string produced by :meth:`to_json`.

        Returns:
            A fully validated ``PlasticityFullConfig`` instance.
        """
        return cls.from_dict(json.loads(json_str))

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    @classmethod
    def minimal(cls) -> "PlasticityFullConfig":
        """Minimal configuration for unit tests.

        Uses simple accumulating traces with rate-based kernel, fixed
        weighted-sum combination, online mode, and minimal logging.
        Suitable for fast iteration and CI pipelines.

        Returns:
            A ``PlasticityFullConfig`` with lightweight settings.
        """
        return cls(
            eligibility=EligibilityConfig(
                trace_type="accumulating",
                tau_e=10.0,
                kernel="rate",
                stdp_tau_plus=20.0,
                stdp_tau_minus=20.0,
                a_plus=1.0,
                a_minus=1.0,
                dutch_alpha=0.1,
                clamp_range=(-1.0, 1.0),
                diagonal=True,
            ),
            neuromod=NeuromodConfig(
                combination_fn="weighted_sum",
                modulator_hidden_dim=16,
                learnable_weights=False,
                w_da=1.0,
                w_ach=0.5,
                w_ne=0.3,
                w_sht=0.2,
            ),
            three_factor=ThreeFactorConfig(
                mode="online",
                lr=0.01,
                weight_clamp=(-1.0, 1.0),
                delta_clamp=(-0.5, 0.5),
                target_layers="all_eligible",
                update_frequency=1,
                auxiliary_weight=0.0,
            ),
            diagnostics=PlasticityDiagnosticsConfig(
                log_every_n_steps=100,
                max_history=100,
                detailed=False,
                save_with_checkpoint=False,
                log_modulator_stats=True,
                log_trace_norms=False,
                log_update_norms=False,
                log_clamp_hits=False,
            ),
            use_neuromodulation=True,
            use_eligibility=True,
        )

    @classmethod
    def dev(cls) -> "PlasticityFullConfig":
        """Development configuration with STDP traces and detailed logging.

        Uses STDP pair-based kernel for biologically realistic traces,
        weighted-sum combination, online mode, and full diagnostic logging.
        Good for prototyping and debugging plasticity behaviour.

        Returns:
            A ``PlasticityFullConfig`` tuned for development.
        """
        return cls(
            eligibility=EligibilityConfig(
                trace_type="accumulating",
                tau_e=20.0,
                kernel="stdp_pair",
                stdp_tau_plus=20.0,
                stdp_tau_minus=20.0,
                a_plus=1.0,
                a_minus=1.0,
                dutch_alpha=0.1,
                clamp_range=(-5.0, 5.0),
                diagonal=False,
            ),
            neuromod=NeuromodConfig(
                combination_fn="weighted_sum",
                modulator_hidden_dim=64,
                learnable_weights=False,
                w_da=1.0,
                w_ach=0.5,
                w_ne=0.3,
                w_sht=0.2,
                da_ema_alpha=0.01,
                ne_smoothing_beta=0.1,
            ),
            three_factor=ThreeFactorConfig(
                mode="online",
                lr=0.001,
                weight_clamp=(-1.0, 1.0),
                delta_clamp=(-0.1, 0.1),
                target_layers="all_eligible",
                update_frequency=1,
                auxiliary_weight=0.1,
            ),
            diagnostics=PlasticityDiagnosticsConfig(
                log_every_n_steps=10,
                max_history=10000,
                detailed=True,
                save_with_checkpoint=True,
                log_modulator_stats=True,
                log_trace_norms=True,
                log_update_norms=True,
                log_clamp_hits=True,
            ),
            use_neuromodulation=True,
            use_eligibility=True,
        )

    @classmethod
    def production(cls) -> "PlasticityFullConfig":
        """Production configuration with MLP combination and hybrid mode.

        Uses STDP pair-based kernel, learned MLP combination for the
        neuromodulatory gate, hybrid mode (online + auxiliary loss), and
        checkpoint-friendly logging.  Designed for multi-GPU training
        with AMP.

        Returns:
            A ``PlasticityFullConfig`` tuned for production training.
        """
        return cls(
            eligibility=EligibilityConfig(
                trace_type="accumulating",
                tau_e=30.0,
                kernel="stdp_pair",
                stdp_tau_plus=20.0,
                stdp_tau_minus=25.0,
                a_plus=1.0,
                a_minus=0.8,
                dutch_alpha=0.1,
                clamp_range=(-5.0, 5.0),
                diagonal=False,
            ),
            neuromod=NeuromodConfig(
                combination_fn="mlp",
                modulator_hidden_dim=128,
                learnable_weights=True,
                w_da=1.0,
                w_ach=0.5,
                w_ne=0.3,
                w_sht=0.2,
                da_ema_alpha=0.005,
                ne_smoothing_beta=0.05,
            ),
            three_factor=ThreeFactorConfig(
                mode="hybrid",
                lr=0.0005,
                weight_clamp=(-1.0, 1.0),
                delta_clamp=(-0.05, 0.05),
                target_layers="all_eligible",
                update_frequency=1,
                auxiliary_weight=0.1,
            ),
            diagnostics=PlasticityDiagnosticsConfig(
                log_every_n_steps=50,
                max_history=50000,
                detailed=False,
                save_with_checkpoint=True,
                log_modulator_stats=True,
                log_trace_norms=True,
                log_update_norms=True,
                log_clamp_hits=True,
            ),
            use_neuromodulation=True,
            use_eligibility=True,
        )

    # ------------------------------------------------------------------
    # Merge / override
    # ------------------------------------------------------------------

    def override(self, **kwargs: Any) -> "PlasticityFullConfig":
        """Return a new config with selected fields overridden.

        Supports dot-notation keys for nested fields, e.g.::

            cfg.override(**{"eligibility.tau_e": 50.0, "use_eligibility": False})

        Args:
            **kwargs: Field overrides.  Top-level keys are applied directly;
                      dotted keys update nested sub-configs.

        Returns:
            A deep copy of this config with overrides applied.
        """
        new_cfg = copy.deepcopy(self)
        for key, value in kwargs.items():
            parts = key.split(".")
            if len(parts) == 1:
                if not hasattr(new_cfg, parts[0]):
                    raise ValueError(f"Unknown top-level field: {parts[0]}")
                setattr(new_cfg, parts[0], value)
            elif len(parts) == 2:
                sub_cfg = getattr(new_cfg, parts[0], None)
                if sub_cfg is None:
                    raise ValueError(f"Unknown sub-config: {parts[0]}")
                if not hasattr(sub_cfg, parts[1]):
                    raise ValueError(
                        f"Unknown field '{parts[1]}' in {parts[0]}"
                    )
                setattr(sub_cfg, parts[1], value)
            else:
                raise ValueError(
                    f"Override key '{key}' has too many dots; max depth is 2."
                )
        # Re-validate
        new_cfg.eligibility.__post_init__()
        new_cfg.neuromod.__post_init__()
        new_cfg.three_factor.__post_init__()
        new_cfg.diagnostics.__post_init__()
        new_cfg.__post_init__()
        return new_cfg

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable multi-line summary of this config.

        Returns:
            Formatted string suitable for logging or printing.
        """
        lines = [
            "PlasticityFullConfig Summary",
            "=" * 40,
            f"  use_neuromodulation : {self.use_neuromodulation}",
            f"  use_eligibility     : {self.use_eligibility}",
            "",
            "  Eligibility:",
            f"    trace_type  : {self.eligibility.trace_type}",
            f"    tau_e       : {self.eligibility.tau_e}",
            f"    kernel      : {self.eligibility.kernel}",
            f"    diagonal    : {self.eligibility.diagonal}",
            f"    clamp_range : {self.eligibility.clamp_range}",
        ]
        if self.eligibility.is_stdp:
            lines += [
                f"    stdp_tau+   : {self.eligibility.stdp_tau_plus}",
                f"    stdp_tau-   : {self.eligibility.stdp_tau_minus}",
                f"    a+          : {self.eligibility.a_plus}",
                f"    a-          : {self.eligibility.a_minus}",
            ]
        if self.eligibility.trace_type == "dutch":
            lines.append(f"    dutch_alpha : {self.eligibility.dutch_alpha}")

        lines += [
            "",
            "  Neuromodulation:",
            f"    combination : {self.neuromod.combination_fn}",
            f"    learnable   : {self.neuromod.learnable_weights}",
            f"    weights     : DA={self.neuromod.w_da}, ACh={self.neuromod.w_ach}, "
            f"NE={self.neuromod.w_ne}, 5HT={self.neuromod.w_sht}",
            f"    DA source   : {self.neuromod.da_source}",
            f"    ACh source  : {self.neuromod.ach_source}",
            f"    NE source   : {self.neuromod.ne_source}",
            f"    5HT source  : {self.neuromod.sht_source}",
        ]
        if self.neuromod.combination_fn == "mlp":
            lines.append(
                f"    mlp_hidden  : {self.neuromod.modulator_hidden_dim}"
            )

        lines += [
            "",
            "  Three-Factor Update:",
            f"    mode        : {self.three_factor.mode}",
            f"    lr          : {self.three_factor.lr}",
            f"    update_freq : {self.three_factor.update_frequency}",
            f"    weight_clamp: {self.three_factor.weight_clamp}",
            f"    delta_clamp : {self.three_factor.delta_clamp}",
            f"    target      : {self.three_factor.target_layers}",
        ]
        if self.three_factor.has_auxiliary_loss:
            lines.append(
                f"    aux_weight  : {self.three_factor.auxiliary_weight}"
            )

        lines += [
            "",
            "  Diagnostics:",
            f"    log_every   : {self.diagnostics.log_every_n_steps}",
            f"    max_history : {self.diagnostics.max_history}",
            f"    detailed    : {self.diagnostics.detailed}",
            f"    checkpoint  : {self.diagnostics.save_with_checkpoint}",
            f"    categories  : {self.diagnostics.num_enabled_categories}/4",
        ]

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"PlasticityFullConfig("
            f"eligibility={self.eligibility!r}, "
            f"neuromod={self.neuromod!r}, "
            f"three_factor={self.three_factor!r}, "
            f"diagnostics={self.diagnostics!r}, "
            f"use_neuromod={self.use_neuromodulation}, "
            f"use_elig={self.use_eligibility})"
        )


# ---------------------------------------------------------------------------
# Module-level convenience: default config singleton
# ---------------------------------------------------------------------------

def default_plasticity_config() -> PlasticityFullConfig:
    """Return a fresh default ``PlasticityFullConfig`` (same as constructor defaults).

    Returns:
        PlasticityFullConfig with all defaults.
    """
    return PlasticityFullConfig()


def merge_with_brain_config(
    brain_config: Any,
    plasticity_config: Optional[PlasticityFullConfig] = None,
) -> Any:
    """Attach a ``PlasticityFullConfig`` to an existing ``BrainAIConfig``.

    This is a helper for integrating plasticity configuration into the
    main brain_ai config system.  It sets ``brain_config.plasticity``
    to the provided (or default) plasticity config.

    Args:
        brain_config:       An existing ``BrainAIConfig`` instance.
        plasticity_config:  Optional override; uses defaults if None.

    Returns:
        The same ``brain_config`` instance with ``.plasticity`` set.
    """
    if plasticity_config is None:
        plasticity_config = default_plasticity_config()
    brain_config.plasticity = plasticity_config
    return brain_config


# ===================================================================
# Self-Test Block
# ===================================================================

if __name__ == "__main__":
    import sys
    import traceback

    _pass_count = 0
    _fail_count = 0

    def _report(name: str, passed: bool, detail: str = "") -> None:
        """Print a PASS/FAIL line and update counters."""
        global _pass_count, _fail_count
        status = "PASS" if passed else "FAIL"
        suffix = f" -- {detail}" if detail else ""
        print(f"  [{status}] {name}{suffix}")
        if passed:
            _pass_count += 1
        else:
            _fail_count += 1

    print("=" * 60)
    print("Neuromodulation Config Template -- Self-Tests")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # Test 1: Default construction
    # ------------------------------------------------------------------
    try:
        e = EligibilityConfig()
        n = NeuromodConfig()
        t = ThreeFactorConfig()
        d = PlasticityDiagnosticsConfig()
        f_cfg = PlasticityFullConfig()
        _report(
            "1. Default construction",
            True,
            "all configs instantiate with defaults",
        )
    except Exception as exc:
        _report("1. Default construction", False, str(exc))

    # ------------------------------------------------------------------
    # Test 2: minimal() preset
    # ------------------------------------------------------------------
    try:
        cfg_min = PlasticityFullConfig.minimal()
        assert cfg_min.eligibility.trace_type == "accumulating"
        assert cfg_min.eligibility.diagonal is True
        assert cfg_min.neuromod.combination_fn == "weighted_sum"
        assert cfg_min.three_factor.mode == "online"
        assert cfg_min.diagnostics.detailed is False
        _report("2. minimal() preset", True, "creates valid minimal config")
    except Exception as exc:
        _report("2. minimal() preset", False, str(exc))

    # ------------------------------------------------------------------
    # Test 3: dev() preset
    # ------------------------------------------------------------------
    try:
        cfg_dev = PlasticityFullConfig.dev()
        assert cfg_dev.eligibility.kernel == "stdp_pair"
        assert cfg_dev.neuromod.combination_fn == "weighted_sum"
        assert cfg_dev.three_factor.mode == "online"
        assert cfg_dev.diagnostics.detailed is True
        _report("3. dev() preset", True, "creates valid dev config")
    except Exception as exc:
        _report("3. dev() preset", False, str(exc))

    # ------------------------------------------------------------------
    # Test 4: production() preset
    # ------------------------------------------------------------------
    try:
        cfg_prod = PlasticityFullConfig.production()
        assert cfg_prod.eligibility.kernel == "stdp_pair"
        assert cfg_prod.neuromod.combination_fn == "mlp"
        assert cfg_prod.three_factor.mode == "hybrid"
        assert cfg_prod.diagnostics.save_with_checkpoint is True
        _report("4. production() preset", True, "creates valid production config")
    except Exception as exc:
        _report("4. production() preset", False, str(exc))

    # ------------------------------------------------------------------
    # Test 5: to_dict / from_dict round-trip
    # ------------------------------------------------------------------
    try:
        cfg_orig = PlasticityFullConfig.dev()
        d_dict = cfg_orig.to_dict()
        cfg_restored = PlasticityFullConfig.from_dict(d_dict)

        # Compare all leaf fields
        assert cfg_restored.eligibility.trace_type == cfg_orig.eligibility.trace_type
        assert cfg_restored.eligibility.tau_e == cfg_orig.eligibility.tau_e
        assert cfg_restored.eligibility.kernel == cfg_orig.eligibility.kernel
        assert cfg_restored.eligibility.clamp_range == cfg_orig.eligibility.clamp_range
        assert cfg_restored.neuromod.combination_fn == cfg_orig.neuromod.combination_fn
        assert cfg_restored.neuromod.w_da == cfg_orig.neuromod.w_da
        assert cfg_restored.neuromod.learnable_weights == cfg_orig.neuromod.learnable_weights
        assert cfg_restored.three_factor.mode == cfg_orig.three_factor.mode
        assert cfg_restored.three_factor.lr == cfg_orig.three_factor.lr
        assert cfg_restored.three_factor.weight_clamp == cfg_orig.three_factor.weight_clamp
        assert cfg_restored.three_factor.delta_clamp == cfg_orig.three_factor.delta_clamp
        assert cfg_restored.diagnostics.log_every_n_steps == cfg_orig.diagnostics.log_every_n_steps
        assert cfg_restored.diagnostics.detailed == cfg_orig.diagnostics.detailed
        assert cfg_restored.use_neuromodulation == cfg_orig.use_neuromodulation
        assert cfg_restored.use_eligibility == cfg_orig.use_eligibility
        _report("5. to_dict/from_dict round-trip", True, "all fields preserved")
    except Exception as exc:
        _report("5. to_dict/from_dict round-trip", False, str(exc))

    # ------------------------------------------------------------------
    # Test 6: JSON serialization round-trip
    # ------------------------------------------------------------------
    try:
        cfg_orig = PlasticityFullConfig.production()
        json_str = cfg_orig.to_json()
        cfg_from_json = PlasticityFullConfig.from_json(json_str)
        assert cfg_from_json.to_dict() == cfg_orig.to_dict()
        _report("6. JSON round-trip", True, "JSON serialize/deserialize matches")
    except Exception as exc:
        _report("6. JSON round-trip", False, str(exc))

    # ------------------------------------------------------------------
    # Test 7: tau_e <= 0 raises ValueError
    # ------------------------------------------------------------------
    try:
        caught = False
        try:
            EligibilityConfig(tau_e=0.0)
        except ValueError:
            caught = True
        try:
            EligibilityConfig(tau_e=-5.0)
        except ValueError:
            caught = caught and True
        assert caught, "Expected ValueError for tau_e <= 0"
        _report("7. tau_e <= 0 validation", True, "ValueError raised correctly")
    except Exception as exc:
        _report("7. tau_e <= 0 validation", False, str(exc))

    # ------------------------------------------------------------------
    # Test 8: invalid trace_type raises ValueError
    # ------------------------------------------------------------------
    try:
        caught = False
        try:
            EligibilityConfig(trace_type="invalid_type")
        except ValueError:
            caught = True
        assert caught, "Expected ValueError for invalid trace_type"
        _report("8. invalid trace_type validation", True, "ValueError raised correctly")
    except Exception as exc:
        _report("8. invalid trace_type validation", False, str(exc))

    # ------------------------------------------------------------------
    # Test 9: invalid three-factor mode raises ValueError
    # ------------------------------------------------------------------
    try:
        caught = False
        try:
            ThreeFactorConfig(mode="invalid_mode")
        except ValueError:
            caught = True
        assert caught, "Expected ValueError for invalid mode"
        _report("9. invalid mode validation", True, "ValueError raised correctly")
    except Exception as exc:
        _report("9. invalid mode validation", False, str(exc))

    # ------------------------------------------------------------------
    # Test 10: minimal is simpler than production
    # ------------------------------------------------------------------
    try:
        cfg_min = PlasticityFullConfig.minimal()
        cfg_prod = PlasticityFullConfig.production()

        # Minimal should have simpler settings
        assert cfg_min.eligibility.diagonal is True  # diagonal (cheaper)
        assert cfg_prod.eligibility.diagonal is False  # full matrix
        assert cfg_min.eligibility.kernel == "rate"  # simple kernel
        assert cfg_prod.eligibility.kernel == "stdp_pair"  # complex kernel
        assert cfg_min.neuromod.combination_fn == "weighted_sum"  # fixed
        assert cfg_prod.neuromod.combination_fn == "mlp"  # learned
        assert cfg_min.three_factor.mode == "online"  # simple
        assert cfg_prod.three_factor.mode == "hybrid"  # complex
        assert cfg_min.diagnostics.max_history < cfg_prod.diagnostics.max_history
        _report(
            "10. minimal simpler than production",
            True,
            "diagonal, kernel, combination, mode all simpler",
        )
    except Exception as exc:
        _report("10. minimal simpler than production", False, str(exc))

    # ------------------------------------------------------------------
    # Test 11: feature flags default to True
    # ------------------------------------------------------------------
    try:
        cfg = PlasticityFullConfig()
        assert cfg.use_neuromodulation is True
        assert cfg.use_eligibility is True
        _report("11. feature flags default True", True, "both flags are True")
    except Exception as exc:
        _report("11. feature flags default True", False, str(exc))

    # ------------------------------------------------------------------
    # Test 12: deep copy isolation
    # ------------------------------------------------------------------
    try:
        cfg_a = PlasticityFullConfig.dev()
        cfg_b = copy.deepcopy(cfg_a)
        cfg_b.eligibility.tau_e = 999.0
        cfg_b.neuromod.w_da = 99.0
        cfg_b.three_factor.lr = 0.999
        cfg_b.use_neuromodulation = False

        # Original should be unaffected
        assert cfg_a.eligibility.tau_e == 20.0
        assert cfg_a.neuromod.w_da == 1.0
        assert cfg_a.three_factor.lr == 0.001
        assert cfg_a.use_neuromodulation is True
        _report("12. deep copy isolation", True, "modify copy does not affect original")
    except Exception as exc:
        _report("12. deep copy isolation", False, str(exc))

    # ------------------------------------------------------------------
    # Test 13: clamp range validation (min >= max)
    # ------------------------------------------------------------------
    try:
        caught_equal = False
        caught_reversed = False
        try:
            EligibilityConfig(clamp_range=(5.0, 5.0))
        except ValueError:
            caught_equal = True
        try:
            EligibilityConfig(clamp_range=(5.0, -5.0))
        except ValueError:
            caught_reversed = True
        assert caught_equal and caught_reversed
        _report(
            "13. clamp range validation",
            True,
            "ValueError for equal and reversed bounds",
        )
    except Exception as exc:
        _report("13. clamp range validation", False, str(exc))

    # ------------------------------------------------------------------
    # Test 14: invalid combination_fn raises ValueError
    # ------------------------------------------------------------------
    try:
        caught = False
        try:
            NeuromodConfig(combination_fn="attention")
        except ValueError:
            caught = True
        assert caught, "Expected ValueError for invalid combination_fn"
        _report("14. invalid combination_fn", True, "ValueError raised correctly")
    except Exception as exc:
        _report("14. invalid combination_fn", False, str(exc))

    # ------------------------------------------------------------------
    # Test 15: negative weight raises ValueError
    # ------------------------------------------------------------------
    try:
        caught = False
        try:
            NeuromodConfig(w_da=-0.1)
        except ValueError:
            caught = True
        assert caught, "Expected ValueError for negative w_da"
        _report("15. negative modulator weight", True, "ValueError raised correctly")
    except Exception as exc:
        _report("15. negative modulator weight", False, str(exc))

    # ------------------------------------------------------------------
    # Test 16: lr <= 0 raises ValueError
    # ------------------------------------------------------------------
    try:
        caught = False
        try:
            ThreeFactorConfig(lr=0.0)
        except ValueError:
            caught = True
        assert caught, "Expected ValueError for lr=0"
        _report("16. lr <= 0 validation", True, "ValueError raised correctly")
    except Exception as exc:
        _report("16. lr <= 0 validation", False, str(exc))

    # ------------------------------------------------------------------
    # Test 17: dutch_alpha out of [0,1] raises ValueError
    # ------------------------------------------------------------------
    try:
        caught = False
        try:
            EligibilityConfig(dutch_alpha=1.5)
        except ValueError:
            caught = True
        assert caught, "Expected ValueError for dutch_alpha > 1"
        _report("17. dutch_alpha out of range", True, "ValueError raised correctly")
    except Exception as exc:
        _report("17. dutch_alpha out of range", False, str(exc))

    # ------------------------------------------------------------------
    # Test 18: sub-config convenience properties
    # ------------------------------------------------------------------
    try:
        e_cfg = EligibilityConfig(kernel="stdp_pair", tau_e=20.0)
        assert e_cfg.is_stdp is True
        assert abs(e_cfg.decay_factor - 0.95) < 1e-6

        e_rate = EligibilityConfig(kernel="rate")
        assert e_rate.is_stdp is False

        n_cfg = NeuromodConfig()
        assert n_cfg.num_modulators == 4
        assert n_cfg.source_map["DA"] == "reward"
        assert n_cfg.weights_vector == (1.0, 0.5, 0.3, 0.2)

        t_cfg = ThreeFactorConfig(mode="hybrid")
        assert t_cfg.is_online is True
        assert t_cfg.has_auxiliary_loss is True

        t_aux = ThreeFactorConfig(mode="auxiliary_loss")
        assert t_aux.is_online is False
        assert t_aux.has_auxiliary_loss is True

        d_cfg = PlasticityDiagnosticsConfig()
        assert d_cfg.any_logging_enabled is True
        assert d_cfg.num_enabled_categories == 4

        _report("18. convenience properties", True, "all properties correct")
    except Exception as exc:
        _report("18. convenience properties", False, str(exc))

    # ------------------------------------------------------------------
    # Test 19: override() method
    # ------------------------------------------------------------------
    try:
        cfg = PlasticityFullConfig.dev()
        cfg2 = cfg.override(
            **{
                "eligibility.tau_e": 50.0,
                "neuromod.w_da": 2.0,
                "three_factor.lr": 0.01,
                "use_eligibility": False,
            }
        )
        # New config has overrides
        assert cfg2.eligibility.tau_e == 50.0
        assert cfg2.neuromod.w_da == 2.0
        assert cfg2.three_factor.lr == 0.01
        assert cfg2.use_eligibility is False
        # Original unchanged
        assert cfg.eligibility.tau_e == 20.0
        assert cfg.neuromod.w_da == 1.0
        assert cfg.three_factor.lr == 0.001
        assert cfg.use_eligibility is True
        _report("19. override() method", True, "overrides applied, original intact")
    except Exception as exc:
        _report("19. override() method", False, str(exc))

    # ------------------------------------------------------------------
    # Test 20: summary() produces output
    # ------------------------------------------------------------------
    try:
        cfg = PlasticityFullConfig.production()
        summary_text = cfg.summary()
        assert "PlasticityFullConfig Summary" in summary_text
        assert "Eligibility:" in summary_text
        assert "Neuromodulation:" in summary_text
        assert "Three-Factor Update:" in summary_text
        assert "Diagnostics:" in summary_text
        assert "mlp" in summary_text  # production uses MLP
        assert "hybrid" in summary_text  # production uses hybrid
        _report("20. summary() output", True, "all sections present")
    except Exception as exc:
        _report("20. summary() output", False, str(exc))

    # ------------------------------------------------------------------
    # Test 21: update_frequency < 1 raises ValueError
    # ------------------------------------------------------------------
    try:
        caught = False
        try:
            ThreeFactorConfig(update_frequency=0)
        except ValueError:
            caught = True
        assert caught, "Expected ValueError for update_frequency=0"
        _report("21. update_frequency < 1", True, "ValueError raised correctly")
    except Exception as exc:
        _report("21. update_frequency < 1", False, str(exc))

    # ------------------------------------------------------------------
    # Test 22: invalid kernel raises ValueError
    # ------------------------------------------------------------------
    try:
        caught = False
        try:
            EligibilityConfig(kernel="hebbian")
        except ValueError:
            caught = True
        assert caught, "Expected ValueError for invalid kernel"
        _report("22. invalid kernel validation", True, "ValueError raised correctly")
    except Exception as exc:
        _report("22. invalid kernel validation", False, str(exc))

    # ------------------------------------------------------------------
    # Test 23: all presets produce distinct configs
    # ------------------------------------------------------------------
    try:
        cfgs = {
            "minimal": PlasticityFullConfig.minimal(),
            "dev": PlasticityFullConfig.dev(),
            "production": PlasticityFullConfig.production(),
        }
        dicts = {name: cfg.to_dict() for name, cfg in cfgs.items()}
        # Each pair should differ
        assert dicts["minimal"] != dicts["dev"]
        assert dicts["dev"] != dicts["production"]
        assert dicts["minimal"] != dicts["production"]
        _report("23. presets are distinct", True, "all three presets differ")
    except Exception as exc:
        _report("23. presets are distinct", False, str(exc))

    # ------------------------------------------------------------------
    # Test 24: weight_clamp reversed raises ValueError
    # ------------------------------------------------------------------
    try:
        caught = False
        try:
            ThreeFactorConfig(weight_clamp=(1.0, -1.0))
        except ValueError:
            caught = True
        assert caught
        _report("24. weight_clamp reversed", True, "ValueError raised correctly")
    except Exception as exc:
        _report("24. weight_clamp reversed", False, str(exc))

    # ------------------------------------------------------------------
    # Test 25: delta_clamp reversed raises ValueError
    # ------------------------------------------------------------------
    try:
        caught = False
        try:
            ThreeFactorConfig(delta_clamp=(0.5, -0.5))
        except ValueError:
            caught = True
        assert caught
        _report("25. delta_clamp reversed", True, "ValueError raised correctly")
    except Exception as exc:
        _report("25. delta_clamp reversed", False, str(exc))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    total = _pass_count + _fail_count
    print(f"Results: {_pass_count}/{total} passed, {_fail_count}/{total} failed")
    if _fail_count == 0:
        print("All tests passed.")
    else:
        print(f"WARNING: {_fail_count} test(s) failed!")
    print("=" * 60)

    sys.exit(0 if _fail_count == 0 else 1)
