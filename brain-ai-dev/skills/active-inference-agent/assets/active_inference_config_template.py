"""
Active Inference Agent -- Comprehensive Configuration Template.

All configuration dataclasses for the active inference decision module,
aggregated into ``ActiveInferenceFullConfig`` with scale presets, JSON
serialization, field validation, and legacy upgrade from ``DecisionConfig``.

Hierarchy:
    ActiveInferenceFullConfig
    +-- GenerativeModelConfig      (encoder, decoder, transition model)
    +-- EFEConfig                  (expected free energy computation)
    +-- PlannerConfig              (CEM / random-shooting rollouts)
    +-- AmortizedPolicyConfig      (distilled fast-policy network)
    +-- PreferenceConfig           (learned / fixed goal preferences)
    +-- PyMDPConfig                (discrete POMDP backend via pymdp)
    +-- OfflineRLConfig            (offline RL dataset training loop)

The decision layer (basal ganglia analog) receives 4096-dim workspace
representations and produces actions via three-component Expected Free Energy.

References:
    - Friston et al., "Active Inference: A Process Theory", 2017
    - Fountas et al., "Deep Active Inference Agents Using MC Methods", 2020
    - Millidge et al., "Deep Active Inference with three-term EFE", 2024
"""

from __future__ import annotations

import copy
import json
import logging
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# ============================================================================
# Module-level constants
# ============================================================================

_VERSION: str = "1.0.0"
"""Schema version embedded in serialized configs for future migration."""

_VALID_LATENT_TYPES: Tuple[str, ...] = ("continuous", "discrete")
_VALID_CTX_MODES: Tuple[str, ...] = ("concat", "cross_attention", "film")
_VALID_PLANNER_TYPES: Tuple[str, ...] = ("cem", "random_shooting", "amortized", "mppi")
_VALID_DISTILL_MODES: Tuple[str, ...] = ("soft", "hard", "mixed")
_VALID_PREFERENCE_MODES: Tuple[str, ...] = ("fixed", "learned", "reward_derived")
_VALID_TERM_NORM_MODES: Tuple[str, ...] = ("running", "sigmoid", "none")
_VALID_DISCRETIZATION_METHODS: Tuple[str, ...] = ("kmeans", "uniform", "quantile")


# ============================================================================
# Helper: tuple round-trip for JSON serialization
# ============================================================================

def _encode_tuples(obj: Any) -> Any:
    """Recursively convert tuples to ``{"__tuple__": true, "items": [...]}`` for JSON."""
    if isinstance(obj, tuple):
        return {"__tuple__": True, "items": [_encode_tuples(v) for v in obj]}
    if isinstance(obj, dict):
        return {k: _encode_tuples(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_encode_tuples(v) for v in obj]
    return obj


def _decode_tuples(obj: Any) -> Any:
    """Recursively restore tuples from tagged dicts produced by ``_encode_tuples``."""
    if isinstance(obj, dict):
        if obj.get("__tuple__") is True:
            return tuple(_decode_tuples(v) for v in obj["items"])
        return {k: _decode_tuples(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_tuples(v) for v in obj]
    return obj


# ============================================================================
# 1. GenerativeModelConfig
# ============================================================================

@dataclass
class GenerativeModelConfig:
    """Configuration for the four-component generative model stack.

    Components: encoder q(s|o), decoder p(o|s), transition p(s'|s,a),
    and preferences p_pref(o).  An ensemble of transition networks
    provides epistemic uncertainty for the EFE epistemic term.

    Attributes:
        obs_dim: Workspace observation dimensionality (4096 for production).
        state_dim: Latent state dimensionality.
        action_dim: Action dimensions (continuous width or discrete count).
        hidden_dim: Width of hidden layers in encoder/decoder/transition.
        latent_type: ``"continuous"`` (Gaussian) or ``"discrete"`` (categorical).
        num_discrete_states: Categories when ``latent_type == "discrete"``.
        encoder_layers: Depth of the latent encoder MLP.
        decoder_layers: Depth of the likelihood decoder MLP.
        transition_ensemble_size: Number of ensemble members (1 disables).
        log_var_clamp: ``(min, max)`` for log-variance clamping.
        kl_weight: ELBO KL weight (< 1.0 gives beta-VAE style objective).
        ctx_mode: Context integration: ``"concat"``, ``"cross_attention"``, ``"film"``.
        ctx_dim: Context vector dimensionality (None disables).
    """

    obs_dim: int = 4096
    state_dim: int = 256
    action_dim: int = 128
    hidden_dim: int = 512
    latent_type: str = "continuous"
    num_discrete_states: int = 32
    encoder_layers: int = 3
    decoder_layers: int = 3
    transition_ensemble_size: int = 5
    log_var_clamp: Tuple[float, float] = (-10.0, 2.0)
    kl_weight: float = 1.0
    ctx_mode: str = "concat"
    ctx_dim: Optional[int] = None

    def validate(self) -> List[str]:
        """Validate field values; return list of error messages (empty if valid)."""
        errors: List[str] = []
        if self.obs_dim <= 0:
            errors.append(f"obs_dim must be positive, got {self.obs_dim}")
        if self.state_dim <= 0:
            errors.append(f"state_dim must be positive, got {self.state_dim}")
        if self.action_dim <= 0:
            errors.append(f"action_dim must be positive, got {self.action_dim}")
        if self.hidden_dim <= 0:
            errors.append(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.latent_type not in _VALID_LATENT_TYPES:
            errors.append(
                f"latent_type must be one of {_VALID_LATENT_TYPES}, "
                f"got '{self.latent_type}'"
            )
        if self.num_discrete_states <= 0:
            errors.append(
                f"num_discrete_states must be positive, got {self.num_discrete_states}"
            )
        if self.encoder_layers < 1:
            errors.append(f"encoder_layers must be >= 1, got {self.encoder_layers}")
        if self.decoder_layers < 1:
            errors.append(f"decoder_layers must be >= 1, got {self.decoder_layers}")
        if self.transition_ensemble_size < 1:
            errors.append(
                f"transition_ensemble_size must be >= 1, "
                f"got {self.transition_ensemble_size}"
            )
        if not isinstance(self.log_var_clamp, (tuple, list)) or len(self.log_var_clamp) != 2:
            errors.append(
                f"log_var_clamp must be a 2-tuple (min, max), "
                f"got {self.log_var_clamp}"
            )
        else:
            lo, hi = self.log_var_clamp
            if lo >= hi:
                errors.append(
                    f"log_var_clamp min ({lo}) must be < max ({hi})"
                )
        if self.kl_weight < 0.0:
            errors.append(f"kl_weight must be non-negative, got {self.kl_weight}")
        if self.ctx_mode not in _VALID_CTX_MODES:
            errors.append(
                f"ctx_mode must be one of {_VALID_CTX_MODES}, "
                f"got '{self.ctx_mode}'"
            )
        if self.ctx_dim is not None and self.ctx_dim <= 0:
            errors.append(f"ctx_dim must be positive or None, got {self.ctx_dim}")
        return errors


# ============================================================================
# 2. EFEConfig
# ============================================================================

@dataclass
class EFEConfig:
    """Configuration for Expected Free Energy (EFE) computation.

    Three-component decomposition (computed in fp32):
        EFE(pi) = w_p * Pragmatic + w_e * Epistemic + w_i * Instrumental
    Instrumental returns negative empowerment (non-positive), so all terms sum directly.
    Hard invariant: |sum(terms) - total| < 1e-5 per batch element.

    Attributes:
        pragmatic_weight: Weight for goal-directed term.
        epistemic_weight: Weight for information-seeking term.
        instrumental_weight: Weight for empowerment term.
        num_samples: Monte Carlo samples for EFE expectations.
        discount_factor: Temporal discount gamma per planning step.
        normalize_by_horizon: Divide trajectory EFE by horizon length.
        normalize_terms: Apply per-term normalization before weighting.
        term_norm_mode: ``"running"``, ``"sigmoid"``, or ``"none"``.
    """

    pragmatic_weight: float = 1.0
    epistemic_weight: float = 1.0
    instrumental_weight: float = 0.1
    num_samples: int = 32
    discount_factor: float = 0.99
    normalize_by_horizon: bool = True
    normalize_terms: bool = False
    term_norm_mode: str = "running"

    def validate(self) -> List[str]:
        """Validate field values; return list of error messages (empty if valid)."""
        errors: List[str] = []
        if self.pragmatic_weight < 0.0:
            errors.append(
                f"pragmatic_weight must be non-negative, got {self.pragmatic_weight}"
            )
        if self.epistemic_weight < 0.0:
            errors.append(
                f"epistemic_weight must be non-negative, got {self.epistemic_weight}"
            )
        if self.instrumental_weight < 0.0:
            errors.append(
                f"instrumental_weight must be non-negative, "
                f"got {self.instrumental_weight}"
            )
        if self.num_samples < 1:
            errors.append(f"num_samples must be >= 1, got {self.num_samples}")
        if not (0.0 < self.discount_factor <= 1.0):
            errors.append(
                f"discount_factor must be in (0, 1], got {self.discount_factor}"
            )
        if self.term_norm_mode not in _VALID_TERM_NORM_MODES:
            errors.append(
                f"term_norm_mode must be one of {_VALID_TERM_NORM_MODES}, "
                f"got '{self.term_norm_mode}'"
            )
        all_zero = (
            self.pragmatic_weight == 0.0
            and self.epistemic_weight == 0.0
            and self.instrumental_weight == 0.0
        )
        if all_zero:
            errors.append(
                "At least one EFE weight (pragmatic, epistemic, instrumental) "
                "must be non-zero"
            )
        return errors


# ============================================================================
# 3. PlannerConfig
# ============================================================================

@dataclass
class PlannerConfig:
    """Configuration for the planning / rollout sub-system.

    Supported planners: ``"cem"`` (Cross-Entropy Method), ``"random_shooting"``,
    ``"mppi"`` (Model Predictive Path Integral), ``"amortized"`` (policy network).

    Attributes:
        planner_type: Planner algorithm name.
        num_rollouts: Candidate action sequences per planning step.
        cem_iterations: CEM refinement rounds.
        cem_elite_fraction: Fraction of top rollouts kept per CEM round.
        cem_temperature: CEM elite re-weighting temperature.
        cem_momentum: CEM mean/std momentum between steps (0 = fresh start).
        action_bounds: ``(lower, upper)`` for continuous action clamping.
        action_temperature: Softmax temperature for action selection.
        discrete_actions: If True, sample discrete action indices.
    """

    planner_type: str = "cem"
    num_rollouts: int = 128
    cem_iterations: int = 5
    cem_elite_fraction: float = 0.1
    cem_temperature: float = 1.0
    cem_momentum: float = 0.0
    action_bounds: Tuple[float, float] = (-1.0, 1.0)
    action_temperature: float = 1.0
    discrete_actions: bool = False

    def validate(self) -> List[str]:
        """Validate field values; return list of error messages (empty if valid)."""
        errors: List[str] = []
        if self.planner_type not in _VALID_PLANNER_TYPES:
            errors.append(
                f"planner_type must be one of {_VALID_PLANNER_TYPES}, "
                f"got '{self.planner_type}'"
            )
        if self.num_rollouts < 1:
            errors.append(f"num_rollouts must be >= 1, got {self.num_rollouts}")
        if self.cem_iterations < 1:
            errors.append(f"cem_iterations must be >= 1, got {self.cem_iterations}")
        if not (0.0 < self.cem_elite_fraction <= 1.0):
            errors.append(
                f"cem_elite_fraction must be in (0, 1], "
                f"got {self.cem_elite_fraction}"
            )
        if self.cem_temperature <= 0.0:
            errors.append(
                f"cem_temperature must be positive, got {self.cem_temperature}"
            )
        if not (0.0 <= self.cem_momentum < 1.0):
            errors.append(
                f"cem_momentum must be in [0, 1), got {self.cem_momentum}"
            )
        if not isinstance(self.action_bounds, (tuple, list)) or len(self.action_bounds) != 2:
            errors.append(
                f"action_bounds must be a 2-tuple (low, high), "
                f"got {self.action_bounds}"
            )
        else:
            lo, hi = self.action_bounds
            if lo >= hi:
                errors.append(
                    f"action_bounds lower ({lo}) must be < upper ({hi})"
                )
        if self.action_temperature <= 0.0:
            errors.append(
                f"action_temperature must be positive, "
                f"got {self.action_temperature}"
            )
        # Check that CEM elite count is at least 1
        elite_count = int(self.num_rollouts * self.cem_elite_fraction)
        if self.planner_type == "cem" and elite_count < 1:
            errors.append(
                f"num_rollouts ({self.num_rollouts}) * cem_elite_fraction "
                f"({self.cem_elite_fraction}) yields 0 elites; "
                f"increase num_rollouts or cem_elite_fraction"
            )
        return errors


# ============================================================================
# 4. AmortizedPolicyConfig
# ============================================================================

@dataclass
class AmortizedPolicyConfig:
    """Configuration for the amortized (distilled) policy network.

    An MLP that imitates the CEM planner for instant action selection.
    Trained online via distillation from planner outputs.

    Attributes:
        enabled: Whether to build the amortized policy.
        hidden_dim: Width of each hidden layer in the policy MLP.
        num_layers: Depth of the policy MLP.
        distill_lr: Learning rate for distillation optimizer.
        distill_mode: ``"soft"`` (MSE), ``"hard"`` (CE), or ``"mixed"``.
        refresh_interval: Re-distill every N planning steps.
    """

    enabled: bool = True
    hidden_dim: int = 512
    num_layers: int = 3
    distill_lr: float = 1e-4
    distill_mode: str = "soft"
    refresh_interval: int = 100

    def validate(self) -> List[str]:
        """Validate field values; return list of error messages (empty if valid)."""
        errors: List[str] = []
        if self.hidden_dim <= 0:
            errors.append(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.num_layers < 1:
            errors.append(f"num_layers must be >= 1, got {self.num_layers}")
        if self.distill_lr <= 0.0:
            errors.append(f"distill_lr must be positive, got {self.distill_lr}")
        if self.distill_mode not in _VALID_DISTILL_MODES:
            errors.append(
                f"distill_mode must be one of {_VALID_DISTILL_MODES}, "
                f"got '{self.distill_mode}'"
            )
        if self.refresh_interval < 1:
            errors.append(
                f"refresh_interval must be >= 1, got {self.refresh_interval}"
            )
        return errors


# ============================================================================
# 5. PreferenceConfig
# ============================================================================

@dataclass
class PreferenceConfig:
    """Configuration for the preference (desired observation) model.

    Goals are encoded as ``p_pref(o)``.  Modes: ``"fixed"`` (static Gaussian),
    ``"learned"`` (jointly optimised network), ``"reward_derived"`` (from reward).

    Attributes:
        mode: Preference strategy.
        preference_dim: Preference embedding dimensionality.
        learn_preferences: Whether preferences receive gradients.
        prior_strength: KL regularisation toward unit-Gaussian prior.
    """

    mode: str = "learned"
    preference_dim: int = 256
    learn_preferences: bool = True
    prior_strength: float = 0.1

    def validate(self) -> List[str]:
        """Validate field values; return list of error messages (empty if valid)."""
        errors: List[str] = []
        if self.mode not in _VALID_PREFERENCE_MODES:
            errors.append(
                f"mode must be one of {_VALID_PREFERENCE_MODES}, "
                f"got '{self.mode}'"
            )
        if self.preference_dim <= 0:
            errors.append(
                f"preference_dim must be positive, got {self.preference_dim}"
            )
        if self.prior_strength < 0.0:
            errors.append(
                f"prior_strength must be non-negative, got {self.prior_strength}"
            )
        if self.mode == "fixed" and self.learn_preferences:
            errors.append(
                "learn_preferences should be False when mode is 'fixed' "
                "(fixed preferences are not learnable)"
            )
        return errors


# ============================================================================
# 6. PyMDPConfig
# ============================================================================

@dataclass
class PyMDPConfig:
    """Configuration for the discrete POMDP backend via pymdp.

    Maintains a parallel discrete model for exact Bayesian inference,
    useful for debugging or small-scale environments.

    Attributes:
        enabled: Activate pymdp backend (False = ignored).
        num_discrete_states: Discrete hidden-state categories.
        num_discrete_obs: Discrete observation categories.
        discretization_method: ``"kmeans"``, ``"uniform"``, or ``"quantile"``.
        regression_tolerance: Max discrete-vs-continuous discrepancy.
    """

    enabled: bool = False
    num_discrete_states: int = 16
    num_discrete_obs: int = 16
    discretization_method: str = "kmeans"
    regression_tolerance: float = 0.1

    def validate(self) -> List[str]:
        """Validate field values; return list of error messages (empty if valid)."""
        errors: List[str] = []
        if self.num_discrete_states < 2:
            errors.append(
                f"num_discrete_states must be >= 2, got {self.num_discrete_states}"
            )
        if self.num_discrete_obs < 2:
            errors.append(
                f"num_discrete_obs must be >= 2, got {self.num_discrete_obs}"
            )
        if self.discretization_method not in _VALID_DISCRETIZATION_METHODS:
            errors.append(
                f"discretization_method must be one of "
                f"{_VALID_DISCRETIZATION_METHODS}, "
                f"got '{self.discretization_method}'"
            )
        if self.regression_tolerance <= 0.0:
            errors.append(
                f"regression_tolerance must be positive, "
                f"got {self.regression_tolerance}"
            )
        return errors


# ============================================================================
# 7. OfflineRLConfig
# ============================================================================

@dataclass
class OfflineRLConfig:
    """Configuration for offline RL dataset training loop.

    Pre-trains the generative model ELBO on collected interaction data
    (D4RL / Minari) with optional KL warm-up and online validation.

    Attributes:
        dataset_name: Offline RL dataset (e.g. ``"pointmaze-medium-v2"``).
        window_size: Observation-action window length for training.
        batch_size: Mini-batch size.
        num_epochs: Training passes through the dataset.
        learning_rate: Peak optimizer learning rate.
        kl_warmup_steps: Steps to linearly ramp KL weight from 0.
        eval_split: Held-out evaluation fraction.
        use_online_validation: Periodically roll out in environment.
    """

    dataset_name: str = "pointmaze-medium-v2"
    window_size: int = 32
    batch_size: int = 256
    num_epochs: int = 50
    learning_rate: float = 3e-4
    kl_warmup_steps: int = 1000
    eval_split: float = 0.2
    use_online_validation: bool = False

    def validate(self) -> List[str]:
        """Validate field values; return list of error messages (empty if valid)."""
        errors: List[str] = []
        if not self.dataset_name or not self.dataset_name.strip():
            errors.append("dataset_name must be a non-empty string")
        if self.window_size < 1:
            errors.append(f"window_size must be >= 1, got {self.window_size}")
        if self.batch_size < 1:
            errors.append(f"batch_size must be >= 1, got {self.batch_size}")
        if self.num_epochs < 1:
            errors.append(f"num_epochs must be >= 1, got {self.num_epochs}")
        if self.learning_rate <= 0.0:
            errors.append(
                f"learning_rate must be positive, got {self.learning_rate}"
            )
        if self.kl_warmup_steps < 0:
            errors.append(
                f"kl_warmup_steps must be non-negative, got {self.kl_warmup_steps}"
            )
        if not (0.0 < self.eval_split < 1.0):
            errors.append(
                f"eval_split must be in (0, 1), got {self.eval_split}"
            )
        return errors


# ============================================================================
# 8. ActiveInferenceFullConfig -- Top-Level Aggregator
# ============================================================================

@dataclass
class ActiveInferenceFullConfig:
    """Aggregate configuration for the entire Active Inference agent.

    Bundles sub-configs with scale presets, JSON serialization, validation,
    and legacy upgrade.  Defaults match ``DecisionConfig`` in ``brain_ai/config.py``.

    Attributes:
        generative: Generative model configuration.
        efe: EFE computation configuration.
        planner: Planner / rollout configuration.
        amortized: Amortized policy configuration.
        preferences: Preference model configuration.
        pymdp: Discrete POMDP backend configuration.
        offline_rl: Offline RL training configuration.
        seed: RNG seed (None = non-deterministic).
        version: Schema version for migration compatibility.
    """

    generative: GenerativeModelConfig = field(default_factory=GenerativeModelConfig)
    efe: EFEConfig = field(default_factory=EFEConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    amortized: AmortizedPolicyConfig = field(default_factory=AmortizedPolicyConfig)
    preferences: PreferenceConfig = field(default_factory=PreferenceConfig)
    pymdp: PyMDPConfig = field(default_factory=PyMDPConfig)
    offline_rl: OfflineRLConfig = field(default_factory=OfflineRLConfig)
    seed: Optional[int] = None
    version: str = _VERSION

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        """Validate all sub-configs and cross-config consistency.

        Returns:
            List of error messages (empty if the configuration is valid).
        """
        errors: List[str] = []

        # --- Sub-config validation ---
        errors.extend(
            [f"[generative] {e}" for e in self.generative.validate()]
        )
        errors.extend([f"[efe] {e}" for e in self.efe.validate()])
        errors.extend([f"[planner] {e}" for e in self.planner.validate()])
        errors.extend([f"[amortized] {e}" for e in self.amortized.validate()])
        errors.extend(
            [f"[preferences] {e}" for e in self.preferences.validate()]
        )
        errors.extend([f"[pymdp] {e}" for e in self.pymdp.validate()])
        errors.extend(
            [f"[offline_rl] {e}" for e in self.offline_rl.validate()]
        )

        # --- Cross-config consistency ---

        # Amortized policy hidden dim should not exceed generative hidden dim
        # by an unreasonable factor (warn, not error)
        if self.amortized.enabled:
            if self.amortized.hidden_dim > self.generative.hidden_dim * 4:
                errors.append(
                    f"[cross] amortized.hidden_dim ({self.amortized.hidden_dim}) "
                    f"is > 4x generative.hidden_dim ({self.generative.hidden_dim}); "
                    f"this may indicate a misconfiguration"
                )

        # Planner type "amortized" requires amortized policy to be enabled
        if (
            self.planner.planner_type == "amortized"
            and not self.amortized.enabled
        ):
            errors.append(
                "[cross] planner_type is 'amortized' but "
                "amortized.enabled is False"
            )

        # Preference dim should be <= obs_dim
        if self.preferences.preference_dim > self.generative.obs_dim:
            errors.append(
                f"[cross] preferences.preference_dim "
                f"({self.preferences.preference_dim}) exceeds "
                f"generative.obs_dim ({self.generative.obs_dim}); "
                f"preference_dim should be <= obs_dim"
            )

        # Discrete action consistency
        if self.planner.discrete_actions and self.generative.latent_type == "continuous":
            # This is allowed but worth flagging
            pass  # Not an error, just different design choice

        # Offline RL window_size should be reasonable relative to planning horizon
        # (not strictly required, but a sanity check)
        if self.offline_rl.window_size < 2:
            errors.append(
                f"[cross] offline_rl.window_size ({self.offline_rl.window_size}) "
                f"should be >= 2 for meaningful sequence training"
            )

        # Seed validation
        if self.seed is not None and self.seed < 0:
            errors.append(f"seed must be non-negative, got {self.seed}")

        return errors

    def validate_or_raise(self) -> None:
        """Validate and raise ``ValueError`` if any errors are found."""
        errors = self.validate()
        if errors:
            msg = "ActiveInferenceFullConfig validation failed:\n"
            msg += "\n".join(f"  - {e}" for e in errors)
            raise ValueError(msg)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-safe dict (tuples encoded as tagged dicts)."""
        raw = asdict(self)
        return _encode_tuples(raw)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ActiveInferenceFullConfig:
        """Deserialize from a dict (handles tagged tuples, ignores unknown keys)."""
        d = _decode_tuples(d)

        # Build each sub-config from its dict, filtering unknown keys
        def _build(cls_type: type, sub_dict: Optional[Dict]) -> Any:
            if sub_dict is None:
                return cls_type()
            valid_keys = {f.name for f in fields(cls_type)}
            filtered = {k: v for k, v in sub_dict.items() if k in valid_keys}
            # Convert list to tuple for tuple-typed fields
            for f in fields(cls_type):
                if f.name in filtered:
                    origin = getattr(f.type, "__origin__", None)
                    type_str = str(f.type) if not isinstance(f.type, str) else f.type
                    if "Tuple" in type_str and isinstance(filtered[f.name], list):
                        filtered[f.name] = tuple(filtered[f.name])
            return cls_type(**filtered)

        generative = _build(GenerativeModelConfig, d.get("generative"))
        efe = _build(EFEConfig, d.get("efe"))
        planner = _build(PlannerConfig, d.get("planner"))
        amortized = _build(AmortizedPolicyConfig, d.get("amortized"))
        preferences = _build(PreferenceConfig, d.get("preferences"))
        pymdp_cfg = _build(PyMDPConfig, d.get("pymdp"))
        offline_rl = _build(OfflineRLConfig, d.get("offline_rl"))

        return cls(
            generative=generative,
            efe=efe,
            planner=planner,
            amortized=amortized,
            preferences=preferences,
            pymdp=pymdp_cfg,
            offline_rl=offline_rl,
            seed=d.get("seed"),
            version=d.get("version", _VERSION),
        )

    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to a JSON file (creates parent dirs)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=False)
        logger.info("Saved ActiveInferenceFullConfig to %s", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> ActiveInferenceFullConfig:
        """Load configuration from a JSON file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        logger.info("Loaded ActiveInferenceFullConfig from %s", path)
        return cls.from_dict(d)

    # ------------------------------------------------------------------
    # Legacy upgrade
    # ------------------------------------------------------------------

    @classmethod
    def from_decision_config(cls, dc: Any) -> ActiveInferenceFullConfig:
        """Upgrade a legacy ``DecisionConfig`` to ``ActiveInferenceFullConfig``."""
        # Extract fields with getattr for robustness
        hidden_dim = getattr(dc, "hidden_dim", 4096)
        planning_horizon = getattr(dc, "planning_horizon", 8)
        epistemic_weight = getattr(dc, "epistemic_weight", 1.0)
        num_policies = getattr(dc, "num_policies", 128)
        use_improved_efe = getattr(dc, "use_improved_efe", True)
        use_empowerment = getattr(dc, "use_empowerment", True)
        empowerment_weight = getattr(dc, "empowerment_weight", 0.1)
        use_amortized_policy = getattr(dc, "use_amortized_policy", True)
        efe_num_samples = getattr(dc, "efe_num_samples", 32)

        # Map legacy -> new structure
        generative = GenerativeModelConfig(
            obs_dim=hidden_dim,
            state_dim=hidden_dim // 16,  # 4096 -> 256
            action_dim=hidden_dim // 32,  # 4096 -> 128
            hidden_dim=hidden_dim // 8,   # 4096 -> 512
        )

        efe = EFEConfig(
            pragmatic_weight=1.0,
            epistemic_weight=epistemic_weight,
            instrumental_weight=empowerment_weight if use_empowerment else 0.0,
            num_samples=efe_num_samples,
            normalize_terms=use_improved_efe,
            term_norm_mode="running" if use_improved_efe else "none",
        )

        planner = PlannerConfig(
            planner_type="cem",
            num_rollouts=num_policies,
        )

        amortized = AmortizedPolicyConfig(
            enabled=use_amortized_policy,
            hidden_dim=hidden_dim // 8,
        )

        preferences = PreferenceConfig(
            mode="learned",
            preference_dim=hidden_dim // 16,
        )

        pymdp_cfg = PyMDPConfig(enabled=False)
        offline_rl = OfflineRLConfig()

        return cls(
            generative=generative,
            efe=efe,
            planner=planner,
            amortized=amortized,
            preferences=preferences,
            pymdp=pymdp_cfg,
            offline_rl=offline_rl,
        )

    # ------------------------------------------------------------------
    # Deep copy
    # ------------------------------------------------------------------

    def copy(self) -> ActiveInferenceFullConfig:
        """Return an independent deep copy of this configuration."""
        return copy.deepcopy(self)

    # ------------------------------------------------------------------
    # Scale presets
    # ------------------------------------------------------------------

    @classmethod
    def minimal(cls) -> ActiveInferenceFullConfig:
        """Tiny configuration for unit tests and CI (~50K params)."""
        return cls(
            generative=GenerativeModelConfig(
                obs_dim=32,
                state_dim=16,
                action_dim=8,
                hidden_dim=32,
                latent_type="continuous",
                num_discrete_states=8,
                encoder_layers=1,
                decoder_layers=1,
                transition_ensemble_size=2,
                log_var_clamp=(-5.0, 2.0),
                kl_weight=1.0,
                ctx_mode="concat",
                ctx_dim=None,
            ),
            efe=EFEConfig(
                pragmatic_weight=1.0,
                epistemic_weight=1.0,
                instrumental_weight=0.0,
                num_samples=4,
                discount_factor=0.99,
                normalize_by_horizon=False,
                normalize_terms=False,
                term_norm_mode="none",
            ),
            planner=PlannerConfig(
                planner_type="random_shooting",
                num_rollouts=8,
                cem_iterations=2,
                cem_elite_fraction=0.25,
                cem_temperature=1.0,
                cem_momentum=0.0,
                action_bounds=(-1.0, 1.0),
                action_temperature=1.0,
                discrete_actions=False,
            ),
            amortized=AmortizedPolicyConfig(
                enabled=False,
                hidden_dim=32,
                num_layers=1,
                distill_lr=1e-3,
                distill_mode="soft",
                refresh_interval=10,
            ),
            preferences=PreferenceConfig(
                mode="fixed",
                preference_dim=16,
                learn_preferences=False,
                prior_strength=0.0,
            ),
            pymdp=PyMDPConfig(
                enabled=False,
                num_discrete_states=4,
                num_discrete_obs=4,
            ),
            offline_rl=OfflineRLConfig(
                window_size=8,
                batch_size=16,
                num_epochs=2,
                learning_rate=1e-3,
                kl_warmup_steps=10,
            ),
            seed=42,
        )

    @classmethod
    def dev(cls) -> ActiveInferenceFullConfig:
        """Development configuration for MNIST-scale experiments (~2M params)."""
        return cls(
            generative=GenerativeModelConfig(
                obs_dim=512,
                state_dim=64,
                action_dim=32,
                hidden_dim=128,
                latent_type="continuous",
                num_discrete_states=16,
                encoder_layers=2,
                decoder_layers=2,
                transition_ensemble_size=3,
                log_var_clamp=(-8.0, 2.0),
                kl_weight=1.0,
                ctx_mode="concat",
                ctx_dim=None,
            ),
            efe=EFEConfig(
                pragmatic_weight=1.0,
                epistemic_weight=1.0,
                instrumental_weight=0.05,
                num_samples=16,
                discount_factor=0.99,
                normalize_by_horizon=True,
                normalize_terms=False,
                term_norm_mode="running",
            ),
            planner=PlannerConfig(
                planner_type="cem",
                num_rollouts=32,
                cem_iterations=3,
                cem_elite_fraction=0.1,
                cem_temperature=1.0,
                cem_momentum=0.0,
                action_bounds=(-1.0, 1.0),
                action_temperature=1.0,
                discrete_actions=False,
            ),
            amortized=AmortizedPolicyConfig(
                enabled=True,
                hidden_dim=128,
                num_layers=2,
                distill_lr=5e-4,
                distill_mode="soft",
                refresh_interval=50,
            ),
            preferences=PreferenceConfig(
                mode="learned",
                preference_dim=64,
                learn_preferences=True,
                prior_strength=0.1,
            ),
            pymdp=PyMDPConfig(
                enabled=False,
                num_discrete_states=8,
                num_discrete_obs=8,
            ),
            offline_rl=OfflineRLConfig(
                window_size=16,
                batch_size=64,
                num_epochs=20,
                learning_rate=3e-4,
                kl_warmup_steps=500,
            ),
        )

    @classmethod
    def production_1b(cls) -> ActiveInferenceFullConfig:
        """1B-scale configuration (~25M decision params, matches DecisionConfig defaults)."""
        return cls(
            generative=GenerativeModelConfig(
                obs_dim=4096,
                state_dim=256,
                action_dim=128,
                hidden_dim=512,
                latent_type="continuous",
                num_discrete_states=32,
                encoder_layers=3,
                decoder_layers=3,
                transition_ensemble_size=5,
                log_var_clamp=(-10.0, 2.0),
                kl_weight=1.0,
                ctx_mode="concat",
                ctx_dim=None,
            ),
            efe=EFEConfig(
                pragmatic_weight=1.0,
                epistemic_weight=1.0,
                instrumental_weight=0.1,
                num_samples=32,
                discount_factor=0.99,
                normalize_by_horizon=True,
                normalize_terms=False,
                term_norm_mode="running",
            ),
            planner=PlannerConfig(
                planner_type="cem",
                num_rollouts=128,
                cem_iterations=5,
                cem_elite_fraction=0.1,
                cem_temperature=1.0,
                cem_momentum=0.0,
                action_bounds=(-1.0, 1.0),
                action_temperature=1.0,
                discrete_actions=False,
            ),
            amortized=AmortizedPolicyConfig(
                enabled=True,
                hidden_dim=512,
                num_layers=3,
                distill_lr=1e-4,
                distill_mode="soft",
                refresh_interval=100,
            ),
            preferences=PreferenceConfig(
                mode="learned",
                preference_dim=256,
                learn_preferences=True,
                prior_strength=0.1,
            ),
            pymdp=PyMDPConfig(enabled=False),
            offline_rl=OfflineRLConfig(
                dataset_name="pointmaze-medium-v2",
                window_size=32,
                batch_size=256,
                num_epochs=50,
                learning_rate=3e-4,
                kl_warmup_steps=1000,
                eval_split=0.2,
                use_online_validation=False,
            ),
        )

    @classmethod
    def production_3b(cls) -> ActiveInferenceFullConfig:
        """3B-scale configuration (~80M decision params, multi-GPU recommended)."""
        return cls(
            generative=GenerativeModelConfig(
                obs_dim=4096,
                state_dim=512,
                action_dim=256,
                hidden_dim=1024,
                latent_type="continuous",
                num_discrete_states=64,
                encoder_layers=4,
                decoder_layers=4,
                transition_ensemble_size=7,
                log_var_clamp=(-10.0, 2.0),
                kl_weight=0.8,
                ctx_mode="cross_attention",
                ctx_dim=4096,
            ),
            efe=EFEConfig(
                pragmatic_weight=1.0,
                epistemic_weight=1.0,
                instrumental_weight=0.1,
                num_samples=64,
                discount_factor=0.995,
                normalize_by_horizon=True,
                normalize_terms=True,
                term_norm_mode="running",
            ),
            planner=PlannerConfig(
                planner_type="cem",
                num_rollouts=256,
                cem_iterations=8,
                cem_elite_fraction=0.1,
                cem_temperature=0.8,
                cem_momentum=0.1,
                action_bounds=(-2.0, 2.0),
                action_temperature=0.8,
                discrete_actions=False,
            ),
            amortized=AmortizedPolicyConfig(
                enabled=True,
                hidden_dim=1024,
                num_layers=4,
                distill_lr=5e-5,
                distill_mode="soft",
                refresh_interval=200,
            ),
            preferences=PreferenceConfig(
                mode="learned",
                preference_dim=512,
                learn_preferences=True,
                prior_strength=0.05,
            ),
            pymdp=PyMDPConfig(enabled=False),
            offline_rl=OfflineRLConfig(
                dataset_name="pointmaze-large-v2",
                window_size=64,
                batch_size=512,
                num_epochs=100,
                learning_rate=1e-4,
                kl_warmup_steps=2000,
                eval_split=0.15,
                use_online_validation=False,
            ),
        )

    @classmethod
    def production_7b(cls) -> ActiveInferenceFullConfig:
        """Full 7B production configuration (~120M decision params, FSDP/DDP)."""
        return cls(
            generative=GenerativeModelConfig(
                obs_dim=4096,
                state_dim=1024,
                action_dim=512,
                hidden_dim=2048,
                latent_type="continuous",
                num_discrete_states=128,
                encoder_layers=6,
                decoder_layers=6,
                transition_ensemble_size=9,
                log_var_clamp=(-10.0, 2.0),
                kl_weight=0.5,
                ctx_mode="cross_attention",
                ctx_dim=4096,
            ),
            efe=EFEConfig(
                pragmatic_weight=1.0,
                epistemic_weight=1.0,
                instrumental_weight=0.1,
                num_samples=128,
                discount_factor=0.997,
                normalize_by_horizon=True,
                normalize_terms=True,
                term_norm_mode="running",
            ),
            planner=PlannerConfig(
                planner_type="cem",
                num_rollouts=512,
                cem_iterations=10,
                cem_elite_fraction=0.05,
                cem_temperature=0.5,
                cem_momentum=0.2,
                action_bounds=(-2.0, 2.0),
                action_temperature=0.5,
                discrete_actions=False,
            ),
            amortized=AmortizedPolicyConfig(
                enabled=True,
                hidden_dim=2048,
                num_layers=6,
                distill_lr=1e-5,
                distill_mode="mixed",
                refresh_interval=500,
            ),
            preferences=PreferenceConfig(
                mode="learned",
                preference_dim=1024,
                learn_preferences=True,
                prior_strength=0.01,
            ),
            pymdp=PyMDPConfig(enabled=False),
            offline_rl=OfflineRLConfig(
                dataset_name="antmaze-large-diverse-v2",
                window_size=128,
                batch_size=1024,
                num_epochs=200,
                learning_rate=5e-5,
                kl_warmup_steps=5000,
                eval_split=0.1,
                use_online_validation=True,
            ),
        )

    # ------------------------------------------------------------------
    # Pretty printing
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable multi-line summary of the configuration."""
        lines: List[str] = [
            "=" * 72,
            "ActiveInferenceFullConfig Summary",
            "=" * 72,
            "",
            f"  Version:  {self.version}",
            f"  Seed:     {self.seed}",
            "",
            "  [Generative Model]",
            f"    obs_dim={self.generative.obs_dim}, "
            f"state_dim={self.generative.state_dim}, "
            f"action_dim={self.generative.action_dim}",
            f"    hidden_dim={self.generative.hidden_dim}, "
            f"latent_type={self.generative.latent_type}",
            f"    encoder_layers={self.generative.encoder_layers}, "
            f"decoder_layers={self.generative.decoder_layers}",
            f"    transition_ensemble_size="
            f"{self.generative.transition_ensemble_size}",
            f"    kl_weight={self.generative.kl_weight}, "
            f"ctx_mode={self.generative.ctx_mode}, "
            f"ctx_dim={self.generative.ctx_dim}",
            "",
            "  [EFE]",
            f"    pragmatic={self.efe.pragmatic_weight}, "
            f"epistemic={self.efe.epistemic_weight}, "
            f"instrumental={self.efe.instrumental_weight}",
            f"    num_samples={self.efe.num_samples}, "
            f"discount={self.efe.discount_factor}",
            f"    normalize_by_horizon={self.efe.normalize_by_horizon}, "
            f"normalize_terms={self.efe.normalize_terms}",
            "",
            "  [Planner]",
            f"    type={self.planner.planner_type}, "
            f"rollouts={self.planner.num_rollouts}",
            f"    cem_iters={self.planner.cem_iterations}, "
            f"elite_frac={self.planner.cem_elite_fraction}",
            f"    action_bounds={self.planner.action_bounds}, "
            f"discrete={self.planner.discrete_actions}",
            "",
            "  [Amortized Policy]",
            f"    enabled={self.amortized.enabled}, "
            f"hidden_dim={self.amortized.hidden_dim}",
            f"    num_layers={self.amortized.num_layers}, "
            f"distill_lr={self.amortized.distill_lr}",
            f"    distill_mode={self.amortized.distill_mode}, "
            f"refresh_interval={self.amortized.refresh_interval}",
            "",
            "  [Preferences]",
            f"    mode={self.preferences.mode}, "
            f"dim={self.preferences.preference_dim}",
            f"    learn={self.preferences.learn_preferences}, "
            f"prior_strength={self.preferences.prior_strength}",
            "",
            "  [PyMDP]",
            f"    enabled={self.pymdp.enabled}",
            f"    states={self.pymdp.num_discrete_states}, "
            f"obs={self.pymdp.num_discrete_obs}",
            f"    method={self.pymdp.discretization_method}",
            "",
            "  [Offline RL]",
            f"    dataset={self.offline_rl.dataset_name}",
            f"    window={self.offline_rl.window_size}, "
            f"batch={self.offline_rl.batch_size}, "
            f"epochs={self.offline_rl.num_epochs}",
            f"    lr={self.offline_rl.learning_rate}, "
            f"kl_warmup={self.offline_rl.kl_warmup_steps}",
            "",
            "=" * 72,
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Compact repr showing scale-relevant dimensions."""
        return (
            f"ActiveInferenceFullConfig("
            f"obs={self.generative.obs_dim}, "
            f"state={self.generative.state_dim}, "
            f"action={self.generative.action_dim}, "
            f"hidden={self.generative.hidden_dim}, "
            f"ensemble={self.generative.transition_ensemble_size}, "
            f"rollouts={self.planner.num_rollouts}, "
            f"v={self.version})"
        )


# ============================================================================
# Convenience factory function
# ============================================================================

def create_config(
    preset: str = "production_1b",
    **overrides: Any,
) -> ActiveInferenceFullConfig:
    """Create config from a named preset with optional ``subconfig__field`` overrides.

    Args:
        preset: ``"minimal"``, ``"dev"``, ``"production_1b"``, ``"production_3b"``,
            or ``"production_7b"``.
        **overrides: ``subconfig__field=value`` pairs (double-underscore separated).
    """
    presets = {
        "minimal": ActiveInferenceFullConfig.minimal,
        "dev": ActiveInferenceFullConfig.dev,
        "production_1b": ActiveInferenceFullConfig.production_1b,
        "production_3b": ActiveInferenceFullConfig.production_3b,
        "production_7b": ActiveInferenceFullConfig.production_7b,
    }
    if preset not in presets:
        raise ValueError(
            f"Unknown preset '{preset}'. Choose from: {list(presets.keys())}"
        )

    cfg = presets[preset]()

    # Apply overrides
    for key, value in overrides.items():
        parts = key.split("__")
        if len(parts) == 1:
            # Top-level field (e.g. seed)
            if not hasattr(cfg, parts[0]):
                raise ValueError(
                    f"ActiveInferenceFullConfig has no field '{parts[0]}'"
                )
            setattr(cfg, parts[0], value)
        elif len(parts) == 2:
            sub_name, field_name = parts
            sub_config = getattr(cfg, sub_name, None)
            if sub_config is None:
                raise ValueError(
                    f"ActiveInferenceFullConfig has no sub-config '{sub_name}'"
                )
            if not hasattr(sub_config, field_name):
                raise ValueError(
                    f"{type(sub_config).__name__} has no field '{field_name}'"
                )
            setattr(sub_config, field_name, value)
        else:
            raise ValueError(
                f"Override key '{key}' has too many parts; "
                f"expected 'field' or 'subconfig__field'"
            )

    return cfg


# ============================================================================
# Self-test block
# ============================================================================

def _run_self_tests() -> None:
    """Execute self-tests covering defaults, presets, serialization, validation, and legacy upgrade."""
    import tempfile

    passed = 0
    failed = 0
    errors_log: List[str] = []

    def _test(name: str, fn: Any) -> None:
        nonlocal passed, failed
        try:
            fn()
            passed += 1
            print(f"  PASS: {name}")
        except Exception as exc:
            failed += 1
            errors_log.append(f"  FAIL: {name} -- {exc}")
            print(f"  FAIL: {name} -- {exc}")

    print("=" * 72)
    print("Active Inference Config -- Self-Tests")
    print("=" * 72)

    # ---------------------------------------------------------------
    # 1. Individual config defaults valid
    # ---------------------------------------------------------------

    def test_generative_defaults_valid():
        cfg = GenerativeModelConfig()
        errs = cfg.validate()
        assert not errs, f"GenerativeModelConfig defaults invalid: {errs}"
        assert cfg.obs_dim == 4096
        assert cfg.state_dim == 256
        assert cfg.action_dim == 128
        assert cfg.hidden_dim == 512
        assert cfg.latent_type == "continuous"
        assert cfg.transition_ensemble_size == 5
        assert isinstance(cfg.log_var_clamp, tuple)
        assert len(cfg.log_var_clamp) == 2
    _test("GenerativeModelConfig defaults valid", test_generative_defaults_valid)

    def test_efe_defaults_valid():
        cfg = EFEConfig()
        errs = cfg.validate()
        assert not errs, f"EFEConfig defaults invalid: {errs}"
        assert cfg.pragmatic_weight == 1.0
        assert cfg.epistemic_weight == 1.0
        assert cfg.instrumental_weight == 0.1
        assert cfg.num_samples == 32
        assert cfg.discount_factor == 0.99
    _test("EFEConfig defaults valid", test_efe_defaults_valid)

    def test_planner_defaults_valid():
        cfg = PlannerConfig()
        errs = cfg.validate()
        assert not errs, f"PlannerConfig defaults invalid: {errs}"
        assert cfg.planner_type == "cem"
        assert cfg.num_rollouts == 128
        assert cfg.cem_iterations == 5
        assert cfg.action_bounds == (-1.0, 1.0)
    _test("PlannerConfig defaults valid", test_planner_defaults_valid)

    def test_amortized_defaults_valid():
        cfg = AmortizedPolicyConfig()
        errs = cfg.validate()
        assert not errs, f"AmortizedPolicyConfig defaults invalid: {errs}"
        assert cfg.enabled is True
        assert cfg.hidden_dim == 512
        assert cfg.distill_mode == "soft"
    _test("AmortizedPolicyConfig defaults valid", test_amortized_defaults_valid)

    def test_preference_defaults_valid():
        cfg = PreferenceConfig()
        errs = cfg.validate()
        assert not errs, f"PreferenceConfig defaults invalid: {errs}"
        assert cfg.mode == "learned"
        assert cfg.preference_dim == 256
        assert cfg.learn_preferences is True
    _test("PreferenceConfig defaults valid", test_preference_defaults_valid)

    def test_pymdp_defaults_valid():
        cfg = PyMDPConfig()
        errs = cfg.validate()
        assert not errs, f"PyMDPConfig defaults invalid: {errs}"
        assert cfg.enabled is False
        assert cfg.discretization_method == "kmeans"
    _test("PyMDPConfig defaults valid", test_pymdp_defaults_valid)

    def test_offline_rl_defaults_valid():
        cfg = OfflineRLConfig()
        errs = cfg.validate()
        assert not errs, f"OfflineRLConfig defaults invalid: {errs}"
        assert cfg.dataset_name == "pointmaze-medium-v2"
        assert cfg.batch_size == 256
        assert cfg.num_epochs == 50
    _test("OfflineRLConfig defaults valid", test_offline_rl_defaults_valid)

    # ---------------------------------------------------------------
    # 2. Field type checks
    # ---------------------------------------------------------------

    def test_field_types():
        cfg = GenerativeModelConfig()
        assert isinstance(cfg.obs_dim, int)
        assert isinstance(cfg.kl_weight, float)
        assert isinstance(cfg.latent_type, str)
        assert isinstance(cfg.log_var_clamp, tuple)
        assert isinstance(cfg.ctx_dim, type(None))
        efe = EFEConfig()
        assert isinstance(efe.normalize_by_horizon, bool)
        assert isinstance(efe.num_samples, int)
        planner = PlannerConfig()
        assert isinstance(planner.action_bounds, tuple)
        assert isinstance(planner.discrete_actions, bool)
    _test("Field types correct", test_field_types)

    # ---------------------------------------------------------------
    # 3. All 5 presets create valid configs
    # ---------------------------------------------------------------

    def test_preset_minimal():
        cfg = ActiveInferenceFullConfig.minimal()
        errs = cfg.validate()
        assert not errs, f"minimal() invalid: {errs}"
        assert cfg.generative.obs_dim == 32
        assert cfg.generative.state_dim == 16
        assert cfg.generative.action_dim == 8
        assert cfg.generative.hidden_dim == 32
        assert cfg.generative.transition_ensemble_size == 2
        assert cfg.planner.num_rollouts == 8
    _test("Preset minimal() valid", test_preset_minimal)

    def test_preset_dev():
        cfg = ActiveInferenceFullConfig.dev()
        errs = cfg.validate()
        assert not errs, f"dev() invalid: {errs}"
        assert cfg.generative.obs_dim == 512
        assert cfg.generative.state_dim == 64
        assert cfg.generative.action_dim == 32
        assert cfg.generative.hidden_dim == 128
        assert cfg.generative.transition_ensemble_size == 3
        assert cfg.planner.num_rollouts == 32
    _test("Preset dev() valid", test_preset_dev)

    def test_preset_production_1b():
        cfg = ActiveInferenceFullConfig.production_1b()
        errs = cfg.validate()
        assert not errs, f"production_1b() invalid: {errs}"
        assert cfg.generative.obs_dim == 4096
        assert cfg.generative.state_dim == 256
        assert cfg.generative.action_dim == 128
        assert cfg.generative.hidden_dim == 512
        assert cfg.generative.transition_ensemble_size == 5
        assert cfg.planner.num_rollouts == 128
    _test("Preset production_1b() valid", test_preset_production_1b)

    def test_preset_production_3b():
        cfg = ActiveInferenceFullConfig.production_3b()
        errs = cfg.validate()
        assert not errs, f"production_3b() invalid: {errs}"
        assert cfg.generative.state_dim > 256  # Larger than 1b
        assert cfg.generative.hidden_dim > 512
    _test("Preset production_3b() valid", test_preset_production_3b)

    def test_preset_production_7b():
        cfg = ActiveInferenceFullConfig.production_7b()
        errs = cfg.validate()
        assert not errs, f"production_7b() invalid: {errs}"
        assert cfg.generative.state_dim >= 1024  # Largest
        assert cfg.generative.hidden_dim >= 2048
        assert cfg.planner.num_rollouts >= 512
    _test("Preset production_7b() valid", test_preset_production_7b)

    # ---------------------------------------------------------------
    # 4. Aggregator sub-config accessibility
    # ---------------------------------------------------------------

    def test_subconfig_access():
        cfg = ActiveInferenceFullConfig.production_1b()
        # All sub-configs should be accessible and of correct type
        assert isinstance(cfg.generative, GenerativeModelConfig)
        assert isinstance(cfg.efe, EFEConfig)
        assert isinstance(cfg.planner, PlannerConfig)
        assert isinstance(cfg.amortized, AmortizedPolicyConfig)
        assert isinstance(cfg.preferences, PreferenceConfig)
        assert isinstance(cfg.pymdp, PyMDPConfig)
        assert isinstance(cfg.offline_rl, OfflineRLConfig)
        # Modify sub-config field
        cfg.generative.obs_dim = 2048
        assert cfg.generative.obs_dim == 2048
    _test("Aggregator sub-config accessibility", test_subconfig_access)

    # ---------------------------------------------------------------
    # 5. Serialization round-trip: to_dict / from_dict
    # ---------------------------------------------------------------

    def test_serialization_roundtrip_dict():
        original = ActiveInferenceFullConfig.production_1b()
        d = original.to_dict()
        assert isinstance(d, dict)
        restored = ActiveInferenceFullConfig.from_dict(d)
        # Compare key fields
        assert restored.generative.obs_dim == original.generative.obs_dim
        assert restored.generative.state_dim == original.generative.state_dim
        assert restored.generative.log_var_clamp == original.generative.log_var_clamp
        assert restored.efe.pragmatic_weight == original.efe.pragmatic_weight
        assert restored.planner.action_bounds == original.planner.action_bounds
        assert restored.amortized.enabled == original.amortized.enabled
        assert restored.preferences.mode == original.preferences.mode
        assert restored.pymdp.enabled == original.pymdp.enabled
        assert restored.offline_rl.dataset_name == original.offline_rl.dataset_name
        assert restored.version == original.version
        assert restored.seed == original.seed
    _test("Serialization round-trip (to_dict/from_dict)", test_serialization_roundtrip_dict)

    def test_serialization_roundtrip_json_file():
        original = ActiveInferenceFullConfig.dev()
        original.seed = 123
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            tmp_path = f.name
        try:
            original.save(tmp_path)
            restored = ActiveInferenceFullConfig.load(tmp_path)
            assert restored.generative.obs_dim == original.generative.obs_dim
            assert restored.seed == 123
            assert restored.planner.action_bounds == original.planner.action_bounds
            assert restored.generative.log_var_clamp == original.generative.log_var_clamp
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    _test("Serialization round-trip (save/load JSON)", test_serialization_roundtrip_json_file)

    # ---------------------------------------------------------------
    # 6. Legacy upgrade from DecisionConfig
    # ---------------------------------------------------------------

    def test_legacy_upgrade():
        # Simulate a DecisionConfig-like object
        class MockDecisionConfig:
            hidden_dim = 4096
            planning_horizon = 8
            epistemic_weight = 1.0
            num_policies = 128
            use_improved_efe = True
            use_empowerment = True
            empowerment_weight = 0.1
            use_amortized_policy = True
            efe_num_samples = 32

        legacy = MockDecisionConfig()
        modern = ActiveInferenceFullConfig.from_decision_config(legacy)
        errs = modern.validate()
        assert not errs, f"Legacy upgrade produced invalid config: {errs}"
        assert modern.generative.obs_dim == 4096
        assert modern.generative.state_dim == 256  # 4096 // 16
        assert modern.generative.action_dim == 128  # 4096 // 32
        assert modern.efe.epistemic_weight == 1.0
        assert modern.efe.instrumental_weight == 0.1
        assert modern.efe.num_samples == 32
        assert modern.planner.num_rollouts == 128
        assert modern.amortized.enabled is True
        assert modern.efe.normalize_terms is True  # use_improved_efe=True
    _test("Legacy upgrade from DecisionConfig", test_legacy_upgrade)

    def test_legacy_upgrade_no_empowerment():
        class MockDecisionConfig:
            hidden_dim = 2048
            planning_horizon = 4
            epistemic_weight = 0.5
            num_policies = 64
            use_improved_efe = False
            use_empowerment = False
            empowerment_weight = 0.0
            use_amortized_policy = False
            efe_num_samples = 16

        legacy = MockDecisionConfig()
        modern = ActiveInferenceFullConfig.from_decision_config(legacy)
        errs = modern.validate()
        assert not errs, f"Legacy upgrade (no empowerment) invalid: {errs}"
        assert modern.generative.obs_dim == 2048
        assert modern.efe.instrumental_weight == 0.0
        assert modern.amortized.enabled is False
        assert modern.efe.normalize_terms is False
    _test("Legacy upgrade (no empowerment)", test_legacy_upgrade_no_empowerment)

    # ---------------------------------------------------------------
    # 7. Validation catches invalid values
    # ---------------------------------------------------------------

    def test_validation_catches_negative_obs_dim():
        cfg = GenerativeModelConfig(obs_dim=-1)
        errs = cfg.validate()
        assert any("obs_dim" in e for e in errs), f"Expected obs_dim error: {errs}"
    _test("Validation catches negative obs_dim", test_validation_catches_negative_obs_dim)

    def test_validation_catches_invalid_latent_type():
        cfg = GenerativeModelConfig(latent_type="banana")
        errs = cfg.validate()
        assert any("latent_type" in e for e in errs), f"Expected latent_type error: {errs}"
    _test("Validation catches invalid latent_type", test_validation_catches_invalid_latent_type)

    def test_validation_catches_all_zero_efe_weights():
        cfg = EFEConfig(
            pragmatic_weight=0.0,
            epistemic_weight=0.0,
            instrumental_weight=0.0,
        )
        errs = cfg.validate()
        assert any("non-zero" in e for e in errs), f"Expected all-zero EFE error: {errs}"
    _test("Validation catches all-zero EFE weights", test_validation_catches_all_zero_efe_weights)

    def test_validation_catches_invalid_planner_type():
        cfg = PlannerConfig(planner_type="magic")
        errs = cfg.validate()
        assert any("planner_type" in e for e in errs), f"Expected planner_type error: {errs}"
    _test("Validation catches invalid planner_type", test_validation_catches_invalid_planner_type)

    def test_validation_catches_invalid_discount_factor():
        cfg = EFEConfig(discount_factor=1.5)
        errs = cfg.validate()
        assert any("discount_factor" in e for e in errs), f"Expected discount error: {errs}"
    _test("Validation catches invalid discount_factor", test_validation_catches_invalid_discount_factor)

    def test_validation_catches_fixed_preference_with_learn():
        cfg = PreferenceConfig(mode="fixed", learn_preferences=True)
        errs = cfg.validate()
        assert any("fixed" in e.lower() for e in errs), f"Expected fixed-pref error: {errs}"
    _test("Validation catches fixed preference with learn=True", test_validation_catches_fixed_preference_with_learn)

    def test_cross_config_validation_amortized_planner():
        cfg = ActiveInferenceFullConfig.dev()
        cfg.planner.planner_type = "amortized"
        cfg.amortized.enabled = False
        errs = cfg.validate()
        assert any("amortized" in e.lower() for e in errs), (
            f"Expected cross-config amortized error: {errs}"
        )
    _test("Cross-config: amortized planner without policy", test_cross_config_validation_amortized_planner)

    def test_validate_or_raise():
        cfg = ActiveInferenceFullConfig.minimal()
        cfg.generative.obs_dim = -1
        try:
            cfg.validate_or_raise()
            assert False, "Should have raised ValueError"
        except ValueError as exc:
            assert "obs_dim" in str(exc)
    _test("validate_or_raise raises on invalid config", test_validate_or_raise)

    # ---------------------------------------------------------------
    # 8. Factory function
    # ---------------------------------------------------------------

    def test_create_config_basic():
        cfg = create_config("dev")
        assert cfg.generative.obs_dim == 512
        errs = cfg.validate()
        assert not errs, f"create_config('dev') invalid: {errs}"
    _test("create_config basic preset", test_create_config_basic)

    def test_create_config_with_overrides():
        cfg = create_config("dev", generative__obs_dim=256, seed=99)
        assert cfg.generative.obs_dim == 256
        assert cfg.seed == 99
    _test("create_config with overrides", test_create_config_with_overrides)

    def test_create_config_invalid_preset():
        try:
            create_config("nonexistent")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
    _test("create_config rejects invalid preset", test_create_config_invalid_preset)

    # ---------------------------------------------------------------
    # 9. Deep copy
    # ---------------------------------------------------------------

    def test_deep_copy():
        original = ActiveInferenceFullConfig.dev()
        copied = original.copy()
        copied.generative.obs_dim = 9999
        assert original.generative.obs_dim == 512  # Unchanged
        assert copied.generative.obs_dim == 9999
    _test("Deep copy independence", test_deep_copy)

    # ---------------------------------------------------------------
    # 10. Summary / repr
    # ---------------------------------------------------------------

    def test_summary_and_repr():
        cfg = ActiveInferenceFullConfig.production_1b()
        summary = cfg.summary()
        assert "ActiveInferenceFullConfig" in summary
        assert "Generative Model" in summary
        assert "EFE" in summary
        r = repr(cfg)
        assert "obs=4096" in r
        assert "state=256" in r
    _test("summary() and __repr__", test_summary_and_repr)

    # ---------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------
    print()
    print("-" * 72)
    total = passed + failed
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if errors_log:
        print()
        for e in errors_log:
            print(e)
    print("-" * 72)

    if failed > 0:
        sys.exit(1)
    else:
        print("All tests passed.")


# ============================================================================
# Main entry point
# ============================================================================

if __name__ == "__main__":
    _run_self_tests()
