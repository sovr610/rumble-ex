"""
SearchConfig and LRFinderConfig dataclasses for hyperparameter search.

Provides validated configuration for search strategies, LR finding,
and phase-specific search space presets for the brain_ai system.

No external dependencies beyond Python stdlib.
"""

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple
import json
import copy

# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

VALID_STRATEGIES = ("grid", "random", "bayesian", "hyperband")
VALID_SAMPLERS = ("tpe", "cmaes", "random")
VALID_DIRECTIONS = ("maximize", "minimize")
VALID_PRUNERS = ("asha", "hyperband", "median", "none")


@dataclass
class SearchConfig:
    """Configuration for hyperparameter search."""

    strategy: str = "bayesian"
    n_trials: int = 100
    metric: str = "val_accuracy"
    direction: str = "maximize"

    # Bayesian
    sampler: str = "tpe"
    gamma: float = 0.25
    n_candidates: int = 100
    n_startup_trials: int = 10

    # Early stopping
    enable_pruning: bool = True
    pruner: str = "asha"
    min_resource: int = 1
    reduction_factor: int = 3
    max_resource: int = 81

    # Storage
    study_dir: str = "studies/"
    study_name: str = "brain_ai_search"

    # Parallelism
    n_jobs: int = 1
    timeout_per_trial: int = 3600

    # Reproducibility
    seed: Optional[int] = 42

    def validate(self) -> None:
        """Validate configuration values."""
        if self.strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy '{self.strategy}'. Must be one of {VALID_STRATEGIES}"
            )
        if self.direction not in VALID_DIRECTIONS:
            raise ValueError(
                f"Invalid direction '{self.direction}'. Must be one of {VALID_DIRECTIONS}"
            )
        if self.sampler not in VALID_SAMPLERS:
            raise ValueError(
                f"Invalid sampler '{self.sampler}'. Must be one of {VALID_SAMPLERS}"
            )
        if self.pruner not in VALID_PRUNERS:
            raise ValueError(
                f"Invalid pruner '{self.pruner}'. Must be one of {VALID_PRUNERS}"
            )
        if self.n_trials <= 0:
            raise ValueError(f"n_trials must be > 0, got {self.n_trials}")
        if self.min_resource < 0:
            raise ValueError(f"min_resource must be >= 0, got {self.min_resource}")
        if self.reduction_factor < 2:
            raise ValueError(
                f"reduction_factor must be >= 2, got {self.reduction_factor}"
            )
        if not 0.0 < self.gamma < 1.0:
            raise ValueError(f"gamma must be in (0, 1), got {self.gamma}")
        if self.n_candidates <= 0:
            raise ValueError(f"n_candidates must be > 0, got {self.n_candidates}")
        if self.n_startup_trials < 0:
            raise ValueError(
                f"n_startup_trials must be >= 0, got {self.n_startup_trials}"
            )
        if self.max_resource < self.min_resource:
            raise ValueError(
                f"max_resource ({self.max_resource}) must be >= "
                f"min_resource ({self.min_resource})"
            )
        if self.timeout_per_trial <= 0:
            raise ValueError(
                f"timeout_per_trial must be > 0, got {self.timeout_per_trial}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SearchConfig":
        """Deserialize from dictionary."""
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "SearchConfig":
        return cls.from_dict(json.loads(s))


@dataclass
class LRFinderConfig:
    """Configuration for learning rate range test."""

    start_lr: float = 1e-7
    end_lr: float = 10.0
    num_steps: int = 100
    smooth_factor: float = 0.05
    divergence_threshold: float = 5.0

    def validate(self) -> None:
        """Validate configuration values."""
        if self.start_lr <= 0:
            raise ValueError(f"start_lr must be > 0, got {self.start_lr}")
        if self.end_lr <= 0:
            raise ValueError(f"end_lr must be > 0, got {self.end_lr}")
        if self.start_lr >= self.end_lr:
            raise ValueError(
                f"start_lr ({self.start_lr}) must be < end_lr ({self.end_lr})"
            )
        if self.num_steps <= 0:
            raise ValueError(f"num_steps must be > 0, got {self.num_steps}")
        if not 0.0 <= self.smooth_factor <= 1.0:
            raise ValueError(
                f"smooth_factor must be in [0, 1], got {self.smooth_factor}"
            )
        if self.divergence_threshold <= 1.0:
            raise ValueError(
                f"divergence_threshold must be > 1.0, got {self.divergence_threshold}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LRFinderConfig":
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Phase-specific search space presets
# ---------------------------------------------------------------------------


class PhasePresets:
    """Phase-specific search space parameter definitions.

    Each preset returns a list of dicts with keys:
        name, type, low, high, log, choices, condition
    """

    @staticmethod
    def _float(name: str, low: float, high: float, log: bool = False,
               condition: Optional[str] = None) -> Dict[str, Any]:
        return dict(name=name, type="float", low=low, high=high,
                    log=log, choices=None, condition=condition)

    @staticmethod
    def _int(name: str, low: int, high: int,
             condition: Optional[str] = None) -> Dict[str, Any]:
        return dict(name=name, type="int", low=low, high=high,
                    log=False, choices=None, condition=condition)

    @staticmethod
    def _cat(name: str, choices: List[Any],
             condition: Optional[str] = None) -> Dict[str, Any]:
        return dict(name=name, type="categorical", low=None, high=None,
                    log=False, choices=choices, condition=condition)

    @classmethod
    def phase1_snn(cls) -> List[Dict[str, Any]]:
        """SNN Core training hyperparameters."""
        return [
            cls._float("learning_rate", 1e-4, 1e-2, log=True),
            cls._float("snn.beta", 0.8, 0.99),
            cls._cat("snn.surrogate", ["atan", "fast_sigmoid", "straight_through"]),
            cls._float("snn.surrogate_alpha", 0.5, 5.0),
            cls._float("snn.dropout", 0.0, 0.3),
            cls._float("weight_decay", 1e-4, 1e-1, log=True),
            cls._int("snn.num_timesteps", 10, 100),
            cls._float("snn.spike_rate_target", 0.05, 0.3),
            cls._float("snn.spike_rate_weight", 1e-3, 1e-1, log=True),
            cls._int("warmup_steps", 500, 5000),
        ]

    @classmethod
    def phase2_encoders(cls) -> List[Dict[str, Any]]:
        """Modality encoder training hyperparameters."""
        return [
            cls._float("learning_rate", 1e-5, 1e-3, log=True),
            cls._float("weight_decay", 0.01, 0.3, log=True),
            cls._int("warmup_steps", 1000, 10000),
            cls._float("grad_clip", 0.5, 2.0),
            cls._int("encoder.vision_num_layers", 12, 32),
            cls._cat("encoder.vision_patch_size", [8, 16, 32]),
            cls._int("encoder.text_num_layers", 12, 48),
        ]

    @classmethod
    def phase3_htm(cls) -> List[Dict[str, Any]]:
        """HTM training hyperparameters."""
        return [
            cls._float("learning_rate", 1e-4, 5e-3, log=True),
            cls._int("htm.column_count", 512, 16384),
            cls._int("htm.cells_per_column", 8, 64),
            cls._float("htm.sparsity", 0.01, 0.1),
            cls._float("htm.permanence_inc", 0.01, 0.3),
            cls._float("htm.permanence_dec", 0.01, 0.3),
            cls._int("htm.activation_threshold", 5, 25),
            cls._int("htm.lstm_num_layers", 2, 8),
            cls._int("htm.reflex_num_tables", 4, 16,
                      condition="htm.use_reflex_memory == True"),
            cls._int("htm.reflex_bits_per_hash", 8, 16,
                      condition="htm.use_reflex_memory == True"),
        ]

    @classmethod
    def phase4_workspace(cls) -> List[Dict[str, Any]]:
        """Global Workspace training hyperparameters."""
        return [
            cls._float("learning_rate", 1e-5, 1e-3, log=True),
            cls._int("workspace.num_heads", 8, 32),
            cls._int("workspace.capacity_limit", 3, 12),
            cls._int("workspace.memory_num_layers", 4, 12),
            cls._int("workspace.selection_rounds", 1, 5,
                      condition="workspace.use_selection_broadcast == True"),
            cls._float("workspace.ignition_threshold", 0.1, 0.7,
                        condition="workspace.use_selection_broadcast == True"),
            cls._int("workspace.broadcast_iterations", 1, 4,
                      condition="workspace.use_selection_broadcast == True"),
            cls._float("workspace.broadcast_decay", 0.7, 0.99,
                        condition="workspace.use_selection_broadcast == True"),
        ]

    @classmethod
    def phase5_decision(cls) -> List[Dict[str, Any]]:
        """Active Inference / Decision training hyperparameters."""
        return [
            cls._float("learning_rate", 1e-5, 1e-3, log=True),
            cls._int("decision.planning_horizon", 1, 16),
            cls._float("decision.epistemic_weight", 0.1, 5.0, log=True),
            cls._int("decision.num_policies", 16, 256),
            cls._float("decision.empowerment_weight", 0.01, 1.0, log=True,
                        condition="decision.use_empowerment == True"),
            cls._int("decision.efe_num_samples", 8, 64,
                      condition="decision.use_improved_efe == True"),
        ]

    @classmethod
    def phase6_reasoning(cls) -> List[Dict[str, Any]]:
        """Reasoning training hyperparameters."""
        return [
            cls._float("learning_rate", 1e-5, 1e-3, log=True),
            cls._float("reasoning.confidence_threshold", 0.5, 0.95),
            cls._int("reasoning.num_reasoning_steps", 4, 32),
            cls._int("reasoning.system2_layers", 4, 12),
            cls._cat("reasoning.logic_type", ["product", "godel", "lukasiewicz"]),
            cls._int("reasoning.ltn_embedding_dim", 64, 256,
                      condition="reasoning.use_ltn == True"),
            cls._int("reasoning.ltn_num_layers", 2, 6,
                      condition="reasoning.use_ltn == True"),
            cls._float("reasoning.ltn_p_forall", 1.0, 4.0,
                        condition="reasoning.use_ltn == True"),
        ]

    @classmethod
    def phase7_meta(cls) -> List[Dict[str, Any]]:
        """Meta-learning training hyperparameters."""
        return [
            cls._float("meta.inner_lr", 0.001, 0.5, log=True),
            cls._float("meta.outer_lr", 1e-5, 1e-3, log=True),
            cls._int("meta.num_inner_steps", 1, 20),
            cls._float("meta.trace_decay", 0.9, 0.999),
            cls._float("meta.ewc_lambda", 10.0, 10000.0, log=True),
            cls._float("meta.gradient_clipping", 0.5, 5.0),
            cls._int("meta.task_embedding_dim", 64, 512,
                      condition="meta.use_task2vec == True"),
        ]

    @classmethod
    def all_presets(cls) -> Dict[str, List[Dict[str, Any]]]:
        """Return all phase presets."""
        return {
            "phase1_snn": cls.phase1_snn(),
            "phase2_encoders": cls.phase2_encoders(),
            "phase3_htm": cls.phase3_htm(),
            "phase4_workspace": cls.phase4_workspace(),
            "phase5_decision": cls.phase5_decision(),
            "phase6_reasoning": cls.phase6_reasoning(),
            "phase7_meta": cls.phase7_meta(),
        }

    @classmethod
    def get_preset(cls, phase: int) -> List[Dict[str, Any]]:
        """Get preset by phase number (1-7)."""
        method_map = {
            1: cls.phase1_snn,
            2: cls.phase2_encoders,
            3: cls.phase3_htm,
            4: cls.phase4_workspace,
            5: cls.phase5_decision,
            6: cls.phase6_reasoning,
            7: cls.phase7_meta,
        }
        if phase not in method_map:
            raise ValueError(f"Phase must be 1-7, got {phase}")
        return method_map[phase]()


# ---------------------------------------------------------------------------
# Helper: build SearchConfig for specific use cases
# ---------------------------------------------------------------------------


def make_quick_search_config(
    n_trials: int = 20,
    strategy: str = "random",
    metric: str = "val_loss",
    direction: str = "minimize",
    seed: int = 42,
) -> SearchConfig:
    """Create a quick search configuration for development."""
    cfg = SearchConfig(
        strategy=strategy,
        n_trials=n_trials,
        metric=metric,
        direction=direction,
        enable_pruning=False,
        seed=seed,
    )
    cfg.validate()
    return cfg


def make_production_search_config(
    n_trials: int = 100,
    metric: str = "val_accuracy",
    direction: str = "maximize",
    study_name: str = "brain_ai_production",
    seed: int = 42,
) -> SearchConfig:
    """Create a production search configuration with TPE + ASHA."""
    cfg = SearchConfig(
        strategy="bayesian",
        n_trials=n_trials,
        metric=metric,
        direction=direction,
        sampler="tpe",
        enable_pruning=True,
        pruner="asha",
        min_resource=1,
        reduction_factor=3,
        max_resource=81,
        study_name=study_name,
        seed=seed,
    )
    cfg.validate()
    return cfg


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------


def _run_tests():
    passed = 0
    failed = 0

    def check(name: str, condition: bool, msg: str = ""):
        nonlocal passed, failed
        if condition:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL: {name} — {msg}")

    print("=== SearchConfig Tests ===")

    # SC-01: Valid config
    cfg = SearchConfig()
    try:
        cfg.validate()
        check("SC-01 valid default config", True)
    except Exception as e:
        check("SC-01 valid default config", False, str(e))

    # SC-02: Invalid strategy
    cfg2 = SearchConfig(strategy="invalid")
    try:
        cfg2.validate()
        check("SC-02 invalid strategy", False, "should have raised")
    except ValueError:
        check("SC-02 invalid strategy", True)

    # SC-03: Invalid direction
    cfg3 = SearchConfig(direction="sideways")
    try:
        cfg3.validate()
        check("SC-03 invalid direction", False, "should have raised")
    except ValueError:
        check("SC-03 invalid direction", True)

    # SC-04: n_trials <= 0
    cfg4 = SearchConfig(n_trials=-1)
    try:
        cfg4.validate()
        check("SC-04 negative n_trials", False, "should have raised")
    except ValueError:
        check("SC-04 negative n_trials", True)

    # SC-05: Invalid sampler
    cfg5 = SearchConfig(sampler="magic")
    try:
        cfg5.validate()
        check("SC-05 invalid sampler", False, "should have raised")
    except ValueError:
        check("SC-05 invalid sampler", True)

    # SC-06: Invalid pruner
    cfg6 = SearchConfig(pruner="none_such")
    try:
        cfg6.validate()
        check("SC-06 invalid pruner", False, "should have raised")
    except ValueError:
        check("SC-06 invalid pruner", True)

    # SC-07: Negative min_resource
    cfg7 = SearchConfig(min_resource=-1)
    try:
        cfg7.validate()
        check("SC-07 negative min_resource", False, "should have raised")
    except ValueError:
        check("SC-07 negative min_resource", True)

    # SC-08: reduction_factor < 2
    cfg8 = SearchConfig(reduction_factor=1)
    try:
        cfg8.validate()
        check("SC-08 reduction_factor < 2", False, "should have raised")
    except ValueError:
        check("SC-08 reduction_factor < 2", True)

    print("\n=== LRFinderConfig Tests ===")

    # SC-09: Valid LR config
    lr_cfg = LRFinderConfig()
    try:
        lr_cfg.validate()
        check("SC-09 valid default LR config", True)
    except Exception as e:
        check("SC-09 valid default LR config", False, str(e))

    # SC-10: start_lr >= end_lr
    lr10 = LRFinderConfig(start_lr=1.0, end_lr=0.1)
    try:
        lr10.validate()
        check("SC-10 start_lr >= end_lr", False, "should have raised")
    except ValueError:
        check("SC-10 start_lr >= end_lr", True)

    # SC-11: start_lr <= 0
    lr11 = LRFinderConfig(start_lr=0)
    try:
        lr11.validate()
        check("SC-11 start_lr <= 0", False, "should have raised")
    except ValueError:
        check("SC-11 start_lr <= 0", True)

    # SC-12: num_steps <= 0
    lr12 = LRFinderConfig(num_steps=0)
    try:
        lr12.validate()
        check("SC-12 num_steps <= 0", False, "should have raised")
    except ValueError:
        check("SC-12 num_steps <= 0", True)

    # SC-13: smooth_factor out of range
    lr13 = LRFinderConfig(smooth_factor=2.0)
    try:
        lr13.validate()
        check("SC-13 smooth_factor out of range", False, "should have raised")
    except ValueError:
        check("SC-13 smooth_factor out of range", True)

    # SC-14: divergence_threshold <= 1
    lr14 = LRFinderConfig(divergence_threshold=0.5)
    try:
        lr14.validate()
        check("SC-14 divergence_threshold <= 1", False, "should have raised")
    except ValueError:
        check("SC-14 divergence_threshold <= 1", True)

    print("\n=== Serialization Tests ===")

    # to_dict / from_dict round-trip
    cfg_orig = SearchConfig(n_trials=50, strategy="random", seed=123)
    d = cfg_orig.to_dict()
    cfg_back = SearchConfig.from_dict(d)
    check("Serialization round-trip", cfg_back.n_trials == 50 and cfg_back.strategy == "random")

    # to_json / from_json
    j = cfg_orig.to_json()
    cfg_json = SearchConfig.from_json(j)
    check("JSON round-trip", cfg_json.seed == 123)

    # LRFinderConfig serialization
    lr_orig = LRFinderConfig(start_lr=1e-6, end_lr=5.0)
    ld = lr_orig.to_dict()
    lr_back = LRFinderConfig.from_dict(ld)
    check("LR config round-trip", abs(lr_back.start_lr - 1e-6) < 1e-10)

    # from_dict with extra keys
    d_extra = {"n_trials": 10, "unknown_key": "ignored"}
    cfg_extra = SearchConfig.from_dict(d_extra)
    check("from_dict ignores unknown keys", cfg_extra.n_trials == 10)

    print("\n=== Phase Preset Tests ===")

    # SC-15: Phase 1 preset
    p1 = PhasePresets.phase1_snn()
    p1_names = [p["name"] for p in p1]
    check("SC-15 Phase 1 has lr", "learning_rate" in p1_names)
    check("SC-15 Phase 1 has beta", "snn.beta" in p1_names)
    check("SC-15 Phase 1 has surrogate", "snn.surrogate" in p1_names)

    # SC-16: Phase 3 preset
    p3 = PhasePresets.phase3_htm()
    p3_names = [p["name"] for p in p3]
    check("SC-16 Phase 3 has column_count", "htm.column_count" in p3_names)
    check("SC-16 Phase 3 has sparsity", "htm.sparsity" in p3_names)

    # SC-17: Phase 7 preset
    p7 = PhasePresets.phase7_meta()
    p7_names = [p["name"] for p in p7]
    check("SC-17 Phase 7 has inner_lr", "meta.inner_lr" in p7_names)
    check("SC-17 Phase 7 has outer_lr", "meta.outer_lr" in p7_names)

    # SC-18: All presets valid (check they return lists)
    all_presets = PhasePresets.all_presets()
    check("SC-18 all 7 presets exist", len(all_presets) == 7)
    for name, preset in all_presets.items():
        check(f"SC-18 {name} is list", isinstance(preset, list) and len(preset) > 0)

    # SC-19: Preset search spaces have correct structure
    for name, preset in all_presets.items():
        for param in preset:
            check(
                f"SC-19 {name}/{param['name']} has type",
                param["type"] in ("float", "int", "categorical"),
                f"type={param.get('type')}",
            )

    # get_preset by phase number
    for phase in range(1, 8):
        p = PhasePresets.get_preset(phase)
        check(f"get_preset({phase})", isinstance(p, list) and len(p) > 0)

    # Invalid phase number
    try:
        PhasePresets.get_preset(0)
        check("Invalid phase 0", False, "should have raised")
    except ValueError:
        check("Invalid phase 0", True)

    try:
        PhasePresets.get_preset(8)
        check("Invalid phase 8", False, "should have raised")
    except ValueError:
        check("Invalid phase 8", True)

    print("\n=== Helper Function Tests ===")

    # make_quick_search_config
    quick = make_quick_search_config(n_trials=10)
    quick.validate()
    check("quick config valid", quick.n_trials == 10 and quick.strategy == "random")

    # make_production_search_config
    prod = make_production_search_config(n_trials=200)
    prod.validate()
    check("production config valid", prod.strategy == "bayesian" and prod.pruner == "asha")

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    if failed == 0:
        print("ALL TESTS PASSED")
    return failed == 0


if __name__ == "__main__":
    import sys
    success = _run_tests()
    sys.exit(0 if success else 1)
