"""
sleep_config_template.py -- Configuration Dataclass for Sleep Consolidation Cycle

Provides the central SleepConfig dataclass that parameterises all aspects of the
sleep consolidation cycle: NREM replay, synaptic homeostasis, REM generative replay,
systems consolidation, and scheduling.

Classes:
    SleepConfig          -- Central configuration with presets and validation.
    ConsolidationResult  -- Result container for a completed consolidation cycle.
    PhaseResult          -- Result container for a single sub-phase (NREM or REM).
    HomeostasisResult    -- Result container for synaptic homeostasis.
    TransferResult       -- Result container for systems consolidation transfer.

Self-contained: no brain_ai imports required.
"""

from __future__ import annotations

import copy
import json
import math
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# =========================================================================
# SleepConfig
# =========================================================================

@dataclass
class SleepConfig:
    """Central configuration for the sleep consolidation cycle.

    Parameters are organised into sections matching the sub-phases of
    consolidation.  Use the class-method presets (``minimal``, ``dev``,
    ``production``) for common configurations.
    """

    # -- NREM phase --------------------------------------------------------
    nrem_replay_steps: int = 100
    compression_ratio: float = 5.0
    priority_exponent: float = 0.6
    priority_correction: float = 0.4
    replay_batch_size: int = 64
    replay_learning_rate: float = 1e-4

    # -- Synaptic homeostasis ----------------------------------------------
    downscale_factor: float = 0.85
    downscale_strategy: str = "global"  # "global", "selective", "layerwise"
    protect_threshold: float = 0.1
    homeostasis_epsilon: float = 1e-8

    # -- REM phase ---------------------------------------------------------
    rem_replay_steps: int = 50
    dream_noise_scale: float = 0.1
    creative_blend_ratio: float = 0.3
    dream_horizon: int = 10
    dream_temperature: float = 1.5

    # -- Systems consolidation ---------------------------------------------
    distillation_temperature: float = 2.0
    transfer_learning_rate: float = 1e-4
    fast_to_slow_ratio: float = 0.5
    transfer_steps: int = 50

    # -- Scheduling --------------------------------------------------------
    consolidate_every_n_epochs: int = 5
    min_buffer_size: int = 1000
    sleep_duration_budget: float = 0.2

    # -- Feature toggles ---------------------------------------------------
    enable_nrem: bool = True
    enable_rem: bool = True
    enable_homeostasis: bool = True
    enable_systems_transfer: bool = True

    # -- Replay buffer -----------------------------------------------------
    replay_buffer_max_size: int = 100_000

    # -- Validation --------------------------------------------------------

    def validate(self) -> List[str]:
        """Return a list of validation error strings.  Empty list means valid."""
        errors: List[str] = []

        if self.nrem_replay_steps < 0:
            errors.append("nrem_replay_steps must be >= 0")
        if self.rem_replay_steps < 0:
            errors.append("rem_replay_steps must be >= 0")
        if not (0.0 < self.compression_ratio <= 100.0):
            errors.append("compression_ratio must be in (0, 100]")
        if not (0.0 <= self.priority_exponent <= 1.0):
            errors.append("priority_exponent must be in [0, 1]")
        if not (0.0 <= self.priority_correction <= 1.0):
            errors.append("priority_correction must be in [0, 1]")
        if not (0.0 < self.downscale_factor < 1.0):
            errors.append("downscale_factor must be in (0, 1)")
        if self.downscale_strategy not in ("global", "selective", "layerwise"):
            errors.append(
                f"downscale_strategy must be 'global', 'selective', or 'layerwise', "
                f"got '{self.downscale_strategy}'"
            )
        if not (0.0 <= self.protect_threshold <= 1.0):
            errors.append("protect_threshold must be in [0, 1]")
        if not (0.0 <= self.creative_blend_ratio <= 1.0):
            errors.append("creative_blend_ratio must be in [0, 1]")
        if self.dream_noise_scale < 0:
            errors.append("dream_noise_scale must be >= 0")
        if not (0.5 <= self.distillation_temperature <= 20.0):
            errors.append("distillation_temperature must be in [0.5, 20]")
        if not (0.0 < self.fast_to_slow_ratio <= 1.0):
            errors.append("fast_to_slow_ratio must be in (0, 1]")
        if not (0.0 < self.sleep_duration_budget <= 1.0):
            errors.append("sleep_duration_budget must be in (0, 1]")
        if self.min_buffer_size < 1:
            errors.append("min_buffer_size must be >= 1")
        if self.replay_buffer_max_size < self.min_buffer_size:
            errors.append("replay_buffer_max_size must be >= min_buffer_size")
        if self.consolidate_every_n_epochs < 1:
            errors.append("consolidate_every_n_epochs must be >= 1")

        return errors

    # -- Presets -----------------------------------------------------------

    @classmethod
    def minimal(cls) -> "SleepConfig":
        """Minimal config for fast smoke tests."""
        return cls(
            nrem_replay_steps=5,
            rem_replay_steps=3,
            transfer_steps=3,
            replay_batch_size=8,
            min_buffer_size=10,
            replay_buffer_max_size=100,
            dream_horizon=3,
            consolidate_every_n_epochs=1,
            sleep_duration_budget=1.0,  # no budget constraint in tests
        )

    @classmethod
    def dev(cls) -> "SleepConfig":
        """Development config with moderate settings."""
        return cls(
            nrem_replay_steps=20,
            rem_replay_steps=10,
            transfer_steps=10,
            replay_batch_size=32,
            min_buffer_size=100,
            replay_buffer_max_size=10_000,
            consolidate_every_n_epochs=2,
        )

    @classmethod
    def production(cls) -> "SleepConfig":
        """Production config with full settings."""
        return cls(
            nrem_replay_steps=200,
            rem_replay_steps=100,
            transfer_steps=100,
            replay_batch_size=128,
            min_buffer_size=5000,
            replay_buffer_max_size=500_000,
            consolidate_every_n_epochs=5,
        )

    # -- Serialization -----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict suitable for JSON / manifest storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SleepConfig":
        """Reconstruct from a dict produced by ``to_dict``."""
        # Filter to only known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)

    def __repr__(self) -> str:
        return (
            f"SleepConfig(nrem_steps={self.nrem_replay_steps}, "
            f"rem_steps={self.rem_replay_steps}, "
            f"downscale={self.downscale_factor}, "
            f"strategy='{self.downscale_strategy}')"
        )


# =========================================================================
# Result Containers
# =========================================================================

@dataclass
class PhaseResult:
    """Result of a single sub-phase (NREM or REM)."""
    loss: float
    steps: int
    duration_seconds: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HomeostasisResult:
    """Result of synaptic homeostasis."""
    wnr_before: float
    wnr_after: float
    scaling_factor: float
    strategy: str
    num_params_scaled: int
    num_params_protected: int = 0
    duration_seconds: float = 0.0


@dataclass
class TransferResult:
    """Result of systems consolidation transfer."""
    distillation_loss: float
    reconstruction_loss: float
    combined_loss: float
    steps: int
    slow_model_improved: bool = False
    duration_seconds: float = 0.0


@dataclass
class ConsolidationResult:
    """Result of a complete consolidation cycle."""
    nrem_loss: Optional[float] = None
    nrem_steps: int = 0
    rem_loss: Optional[float] = None
    rem_steps: int = 0
    homeostasis_wnr_before: Optional[float] = None
    homeostasis_wnr_after: Optional[float] = None
    homeostasis_factor: Optional[float] = None
    transfer_loss: Optional[float] = None
    transfer_steps: int = 0
    total_sleep_time: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def all_metrics_finite(self) -> bool:
        """Check that all reported metrics are finite numbers."""
        for val in [self.nrem_loss, self.rem_loss,
                    self.homeostasis_wnr_before, self.homeostasis_wnr_after,
                    self.homeostasis_factor, self.transfer_loss,
                    self.total_sleep_time]:
            if val is not None and not math.isfinite(val):
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict."""
        return asdict(self)


# =========================================================================
# Self-test suite
# =========================================================================

def _banner(text: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def _pass(name: str) -> None:
    print(f"  [PASS] {name}")


def _fail(name: str, detail: str = "") -> None:
    msg = f"  [FAIL] {name}"
    if detail:
        msg += f" -- {detail}"
    print(msg)


def _run_tests() -> None:
    """Execute the full self-test suite."""

    passed = 0
    failed = 0
    failure_details: List[str] = []

    def check(condition: bool, name: str, detail: str = "") -> None:
        nonlocal passed, failed
        if condition:
            _pass(name)
            passed += 1
        else:
            _fail(name, detail)
            failed += 1
            failure_details.append(name)

    # ==================================================================
    # SleepConfig basic tests
    # ==================================================================
    _banner("SleepConfig -- Defaults and Validation")

    cfg = SleepConfig()

    # 1. Default values
    check(cfg.nrem_replay_steps == 100, "Default nrem_replay_steps = 100")
    check(cfg.rem_replay_steps == 50, "Default rem_replay_steps = 50")
    check(cfg.downscale_factor == 0.85, "Default downscale_factor = 0.85")
    check(cfg.downscale_strategy == "global", "Default downscale_strategy = 'global'")
    check(cfg.priority_exponent == 0.6, "Default priority_exponent = 0.6")
    check(cfg.distillation_temperature == 2.0, "Default distillation_temperature = 2.0")
    check(cfg.enable_nrem is True, "Default enable_nrem = True")
    check(cfg.enable_rem is True, "Default enable_rem = True")
    check(cfg.enable_homeostasis is True, "Default enable_homeostasis = True")
    check(cfg.enable_systems_transfer is True, "Default enable_systems_transfer = True")

    # 2. Validation passes for defaults
    errors = cfg.validate()
    check(len(errors) == 0, "Default config passes validation", str(errors))

    # 3. Validation catches invalid values
    bad_cfg = SleepConfig(downscale_factor=1.5)
    errors = bad_cfg.validate()
    check(len(errors) > 0, "downscale_factor=1.5 fails validation")
    check(any("downscale_factor" in e for e in errors), "Error mentions downscale_factor")

    bad_cfg2 = SleepConfig(priority_exponent=-0.1)
    errors2 = bad_cfg2.validate()
    check(len(errors2) > 0, "priority_exponent=-0.1 fails validation")

    bad_cfg3 = SleepConfig(downscale_strategy="invalid")
    errors3 = bad_cfg3.validate()
    check(len(errors3) > 0, "downscale_strategy='invalid' fails validation")

    bad_cfg4 = SleepConfig(creative_blend_ratio=1.5)
    errors4 = bad_cfg4.validate()
    check(len(errors4) > 0, "creative_blend_ratio=1.5 fails validation")

    bad_cfg5 = SleepConfig(sleep_duration_budget=0.0)
    errors5 = bad_cfg5.validate()
    check(len(errors5) > 0, "sleep_duration_budget=0.0 fails validation")

    bad_cfg6 = SleepConfig(min_buffer_size=0)
    errors6 = bad_cfg6.validate()
    check(len(errors6) > 0, "min_buffer_size=0 fails validation")

    bad_cfg7 = SleepConfig(replay_buffer_max_size=10, min_buffer_size=100)
    errors7 = bad_cfg7.validate()
    check(len(errors7) > 0, "max_size < min_size fails validation")

    # ==================================================================
    # Presets
    # ==================================================================
    _banner("SleepConfig -- Presets")

    minimal = SleepConfig.minimal()
    dev = SleepConfig.dev()
    prod = SleepConfig.production()

    check(len(minimal.validate()) == 0, "Minimal preset passes validation")
    check(len(dev.validate()) == 0, "Dev preset passes validation")
    check(len(prod.validate()) == 0, "Production preset passes validation")

    check(
        minimal.nrem_replay_steps < dev.nrem_replay_steps < prod.nrem_replay_steps,
        "Presets scale: minimal < dev < production (nrem_replay_steps)",
    )
    check(
        minimal.replay_buffer_max_size < dev.replay_buffer_max_size < prod.replay_buffer_max_size,
        "Presets scale: minimal < dev < production (replay_buffer_max_size)",
    )

    # ==================================================================
    # Serialization
    # ==================================================================
    _banner("SleepConfig -- Serialization")

    cfg_orig = SleepConfig(
        nrem_replay_steps=42,
        downscale_factor=0.9,
        downscale_strategy="selective",
        dream_noise_scale=0.05,
    )
    d = cfg_orig.to_dict()
    check(isinstance(d, dict), "to_dict returns dict")
    check(d["nrem_replay_steps"] == 42, "to_dict preserves nrem_replay_steps")
    check(d["downscale_factor"] == 0.9, "to_dict preserves downscale_factor")
    check(d["downscale_strategy"] == "selective", "to_dict preserves downscale_strategy")

    # JSON round-trip
    json_str = json.dumps(d)
    d_loaded = json.loads(json_str)
    cfg_loaded = SleepConfig.from_dict(d_loaded)
    check(cfg_loaded.nrem_replay_steps == 42, "JSON round-trip preserves nrem_replay_steps")
    check(cfg_loaded.downscale_factor == 0.9, "JSON round-trip preserves downscale_factor")
    check(cfg_loaded.downscale_strategy == "selective", "JSON round-trip preserves strategy")
    check(cfg_loaded.dream_noise_scale == 0.05, "JSON round-trip preserves dream_noise_scale")

    # from_dict ignores unknown keys
    d_extra = copy.deepcopy(d)
    d_extra["unknown_key_xyz"] = 999
    cfg_extra = SleepConfig.from_dict(d_extra)
    check(cfg_extra.nrem_replay_steps == 42, "from_dict ignores unknown keys")

    # ==================================================================
    # Result containers
    # ==================================================================
    _banner("Result Containers")

    phase_r = PhaseResult(loss=0.5, steps=100, duration_seconds=1.0)
    check(phase_r.loss == 0.5, "PhaseResult.loss")
    check(phase_r.steps == 100, "PhaseResult.steps")

    home_r = HomeostasisResult(
        wnr_before=1.5, wnr_after=1.1, scaling_factor=0.85,
        strategy="global", num_params_scaled=1000,
    )
    check(home_r.wnr_before == 1.5, "HomeostasisResult.wnr_before")
    check(home_r.wnr_after < home_r.wnr_before, "HomeostasisResult wnr decreased")

    transfer_r = TransferResult(
        distillation_loss=0.8, reconstruction_loss=0.3,
        combined_loss=0.55, steps=50,
    )
    check(transfer_r.combined_loss == 0.55, "TransferResult.combined_loss")

    consol_r = ConsolidationResult(
        nrem_loss=0.4, nrem_steps=100,
        rem_loss=0.3, rem_steps=50,
        homeostasis_wnr_before=1.5, homeostasis_wnr_after=1.1,
        homeostasis_factor=0.85,
        transfer_loss=0.5, transfer_steps=50,
        total_sleep_time=10.0,
    )
    check(consol_r.all_metrics_finite(), "ConsolidationResult all metrics finite")
    check(consol_r.total_sleep_time == 10.0, "ConsolidationResult.total_sleep_time")

    # Test with NaN
    bad_consol = ConsolidationResult(nrem_loss=float('nan'))
    check(not bad_consol.all_metrics_finite(), "NaN metric detected by all_metrics_finite")

    # Test with None (should be fine)
    partial_consol = ConsolidationResult(nrem_loss=0.4, rem_loss=None)
    check(partial_consol.all_metrics_finite(), "None metric treated as valid by all_metrics_finite")

    # Test to_dict
    consol_d = consol_r.to_dict()
    check(isinstance(consol_d, dict), "ConsolidationResult.to_dict returns dict")
    check(consol_d["nrem_loss"] == 0.4, "ConsolidationResult.to_dict preserves nrem_loss")

    # ==================================================================
    # Repr
    # ==================================================================
    _banner("Repr")

    r = repr(cfg)
    check("SleepConfig" in r, "SleepConfig repr contains class name")
    check("100" in r, "SleepConfig repr contains nrem_steps")

    # ==================================================================
    # Summary
    # ==================================================================
    _banner("SUMMARY")

    total = passed + failed
    print(f"\n  Total : {total}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")

    if failure_details:
        print("\n  Failed tests:")
        for name in failure_details:
            print(f"    - {name}")

    if failed > 0:
        print(f"\n  EXIT CODE: 1 ({failed} failure(s))")
        sys.exit(1)
    else:
        print("\n  All tests passed.")
        sys.exit(0)


if __name__ == "__main__":
    _run_tests()
