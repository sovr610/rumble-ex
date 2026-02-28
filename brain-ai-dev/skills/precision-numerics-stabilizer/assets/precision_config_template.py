"""
Precision + Numerics Stabilizer — Configuration Templates
=========================================================
All config dataclasses: PrecisionConfig, SentinelConfig, FailureConfig.
Includes mode derivation, validation, and JSON round-trip serialization.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Derivation helpers
# ---------------------------------------------------------------------------

def resolve_autocast_dtype(mode: str):
    """Return torch.dtype for autocast, or None if autocast is disabled.

    Returns None (not a string) for fp32 mode.
    Returns torch.bfloat16 for bf16.
    Returns torch.float16 for fp16.
    """
    if mode == "bf16":
        if _TORCH_AVAILABLE:
            return torch.bfloat16
        return "bfloat16"
    elif mode == "fp16":
        if _TORCH_AVAILABLE:
            return torch.float16
        return "float16"
    elif mode == "fp32":
        return None
    else:
        raise ValueError(
            f"Unknown precision mode: {mode!r}. Must be one of: fp32, bf16, fp16."
        )


def resolve_scaler_enabled(mode: str) -> bool:
    """Return True only for fp16 — the only mode requiring GradScaler."""
    return mode == "fp16"


# ---------------------------------------------------------------------------
# PrecisionConfig
# ---------------------------------------------------------------------------

@dataclass
class PrecisionConfig:
    """Configuration for precision mode and AMP components.

    Attributes
    ----------
    mode : str
        One of "fp32", "bf16", "fp16".
    autocast_dtype : str or None
        Explicit dtype string ("bfloat16", "float16") or None to derive from mode.
    grad_scaler_enabled : bool or None
        Explicit flag or None to derive from mode (True only for fp16).
    grad_scaler_init_scale : float
        Initial GradScaler loss scale. Default: 65536.0 (2**16).
    grad_scaler_growth_factor : float
        Scale growth factor after growth_interval clean steps. Default: 2.0.
    grad_scaler_backoff_factor : float
        Scale reduction factor on overflow. Default: 0.5.
    grad_scaler_growth_interval : int
        Consecutive clean steps before scale grows. Default: 2000.
    max_grad_norm : float
        Maximum gradient L2 norm for clipping. Default: 1.0.
    """

    mode: str = "bf16"
    autocast_dtype: Optional[str] = None
    grad_scaler_enabled: Optional[bool] = None
    grad_scaler_init_scale: float = 65536.0
    grad_scaler_growth_factor: float = 2.0
    grad_scaler_backoff_factor: float = 0.5
    grad_scaler_growth_interval: int = 2000
    max_grad_norm: float = 1.0

    def __post_init__(self):
        self.validate()

    def validate(self):
        """Raise ValueError for invalid configuration."""
        valid_modes = {"fp32", "bf16", "fp16"}
        if self.mode not in valid_modes:
            raise ValueError(
                f"PrecisionConfig.mode={self.mode!r} invalid. "
                f"Must be one of: {sorted(valid_modes)}"
            )
        if self.grad_scaler_init_scale <= 0:
            raise ValueError(
                f"grad_scaler_init_scale must be > 0, got {self.grad_scaler_init_scale}"
            )
        if self.grad_scaler_growth_factor <= 1.0:
            raise ValueError(
                f"grad_scaler_growth_factor must be > 1.0, got {self.grad_scaler_growth_factor}"
            )
        if not (0.0 < self.grad_scaler_backoff_factor < 1.0):
            raise ValueError(
                f"grad_scaler_backoff_factor must be in (0, 1), got {self.grad_scaler_backoff_factor}"
            )
        if self.grad_scaler_growth_interval < 1:
            raise ValueError(
                f"grad_scaler_growth_interval must be >= 1, got {self.grad_scaler_growth_interval}"
            )
        if self.max_grad_norm <= 0:
            raise ValueError(
                f"max_grad_norm must be > 0, got {self.max_grad_norm}"
            )

    @property
    def resolved_autocast_dtype(self):
        """Return the resolved torch.dtype (or None for fp32 / disabled autocast)."""
        if self.autocast_dtype is not None:
            if _TORCH_AVAILABLE:
                return getattr(torch, self.autocast_dtype)
            return self.autocast_dtype
        return resolve_autocast_dtype(self.mode)

    @property
    def resolved_scaler_enabled(self) -> bool:
        """Return the resolved GradScaler enabled flag."""
        if self.grad_scaler_enabled is not None:
            return bool(self.grad_scaler_enabled)
        return resolve_scaler_enabled(self.mode)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PrecisionConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "PrecisionConfig":
        return cls.from_dict(json.loads(s))


# ---------------------------------------------------------------------------
# SentinelConfig
# ---------------------------------------------------------------------------

@dataclass
class SentinelConfig:
    """Configuration for the numerics sentinel system.

    Attributes
    ----------
    every_n_steps : int
        Run sentinel checks every N training steps. Default: 50.
    grad_norm_topk : int
        Number of top-norm modules to report per check. Default: 20.
    nan_check_sample_layers : tuple of str
        Glob patterns matching module names to hook for activation checks.
    logit_max_abs_threshold : float
        Logit magnitude above which consecutive_violations is incremented.
    logit_alert_consecutive : int
        Number of consecutive threshold violations before emitting an alert.
    loss_spike_pct : float
        Percentage increase above rolling median to classify as a loss spike.
    loss_spike_window : int
        Rolling window size (in steps) for loss spike detection.
    """

    every_n_steps: int = 50
    grad_norm_topk: int = 20
    nan_check_sample_layers: Tuple[str, ...] = (
        "embeddings",
        "blocks.*.attn",
        "blocks.*.mlp",
        "lm_head",
    )
    logit_max_abs_threshold: float = 80.0
    logit_alert_consecutive: int = 3
    loss_spike_pct: float = 200.0
    loss_spike_window: int = 100

    def __post_init__(self):
        # Ensure nan_check_sample_layers is stored as a tuple
        self.nan_check_sample_layers = tuple(self.nan_check_sample_layers)
        self.validate()

    def validate(self):
        if self.every_n_steps < 1:
            raise ValueError(
                f"every_n_steps must be >= 1, got {self.every_n_steps}"
            )
        if self.grad_norm_topk < 1:
            raise ValueError(
                f"grad_norm_topk must be >= 1, got {self.grad_norm_topk}"
            )
        if self.logit_max_abs_threshold <= 0:
            raise ValueError(
                f"logit_max_abs_threshold must be > 0, got {self.logit_max_abs_threshold}"
            )
        if self.logit_alert_consecutive < 1:
            raise ValueError(
                f"logit_alert_consecutive must be >= 1, got {self.logit_alert_consecutive}"
            )
        if self.loss_spike_pct <= 0:
            raise ValueError(
                f"loss_spike_pct must be > 0, got {self.loss_spike_pct}"
            )
        if self.loss_spike_window < 2:
            raise ValueError(
                f"loss_spike_window must be >= 2, got {self.loss_spike_window}"
            )

    def to_dict(self) -> dict:
        d = asdict(self)
        # asdict converts tuples to lists; keep list for JSON compatibility
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SentinelConfig":
        d = dict(d)
        # Convert list back to tuple for nan_check_sample_layers
        if "nan_check_sample_layers" in d and isinstance(d["nan_check_sample_layers"], list):
            d["nan_check_sample_layers"] = tuple(d["nan_check_sample_layers"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "SentinelConfig":
        return cls.from_dict(json.loads(s))


# ---------------------------------------------------------------------------
# FailureConfig
# ---------------------------------------------------------------------------

@dataclass
class FailureConfig:
    """Configuration for failure detection and snapshot behavior.

    Attributes
    ----------
    nan_persist_steps : int
        Number of consecutive NaN steps before abort action is taken. Default: 3.
    snapshot_dir : str
        Directory template for snapshot output. May contain {run_id}. Default: "runs/{run_id}/numerics".
    on_error : str
        Action on persistent NaN: "abort" (sys.exit(1)) or "raise" (RuntimeError).
    deterministic_debug : bool
        Whether to enable deterministic algorithms for failure reproduction. Default: False.
    """

    nan_persist_steps: int = 3
    snapshot_dir: str = "runs/{run_id}/numerics"
    on_error: str = "abort"
    deterministic_debug: bool = False

    def __post_init__(self):
        self.validate()

    def validate(self):
        if self.nan_persist_steps < 1:
            raise ValueError(
                f"nan_persist_steps must be >= 1, got {self.nan_persist_steps}"
            )
        valid_on_error = {"abort", "raise"}
        if self.on_error not in valid_on_error:
            raise ValueError(
                f"on_error={self.on_error!r} invalid. Must be one of: {sorted(valid_on_error)}"
            )

    def resolve_snapshot_dir(self, run_id: str = "default") -> str:
        """Expand {run_id} template in snapshot_dir."""
        return self.snapshot_dir.format(run_id=run_id)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FailureConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "FailureConfig":
        return cls.from_dict(json.loads(s))


# ---------------------------------------------------------------------------
# Unified Config Container
# ---------------------------------------------------------------------------

@dataclass
class FullConfig:
    """Container for all three config groups."""
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    sentinel: SentinelConfig = field(default_factory=SentinelConfig)
    failure: FailureConfig = field(default_factory=FailureConfig)

    def to_dict(self) -> dict:
        return {
            "precision_config": self.precision.to_dict(),
            "sentinel_config": self.sentinel.to_dict(),
            "failure_config": self.failure.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "FullConfig":
        return cls(
            precision=PrecisionConfig.from_dict(d.get("precision_config", {})),
            sentinel=SentinelConfig.from_dict(d.get("sentinel_config", {})),
            failure=FailureConfig.from_dict(d.get("failure_config", {})),
        )

    @classmethod
    def from_json(cls, s: str) -> "FullConfig":
        return cls.from_dict(json.loads(s))


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("Running precision_config_template.py self-tests...")
    failures = []

    # --- Test 1: Derivation logic for bf16 ---
    try:
        cfg = PrecisionConfig(mode="bf16")
        assert cfg.resolved_scaler_enabled is False, (
            f"bf16 should not enable scaler, got {cfg.resolved_scaler_enabled}"
        )
        dtype = cfg.resolved_autocast_dtype
        if _TORCH_AVAILABLE:
            assert dtype == torch.bfloat16, f"Expected bfloat16, got {dtype}"
        else:
            assert dtype == "bfloat16", f"Expected 'bfloat16' string, got {dtype}"
        print("  [PASS] T1: bf16 derivation correct")
    except Exception as e:
        failures.append(f"T1 bf16 derivation: {e}")
        print(f"  [FAIL] T1: {e}")

    # --- Test 2: Derivation logic for fp16 ---
    try:
        cfg = PrecisionConfig(mode="fp16")
        assert cfg.resolved_scaler_enabled is True, (
            f"fp16 should enable scaler, got {cfg.resolved_scaler_enabled}"
        )
        dtype = cfg.resolved_autocast_dtype
        if _TORCH_AVAILABLE:
            assert dtype == torch.float16, f"Expected float16, got {dtype}"
        else:
            assert dtype == "float16", f"Expected 'float16' string, got {dtype}"
        print("  [PASS] T2: fp16 derivation correct")
    except Exception as e:
        failures.append(f"T2 fp16 derivation: {e}")
        print(f"  [FAIL] T2: {e}")

    # --- Test 3: Derivation logic for fp32 ---
    try:
        cfg = PrecisionConfig(mode="fp32")
        assert cfg.resolved_scaler_enabled is False, (
            f"fp32 should not enable scaler, got {cfg.resolved_scaler_enabled}"
        )
        assert cfg.resolved_autocast_dtype is None, (
            f"fp32 should have None dtype, got {cfg.resolved_autocast_dtype}"
        )
        print("  [PASS] T3: fp32 derivation correct")
    except Exception as e:
        failures.append(f"T3 fp32 derivation: {e}")
        print(f"  [FAIL] T3: {e}")

    # --- Test 4: Validation catches bad mode ---
    try:
        raised = False
        try:
            cfg = PrecisionConfig(mode="mixed16")  # invalid
        except ValueError:
            raised = True
        assert raised, "Should have raised ValueError for invalid mode"
        print("  [PASS] T4: Invalid mode raises ValueError")
    except Exception as e:
        failures.append(f"T4 bad mode validation: {e}")
        print(f"  [FAIL] T4: {e}")

    # --- Test 5: PrecisionConfig round-trip serialization ---
    try:
        cfg = PrecisionConfig(
            mode="fp16",
            grad_scaler_init_scale=32768.0,
            max_grad_norm=0.5,
        )
        json_str = cfg.to_json()
        cfg2 = PrecisionConfig.from_json(json_str)
        assert cfg2.mode == "fp16", f"mode mismatch: {cfg2.mode}"
        assert cfg2.grad_scaler_init_scale == 32768.0, (
            f"init_scale mismatch: {cfg2.grad_scaler_init_scale}"
        )
        assert cfg2.max_grad_norm == 0.5, f"max_grad_norm mismatch: {cfg2.max_grad_norm}"
        print("  [PASS] T5: PrecisionConfig JSON round-trip")
    except Exception as e:
        failures.append(f"T5 PrecisionConfig serialization: {e}")
        print(f"  [FAIL] T5: {e}")

    # --- Test 6: SentinelConfig round-trip ---
    try:
        cfg = SentinelConfig(
            every_n_steps=25,
            logit_max_abs_threshold=60.0,
            nan_check_sample_layers=("layer1", "layer2.*"),
        )
        json_str = cfg.to_json()
        cfg2 = SentinelConfig.from_json(json_str)
        assert cfg2.every_n_steps == 25, f"every_n_steps mismatch: {cfg2.every_n_steps}"
        assert cfg2.logit_max_abs_threshold == 60.0, (
            f"threshold mismatch: {cfg2.logit_max_abs_threshold}"
        )
        assert cfg2.nan_check_sample_layers == ("layer1", "layer2.*"), (
            f"patterns mismatch: {cfg2.nan_check_sample_layers}"
        )
        print("  [PASS] T6: SentinelConfig JSON round-trip")
    except Exception as e:
        failures.append(f"T6 SentinelConfig serialization: {e}")
        print(f"  [FAIL] T6: {e}")

    # --- Test 7: FailureConfig round-trip ---
    try:
        cfg = FailureConfig(nan_persist_steps=5, on_error="raise")
        json_str = cfg.to_json()
        cfg2 = FailureConfig.from_json(json_str)
        assert cfg2.nan_persist_steps == 5, f"persist mismatch: {cfg2.nan_persist_steps}"
        assert cfg2.on_error == "raise", f"on_error mismatch: {cfg2.on_error}"
        print("  [PASS] T7: FailureConfig JSON round-trip")
    except Exception as e:
        failures.append(f"T7 FailureConfig serialization: {e}")
        print(f"  [FAIL] T7: {e}")

    # --- Test 8: FailureConfig invalid on_error ---
    try:
        raised = False
        try:
            cfg = FailureConfig(on_error="exit")  # invalid
        except ValueError:
            raised = True
        assert raised, "Should have raised ValueError for invalid on_error"
        print("  [PASS] T8: Invalid on_error raises ValueError")
    except Exception as e:
        failures.append(f"T8 on_error validation: {e}")
        print(f"  [FAIL] T8: {e}")

    # --- Test 9: FullConfig round-trip ---
    try:
        full = FullConfig(
            precision=PrecisionConfig(mode="fp16"),
            sentinel=SentinelConfig(every_n_steps=10),
            failure=FailureConfig(on_error="raise"),
        )
        json_str = full.to_json()
        full2 = FullConfig.from_json(json_str)
        assert full2.precision.mode == "fp16"
        assert full2.sentinel.every_n_steps == 10
        assert full2.failure.on_error == "raise"
        print("  [PASS] T9: FullConfig JSON round-trip")
    except Exception as e:
        failures.append(f"T9 FullConfig serialization: {e}")
        print(f"  [FAIL] T9: {e}")

    # --- Test 10: Defaults are correct ---
    try:
        p = PrecisionConfig()
        s = SentinelConfig()
        f = FailureConfig()
        assert p.mode == "bf16"
        assert p.grad_scaler_init_scale == 65536.0
        assert p.max_grad_norm == 1.0
        assert s.every_n_steps == 50
        assert s.logit_max_abs_threshold == 80.0
        assert f.nan_persist_steps == 3
        assert f.on_error == "abort"
        print("  [PASS] T10: Default values correct")
    except Exception as e:
        failures.append(f"T10 defaults: {e}")
        print(f"  [FAIL] T10: {e}")

    # --- Test 11: resolve_snapshot_dir substitution ---
    try:
        cfg = FailureConfig(snapshot_dir="runs/{run_id}/numerics")
        resolved = cfg.resolve_snapshot_dir("run_abc123")
        assert resolved == "runs/run_abc123/numerics", f"Got: {resolved}"
        print("  [PASS] T11: snapshot_dir template substitution")
    except Exception as e:
        failures.append(f"T11 snapshot_dir: {e}")
        print(f"  [FAIL] T11: {e}")

    print()
    if failures:
        print(f"FAILED: {len(failures)} test(s) failed:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print(f"All 11 self-tests passed.")
