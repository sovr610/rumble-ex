"""
budget_config_template.py
=========================
All configuration and result dataclasses for the Compute-Optimal Budget Planner.

Provides:
    BudgetConfig    — global planner settings (k, tokens_per_param_target, etc.)
    RunSpec         — describes an existing or planned training run (Mode A)
    ModelSpec       — describes a model to plan around (Mode B)
    ComputeBudget   — describes a compute budget to solve from (Mode C)
    BudgetResult    — output from any planner mode
    GPUSpec         — single GPU hardware spec

All dataclasses support:
    - .to_dict()     → dict (JSON-serializable)
    - .from_dict(d)  → instance (classmethod)
    - ._validate()   → raises ValueError on invalid fields

Usage
-----
    from budget_config_template import BudgetConfig, RunSpec, BudgetResult
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_positive(value: float, name: str) -> None:
    """Raise ValueError if value <= 0."""
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")


def _require_positive_or_zero(value: float, name: str) -> None:
    """Raise ValueError if value < 0."""
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number, got {value!r}")


def _require_in_range(value: float, lo: float, hi: float, name: str) -> None:
    """Raise ValueError if value not in [lo, hi]."""
    if not math.isfinite(value) or value < lo or value > hi:
        raise ValueError(f"{name} must be in [{lo}, {hi}], got {value!r}")


def _to_json_compatible(obj: Any) -> Any:
    """Recursively convert an object to JSON-serializable form."""
    if isinstance(obj, dict):
        return {k: _to_json_compatible(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_compatible(v) for v in obj]
    if isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
        return None  # JSON cannot represent inf/nan
    return obj


# ---------------------------------------------------------------------------
# GPUSpec
# ---------------------------------------------------------------------------

@dataclass
class GPUSpec:
    """Hardware specification for a single GPU.

    Attributes
    ----------
    name : str
        Canonical GPU name (e.g., "H100 SXM").
    peak_tflops : float
        Peak TFLOPS for the selected dtype (dense, not sparse).
    mem_gb : float
        GPU memory in gigabytes.
    is_sparse : bool
        Whether peak_tflops reflects sparse (2:4) operation. Default False.
    compute_capability : float
        NVIDIA compute capability (e.g., 9.0 for H100).
    peak_tflops_bf16 : float
        BF16 dense TFLOPS.
    peak_tflops_fp16 : float
        FP16 dense TFLOPS.
    peak_tflops_fp32 : float
        FP32 dense TFLOPS.
    is_user_supplied : bool
        True if specs were provided by user rather than from internal table.
    """

    name: str
    peak_tflops: float                   # resolved for selected dtype
    mem_gb: float
    is_sparse: bool = False
    compute_capability: float = 0.0
    peak_tflops_bf16: float = 0.0
    peak_tflops_fp16: float = 0.0
    peak_tflops_fp32: float = 0.0
    is_user_supplied: bool = False

    def _validate(self) -> None:
        _require_positive(self.peak_tflops, "GPUSpec.peak_tflops")
        _require_positive(self.mem_gb, "GPUSpec.mem_gb")

    def to_dict(self) -> Dict[str, Any]:
        return _to_json_compatible(asdict(self))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GPUSpec":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def user_supplied(cls, peak_tflops: float, mem_gb: float) -> "GPUSpec":
        """Create a GPUSpec from user-provided values (fallback mode)."""
        spec = cls(
            name="user_supplied",
            peak_tflops=peak_tflops,
            mem_gb=mem_gb,
            peak_tflops_bf16=peak_tflops,
            peak_tflops_fp16=peak_tflops,
            peak_tflops_fp32=peak_tflops / 32,
            is_user_supplied=True,
        )
        spec._validate()
        return spec


# ---------------------------------------------------------------------------
# BudgetConfig
# ---------------------------------------------------------------------------

@dataclass
class BudgetConfig:
    """Global planner configuration.

    Attributes
    ----------
    k : float
        FLOPs coefficient in C = k * N * D. Default 6 (2 forward + 4 backward).
    tokens_per_param_target : float
        Chinchilla-style target tokens per non-embedding parameter. Default 20.0.
    default_utilization : float
        Model FLOPs Utilization (MFU) assumption. Must be in (0, 1]. Default 0.35.
    cost_per_gpu_hour : float or None
        GPU cost in USD per GPU-hour. None disables cost estimation.
    num_gpus : int
        Default number of GPUs for wallclock estimation.
    gpu_type : str
        Default GPU model name for spec lookup.
    dtype : str
        Default training dtype for TFLOPS lookup. One of "bf16", "fp16", "fp32".
    log_assumptions : bool
        Whether to log assumptions at planning time. Default True.
    """

    k: float = 6.0
    tokens_per_param_target: float = 20.0
    default_utilization: float = 0.35
    cost_per_gpu_hour: Optional[float] = None
    num_gpus: int = 8
    gpu_type: str = "H100_SXM"
    dtype: str = "bf16"
    log_assumptions: bool = True

    def _validate(self) -> None:
        _require_positive(self.k, "BudgetConfig.k")
        _require_positive(self.tokens_per_param_target, "BudgetConfig.tokens_per_param_target")
        _require_in_range(self.default_utilization, 1e-6, 1.0, "BudgetConfig.default_utilization")
        if self.cost_per_gpu_hour is not None:
            _require_positive(self.cost_per_gpu_hour, "BudgetConfig.cost_per_gpu_hour")
        if self.num_gpus < 1:
            raise ValueError(f"BudgetConfig.num_gpus must be >= 1, got {self.num_gpus}")
        if self.dtype not in ("bf16", "fp16", "fp32", "fp8", "tf32"):
            raise ValueError(f"BudgetConfig.dtype must be one of bf16/fp16/fp32/fp8/tf32, got {self.dtype!r}")

    def to_dict(self) -> Dict[str, Any]:
        return _to_json_compatible(asdict(self))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BudgetConfig":
        valid_fields = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str) -> "BudgetConfig":
        """Load BudgetConfig from a YAML file. Requires pyyaml."""
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("pyyaml is required to load from YAML. pip install pyyaml") from exc
        with open(path, "r") as fh:
            data = yaml.safe_load(fh)
        return cls.from_dict(data)

    def __post_init__(self) -> None:
        self._validate()


# ---------------------------------------------------------------------------
# RunSpec  (Mode A input)
# ---------------------------------------------------------------------------

@dataclass
class RunSpec:
    """Describes an existing or planned training run for validation (Mode A).

    Attributes
    ----------
    n_params : float
        Non-embedding parameter count.
    seq_len : int
        Sequence length in tokens.
    global_batch : int
        Global batch size across all data-parallel ranks.
    steps : int
        Number of optimizer steps (after gradient accumulation).
    num_gpus : int
        Number of GPUs in the training cluster.
    gpu_type : str
        GPU model name for spec lookup (e.g., "H100_SXM").
    dtype : str
        Training precision ("bf16", "fp16", "fp32").
    utilization : float or None
        MFU override. If None, uses BudgetConfig.default_utilization.
    cost_per_gpu_hour : float or None
        Per-GPU-hour cost override. If None, uses BudgetConfig.cost_per_gpu_hour.
    """

    n_params: float
    seq_len: int
    global_batch: int
    steps: int
    num_gpus: int = 1
    gpu_type: str = "H100_SXM"
    dtype: str = "bf16"
    utilization: Optional[float] = None
    cost_per_gpu_hour: Optional[float] = None

    def _validate(self) -> None:
        _require_positive(self.n_params, "RunSpec.n_params")
        if self.seq_len <= 0:
            raise ValueError(f"RunSpec.seq_len must be > 0, got {self.seq_len}")
        if self.global_batch <= 0:
            raise ValueError(f"RunSpec.global_batch must be > 0, got {self.global_batch}")
        if self.steps <= 0:
            raise ValueError(f"RunSpec.steps must be > 0, got {self.steps}")
        if self.num_gpus < 1:
            raise ValueError(f"RunSpec.num_gpus must be >= 1, got {self.num_gpus}")
        if self.utilization is not None:
            _require_in_range(self.utilization, 1e-6, 1.0, "RunSpec.utilization")
        if self.cost_per_gpu_hour is not None:
            _require_positive(self.cost_per_gpu_hour, "RunSpec.cost_per_gpu_hour")

    @property
    def total_tokens(self) -> float:
        """Total training tokens: steps * global_batch * seq_len."""
        return float(self.steps) * float(self.global_batch) * float(self.seq_len)

    def to_dict(self) -> Dict[str, Any]:
        return _to_json_compatible(asdict(self))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunSpec":
        valid_fields = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)

    def __post_init__(self) -> None:
        self._validate()


# ---------------------------------------------------------------------------
# ModelSpec  (Mode B input)
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    """Describes a model for which to compute the required training budget (Mode B).

    Attributes
    ----------
    n_params : float
        Non-embedding parameter count.
    seq_len : int
        Sequence length. Default 2048.
    global_batch : int
        Global batch size. Default 2048.
    tokens_per_param_target : float or None
        Override for tokens_per_param_target. If None, uses BudgetConfig default.
    num_gpus : int or None
        Override for number of GPUs. If None, uses BudgetConfig default.
    gpu_type : str or None
        Override for GPU type. If None, uses BudgetConfig default.
    dtype : str
        Training precision.
    utilization : float or None
        MFU override.
    cost_per_gpu_hour : float or None
        Per-GPU-hour cost override.
    """

    n_params: float
    seq_len: int = 2048
    global_batch: int = 2048
    tokens_per_param_target: Optional[float] = None
    num_gpus: Optional[int] = None
    gpu_type: Optional[str] = None
    dtype: str = "bf16"
    utilization: Optional[float] = None
    cost_per_gpu_hour: Optional[float] = None

    def _validate(self) -> None:
        _require_positive(self.n_params, "ModelSpec.n_params")
        if self.seq_len <= 0:
            raise ValueError(f"ModelSpec.seq_len must be > 0, got {self.seq_len}")
        if self.global_batch <= 0:
            raise ValueError(f"ModelSpec.global_batch must be > 0, got {self.global_batch}")
        if self.tokens_per_param_target is not None:
            _require_positive(self.tokens_per_param_target, "ModelSpec.tokens_per_param_target")
        if self.num_gpus is not None and self.num_gpus < 1:
            raise ValueError(f"ModelSpec.num_gpus must be >= 1, got {self.num_gpus}")
        if self.utilization is not None:
            _require_in_range(self.utilization, 1e-6, 1.0, "ModelSpec.utilization")

    def to_dict(self) -> Dict[str, Any]:
        return _to_json_compatible(asdict(self))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelSpec":
        valid_fields = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)

    def __post_init__(self) -> None:
        self._validate()


# ---------------------------------------------------------------------------
# ComputeBudget  (Mode C input)
# ---------------------------------------------------------------------------

@dataclass
class ComputeBudget:
    """Describes a compute budget for solving optimal (N, D) (Mode C).

    mode must be one of "flops", "time", or "money".

    Attributes
    ----------
    mode : str
        Budget specification mode: "flops" | "time" | "money".
    total_flops : float or None
        Total training FLOPs (required if mode="flops").
    num_gpus : int or None
        Number of GPUs (required if mode="time" or "money").
    gpu_type : str or None
        GPU type for spec lookup (required if mode="time" or "money").
    hours : float or None
        Cluster-hours budget (required if mode="time").
    utilization : float
        MFU assumption for converting time to FLOPs. Default 0.35.
    budget_dollars : float or None
        Total dollar budget (required if mode="money").
    cost_per_gpu_hour : float or None
        Per-GPU-hour cost (required if mode="money").
    tokens_per_param_target : float or None
        Override target. If None, uses BudgetConfig default.
    k : float or None
        Override FLOPs coefficient. If None, uses BudgetConfig default.
    """

    mode: str = "flops"
    total_flops: Optional[float] = None
    num_gpus: Optional[int] = None
    gpu_type: Optional[str] = None
    hours: Optional[float] = None
    utilization: float = 0.35
    budget_dollars: Optional[float] = None
    cost_per_gpu_hour: Optional[float] = None
    tokens_per_param_target: Optional[float] = None
    k: Optional[float] = None

    _VALID_MODES = ("flops", "time", "money")

    def _validate(self) -> None:
        if self.mode not in self._VALID_MODES:
            raise ValueError(
                f"ComputeBudget.mode must be one of {self._VALID_MODES}, got {self.mode!r}"
            )
        _require_in_range(self.utilization, 1e-6, 1.0, "ComputeBudget.utilization")

        if self.mode == "flops":
            if self.total_flops is None:
                raise ValueError("ComputeBudget.total_flops required when mode='flops'")
            _require_positive(self.total_flops, "ComputeBudget.total_flops")

        elif self.mode == "time":
            if self.hours is None:
                raise ValueError("ComputeBudget.hours required when mode='time'")
            if self.num_gpus is None:
                raise ValueError("ComputeBudget.num_gpus required when mode='time'")
            _require_positive(self.hours, "ComputeBudget.hours")
            if self.num_gpus < 1:
                raise ValueError("ComputeBudget.num_gpus must be >= 1")

        elif self.mode == "money":
            if self.budget_dollars is None:
                raise ValueError("ComputeBudget.budget_dollars required when mode='money'")
            if self.cost_per_gpu_hour is None:
                raise ValueError("ComputeBudget.cost_per_gpu_hour required when mode='money'")
            if self.num_gpus is None:
                raise ValueError("ComputeBudget.num_gpus required when mode='money'")
            _require_positive(self.budget_dollars, "ComputeBudget.budget_dollars")
            _require_positive(self.cost_per_gpu_hour, "ComputeBudget.cost_per_gpu_hour")

        if self.tokens_per_param_target is not None:
            _require_positive(self.tokens_per_param_target, "ComputeBudget.tokens_per_param_target")
        if self.k is not None:
            _require_positive(self.k, "ComputeBudget.k")

    def to_dict(self) -> Dict[str, Any]:
        return _to_json_compatible(asdict(self))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ComputeBudget":
        valid_fields = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)

    def __post_init__(self) -> None:
        self._validate()


# ---------------------------------------------------------------------------
# BudgetResult  (output from any planner mode)
# ---------------------------------------------------------------------------

@dataclass
class BudgetResult:
    """Output from any BudgetPlanner mode.

    Attributes
    ----------
    mode : str
        Which planner mode produced this result.
    inputs : dict
        The raw input spec fields.
    assumptions : dict
        All configuration assumptions that were used (k, utilization, etc.).
    derived : dict
        All computed values:
            total_tokens          — total training tokens
            tokens_per_param      — computed tokens per parameter
            total_flops           — total estimated FLOPs
            predicted_wallclock_hours — estimated wall time in hours
            predicted_cost_usd    — estimated cost (None if no pricing)
            undertraining_ratio   — ratio vs tokens_per_param_target
            n_opt                 — compute-optimal N for this budget
            d_opt                 — compute-optimal D for this budget
    warnings : list of str
        Severity-prefixed warning messages (CRITICAL / WARNING / INFO).
    suggestions : list of str
        Actionable corrective suggestions.
    generated_at : str
        ISO 8601 timestamp of when the result was produced.
    schema_version : str
        Version identifier for the output schema.
    """

    mode: str
    inputs: Dict[str, Any]
    assumptions: Dict[str, Any]
    derived: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: str = "1.0"

    # Convenience accessors for commonly used derived fields
    @property
    def total_tokens(self) -> Optional[float]:
        return self.derived.get("total_tokens")

    @property
    def tokens_per_param(self) -> Optional[float]:
        return self.derived.get("tokens_per_param")

    @property
    def total_flops(self) -> Optional[float]:
        return self.derived.get("total_flops")

    @property
    def predicted_wallclock_hours(self) -> Optional[float]:
        return self.derived.get("predicted_wallclock_hours")

    @property
    def predicted_cost_usd(self) -> Optional[float]:
        return self.derived.get("predicted_cost_usd")

    @property
    def undertraining_ratio(self) -> Optional[float]:
        return self.derived.get("undertraining_ratio")

    @property
    def n_opt(self) -> Optional[float]:
        return self.derived.get("n_opt")

    @property
    def d_opt(self) -> Optional[float]:
        return self.derived.get("d_opt")

    def has_critical_warnings(self) -> bool:
        """Return True if any warning has CRITICAL severity."""
        return any("[CRITICAL]" in w or w.startswith("CRITICAL") for w in self.warnings)

    def has_warnings(self) -> bool:
        """Return True if there are any warnings (any severity)."""
        return len(self.warnings) > 0

    def to_dict(self) -> Dict[str, Any]:
        return _to_json_compatible({
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "mode": self.mode,
            "inputs": self.inputs,
            "assumptions": self.assumptions,
            "derived": self.derived,
            "warnings": self.warnings,
            "suggestions": self.suggestions,
        })

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BudgetResult":
        return cls(
            mode=d.get("mode", "unknown"),
            inputs=d.get("inputs", {}),
            assumptions=d.get("assumptions", {}),
            derived=d.get("derived", {}),
            warnings=d.get("warnings", []),
            suggestions=d.get("suggestions", []),
            generated_at=d.get("generated_at", datetime.now(timezone.utc).isoformat()),
            schema_version=d.get("schema_version", "1.0"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "BudgetResult":
        return cls.from_dict(json.loads(json_str))


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _run_self_tests() -> None:
    print("Running budget_config_template.py self-tests...")
    errors = []

    # --- BudgetConfig defaults ---
    try:
        cfg = BudgetConfig()
        assert cfg.k == 6.0, f"Expected k=6.0, got {cfg.k}"
        assert cfg.tokens_per_param_target == 20.0
        assert cfg.default_utilization == 0.35
        print("  [PASS] BudgetConfig defaults valid")
    except Exception as e:
        errors.append(f"  [FAIL] BudgetConfig defaults: {e}")

    # --- BudgetConfig validation catches bad values ---
    try:
        try:
            BudgetConfig(k=-1.0)
            errors.append("  [FAIL] BudgetConfig(k=-1.0) should raise ValueError")
        except ValueError:
            pass

        try:
            BudgetConfig(default_utilization=0.0)
            errors.append("  [FAIL] BudgetConfig(utilization=0.0) should raise ValueError")
        except ValueError:
            pass

        try:
            BudgetConfig(default_utilization=1.5)
            errors.append("  [FAIL] BudgetConfig(utilization=1.5) should raise ValueError")
        except ValueError:
            pass

        try:
            BudgetConfig(tokens_per_param_target=-5.0)
            errors.append("  [FAIL] BudgetConfig(tokens_per_param_target=-5) should raise ValueError")
        except ValueError:
            pass

        print("  [PASS] BudgetConfig validation catches bad values")
    except Exception as e:
        errors.append(f"  [FAIL] BudgetConfig validation: {e}")

    # --- BudgetConfig serialization roundtrip ---
    try:
        cfg = BudgetConfig(k=8.0, tokens_per_param_target=30.0, default_utilization=0.4)
        d = cfg.to_dict()
        cfg2 = BudgetConfig.from_dict(d)
        assert cfg2.k == 8.0, f"k roundtrip failed: {cfg2.k}"
        assert cfg2.tokens_per_param_target == 30.0
        assert cfg2.default_utilization == 0.4
        print("  [PASS] BudgetConfig serialization roundtrip")
    except Exception as e:
        errors.append(f"  [FAIL] BudgetConfig serialization: {e}")

    # --- RunSpec validation ---
    try:
        run = RunSpec(n_params=7e9, seq_len=2048, global_batch=2048, steps=100000)
        assert run.total_tokens == 100000 * 2048 * 2048
        print("  [PASS] RunSpec total_tokens property")

        try:
            RunSpec(n_params=-1, seq_len=2048, global_batch=2048, steps=100)
            errors.append("  [FAIL] RunSpec(n_params=-1) should raise ValueError")
        except ValueError:
            pass

        try:
            RunSpec(n_params=7e9, seq_len=0, global_batch=2048, steps=100)
            errors.append("  [FAIL] RunSpec(seq_len=0) should raise ValueError")
        except ValueError:
            pass

        print("  [PASS] RunSpec validation catches bad values")
    except Exception as e:
        errors.append(f"  [FAIL] RunSpec: {e}")

    # --- RunSpec serialization ---
    try:
        run = RunSpec(n_params=7e9, seq_len=2048, global_batch=2048, steps=500, num_gpus=8)
        d = run.to_dict()
        run2 = RunSpec.from_dict(d)
        assert run2.n_params == 7e9
        assert run2.num_gpus == 8
        print("  [PASS] RunSpec serialization roundtrip")
    except Exception as e:
        errors.append(f"  [FAIL] RunSpec serialization: {e}")

    # --- ModelSpec ---
    try:
        m = ModelSpec(n_params=70e9)
        assert m.seq_len == 2048
        assert m.global_batch == 2048
        d = m.to_dict()
        m2 = ModelSpec.from_dict(d)
        assert m2.n_params == 70e9
        print("  [PASS] ModelSpec defaults and roundtrip")
    except Exception as e:
        errors.append(f"  [FAIL] ModelSpec: {e}")

    # --- ComputeBudget modes ---
    try:
        cb_flops = ComputeBudget(mode="flops", total_flops=5e23)
        assert cb_flops.total_flops == 5e23

        cb_time = ComputeBudget(mode="time", hours=1000.0, num_gpus=8, gpu_type="H100_SXM")
        assert cb_time.hours == 1000.0

        cb_money = ComputeBudget(
            mode="money", budget_dollars=50000, cost_per_gpu_hour=4.0,
            num_gpus=8, gpu_type="H100_SXM"
        )
        assert cb_money.budget_dollars == 50000

        print("  [PASS] ComputeBudget modes construct correctly")
    except Exception as e:
        errors.append(f"  [FAIL] ComputeBudget: {e}")

    # --- ComputeBudget validation ---
    try:
        try:
            ComputeBudget(mode="invalid_mode")
            errors.append("  [FAIL] ComputeBudget(mode='invalid') should raise ValueError")
        except ValueError:
            pass

        try:
            ComputeBudget(mode="flops")  # missing total_flops
            errors.append("  [FAIL] ComputeBudget(mode='flops', no total_flops) should raise ValueError")
        except ValueError:
            pass

        print("  [PASS] ComputeBudget validation catches bad values")
    except Exception as e:
        errors.append(f"  [FAIL] ComputeBudget validation: {e}")

    # --- BudgetResult construction and serialization ---
    try:
        result = BudgetResult(
            mode="validate_run",
            inputs={"n_params": 7e9},
            assumptions={"k": 6.0},
            derived={
                "total_tokens": 2e12,
                "tokens_per_param": 285.7,
                "total_flops": 8.4e22,
                "predicted_wallclock_hours": 131.6,
                "predicted_cost_usd": None,
                "undertraining_ratio": 14.3,
                "n_opt": 2.65e9,
                "d_opt": 5.3e10,
            },
            warnings=["[INFO] overtrain regime detected"],
            suggestions=["Confirm this is intentional."],
        )
        assert result.total_tokens == 2e12
        assert result.tokens_per_param == 285.7
        assert result.has_warnings()
        assert not result.has_critical_warnings()
        json_str = result.to_json()
        result2 = BudgetResult.from_json(json_str)
        assert result2.mode == "validate_run"
        assert result2.derived["total_tokens"] == 2e12
        print("  [PASS] BudgetResult construction and JSON roundtrip")
    except Exception as e:
        errors.append(f"  [FAIL] BudgetResult: {e}")

    # --- GPUSpec ---
    try:
        spec = GPUSpec(name="H100 SXM", peak_tflops=989.0, mem_gb=80.0,
                       peak_tflops_bf16=989.0, peak_tflops_fp16=989.0)
        spec._validate()
        d = spec.to_dict()
        spec2 = GPUSpec.from_dict(d)
        assert spec2.peak_tflops == 989.0
        assert spec2.mem_gb == 80.0

        us = GPUSpec.user_supplied(peak_tflops=500.0, mem_gb=40.0)
        assert us.is_user_supplied
        print("  [PASS] GPUSpec construction, validation, roundtrip, user_supplied")
    except Exception as e:
        errors.append(f"  [FAIL] GPUSpec: {e}")

    # --- _to_json_compatible handles inf/nan ---
    try:
        d = {"a": float("inf"), "b": float("nan"), "c": 1.0}
        result_d = _to_json_compatible(d)
        assert result_d["a"] is None
        assert result_d["b"] is None
        assert result_d["c"] == 1.0
        print("  [PASS] _to_json_compatible handles inf/nan")
    except Exception as e:
        errors.append(f"  [FAIL] _to_json_compatible: {e}")

    # --- Summary ---
    if errors:
        print("\nFailed tests:")
        for err in errors:
            print(err)
        raise SystemExit(1)
    else:
        print("\nAll self-tests passed.")


if __name__ == "__main__":
    _run_self_tests()
