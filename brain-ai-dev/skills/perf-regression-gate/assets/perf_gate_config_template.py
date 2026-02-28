"""
perf_gate_config_template.py
=============================
All configuration dataclasses, result dataclasses, and the GPU peak TFLOPS
registry for the Compute/Throughput Baseline & Regression Gate skill.

Usage:
    from perf_gate_config_template import (
        BenchConfig, EvalConfig, ToleranceConfig,
        BenchResult, EvalResult, CheckResult, CompareResult,
        GPU_PEAK_TFLOPS, get_peak_tflops,
    )

Self-test:
    python perf_gate_config_template.py
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Optional YAML support
# ---------------------------------------------------------------------------
try:
    import yaml as _yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


# ===========================================================================
# GPU Peak TFLOPS Registry
# ===========================================================================
# Values are dense (non-sparse) bf16 TFLOPS from official NVIDIA specs.
# fp16 is listed separately where it differs significantly.
# For V100 (no native bf16), fp16 values are used.

GPU_PEAK_TFLOPS: Dict[str, Dict[str, float]] = {
    # H100 family
    "H100 SXM5":   {"bf16": 989.0,  "fp16": 1979.0},
    "H100 SXM":    {"bf16": 989.0,  "fp16": 1979.0},
    "H100 NVL":    {"bf16": 835.0,  "fp16": 1671.0},
    "H100 PCIe":   {"bf16": 756.0,  "fp16": 1513.0},
    # A100 family (bf16 == fp16 on A100)
    "A100 SXM4 80GB": {"bf16": 312.0, "fp16": 312.0},
    "A100 SXM4 40GB": {"bf16": 312.0, "fp16": 312.0},
    "A100 SXM":    {"bf16": 312.0,  "fp16": 312.0},
    "A100 PCIe 80GB": {"bf16": 312.0, "fp16": 312.0},
    "A100 PCIe 40GB": {"bf16": 312.0, "fp16": 312.0},
    "A100 PCIe":   {"bf16": 312.0,  "fp16": 312.0},
    "A100":        {"bf16": 312.0,  "fp16": 312.0},
    # L40 family
    "L40S":        {"bf16": 362.0,  "fp16": 362.0},
    "L40":         {"bf16": 181.0,  "fp16": 181.0},
    # RTX Ada Lovelace
    "RTX 4090":    {"bf16": 330.0,  "fp16": 165.0},
    "RTX 4080 SUPER": {"bf16": 260.0, "fp16": 130.0},
    "RTX 4080":    {"bf16": 242.0,  "fp16": 121.0},
    "RTX 4070 Ti": {"bf16": 183.0,  "fp16": 91.6},
    "RTX 4070":    {"bf16": 165.0,  "fp16": 82.6},
    # RTX Ampere
    "RTX 3090 Ti": {"bf16": 40.0,   "fp16": 40.0},
    "RTX 3090":    {"bf16": 35.6,   "fp16": 35.6},
    "RTX 3080 Ti": {"bf16": 34.1,   "fp16": 34.1},
    "RTX 3080":    {"bf16": 29.8,   "fp16": 29.8},
    # A-series workstation / data center
    "A40":         {"bf16": 149.7,  "fp16": 149.7},
    "A30":         {"bf16": 165.0,  "fp16": 165.0},
    "A10G":        {"bf16": 125.0,  "fp16": 125.0},
    "A10":         {"bf16": 125.0,  "fp16": 125.0},
    # V100 (no native bf16; fp16 used)
    "V100 SXM2":   {"bf16": 125.0,  "fp16": 125.0},
    "V100 SXM":    {"bf16": 125.0,  "fp16": 125.0},
    "V100 PCIe":   {"bf16": 112.0,  "fp16": 112.0},
    "V100":        {"bf16": 112.0,  "fp16": 112.0},
    # T4 inference card
    "T4":          {"bf16": 65.0,   "fp16": 65.0},
    # H200 (future-proofing)
    "H200 SXM":    {"bf16": 1979.0, "fp16": 3958.0},
    "H200":        {"bf16": 1979.0, "fp16": 3958.0},
}


def get_peak_tflops(
    device_name: str,
    dtype: str = "bf16",
) -> Optional[float]:
    """
    Look up peak TFLOPS for a GPU given its name from torch.cuda.get_device_name().

    Args:
        device_name: GPU name string, e.g. "NVIDIA H100 SXM5 80GB HBM3"
        dtype: "bf16" or "fp16" (default: "bf16")

    Returns:
        Peak TFLOPS as float, or None if GPU not found in registry.
    """
    # Try exact prefix matches first (longest match wins)
    best_match: Optional[str] = None
    best_len = 0
    for key in GPU_PEAK_TFLOPS:
        if key in device_name and len(key) > best_len:
            best_match = key
            best_len = len(key)

    if best_match is None:
        return None

    dtype_key = dtype.lower().replace("-", "")
    if dtype_key not in GPU_PEAK_TFLOPS[best_match]:
        # Fall back to bf16 if requested dtype not in registry
        dtype_key = "bf16"

    return GPU_PEAK_TFLOPS[best_match].get(dtype_key)


# ===========================================================================
# Status Enum
# ===========================================================================

class Status(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    SKIP = "SKIP"


# ===========================================================================
# Configuration Dataclasses
# ===========================================================================

@dataclass
class BenchConfig:
    """
    Configuration for the training benchmark harness (Phase 2).

    Fields:
        warmup_steps:    Number of steps to run before measurement begins.
                         JIT compilation and cache warming happen here.
        measure_steps:   Number of steps to measure. Gate on p50 of these.
        mode:            "synthetic" (random tensors) or "e2e" (real dataloader).
        world_size:      Number of GPUs. -1 = auto-detect.
        profile:         "off" | "trace" (Chrome JSON) | "tb" (TensorBoard).
        repeat:          Run K full bench cycles; take best median. Reduces noise.
        peak_tflops:     Override GPU peak TFLOPS for MFU computation.
                         If None, looks up from GPU_PEAK_TFLOPS registry.
        mfu_estimator:   "6ND" or "transformer_aware".
        per_device_batch_size: Batch size per GPU.
        seq_len:         Sequence length in tokens.
        grad_accum_steps: Gradient accumulation steps per optimizer step.
        dtype:           Training dtype ("bfloat16" or "float16").
        vocab_size:      Vocab size for synthetic token generation.
        hidden_dim:      Model hidden dimension (required for transformer_aware MFU).
        num_layers:      Number of transformer layers (required for transformer_aware).
    """
    warmup_steps: int = 200
    measure_steps: int = 100
    mode: str = "synthetic"            # "synthetic" | "e2e"
    world_size: int = 1                # 1 | N | -1 for auto
    profile: str = "off"              # "off" | "trace" | "tb"
    repeat: int = 1
    peak_tflops: Optional[float] = None
    mfu_estimator: str = "6ND"        # "6ND" | "transformer_aware"
    per_device_batch_size: int = 8
    seq_len: int = 2048
    grad_accum_steps: int = 1
    dtype: str = "bfloat16"
    vocab_size: int = 32000
    hidden_dim: int = 4096
    num_layers: int = 32

    def validate(self) -> None:
        """Raise ValueError if any field is invalid."""
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        if self.measure_steps < 1:
            raise ValueError(f"measure_steps must be >= 1, got {self.measure_steps}")
        if self.mode not in ("synthetic", "e2e"):
            raise ValueError(f"mode must be 'synthetic' or 'e2e', got '{self.mode}'")
        if self.world_size < -1 or self.world_size == 0:
            raise ValueError(f"world_size must be -1 or >= 1, got {self.world_size}")
        if self.profile not in ("off", "trace", "tb"):
            raise ValueError(f"profile must be 'off', 'trace', or 'tb', got '{self.profile}'")
        if self.repeat < 1:
            raise ValueError(f"repeat must be >= 1, got {self.repeat}")
        if self.mfu_estimator not in ("6ND", "transformer_aware"):
            raise ValueError(
                f"mfu_estimator must be '6ND' or 'transformer_aware', got '{self.mfu_estimator}'"
            )
        if self.per_device_batch_size < 1:
            raise ValueError(f"per_device_batch_size must be >= 1")
        if self.seq_len < 1:
            raise ValueError(f"seq_len must be >= 1")
        if self.grad_accum_steps < 1:
            raise ValueError(f"grad_accum_steps must be >= 1")
        if self.dtype not in ("bfloat16", "float16", "float32"):
            raise ValueError(f"dtype must be bfloat16, float16, or float32")

    @property
    def global_batch_size(self) -> int:
        ws = max(1, self.world_size)
        return self.per_device_batch_size * ws * self.grad_accum_steps

    @property
    def tokens_per_step(self) -> int:
        return self.global_batch_size * self.seq_len


@dataclass
class EvalConfig:
    """
    Configuration for the quality evaluation harness (Phase 3).

    Fields:
        fixed_shard_path: Path to the bundled fixed evaluation text shard.
        fixed_shard_sha256: Expected SHA256 of the shard file. Verified before use.
        probes:           List of probe set names to run.
        temperature:      Generation temperature. Must be 0.0 for deterministic eval.
        seed:             Random seed for all RNG sources.
        max_new_tokens:   Maximum new tokens generated per probe.
        max_length:       Maximum tokenizer sequence length.
        stride:           Sliding window stride for perplexity computation.
    """
    fixed_shard_path: str = "bench/data/fixed_shard.txt"
    fixed_shard_sha256: str = ""       # Empty = skip verification
    probes: List[str] = field(default_factory=lambda: [
        "basic_reasoning_25",
        "format_following_30",
        "code_sanity_20",
    ])
    temperature: float = 0.0
    seed: int = 42
    max_new_tokens: int = 32
    max_length: int = 2048
    stride: int = 512

    def validate(self) -> None:
        """Raise ValueError if any field is invalid."""
        if self.temperature != 0.0:
            raise ValueError(
                f"temperature must be 0.0 for deterministic CI eval, got {self.temperature}. "
                "Non-zero temperature produces non-deterministic outputs."
            )
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {self.max_new_tokens}")
        if self.max_length < 64:
            raise ValueError(f"max_length must be >= 64, got {self.max_length}")
        if self.stride < 1 or self.stride > self.max_length:
            raise ValueError(
                f"stride must be in [1, max_length], got stride={self.stride}, "
                f"max_length={self.max_length}"
            )
        if not self.probes:
            raise ValueError("probes list must not be empty")


@dataclass
class ToleranceConfig:
    """
    Tolerance thresholds for baseline comparison (Phase 4).

    Performance gate (FAIL on violation):
        throughput_drop_pct:    tokens/sec p50 drop percentage that triggers FAIL.
        step_time_increase_pct: step time p50 increase percentage that triggers FAIL.

    Memory gate (WARN only):
        memory_increase_pct:    peak memory increase percentage that triggers WARN.

    Quality gate (FAIL on violation):
        ppl_increase_pct:       relative perplexity increase percentage that triggers FAIL.
        probe_drop_abs:         absolute probe accuracy drop in percentage points that triggers FAIL.

    Stability gate (FAIL on violation):
        loss_slope_threshold:   positive loss slope above this triggers FAIL.
    """
    # Performance
    throughput_drop_pct: float = 5.0
    step_time_increase_pct: float = 5.0
    # Memory (warn only)
    memory_increase_pct: float = 10.0
    # Quality
    ppl_increase_pct: float = 1.5
    probe_drop_abs: float = 2.0        # percentage points, not relative %
    # Stability
    loss_slope_threshold: float = 0.001

    def validate(self) -> None:
        """Raise ValueError if any threshold is invalid."""
        if self.throughput_drop_pct <= 0:
            raise ValueError(f"throughput_drop_pct must be > 0, got {self.throughput_drop_pct}")
        if self.step_time_increase_pct <= 0:
            raise ValueError(
                f"step_time_increase_pct must be > 0, got {self.step_time_increase_pct}"
            )
        if self.memory_increase_pct <= 0:
            raise ValueError(f"memory_increase_pct must be > 0, got {self.memory_increase_pct}")
        if self.ppl_increase_pct <= 0:
            raise ValueError(f"ppl_increase_pct must be > 0, got {self.ppl_increase_pct}")
        if self.probe_drop_abs < 0:
            raise ValueError(f"probe_drop_abs must be >= 0, got {self.probe_drop_abs}")
        if self.loss_slope_threshold <= 0:
            raise ValueError(
                f"loss_slope_threshold must be > 0, got {self.loss_slope_threshold}"
            )


# ===========================================================================
# Result Dataclasses
# ===========================================================================

@dataclass
class BenchResult:
    """
    Output of BenchTrain.run(). Contains all measured metrics.
    """
    # Timing
    step_times: List[float]            # All step times (seconds)
    step_time_mean: float
    step_time_p50: float
    step_time_p90: float
    step_time_min: float
    step_time_max: float

    # Throughput
    tokens_per_sec_mean: float
    tokens_per_sec_p50: float
    tokens_per_sec_p10: float
    tokens_per_step: int

    # Memory
    peak_allocated_bytes: int
    peak_allocated_gb: float
    peak_reserved_bytes: int
    peak_reserved_gb: float

    # MFU
    mfu_p50: Optional[float]           # None if GPU not in registry
    mfu_mean: Optional[float]
    n_non_embedding_params: int
    peak_tflops_per_gpu: Optional[float]
    mfu_estimator: str

    # Loss
    loss_values: List[float]
    final_loss: Optional[float]
    loss_slope: Optional[float]        # Linear regression slope

    # Config echo
    machine_profile: str
    run_config: dict

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict matching metrics.json schema."""
        import time as _time
        return {
            "schema_version": "1.0",
            "machine_profile": self.machine_profile,
            "run_config": self.run_config,
            "throughput": {
                "tokens_per_sec_mean": self.tokens_per_sec_mean,
                "tokens_per_sec_p50": self.tokens_per_sec_p50,
                "tokens_per_sec_p10": self.tokens_per_sec_p10,
                "tokens_per_step": self.tokens_per_step,
            },
            "timing": {
                "step_time_mean_s": self.step_time_mean,
                "step_time_p50_s": self.step_time_p50,
                "step_time_p90_s": self.step_time_p90,
                "step_time_min_s": self.step_time_min,
                "step_time_max_s": self.step_time_max,
            },
            "memory": {
                "peak_allocated_bytes": self.peak_allocated_bytes,
                "peak_allocated_gb": self.peak_allocated_gb,
                "peak_reserved_bytes": self.peak_reserved_bytes,
                "peak_reserved_gb": self.peak_reserved_gb,
            },
            "mfu": {
                "mfu_p50": self.mfu_p50,
                "mfu_mean": self.mfu_mean,
                "n_non_embedding_params": self.n_non_embedding_params,
                "peak_tflops_per_gpu": self.peak_tflops_per_gpu,
                "estimator": self.mfu_estimator,
            },
            "loss": {
                "final_loss": self.final_loss,
                "loss_slope": self.loss_slope,
                "loss_values": self.loss_values,
            },
            "timestamp": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        }


@dataclass
class EvalResult:
    """
    Output of EvalSmall.run(). Contains perplexity and probe accuracies.
    """
    ppl_fixed_shard: float
    task_probe_accuracy: Dict[str, float]
    machine_profile: str
    eval_config: dict

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict matching eval.json schema."""
        import time as _time
        return {
            "schema_version": "1.0",
            "machine_profile": self.machine_profile,
            "eval_config": self.eval_config,
            "ppl_fixed_shard": self.ppl_fixed_shard,
            "task_probe_accuracy": self.task_probe_accuracy,
            "timestamp": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        }


@dataclass
class CheckResult:
    """
    Result of a single metric comparison between current and baseline.
    """
    metric: str                        # e.g. "tokens_per_sec_p50"
    baseline_val: float
    current_val: float
    delta_pct: float                   # (current - baseline) / |baseline| * 100
    threshold: float                   # Gate threshold
    status: Status
    message: str

    @property
    def is_pass(self) -> bool:
        return self.status == Status.PASS

    @property
    def is_fail(self) -> bool:
        return self.status == Status.FAIL

    @property
    def is_warn(self) -> bool:
        return self.status == Status.WARN


@dataclass
class CompareResult:
    """
    Aggregated result of baseline comparison. Contains all per-metric checks.
    """
    overall_status: Status
    checks: List[CheckResult]
    report: str
    machine_profile: str

    @property
    def failed_checks(self) -> List[CheckResult]:
        return [c for c in self.checks if c.status == Status.FAIL]

    @property
    def warned_checks(self) -> List[CheckResult]:
        return [c for c in self.checks if c.status == Status.WARN]

    @property
    def passed(self) -> bool:
        return self.overall_status in (Status.PASS, Status.WARN)

    @property
    def failed(self) -> bool:
        return self.overall_status == Status.FAIL


# ===========================================================================
# YAML Utilities
# ===========================================================================

def _dataclass_to_dict(obj: Any) -> dict:
    """Recursively convert a dataclass to a plain dict."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _dataclass_to_dict(v) for k, v in asdict(obj).items()}
    return obj


def to_yaml(config: Any) -> str:
    """
    Serialize a config dataclass to YAML string.

    Raises:
        ImportError: If PyYAML is not installed.
    """
    if not _YAML_AVAILABLE:
        raise ImportError("PyYAML is required for YAML serialization. pip install pyyaml")
    d = _dataclass_to_dict(config) if hasattr(config, "__dataclass_fields__") else config
    return _yaml.safe_dump(d, default_flow_style=False, sort_keys=True)


def from_yaml(yaml_str: str, config_class: type) -> Any:
    """
    Deserialize a YAML string into a config dataclass.

    Args:
        yaml_str:     YAML content string.
        config_class: Target dataclass class (e.g., BenchConfig).

    Returns:
        Populated dataclass instance.

    Raises:
        ImportError: If PyYAML is not installed.
        TypeError:   If YAML keys don't match dataclass fields.
    """
    if not _YAML_AVAILABLE:
        raise ImportError("PyYAML is required for YAML deserialization. pip install pyyaml")
    data = _yaml.safe_load(yaml_str)
    if data is None:
        data = {}
    # Filter to only known fields
    known_fields = {f.name for f in config_class.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in known_fields}
    return config_class(**filtered)


# ===========================================================================
# Self-Tests
# ===========================================================================

def _run_self_tests() -> None:
    """Run all self-tests. Raises AssertionError on failure."""
    import json

    print("Running perf_gate_config_template self-tests...")
    failures: List[str] = []

    def check(name: str, condition: bool, msg: str = "") -> None:
        if not condition:
            failures.append(f"FAIL [{name}]: {msg}")
        else:
            print(f"  PASS  {name}")

    # --- BenchConfig defaults ---
    cfg = BenchConfig()
    check("BenchConfig.defaults_valid", cfg.warmup_steps == 200)
    check("BenchConfig.measure_steps_default", cfg.measure_steps == 100)
    check("BenchConfig.mode_default", cfg.mode == "synthetic")
    check("BenchConfig.world_size_default", cfg.world_size == 1)
    check("BenchConfig.repeat_default", cfg.repeat == 1)
    check("BenchConfig.mfu_estimator_default", cfg.mfu_estimator == "6ND")

    # --- BenchConfig.validate passes on defaults ---
    try:
        cfg.validate()
        check("BenchConfig.validate_passes_defaults", True)
    except ValueError as exc:
        check("BenchConfig.validate_passes_defaults", False, str(exc))

    # --- BenchConfig.validate catches bad values ---
    bad = BenchConfig(warmup_steps=-1)
    try:
        bad.validate()
        check("BenchConfig.validate_rejects_negative_warmup", False, "Should have raised ValueError")
    except ValueError:
        check("BenchConfig.validate_rejects_negative_warmup", True)

    bad_mode = BenchConfig(mode="invalid")
    try:
        bad_mode.validate()
        check("BenchConfig.validate_rejects_bad_mode", False, "Should have raised ValueError")
    except ValueError:
        check("BenchConfig.validate_rejects_bad_mode", True)

    # --- tokens_per_step computation ---
    cfg2 = BenchConfig(per_device_batch_size=4, seq_len=512, world_size=2, grad_accum_steps=2)
    expected_tokens = 4 * 512 * 2 * 2  # 8192
    check(
        "BenchConfig.tokens_per_step",
        cfg2.tokens_per_step == expected_tokens,
        f"Expected {expected_tokens}, got {cfg2.tokens_per_step}",
    )

    # --- EvalConfig defaults ---
    ecfg = EvalConfig()
    check("EvalConfig.temperature_zero", ecfg.temperature == 0.0)
    check("EvalConfig.seed_default", ecfg.seed == 42)
    check("EvalConfig.probes_nonempty", len(ecfg.probes) > 0)

    # --- EvalConfig.validate catches non-zero temperature ---
    bad_eval = EvalConfig(temperature=0.7)
    try:
        bad_eval.validate()
        check("EvalConfig.validate_rejects_nonzero_temp", False, "Should have raised ValueError")
    except ValueError:
        check("EvalConfig.validate_rejects_nonzero_temp", True)

    # --- ToleranceConfig defaults ---
    tol = ToleranceConfig()
    check("ToleranceConfig.throughput_drop_pct_default", tol.throughput_drop_pct == 5.0)
    check("ToleranceConfig.ppl_increase_pct_default", tol.ppl_increase_pct == 1.5)
    check("ToleranceConfig.probe_drop_abs_default", tol.probe_drop_abs == 2.0)

    # --- ToleranceConfig.validate passes on defaults ---
    try:
        tol.validate()
        check("ToleranceConfig.validate_passes_defaults", True)
    except ValueError as exc:
        check("ToleranceConfig.validate_passes_defaults", False, str(exc))

    # --- ToleranceConfig.validate catches bad values ---
    bad_tol = ToleranceConfig(throughput_drop_pct=-1.0)
    try:
        bad_tol.validate()
        check("ToleranceConfig.validate_rejects_negative_threshold", False, "Should have raised ValueError")
    except ValueError:
        check("ToleranceConfig.validate_rejects_negative_threshold", True)

    # --- GPU Registry has common GPUs ---
    required_gpus = ["H100 SXM", "A100 SXM", "A100 PCIe", "L40S", "RTX 4090", "V100"]
    for gpu in required_gpus:
        check(f"GPU_REGISTRY.has_{gpu}", gpu in GPU_PEAK_TFLOPS, f"Missing {gpu} in registry")

    # --- get_peak_tflops lookup ---
    h100_tflops = get_peak_tflops("NVIDIA H100 SXM5 80GB HBM3")
    check(
        "get_peak_tflops.H100_SXM5",
        h100_tflops == 989.0,
        f"Expected 989.0, got {h100_tflops}",
    )
    a100_tflops = get_peak_tflops("NVIDIA A100-SXM4-80GB")
    check(
        "get_peak_tflops.A100",
        a100_tflops == 312.0,
        f"Expected 312.0, got {a100_tflops}",
    )
    unknown = get_peak_tflops("NVIDIA GeForce GTX 1080 Ti")
    check(
        "get_peak_tflops.unknown_returns_None",
        unknown is None,
        f"Expected None, got {unknown}",
    )

    # --- All registry values are positive ---
    for gpu_name, dtype_map in GPU_PEAK_TFLOPS.items():
        for dtype_label, tflops_val in dtype_map.items():
            check(
                f"GPU_REGISTRY.{gpu_name}.{dtype_label}_positive",
                tflops_val > 0,
                f"{gpu_name} {dtype_label}: {tflops_val} <= 0",
            )

    # --- BenchResult.to_dict schema ---
    bench_result = BenchResult(
        step_times=[1.0, 1.1, 1.05],
        step_time_mean=1.05,
        step_time_p50=1.05,
        step_time_p90=1.09,
        step_time_min=1.0,
        step_time_max=1.1,
        tokens_per_sec_mean=10000.0,
        tokens_per_sec_p50=10000.0,
        tokens_per_sec_p10=9500.0,
        tokens_per_step=65536,
        peak_allocated_bytes=10 * 1024**3,
        peak_allocated_gb=10.0,
        peak_reserved_bytes=11 * 1024**3,
        peak_reserved_gb=11.0,
        mfu_p50=0.35,
        mfu_mean=0.34,
        n_non_embedding_params=7_000_000_000,
        peak_tflops_per_gpu=989.0,
        mfu_estimator="6ND",
        loss_values=[2.5, 2.4, 2.3],
        final_loss=2.3,
        loss_slope=-0.1,
        machine_profile="test_profile",
        run_config={"mode": "synthetic"},
    )
    d = bench_result.to_dict()
    required_sections = ["schema_version", "machine_profile", "throughput", "timing", "memory", "mfu", "loss"]
    for section in required_sections:
        check(
            f"BenchResult.to_dict.has_{section}",
            section in d,
            f"Missing section '{section}' in to_dict() output",
        )

    # --- JSON serializable ---
    try:
        json_str = json.dumps(d)
        check("BenchResult.to_dict.json_serializable", len(json_str) > 0)
    except (TypeError, ValueError) as exc:
        check("BenchResult.to_dict.json_serializable", False, str(exc))

    # --- EvalResult.to_dict schema ---
    eval_result = EvalResult(
        ppl_fixed_shard=12.34,
        task_probe_accuracy={"basic_reasoning_25": 0.92},
        machine_profile="test_profile",
        eval_config={"seed": 42},
    )
    ed = eval_result.to_dict()
    for key in ["schema_version", "ppl_fixed_shard", "task_probe_accuracy", "machine_profile"]:
        check(
            f"EvalResult.to_dict.has_{key}",
            key in ed,
            f"Missing key '{key}' in EvalResult.to_dict()",
        )

    # --- Status enum ---
    check("Status.PASS_is_string", Status.PASS == "PASS")
    check("Status.FAIL_is_string", Status.FAIL == "FAIL")
    check("Status.WARN_is_string", Status.WARN == "WARN")
    check("Status.SKIP_is_string", Status.SKIP == "SKIP")

    # --- CompareResult helpers ---
    passing = CheckResult(
        metric="foo", baseline_val=100.0, current_val=98.0,
        delta_pct=-2.0, threshold=5.0, status=Status.PASS, message="OK"
    )
    failing = CheckResult(
        metric="bar", baseline_val=100.0, current_val=90.0,
        delta_pct=-10.0, threshold=5.0, status=Status.FAIL, message="FAIL"
    )
    cr = CompareResult(
        overall_status=Status.FAIL,
        checks=[passing, failing],
        report="",
        machine_profile="test",
    )
    check("CompareResult.failed_checks_count", len(cr.failed_checks) == 1)
    check("CompareResult.passed_false_on_fail", not cr.passed)
    check("CompareResult.failed_true_on_fail", cr.failed)

    # --- YAML roundtrip ---
    if _YAML_AVAILABLE:
        yaml_str = to_yaml(cfg)
        restored = from_yaml(yaml_str, BenchConfig)
        check("YAML.roundtrip_warmup_steps", restored.warmup_steps == cfg.warmup_steps)
        check("YAML.roundtrip_mode", restored.mode == cfg.mode)
    else:
        print("  SKIP  YAML tests (pyyaml not installed)")

    # Summary
    print()
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    else:
        print(f"All perf_gate_config_template self-tests passed.")


if __name__ == "__main__":
    _run_self_tests()
