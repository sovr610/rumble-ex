"""
brain_ai/inference/config.py — Inference Optimization Configuration Dataclasses

This module provides all configuration dataclasses for the inference optimization
pipeline. Each component (batch engine, cache, async engine, memory optimizer,
latency profiler) has a dedicated config, and InferenceOptConfig aggregates them
into a single top-level configuration surface.

Key classes:
    CacheConfig          — L1/L2/L3 cache sizing and eviction policy
    MemoryConfig         — Dtype, offloading, gradient checkpointing
    AsyncConfig          — Thread pool sizing and timeout
    ProfileConfig        — Profiler sampling and output options
    BatchConfig          — Batch assembly triggers and padding
    InferenceOptConfig   — Top-level config aggregating all sub-configs

Design principles:
    1. Every field has a sensible default matching SKILL.md specification.
    2. Validation is explicit — call validate() or use __post_init__.
    3. Serialization to/from dict and JSON is supported.
    4. Immutable after construction (frozen=False but intended as read-only at runtime).

References:
    SKILL.md § Configuration Surface
    references/batch-strategies.md § Throughput vs Latency Trade-offs
    references/caching-architecture.md § Memory Budgeting
    references/memory-optimization.md § Reduced Precision Inference
"""

from __future__ import annotations

import json
import copy
import logging
from dataclasses import dataclass, field, asdict, fields
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid value sets
# ---------------------------------------------------------------------------

VALID_DTYPES = ("fp32", "fp16", "bf16")
VALID_DEVICES = ("auto", "cpu", "cuda")
VALID_EVICTION_POLICIES = ("lru", "ttl", "size_aware")
VALID_HASH_STRATEGIES = ("full", "sampled", "stats")
VALID_PADDING_STRATEGIES = ("right", "longest", "max_length")
VALID_PRIORITY_LEVELS = (0, 1, 2)

# Default bucket boundaries for sequence length bucketing
DEFAULT_BUCKETS: List[Tuple[int, int]] = [
    (0, 32),
    (33, 128),
    (129, 512),
    (513, 2048),
    (2049, 8192),
]


# ===========================================================================
# SECTION 1: CacheConfig
# ===========================================================================

@dataclass
class CacheConfig:
    """Configuration for the multi-level cache system.

    Attributes:
        enable_cache: Whether caching is enabled.
        l1_cache_mb: L1 (GPU tensor) cache budget in MB.
        l2_cache_mb: L2 (CPU pinned memory) cache budget in MB.
        l3_cache_mb: L3 (disk mmap) cache budget in MB. 0 = unlimited.
        l3_cache_dir: Directory path for L3 disk cache.
        eviction_policy: Eviction strategy for L1 and L2 caches.
        ttl_seconds: Default time-to-live for cache entries (0 = no expiry).
        hash_strategy: How to hash tensor inputs for cache keys.
        hash_sample_size: Number of elements to sample for sampled hashing.
        enable_l2: Whether to enable L2 cache level.
        enable_l3: Whether to enable L3 cache level.
        max_entries_l1: Hard limit on number of L1 entries (0 = no limit).
        max_entries_l2: Hard limit on number of L2 entries (0 = no limit).
    """

    enable_cache: bool = True
    l1_cache_mb: int = 512
    l2_cache_mb: int = 2048
    l3_cache_mb: int = 0
    l3_cache_dir: str = "/tmp/brain_ai_cache"
    eviction_policy: str = "lru"
    ttl_seconds: float = 0.0
    hash_strategy: str = "sampled"
    hash_sample_size: int = 1024
    enable_l2: bool = True
    enable_l3: bool = False
    max_entries_l1: int = 0
    max_entries_l2: int = 0

    def validate(self) -> None:
        """Validate all cache configuration values."""
        if self.l1_cache_mb < 0:
            raise ValueError(f"l1_cache_mb must be >= 0, got {self.l1_cache_mb}")
        if self.l2_cache_mb < 0:
            raise ValueError(f"l2_cache_mb must be >= 0, got {self.l2_cache_mb}")
        if self.l3_cache_mb < 0:
            raise ValueError(f"l3_cache_mb must be >= 0, got {self.l3_cache_mb}")
        if self.eviction_policy not in VALID_EVICTION_POLICIES:
            raise ValueError(
                f"eviction_policy must be one of {VALID_EVICTION_POLICIES}, "
                f"got '{self.eviction_policy}'"
            )
        if self.ttl_seconds < 0:
            raise ValueError(f"ttl_seconds must be >= 0, got {self.ttl_seconds}")
        if self.hash_strategy not in VALID_HASH_STRATEGIES:
            raise ValueError(
                f"hash_strategy must be one of {VALID_HASH_STRATEGIES}, "
                f"got '{self.hash_strategy}'"
            )
        if self.hash_sample_size < 1:
            raise ValueError(
                f"hash_sample_size must be >= 1, got {self.hash_sample_size}"
            )

    def __post_init__(self):
        self.validate()


# ===========================================================================
# SECTION 2: MemoryConfig
# ===========================================================================

@dataclass
class MemoryConfig:
    """Configuration for memory optimization.

    Attributes:
        dtype: Target dtype for model parameters during inference.
        gradient_checkpointing: Enable gradient checkpointing to reduce memory.
        cpu_offload_modules: List of module names to offload to CPU.
        use_pinned_memory: Use pinned CPU memory for offloaded modules.
        inference_mode: Use torch.inference_mode() context.
        mixed_precision_modules: Module names to keep in FP32 when using FP16.
        auto_optimize: Automatically apply optimizations based on GPU memory.
        target_gpu_memory_mb: Target GPU memory budget for auto_optimize.
        enable_memory_tracking: Track memory usage during inference.
        empty_cache_after_offload: Call torch.cuda.empty_cache() after offloading.
    """

    dtype: str = "fp32"
    gradient_checkpointing: bool = False
    cpu_offload_modules: List[str] = field(default_factory=list)
    use_pinned_memory: bool = False
    inference_mode: bool = True
    mixed_precision_modules: List[str] = field(
        default_factory=lambda: ["htm", "active_inference", "neuromodulation"]
    )
    auto_optimize: bool = False
    target_gpu_memory_mb: float = 0.0
    enable_memory_tracking: bool = False
    empty_cache_after_offload: bool = True

    def validate(self) -> None:
        """Validate all memory configuration values."""
        if self.dtype not in VALID_DTYPES:
            raise ValueError(
                f"dtype must be one of {VALID_DTYPES}, got '{self.dtype}'"
            )
        if self.target_gpu_memory_mb < 0:
            raise ValueError(
                f"target_gpu_memory_mb must be >= 0, got {self.target_gpu_memory_mb}"
            )
        if not isinstance(self.cpu_offload_modules, list):
            raise ValueError("cpu_offload_modules must be a list of strings")
        for mod in self.cpu_offload_modules:
            if not isinstance(mod, str):
                raise ValueError(
                    f"Each cpu_offload_module must be a string, got {type(mod)}"
                )

    def __post_init__(self):
        # Convert tuples to lists for compatibility
        if isinstance(self.cpu_offload_modules, tuple):
            self.cpu_offload_modules = list(self.cpu_offload_modules)
        if isinstance(self.mixed_precision_modules, tuple):
            self.mixed_precision_modules = list(self.mixed_precision_modules)
        self.validate()

    @property
    def torch_dtype(self):
        """Convert string dtype to torch dtype."""
        import torch
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        return dtype_map[self.dtype]


# ===========================================================================
# SECTION 3: AsyncConfig
# ===========================================================================

@dataclass
class AsyncConfig:
    """Configuration for async inference engine.

    Attributes:
        enable_async: Whether async inference is enabled.
        num_inference_threads: Number of worker threads in the pool.
        num_model_copies: Number of model copies for thread-safe inference.
        request_timeout_seconds: Timeout for individual inference requests.
        shutdown_timeout_seconds: Timeout for graceful shutdown.
        max_queue_size: Maximum number of pending requests.
        use_cuda_streams: Use separate CUDA streams per thread.
        circuit_breaker_threshold: Number of failures before circuit opens.
        circuit_breaker_reset_seconds: Time before circuit breaker resets.
    """

    enable_async: bool = False
    num_inference_threads: int = 2
    num_model_copies: int = 2
    request_timeout_seconds: float = 5.0
    shutdown_timeout_seconds: float = 10.0
    max_queue_size: int = 100
    use_cuda_streams: bool = False
    circuit_breaker_threshold: int = 5
    circuit_breaker_reset_seconds: float = 60.0

    def validate(self) -> None:
        """Validate all async configuration values."""
        if self.num_inference_threads < 1:
            raise ValueError(
                f"num_inference_threads must be >= 1, got {self.num_inference_threads}"
            )
        if self.num_model_copies < 1:
            raise ValueError(
                f"num_model_copies must be >= 1, got {self.num_model_copies}"
            )
        if self.request_timeout_seconds <= 0:
            raise ValueError(
                f"request_timeout_seconds must be > 0, "
                f"got {self.request_timeout_seconds}"
            )
        if self.shutdown_timeout_seconds <= 0:
            raise ValueError(
                f"shutdown_timeout_seconds must be > 0, "
                f"got {self.shutdown_timeout_seconds}"
            )
        if self.max_queue_size < 1:
            raise ValueError(
                f"max_queue_size must be >= 1, got {self.max_queue_size}"
            )
        if self.circuit_breaker_threshold < 1:
            raise ValueError(
                f"circuit_breaker_threshold must be >= 1, "
                f"got {self.circuit_breaker_threshold}"
            )

    def __post_init__(self):
        self.validate()


# ===========================================================================
# SECTION 4: ProfileConfig
# ===========================================================================

@dataclass
class ProfileConfig:
    """Configuration for latency profiling.

    Attributes:
        enable_profiling: Whether profiling is active.
        n_warmup_runs: Number of warmup runs before profiling.
        n_profile_runs: Number of profiling runs to average.
        export_chrome_trace: Whether to export Chrome trace JSON.
        trace_output_path: Path for Chrome trace output.
        profile_memory: Also profile memory alongside latency.
        percentiles: Latency percentiles to compute.
        per_module: Enable per-module breakdown.
    """

    enable_profiling: bool = False
    n_warmup_runs: int = 10
    n_profile_runs: int = 100
    export_chrome_trace: bool = False
    trace_output_path: str = "/tmp/brain_ai_trace.json"
    profile_memory: bool = False
    percentiles: List[float] = field(default_factory=lambda: [50.0, 90.0, 95.0, 99.0])
    per_module: bool = True

    def validate(self) -> None:
        """Validate all profiling configuration values."""
        if self.n_warmup_runs < 0:
            raise ValueError(
                f"n_warmup_runs must be >= 0, got {self.n_warmup_runs}"
            )
        if self.n_profile_runs < 1:
            raise ValueError(
                f"n_profile_runs must be >= 1, got {self.n_profile_runs}"
            )
        for p in self.percentiles:
            if not (0.0 <= p <= 100.0):
                raise ValueError(
                    f"Percentile must be in [0, 100], got {p}"
                )

    def __post_init__(self):
        if isinstance(self.percentiles, tuple):
            self.percentiles = list(self.percentiles)
        self.validate()


# ===========================================================================
# SECTION 5: BatchConfig
# ===========================================================================

@dataclass
class BatchConfig:
    """Configuration for batch inference engine.

    Attributes:
        batch_size: Default batch size for inference.
        max_batch_size: Maximum batch size (larger batches are split).
        batch_timeout_ms: Timeout before dispatching a partial batch (ms).
        padding_strategy: How to pad variable-length inputs.
        pad_value: Value to use for padding.
        enable_bucketing: Group requests by sequence length.
        bucket_boundaries: Sequence length bucket boundaries.
        max_queue_size: Maximum pending requests in queue.
        warmup_steps: Number of warmup iterations.
        enable_early_exit: Enable dual-process early exit.
        confidence_threshold: Threshold for System 1 early exit.
    """

    batch_size: int = 1
    max_batch_size: int = 64
    batch_timeout_ms: float = 10.0
    padding_strategy: str = "longest"
    pad_value: float = 0.0
    enable_bucketing: bool = False
    bucket_boundaries: List[Tuple[int, int]] = field(
        default_factory=lambda: list(DEFAULT_BUCKETS)
    )
    max_queue_size: int = 256
    warmup_steps: int = 10
    enable_early_exit: bool = True
    confidence_threshold: float = 0.7

    def validate(self) -> None:
        """Validate all batch configuration values."""
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.max_batch_size < 1:
            raise ValueError(
                f"max_batch_size must be >= 1, got {self.max_batch_size}"
            )
        if self.batch_size > self.max_batch_size:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be <= max_batch_size "
                f"({self.max_batch_size})"
            )
        if self.batch_timeout_ms < 0:
            raise ValueError(
                f"batch_timeout_ms must be >= 0, got {self.batch_timeout_ms}"
            )
        if self.padding_strategy not in VALID_PADDING_STRATEGIES:
            raise ValueError(
                f"padding_strategy must be one of {VALID_PADDING_STRATEGIES}, "
                f"got '{self.padding_strategy}'"
            )
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be in [0, 1], "
                f"got {self.confidence_threshold}"
            )
        if self.warmup_steps < 0:
            raise ValueError(
                f"warmup_steps must be >= 0, got {self.warmup_steps}"
            )
        if self.max_queue_size < 1:
            raise ValueError(
                f"max_queue_size must be >= 1, got {self.max_queue_size}"
            )

    def __post_init__(self):
        if isinstance(self.bucket_boundaries, tuple):
            self.bucket_boundaries = list(self.bucket_boundaries)
        self.validate()


# ===========================================================================
# SECTION 6: InferenceOptConfig (top-level aggregator)
# ===========================================================================

@dataclass
class InferenceOptConfig:
    """Top-level inference optimization configuration.

    Aggregates all sub-configs and provides convenience accessors matching
    the SKILL.md specification. Fields at this level are shortcuts; the
    sub-configs hold the detailed settings.

    Attributes:
        batch_size: Default batch size (shortcut for batch.batch_size).
        max_batch_size: Maximum batch size (shortcut for batch.max_batch_size).
        device: Target device for inference.
        dtype: Target dtype for inference.
        enable_cache: Enable caching (shortcut for cache.enable_cache).
        l1_cache_mb: L1 cache budget (shortcut for cache.l1_cache_mb).
        l2_cache_mb: L2 cache budget (shortcut for cache.l2_cache_mb).
        enable_async: Enable async inference (shortcut for async_cfg.enable_async).
        num_inference_threads: Thread count (shortcut for async_cfg).
        enable_early_exit: Enable early exit (shortcut for batch.enable_early_exit).
        confidence_threshold: Early exit threshold (shortcut for batch).
        gradient_checkpointing: Enable checkpointing (shortcut for memory).
        cpu_offload_modules: Modules to offload (shortcut for memory).
        enable_profiling: Enable profiling (shortcut for profile).
        warmup_steps: Warmup iterations (shortcut for batch.warmup_steps).
        cache: Detailed cache configuration.
        memory: Detailed memory configuration.
        async_cfg: Detailed async configuration.
        profile: Detailed profile configuration.
        batch: Detailed batch configuration.
    """

    # Top-level shortcuts (match SKILL.md specification)
    batch_size: int = 1
    max_batch_size: int = 64
    device: str = "auto"
    dtype: str = "fp16"
    enable_cache: bool = True
    l1_cache_mb: int = 512
    l2_cache_mb: int = 2048
    enable_async: bool = False
    num_inference_threads: int = 2
    enable_early_exit: bool = True
    confidence_threshold: float = 0.7
    gradient_checkpointing: bool = False
    cpu_offload_modules: List[str] = field(default_factory=list)
    enable_profiling: bool = False
    warmup_steps: int = 10

    # Sub-configs (created in __post_init__ if not provided)
    cache: Optional[CacheConfig] = None
    memory: Optional[MemoryConfig] = None
    async_cfg: Optional[AsyncConfig] = None
    profile: Optional[ProfileConfig] = None
    batch: Optional[BatchConfig] = None

    def __post_init__(self):
        # Convert tuples to lists
        if isinstance(self.cpu_offload_modules, tuple):
            self.cpu_offload_modules = list(self.cpu_offload_modules)

        # Create sub-configs from top-level shortcuts if not provided
        if self.cache is None:
            self.cache = CacheConfig(
                enable_cache=self.enable_cache,
                l1_cache_mb=self.l1_cache_mb,
                l2_cache_mb=self.l2_cache_mb,
            )
        if self.memory is None:
            self.memory = MemoryConfig(
                dtype=self.dtype,
                gradient_checkpointing=self.gradient_checkpointing,
                cpu_offload_modules=list(self.cpu_offload_modules),
            )
        if self.async_cfg is None:
            self.async_cfg = AsyncConfig(
                enable_async=self.enable_async,
                num_inference_threads=self.num_inference_threads,
            )
        if self.profile is None:
            self.profile = ProfileConfig(
                enable_profiling=self.enable_profiling,
            )
        if self.batch is None:
            self.batch = BatchConfig(
                batch_size=self.batch_size,
                max_batch_size=self.max_batch_size,
                warmup_steps=self.warmup_steps,
                enable_early_exit=self.enable_early_exit,
                confidence_threshold=self.confidence_threshold,
            )

        self.validate()

    def validate(self) -> None:
        """Validate the top-level config and all sub-configs."""
        if self.max_batch_size < 1:
            raise ValueError(
                f"max_batch_size must be >= 1, got {self.max_batch_size}"
            )
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.device not in VALID_DEVICES and not self.device.startswith("cuda:"):
            raise ValueError(
                f"device must be one of {VALID_DEVICES} or 'cuda:N', "
                f"got '{self.device}'"
            )
        if self.dtype not in VALID_DTYPES:
            raise ValueError(
                f"dtype must be one of {VALID_DTYPES}, got '{self.dtype}'"
            )
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be in [0, 1], "
                f"got {self.confidence_threshold}"
            )
        if self.warmup_steps < 0:
            raise ValueError(
                f"warmup_steps must be >= 0, got {self.warmup_steps}"
            )
        if self.l1_cache_mb < 0:
            raise ValueError(
                f"l1_cache_mb must be >= 0, got {self.l1_cache_mb}"
            )

        # Validate sub-configs
        if self.cache is not None:
            self.cache.validate()
        if self.memory is not None:
            self.memory.validate()
        if self.async_cfg is not None:
            self.async_cfg.validate()
        if self.profile is not None:
            self.profile.validate()
        if self.batch is not None:
            self.batch.validate()

    def resolve_device(self) -> str:
        """Resolve 'auto' device to an actual device string."""
        import torch
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the entire config to a nested dict."""
        result = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if hasattr(val, '__dataclass_fields__'):
                result[f.name] = asdict(val)
            else:
                result[f.name] = copy.deepcopy(val)
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "InferenceOptConfig":
        """Reconstruct config from a dict (as produced by to_dict)."""
        d = copy.deepcopy(d)

        # Reconstruct sub-configs
        sub_config_map = {
            "cache": CacheConfig,
            "memory": MemoryConfig,
            "async_cfg": AsyncConfig,
            "profile": ProfileConfig,
            "batch": BatchConfig,
        }
        for key, config_cls in sub_config_map.items():
            if key in d and isinstance(d[key], dict):
                d[key] = config_cls(**d[key])

        return cls(**d)

    def to_json(self) -> str:
        """Serialize config to JSON string."""
        d = self.to_dict()
        # Convert tuples in bucket_boundaries to lists for JSON
        if "batch" in d and isinstance(d["batch"], dict):
            if "bucket_boundaries" in d["batch"]:
                d["batch"]["bucket_boundaries"] = [
                    list(b) for b in d["batch"]["bucket_boundaries"]
                ]
        return json.dumps(d, indent=2, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "InferenceOptConfig":
        """Reconstruct config from a JSON string."""
        d = json.loads(json_str)
        # Convert bucket boundary lists back to tuples
        if "batch" in d and isinstance(d["batch"], dict):
            if "bucket_boundaries" in d["batch"]:
                d["batch"]["bucket_boundaries"] = [
                    tuple(b) for b in d["batch"]["bucket_boundaries"]
                ]
        return cls.from_dict(d)

    @classmethod
    def minimal(cls) -> "InferenceOptConfig":
        """Create minimal config suitable for testing."""
        return cls(
            batch_size=1,
            max_batch_size=8,
            device="cpu",
            dtype="fp32",
            enable_cache=True,
            l1_cache_mb=64,
            l2_cache_mb=128,
            enable_async=False,
            warmup_steps=2,
        )

    @classmethod
    def production(cls) -> "InferenceOptConfig":
        """Create production config for GPU inference."""
        return cls(
            batch_size=8,
            max_batch_size=64,
            device="auto",
            dtype="fp16",
            enable_cache=True,
            l1_cache_mb=512,
            l2_cache_mb=2048,
            enable_async=True,
            num_inference_threads=4,
            enable_early_exit=True,
            confidence_threshold=0.7,
            warmup_steps=10,
        )

    def summary(self) -> str:
        """Return a human-readable summary of the config."""
        lines = [
            "InferenceOptConfig:",
            f"  device={self.device} dtype={self.dtype}",
            f"  batch_size={self.batch_size} max_batch_size={self.max_batch_size}",
            f"  cache={'ON' if self.enable_cache else 'OFF'}"
            f" (L1={self.l1_cache_mb}MB, L2={self.l2_cache_mb}MB)",
            f"  async={'ON' if self.enable_async else 'OFF'}"
            f" (threads={self.num_inference_threads})",
            f"  early_exit={'ON' if self.enable_early_exit else 'OFF'}"
            f" (threshold={self.confidence_threshold})",
            f"  profiling={'ON' if self.enable_profiling else 'OFF'}",
            f"  warmup_steps={self.warmup_steps}",
        ]
        return "\n".join(lines)


# ===========================================================================
# SECTION 7: InferenceResult dataclass
# ===========================================================================

@dataclass
class InferenceResult:
    """Result container for a single inference.

    Attributes:
        output: The model output tensor (None if error).
        error: Error message (None if success).
        latency_ms: Inference latency in milliseconds.
        cache_hit: Whether this result came from cache.
        request_idx: Index within the batch (for batch results).
        metadata: Arbitrary metadata dict.
    """

    output: Any = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    cache_hit: bool = False
    request_idx: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Whether inference succeeded."""
        return self.error is None and self.output is not None

    def __repr__(self) -> str:
        if self.success:
            shape = (
                tuple(self.output.shape) if hasattr(self.output, "shape") else "?"
            )
            return (
                f"InferenceResult(shape={shape}, "
                f"latency={self.latency_ms:.2f}ms, "
                f"cache_hit={self.cache_hit})"
            )
        return f"InferenceResult(error='{self.error}')"


# ===========================================================================
# SECTION 8: ProfileReport and Bottleneck dataclasses
# ===========================================================================

@dataclass
class Bottleneck:
    """A single bottleneck identified by profiling.

    Attributes:
        module_name: Name of the bottleneck module.
        latency_ms: Average latency in milliseconds.
        percentage: Percentage of total latency.
        recommendation: Suggested optimization.
    """

    module_name: str = ""
    latency_ms: float = 0.0
    percentage: float = 0.0
    recommendation: str = ""


@dataclass
class ProfileReport:
    """Report produced by LatencyProfiler.profile().

    Attributes:
        total_latency_ms: Mean total forward pass latency.
        per_module_ms: Per-module latency breakdown.
        n_runs: Number of profiling runs.
        std_ms: Standard deviation of total latency.
        p50_ms: 50th percentile latency.
        p99_ms: 99th percentile latency.
        p90_ms: 90th percentile latency.
        p95_ms: 95th percentile latency.
        bottlenecks: List of identified bottlenecks.
        latencies_ms: Raw latency measurements.
    """

    total_latency_ms: float = 0.0
    per_module_ms: Dict[str, float] = field(default_factory=dict)
    n_runs: int = 0
    std_ms: float = 0.0
    p50_ms: float = 0.0
    p99_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    bottlenecks: List[Bottleneck] = field(default_factory=list)
    latencies_ms: List[float] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"ProfileReport (n_runs={self.n_runs}):",
            f"  Total: {self.total_latency_ms:.3f}ms "
            f"(std={self.std_ms:.3f}ms)",
            f"  p50={self.p50_ms:.3f}ms  p90={self.p90_ms:.3f}ms  "
            f"p99={self.p99_ms:.3f}ms",
        ]
        if self.per_module_ms:
            lines.append("  Per-module breakdown:")
            for name, ms in sorted(
                self.per_module_ms.items(), key=lambda x: x[1], reverse=True
            ):
                pct = (ms / self.total_latency_ms * 100) if self.total_latency_ms > 0 else 0
                lines.append(f"    {name}: {ms:.3f}ms ({pct:.1f}%)")
        return "\n".join(lines)


# ===========================================================================
# SECTION 9: MemoryReport dataclass
# ===========================================================================

@dataclass
class MemoryReport:
    """Report produced by MemoryOptimizer.measure_memory().

    Attributes:
        model_size_mb: Total model parameters + buffers in MB.
        param_size_mb: Parameters only in MB.
        buffer_size_mb: Buffers only in MB.
        peak_inference_mb: Peak GPU memory during inference.
        activation_mb: Activation memory (peak - model).
        dtype: Current model dtype string.
        device: Current device string.
        per_module: Per-module memory breakdown.
        recommendations: Optimization suggestions.
    """

    model_size_mb: float = 0.0
    param_size_mb: float = 0.0
    buffer_size_mb: float = 0.0
    peak_inference_mb: float = 0.0
    activation_mb: float = 0.0
    dtype: str = "fp32"
    device: str = "cpu"
    per_module: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Model: {self.model_size_mb:.1f}MB ({self.dtype} on {self.device})",
            f"  Params: {self.param_size_mb:.1f}MB  Buffers: {self.buffer_size_mb:.1f}MB",
            f"Peak inference: {self.peak_inference_mb:.1f}MB",
            f"Activations: {self.activation_mb:.1f}MB",
        ]
        if self.per_module:
            lines.append("Per-module memory:")
            for name, mb in sorted(
                self.per_module.items(), key=lambda x: x[1], reverse=True
            ):
                lines.append(f"  {name}: {mb:.2f}MB")
        if self.recommendations:
            lines.append("Recommendations:")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")
        return "\n".join(lines)


# ===========================================================================
# SECTION 10: CacheStats dataclass
# ===========================================================================

@dataclass
class CacheStats:
    """Statistics for the cache system.

    Attributes:
        hits: Total cache hits across all levels.
        misses: Total cache misses.
        evictions: Total evictions across all levels.
        l1_size_mb: Current L1 cache size in MB.
        l2_size_mb: Current L2 cache size in MB.
        l3_size_mb: Current L3 cache size in MB.
        l1_entries: Number of entries in L1.
        l2_entries: Number of entries in L2.
        l3_entries: Number of entries in L3.
        l1_hits: L1-specific hits.
        l1_misses: L1-specific misses.
        l2_hits: L2-specific hits.
        l2_misses: L2-specific misses.
    """

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    l1_size_mb: float = 0.0
    l2_size_mb: float = 0.0
    l3_size_mb: float = 0.0
    l1_entries: int = 0
    l2_entries: int = 0
    l3_entries: int = 0
    l1_hits: int = 0
    l1_misses: int = 0
    l2_hits: int = 0
    l2_misses: int = 0

    @property
    def hit_rate(self) -> float:
        """Overall cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def l1_hit_rate(self) -> float:
        """L1-specific hit rate."""
        total = self.l1_hits + self.l1_misses
        return self.l1_hits / total if total > 0 else 0.0

    @property
    def total_size_mb(self) -> float:
        """Total cache size across all levels."""
        return self.l1_size_mb + self.l2_size_mb + self.l3_size_mb

    @property
    def total_entries(self) -> int:
        """Total number of cached entries."""
        return self.l1_entries + self.l2_entries + self.l3_entries

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"CacheStats: hit_rate={self.hit_rate:.1%} "
            f"({self.hits}/{self.hits + self.misses}) "
            f"entries={self.total_entries} "
            f"size={self.total_size_mb:.1f}MB "
            f"evictions={self.evictions}"
        )


# ===========================================================================
# SECTION 11: Self-tests
# ===========================================================================

def _run_self_tests():
    """Run self-tests for all configuration dataclasses."""
    import traceback

    passed = 0
    failed = 0
    test_results = []

    def _test(name: str, fn):
        nonlocal passed, failed
        try:
            fn()
            passed += 1
            test_results.append(f"  PASS: {name}")
        except Exception as e:
            failed += 1
            test_results.append(f"  FAIL: {name} — {e}")
            traceback.print_exc()

    # --- CacheConfig tests ---
    def test_cache_config_defaults():
        cfg = CacheConfig()
        assert cfg.enable_cache is True
        assert cfg.l1_cache_mb == 512
        assert cfg.l2_cache_mb == 2048
        assert cfg.eviction_policy == "lru"

    def test_cache_config_negative_l1():
        try:
            CacheConfig(l1_cache_mb=-1)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_cache_config_invalid_eviction():
        try:
            CacheConfig(eviction_policy="random")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_cache_config_invalid_hash():
        try:
            CacheConfig(hash_strategy="md5")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_cache_config_negative_ttl():
        try:
            CacheConfig(ttl_seconds=-5.0)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    # --- MemoryConfig tests ---
    def test_memory_config_defaults():
        cfg = MemoryConfig()
        assert cfg.dtype == "fp32"
        assert cfg.inference_mode is True
        assert cfg.gradient_checkpointing is False

    def test_memory_config_invalid_dtype():
        try:
            MemoryConfig(dtype="int4")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_memory_config_torch_dtype():
        import torch
        cfg = MemoryConfig(dtype="fp16")
        assert cfg.torch_dtype == torch.float16

    def test_memory_config_tuple_offload():
        cfg = MemoryConfig(cpu_offload_modules=("a", "b"))
        assert isinstance(cfg.cpu_offload_modules, list)
        assert cfg.cpu_offload_modules == ["a", "b"]

    # --- AsyncConfig tests ---
    def test_async_config_defaults():
        cfg = AsyncConfig()
        assert cfg.enable_async is False
        assert cfg.num_inference_threads == 2

    def test_async_config_invalid_threads():
        try:
            AsyncConfig(num_inference_threads=0)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_async_config_invalid_timeout():
        try:
            AsyncConfig(request_timeout_seconds=-1.0)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_async_config_invalid_queue():
        try:
            AsyncConfig(max_queue_size=0)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    # --- ProfileConfig tests ---
    def test_profile_config_defaults():
        cfg = ProfileConfig()
        assert cfg.enable_profiling is False
        assert cfg.n_profile_runs == 100

    def test_profile_config_invalid_runs():
        try:
            ProfileConfig(n_profile_runs=0)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_profile_config_invalid_percentile():
        try:
            ProfileConfig(percentiles=[101.0])
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    # --- BatchConfig tests ---
    def test_batch_config_defaults():
        cfg = BatchConfig()
        assert cfg.batch_size == 1
        assert cfg.max_batch_size == 64
        assert cfg.confidence_threshold == 0.7

    def test_batch_config_zero_batch():
        try:
            BatchConfig(max_batch_size=0)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_batch_config_invalid_threshold():
        try:
            BatchConfig(confidence_threshold=1.5)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_batch_config_invalid_padding():
        try:
            BatchConfig(padding_strategy="center")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_batch_config_batch_exceeds_max():
        try:
            BatchConfig(batch_size=128, max_batch_size=64)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    # --- InferenceOptConfig tests ---
    def test_opt_config_defaults():
        cfg = InferenceOptConfig()
        assert cfg.batch_size == 1
        assert cfg.max_batch_size == 64
        assert cfg.dtype == "fp16"
        assert cfg.enable_cache is True
        assert cfg.confidence_threshold == 0.7
        assert cfg.warmup_steps == 10

    def test_opt_config_sub_configs_created():
        cfg = InferenceOptConfig()
        assert cfg.cache is not None
        assert cfg.memory is not None
        assert cfg.async_cfg is not None
        assert cfg.profile is not None
        assert cfg.batch is not None

    def test_opt_config_invalid_dtype():
        try:
            InferenceOptConfig(dtype="int4")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_opt_config_invalid_device():
        try:
            InferenceOptConfig(device="tpu")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_opt_config_invalid_confidence():
        try:
            InferenceOptConfig(confidence_threshold=1.5)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_opt_config_to_dict():
        cfg = InferenceOptConfig(batch_size=4, dtype="fp32")
        d = cfg.to_dict()
        assert d["batch_size"] == 4
        assert d["dtype"] == "fp32"
        assert isinstance(d["cache"], dict)

    def test_opt_config_from_dict():
        cfg1 = InferenceOptConfig(batch_size=16, dtype="fp32")
        d = cfg1.to_dict()
        cfg2 = InferenceOptConfig.from_dict(d)
        assert cfg2.batch_size == 16
        assert cfg2.dtype == "fp32"

    def test_opt_config_to_json():
        cfg = InferenceOptConfig(batch_size=8)
        j = cfg.to_json()
        parsed = json.loads(j)
        assert parsed["batch_size"] == 8

    def test_opt_config_from_json():
        cfg1 = InferenceOptConfig(batch_size=32, dtype="fp32")
        j = cfg1.to_json()
        cfg2 = InferenceOptConfig.from_json(j)
        assert cfg2.batch_size == 32
        assert cfg2.dtype == "fp32"

    def test_opt_config_roundtrip():
        cfg1 = InferenceOptConfig(
            batch_size=4, max_batch_size=32, dtype="bf16",
            enable_cache=False, warmup_steps=5,
        )
        j = cfg1.to_json()
        cfg2 = InferenceOptConfig.from_json(j)
        assert cfg2.batch_size == cfg1.batch_size
        assert cfg2.max_batch_size == cfg1.max_batch_size
        assert cfg2.dtype == cfg1.dtype
        assert cfg2.enable_cache == cfg1.enable_cache
        assert cfg2.warmup_steps == cfg1.warmup_steps

    def test_opt_config_minimal():
        cfg = InferenceOptConfig.minimal()
        assert cfg.device == "cpu"
        assert cfg.dtype == "fp32"
        assert cfg.max_batch_size == 8

    def test_opt_config_production():
        cfg = InferenceOptConfig.production()
        assert cfg.batch_size == 8
        assert cfg.enable_async is True

    def test_opt_config_summary():
        cfg = InferenceOptConfig()
        s = cfg.summary()
        assert "InferenceOptConfig" in s
        assert "fp16" in s

    def test_opt_config_resolve_device():
        cfg = InferenceOptConfig(device="cpu")
        assert cfg.resolve_device() == "cpu"

    # --- InferenceResult tests ---
    def test_inference_result_success():
        import torch
        r = InferenceResult(output=torch.randn(2, 10))
        assert r.success is True
        assert "shape" in repr(r)

    def test_inference_result_error():
        r = InferenceResult(error="OOM")
        assert r.success is False
        assert "OOM" in repr(r)

    # --- CacheStats tests ---
    def test_cache_stats_hit_rate():
        s = CacheStats(hits=80, misses=20)
        assert abs(s.hit_rate - 0.8) < 1e-6

    def test_cache_stats_hit_rate_empty():
        s = CacheStats()
        assert s.hit_rate == 0.0

    def test_cache_stats_total():
        s = CacheStats(l1_entries=10, l2_entries=5, l3_entries=3)
        assert s.total_entries == 18

    def test_cache_stats_summary():
        s = CacheStats(hits=50, misses=50, evictions=10)
        text = s.summary()
        assert "50.0%" in text

    # --- Bottleneck tests ---
    def test_bottleneck_creation():
        b = Bottleneck(module_name="slow", latency_ms=100.0, percentage=80.0)
        assert b.module_name == "slow"

    # --- ProfileReport tests ---
    def test_profile_report_summary():
        r = ProfileReport(
            total_latency_ms=10.0, n_runs=100, std_ms=1.0,
            p50_ms=9.5, p90_ms=11.0, p99_ms=15.0,
            per_module_ms={"layer1": 5.0, "layer2": 5.0},
        )
        s = r.summary()
        assert "10.000ms" in s
        assert "layer1" in s

    # --- MemoryReport tests ---
    def test_memory_report_summary():
        r = MemoryReport(
            model_size_mb=100.0, param_size_mb=90.0, buffer_size_mb=10.0,
            dtype="fp32", device="cpu",
            recommendations=["Use FP16"],
        )
        s = r.summary()
        assert "100.0MB" in s
        assert "Use FP16" in s

    # Run all tests
    tests = [
        ("CacheConfig defaults", test_cache_config_defaults),
        ("CacheConfig negative L1", test_cache_config_negative_l1),
        ("CacheConfig invalid eviction", test_cache_config_invalid_eviction),
        ("CacheConfig invalid hash", test_cache_config_invalid_hash),
        ("CacheConfig negative TTL", test_cache_config_negative_ttl),
        ("MemoryConfig defaults", test_memory_config_defaults),
        ("MemoryConfig invalid dtype", test_memory_config_invalid_dtype),
        ("MemoryConfig torch_dtype", test_memory_config_torch_dtype),
        ("MemoryConfig tuple offload", test_memory_config_tuple_offload),
        ("AsyncConfig defaults", test_async_config_defaults),
        ("AsyncConfig invalid threads", test_async_config_invalid_threads),
        ("AsyncConfig invalid timeout", test_async_config_invalid_timeout),
        ("AsyncConfig invalid queue", test_async_config_invalid_queue),
        ("ProfileConfig defaults", test_profile_config_defaults),
        ("ProfileConfig invalid runs", test_profile_config_invalid_runs),
        ("ProfileConfig invalid percentile", test_profile_config_invalid_percentile),
        ("BatchConfig defaults", test_batch_config_defaults),
        ("BatchConfig zero batch", test_batch_config_zero_batch),
        ("BatchConfig invalid threshold", test_batch_config_invalid_threshold),
        ("BatchConfig invalid padding", test_batch_config_invalid_padding),
        ("BatchConfig batch exceeds max", test_batch_config_batch_exceeds_max),
        ("InferenceOptConfig defaults", test_opt_config_defaults),
        ("InferenceOptConfig sub-configs created", test_opt_config_sub_configs_created),
        ("InferenceOptConfig invalid dtype", test_opt_config_invalid_dtype),
        ("InferenceOptConfig invalid device", test_opt_config_invalid_device),
        ("InferenceOptConfig invalid confidence", test_opt_config_invalid_confidence),
        ("InferenceOptConfig to_dict", test_opt_config_to_dict),
        ("InferenceOptConfig from_dict", test_opt_config_from_dict),
        ("InferenceOptConfig to_json", test_opt_config_to_json),
        ("InferenceOptConfig from_json", test_opt_config_from_json),
        ("InferenceOptConfig roundtrip", test_opt_config_roundtrip),
        ("InferenceOptConfig minimal", test_opt_config_minimal),
        ("InferenceOptConfig production", test_opt_config_production),
        ("InferenceOptConfig summary", test_opt_config_summary),
        ("InferenceOptConfig resolve_device", test_opt_config_resolve_device),
        ("InferenceResult success", test_inference_result_success),
        ("InferenceResult error", test_inference_result_error),
        ("CacheStats hit_rate", test_cache_stats_hit_rate),
        ("CacheStats hit_rate empty", test_cache_stats_hit_rate_empty),
        ("CacheStats total", test_cache_stats_total),
        ("CacheStats summary", test_cache_stats_summary),
        ("Bottleneck creation", test_bottleneck_creation),
        ("ProfileReport summary", test_profile_report_summary),
        ("MemoryReport summary", test_memory_report_summary),
    ]

    print(f"Running {len(tests)} self-tests for inference_opt_config_template...")
    for name, fn in tests:
        _test(name, fn)

    print("\n".join(test_results))
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    if failed == 0:
        print("ALL TESTS PASSED")
    return failed == 0


if __name__ == "__main__":
    _run_self_tests()
