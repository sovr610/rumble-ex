#!/usr/bin/env python3
"""
distributed_config_template.py
-------------------------------
All configuration dataclasses for the distributed-memory-scaling skill.

Includes:
  - DistributedConfig: top-level strategy selection
  - FSDPConfig: FSDP-specific wrapping and precision settings
  - DeepSpeedConfig: ZeRO stage and offload settings
  - OptimizerConfig: optimizer hyperparameters
  - StrategyContext: runtime state after distributed setup
  - ScalingReport: result of a scaling efficiency measurement
  - BenchResult: raw timing data from a single benchmark run

All configs support to_dict() / from_dict() serialization and validate()
cross-field invariants.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_STRATEGIES = frozenset(["ddp", "fsdp", "deepspeed_zero2", "deepspeed_zero3"])
VALID_SHARDING_STRATEGIES = frozenset(["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD"])
VALID_WRAP_POLICIES = frozenset(["transformer_block", "size_based"])
VALID_MIXED_PRECISION = frozenset(["bf16", "fp16", "none"])
VALID_ACTIVATION_CHECKPOINTING = frozenset(["off", "transformer_block"])
VALID_STATE_DICT_TYPES = frozenset(["full", "sharded"])
VALID_OFFLOAD_DEVICES = frozenset(["none", "cpu", "nvme"])
VALID_ZERO_STAGES = frozenset([2, 3])
VALID_BACKENDS = frozenset(["nccl", "gloo", "mpi"])


# ---------------------------------------------------------------------------
# DistributedConfig
# ---------------------------------------------------------------------------


@dataclass
class DistributedConfig:
    """Top-level distributed training configuration.

    Attributes
    ----------
    strategy:
        One of 'ddp', 'fsdp', 'deepspeed_zero2', 'deepspeed_zero3'.
    world_size:
        Total number of processes. -1 means auto-detect from launcher
        environment variables (WORLD_SIZE).
    backend:
        Process group backend. 'nccl' for GPU training, 'gloo' for CPU.
    grad_accum:
        Gradient accumulation steps. All strategies support this.
    """

    strategy: str = "ddp"
    world_size: int = -1
    backend: str = "nccl"
    grad_accum: int = 1

    def validate(self) -> None:
        """Raise ValueError for any invalid field combination."""
        if self.strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy '{self.strategy}'. "
                f"Must be one of: {sorted(VALID_STRATEGIES)}"
            )
        if self.strategy is None:
            raise TypeError("strategy must be a string, got None")
        if self.world_size != -1 and self.world_size < 1:
            raise ValueError(
                f"world_size must be -1 (auto) or >= 1, got {self.world_size}"
            )
        if self.backend not in VALID_BACKENDS:
            raise ValueError(
                f"Invalid backend '{self.backend}'. "
                f"Must be one of: {sorted(VALID_BACKENDS)}"
            )
        if self.grad_accum < 1:
            raise ValueError(f"grad_accum must be >= 1, got {self.grad_accum}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DistributedConfig":
        cfg = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        cfg.validate()
        return cfg


# ---------------------------------------------------------------------------
# FSDPConfig
# ---------------------------------------------------------------------------


@dataclass
class FSDPConfig:
    """FSDP-specific configuration.

    Attributes
    ----------
    sharding_strategy:
        One of FULL_SHARD, SHARD_GRAD_OP, NO_SHARD, HYBRID_SHARD.
    wrap_policy:
        'transformer_block' for ModuleWrapPolicy, 'size_based' for size-based.
    wrap_module_classes:
        List of fully-qualified or simple class names to wrap when using
        'transformer_block' policy. Required if wrap_policy='transformer_block'.
    mixed_precision:
        'bf16', 'fp16', or 'none'.
    activation_checkpointing:
        'off' or 'transformer_block'. Applied after FSDP wrapping.
    state_dict_type:
        'full' for FULL_STATE_DICT, 'sharded' for DCP sharded checkpoint.
    sync_module_states:
        If True, FSDP broadcasts rank-0 parameters to all ranks at wrap time.
        Required when using the rank-0-only load pattern.
    cpu_offload:
        If True, enable CPUOffload(offload_params=True). Use as last resort.
    min_num_params:
        Minimum parameter count for size_based policy. Ignored for
        transformer_block policy.
    """

    sharding_strategy: str = "FULL_SHARD"
    wrap_policy: str = "transformer_block"
    wrap_module_classes: List[str] = field(default_factory=list)
    mixed_precision: str = "bf16"
    activation_checkpointing: str = "off"
    state_dict_type: str = "full"
    sync_module_states: bool = True
    cpu_offload: bool = False
    min_num_params: int = 100_000_000

    def validate(self) -> None:
        if self.sharding_strategy not in VALID_SHARDING_STRATEGIES:
            raise ValueError(
                f"Invalid sharding_strategy '{self.sharding_strategy}'. "
                f"Must be one of: {sorted(VALID_SHARDING_STRATEGIES)}"
            )
        if self.wrap_policy not in VALID_WRAP_POLICIES:
            raise ValueError(
                f"Invalid wrap_policy '{self.wrap_policy}'. "
                f"Must be one of: {sorted(VALID_WRAP_POLICIES)}"
            )
        if self.wrap_policy == "transformer_block" and not self.wrap_module_classes:
            raise ValueError(
                "wrap_module_classes must be non-empty when wrap_policy='transformer_block'. "
                "Specify class names like ['TransformerBlock']."
            )
        if self.mixed_precision not in VALID_MIXED_PRECISION:
            raise ValueError(
                f"Invalid mixed_precision '{self.mixed_precision}'. "
                f"Must be one of: {sorted(VALID_MIXED_PRECISION)}"
            )
        if self.activation_checkpointing not in VALID_ACTIVATION_CHECKPOINTING:
            raise ValueError(
                f"Invalid activation_checkpointing '{self.activation_checkpointing}'. "
                f"Must be one of: {sorted(VALID_ACTIVATION_CHECKPOINTING)}"
            )
        if self.state_dict_type not in VALID_STATE_DICT_TYPES:
            raise ValueError(
                f"Invalid state_dict_type '{self.state_dict_type}'. "
                f"Must be one of: {sorted(VALID_STATE_DICT_TYPES)}"
            )
        if self.min_num_params < 1:
            raise ValueError(
                f"min_num_params must be >= 1, got {self.min_num_params}"
            )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FSDPConfig":
        cfg = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        cfg.validate()
        return cfg


# ---------------------------------------------------------------------------
# DeepSpeedConfig
# ---------------------------------------------------------------------------


@dataclass
class DeepSpeedConfig:
    """DeepSpeed ZeRO configuration.

    Attributes
    ----------
    zero_stage:
        ZeRO stage: 2 or 3. Stage 2 shards gradients and optimizer states.
        Stage 3 additionally shards parameters.
    offload_optimizer:
        Where to offload optimizer states: 'none', 'cpu', or 'nvme'.
        Valid for stage 2 and 3.
    offload_param:
        Where to offload parameters: 'none', 'cpu', or 'nvme'.
        Valid ONLY for stage 3. Raises ValueError for stage 2.
    reduce_bucket_size:
        Bytes per gradient reduction bucket.
    allgather_bucket_size:
        Bytes per parameter all-gather bucket.
    stage3_prefetch_bucket_size:
        Bytes to prefetch for the next parameter partition (stage 3 only).
    stage3_param_persistence_threshold:
        Parameters smaller than this (bytes) stay resident on all ranks.
    stage3_max_live_parameters:
        Maximum bytes of simultaneously-gathered parameters.
    stage3_max_reuse_distance:
        If a param will be reused within this many bytes, keep it gathered.
    overlap_comm:
        Overlap gradient reduction with backward computation.
    contiguous_gradients:
        Copy gradients into a contiguous buffer before reduction.
    nvme_path:
        File system path for NVMe offload. Required when offload device is nvme.
    fp16:
        Enable FP16 mixed precision in DeepSpeed config.
    bf16:
        Enable BF16 mixed precision in DeepSpeed config.
    gradient_clipping:
        Global gradient norm clipping threshold.
    """

    zero_stage: int = 3
    offload_optimizer: str = "none"
    offload_param: str = "none"
    reduce_bucket_size: int = 500_000_000
    allgather_bucket_size: int = 500_000_000
    stage3_prefetch_bucket_size: int = 50_000_000
    stage3_param_persistence_threshold: int = 100_000
    stage3_max_live_parameters: int = 1_000_000_000
    stage3_max_reuse_distance: int = 1_000_000_000
    overlap_comm: bool = True
    contiguous_gradients: bool = True
    nvme_path: Optional[str] = None
    fp16: bool = False
    bf16: bool = False
    gradient_clipping: float = 1.0

    def validate(self) -> None:
        if self.zero_stage not in VALID_ZERO_STAGES:
            raise ValueError(
                f"Invalid zero_stage {self.zero_stage}. Must be 2 or 3."
            )
        if self.offload_optimizer not in VALID_OFFLOAD_DEVICES:
            raise ValueError(
                f"Invalid offload_optimizer '{self.offload_optimizer}'. "
                f"Must be one of: {sorted(VALID_OFFLOAD_DEVICES)}"
            )
        if self.offload_param not in VALID_OFFLOAD_DEVICES:
            raise ValueError(
                f"Invalid offload_param '{self.offload_param}'. "
                f"Must be one of: {sorted(VALID_OFFLOAD_DEVICES)}"
            )
        if self.offload_param != "none" and self.zero_stage != 3:
            raise ValueError(
                f"offload_param='{self.offload_param}' requires zero_stage=3, "
                f"got zero_stage={self.zero_stage}."
            )
        if self.offload_param == "nvme" and not self.nvme_path:
            raise ValueError(
                "nvme_path must be set when offload_param='nvme'."
            )
        if self.offload_optimizer == "nvme" and not self.nvme_path:
            raise ValueError(
                "nvme_path must be set when offload_optimizer='nvme'."
            )
        if self.reduce_bucket_size < 1:
            raise ValueError(
                f"reduce_bucket_size must be >= 1, got {self.reduce_bucket_size}"
            )
        if self.allgather_bucket_size < 1:
            raise ValueError(
                f"allgather_bucket_size must be >= 1, got {self.allgather_bucket_size}"
            )
        if self.fp16 and self.bf16:
            raise ValueError("Cannot enable both fp16 and bf16 simultaneously.")
        if self.gradient_clipping <= 0:
            raise ValueError(
                f"gradient_clipping must be > 0, got {self.gradient_clipping}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DeepSpeedConfig":
        cfg = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        cfg.validate()
        return cfg


# ---------------------------------------------------------------------------
# OptimizerConfig
# ---------------------------------------------------------------------------


@dataclass
class OptimizerConfig:
    """Optimizer hyperparameters.

    Attributes
    ----------
    optimizer_type:
        Optimizer class name: 'AdamW', 'Adam', 'SGD'.
    lr:
        Base learning rate.
    weight_decay:
        L2 regularization coefficient.
    betas:
        Adam beta1 and beta2 coefficients.
    eps:
        Adam epsilon for numerical stability.
    momentum:
        SGD momentum coefficient. Ignored for Adam variants.
    grad_clip:
        Global gradient norm clipping. 0.0 disables clipping.
    """

    optimizer_type: str = "AdamW"
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    momentum: float = 0.9
    grad_clip: float = 1.0

    def validate(self) -> None:
        valid_optimizers = frozenset(["AdamW", "Adam", "SGD"])
        if self.optimizer_type not in valid_optimizers:
            raise ValueError(
                f"Invalid optimizer_type '{self.optimizer_type}'. "
                f"Must be one of: {sorted(valid_optimizers)}"
            )
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if not (0.0 < self.betas[0] < 1.0):
            raise ValueError(f"betas[0] must be in (0, 1), got {self.betas[0]}")
        if not (0.0 < self.betas[1] < 1.0):
            raise ValueError(f"betas[1] must be in (0, 1), got {self.betas[1]}")
        if self.eps <= 0:
            raise ValueError(f"eps must be > 0, got {self.eps}")
        if self.grad_clip < 0:
            raise ValueError(f"grad_clip must be >= 0, got {self.grad_clip}")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["betas"] = list(d["betas"])
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OptimizerConfig":
        d = dict(d)
        if "betas" in d:
            d["betas"] = tuple(d["betas"])
        cfg = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        cfg.validate()
        return cfg


# ---------------------------------------------------------------------------
# StrategyContext
# ---------------------------------------------------------------------------


@dataclass
class StrategyContext:
    """Runtime context established after distributed setup.

    Populated by StrategyRouter.setup_distributed() and updated by
    StrategyRouter.wrap_model().

    Attributes
    ----------
    strategy:
        Active strategy string.
    rank:
        Global rank of this process.
    world_size:
        Total number of processes.
    local_rank:
        Rank within the current node.
    device:
        torch.device for this process.
    is_distributed:
        True if world_size > 1.
    """

    strategy: str = "ddp"
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    device: str = "cuda:0"
    is_distributed: bool = False

    def __post_init__(self) -> None:
        self.is_distributed = self.world_size > 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StrategyContext":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# BenchResult
# ---------------------------------------------------------------------------


@dataclass
class BenchResult:
    """Raw timing data from a single benchmark run.

    Attributes
    ----------
    strategy:
        Strategy used for this run.
    world_size:
        Number of GPUs/processes.
    throughput_p50:
        Median tokens per second over measured steps.
    throughput_p90:
        10th-percentile throughput (conservative bound).
    step_time_p50_ms:
        Median step time in milliseconds.
    step_time_p90_ms:
        90th-percentile step time in milliseconds.
    memory_peak_gb:
        Peak GPU memory usage in GB.
    measured_steps:
        Number of steps included in statistics.
    warmup_steps:
        Number of steps excluded from statistics.
    """

    strategy: str = "ddp"
    world_size: int = 1
    throughput_p50: float = 0.0
    throughput_p90: float = 0.0
    step_time_p50_ms: float = 0.0
    step_time_p90_ms: float = 0.0
    memory_peak_gb: float = 0.0
    measured_steps: int = 0
    warmup_steps: int = 0

    def validate(self) -> None:
        if self.throughput_p50 < 0:
            raise ValueError(f"throughput_p50 must be >= 0, got {self.throughput_p50}")
        if self.step_time_p50_ms < 0:
            raise ValueError(
                f"step_time_p50_ms must be >= 0, got {self.step_time_p50_ms}"
            )
        if self.memory_peak_gb < 0:
            raise ValueError(
                f"memory_peak_gb must be >= 0, got {self.memory_peak_gb}"
            )
        if self.measured_steps < 0:
            raise ValueError(
                f"measured_steps must be >= 0, got {self.measured_steps}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BenchResult":
        result = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        result.validate()
        return result


# ---------------------------------------------------------------------------
# ScalingReport
# ---------------------------------------------------------------------------


@dataclass
class ScalingReport:
    """Result of a scaling efficiency computation.

    Attributes
    ----------
    strategy:
        Strategy used for this measurement.
    world_size:
        N in the scaling_efficiency formula.
    throughput_1:
        Tokens per second at world_size=1 (baseline).
    throughput_n:
        Tokens per second at world_size=N.
    scaling_efficiency:
        throughput_n / (N * throughput_1). 1.0 is perfect linear scaling.
    memory_peak_gb:
        Peak GPU memory in GB (max across ranks).
    step_time_p50_ms:
        Median step time in milliseconds.
    step_time_p90_ms:
        90th-percentile step time in milliseconds.
    model_params_b:
        Model size in billions of parameters (optional metadata).
    per_gpu_batch_size:
        Per-GPU batch size during benchmark.
    seq_len:
        Sequence length in tokens.
    mixed_precision:
        Mixed precision setting used.
    activation_checkpointing:
        Whether activation checkpointing was enabled.
    """

    strategy: str = "ddp"
    world_size: int = 1
    throughput_1: float = 0.0
    throughput_n: float = 0.0
    scaling_efficiency: float = 0.0
    memory_peak_gb: float = 0.0
    step_time_p50_ms: float = 0.0
    step_time_p90_ms: float = 0.0
    model_params_b: float = 0.0
    per_gpu_batch_size: int = 1
    seq_len: int = 512
    mixed_precision: str = "bf16"
    activation_checkpointing: bool = False

    def validate(self) -> None:
        if self.world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {self.world_size}")
        if self.throughput_1 < 0:
            raise ValueError(f"throughput_1 must be >= 0, got {self.throughput_1}")
        if self.throughput_n < 0:
            raise ValueError(f"throughput_n must be >= 0, got {self.throughput_n}")
        if self.scaling_efficiency < 0:
            raise ValueError(
                f"scaling_efficiency must be >= 0, got {self.scaling_efficiency}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ScalingReport":
        report = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        report.validate()
        return report


# ---------------------------------------------------------------------------
# validate_config: cross-field validation of the full config bundle
# ---------------------------------------------------------------------------


def validate_config(
    dist_cfg: DistributedConfig,
    fsdp_cfg: Optional[FSDPConfig] = None,
    ds_cfg: Optional[DeepSpeedConfig] = None,
    opt_cfg: Optional[OptimizerConfig] = None,
) -> None:
    """Run all individual validations and cross-field checks.

    Parameters
    ----------
    dist_cfg:
        Required. The top-level distributed config.
    fsdp_cfg:
        Required if strategy is 'fsdp'.
    ds_cfg:
        Required if strategy is 'deepspeed_zero2' or 'deepspeed_zero3'.
    opt_cfg:
        Optional optimizer config. Validated independently if provided.

    Raises
    ------
    ValueError
        For any invalid field or cross-field constraint violation.
    """
    dist_cfg.validate()

    if dist_cfg.strategy == "fsdp":
        if fsdp_cfg is None:
            raise ValueError("FSDPConfig is required when strategy='fsdp'.")
        fsdp_cfg.validate()

    if dist_cfg.strategy in ("deepspeed_zero2", "deepspeed_zero3"):
        if ds_cfg is None:
            raise ValueError(
                f"DeepSpeedConfig is required when strategy='{dist_cfg.strategy}'."
            )
        # Enforce that the zero_stage in DeepSpeedConfig matches the strategy
        expected_stage = 2 if dist_cfg.strategy == "deepspeed_zero2" else 3
        if ds_cfg.zero_stage != expected_stage:
            raise ValueError(
                f"strategy='{dist_cfg.strategy}' requires zero_stage={expected_stage}, "
                f"but DeepSpeedConfig.zero_stage={ds_cfg.zero_stage}."
            )
        ds_cfg.validate()

    if opt_cfg is not None:
        opt_cfg.validate()


# ---------------------------------------------------------------------------
# Self-Tests
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    failures: List[str] = []

    def check(name: str, condition: bool) -> None:
        if condition:
            print(f"  PASS  {name}")
        else:
            print(f"  FAIL  {name}")
            failures.append(name)

    def expect_raises(name: str, exc_type: type, fn) -> None:
        try:
            fn()
            print(f"  FAIL  {name} (no exception raised)")
            failures.append(name)
        except exc_type:
            print(f"  PASS  {name}")
        except Exception as e:
            print(f"  FAIL  {name} (wrong exception: {type(e).__name__}: {e})")
            failures.append(name)

    print("=== DistributedConfig tests ===")
    # Valid strategies
    for s in ["ddp", "fsdp", "deepspeed_zero2", "deepspeed_zero3"]:
        cfg = DistributedConfig(strategy=s)
        cfg.validate()
        check(f"valid_strategy_{s}", True)

    # Invalid strategy
    expect_raises(
        "invalid_strategy_unknown",
        ValueError,
        lambda: DistributedConfig(strategy="unknown").validate(),
    )
    expect_raises(
        "invalid_strategy_empty",
        ValueError,
        lambda: DistributedConfig(strategy="").validate(),
    )

    # Serialization round-trip
    orig = DistributedConfig(strategy="fsdp", world_size=8, grad_accum=4)
    reconstructed = DistributedConfig.from_dict(orig.to_dict())
    check("dist_config_roundtrip", reconstructed == orig)

    print("\n=== FSDPConfig tests ===")
    # Valid sharding strategies
    for ss in ["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD"]:
        cfg = FSDPConfig(sharding_strategy=ss, wrap_module_classes=["TransformerBlock"])
        cfg.validate()
        check(f"valid_sharding_{ss}", True)

    # transformer_block without classes raises
    expect_raises(
        "transformer_block_no_classes",
        ValueError,
        lambda: FSDPConfig(wrap_policy="transformer_block", wrap_module_classes=[]).validate(),
    )

    # size_based without classes is fine
    cfg = FSDPConfig(wrap_policy="size_based", wrap_module_classes=[])
    cfg.validate()
    check("size_based_no_classes_ok", True)

    # Invalid mixed precision
    expect_raises(
        "invalid_mixed_precision",
        ValueError,
        lambda: FSDPConfig(
            mixed_precision="fp8", wrap_module_classes=["B"]
        ).validate(),
    )

    # Round-trip
    orig_fsdp = FSDPConfig(
        sharding_strategy="HYBRID_SHARD",
        wrap_module_classes=["TransformerBlock"],
        mixed_precision="fp16",
        activation_checkpointing="transformer_block",
    )
    reconstructed_fsdp = FSDPConfig.from_dict(orig_fsdp.to_dict())
    check("fsdp_config_roundtrip", reconstructed_fsdp == orig_fsdp)

    print("\n=== DeepSpeedConfig tests ===")
    # Valid stages
    for stage in [2, 3]:
        cfg = DeepSpeedConfig(zero_stage=stage)
        cfg.validate()
        check(f"valid_zero_stage_{stage}", True)

    # Invalid stage
    expect_raises(
        "invalid_zero_stage_1",
        ValueError,
        lambda: DeepSpeedConfig(zero_stage=1).validate(),
    )
    expect_raises(
        "invalid_zero_stage_4",
        ValueError,
        lambda: DeepSpeedConfig(zero_stage=4).validate(),
    )

    # offload_param stage 3 valid
    cfg = DeepSpeedConfig(zero_stage=3, offload_param="cpu")
    cfg.validate()
    check("offload_param_stage3_ok", True)

    # offload_param stage 2 raises
    expect_raises(
        "offload_param_stage2_raises",
        ValueError,
        lambda: DeepSpeedConfig(zero_stage=2, offload_param="cpu").validate(),
    )

    # fp16 + bf16 raises
    expect_raises(
        "fp16_bf16_mutual_exclusion",
        ValueError,
        lambda: DeepSpeedConfig(fp16=True, bf16=True).validate(),
    )

    # nvme without path raises
    expect_raises(
        "nvme_no_path_raises",
        ValueError,
        lambda: DeepSpeedConfig(
            zero_stage=3, offload_param="nvme", nvme_path=None
        ).validate(),
    )

    # Round-trip stage 3
    orig_ds = DeepSpeedConfig(
        zero_stage=3,
        offload_optimizer="cpu",
        reduce_bucket_size=200_000_000,
        bf16=True,
    )
    reconstructed_ds = DeepSpeedConfig.from_dict(orig_ds.to_dict())
    check("ds_config_roundtrip", reconstructed_ds == orig_ds)

    print("\n=== OptimizerConfig tests ===")
    cfg_opt = OptimizerConfig(lr=1e-3, weight_decay=0.05)
    cfg_opt.validate()
    check("valid_optimizer_config", True)

    expect_raises(
        "invalid_optimizer_type",
        ValueError,
        lambda: OptimizerConfig(optimizer_type="RMSprop").validate(),
    )
    expect_raises(
        "negative_lr",
        ValueError,
        lambda: OptimizerConfig(lr=-1e-4).validate(),
    )

    # Round-trip (betas are tuple -> list -> tuple)
    orig_opt = OptimizerConfig(lr=2e-5, betas=(0.9, 0.98))
    reconstructed_opt = OptimizerConfig.from_dict(orig_opt.to_dict())
    check("optimizer_config_roundtrip", reconstructed_opt == orig_opt)

    print("\n=== validate_config cross-field tests ===")
    dist = DistributedConfig(strategy="fsdp")
    fsdp = FSDPConfig(wrap_module_classes=["TransformerBlock"])
    validate_config(dist, fsdp_cfg=fsdp)
    check("validate_config_fsdp_ok", True)

    # fsdp strategy without FSDPConfig raises
    expect_raises(
        "validate_config_fsdp_missing",
        ValueError,
        lambda: validate_config(DistributedConfig(strategy="fsdp")),
    )

    # deepspeed_zero3 with wrong stage raises
    expect_raises(
        "validate_config_stage_mismatch",
        ValueError,
        lambda: validate_config(
            DistributedConfig(strategy="deepspeed_zero3"),
            ds_cfg=DeepSpeedConfig(zero_stage=2),
        ),
    )

    print("\n=== ScalingReport tests ===")
    report = ScalingReport(
        strategy="fsdp",
        world_size=4,
        throughput_1=10000.0,
        throughput_n=38000.0,
        scaling_efficiency=0.95,
    )
    report.validate()
    reconstructed_report = ScalingReport.from_dict(report.to_dict())
    check("scaling_report_roundtrip", reconstructed_report == report)

    print("\n=== BenchResult tests ===")
    bench = BenchResult(
        strategy="fsdp",
        world_size=8,
        throughput_p50=50000.0,
        step_time_p50_ms=120.0,
        memory_peak_gb=35.5,
    )
    bench.validate()
    reconstructed_bench = BenchResult.from_dict(bench.to_dict())
    check("bench_result_roundtrip", reconstructed_bench == bench)

    # Summary
    print(f"\n{'='*50}")
    total = len(failures) + (
        # count checks above — approximate via absence of failures
        0
    )
    if failures:
        print(f"FAIL: {len(failures)} test(s) failed: {failures}")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")
        sys.exit(0)
