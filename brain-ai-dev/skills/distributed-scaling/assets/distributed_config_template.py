"""
DistributedConfig — Configuration dataclass with presets and validation
for distributed training of the brain_ai system.

Aggregates all distributed-related configuration: strategy, sharding,
gradient accumulation, mixed precision, multi-node, memory optimization,
and scale-specific presets (1B/3B/7B).

Key classes:
    DistributedConfig     — Main configuration dataclass
    MemoryConfig          — GPU memory optimization settings
    CommunicationConfig   — Backend and timeout settings
    CheckpointConfig      — Distributed checkpoint settings

Self-tests in __main__ validate all configuration, presets, and validation
rules without multi-GPU hardware.
"""

from __future__ import annotations

import copy
import logging
import math
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ===========================================================================
# SECTION 1: Enums
# ===========================================================================

class DistStrategy(str, Enum):
    """Distributed training strategies."""
    SINGLE = "single"
    DDP = "ddp"
    FSDP = "fsdp"


class FSDPSharding(str, Enum):
    """FSDP sharding strategies."""
    FULL_SHARD = "full_shard"
    SHARD_GRAD_OP = "shard_grad_op"
    NO_SHARD = "no_shard"


class MixedPrecisionMode(str, Enum):
    """Mixed precision modes."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


# ===========================================================================
# SECTION 2: Sub-configs
# ===========================================================================

@dataclass
class MemoryConfig:
    """GPU memory optimization settings.

    Attributes:
        activation_checkpointing: Recompute activations to save memory.
        cpu_offload: Offload FSDP shards to CPU.
        limit_all_gathers: Prevent FSDP prefetching (reduces peak memory).
        pin_memory: Pin DataLoader memory for faster GPU transfer.
        gradient_as_bucket_view: Alias gradients into DDP comm buffers.
        empty_cache_frequency: How often (in steps) to call cuda.empty_cache().
    """
    activation_checkpointing: bool = False
    cpu_offload: bool = False
    limit_all_gathers: bool = False
    pin_memory: bool = True
    gradient_as_bucket_view: bool = True
    empty_cache_frequency: int = 0  # 0 = never

    def validate(self) -> List[str]:
        errors = []
        if self.empty_cache_frequency < 0:
            errors.append(f"empty_cache_frequency must be >= 0, got {self.empty_cache_frequency}")
        return errors


@dataclass
class CommunicationConfig:
    """Communication backend and timeout settings.

    Attributes:
        backend: "nccl" for GPU, "gloo" for CPU.
        timeout_seconds: Timeout for collective operations.
        bucket_cap_mb: DDP bucket size in MB.
        nccl_socket_ifname: Network interface for NCCL.
        nccl_debug: NCCL debug level ("OFF", "INFO", "WARN").
    """
    backend: str = "nccl"
    timeout_seconds: int = 1800
    bucket_cap_mb: int = 25
    nccl_socket_ifname: str = ""
    nccl_debug: str = "OFF"

    def validate(self) -> List[str]:
        errors = []
        if self.backend not in ("nccl", "gloo"):
            errors.append(f"Invalid backend: {self.backend}")
        if self.timeout_seconds < 1:
            errors.append(f"timeout_seconds must be >= 1, got {self.timeout_seconds}")
        if self.bucket_cap_mb < 1:
            errors.append(f"bucket_cap_mb must be >= 1, got {self.bucket_cap_mb}")
        if self.nccl_debug not in ("OFF", "INFO", "WARN", "TRACE"):
            errors.append(f"Invalid nccl_debug: {self.nccl_debug}")
        return errors


@dataclass
class CheckpointConfig:
    """Distributed checkpointing settings.

    Attributes:
        save_sharded: Save sharded checkpoints (fast, same config only).
        save_full: Save full state dict (portable across configs).
        save_optimizer: Include optimizer state in checkpoint.
        save_on_rank0_only: Only rank 0 saves (DDP mode).
        checkpoint_dir: Base directory for checkpoints.
        save_interval_steps: Save every N optimizer steps (0 = only at phase end).
    """
    save_sharded: bool = True
    save_full: bool = True
    save_optimizer: bool = True
    save_on_rank0_only: bool = True
    checkpoint_dir: str = "checkpoints"
    save_interval_steps: int = 0

    def validate(self) -> List[str]:
        errors = []
        if self.save_interval_steps < 0:
            errors.append(f"save_interval_steps must be >= 0, got {self.save_interval_steps}")
        if not self.save_sharded and not self.save_full:
            errors.append("At least one of save_sharded or save_full must be True")
        return errors


# ===========================================================================
# SECTION 3: DistributedConfig
# ===========================================================================

@dataclass
class DistributedConfig:
    """Main configuration for distributed training.

    Aggregates strategy, sharding, accumulation, precision, multi-node,
    and memory settings into one config object.

    Attributes:
        strategy: Distribution strategy ("single", "ddp", "fsdp").
        fsdp_sharding: FSDP sharding mode ("full_shard", "shard_grad_op", "no_shard").
        gradient_accumulation_steps: Micro-batches per optimizer step.
        find_unused_parameters: DDP unused parameter detection.
        sync_batchnorm: Convert BatchNorm to SyncBatchNorm.
        mixed_precision_policy: Precision mode ("fp32", "fp16", "bf16").
        micro_batch_size: Per-GPU, per-step batch size.
        num_workers: DataLoader worker count per rank.
        # Multi-node
        num_nodes: Number of nodes in the cluster.
        node_rank: This node's rank.
        nproc_per_node: Processes per node.
        master_addr: Master node address.
        master_port: Master node port.
        # Learning rate
        learning_rate: Base learning rate.
        reference_batch_size: Batch size the base LR was tuned for.
        lr_scaling_mode: "linear" or "sqrt".
        warmup_steps: LR warmup steps (before scaling).
        max_grad_norm: Maximum gradient norm for clipping.
        # Sub-configs
        memory: MemoryConfig
        communication: CommunicationConfig
        checkpoint: CheckpointConfig
    """
    # Strategy
    strategy: str = "ddp"
    fsdp_sharding: str = "full_shard"
    gradient_accumulation_steps: int = 1
    find_unused_parameters: bool = False
    sync_batchnorm: bool = True
    mixed_precision_policy: str = "bf16"

    # Batch and data loading
    micro_batch_size: int = 32
    num_workers: int = 8

    # Multi-node
    num_nodes: int = 1
    node_rank: int = 0
    nproc_per_node: int = 1
    master_addr: str = "localhost"
    master_port: str = "29500"

    # Learning rate
    learning_rate: float = 3e-4
    reference_batch_size: int = 256
    lr_scaling_mode: str = "linear"
    warmup_steps: int = 2000
    max_grad_norm: float = 1.0

    # Sub-configs
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    # -------------------------------------------------------------------
    # Computed properties
    # -------------------------------------------------------------------

    @property
    def world_size(self) -> int:
        """Total number of processes."""
        return self.num_nodes * self.nproc_per_node

    @property
    def effective_batch_size(self) -> int:
        """Effective batch size = micro_batch * accum_steps * world_size."""
        return self.micro_batch_size * self.gradient_accumulation_steps * self.world_size

    @property
    def scaled_learning_rate(self) -> float:
        """Learning rate scaled by effective batch size."""
        ratio = self.effective_batch_size / self.reference_batch_size
        if self.lr_scaling_mode == "linear":
            return self.learning_rate * ratio
        elif self.lr_scaling_mode == "sqrt":
            return self.learning_rate * math.sqrt(ratio)
        return self.learning_rate

    @property
    def scaled_warmup_steps(self) -> int:
        """Warmup steps scaled by effective batch ratio."""
        ratio = self.effective_batch_size / self.reference_batch_size
        return int(self.warmup_steps * ratio)

    @property
    def is_distributed(self) -> bool:
        """Whether this config uses distributed training."""
        return self.strategy != "single" and self.world_size > 1

    @property
    def needs_fsdp(self) -> bool:
        """Whether FSDP is required."""
        return self.strategy == "fsdp"

    @property
    def backend(self) -> str:
        """Communication backend (shorthand)."""
        return self.communication.backend

    # -------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------

    def validate(self) -> List[str]:
        """Validate the entire configuration.

        Returns:
            List of error messages (empty if valid).
        """
        errors = []

        # Strategy
        if self.strategy not in ("single", "ddp", "fsdp"):
            errors.append(f"Invalid strategy: {self.strategy}")

        if self.fsdp_sharding not in ("full_shard", "shard_grad_op", "no_shard"):
            errors.append(f"Invalid fsdp_sharding: {self.fsdp_sharding}")

        if self.mixed_precision_policy not in ("fp32", "fp16", "bf16"):
            errors.append(f"Invalid mixed_precision_policy: {self.mixed_precision_policy}")

        if self.lr_scaling_mode not in ("linear", "sqrt"):
            errors.append(f"Invalid lr_scaling_mode: {self.lr_scaling_mode}")

        # Numeric ranges
        if self.gradient_accumulation_steps < 1:
            errors.append(
                f"gradient_accumulation_steps must be >= 1, "
                f"got {self.gradient_accumulation_steps}"
            )

        if self.micro_batch_size < 1:
            errors.append(f"micro_batch_size must be >= 1, got {self.micro_batch_size}")

        if self.num_workers < 0:
            errors.append(f"num_workers must be >= 0, got {self.num_workers}")

        if self.num_nodes < 1:
            errors.append(f"num_nodes must be >= 1, got {self.num_nodes}")

        if self.nproc_per_node < 1:
            errors.append(f"nproc_per_node must be >= 1, got {self.nproc_per_node}")

        if self.node_rank < 0 or self.node_rank >= max(self.num_nodes, 1):
            errors.append(
                f"node_rank ({self.node_rank}) must be in [0, {self.num_nodes})"
            )

        if self.learning_rate <= 0:
            errors.append(f"learning_rate must be > 0, got {self.learning_rate}")

        if self.reference_batch_size < 1:
            errors.append(f"reference_batch_size must be >= 1, got {self.reference_batch_size}")

        if self.warmup_steps < 0:
            errors.append(f"warmup_steps must be >= 0, got {self.warmup_steps}")

        if self.max_grad_norm < 0:
            errors.append(f"max_grad_norm must be >= 0, got {self.max_grad_norm}")

        # Port validation
        try:
            port = int(self.master_port)
            if port < 1 or port > 65535:
                errors.append(f"master_port must be 1-65535, got {port}")
        except ValueError:
            errors.append(f"master_port must be numeric, got {self.master_port}")

        # Strategy-specific checks
        if self.strategy == "fsdp" and self.world_size < 2:
            errors.append(
                "FSDP requires world_size >= 2; consider strategy='ddp' or 'single'"
            )

        if self.strategy == "ddp" and self.memory.cpu_offload:
            errors.append("cpu_offload is only supported with FSDP, not DDP")

        if self.strategy == "single" and self.sync_batchnorm:
            # Not an error, but a warning-level note
            pass

        # Sub-config validation
        errors.extend(self.memory.validate())
        errors.extend(self.communication.validate())
        errors.extend(self.checkpoint.validate())

        return errors

    def is_valid(self) -> bool:
        """Return True if configuration is valid."""
        return len(self.validate()) == 0

    # -------------------------------------------------------------------
    # Presets
    # -------------------------------------------------------------------

    @classmethod
    def for_dev(cls) -> "DistributedConfig":
        """Dev preset: single GPU, MNIST, fast iteration.

        micro_batch=32, no accumulation, fp32.
        """
        return cls(
            strategy="single",
            fsdp_sharding="no_shard",
            gradient_accumulation_steps=1,
            micro_batch_size=32,
            mixed_precision_policy="fp32",
            num_workers=0,
            nproc_per_node=1,
            num_nodes=1,
            learning_rate=3e-4,
            warmup_steps=100,
            memory=MemoryConfig(
                activation_checkpointing=False,
                cpu_offload=False,
                pin_memory=False,
            ),
            communication=CommunicationConfig(backend="gloo"),
            checkpoint=CheckpointConfig(
                save_sharded=False,
                save_full=True,
            ),
        )

    @classmethod
    def for_1b(cls, num_gpus: int = 4) -> "DistributedConfig":
        """1B preset: DDP across 2-4 GPUs, 24GB each.

        micro_batch=16, accum=4, effective=256 on 4 GPUs.
        """
        return cls(
            strategy="ddp",
            fsdp_sharding="no_shard",
            gradient_accumulation_steps=4,
            micro_batch_size=16,
            mixed_precision_policy="bf16",
            num_workers=8,
            nproc_per_node=num_gpus,
            num_nodes=1,
            learning_rate=3e-4,
            reference_batch_size=256,
            warmup_steps=2000,
            sync_batchnorm=True,
            memory=MemoryConfig(
                activation_checkpointing=False,
                cpu_offload=False,
                pin_memory=True,
                gradient_as_bucket_view=True,
            ),
            communication=CommunicationConfig(
                backend="nccl",
                bucket_cap_mb=25,
            ),
            checkpoint=CheckpointConfig(
                save_sharded=False,
                save_full=True,
                save_optimizer=True,
            ),
        )

    @classmethod
    def for_3b(cls, num_gpus: int = 8) -> "DistributedConfig":
        """3B preset: FSDP SHARD_GRAD_OP across 4-8 GPUs, 40GB each.

        micro_batch=8, accum=8, effective=512 on 8 GPUs.
        """
        return cls(
            strategy="fsdp",
            fsdp_sharding="shard_grad_op",
            gradient_accumulation_steps=8,
            micro_batch_size=8,
            mixed_precision_policy="bf16",
            num_workers=8,
            nproc_per_node=num_gpus,
            num_nodes=1,
            learning_rate=3e-4,
            reference_batch_size=256,
            warmup_steps=2000,
            find_unused_parameters=False,
            sync_batchnorm=True,
            memory=MemoryConfig(
                activation_checkpointing=True,
                cpu_offload=False,
                pin_memory=True,
                limit_all_gathers=False,
            ),
            communication=CommunicationConfig(
                backend="nccl",
                bucket_cap_mb=50,
            ),
            checkpoint=CheckpointConfig(
                save_sharded=True,
                save_full=True,
                save_optimizer=True,
            ),
        )

    @classmethod
    def for_7b(cls, num_gpus: int = 8, num_nodes: int = 1) -> "DistributedConfig":
        """7B preset: FSDP FULL_SHARD across 8 GPUs, 80GB each.

        micro_batch=4, accum=16, effective=512 on 8 GPUs.
        """
        return cls(
            strategy="fsdp",
            fsdp_sharding="full_shard",
            gradient_accumulation_steps=16,
            micro_batch_size=4,
            mixed_precision_policy="bf16",
            num_workers=8,
            nproc_per_node=num_gpus,
            num_nodes=num_nodes,
            learning_rate=3e-4,
            reference_batch_size=256,
            warmup_steps=2000,
            find_unused_parameters=False,
            sync_batchnorm=True,
            memory=MemoryConfig(
                activation_checkpointing=True,
                cpu_offload=False,
                pin_memory=True,
                limit_all_gathers=True,
            ),
            communication=CommunicationConfig(
                backend="nccl",
                bucket_cap_mb=50,
                timeout_seconds=3600,
            ),
            checkpoint=CheckpointConfig(
                save_sharded=True,
                save_full=True,
                save_optimizer=True,
                save_interval_steps=1000,
            ),
        )

    @classmethod
    def for_7b_constrained(cls, num_gpus: int = 8) -> "DistributedConfig":
        """7B on constrained hardware (40GB GPUs) with CPU offload.

        micro_batch=2, accum=32, effective=512 on 8 GPUs.
        """
        return cls(
            strategy="fsdp",
            fsdp_sharding="full_shard",
            gradient_accumulation_steps=32,
            micro_batch_size=2,
            mixed_precision_policy="bf16",
            num_workers=4,
            nproc_per_node=num_gpus,
            num_nodes=1,
            learning_rate=3e-4,
            reference_batch_size=256,
            warmup_steps=2000,
            memory=MemoryConfig(
                activation_checkpointing=True,
                cpu_offload=True,
                pin_memory=True,
                limit_all_gathers=True,
            ),
            communication=CommunicationConfig(
                backend="nccl",
                bucket_cap_mb=25,
                timeout_seconds=3600,
            ),
            checkpoint=CheckpointConfig(
                save_sharded=True,
                save_full=True,
                save_optimizer=True,
                save_interval_steps=500,
            ),
        )

    @classmethod
    def for_multi_node_7b(
        cls, num_nodes: int = 2, num_gpus_per_node: int = 8
    ) -> "DistributedConfig":
        """7B multi-node preset: 2+ nodes, 8 GPUs each.

        micro_batch=4, accum=8, effective=512 on 16 GPUs (2 nodes x 8).
        """
        return cls(
            strategy="fsdp",
            fsdp_sharding="full_shard",
            gradient_accumulation_steps=8,
            micro_batch_size=4,
            mixed_precision_policy="bf16",
            num_workers=8,
            nproc_per_node=num_gpus_per_node,
            num_nodes=num_nodes,
            learning_rate=3e-4,
            reference_batch_size=256,
            warmup_steps=2000,
            memory=MemoryConfig(
                activation_checkpointing=True,
                cpu_offload=False,
                pin_memory=True,
                limit_all_gathers=True,
            ),
            communication=CommunicationConfig(
                backend="nccl",
                bucket_cap_mb=50,
                timeout_seconds=7200,
            ),
            checkpoint=CheckpointConfig(
                save_sharded=True,
                save_full=True,
                save_optimizer=True,
                save_interval_steps=500,
            ),
        )

    # -------------------------------------------------------------------
    # Utility methods
    # -------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a flat summary dict for logging."""
        return {
            "strategy": self.strategy,
            "fsdp_sharding": self.fsdp_sharding,
            "world_size": self.world_size,
            "micro_batch_size": self.micro_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_batch_size": self.effective_batch_size,
            "mixed_precision": self.mixed_precision_policy,
            "learning_rate": self.learning_rate,
            "scaled_lr": self.scaled_learning_rate,
            "warmup_steps": self.warmup_steps,
            "scaled_warmup_steps": self.scaled_warmup_steps,
            "activation_checkpointing": self.memory.activation_checkpointing,
            "cpu_offload": self.memory.cpu_offload,
            "backend": self.communication.backend,
            "num_nodes": self.num_nodes,
            "nproc_per_node": self.nproc_per_node,
        }

    def estimate_memory_gb(self, param_count: int) -> Dict[str, float]:
        """Estimate per-GPU memory usage in GB.

        Args:
            param_count: Total model parameter count.

        Returns:
            Dict with estimated memory components.
        """
        bytes_per_param = 4  # fp32
        if self.mixed_precision_policy in ("fp16", "bf16"):
            bytes_per_param = 2

        param_gb = param_count * bytes_per_param / 1e9
        grad_gb = param_gb
        # Adam optimizer: 2 states per parameter in fp32
        optimizer_gb = param_count * 4 * 2 / 1e9

        if self.strategy == "fsdp":
            ws = max(self.world_size, 1)
            if self.fsdp_sharding == "full_shard":
                param_gb_per_gpu = param_gb / ws
                grad_gb_per_gpu = grad_gb / ws
                opt_gb_per_gpu = optimizer_gb / ws
            elif self.fsdp_sharding == "shard_grad_op":
                param_gb_per_gpu = param_gb  # Full params on each GPU
                grad_gb_per_gpu = grad_gb / ws
                opt_gb_per_gpu = optimizer_gb / ws
            else:
                param_gb_per_gpu = param_gb
                grad_gb_per_gpu = grad_gb
                opt_gb_per_gpu = optimizer_gb
        else:
            param_gb_per_gpu = param_gb
            grad_gb_per_gpu = grad_gb
            opt_gb_per_gpu = optimizer_gb

        # Rough activation estimate (depends on batch size and model architecture)
        activation_gb = self.micro_batch_size * 0.5  # Rough heuristic
        if self.memory.activation_checkpointing:
            activation_gb *= 0.3

        total = param_gb_per_gpu + grad_gb_per_gpu + opt_gb_per_gpu + activation_gb

        return {
            "params_gb": round(param_gb_per_gpu, 2),
            "gradients_gb": round(grad_gb_per_gpu, 2),
            "optimizer_gb": round(opt_gb_per_gpu, 2),
            "activations_gb": round(activation_gb, 2),
            "total_gb": round(total, 2),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a flat dictionary."""
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, (MemoryConfig, CommunicationConfig, CheckpointConfig)):
                for sk, sv in v.__dict__.items():
                    d[f"{k}.{sk}"] = sv
            else:
                d[k] = v
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DistributedConfig":
        """Deserialize from a flat dictionary."""
        mem_kwargs = {}
        comm_kwargs = {}
        ckpt_kwargs = {}
        main_kwargs = {}

        for k, v in d.items():
            if k.startswith("memory."):
                mem_kwargs[k.split(".", 1)[1]] = v
            elif k.startswith("communication."):
                comm_kwargs[k.split(".", 1)[1]] = v
            elif k.startswith("checkpoint."):
                ckpt_kwargs[k.split(".", 1)[1]] = v
            else:
                main_kwargs[k] = v

        main_kwargs["memory"] = MemoryConfig(**mem_kwargs) if mem_kwargs else MemoryConfig()
        main_kwargs["communication"] = CommunicationConfig(**comm_kwargs) if comm_kwargs else CommunicationConfig()
        main_kwargs["checkpoint"] = CheckpointConfig(**ckpt_kwargs) if ckpt_kwargs else CheckpointConfig()

        return cls(**main_kwargs)


# ===========================================================================
# SECTION 4: Self-tests
# ===========================================================================

if __name__ == "__main__":
    import sys
    import traceback

    passed = 0
    failed = 0
    total = 0

    def run_test(name: str, fn: Callable):
        global passed, failed, total
        total += 1
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            traceback.print_exc()
            failed += 1

    print("=" * 70)
    print("DistributedConfig Self-Tests")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # DistStrategy / FSDPSharding / MixedPrecisionMode enum tests
    # -----------------------------------------------------------------------

    def test_dist_strategy_values():
        assert DistStrategy.SINGLE.value == "single"
        assert DistStrategy.DDP.value == "ddp"
        assert DistStrategy.FSDP.value == "fsdp"

    run_test("DistStrategy enum values", test_dist_strategy_values)

    def test_fsdp_sharding_values():
        assert FSDPSharding.FULL_SHARD.value == "full_shard"
        assert FSDPSharding.SHARD_GRAD_OP.value == "shard_grad_op"
        assert FSDPSharding.NO_SHARD.value == "no_shard"

    run_test("FSDPSharding enum values", test_fsdp_sharding_values)

    def test_mixed_precision_values():
        assert MixedPrecisionMode.FP32.value == "fp32"
        assert MixedPrecisionMode.FP16.value == "fp16"
        assert MixedPrecisionMode.BF16.value == "bf16"

    run_test("MixedPrecisionMode enum values", test_mixed_precision_values)

    # -----------------------------------------------------------------------
    # MemoryConfig tests
    # -----------------------------------------------------------------------

    def test_memory_config_defaults():
        mc = MemoryConfig()
        assert mc.activation_checkpointing is False
        assert mc.cpu_offload is False
        assert mc.pin_memory is True
        assert mc.gradient_as_bucket_view is True

    run_test("MemoryConfig defaults", test_memory_config_defaults)

    def test_memory_config_validate_valid():
        mc = MemoryConfig()
        assert len(mc.validate()) == 0

    run_test("MemoryConfig validate (valid)", test_memory_config_validate_valid)

    def test_memory_config_validate_invalid():
        mc = MemoryConfig(empty_cache_frequency=-1)
        errors = mc.validate()
        assert len(errors) > 0

    run_test("MemoryConfig validate (invalid)", test_memory_config_validate_invalid)

    # -----------------------------------------------------------------------
    # CommunicationConfig tests
    # -----------------------------------------------------------------------

    def test_comm_config_defaults():
        cc = CommunicationConfig()
        assert cc.backend == "nccl"
        assert cc.timeout_seconds == 1800
        assert cc.bucket_cap_mb == 25

    run_test("CommunicationConfig defaults", test_comm_config_defaults)

    def test_comm_config_validate_valid():
        cc = CommunicationConfig()
        assert len(cc.validate()) == 0

    run_test("CommunicationConfig validate (valid)", test_comm_config_validate_valid)

    def test_comm_config_validate_bad_backend():
        cc = CommunicationConfig(backend="mpi")
        errors = cc.validate()
        assert any("backend" in e for e in errors)

    run_test("CommunicationConfig validate (bad backend)", test_comm_config_validate_bad_backend)

    def test_comm_config_validate_bad_timeout():
        cc = CommunicationConfig(timeout_seconds=0)
        errors = cc.validate()
        assert any("timeout" in e for e in errors)

    run_test("CommunicationConfig validate (bad timeout)", test_comm_config_validate_bad_timeout)

    # -----------------------------------------------------------------------
    # CheckpointConfig tests
    # -----------------------------------------------------------------------

    def test_ckpt_config_defaults():
        ck = CheckpointConfig()
        assert ck.save_sharded is True
        assert ck.save_full is True
        assert ck.save_optimizer is True

    run_test("CheckpointConfig defaults", test_ckpt_config_defaults)

    def test_ckpt_config_validate_both_false():
        ck = CheckpointConfig(save_sharded=False, save_full=False)
        errors = ck.validate()
        assert any("save_sharded" in e or "save_full" in e for e in errors)

    run_test("CheckpointConfig validate (both save modes false)", test_ckpt_config_validate_both_false)

    def test_ckpt_config_validate_negative_interval():
        ck = CheckpointConfig(save_interval_steps=-1)
        errors = ck.validate()
        assert len(errors) > 0

    run_test("CheckpointConfig validate (negative interval)", test_ckpt_config_validate_negative_interval)

    # -----------------------------------------------------------------------
    # DistributedConfig defaults and basic properties
    # -----------------------------------------------------------------------

    def test_config_defaults():
        cfg = DistributedConfig()
        assert cfg.strategy == "ddp"
        assert cfg.fsdp_sharding == "full_shard"
        assert cfg.gradient_accumulation_steps == 1
        assert cfg.micro_batch_size == 32
        assert cfg.learning_rate == 3e-4

    run_test("DistributedConfig defaults", test_config_defaults)

    def test_config_world_size():
        cfg = DistributedConfig(num_nodes=2, nproc_per_node=8)
        assert cfg.world_size == 16

    run_test("DistributedConfig world_size", test_config_world_size)

    def test_config_effective_batch_size():
        cfg = DistributedConfig(
            micro_batch_size=4,
            gradient_accumulation_steps=16,
            nproc_per_node=8,
        )
        assert cfg.effective_batch_size == 4 * 16 * 8

    run_test("DistributedConfig effective_batch_size", test_config_effective_batch_size)

    def test_config_is_distributed():
        cfg1 = DistributedConfig(strategy="single")
        assert cfg1.is_distributed is False

        cfg2 = DistributedConfig(strategy="ddp", nproc_per_node=4)
        assert cfg2.is_distributed is True

    run_test("DistributedConfig is_distributed", test_config_is_distributed)

    def test_config_needs_fsdp():
        cfg1 = DistributedConfig(strategy="ddp")
        assert cfg1.needs_fsdp is False

        cfg2 = DistributedConfig(strategy="fsdp")
        assert cfg2.needs_fsdp is True

    run_test("DistributedConfig needs_fsdp", test_config_needs_fsdp)

    def test_config_backend_shorthand():
        cfg = DistributedConfig(communication=CommunicationConfig(backend="gloo"))
        assert cfg.backend == "gloo"

    run_test("DistributedConfig backend shorthand", test_config_backend_shorthand)

    # -----------------------------------------------------------------------
    # Scaled learning rate tests
    # -----------------------------------------------------------------------

    def test_scaled_lr_identity():
        cfg = DistributedConfig(
            learning_rate=3e-4,
            micro_batch_size=256,
            gradient_accumulation_steps=1,
            nproc_per_node=1,
            reference_batch_size=256,
            lr_scaling_mode="linear",
        )
        assert abs(cfg.scaled_learning_rate - 3e-4) < 1e-10

    run_test("scaled_lr identity when batch matches ref", test_scaled_lr_identity)

    def test_scaled_lr_linear():
        cfg = DistributedConfig(
            learning_rate=3e-4,
            micro_batch_size=4,
            gradient_accumulation_steps=16,
            nproc_per_node=8,
            reference_batch_size=256,
            lr_scaling_mode="linear",
        )
        # effective = 4*16*8 = 512, ratio = 2
        assert abs(cfg.scaled_learning_rate - 6e-4) < 1e-10

    run_test("scaled_lr linear 2x", test_scaled_lr_linear)

    def test_scaled_lr_sqrt():
        cfg = DistributedConfig(
            learning_rate=3e-4,
            micro_batch_size=4,
            gradient_accumulation_steps=16,
            nproc_per_node=8,
            reference_batch_size=256,
            lr_scaling_mode="sqrt",
        )
        expected = 3e-4 * math.sqrt(2.0)
        assert abs(cfg.scaled_learning_rate - expected) < 1e-10

    run_test("scaled_lr sqrt 2x", test_scaled_lr_sqrt)

    def test_scaled_warmup():
        cfg = DistributedConfig(
            warmup_steps=2000,
            micro_batch_size=4,
            gradient_accumulation_steps=16,
            nproc_per_node=8,
            reference_batch_size=256,
        )
        # ratio = 2, scaled_warmup = 4000
        assert cfg.scaled_warmup_steps == 4000

    run_test("scaled_warmup_steps", test_scaled_warmup)

    # -----------------------------------------------------------------------
    # Validation tests
    # -----------------------------------------------------------------------

    def test_validate_default_valid():
        cfg = DistributedConfig()
        errors = cfg.validate()
        assert len(errors) == 0, f"Errors: {errors}"

    run_test("validate default config is valid", test_validate_default_valid)

    def test_validate_bad_strategy():
        cfg = DistributedConfig(strategy="horovod")
        errors = cfg.validate()
        assert any("strategy" in e for e in errors)

    run_test("validate bad strategy", test_validate_bad_strategy)

    def test_validate_bad_sharding():
        cfg = DistributedConfig(fsdp_sharding="hybrid")
        errors = cfg.validate()
        assert any("fsdp_sharding" in e for e in errors)

    run_test("validate bad fsdp_sharding", test_validate_bad_sharding)

    def test_validate_bad_precision():
        cfg = DistributedConfig(mixed_precision_policy="int8")
        errors = cfg.validate()
        assert any("mixed_precision" in e for e in errors)

    run_test("validate bad mixed_precision_policy", test_validate_bad_precision)

    def test_validate_bad_accum_steps():
        cfg = DistributedConfig(gradient_accumulation_steps=0)
        errors = cfg.validate()
        assert any("gradient_accumulation" in e for e in errors)

    run_test("validate bad gradient_accumulation_steps", test_validate_bad_accum_steps)

    def test_validate_fsdp_single_gpu():
        cfg = DistributedConfig(strategy="fsdp", nproc_per_node=1)
        errors = cfg.validate()
        assert any("FSDP" in e for e in errors)

    run_test("validate FSDP requires multi-GPU", test_validate_fsdp_single_gpu)

    def test_validate_ddp_cpu_offload():
        cfg = DistributedConfig(
            strategy="ddp",
            memory=MemoryConfig(cpu_offload=True),
        )
        errors = cfg.validate()
        assert any("cpu_offload" in e for e in errors)

    run_test("validate DDP with cpu_offload", test_validate_ddp_cpu_offload)

    def test_validate_bad_lr_mode():
        cfg = DistributedConfig(lr_scaling_mode="cubic")
        errors = cfg.validate()
        assert any("lr_scaling" in e for e in errors)

    run_test("validate bad lr_scaling_mode", test_validate_bad_lr_mode)

    def test_validate_bad_port():
        cfg = DistributedConfig(master_port="not_a_number")
        errors = cfg.validate()
        assert any("master_port" in e for e in errors)

    run_test("validate bad master_port", test_validate_bad_port)

    # -----------------------------------------------------------------------
    # Preset tests
    # -----------------------------------------------------------------------

    def test_preset_dev():
        cfg = DistributedConfig.for_dev()
        assert cfg.strategy == "single"
        assert cfg.micro_batch_size == 32
        assert cfg.gradient_accumulation_steps == 1
        assert cfg.is_valid()

    run_test("preset: dev", test_preset_dev)

    def test_preset_1b():
        cfg = DistributedConfig.for_1b(num_gpus=4)
        assert cfg.strategy == "ddp"
        assert cfg.nproc_per_node == 4
        assert cfg.gradient_accumulation_steps == 4
        assert cfg.micro_batch_size == 16
        assert cfg.effective_batch_size == 16 * 4 * 4
        assert cfg.is_valid()

    run_test("preset: 1B", test_preset_1b)

    def test_preset_3b():
        cfg = DistributedConfig.for_3b(num_gpus=8)
        assert cfg.strategy == "fsdp"
        assert cfg.fsdp_sharding == "shard_grad_op"
        assert cfg.nproc_per_node == 8
        assert cfg.memory.activation_checkpointing is True
        assert cfg.is_valid()

    run_test("preset: 3B", test_preset_3b)

    def test_preset_7b():
        cfg = DistributedConfig.for_7b(num_gpus=8)
        assert cfg.strategy == "fsdp"
        assert cfg.fsdp_sharding == "full_shard"
        assert cfg.gradient_accumulation_steps == 16
        assert cfg.micro_batch_size == 4
        assert cfg.effective_batch_size == 4 * 16 * 8
        assert cfg.memory.activation_checkpointing is True
        assert cfg.memory.limit_all_gathers is True
        assert cfg.is_valid()

    run_test("preset: 7B", test_preset_7b)

    def test_preset_7b_constrained():
        cfg = DistributedConfig.for_7b_constrained(num_gpus=8)
        assert cfg.strategy == "fsdp"
        assert cfg.memory.cpu_offload is True
        assert cfg.gradient_accumulation_steps == 32
        assert cfg.micro_batch_size == 2
        assert cfg.is_valid()

    run_test("preset: 7B constrained", test_preset_7b_constrained)

    def test_preset_multi_node_7b():
        cfg = DistributedConfig.for_multi_node_7b(num_nodes=2, num_gpus_per_node=8)
        assert cfg.num_nodes == 2
        assert cfg.world_size == 16
        assert cfg.effective_batch_size == 4 * 8 * 16
        assert cfg.is_valid()

    run_test("preset: multi-node 7B", test_preset_multi_node_7b)

    # -----------------------------------------------------------------------
    # Memory estimation tests
    # -----------------------------------------------------------------------

    def test_estimate_memory_1b():
        cfg = DistributedConfig.for_1b(num_gpus=4)
        mem = cfg.estimate_memory_gb(1_000_000_000)
        assert mem["total_gb"] > 0
        assert mem["params_gb"] > 0

    run_test("estimate_memory_gb 1B", test_estimate_memory_1b)

    def test_estimate_memory_7b():
        cfg = DistributedConfig.for_7b(num_gpus=8)
        mem = cfg.estimate_memory_gb(7_000_000_000)
        # With FULL_SHARD on 8 GPUs, params should be ~1/8
        assert mem["params_gb"] < 5.0  # 7B * 2 bytes / 8 GPUs = ~1.75GB

    run_test("estimate_memory_gb 7B FULL_SHARD", test_estimate_memory_7b)

    def test_estimate_memory_reduces_with_checkpointing():
        cfg_no_ckpt = DistributedConfig.for_7b(num_gpus=8)
        cfg_no_ckpt.memory.activation_checkpointing = False
        mem_no = cfg_no_ckpt.estimate_memory_gb(7_000_000_000)

        cfg_ckpt = DistributedConfig.for_7b(num_gpus=8)
        cfg_ckpt.memory.activation_checkpointing = True
        mem_yes = cfg_ckpt.estimate_memory_gb(7_000_000_000)

        assert mem_yes["activations_gb"] < mem_no["activations_gb"]

    run_test("activation checkpointing reduces memory estimate", test_estimate_memory_reduces_with_checkpointing)

    # -----------------------------------------------------------------------
    # Serialization tests
    # -----------------------------------------------------------------------

    def test_to_dict():
        cfg = DistributedConfig.for_7b()
        d = cfg.to_dict()
        assert "strategy" in d
        assert "memory.activation_checkpointing" in d
        assert "communication.backend" in d
        assert "checkpoint.save_full" in d

    run_test("to_dict", test_to_dict)

    def test_from_dict_roundtrip():
        cfg = DistributedConfig.for_3b()
        d = cfg.to_dict()
        cfg2 = DistributedConfig.from_dict(d)
        assert cfg2.strategy == cfg.strategy
        assert cfg2.fsdp_sharding == cfg.fsdp_sharding
        assert cfg2.micro_batch_size == cfg.micro_batch_size
        assert cfg2.memory.activation_checkpointing == cfg.memory.activation_checkpointing
        assert cfg2.communication.backend == cfg.communication.backend

    run_test("from_dict roundtrip", test_from_dict_roundtrip)

    # -----------------------------------------------------------------------
    # Summary test
    # -----------------------------------------------------------------------

    def test_summary():
        cfg = DistributedConfig.for_7b()
        s = cfg.summary()
        assert s["strategy"] == "fsdp"
        assert s["effective_batch_size"] == 512
        assert "scaled_lr" in s
        assert s["world_size"] == 8

    run_test("summary", test_summary)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed, {total} total")
    print("=" * 70)

    sys.exit(0 if failed == 0 else 1)
