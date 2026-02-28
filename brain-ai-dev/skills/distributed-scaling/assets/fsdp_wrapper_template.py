"""
FSDPWrapper — FullyShardedDataParallel wrapper for the brain_ai system.

Provides FSDP wrapping with BrainAI-specific sharding policies,
activation checkpointing, CPU offloading, mixed precision, and
distributed checkpointing.

Key classes:
    FSDPWrapper      — Main wrapper for FSDP training
    ShardingPolicy   — Configuration for how modules are sharded

Self-tests in __main__ validate all functionality without actual multi-GPU hardware.
"""

from __future__ import annotations

import copy
import logging
import os
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports with fallbacks
# ---------------------------------------------------------------------------

_FSDP_AVAILABLE = False
_DCP_AVAILABLE = False

try:
    import torch.distributed as dist
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        ShardingStrategy,
        CPUOffload,
        MixedPrecision,
        FullStateDictConfig,
        StateDictType,
    )
    from torch.distributed.fsdp.wrap import (
        ModuleWrapPolicy,
        size_based_auto_wrap_policy,
    )
    _FSDP_AVAILABLE = True
except ImportError:
    dist = None
    FSDP = None
    ShardingStrategy = None

try:
    import torch.distributed.checkpoint as dcp
    _DCP_AVAILABLE = True
except ImportError:
    dcp = None


# ===========================================================================
# SECTION 1: ShardingPolicy dataclass
# ===========================================================================

class ShardingStrategyEnum(str, Enum):
    """Supported FSDP sharding strategies."""
    FULL_SHARD = "full_shard"
    SHARD_GRAD_OP = "shard_grad_op"
    NO_SHARD = "no_shard"


@dataclass
class ShardingPolicy:
    """Configuration for FSDP sharding.

    Attributes:
        strategy: Sharding strategy name.
        wrap_module_classes: Set of module class names to wrap individually.
        min_num_params: Minimum param count for size-based wrapping (alternative).
        activation_checkpointing: Enable gradient checkpointing on wrapped modules.
        cpu_offload: Offload sharded parameters to CPU.
        mixed_precision_dtype: Precision for mixed-precision training.
    """
    strategy: str = "full_shard"
    wrap_module_classes: List[str] = field(default_factory=lambda: [
        "VisionEncoder", "TextEncoder", "AudioEncoder", "SensorEncoder",
        "SNNCore", "HTMLayer", "GlobalWorkspace",
        "ActiveInferenceAgent", "DecisionHeads",
        "DualProcessReasoner", "NeuromodulatoryGate",
    ])
    min_num_params: int = 100_000_000  # 100M for size-based policy
    activation_checkpointing: bool = False
    cpu_offload: bool = False
    mixed_precision_dtype: str = "bfloat16"

    def get_torch_strategy(self) -> Optional[Any]:
        """Convert string strategy to torch ShardingStrategy enum."""
        if not _FSDP_AVAILABLE:
            return None
        mapping = {
            "full_shard": ShardingStrategy.FULL_SHARD,
            "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
            "no_shard": ShardingStrategy.NO_SHARD,
        }
        return mapping.get(self.strategy, ShardingStrategy.FULL_SHARD)

    def get_mixed_precision(self) -> Optional[Any]:
        """Create MixedPrecision policy."""
        if not _FSDP_AVAILABLE:
            return None
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        dt = dtype_map.get(self.mixed_precision_dtype, torch.bfloat16)
        if dt == torch.float32:
            return None  # No mixed precision needed
        return MixedPrecision(
            param_dtype=dt,
            reduce_dtype=dt,
            buffer_dtype=dt,
        )

    @classmethod
    def for_1b(cls) -> "ShardingPolicy":
        return cls(
            strategy="shard_grad_op",
            activation_checkpointing=False,
            cpu_offload=False,
            mixed_precision_dtype="bfloat16",
        )

    @classmethod
    def for_3b(cls) -> "ShardingPolicy":
        return cls(
            strategy="shard_grad_op",
            activation_checkpointing=True,
            cpu_offload=False,
            mixed_precision_dtype="bfloat16",
        )

    @classmethod
    def for_7b(cls) -> "ShardingPolicy":
        return cls(
            strategy="full_shard",
            activation_checkpointing=True,
            cpu_offload=False,
            mixed_precision_dtype="bfloat16",
        )

    @classmethod
    def for_7b_constrained(cls) -> "ShardingPolicy":
        """7B on smaller GPUs (40GB) with CPU offload."""
        return cls(
            strategy="full_shard",
            activation_checkpointing=True,
            cpu_offload=True,
            mixed_precision_dtype="bfloat16",
        )


# ===========================================================================
# SECTION 2: FSDPWrapper
# ===========================================================================

class FSDPWrapper:
    """FullyShardedDataParallel wrapper for BrainAI models.

    Handles:
    - FSDP wrapping with brain_ai-specific module policies
    - Sharding strategy selection by model scale
    - Activation checkpointing
    - CPU offloading
    - Mixed precision
    - Distributed checkpoint save/load
    - Full state dict consolidation at phase boundaries

    Args:
        model: The nn.Module to wrap (typically BrainAI).
        policy: ShardingPolicy configuration.
        backend: Communication backend ("nccl" or "gloo").
    """

    def __init__(
        self,
        model: nn.Module,
        policy: Optional[ShardingPolicy] = None,
        backend: str = "nccl",
    ):
        self.model = model
        self.policy = policy or ShardingPolicy()
        self.backend = backend

        self._fsdp_model: Optional[nn.Module] = None
        self._rank = 0
        self._world_size = 1

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def is_main(self) -> bool:
        return self._rank == 0

    @property
    def wrapped_model(self) -> nn.Module:
        if self._fsdp_model is not None:
            return self._fsdp_model
        return self.model

    @property
    def inner_model(self) -> nn.Module:
        """Access the inner unwrapped model (for state_dict, etc.)."""
        if self._fsdp_model is not None:
            # FSDP doesn't have .module like DDP
            return self.model
        return self.model

    def get_sharding_policy(self) -> ShardingPolicy:
        """Return the current sharding policy."""
        return self.policy

    def _build_wrap_policy(self) -> Optional[Any]:
        """Build the FSDP auto_wrap_policy from the sharding policy."""
        if not _FSDP_AVAILABLE:
            return None

        # Collect actual module classes from the model
        target_names = set(self.policy.wrap_module_classes)
        wrap_classes = set()

        for module in self.model.modules():
            if type(module).__name__ in target_names:
                wrap_classes.add(type(module))

        if wrap_classes:
            return ModuleWrapPolicy(wrap_classes)

        # Fallback to size-based
        import functools
        return functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=self.policy.min_num_params,
        )

    def setup(
        self,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ) -> nn.Module:
        """Initialize process group and wrap model with FSDP.

        Args:
            rank: Process rank (auto-detected if None).
            world_size: Total number of processes (auto-detected if None).

        Returns:
            The FSDP-wrapped model (or raw model if single-process).
        """
        # Determine rank and world_size
        if rank is not None:
            self._rank = rank
        elif _FSDP_AVAILABLE and dist is not None and dist.is_initialized():
            self._rank = dist.get_rank()
        else:
            self._rank = int(os.environ.get("RANK", "0"))

        if world_size is not None:
            self._world_size = world_size
        elif _FSDP_AVAILABLE and dist is not None and dist.is_initialized():
            self._world_size = dist.get_world_size()
        else:
            self._world_size = int(os.environ.get("WORLD_SIZE", "1"))

        if self._world_size <= 1 or not _FSDP_AVAILABLE:
            logger.info("Single-process mode; returning unwrapped model.")
            return self.model

        if not dist.is_initialized():
            logger.warning("Process group not initialized; returning unwrapped model.")
            return self.model

        # Build wrapping policy
        auto_wrap_policy = self._build_wrap_policy()

        # Build FSDP kwargs
        fsdp_kwargs = {
            "sharding_strategy": self.policy.get_torch_strategy(),
            "auto_wrap_policy": auto_wrap_policy,
        }

        # CPU offload
        if self.policy.cpu_offload:
            fsdp_kwargs["cpu_offload"] = CPUOffload(offload_params=True)

        # Mixed precision
        mp = self.policy.get_mixed_precision()
        if mp is not None:
            fsdp_kwargs["mixed_precision"] = mp

        # Wrap
        self._fsdp_model = FSDP(self.model, **fsdp_kwargs)

        # Activation checkpointing
        if self.policy.activation_checkpointing:
            self._apply_activation_checkpointing()

        logger.info(
            f"Model wrapped with FSDP on rank {self._rank} "
            f"(strategy={self.policy.strategy}, "
            f"cpu_offload={self.policy.cpu_offload})"
        )
        return self._fsdp_model

    def _apply_activation_checkpointing(self) -> None:
        """Apply activation checkpointing to large modules."""
        if not _FSDP_AVAILABLE or self._fsdp_model is None:
            return

        try:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                checkpoint_wrapper,
                apply_activation_checkpointing,
            )
        except ImportError:
            logger.warning("Activation checkpointing not available in this PyTorch version.")
            return

        target_names = set(self.policy.wrap_module_classes)

        def check_fn(module: nn.Module) -> bool:
            return type(module).__name__ in target_names

        apply_activation_checkpointing(
            self._fsdp_model,
            checkpoint_wrapper_fn=checkpoint_wrapper,
            check_fn=check_fn,
        )
        logger.info("Activation checkpointing applied.")

    def cleanup(self) -> None:
        """Release FSDP resources."""
        self._fsdp_model = None

    def save_distributed_checkpoint(self, path: str) -> None:
        """Save a sharded distributed checkpoint.

        Each rank saves its own shard. Fast save/load with same config.

        Args:
            path: Directory path for the checkpoint.
        """
        if self._fsdp_model is None:
            # Single process: standard save
            if self.is_main:
                os.makedirs(path, exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(path, "model.pt"))
            return

        if not _DCP_AVAILABLE:
            logger.warning("torch.distributed.checkpoint not available; using fallback.")
            self._save_full_state_dict(os.path.join(path, "model_full.pt"))
            return

        os.makedirs(path, exist_ok=True)
        dcp.save(
            state_dict={"model": self._fsdp_model.state_dict()},
            storage_writer=dcp.FileSystemWriter(path),
        )
        if _FSDP_AVAILABLE and dist.is_initialized():
            dist.barrier()
        logger.info(f"Distributed checkpoint saved to {path}")

    def load_distributed_checkpoint(self, path: str) -> None:
        """Load a sharded distributed checkpoint.

        Args:
            path: Directory path of the checkpoint.
        """
        if self._fsdp_model is None:
            # Single process: standard load
            state_path = os.path.join(path, "model.pt")
            if os.path.exists(state_path):
                state = torch.load(state_path, weights_only=False)
                self.model.load_state_dict(state)
            return

        if not _DCP_AVAILABLE:
            logger.warning("torch.distributed.checkpoint not available; using fallback.")
            return

        state_dict = {"model": self._fsdp_model.state_dict()}
        dcp.load(
            state_dict=state_dict,
            storage_reader=dcp.FileSystemReader(path),
        )
        self._fsdp_model.load_state_dict(state_dict["model"])
        logger.info(f"Distributed checkpoint loaded from {path}")

    def _save_full_state_dict(self, path: str) -> None:
        """Save a full (non-sharded) state dict for portability."""
        if not _FSDP_AVAILABLE or self._fsdp_model is None:
            if self.is_main:
                torch.save(self.model.state_dict(), path)
            return

        full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
            self._fsdp_model, StateDictType.FULL_STATE_DICT, full_cfg
        ):
            state = self._fsdp_model.state_dict()
            if self.is_main:
                torch.save(state, path)

        if dist.is_initialized():
            dist.barrier()
        logger.info(f"Full state dict saved to {path}")

    def save_phase_checkpoint(self, phase: int, base_dir: str) -> None:
        """Save checkpoint at phase boundary: both sharded and full.

        Args:
            phase: Current training phase number (1-7).
            base_dir: Base directory for checkpoints.
        """
        sharded_dir = os.path.join(base_dir, f"phase{phase}_sharded")
        full_path = os.path.join(base_dir, f"phase{phase}_full.pt")

        self.save_distributed_checkpoint(sharded_dir)
        self._save_full_state_dict(full_path)

    def get_parameter_summary(self) -> Dict[str, int]:
        """Return parameter count summary."""
        model = self.wrapped_model
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": total - trainable,
        }

    def get_memory_summary(self) -> Dict[str, float]:
        """Return GPU memory usage in GB."""
        if not torch.cuda.is_available():
            return {
                "allocated_gb": 0.0,
                "reserved_gb": 0.0,
                "max_allocated_gb": 0.0,
            }
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        }


# ===========================================================================
# SECTION 3: Factory functions
# ===========================================================================

def create_fsdp_wrapper(
    model: nn.Module,
    scale: str = "7b",
    backend: str = "nccl",
) -> FSDPWrapper:
    """Create an FSDPWrapper with preset policy for the given scale.

    Args:
        model: The model to wrap.
        scale: "1b", "3b", "7b", or "7b_constrained".
        backend: Communication backend.

    Returns:
        Configured FSDPWrapper.
    """
    policy_map = {
        "1b": ShardingPolicy.for_1b,
        "3b": ShardingPolicy.for_3b,
        "7b": ShardingPolicy.for_7b,
        "7b_constrained": ShardingPolicy.for_7b_constrained,
    }
    factory = policy_map.get(scale, ShardingPolicy.for_7b)
    policy = factory()
    return FSDPWrapper(model, policy=policy, backend=backend)


# ===========================================================================
# SECTION 4: Self-tests
# ===========================================================================

if __name__ == "__main__":
    import sys
    import tempfile
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
    print("FSDPWrapper Self-Tests (single-process simulation)")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Helper models
    # -----------------------------------------------------------------------

    class SimpleModel(nn.Module):
        def __init__(self, dim=64):
            super().__init__()
            self.linear1 = nn.Linear(dim, dim * 2)
            self.bn = nn.BatchNorm1d(dim * 2)
            self.linear2 = nn.Linear(dim * 2, 10)

        def forward(self, x):
            return self.linear2(torch.relu(self.bn(self.linear1(x))))

    class LargeModule(nn.Module):
        """Simulates a large BrainAI submodule."""
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, out_dim),
            )

        def forward(self, x):
            return self.layers(x)

    class MockBrainAI(nn.Module):
        """Simplified BrainAI-like model for testing."""
        def __init__(self):
            super().__init__()
            self.encoder = LargeModule(32, 64)
            self.workspace = LargeModule(64, 64)
            self.head = nn.Linear(64, 10)

        def forward(self, x):
            enc = self.encoder(x)
            ws = self.workspace(enc)
            return self.head(ws)

    # -----------------------------------------------------------------------
    # ShardingPolicy tests
    # -----------------------------------------------------------------------

    def test_policy_defaults():
        policy = ShardingPolicy()
        assert policy.strategy == "full_shard"
        assert "VisionEncoder" in policy.wrap_module_classes
        assert policy.activation_checkpointing is False
        assert policy.cpu_offload is False

    run_test("ShardingPolicy defaults", test_policy_defaults)

    def test_policy_for_1b():
        policy = ShardingPolicy.for_1b()
        assert policy.strategy == "shard_grad_op"
        assert policy.activation_checkpointing is False
        assert policy.cpu_offload is False

    run_test("ShardingPolicy.for_1b preset", test_policy_for_1b)

    def test_policy_for_3b():
        policy = ShardingPolicy.for_3b()
        assert policy.strategy == "shard_grad_op"
        assert policy.activation_checkpointing is True

    run_test("ShardingPolicy.for_3b preset", test_policy_for_3b)

    def test_policy_for_7b():
        policy = ShardingPolicy.for_7b()
        assert policy.strategy == "full_shard"
        assert policy.activation_checkpointing is True
        assert policy.cpu_offload is False

    run_test("ShardingPolicy.for_7b preset", test_policy_for_7b)

    def test_policy_for_7b_constrained():
        policy = ShardingPolicy.for_7b_constrained()
        assert policy.strategy == "full_shard"
        assert policy.activation_checkpointing is True
        assert policy.cpu_offload is True

    run_test("ShardingPolicy.for_7b_constrained preset", test_policy_for_7b_constrained)

    def test_policy_mixed_precision_bf16():
        policy = ShardingPolicy(mixed_precision_dtype="bfloat16")
        mp = policy.get_mixed_precision()
        if _FSDP_AVAILABLE:
            assert mp is not None
        # Without FSDP, returns None gracefully

    run_test("ShardingPolicy mixed precision bf16", test_policy_mixed_precision_bf16)

    def test_policy_mixed_precision_fp32():
        policy = ShardingPolicy(mixed_precision_dtype="float32")
        mp = policy.get_mixed_precision()
        assert mp is None  # No mixed precision for fp32

    run_test("ShardingPolicy mixed precision fp32 returns None", test_policy_mixed_precision_fp32)

    def test_policy_strategy_mapping():
        for strategy in ["full_shard", "shard_grad_op", "no_shard"]:
            policy = ShardingPolicy(strategy=strategy)
            ts = policy.get_torch_strategy()
            if _FSDP_AVAILABLE:
                assert ts is not None

    run_test("ShardingPolicy strategy mapping", test_policy_strategy_mapping)

    # -----------------------------------------------------------------------
    # FSDPWrapper construction tests
    # -----------------------------------------------------------------------

    def test_wrapper_construction():
        model = SimpleModel()
        wrapper = FSDPWrapper(model)
        assert wrapper.rank == 0
        assert wrapper.world_size == 1
        assert wrapper.is_main is True

    run_test("FSDPWrapper construction", test_wrapper_construction)

    def test_wrapper_setup_single():
        model = SimpleModel()
        wrapper = FSDPWrapper(model, backend="gloo")
        returned = wrapper.setup(rank=0, world_size=1)
        assert returned is model
        wrapper.cleanup()

    run_test("FSDPWrapper setup single-process", test_wrapper_setup_single)

    def test_wrapper_with_custom_policy():
        model = SimpleModel()
        policy = ShardingPolicy(strategy="shard_grad_op", cpu_offload=True)
        wrapper = FSDPWrapper(model, policy=policy)
        assert wrapper.policy.strategy == "shard_grad_op"
        assert wrapper.policy.cpu_offload is True

    run_test("FSDPWrapper with custom policy", test_wrapper_with_custom_policy)

    # -----------------------------------------------------------------------
    # Model forward/backward in single-process
    # -----------------------------------------------------------------------

    def test_forward_single():
        model = SimpleModel()
        wrapper = FSDPWrapper(model, backend="gloo")
        wrapper.setup(rank=0, world_size=1)
        x = torch.randn(4, 64)
        out = wrapper.wrapped_model(x)
        assert out.shape == (4, 10)
        wrapper.cleanup()

    run_test("forward pass single-process", test_forward_single)

    def test_backward_single():
        model = SimpleModel()
        wrapper = FSDPWrapper(model, backend="gloo")
        wrapper.setup(rank=0, world_size=1)
        x = torch.randn(4, 64)
        out = wrapper.wrapped_model(x)
        loss = out.sum()
        loss.backward()
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad
        wrapper.cleanup()

    run_test("backward pass single-process", test_backward_single)

    # -----------------------------------------------------------------------
    # Checkpoint tests (single-process, file-based)
    # -----------------------------------------------------------------------

    def test_save_distributed_checkpoint_single():
        model = SimpleModel()
        wrapper = FSDPWrapper(model, backend="gloo")
        wrapper.setup(rank=0, world_size=1)
        with tempfile.TemporaryDirectory() as tmp_dir:
            wrapper.save_distributed_checkpoint(tmp_dir)
            assert os.path.exists(os.path.join(tmp_dir, "model.pt"))
        wrapper.cleanup()

    run_test("save_distributed_checkpoint single-process", test_save_distributed_checkpoint_single)

    def test_load_distributed_checkpoint_single():
        model = SimpleModel()
        wrapper = FSDPWrapper(model, backend="gloo")
        wrapper.setup(rank=0, world_size=1)

        with tempfile.TemporaryDirectory() as tmp_dir:
            wrapper.save_distributed_checkpoint(tmp_dir)

            # Modify model
            with torch.no_grad():
                for p in model.parameters():
                    p.fill_(999.0)

            wrapper.load_distributed_checkpoint(tmp_dir)

            # Should be restored
            first_param = next(model.parameters())
            assert not torch.all(first_param == 999.0)

        wrapper.cleanup()

    run_test("load_distributed_checkpoint roundtrip", test_load_distributed_checkpoint_single)

    def test_save_full_state_dict_single():
        model = SimpleModel()
        wrapper = FSDPWrapper(model, backend="gloo")
        wrapper.setup(rank=0, world_size=1)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "full.pt")
            wrapper._save_full_state_dict(path)
            assert os.path.exists(path)
            state = torch.load(path, weights_only=False)
            assert "linear1.weight" in state
        wrapper.cleanup()

    run_test("save full state dict single-process", test_save_full_state_dict_single)

    def test_phase_checkpoint_single():
        model = SimpleModel()
        wrapper = FSDPWrapper(model, backend="gloo")
        wrapper.setup(rank=0, world_size=1)
        with tempfile.TemporaryDirectory() as tmp_dir:
            wrapper.save_phase_checkpoint(phase=1, base_dir=tmp_dir)
            assert os.path.isdir(os.path.join(tmp_dir, "phase1_sharded"))
            assert os.path.exists(os.path.join(tmp_dir, "phase1_full.pt"))
        wrapper.cleanup()

    run_test("save_phase_checkpoint creates both formats", test_phase_checkpoint_single)

    # -----------------------------------------------------------------------
    # Parameter summary tests
    # -----------------------------------------------------------------------

    def test_parameter_summary():
        model = SimpleModel(64)
        wrapper = FSDPWrapper(model, backend="gloo")
        wrapper.setup(rank=0, world_size=1)
        summary = wrapper.get_parameter_summary()
        assert summary["total_parameters"] > 0
        assert summary["trainable_parameters"] > 0
        assert summary["frozen_parameters"] == 0
        wrapper.cleanup()

    run_test("get_parameter_summary", test_parameter_summary)

    def test_parameter_summary_frozen():
        model = SimpleModel(64)
        # Freeze first layer
        for p in model.linear1.parameters():
            p.requires_grad = False
        wrapper = FSDPWrapper(model, backend="gloo")
        wrapper.setup(rank=0, world_size=1)
        summary = wrapper.get_parameter_summary()
        assert summary["frozen_parameters"] > 0
        assert summary["trainable_parameters"] < summary["total_parameters"]
        wrapper.cleanup()

    run_test("get_parameter_summary with frozen params", test_parameter_summary_frozen)

    # -----------------------------------------------------------------------
    # Memory summary tests
    # -----------------------------------------------------------------------

    def test_memory_summary():
        model = SimpleModel()
        wrapper = FSDPWrapper(model, backend="gloo")
        wrapper.setup(rank=0, world_size=1)
        mem = wrapper.get_memory_summary()
        assert "allocated_gb" in mem
        assert "reserved_gb" in mem
        assert "max_allocated_gb" in mem
        wrapper.cleanup()

    run_test("get_memory_summary", test_memory_summary)

    # -----------------------------------------------------------------------
    # Wrap policy building tests
    # -----------------------------------------------------------------------

    def test_build_wrap_policy_known_classes():
        model = MockBrainAI()
        policy = ShardingPolicy(wrap_module_classes=["LargeModule"])
        wrapper = FSDPWrapper(model, policy=policy)
        wp = wrapper._build_wrap_policy()
        if _FSDP_AVAILABLE:
            assert wp is not None

    run_test("build_wrap_policy with known classes", test_build_wrap_policy_known_classes)

    def test_build_wrap_policy_no_match():
        model = SimpleModel()
        policy = ShardingPolicy(wrap_module_classes=["NonexistentModule"])
        wrapper = FSDPWrapper(model, policy=policy)
        wp = wrapper._build_wrap_policy()
        # Falls back to size-based policy
        if _FSDP_AVAILABLE:
            assert wp is not None

    run_test("build_wrap_policy fallback to size-based", test_build_wrap_policy_no_match)

    # -----------------------------------------------------------------------
    # Factory function tests
    # -----------------------------------------------------------------------

    def test_create_fsdp_wrapper_1b():
        model = SimpleModel()
        wrapper = create_fsdp_wrapper(model, scale="1b", backend="gloo")
        assert wrapper.policy.strategy == "shard_grad_op"

    run_test("create_fsdp_wrapper scale=1b", test_create_fsdp_wrapper_1b)

    def test_create_fsdp_wrapper_7b():
        model = SimpleModel()
        wrapper = create_fsdp_wrapper(model, scale="7b", backend="gloo")
        assert wrapper.policy.strategy == "full_shard"
        assert wrapper.policy.activation_checkpointing is True

    run_test("create_fsdp_wrapper scale=7b", test_create_fsdp_wrapper_7b)

    def test_create_fsdp_wrapper_unknown():
        model = SimpleModel()
        wrapper = create_fsdp_wrapper(model, scale="unknown", backend="gloo")
        # Defaults to 7b
        assert wrapper.policy.strategy == "full_shard"

    run_test("create_fsdp_wrapper unknown scale defaults to 7b", test_create_fsdp_wrapper_unknown)

    # -----------------------------------------------------------------------
    # MockBrainAI integration
    # -----------------------------------------------------------------------

    def test_mock_brain_ai_forward():
        model = MockBrainAI()
        wrapper = FSDPWrapper(model, backend="gloo")
        wrapper.setup(rank=0, world_size=1)
        x = torch.randn(8, 32)
        out = wrapper.wrapped_model(x)
        assert out.shape == (8, 10)
        wrapper.cleanup()

    run_test("MockBrainAI forward through wrapper", test_mock_brain_ai_forward)

    def test_mock_brain_ai_train_step():
        model = MockBrainAI()
        wrapper = FSDPWrapper(model, backend="gloo")
        wrapper.setup(rank=0, world_size=1)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        params_before = {n: p.clone() for n, p in model.named_parameters()}

        x = torch.randn(4, 32)
        target = torch.randint(0, 10, (4,))
        out = wrapper.wrapped_model(x)
        loss = nn.functional.cross_entropy(out, target)
        loss.backward()
        opt.step()
        opt.zero_grad()

        changed = any(
            not torch.equal(p, params_before[n])
            for n, p in model.named_parameters()
        )
        assert changed
        wrapper.cleanup()

    run_test("MockBrainAI training step", test_mock_brain_ai_train_step)

    # -----------------------------------------------------------------------
    # Multiple cleanup cycles
    # -----------------------------------------------------------------------

    def test_multiple_setup_cleanup():
        model = SimpleModel()
        wrapper = FSDPWrapper(model, backend="gloo")
        for _ in range(3):
            wrapper.setup(rank=0, world_size=1)
            x = torch.randn(2, 64)
            out = wrapper.wrapped_model(x)
            assert out.shape == (2, 10)
            wrapper.cleanup()

    run_test("multiple setup/cleanup cycles", test_multiple_setup_cleanup)

    # -----------------------------------------------------------------------
    # ShardingStrategyEnum
    # -----------------------------------------------------------------------

    def test_strategy_enum_values():
        assert ShardingStrategyEnum.FULL_SHARD.value == "full_shard"
        assert ShardingStrategyEnum.SHARD_GRAD_OP.value == "shard_grad_op"
        assert ShardingStrategyEnum.NO_SHARD.value == "no_shard"

    run_test("ShardingStrategyEnum values", test_strategy_enum_values)

    # -----------------------------------------------------------------------
    # Edge cases
    # -----------------------------------------------------------------------

    def test_inner_model_no_wrap():
        model = SimpleModel()
        wrapper = FSDPWrapper(model, backend="gloo")
        # Don't call setup
        assert wrapper.inner_model is model

    run_test("inner_model before setup", test_inner_model_no_wrap)

    def test_cleanup_before_setup():
        model = SimpleModel()
        wrapper = FSDPWrapper(model, backend="gloo")
        wrapper.cleanup()  # Should not raise

    run_test("cleanup before setup", test_cleanup_before_setup)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed, {total} total")
    print("=" * 70)

    sys.exit(0 if failed == 0 else 1)
