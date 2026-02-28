#!/usr/bin/env python3
"""
fsdp_wrapper_template.py
------------------------
FSDPWrapper: wraps an nn.Module with FSDP policies, applies mixed precision,
activation checkpointing, and provides two-tier checkpoint save/load.

Tier 1 — Portable weights: FULL_STATE_DICT with offload_to_cpu + rank0_only.
          Loadable by any strategy.

Tier 2 — Efficient resume: torch.distributed.checkpoint (DCP) sharded
          state dict. Same-strategy same-world_size resume.

Usage
-----
    wrapper = FSDPWrapper()
    model = wrapper.wrap(model, fsdp_cfg)
    wrapper.save_full_state_dict(model, "/path/full_ckpt.pt")
    wrapper.save_sharded_checkpoint(model, optimizer, "/path/sharded/")
    wrapper.load_full_state_dict(model, "/path/full_ckpt.pt")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Set, Tuple, Type

import torch
import torch.distributed as dist
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — only loaded when FSDP is actually used
# ---------------------------------------------------------------------------

def _import_fsdp():
    """Import FSDP components with version-tolerant fallback paths.

    apply_activation_checkpointing moved across PyTorch 2.x versions:
      - PyTorch <= 2.1:  torch.distributed.fsdp.wrap
      - PyTorch >= 2.2:  torch.distributed.fsdp  (top-level)
      - Some builds:     torch.distributed.fsdp.fully_sharded_data_parallel
    We try each location in order.
    """
    try:
        from torch.distributed.fsdp import (
            CPUOffload,
            FullStateDictConfig,
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
            ShardingStrategy,
            StateDictType,
        )
    except ImportError as e:
        raise ImportError(
            "PyTorch FSDP core not available. "
            f"Installed torch version: {torch.__version__}. Error: {e}"
        ) from e

    try:
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy, size_based_auto_wrap_policy
    except ImportError as e:
        raise ImportError(
            f"Cannot import ModuleWrapPolicy from torch.distributed.fsdp.wrap. Error: {e}"
        ) from e

    # apply_activation_checkpointing: try multiple locations across versions
    apply_activation_checkpointing = None
    _ac_locations = [
        ("torch.distributed.fsdp.wrap", "apply_activation_checkpointing"),
        ("torch.distributed.fsdp", "apply_activation_checkpointing"),
        ("torch.distributed.fsdp.fully_sharded_data_parallel", "apply_activation_checkpointing"),
    ]
    for mod_path, func_name in _ac_locations:
        try:
            import importlib
            _mod = importlib.import_module(mod_path)
            apply_activation_checkpointing = getattr(_mod, func_name, None)
            if apply_activation_checkpointing is not None:
                break
        except ImportError:
            continue

    if apply_activation_checkpointing is None:
        # Provide a no-op stub with a warning — activation checkpointing
        # may not be available in this PyTorch build but other FSDP features work.
        import warnings
        warnings.warn(
            "apply_activation_checkpointing not found in this PyTorch installation "
            f"({torch.__version__}). Activation checkpointing will be skipped.",
            ImportWarning,
            stacklevel=3,
        )

        def apply_activation_checkpointing(model, check_fn=None, **kwargs):
            pass  # no-op stub

    return (
        FSDP, ShardingStrategy, MixedPrecision, CPUOffload,
        StateDictType, FullStateDictConfig,
        ModuleWrapPolicy, size_based_auto_wrap_policy,
        apply_activation_checkpointing,
    )


def _import_dcp():
    """Import torch.distributed.checkpoint components."""
    try:
        import torch.distributed.checkpoint as dist_cp
        from torch.distributed.checkpoint import (
            FileSystemReader,
            FileSystemWriter,
        )
        return dist_cp, FileSystemWriter, FileSystemReader
    except ImportError as e:
        raise ImportError(
            "torch.distributed.checkpoint requires torch >= 2.0.0. "
            f"Error: {e}"
        ) from e


# ---------------------------------------------------------------------------
# Sharding strategy map
# ---------------------------------------------------------------------------

SHARDING_STRATEGY_MAP: Dict[str, str] = {
    "FULL_SHARD": "FULL_SHARD",
    "SHARD_GRAD_OP": "SHARD_GRAD_OP",
    "NO_SHARD": "NO_SHARD",
    "HYBRID_SHARD": "HYBRID_SHARD",
}


# ---------------------------------------------------------------------------
# FSDPWrapper
# ---------------------------------------------------------------------------


class FSDPWrapper:
    """Encapsulates FSDP wrapping, mixed precision, and checkpoint I/O.

    All methods assume that `dist.init_process_group()` has already been
    called and that the current device has been set via
    `torch.cuda.set_device(local_rank)`.
    """

    def __init__(self) -> None:
        self._wrapped: bool = False
        self._wrap_module_classes: List[str] = []

    # ------------------------------------------------------------------
    # Wrap
    # ------------------------------------------------------------------

    def wrap(self, model: nn.Module, cfg: Any) -> nn.Module:
        """Apply FSDP wrapping to model according to cfg.

        Parameters
        ----------
        model:
            Unwrapped nn.Module. Must already be on the correct device.
        cfg:
            FSDPConfig instance.

        Returns
        -------
        nn.Module
            FSDP-wrapped model.
        """
        (
            FSDP, ShardingStrategy, MixedPrecision, CPUOffload,
            StateDictType, FullStateDictConfig,
            ModuleWrapPolicy, size_based_auto_wrap_policy,
            apply_activation_checkpointing_fn,
        ) = _import_fsdp()

        self._wrap_module_classes = list(cfg.wrap_module_classes)

        # Build auto_wrap_policy
        auto_wrap_policy = self._build_wrap_policy(
            cfg, ModuleWrapPolicy, size_based_auto_wrap_policy
        )

        # Build mixed precision policy
        mixed_precision = self._build_mixed_precision(cfg.mixed_precision, MixedPrecision)

        # Build sharding strategy
        sharding_strategy = self._resolve_sharding_strategy(
            cfg.sharding_strategy, ShardingStrategy
        )

        # CPU offload
        cpu_offload = CPUOffload(offload_params=True) if cfg.cpu_offload else None

        # Wrap
        wrapped = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            sharding_strategy=sharding_strategy,
            cpu_offload=cpu_offload,
            sync_module_states=cfg.sync_module_states,
            device_id=torch.cuda.current_device(),
        )

        # Log shard boundaries on rank 0
        if dist.is_initialized() and dist.get_rank() == 0:
            logger.info("FSDP-wrapped model structure:\n%s", wrapped)

        # Apply activation checkpointing AFTER FSDP wrap
        if cfg.activation_checkpointing != "off":
            self.apply_activation_checkpointing(wrapped, cfg)

        self._wrapped = True
        return wrapped

    def _build_wrap_policy(
        self,
        cfg: Any,
        ModuleWrapPolicy: type,
        size_based_auto_wrap_policy: Any,
    ) -> Any:
        """Return the appropriate auto_wrap_policy callable."""
        if cfg.wrap_policy == "transformer_block":
            if not cfg.wrap_module_classes:
                raise ValueError(
                    "wrap_module_classes must be non-empty for 'transformer_block' policy."
                )
            module_classes = self._resolve_module_classes(cfg.wrap_module_classes)
            return ModuleWrapPolicy(module_classes)
        elif cfg.wrap_policy == "size_based":
            return partial(
                size_based_auto_wrap_policy,
                min_num_params=cfg.min_num_params,
            )
        else:
            raise ValueError(
                f"Unknown wrap_policy '{cfg.wrap_policy}'. "
                "Must be 'transformer_block' or 'size_based'."
            )

    def _resolve_module_classes(self, class_names: List[str]) -> Set[type]:
        """Resolve class name strings to actual class objects.

        Supports both simple names (resolved against caller's globals at
        import time) and fully-qualified dotted names.
        """
        import importlib

        classes: Set[type] = set()
        for name in class_names:
            if "." in name:
                # Fully qualified: e.g. "transformers.models.gpt2.GPT2Block"
                module_path, cls_name = name.rsplit(".", 1)
                try:
                    module = importlib.import_module(module_path)
                    cls = getattr(module, cls_name)
                except (ImportError, AttributeError) as e:
                    raise ValueError(
                        f"Cannot resolve class '{name}': {e}"
                    ) from e
            else:
                # Simple name: search through all loaded modules
                cls = self._find_class_by_name(name)
                if cls is None:
                    raise ValueError(
                        f"Cannot find class '{name}' in any loaded module. "
                        "Use a fully-qualified dotted name or ensure the class "
                        "is imported before calling wrap()."
                    )
            classes.add(cls)
        return classes

    @staticmethod
    def _find_class_by_name(name: str) -> Optional[type]:
        """Search sys.modules for a class with the given simple name."""
        import sys
        for module in sys.modules.values():
            cls = getattr(module, name, None)
            if cls is not None and isinstance(cls, type):
                return cls
        return None

    @staticmethod
    def _build_mixed_precision(
        precision: str,
        MixedPrecision: type,
    ) -> Optional[Any]:
        """Return a MixedPrecision instance or None."""
        if precision == "bf16":
            return MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        elif precision == "fp16":
            return MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
        elif precision == "none":
            return None
        else:
            raise ValueError(
                f"Unknown mixed_precision '{precision}'. "
                "Must be 'bf16', 'fp16', or 'none'."
            )

    @staticmethod
    def _resolve_sharding_strategy(
        strategy_str: str,
        ShardingStrategy: type,
    ) -> Any:
        """Map string to ShardingStrategy enum value."""
        mapping = {
            "FULL_SHARD": ShardingStrategy.FULL_SHARD,
            "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
            "NO_SHARD": ShardingStrategy.NO_SHARD,
            "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
        }
        if strategy_str not in mapping:
            raise ValueError(
                f"Unknown sharding_strategy '{strategy_str}'. "
                f"Must be one of: {sorted(mapping.keys())}"
            )
        return mapping[strategy_str]

    # ------------------------------------------------------------------
    # Activation Checkpointing
    # ------------------------------------------------------------------

    def apply_activation_checkpointing(
        self,
        model: nn.Module,
        cfg: Any,
    ) -> None:
        """Apply activation checkpointing to FSDP-wrapped transformer blocks.

        Must be called AFTER FSDP.wrap(). Applies to all submodules whose
        class name matches cfg.wrap_module_classes.

        Parameters
        ----------
        model:
            Already FSDP-wrapped model.
        cfg:
            FSDPConfig instance.
        """
        (
            FSDP, _ShardingStrategy, _MixedPrecision, _CPUOffload,
            _StateDictType, _FullStateDictConfig,
            _ModuleWrapPolicy, _size_based_auto_wrap_policy,
            apply_activation_checkpointing_fn,
        ) = _import_fsdp()

        if cfg.activation_checkpointing == "off":
            return

        if not cfg.wrap_module_classes:
            logger.warning(
                "activation_checkpointing='%s' but wrap_module_classes is empty. "
                "No checkpointing applied.",
                cfg.activation_checkpointing,
            )
            return

        module_classes = self._resolve_module_classes(cfg.wrap_module_classes)

        def check_fn(module: nn.Module) -> bool:
            return type(module) in module_classes

        apply_activation_checkpointing_fn(model, check_fn=check_fn)
        logger.info(
            "Activation checkpointing applied to modules of type: %s",
            [c.__name__ for c in module_classes],
        )

    # ------------------------------------------------------------------
    # Save Full State Dict (Portable)
    # ------------------------------------------------------------------

    def save_full_state_dict(self, model: nn.Module, path: str) -> None:
        """Save a portable FULL_STATE_DICT checkpoint.

        Uses offload_to_cpu=True and rank0_only=True. Only rank 0 writes
        to disk. All ranks must call this method (collective operation).

        Parameters
        ----------
        model:
            FSDP-wrapped model.
        path:
            File path to write on rank 0 (e.g. '/checkpoints/weights.pt').
        """
        (
            FSDP, _ShardingStrategy, _MixedPrecision, _CPUOffload,
            StateDictType, FullStateDictConfig,
            _ModuleWrapPolicy, _size_based_auto_wrap_policy,
            _apply_activation_checkpointing_fn,
        ) = _import_fsdp()

        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            state_dict = model.state_dict()

        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0

        if rank == 0:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            torch.save(state_dict, path)
            logger.info("Saved FSDP full state dict to %s", path)

        if dist.is_initialized():
            dist.barrier()

    # ------------------------------------------------------------------
    # Save Sharded Checkpoint (Efficient Resume)
    # ------------------------------------------------------------------

    def save_sharded_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        path: str,
        step: int = 0,
    ) -> None:
        """Save a sharded checkpoint via torch.distributed.checkpoint.

        All ranks write their shards simultaneously to the given directory.

        Parameters
        ----------
        model:
            FSDP-wrapped model.
        optimizer:
            Optimizer holding optimizer states.
        path:
            Directory path for the sharded checkpoint.
        step:
            Training step to embed in the checkpoint metadata.
        """
        (
            FSDP, _ShardingStrategy, _MixedPrecision, _CPUOffload,
            StateDictType, _FullStateDictConfig,
            _ModuleWrapPolicy, _size_based_auto_wrap_policy,
            _apply_activation_checkpointing_fn,
        ) = _import_fsdp()
        dist_cp, FileSystemWriter, _FileSystemReader = _import_dcp()

        os.makedirs(path, exist_ok=True)

        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            state_dict: Dict[str, Any] = {
                "model": model.state_dict(),
                "optimizer": FSDP.optim_state_dict(model, optimizer),
                "step": step,
            }

        dist_cp.save_state_dict(
            state_dict=state_dict,
            storage_writer=FileSystemWriter(path),
        )

        if dist.is_initialized():
            dist.barrier()

        logger.info("Saved FSDP sharded checkpoint to %s (step=%d)", path, step)

    # ------------------------------------------------------------------
    # Load Full State Dict
    # ------------------------------------------------------------------

    def load_full_state_dict(self, model: nn.Module, path: str) -> None:
        """Load a portable FULL_STATE_DICT checkpoint into an FSDP model.

        Rank 0 reads the file; FSDP distributes shards to all ranks.
        All ranks must call this method (collective operation).

        Parameters
        ----------
        model:
            FSDP-wrapped model (must be wrapped before calling this).
        path:
            Path to the checkpoint file saved by save_full_state_dict().
        """
        (
            FSDP, _ShardingStrategy, _MixedPrecision, _CPUOffload,
            StateDictType, FullStateDictConfig,
            _ModuleWrapPolicy, _size_based_auto_wrap_policy,
            _apply_activation_checkpointing_fn,
        ) = _import_fsdp()

        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0

        if rank == 0:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Checkpoint file not found: {path}"
                )
            full_state = torch.load(path, map_location="cpu")
            logger.info("Rank 0 loaded full state dict from %s", path)
        else:
            full_state = {}

        load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, load_policy):
            model.load_state_dict(full_state)

        if dist.is_initialized():
            dist.barrier()

    # ------------------------------------------------------------------
    # Load Sharded Checkpoint
    # ------------------------------------------------------------------

    def load_sharded_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        path: str,
    ) -> int:
        """Load a sharded DCP checkpoint into an FSDP model.

        Returns the training step stored in the checkpoint.

        Parameters
        ----------
        model:
            FSDP-wrapped model.
        optimizer:
            Optimizer to restore states into.
        path:
            Directory path of the sharded checkpoint.

        Returns
        -------
        int
            Step number stored in the checkpoint.
        """
        (
            FSDP, _ShardingStrategy, _MixedPrecision, _CPUOffload,
            StateDictType, _FullStateDictConfig,
            _ModuleWrapPolicy, _size_based_auto_wrap_policy,
            _apply_activation_checkpointing_fn,
        ) = _import_fsdp()
        dist_cp, _FileSystemWriter, FileSystemReader = _import_dcp()

        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            state_dict: Dict[str, Any] = {
                "model": model.state_dict(),
                "optimizer": FSDP.optim_state_dict(model, optimizer),
                "step": 0,
            }

            dist_cp.load_state_dict(
                state_dict=state_dict,
                storage_reader=FileSystemReader(path),
            )

            model.load_state_dict(state_dict["model"])
            FSDP.optim_state_dict_to_load(
                model, optimizer, state_dict["optimizer"]
            )

        step: int = state_dict.get("step", 0)
        logger.info(
            "Loaded FSDP sharded checkpoint from %s (step=%d)", path, step
        )
        return step

    # ------------------------------------------------------------------
    # Build Mixed Precision (public helper for external use)
    # ------------------------------------------------------------------

    @staticmethod
    def build_mixed_precision_policy(precision: str) -> Optional[Any]:
        """Build a MixedPrecision policy dataclass from a precision string.

        Parameters
        ----------
        precision:
            'bf16', 'fp16', or 'none'.

        Returns
        -------
        MixedPrecision or None
        """
        (
            _FSDP, _ShardingStrategy, MixedPrecision, _CPUOffload,
            _StateDictType, _FullStateDictConfig,
            _ModuleWrapPolicy, _size_based_auto_wrap_policy,
            _apply_activation_checkpointing_fn,
        ) = _import_fsdp()
        return FSDPWrapper._build_mixed_precision(precision, MixedPrecision)


# ---------------------------------------------------------------------------
# Self-Tests
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    # These tests do not require a real process group — they test the
    # configuration building and class resolution logic only.

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
            print(f"  FAIL  {name} (wrong exception {type(e).__name__}: {e})")
            failures.append(name)

    # ------------------------------------------------------------------
    # Test: Mixed precision config construction
    # ------------------------------------------------------------------
    print("=== Mixed Precision Config tests ===")

    try:
        (
            _FSDP, _ShardingStrategy, MixedPrecision, _CPUOffload,
            _StateDictType, _FullStateDictConfig,
            _ModuleWrapPolicy, _size_based_auto_wrap_policy,
            _apply_activation_checkpointing_fn,
        ) = _import_fsdp()
        fsdp_available = True
    except ImportError:
        fsdp_available = False
        print("  SKIP  FSDP not available (torch version too old)")

    if fsdp_available:
        bf16_policy = FSDPWrapper._build_mixed_precision("bf16", MixedPrecision)
        check(
            "bf16_param_dtype",
            bf16_policy is not None and bf16_policy.param_dtype == torch.bfloat16,
        )
        check(
            "bf16_reduce_dtype",
            bf16_policy is not None and bf16_policy.reduce_dtype == torch.bfloat16,
        )
        check(
            "bf16_buffer_dtype",
            bf16_policy is not None and bf16_policy.buffer_dtype == torch.bfloat16,
        )

        fp16_policy = FSDPWrapper._build_mixed_precision("fp16", MixedPrecision)
        check(
            "fp16_param_dtype",
            fp16_policy is not None and fp16_policy.param_dtype == torch.float16,
        )

        none_policy = FSDPWrapper._build_mixed_precision("none", MixedPrecision)
        check("none_precision_is_none", none_policy is None)

        expect_raises(
            "invalid_precision_raises",
            ValueError,
            lambda: FSDPWrapper._build_mixed_precision("fp8", MixedPrecision),
        )

    # ------------------------------------------------------------------
    # Test: Sharding strategy resolution
    # ------------------------------------------------------------------
    print("\n=== Sharding Strategy Resolution tests ===")

    if fsdp_available:
        from torch.distributed.fsdp import ShardingStrategy

        for ss_name in ["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD"]:
            resolved = FSDPWrapper._resolve_sharding_strategy(ss_name, ShardingStrategy)
            check(f"resolve_{ss_name}", resolved is not None)

        expect_raises(
            "invalid_sharding_strategy_raises",
            ValueError,
            lambda: FSDPWrapper._resolve_sharding_strategy("INVALID", ShardingStrategy),
        )

    # ------------------------------------------------------------------
    # Test: Class name resolution
    # ------------------------------------------------------------------
    print("\n=== Class Name Resolution tests ===")

    wrapper = FSDPWrapper()

    # Resolve a known class by simple name
    classes = wrapper._resolve_module_classes(["Linear"])
    check("resolve_Linear_class", nn.Linear in classes)

    # Resolve by fully-qualified name
    classes_fq = wrapper._resolve_module_classes(["torch.nn.Linear"])
    check("resolve_fq_Linear_class", nn.Linear in classes_fq)

    # Non-existent simple name raises
    expect_raises(
        "resolve_nonexistent_simple_name",
        ValueError,
        lambda: wrapper._resolve_module_classes(["NonExistentClass123XYZ"]),
    )

    # Non-existent dotted name raises
    expect_raises(
        "resolve_nonexistent_dotted_name",
        ValueError,
        lambda: wrapper._resolve_module_classes(["fake.module.FakeClass"]),
    )

    # ------------------------------------------------------------------
    # Test: Build wrap policy selection
    # ------------------------------------------------------------------
    print("\n=== Wrap Policy Selection tests ===")

    if fsdp_available:
        from torch.distributed.fsdp.wrap import (
            ModuleWrapPolicy,
            size_based_auto_wrap_policy,
        )

        @dataclass
        class MockFSDPConfig:
            wrap_policy: str = "transformer_block"
            wrap_module_classes: List[str] = None
            min_num_params: int = 100_000_000

            def __post_init__(self):
                if self.wrap_module_classes is None:
                    self.wrap_module_classes = []

        wrapper2 = FSDPWrapper()

        # transformer_block policy with valid class
        cfg_tb = MockFSDPConfig(
            wrap_policy="transformer_block",
            wrap_module_classes=["Linear"],
        )
        policy_tb = wrapper2._build_wrap_policy(
            cfg_tb, ModuleWrapPolicy, size_based_auto_wrap_policy
        )
        check(
            "transformer_block_policy_is_ModuleWrapPolicy",
            isinstance(policy_tb, ModuleWrapPolicy),
        )

        # size_based policy
        cfg_sb = MockFSDPConfig(wrap_policy="size_based", wrap_module_classes=[])
        policy_sb = wrapper2._build_wrap_policy(
            cfg_sb, ModuleWrapPolicy, size_based_auto_wrap_policy
        )
        check(
            "size_based_policy_is_callable",
            callable(policy_sb),
        )

        # transformer_block without classes raises
        cfg_no_cls = MockFSDPConfig(
            wrap_policy="transformer_block", wrap_module_classes=[]
        )
        expect_raises(
            "transformer_block_no_classes_raises",
            ValueError,
            lambda: wrapper2._build_wrap_policy(
                cfg_no_cls, ModuleWrapPolicy, size_based_auto_wrap_policy
            ),
        )

        # Invalid policy name raises
        cfg_bad = MockFSDPConfig(wrap_policy="unknown_policy", wrap_module_classes=[])
        expect_raises(
            "invalid_wrap_policy_raises",
            ValueError,
            lambda: wrapper2._build_wrap_policy(
                cfg_bad, ModuleWrapPolicy, size_based_auto_wrap_policy
            ),
        )

    # ------------------------------------------------------------------
    # Test: FullStateDictConfig construction
    # ------------------------------------------------------------------
    print("\n=== FullStateDictConfig Construction tests ===")

    if fsdp_available:
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        check("full_state_dict_config_offload_to_cpu", policy.offload_to_cpu is True)
        check("full_state_dict_config_rank0_only", policy.rank0_only is True)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*50}")
    if failures:
        print(f"FAIL: {len(failures)} test(s) failed: {failures}")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")
        sys.exit(0)
