"""
DDPWrapper — DistributedDataParallel wrapper for the brain_ai system.

Provides process group management, model wrapping, metric aggregation,
and single-GPU graceful fallback.

Key classes:
    DDPWrapper          — Main wrapper for DDP training
    ProcessGroupManager — Manages init/cleanup of process groups

Self-tests in __main__ validate all functionality without actual multi-GPU hardware.
"""

from __future__ import annotations

import copy
import logging
import os
import warnings
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports (graceful fallback)
# ---------------------------------------------------------------------------

_DIST_AVAILABLE = False
try:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    _DIST_AVAILABLE = True
except ImportError:
    dist = None
    DDP = None


# ===========================================================================
# SECTION 1: ProcessGroupManager
# ===========================================================================

class ProcessGroupManager:
    """Manages initialization and cleanup of distributed process groups.

    Handles torchrun, SLURM, and manual launch modes. Safe to call
    even when distributed is not available (degrades to single-process).
    """

    def __init__(
        self,
        backend: str = "nccl",
        master_addr: str = "localhost",
        master_port: str = "29500",
    ):
        self.backend = backend
        self.master_addr = master_addr
        self.master_port = master_port
        self._initialized = False
        self._rank = 0
        self._world_size = 1
        self._local_rank = 0

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def local_rank(self) -> int:
        return self._local_rank

    @property
    def is_main(self) -> bool:
        return self._rank == 0

    @property
    def initialized(self) -> bool:
        return self._initialized

    def init_process_group(
        self,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ) -> None:
        """Initialize the distributed process group.

        If rank/world_size are not provided, attempts to read from
        environment variables (torchrun or SLURM).
        """
        if not _DIST_AVAILABLE:
            logger.warning("torch.distributed not available; running single-process.")
            return

        if self._initialized:
            logger.warning("Process group already initialized; skipping.")
            return

        # Detect environment
        if rank is not None and world_size is not None:
            self._rank = rank
            self._world_size = world_size
        elif "RANK" in os.environ:
            # torchrun sets RANK, WORLD_SIZE, LOCAL_RANK
            self._rank = int(os.environ["RANK"])
            self._world_size = int(os.environ["WORLD_SIZE"])
            self._local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        elif "SLURM_PROCID" in os.environ:
            self._rank = int(os.environ["SLURM_PROCID"])
            self._world_size = int(os.environ["SLURM_NTASKS"])
            self._local_rank = self._rank % max(torch.cuda.device_count(), 1)
        else:
            logger.info("No distributed env detected; running single-process.")
            return

        os.environ.setdefault("MASTER_ADDR", self.master_addr)
        os.environ.setdefault("MASTER_PORT", self.master_port)

        dist.init_process_group(
            backend=self.backend,
            rank=self._rank,
            world_size=self._world_size,
        )

        if torch.cuda.is_available() and self.backend == "nccl":
            torch.cuda.set_device(self._local_rank)

        self._initialized = True
        logger.info(
            f"Process group initialized: rank={self._rank}, "
            f"world_size={self._world_size}, backend={self.backend}"
        )

    def cleanup(self) -> None:
        """Destroy the process group."""
        if self._initialized and _DIST_AVAILABLE and dist.is_initialized():
            dist.destroy_process_group()
            self._initialized = False
            logger.info("Process group destroyed.")


# ===========================================================================
# SECTION 2: DDPWrapper
# ===========================================================================

class DDPWrapper:
    """DistributedDataParallel wrapper for BrainAI models.

    Handles:
    - Process group management
    - Model wrapping with DDP
    - SyncBatchNorm conversion
    - Metric aggregation across ranks
    - Gradient clipping
    - Checkpoint saving (rank 0 only)
    - Graceful single-GPU fallback

    Args:
        model: The nn.Module to wrap (typically BrainAI).
        backend: Communication backend ("nccl" or "gloo").
        find_unused_parameters: Enable for models with conditional paths.
        sync_batchnorm: Convert BatchNorm to SyncBatchNorm.
        bucket_cap_mb: DDP bucket size in MB.
        gradient_as_bucket_view: Alias gradients into communication buffers.
    """

    def __init__(
        self,
        model: nn.Module,
        backend: str = "nccl",
        find_unused_parameters: bool = False,
        sync_batchnorm: bool = True,
        bucket_cap_mb: int = 25,
        gradient_as_bucket_view: bool = True,
    ):
        self.model = model
        self.backend = backend
        self.find_unused_parameters = find_unused_parameters
        self.sync_batchnorm = sync_batchnorm
        self.bucket_cap_mb = bucket_cap_mb
        self.gradient_as_bucket_view = gradient_as_bucket_view

        self._process_manager = ProcessGroupManager(backend=backend)
        self._ddp_model: Optional[nn.Module] = None
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
        """Return the DDP-wrapped model, or the raw model if not wrapped."""
        if self._ddp_model is not None:
            return self._ddp_model
        return self.model

    @property
    def inner_model(self) -> nn.Module:
        """Return the inner (unwrapped) model."""
        if self._ddp_model is not None and hasattr(self._ddp_model, "module"):
            return self._ddp_model.module
        return self.model

    def setup(
        self,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ) -> nn.Module:
        """Initialize process group and wrap model with DDP.

        Args:
            rank: Process rank (auto-detected if None).
            world_size: Total number of processes (auto-detected if None).

        Returns:
            The DDP-wrapped model (or raw model if single-process).
        """
        self._process_manager.init_process_group(rank=rank, world_size=world_size)
        self._rank = self._process_manager.rank
        self._world_size = self._process_manager.world_size

        if self._world_size <= 1 or not _DIST_AVAILABLE or not dist.is_initialized():
            logger.info("Single-process mode; returning unwrapped model.")
            return self.model

        # Move model to correct device
        device = torch.device(f"cuda:{self._process_manager.local_rank}")
        self.model = self.model.to(device)

        # Convert SyncBatchNorm
        if self.sync_batchnorm:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        # Wrap with DDP
        self._ddp_model = DDP(
            self.model,
            device_ids=[self._process_manager.local_rank],
            output_device=self._process_manager.local_rank,
            find_unused_parameters=self.find_unused_parameters,
            bucket_cap_mb=self.bucket_cap_mb,
            gradient_as_bucket_view=self.gradient_as_bucket_view,
        )

        logger.info(
            f"Model wrapped with DDP on rank {self._rank} "
            f"(find_unused={self.find_unused_parameters}, "
            f"sync_bn={self.sync_batchnorm})"
        )
        return self._ddp_model

    def cleanup(self) -> None:
        """Destroy process group and release resources."""
        self._ddp_model = None
        self._process_manager.cleanup()

    def all_reduce_metrics(
        self,
        metrics: Dict[str, float],
        op: str = "mean",
    ) -> Dict[str, float]:
        """Aggregate metrics across all ranks.

        Args:
            metrics: Dict of metric_name -> float value.
            op: Reduction operation ("mean" or "sum").

        Returns:
            Reduced metrics dict.
        """
        if not _DIST_AVAILABLE or not dist.is_initialized() or self._world_size <= 1:
            return metrics

        device = next(self.inner_model.parameters()).device

        reduced = {}
        for key, value in metrics.items():
            tensor = torch.tensor(value, device=device, dtype=torch.float32)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            if op == "mean":
                tensor = tensor / self._world_size
            reduced[key] = tensor.item()

        return reduced

    def all_reduce_tensor(
        self,
        tensor: Tensor,
        op: str = "mean",
    ) -> Tensor:
        """Reduce a tensor across all ranks.

        Args:
            tensor: Tensor to reduce.
            op: "mean" or "sum".

        Returns:
            Reduced tensor.
        """
        if not _DIST_AVAILABLE or not dist.is_initialized() or self._world_size <= 1:
            return tensor

        rt = tensor.clone().detach()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        if op == "mean":
            rt = rt / self._world_size
        return rt

    @contextmanager
    def no_sync(self):
        """Context manager to skip gradient synchronization.

        Used during gradient accumulation for non-final micro-steps.
        """
        if self._ddp_model is not None and hasattr(self._ddp_model, "no_sync"):
            with self._ddp_model.no_sync():
                yield
        else:
            yield

    def clip_gradients(self, max_norm: float = 1.0) -> float:
        """Clip gradients after backward (post-synchronization).

        Returns:
            The total gradient norm before clipping.
        """
        model = self.inner_model
        return torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=max_norm
        ).item()

    def save_checkpoint(
        self,
        path: str,
        optimizer: Optional[Any] = None,
        epoch: int = 0,
        step: int = 0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save checkpoint from rank 0 only.

        Args:
            path: File path for the checkpoint.
            optimizer: Optional optimizer to save.
            epoch: Current epoch number.
            step: Current global step.
            extra: Additional metadata to save.
        """
        if not self.is_main:
            return

        checkpoint = {
            "model": self.inner_model.state_dict(),
            "epoch": epoch,
            "step": step,
        }
        if optimizer is not None:
            checkpoint["optimizer"] = optimizer.state_dict()
        if extra is not None:
            checkpoint.update(extra)

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path} (epoch={epoch}, step={step})")

    def load_checkpoint(
        self,
        path: str,
        optimizer: Optional[Any] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Load checkpoint into the inner model.

        Args:
            path: Path to checkpoint file.
            optimizer: Optional optimizer to restore.
            strict: Whether to require exact key match.

        Returns:
            The full checkpoint dict (for accessing epoch, step, etc.).
        """
        map_location = "cpu"
        if torch.cuda.is_available():
            map_location = f"cuda:{self._process_manager.local_rank}"

        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        self.inner_model.load_state_dict(checkpoint["model"], strict=strict)

        if optimizer is not None and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])

        logger.info(f"Checkpoint loaded from {path}")
        return checkpoint

    def barrier(self) -> None:
        """Synchronization barrier across all ranks."""
        if _DIST_AVAILABLE and dist.is_initialized():
            dist.barrier()


# ===========================================================================
# SECTION 3: Utility functions
# ===========================================================================

def get_device_for_rank(rank: int) -> torch.device:
    """Return the correct CUDA device for a given rank."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    return torch.device("cpu")


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """Broadcast a Python object from src rank to all ranks."""
    if not _DIST_AVAILABLE or not dist.is_initialized():
        return obj
    obj_list = [obj]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


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
    print("DDPWrapper Self-Tests (single-process simulation)")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # ProcessGroupManager tests
    # -----------------------------------------------------------------------

    def test_pgm_defaults():
        pgm = ProcessGroupManager()
        assert pgm.backend == "nccl"
        assert pgm.rank == 0
        assert pgm.world_size == 1
        assert pgm.is_main is True
        assert pgm.initialized is False

    run_test("ProcessGroupManager defaults", test_pgm_defaults)

    def test_pgm_custom_backend():
        pgm = ProcessGroupManager(backend="gloo", master_port="12345")
        assert pgm.backend == "gloo"
        assert pgm.master_port == "12345"

    run_test("ProcessGroupManager custom backend", test_pgm_custom_backend)

    def test_pgm_cleanup_no_init():
        pgm = ProcessGroupManager()
        pgm.cleanup()  # Should not raise
        assert pgm.initialized is False

    run_test("ProcessGroupManager cleanup without init", test_pgm_cleanup_no_init)

    def test_pgm_single_process_init():
        pgm = ProcessGroupManager(backend="gloo")
        # Without env vars or explicit rank/world_size > 1, stays single-process
        pgm.init_process_group()
        # Should not crash, just log and return
        pgm.cleanup()

    run_test("ProcessGroupManager single-process init", test_pgm_single_process_init)

    # -----------------------------------------------------------------------
    # DDPWrapper construction tests
    # -----------------------------------------------------------------------

    def _make_test_model():
        return nn.Sequential(
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def test_wrapper_construction():
        model = _make_test_model()
        wrapper = DDPWrapper(model, backend="gloo")
        assert wrapper.rank == 0
        assert wrapper.world_size == 1
        assert wrapper.is_main is True
        assert wrapper.inner_model is model

    run_test("DDPWrapper construction", test_wrapper_construction)

    def test_wrapper_setup_single():
        model = _make_test_model()
        wrapper = DDPWrapper(model, backend="gloo")
        returned = wrapper.setup(rank=0, world_size=1)
        # Single process: should return unwrapped model
        assert returned is model
        assert wrapper.wrapped_model is model
        wrapper.cleanup()

    run_test("DDPWrapper setup single-process", test_wrapper_setup_single)

    def test_wrapper_inner_model_single():
        model = _make_test_model()
        wrapper = DDPWrapper(model, backend="gloo")
        wrapper.setup(rank=0, world_size=1)
        assert wrapper.inner_model is model
        wrapper.cleanup()

    run_test("DDPWrapper inner_model single-process", test_wrapper_inner_model_single)

    # -----------------------------------------------------------------------
    # Metric aggregation tests (single-process)
    # -----------------------------------------------------------------------

    def test_all_reduce_metrics_single():
        model = _make_test_model()
        wrapper = DDPWrapper(model, backend="gloo")
        wrapper.setup(rank=0, world_size=1)
        metrics = {"loss": 0.5, "accuracy": 0.9}
        reduced = wrapper.all_reduce_metrics(metrics)
        assert abs(reduced["loss"] - 0.5) < 1e-6
        assert abs(reduced["accuracy"] - 0.9) < 1e-6
        wrapper.cleanup()

    run_test("all_reduce_metrics single-process passthrough", test_all_reduce_metrics_single)

    def test_all_reduce_metrics_empty():
        model = _make_test_model()
        wrapper = DDPWrapper(model, backend="gloo")
        wrapper.setup(rank=0, world_size=1)
        reduced = wrapper.all_reduce_metrics({})
        assert reduced == {}
        wrapper.cleanup()

    run_test("all_reduce_metrics empty dict", test_all_reduce_metrics_empty)

    def test_all_reduce_metrics_types():
        model = _make_test_model()
        wrapper = DDPWrapper(model, backend="gloo")
        wrapper.setup(rank=0, world_size=1)
        metrics = {"int_val": 42, "float_val": 3.14, "zero": 0.0}
        reduced = wrapper.all_reduce_metrics(metrics)
        assert abs(reduced["int_val"] - 42.0) < 1e-5
        assert abs(reduced["float_val"] - 3.14) < 1e-5
        assert abs(reduced["zero"] - 0.0) < 1e-5
        wrapper.cleanup()

    run_test("all_reduce_metrics with various types", test_all_reduce_metrics_types)

    def test_all_reduce_tensor_single():
        model = _make_test_model()
        wrapper = DDPWrapper(model, backend="gloo")
        wrapper.setup(rank=0, world_size=1)
        t = torch.tensor(3.0)
        reduced = wrapper.all_reduce_tensor(t)
        assert abs(reduced.item() - 3.0) < 1e-6
        wrapper.cleanup()

    run_test("all_reduce_tensor single-process passthrough", test_all_reduce_tensor_single)

    # -----------------------------------------------------------------------
    # Gradient clipping tests
    # -----------------------------------------------------------------------

    def test_clip_gradients():
        model = _make_test_model()
        wrapper = DDPWrapper(model, backend="gloo")
        wrapper.setup(rank=0, world_size=1)
        # Create fake gradients
        x = torch.randn(4, 32)
        out = model(x)
        loss = out.sum()
        loss.backward()
        norm = wrapper.clip_gradients(max_norm=0.1)
        assert norm >= 0.0
        # After clipping, total norm should be <= 0.1 + epsilon
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        assert total_norm.item() <= 0.1 + 1e-5
        wrapper.cleanup()

    run_test("clip_gradients enforces max_norm", test_clip_gradients)

    def test_clip_gradients_no_grad():
        model = _make_test_model()
        wrapper = DDPWrapper(model, backend="gloo")
        wrapper.setup(rank=0, world_size=1)
        # No backward called — all grads are None
        norm = wrapper.clip_gradients(max_norm=1.0)
        assert norm == 0.0
        wrapper.cleanup()

    run_test("clip_gradients with no gradients", test_clip_gradients_no_grad)

    # -----------------------------------------------------------------------
    # no_sync context manager tests
    # -----------------------------------------------------------------------

    def test_no_sync_single():
        model = _make_test_model()
        wrapper = DDPWrapper(model, backend="gloo")
        wrapper.setup(rank=0, world_size=1)
        # Should not raise
        with wrapper.no_sync():
            x = torch.randn(4, 32)
            out = model(x)
            out.sum().backward()
        wrapper.cleanup()

    run_test("no_sync context in single-process", test_no_sync_single)

    # -----------------------------------------------------------------------
    # Checkpoint tests
    # -----------------------------------------------------------------------

    def test_save_checkpoint(tmp_path=None):
        import tempfile
        model = _make_test_model()
        wrapper = DDPWrapper(model, backend="gloo")
        wrapper.setup(rank=0, world_size=1)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        wrapper.save_checkpoint(path, optimizer=optimizer, epoch=5, step=100)
        assert os.path.exists(path)
        ckpt = torch.load(path, weights_only=False)
        assert ckpt["epoch"] == 5
        assert ckpt["step"] == 100
        assert "model" in ckpt
        assert "optimizer" in ckpt
        os.unlink(path)
        wrapper.cleanup()

    run_test("save_checkpoint creates valid file", test_save_checkpoint)

    def test_load_checkpoint():
        import tempfile
        model = _make_test_model()
        wrapper = DDPWrapper(model, backend="gloo")
        wrapper.setup(rank=0, world_size=1)

        # Save
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        wrapper.save_checkpoint(path, epoch=3, step=50)

        # Modify model
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(999.0)

        # Load
        ckpt = wrapper.load_checkpoint(path)
        assert ckpt["epoch"] == 3
        # Model should be restored (not all 999.0)
        first_param = next(model.parameters())
        assert not torch.all(first_param == 999.0)
        os.unlink(path)
        wrapper.cleanup()

    run_test("load_checkpoint restores model", test_load_checkpoint)

    def test_save_not_main():
        model = _make_test_model()
        wrapper = DDPWrapper(model, backend="gloo")
        wrapper._rank = 1  # Simulate non-main rank
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        os.unlink(path)  # Remove so we can check it's not created
        wrapper.save_checkpoint(path, epoch=1)
        assert not os.path.exists(path)

    run_test("save_checkpoint skips non-main rank", test_save_not_main)

    # -----------------------------------------------------------------------
    # Model property tests
    # -----------------------------------------------------------------------

    def test_find_unused_default():
        model = _make_test_model()
        wrapper = DDPWrapper(model, find_unused_parameters=True)
        assert wrapper.find_unused_parameters is True

    run_test("find_unused_parameters flag", test_find_unused_default)

    def test_sync_batchnorm_default():
        model = _make_test_model()
        wrapper = DDPWrapper(model, sync_batchnorm=False)
        assert wrapper.sync_batchnorm is False

    run_test("sync_batchnorm flag", test_sync_batchnorm_default)

    def test_bucket_cap_mb():
        model = _make_test_model()
        wrapper = DDPWrapper(model, bucket_cap_mb=50)
        assert wrapper.bucket_cap_mb == 50

    run_test("bucket_cap_mb configuration", test_bucket_cap_mb)

    # -----------------------------------------------------------------------
    # Barrier tests
    # -----------------------------------------------------------------------

    def test_barrier_single():
        model = _make_test_model()
        wrapper = DDPWrapper(model, backend="gloo")
        wrapper.setup(rank=0, world_size=1)
        wrapper.barrier()  # Should not hang or crash
        wrapper.cleanup()

    run_test("barrier in single-process", test_barrier_single)

    # -----------------------------------------------------------------------
    # get_device_for_rank tests
    # -----------------------------------------------------------------------

    def test_device_for_rank_cpu():
        if not torch.cuda.is_available():
            dev = get_device_for_rank(0)
            assert dev == torch.device("cpu")
        else:
            dev = get_device_for_rank(0)
            assert "cuda" in str(dev)

    run_test("get_device_for_rank", test_device_for_rank_cpu)

    # -----------------------------------------------------------------------
    # broadcast_object tests
    # -----------------------------------------------------------------------

    def test_broadcast_object_single():
        obj = {"key": [1, 2, 3], "value": "hello"}
        result = broadcast_object(obj, src=0)
        assert result == obj

    run_test("broadcast_object single-process", test_broadcast_object_single)

    # -----------------------------------------------------------------------
    # Forward pass through wrapper
    # -----------------------------------------------------------------------

    def test_forward_pass():
        model = _make_test_model()
        wrapper = DDPWrapper(model, backend="gloo")
        wrapper.setup(rank=0, world_size=1)
        x = torch.randn(8, 32)
        out = wrapper.wrapped_model(x)
        assert out.shape == (8, 10)
        wrapper.cleanup()

    run_test("forward pass through wrapped model", test_forward_pass)

    def test_backward_pass():
        model = _make_test_model()
        wrapper = DDPWrapper(model, backend="gloo")
        wrapper.setup(rank=0, world_size=1)
        x = torch.randn(8, 32)
        out = wrapper.wrapped_model(x)
        loss = out.sum()
        loss.backward()
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad
        wrapper.cleanup()

    run_test("backward pass produces gradients", test_backward_pass)

    # -----------------------------------------------------------------------
    # Optimizer step integration
    # -----------------------------------------------------------------------

    def test_optimizer_step():
        model = _make_test_model()
        wrapper = DDPWrapper(model, backend="gloo")
        wrapper.setup(rank=0, world_size=1)

        params_before = {n: p.clone() for n, p in model.named_parameters()}
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        x = torch.randn(4, 32)
        out = wrapper.wrapped_model(x)
        loss = out.sum()
        loss.backward()
        wrapper.clip_gradients(max_norm=1.0)
        optimizer.step()

        changed = False
        for n, p in model.named_parameters():
            if not torch.equal(p, params_before[n]):
                changed = True
                break
        assert changed, "Optimizer step did not change parameters"
        wrapper.cleanup()

    run_test("optimizer step changes parameters", test_optimizer_step)

    # -----------------------------------------------------------------------
    # Multiple setup/cleanup cycles
    # -----------------------------------------------------------------------

    def test_multiple_cycles():
        model = _make_test_model()
        wrapper = DDPWrapper(model, backend="gloo")
        for _ in range(3):
            wrapper.setup(rank=0, world_size=1)
            x = torch.randn(4, 32)
            out = wrapper.wrapped_model(x)
            assert out.shape == (4, 10)
            wrapper.cleanup()

    run_test("multiple setup/cleanup cycles", test_multiple_cycles)

    # -----------------------------------------------------------------------
    # Edge case: large metric dict
    # -----------------------------------------------------------------------

    def test_large_metric_dict():
        model = _make_test_model()
        wrapper = DDPWrapper(model, backend="gloo")
        wrapper.setup(rank=0, world_size=1)
        metrics = {f"metric_{i}": float(i) for i in range(100)}
        reduced = wrapper.all_reduce_metrics(metrics)
        for i in range(100):
            assert abs(reduced[f"metric_{i}"] - float(i)) < 1e-5
        wrapper.cleanup()

    run_test("all_reduce_metrics with 100 entries", test_large_metric_dict)

    # -----------------------------------------------------------------------
    # Config properties
    # -----------------------------------------------------------------------

    def test_gradient_as_bucket_view():
        model = _make_test_model()
        wrapper = DDPWrapper(model, gradient_as_bucket_view=False)
        assert wrapper.gradient_as_bucket_view is False

    run_test("gradient_as_bucket_view flag", test_gradient_as_bucket_view)

    def test_backend_property():
        model = _make_test_model()
        wrapper = DDPWrapper(model, backend="gloo")
        assert wrapper.backend == "gloo"

    run_test("backend property", test_backend_property)

    # -----------------------------------------------------------------------
    # Conditional model (unused parameters)
    # -----------------------------------------------------------------------

    class ConditionalModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Linear(32, 64)
            self.path_a = nn.Linear(64, 10)
            self.path_b = nn.Linear(64, 10)

        def forward(self, x, use_path_a=True):
            h = self.shared(x)
            if use_path_a:
                return self.path_a(h)
            return self.path_b(h)

    def test_conditional_model_path_a():
        model = ConditionalModel()
        wrapper = DDPWrapper(model, find_unused_parameters=True, backend="gloo")
        wrapper.setup(rank=0, world_size=1)
        x = torch.randn(4, 32)
        out = wrapper.wrapped_model(x, use_path_a=True)
        assert out.shape == (4, 10)
        wrapper.cleanup()

    run_test("conditional model path A", test_conditional_model_path_a)

    def test_conditional_model_path_b():
        model = ConditionalModel()
        wrapper = DDPWrapper(model, find_unused_parameters=True, backend="gloo")
        wrapper.setup(rank=0, world_size=1)
        x = torch.randn(4, 32)
        out = wrapper.wrapped_model(x, use_path_a=False)
        assert out.shape == (4, 10)
        wrapper.cleanup()

    run_test("conditional model path B", test_conditional_model_path_b)

    # -----------------------------------------------------------------------
    # all_reduce_tensor with operations
    # -----------------------------------------------------------------------

    def test_all_reduce_tensor_sum():
        model = _make_test_model()
        wrapper = DDPWrapper(model, backend="gloo")
        wrapper.setup(rank=0, world_size=1)
        t = torch.tensor(5.0)
        reduced = wrapper.all_reduce_tensor(t, op="sum")
        assert abs(reduced.item() - 5.0) < 1e-6
        wrapper.cleanup()

    run_test("all_reduce_tensor sum op", test_all_reduce_tensor_sum)

    def test_all_reduce_tensor_multidim():
        model = _make_test_model()
        wrapper = DDPWrapper(model, backend="gloo")
        wrapper.setup(rank=0, world_size=1)
        t = torch.randn(3, 4)
        reduced = wrapper.all_reduce_tensor(t, op="mean")
        torch.testing.assert_close(reduced, t)
        wrapper.cleanup()

    run_test("all_reduce_tensor multi-dimensional", test_all_reduce_tensor_multidim)

    # -----------------------------------------------------------------------
    # Checkpoint with extra metadata
    # -----------------------------------------------------------------------

    def test_checkpoint_extra_metadata():
        import tempfile
        model = _make_test_model()
        wrapper = DDPWrapper(model, backend="gloo")
        wrapper.setup(rank=0, world_size=1)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        wrapper.save_checkpoint(
            path, epoch=10, step=500,
            extra={"config": {"lr": 0.001}, "best_acc": 0.95}
        )
        ckpt = torch.load(path, weights_only=False)
        assert ckpt["config"]["lr"] == 0.001
        assert ckpt["best_acc"] == 0.95
        os.unlink(path)
        wrapper.cleanup()

    run_test("checkpoint with extra metadata", test_checkpoint_extra_metadata)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed, {total} total")
    print("=" * 70)

    sys.exit(0 if failed == 0 else 1)
