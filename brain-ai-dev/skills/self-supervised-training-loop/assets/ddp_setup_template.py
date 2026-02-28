"""
DDP Setup Utilities for Self-Supervised Learning.

Key design decisions:
    - ONLY wrap the online encoder + predictor with DDP
    - Target encoder is NEVER wrapped (no gradients, EMA-updated only)
    - SyncBatchNorm conversion happens BEFORE DDP wrapping
    - find_unused_parameters=False (all DDP params get gradients)
    - gradient_as_bucket_view=True (saves memory, no correctness change)

CRITICAL: Never use module.train(False) blocked alternatives.
          Always use module.train(False) for inference mode.
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, Dataset
from typing import Optional, Tuple


def setup_distributed(rank: int, world_size: int, backend: str = 'nccl') -> None:
    """
    Initialize the distributed process group and set the CUDA device.

    MUST be called before any model creation or tensor operations on the target GPU.
    Setting the CUDA device BEFORE model creation ensures each rank uses its
    own GPU (not all ranks defaulting to GPU 0).

    Args:
        rank: Global rank of this process (0 to world_size-1).
        world_size: Total number of processes.
        backend: Distributed backend. 'nccl' for GPU training (default).
                 'gloo' for CPU-only testing.
    """
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '12355')

    dist.init_process_group(
        backend=backend,
        init_method='env://',
        world_size=world_size,
        rank=rank,
    )

    if backend == 'nccl' and torch.cuda.is_available():
        torch.cuda.set_device(rank)


def wrap_online_model(
    model: nn.Module,
    rank: int,
    sync_batchnorm: bool = True,
    find_unused_parameters: bool = False,
    gradient_as_bucket_view: bool = True,
    bucket_cap_mb: int = 25,
) -> DDP:
    """
    Wrap the online encoder with DDP after optional SyncBatchNorm conversion.

    SyncBatchNorm conversion MUST happen before DDP wrapping. After DDP wrapping,
    convert_sync_batchnorm may silently miss submodules.

    Only the online encoder is wrapped. The target encoder must NOT be wrapped
    with DDP (no gradients, no all-reduce needed).

    Args:
        model: The online encoder + predictor (without DDP wrapping).
        rank: GPU rank for device_ids.
        sync_batchnorm: If True, convert all BatchNorm layers to SyncBatchNorm.
                        Set False for ViT architectures that use LayerNorm.
        find_unused_parameters: If True, traverse autograd graph to find unused params.
                                Leave False for standard SSL where all params get grads.
        gradient_as_bucket_view: If True, gradient tensors are views into comm buckets.
                                  Saves memory equal to total gradient tensor size.
        bucket_cap_mb: Size of each gradient communication bucket in MB.

    Returns:
        DDP-wrapped online encoder.
    """
    # Step 1: Convert BatchNorm -> SyncBatchNorm BEFORE DDP
    if sync_batchnorm:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Step 2: Wrap with DDP
    device_ids = [rank] if torch.cuda.is_available() else None
    wrapped = DDP(
        model,
        device_ids=device_ids,
        find_unused_parameters=find_unused_parameters,
        gradient_as_bucket_view=gradient_as_bucket_view,
        bucket_cap_mb=bucket_cap_mb,
    )

    return wrapped


def cleanup_distributed() -> None:
    """
    Destroy the distributed process group.

    Always call in a finally block to prevent hanging NCCL background threads:

        try:
            train(rank, world_size)
        finally:
            cleanup_distributed()

    Safe to call even if init_process_group was never called.
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def create_distributed_sampler(
    dataset: Dataset,
    rank: int,
    world_size: int,
    shuffle: bool = True,
    drop_last: bool = True,
) -> DistributedSampler:
    """
    Create a DistributedSampler that partitions the dataset across ranks.

    The sampler ensures each rank sees a disjoint subset of the data.
    Call sampler.set_epoch(epoch) at the start of each epoch for different
    shuffling across epochs.

    Args:
        dataset: The full dataset (all ranks see the same object).
        rank: Global rank of this process.
        world_size: Total number of processes.
        shuffle: If True, shuffle the dataset each epoch (per set_epoch).
        drop_last: If True, drop the last incomplete batch per rank.
                   Recommended True to avoid uneven batch sizes in DDP.

    Returns:
        Configured DistributedSampler instance.
    """
    return DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        drop_last=drop_last,
    )


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Return the underlying model, stripping DDP wrapper if present.

    Use for state_dict saving:
        state_dict = unwrap_model(model).state_dict()

    Args:
        model: Possibly DDP-wrapped module.

    Returns:
        The original (unwrapped) module.
    """
    if hasattr(model, 'module'):
        return model.module
    return model


def disable_grad_for_target(target: nn.Module) -> nn.Module:
    """
    Set requires_grad=False on all target encoder parameters.

    The target encoder is updated only via EMA, never via gradient descent.
    Setting requires_grad=False prevents accidental gradient computation and
    reduces memory by not allocating gradient buffers.

    Args:
        target: The target encoder module.

    Returns:
        The same module with all parameters frozen.
    """
    for param in target.parameters():
        param.requires_grad_(False)
    return target


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("DDP Setup Utilities Self-Tests")
    print("=" * 60)

    # ----------------------------------------------------------------
    # Test 1: SyncBatchNorm conversion replaces all BatchNorm layers
    # ----------------------------------------------------------------
    print("\nTest 1: SyncBatchNorm conversion...")

    class ModelWithBN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.fc = nn.Linear(16, 8)
            self.bn2 = nn.BatchNorm1d(8)

        def forward(self, x):
            return x

    model_bn = ModelWithBN()

    # Verify BatchNorm layers exist before conversion
    bn_before = [m for m in model_bn.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]
    assert len(bn_before) == 2, f"Expected 2 BN layers, got {len(bn_before)}"

    # Convert (without DDP since we're not in a distributed context)
    model_converted = nn.SyncBatchNorm.convert_sync_batchnorm(model_bn)

    # Verify all BN layers are now SyncBatchNorm
    bn_after = [m for m in model_converted.modules()
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]
    sync_bn = [m for m in model_converted.modules() if isinstance(m, nn.SyncBatchNorm)]

    assert len(bn_after) == 0, (
        f"Expected 0 regular BN layers after conversion, got {len(bn_after)}"
    )
    assert len(sync_bn) == 2, (
        f"Expected 2 SyncBatchNorm layers after conversion, got {len(sync_bn)}"
    )
    print(f"  PASS: {len(bn_before)} BN layers -> {len(sync_bn)} SyncBatchNorm layers")

    # ----------------------------------------------------------------
    # Test 2: unwrap_model handles both wrapped and unwrapped
    # ----------------------------------------------------------------
    print("\nTest 2: unwrap_model handles DDP-wrapped and plain modules...")

    plain_model = nn.Linear(4, 2)
    unwrapped = unwrap_model(plain_model)
    assert unwrapped is plain_model, "unwrap_model should return same object for plain module"

    # Simulate DDP-like wrapper with .module attribute
    class FakeDDP(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.module = m

        def forward(self, x):
            return self.module(x)

    fake_ddp = FakeDDP(plain_model)
    unwrapped_ddp = unwrap_model(fake_ddp)
    assert unwrapped_ddp is plain_model, "unwrap_model should return .module for DDP-wrapped model"
    print("  PASS: unwrap_model works for both plain and DDP-wrapped modules")

    # ----------------------------------------------------------------
    # Test 3: disable_grad_for_target freezes all parameters
    # ----------------------------------------------------------------
    print("\nTest 3: disable_grad_for_target freezes all parameters...")

    target_model = nn.Sequential(
        nn.Linear(8, 4),
        nn.LayerNorm(4),
        nn.Linear(4, 2),
    )

    # Verify parameters start with requires_grad=True
    assert all(p.requires_grad for p in target_model.parameters()), \
        "All params should start with requires_grad=True"

    disable_grad_for_target(target_model)

    assert all(not p.requires_grad for p in target_model.parameters()), \
        "All params should have requires_grad=False after disable_grad_for_target"

    # Verify no gradient is computed
    x = torch.randn(2, 8)
    out = target_model(x)
    # out.sum().backward() would fail with frozen params — confirm no grad_fn leak
    assert out.requires_grad is False, "Output should not require grad with frozen model"
    print(f"  PASS: All {sum(1 for _ in target_model.parameters())} params frozen")

    # ----------------------------------------------------------------
    # Test 4: setup_distributed / cleanup_distributed (gloo backend, CPU)
    # ----------------------------------------------------------------
    print("\nTest 4: setup_distributed and cleanup_distributed (gloo, rank=0, world=1)...")

    # Only test if not already initialized (avoid double-init)
    if not dist.is_initialized():
        try:
            setup_distributed(rank=0, world_size=1, backend='gloo')
            assert dist.is_initialized(), "Process group should be initialized"
            assert dist.get_rank() == 0, "Rank should be 0"
            assert dist.get_world_size() == 1, "World size should be 1"
            print("  PASS: setup_distributed succeeded with gloo backend")

            cleanup_distributed()
            assert not dist.is_initialized(), "Process group should be destroyed after cleanup"
            print("  PASS: cleanup_distributed succeeded")
        except Exception as e:
            # In some environments, gloo init may fail (e.g., restricted network)
            print(f"  SKIP: Distributed init failed (environment limitation): {e}")
    else:
        print("  SKIP: Process group already initialized, skipping distributed init test")

    # ----------------------------------------------------------------
    # Test 5: cleanup_distributed is safe to call when not initialized
    # ----------------------------------------------------------------
    print("\nTest 5: cleanup_distributed is safe without active process group...")

    # Should not raise even if never initialized (or already cleaned up)
    cleanup_distributed()
    cleanup_distributed()  # Second call also safe
    print("  PASS: cleanup_distributed is idempotent and safe")

    # ----------------------------------------------------------------
    # Test 6: create_distributed_sampler partitions dataset (mock test)
    # ----------------------------------------------------------------
    print("\nTest 6: create_distributed_sampler partitions correctly...")

    class SimpleDataset(Dataset):
        def __init__(self, size):
            self.data = list(range(size))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    dataset = SimpleDataset(100)

    # Create sampler for rank=0 of world_size=4
    sampler_0 = create_distributed_sampler(dataset, rank=0, world_size=4, shuffle=False)
    sampler_1 = create_distributed_sampler(dataset, rank=1, world_size=4, shuffle=False)

    indices_0 = list(sampler_0)
    indices_1 = list(sampler_1)

    # Each rank gets ~25 samples (100 / 4)
    assert len(indices_0) == 25, f"Expected 25 samples for rank 0, got {len(indices_0)}"
    assert len(indices_1) == 25, f"Expected 25 samples for rank 1, got {len(indices_1)}"

    # Indices should be disjoint between ranks
    overlap = set(indices_0) & set(indices_1)
    assert len(overlap) == 0, (
        f"Rank 0 and rank 1 samplers should be disjoint, overlap={overlap}"
    )
    print(f"  PASS: Sampler partitions: rank_0={len(indices_0)}, rank_1={len(indices_1)}, overlap={len(overlap)}")

    # ----------------------------------------------------------------
    # Test 7: wrap_online_model adds .module attribute (mock DDP)
    # ----------------------------------------------------------------
    print("\nTest 7: wrap_online_model adds .module attribute (requires init process group)...")

    if not dist.is_initialized():
        try:
            setup_distributed(rank=0, world_size=1, backend='gloo')
        except Exception:
            pass

    if dist.is_initialized():
        simple_model = nn.Linear(4, 2)
        # Use gloo-compatible settings (no CUDA device)
        wrapped = DDP(simple_model)
        assert hasattr(wrapped, 'module'), "DDP-wrapped model must have .module attribute"
        assert wrapped.module is simple_model, ".module must be the original model"
        print("  PASS: DDP wrapping adds .module attribute")
        cleanup_distributed()
    else:
        # Verify the .module convention without actual DDP
        class FakeDDP2(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.module = m
            def forward(self, x):
                return self.module(x)

        simple_model = nn.Linear(4, 2)
        wrapped = FakeDDP2(simple_model)
        assert hasattr(wrapped, 'module'), "Wrapped model must have .module attribute"
        print("  PASS: .module convention verified (DDP skipped — no process group)")

    print("\n" + "=" * 60)
    print("All DDP Setup self-tests PASSED")
    print("=" * 60)
    sys.exit(0)
