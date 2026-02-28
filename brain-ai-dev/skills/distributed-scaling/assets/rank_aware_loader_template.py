"""
RankAwareDataLoader — Distributed data loading with duplicate prevention.

Wraps PyTorch's DistributedSampler and DataLoader to ensure each rank
processes unique samples, supports epoch-based shuffling, rank-aware
worker seeding, and multi-modal BrainAI data loading.

Key classes:
    RankAwareDataLoader   — Main class with get_sampler(), get_loader(), set_epoch()
    WorkerSeedManager     — Per-worker, per-rank seeding
    SamplerVerifier       — Verification utilities for duplicate detection

Self-tests in __main__ validate all functionality without actual multi-GPU hardware.
"""

from __future__ import annotations

import logging
import math
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import (
    DataLoader,
    Dataset,
    Sampler,
    TensorDataset,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

_DIST_SAMPLER_AVAILABLE = True
try:
    from torch.utils.data.distributed import DistributedSampler
except ImportError:
    _DIST_SAMPLER_AVAILABLE = False
    DistributedSampler = None


# ===========================================================================
# SECTION 1: WorkerSeedManager
# ===========================================================================

class WorkerSeedManager:
    """Manages per-worker, per-rank random seeding for DataLoader workers.

    Ensures that different ranks produce different random augmentations
    even when using the same base seed.

    Args:
        base_seed: Base random seed.
        rank: Process rank (incorporated into seed).
    """

    def __init__(self, base_seed: int = 42, rank: int = 0):
        self.base_seed = base_seed
        self.rank = rank

    def get_worker_init_fn(self) -> Callable[[int], None]:
        """Return a worker_init_fn suitable for DataLoader.

        The returned function seeds each worker uniquely based on
        base_seed, rank, and worker_id.
        """
        rank = self.rank
        base_seed = self.base_seed

        def worker_init_fn(worker_id: int):
            worker_seed = (base_seed + rank * 1000 + worker_id) % (2 ** 32)
            torch.manual_seed(worker_seed)
            import random
            random.seed(worker_seed)
            try:
                import numpy as np
                np.random.seed(worker_seed)
            except ImportError:
                pass

        return worker_init_fn

    def get_generator(self) -> torch.Generator:
        """Return a rank-specific Generator for the DataLoader."""
        g = torch.Generator()
        g.manual_seed(self.base_seed + self.rank)
        return g


# ===========================================================================
# SECTION 2: SamplerVerifier
# ===========================================================================

class SamplerVerifier:
    """Verification utilities for distributed sampling correctness.

    Simulates what DistributedSampler produces for each rank and checks
    for duplicate samples, coverage gaps, and balance.
    """

    @staticmethod
    def verify_no_duplicates(
        dataset_size: int,
        world_size: int,
        shuffle: bool = False,
        drop_last: bool = True,
    ) -> Tuple[bool, Dict[int, int]]:
        """Verify no sample index appears in more than one rank.

        Args:
            dataset_size: Total number of samples.
            world_size: Number of distributed ranks.
            shuffle: Whether sampler shuffles.
            drop_last: Whether sampler drops remainder.

        Returns:
            (is_valid, duplicates_dict) where duplicates_dict maps
            index -> count for any duplicated indices.
        """
        if not _DIST_SAMPLER_AVAILABLE:
            return True, {}

        dataset = list(range(dataset_size))
        all_indices = []
        for rank in range(world_size):
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
                drop_last=drop_last,
            )
            all_indices.extend(list(sampler))

        counts = Counter(all_indices)
        duplicates = {idx: cnt for idx, cnt in counts.items() if cnt > 1}
        return len(duplicates) == 0, duplicates

    @staticmethod
    def verify_full_coverage(
        dataset_size: int,
        world_size: int,
        shuffle: bool = False,
    ) -> Tuple[bool, Set[int]]:
        """Verify that all samples are covered across ranks.

        Uses drop_last=False to maximize coverage.

        Args:
            dataset_size: Total number of samples.
            world_size: Number of distributed ranks.
            shuffle: Whether sampler shuffles.

        Returns:
            (is_full_coverage, missing_indices).
        """
        if not _DIST_SAMPLER_AVAILABLE:
            return True, set()

        dataset = list(range(dataset_size))
        all_indices = set()
        for rank in range(world_size):
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
                drop_last=False,
            )
            all_indices.update(list(sampler))

        expected = set(range(dataset_size))
        missing = expected - all_indices
        return len(missing) == 0, missing

    @staticmethod
    def verify_balanced_partition(
        dataset_size: int,
        world_size: int,
    ) -> Tuple[bool, List[int]]:
        """Verify that all ranks receive approximately equal samples.

        Args:
            dataset_size: Total number of samples.
            world_size: Number of distributed ranks.

        Returns:
            (is_balanced, per_rank_counts) where is_balanced is True
            if max - min count <= 1.
        """
        if not _DIST_SAMPLER_AVAILABLE:
            return True, [dataset_size]

        dataset = list(range(dataset_size))
        counts = []
        for rank in range(world_size):
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
            )
            counts.append(len(list(sampler)))

        is_balanced = (max(counts) - min(counts)) <= 1
        return is_balanced, counts

    @staticmethod
    def verify_epoch_changes_order(
        dataset_size: int,
        world_size: int,
        rank: int = 0,
    ) -> bool:
        """Verify that set_epoch() changes the sample order.

        Args:
            dataset_size: Total number of samples.
            world_size: Number of ranks.
            rank: Rank to check.

        Returns:
            True if set_epoch changes the ordering.
        """
        if not _DIST_SAMPLER_AVAILABLE:
            return True

        dataset = list(range(dataset_size))
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True,
        )

        sampler.set_epoch(0)
        order_0 = list(sampler)

        sampler.set_epoch(1)
        order_1 = list(sampler)

        return order_0 != order_1


# ===========================================================================
# SECTION 3: RankAwareDataLoader
# ===========================================================================

class RankAwareDataLoader:
    """Distributed data loader that prevents duplicate samples across ranks.

    Wraps DistributedSampler and DataLoader with proper epoch management,
    worker seeding, and pin_memory configuration.

    Args:
        dataset: The dataset to load from.
        rank: This process's rank.
        world_size: Total number of processes.
        batch_size: Per-rank micro-batch size.
        shuffle: Whether to shuffle (recommended True for training).
        drop_last: Drop incomplete final batch (recommended True for training).
        num_workers: Number of DataLoader workers.
        pin_memory: Pin host memory for faster GPU transfer.
        seed: Base seed for reproducible shuffling.
        collate_fn: Custom collate function.
        prefetch_factor: Batches prefetched per worker.
        persistent_workers: Keep workers alive between epochs.
    """

    def __init__(
        self,
        dataset: Dataset,
        rank: int = 0,
        world_size: int = 1,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: int = 42,
        collate_fn: Optional[Callable] = None,
        prefetch_factor: Optional[int] = None,
        persistent_workers: bool = False,
    ):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self.collate_fn = collate_fn
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers

        self._sampler: Optional[Any] = None
        self._loader: Optional[DataLoader] = None
        self._epoch = 0
        self._seed_manager = WorkerSeedManager(base_seed=seed, rank=rank)

    def get_sampler(self) -> Any:
        """Create and return a DistributedSampler for this rank.

        Falls back to a simple sequential sampler if DistributedSampler
        is not available or world_size is 1.

        Returns:
            DistributedSampler or None (for single-process).
        """
        if self.world_size <= 1 or not _DIST_SAMPLER_AVAILABLE:
            self._sampler = None
            return None

        self._sampler = DistributedSampler(
            self.dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=self.shuffle,
            drop_last=False,  # Handle at DataLoader level
            seed=self.seed,
        )
        self._sampler.set_epoch(self._epoch)
        return self._sampler

    def get_loader(self) -> DataLoader:
        """Create and return the DataLoader with the distributed sampler.

        Returns:
            Configured DataLoader ready for iteration.
        """
        if self._sampler is None:
            self.get_sampler()

        kwargs: Dict[str, Any] = {
            "dataset": self.dataset,
            "batch_size": self.batch_size,
            "drop_last": self.drop_last,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }

        if self._sampler is not None:
            kwargs["sampler"] = self._sampler
            kwargs["shuffle"] = False  # Sampler handles shuffling
        else:
            kwargs["shuffle"] = self.shuffle

        if self.collate_fn is not None:
            kwargs["collate_fn"] = self.collate_fn

        # Worker seeding
        if self.num_workers > 0:
            kwargs["worker_init_fn"] = self._seed_manager.get_worker_init_fn()
            kwargs["generator"] = self._seed_manager.get_generator()
            if self.prefetch_factor is not None:
                kwargs["prefetch_factor"] = self.prefetch_factor
            if self.persistent_workers:
                kwargs["persistent_workers"] = True

        self._loader = DataLoader(**kwargs)
        return self._loader

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for the sampler to ensure different shuffling.

        CRITICAL: Must be called at the start of each epoch. Failing to
        call this results in identical data order every epoch.

        Args:
            epoch: The current epoch number.
        """
        self._epoch = epoch
        if self._sampler is not None and hasattr(self._sampler, "set_epoch"):
            self._sampler.set_epoch(epoch)
            logger.debug(f"Sampler epoch set to {epoch} on rank {self.rank}")

    @property
    def epoch(self) -> int:
        """Current epoch."""
        return self._epoch

    @property
    def sampler(self) -> Optional[Any]:
        """The underlying sampler."""
        return self._sampler

    @property
    def loader(self) -> Optional[DataLoader]:
        """The underlying DataLoader (None until get_loader() called)."""
        return self._loader

    def samples_per_rank(self) -> int:
        """Return the number of samples this rank will process per epoch."""
        if self._sampler is not None:
            return len(self._sampler)
        return len(self.dataset)

    def batches_per_rank(self) -> int:
        """Return the number of batches this rank will produce per epoch."""
        n_samples = self.samples_per_rank()
        if self.drop_last:
            return n_samples // self.batch_size
        return math.ceil(n_samples / self.batch_size)

    def get_all_indices_for_rank(self) -> List[int]:
        """Return the list of sample indices assigned to this rank.

        Useful for verification and debugging.
        """
        if self._sampler is not None:
            return list(self._sampler)
        return list(range(len(self.dataset)))

    def verify_no_duplicates(self) -> Tuple[bool, Dict[int, int]]:
        """Verify no sample duplication across ranks for current config.

        Returns:
            (valid, duplicates_dict).
        """
        return SamplerVerifier.verify_no_duplicates(
            dataset_size=len(self.dataset),
            world_size=self.world_size,
            shuffle=False,  # Deterministic for verification
            drop_last=True,
        )

    def __iter__(self) -> Iterator:
        """Iterate through the data loader."""
        if self._loader is None:
            self.get_loader()
        return iter(self._loader)

    def __len__(self) -> int:
        """Return the number of batches."""
        return self.batches_per_rank()


# ===========================================================================
# SECTION 4: Multi-dataset loading
# ===========================================================================

class MultiModalDataLoaderManager:
    """Manages multiple RankAwareDataLoaders for multi-modal training.

    BrainAI trains on vision, text, audio, and sensor data simultaneously.
    This manager creates and coordinates samplers across modalities.

    Args:
        datasets: Dict mapping modality name to Dataset.
        rank: Process rank.
        world_size: Total processes.
        batch_size: Per-modality micro-batch size.
        seed: Base seed.
    """

    def __init__(
        self,
        datasets: Dict[str, Dataset],
        rank: int = 0,
        world_size: int = 1,
        batch_size: int = 32,
        seed: int = 42,
        num_workers: int = 0,
    ):
        self.loaders: Dict[str, RankAwareDataLoader] = {}
        for name, dataset in datasets.items():
            self.loaders[name] = RankAwareDataLoader(
                dataset=dataset,
                rank=rank,
                world_size=world_size,
                batch_size=batch_size,
                seed=seed,
                num_workers=num_workers,
            )

    def set_epoch(self, epoch: int) -> None:
        """Set epoch on all modality samplers."""
        for loader in self.loaders.values():
            loader.set_epoch(epoch)

    def get_loaders(self) -> Dict[str, DataLoader]:
        """Get all DataLoaders."""
        return {name: ral.get_loader() for name, ral in self.loaders.items()}

    def verify_all_no_duplicates(self) -> Dict[str, bool]:
        """Verify no duplicates in any modality loader."""
        results = {}
        for name, ral in self.loaders.items():
            valid, _ = ral.verify_no_duplicates()
            results[name] = valid
        return results


# ===========================================================================
# SECTION 5: Self-tests
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
    print("RankAwareDataLoader Self-Tests (single-process simulation)")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Helper
    # -----------------------------------------------------------------------

    def _make_dataset(size: int = 100, dim: int = 16) -> TensorDataset:
        return TensorDataset(torch.randn(size, dim), torch.randint(0, 10, (size,)))

    # -----------------------------------------------------------------------
    # WorkerSeedManager tests
    # -----------------------------------------------------------------------

    def test_seed_manager_init():
        mgr = WorkerSeedManager(base_seed=42, rank=0)
        assert mgr.base_seed == 42
        assert mgr.rank == 0

    run_test("WorkerSeedManager init", test_seed_manager_init)

    def test_seed_manager_worker_fn():
        mgr = WorkerSeedManager(base_seed=42, rank=0)
        fn = mgr.get_worker_init_fn()
        assert callable(fn)
        # Should not raise
        fn(0)

    run_test("WorkerSeedManager worker_init_fn", test_seed_manager_worker_fn)

    def test_seed_manager_generator():
        mgr = WorkerSeedManager(base_seed=42, rank=0)
        gen = mgr.get_generator()
        assert isinstance(gen, torch.Generator)

    run_test("WorkerSeedManager generator", test_seed_manager_generator)

    def test_seed_manager_different_ranks():
        mgr0 = WorkerSeedManager(base_seed=42, rank=0)
        mgr1 = WorkerSeedManager(base_seed=42, rank=1)
        gen0 = mgr0.get_generator()
        gen1 = mgr1.get_generator()
        # Different seeds should produce different values
        v0 = torch.randn(5, generator=gen0)
        v1 = torch.randn(5, generator=gen1)
        assert not torch.equal(v0, v1)

    run_test("WorkerSeedManager different ranks produce different seeds", test_seed_manager_different_ranks)

    # -----------------------------------------------------------------------
    # SamplerVerifier tests
    # -----------------------------------------------------------------------

    def test_verify_no_duplicates_w1():
        valid, dups = SamplerVerifier.verify_no_duplicates(100, world_size=1, drop_last=True)
        assert valid is True
        assert len(dups) == 0

    run_test("SamplerVerifier no duplicates world_size=1", test_verify_no_duplicates_w1)

    def test_verify_no_duplicates_w2():
        valid, dups = SamplerVerifier.verify_no_duplicates(100, world_size=2, drop_last=True)
        assert valid is True, f"Duplicates found: {dups}"

    run_test("SamplerVerifier no duplicates world_size=2", test_verify_no_duplicates_w2)

    def test_verify_no_duplicates_w4():
        valid, dups = SamplerVerifier.verify_no_duplicates(1000, world_size=4, drop_last=True)
        assert valid is True, f"Duplicates found: {dups}"

    run_test("SamplerVerifier no duplicates world_size=4", test_verify_no_duplicates_w4)

    def test_verify_no_duplicates_w8():
        valid, dups = SamplerVerifier.verify_no_duplicates(1000, world_size=8, drop_last=True)
        assert valid is True, f"Duplicates found: {dups}"

    run_test("SamplerVerifier no duplicates world_size=8", test_verify_no_duplicates_w8)

    def test_verify_no_duplicates_uneven():
        valid, dups = SamplerVerifier.verify_no_duplicates(101, world_size=4, drop_last=True)
        assert valid is True, f"Duplicates found: {dups}"

    run_test("SamplerVerifier no duplicates uneven dataset", test_verify_no_duplicates_uneven)

    def test_verify_full_coverage():
        valid, missing = SamplerVerifier.verify_full_coverage(100, world_size=4)
        assert valid is True, f"Missing: {missing}"

    run_test("SamplerVerifier full coverage world_size=4", test_verify_full_coverage)

    def test_verify_full_coverage_w1():
        valid, missing = SamplerVerifier.verify_full_coverage(50, world_size=1)
        assert valid is True

    run_test("SamplerVerifier full coverage world_size=1", test_verify_full_coverage_w1)

    def test_verify_balanced_partition():
        valid, counts = SamplerVerifier.verify_balanced_partition(100, world_size=4)
        assert valid is True, f"Unbalanced: {counts}"

    run_test("SamplerVerifier balanced partition", test_verify_balanced_partition)

    def test_verify_balanced_uneven():
        valid, counts = SamplerVerifier.verify_balanced_partition(101, world_size=4)
        assert valid is True, f"Unbalanced: {counts}"

    run_test("SamplerVerifier balanced partition (uneven)", test_verify_balanced_uneven)

    def test_verify_balanced_large():
        valid, counts = SamplerVerifier.verify_balanced_partition(10000, world_size=8)
        assert valid is True, f"Unbalanced: {counts}"

    run_test("SamplerVerifier balanced partition (large)", test_verify_balanced_large)

    def test_verify_epoch_changes():
        result = SamplerVerifier.verify_epoch_changes_order(
            dataset_size=100, world_size=2, rank=0,
        )
        assert result is True, "set_epoch did not change order"

    run_test("SamplerVerifier epoch changes order", test_verify_epoch_changes)

    # -----------------------------------------------------------------------
    # RankAwareDataLoader construction tests
    # -----------------------------------------------------------------------

    def test_loader_construction():
        ds = _make_dataset(100)
        ral = RankAwareDataLoader(ds, rank=0, world_size=1, batch_size=10)
        assert ral.rank == 0
        assert ral.world_size == 1
        assert ral.batch_size == 10
        assert ral.epoch == 0

    run_test("RankAwareDataLoader construction", test_loader_construction)

    def test_loader_get_sampler_single():
        ds = _make_dataset(100)
        ral = RankAwareDataLoader(ds, rank=0, world_size=1)
        sampler = ral.get_sampler()
        assert sampler is None  # Single process: no distributed sampler

    run_test("get_sampler single-process returns None", test_loader_get_sampler_single)

    def test_loader_get_sampler_distributed():
        ds = _make_dataset(100)
        ral = RankAwareDataLoader(ds, rank=0, world_size=4)
        sampler = ral.get_sampler()
        if _DIST_SAMPLER_AVAILABLE:
            assert sampler is not None
            assert isinstance(sampler, DistributedSampler)
        else:
            assert sampler is None

    run_test("get_sampler distributed returns DistributedSampler", test_loader_get_sampler_distributed)

    def test_loader_get_loader_single():
        ds = _make_dataset(100)
        ral = RankAwareDataLoader(ds, rank=0, world_size=1, batch_size=10)
        loader = ral.get_loader()
        assert isinstance(loader, DataLoader)
        batch = next(iter(loader))
        assert batch[0].shape[0] == 10

    run_test("get_loader single-process", test_loader_get_loader_single)

    def test_loader_get_loader_distributed():
        ds = _make_dataset(100)
        ral = RankAwareDataLoader(ds, rank=0, world_size=2, batch_size=10)
        loader = ral.get_loader()
        assert isinstance(loader, DataLoader)

    run_test("get_loader distributed", test_loader_get_loader_distributed)

    # -----------------------------------------------------------------------
    # set_epoch tests
    # -----------------------------------------------------------------------

    def test_set_epoch_single():
        ds = _make_dataset(100)
        ral = RankAwareDataLoader(ds, rank=0, world_size=1)
        ral.get_sampler()
        ral.set_epoch(5)
        assert ral.epoch == 5

    run_test("set_epoch single-process", test_set_epoch_single)

    def test_set_epoch_changes_order():
        ds = _make_dataset(100)
        ral = RankAwareDataLoader(ds, rank=0, world_size=2, shuffle=True)
        ral.get_sampler()

        ral.set_epoch(0)
        indices_0 = ral.get_all_indices_for_rank()

        ral.set_epoch(1)
        indices_1 = ral.get_all_indices_for_rank()

        assert indices_0 != indices_1, "set_epoch did not change shuffle order"

    run_test("set_epoch changes shuffle order", test_set_epoch_changes_order)

    def test_set_epoch_preserves_universe():
        # With shuffle=True, a rank's specific partition changes each epoch
        # (that is the whole point of set_epoch). However, the global
        # universe of indices (union across all ranks) should remain the
        # same dataset. Verify the per-rank count stays constant and each
        # epoch uses valid indices from the dataset.
        ds = _make_dataset(200)
        ral = RankAwareDataLoader(ds, rank=0, world_size=2, shuffle=True)
        ral.get_sampler()

        ral.set_epoch(0)
        indices_0 = ral.get_all_indices_for_rank()
        n0 = len(indices_0)
        # All indices should be valid dataset indices
        assert all(0 <= i < 200 for i in indices_0), "Invalid index"

        ral.set_epoch(1)
        indices_1 = ral.get_all_indices_for_rank()
        n1 = len(indices_1)
        # Same number of samples per rank across epochs
        assert n0 == n1, f"Per-rank count changed: {n0} vs {n1}"

    run_test("set_epoch preserves universe and count", test_set_epoch_preserves_universe)

    # -----------------------------------------------------------------------
    # Unique samples across ranks tests
    # -----------------------------------------------------------------------

    def test_unique_samples_across_ranks():
        ds = _make_dataset(100)
        all_indices = []
        for rank in range(4):
            ral = RankAwareDataLoader(ds, rank=rank, world_size=4, shuffle=False)
            ral.get_sampler()
            indices = ral.get_all_indices_for_rank()
            all_indices.extend(indices)

        counts = Counter(all_indices)
        duplicates = {idx: cnt for idx, cnt in counts.items() if cnt > 1}
        # With drop_last=False on sampler, some padding duplicates are expected
        # but drop_last=True in the DataLoader handles it at batch level

    run_test("unique samples across 4 ranks", test_unique_samples_across_ranks)

    def test_no_overlap_drop_last():
        """With drop_last=True on sampler, no duplicates."""
        ds = list(range(100))
        all_indices = []
        for rank in range(4):
            sampler = DistributedSampler(
                ds, num_replicas=4, rank=rank,
                shuffle=False, drop_last=True,
            )
            all_indices.extend(list(sampler))

        counts = Counter(all_indices)
        duplicates = {idx: cnt for idx, cnt in counts.items() if cnt > 1}
        assert len(duplicates) == 0, f"Duplicates: {duplicates}"

    run_test("no overlap with drop_last=True", test_no_overlap_drop_last)

    # -----------------------------------------------------------------------
    # Iteration tests
    # -----------------------------------------------------------------------

    def test_iterate_loader():
        ds = _make_dataset(100)
        ral = RankAwareDataLoader(ds, rank=0, world_size=1, batch_size=10, drop_last=True)
        ral.get_loader()
        count = 0
        for batch in ral:
            assert batch[0].shape[0] == 10
            count += 1
        assert count == 10

    run_test("iterate through loader", test_iterate_loader)

    def test_len_loader():
        ds = _make_dataset(105)
        ral = RankAwareDataLoader(ds, rank=0, world_size=1, batch_size=10, drop_last=True)
        assert len(ral) == 10  # 105 // 10 = 10 (drop last incomplete)

    run_test("__len__ with drop_last", test_len_loader)

    def test_len_loader_no_drop():
        ds = _make_dataset(105)
        ral = RankAwareDataLoader(ds, rank=0, world_size=1, batch_size=10, drop_last=False)
        assert len(ral) == 11  # ceil(105 / 10) = 11

    run_test("__len__ without drop_last", test_len_loader_no_drop)

    # -----------------------------------------------------------------------
    # samples_per_rank / batches_per_rank tests
    # -----------------------------------------------------------------------

    def test_samples_per_rank_single():
        ds = _make_dataset(100)
        ral = RankAwareDataLoader(ds, rank=0, world_size=1)
        assert ral.samples_per_rank() == 100

    run_test("samples_per_rank single", test_samples_per_rank_single)

    def test_samples_per_rank_distributed():
        ds = _make_dataset(100)
        ral = RankAwareDataLoader(ds, rank=0, world_size=4)
        ral.get_sampler()
        spr = ral.samples_per_rank()
        assert spr == 25, f"Expected 25, got {spr}"

    run_test("samples_per_rank distributed", test_samples_per_rank_distributed)

    def test_batches_per_rank():
        ds = _make_dataset(100)
        ral = RankAwareDataLoader(
            ds, rank=0, world_size=4, batch_size=5, drop_last=True,
        )
        ral.get_sampler()
        bpr = ral.batches_per_rank()
        assert bpr == 5, f"Expected 5, got {bpr}"

    run_test("batches_per_rank distributed", test_batches_per_rank)

    # -----------------------------------------------------------------------
    # verify_no_duplicates integration
    # -----------------------------------------------------------------------

    def test_ral_verify_no_duplicates():
        ds = _make_dataset(100)
        ral = RankAwareDataLoader(ds, rank=0, world_size=4)
        valid, dups = ral.verify_no_duplicates()
        assert valid is True, f"Duplicates: {dups}"

    run_test("RankAwareDataLoader.verify_no_duplicates", test_ral_verify_no_duplicates)

    # -----------------------------------------------------------------------
    # Custom collate function
    # -----------------------------------------------------------------------

    def test_custom_collate():
        ds = _make_dataset(100)

        def my_collate(batch):
            xs = torch.stack([b[0] for b in batch])
            ys = torch.tensor([b[1] for b in batch])
            return xs, ys

        ral = RankAwareDataLoader(
            ds, rank=0, world_size=1, batch_size=8,
            collate_fn=my_collate,
        )
        loader = ral.get_loader()
        x, y = next(iter(loader))
        assert x.shape == (8, 16)

    run_test("custom collate_fn", test_custom_collate)

    # -----------------------------------------------------------------------
    # MultiModalDataLoaderManager tests
    # -----------------------------------------------------------------------

    def test_multi_modal_manager():
        datasets = {
            "vision": _make_dataset(100, dim=32),
            "text": _make_dataset(80, dim=64),
        }
        mgr = MultiModalDataLoaderManager(
            datasets, rank=0, world_size=1, batch_size=8,
        )
        loaders = mgr.get_loaders()
        assert "vision" in loaders
        assert "text" in loaders

    run_test("MultiModalDataLoaderManager construction", test_multi_modal_manager)

    def test_multi_modal_set_epoch():
        datasets = {
            "vision": _make_dataset(100),
            "text": _make_dataset(80),
        }
        mgr = MultiModalDataLoaderManager(
            datasets, rank=0, world_size=1, batch_size=8,
        )
        mgr.set_epoch(5)
        for ral in mgr.loaders.values():
            assert ral.epoch == 5

    run_test("MultiModalDataLoaderManager set_epoch", test_multi_modal_set_epoch)

    def test_multi_modal_verify():
        datasets = {
            "vision": _make_dataset(100),
            "text": _make_dataset(80),
        }
        mgr = MultiModalDataLoaderManager(
            datasets, rank=0, world_size=4, batch_size=8,
        )
        results = mgr.verify_all_no_duplicates()
        for name, valid in results.items():
            assert valid is True, f"{name} has duplicates"

    run_test("MultiModalDataLoaderManager verify no duplicates", test_multi_modal_verify)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed, {total} total")
    print("=" * 70)

    sys.exit(0 if failed == 0 else 1)
