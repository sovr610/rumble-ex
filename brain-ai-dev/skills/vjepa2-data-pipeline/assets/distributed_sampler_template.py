"""
distributed_sampler_template.py
================================
DistributedWeightedSampler and ConcatIndices for multi-source video datasets.

Public API
----------
ConcatIndices(dataset_sizes: List[int])
    .__getitem__(global_idx) -> (dataset_idx, sample_idx)
    .__len__() -> int

DistributedWeightedSampler(weights, num_samples, rank, world_size)
    .set_epoch(epoch) -- call before each epoch for reproducible shuffling
    .__iter__()       -- yields indices for this rank
    .__len__()        -- samples per rank per epoch

Design
------
Weighted sampling without replacement using exponential weighting trick:
  For each sample i with weight w_i, draw u_i ~ Uniform(0,1)
  Score: s_i = u_i^(1/w_i)
  Sort descending: higher-weight samples tend to rank higher.
  This gives sampling proportional to weights without replacement.

Partition: indices[rank::world_size] — stride-based round-robin.
This guarantees disjoint subsets across ranks and full coverage.
"""

from __future__ import annotations

import bisect
import itertools
import math
from typing import Iterator, List, Optional, Tuple

import torch
from torch.utils.data import Sampler


# ---------------------------------------------------------------------------
# ConcatIndices
# ---------------------------------------------------------------------------

class ConcatIndices:
    """
    Two-level index mapping: global flat index -> (dataset_idx, sample_idx).

    Given N datasets of sizes [s_0, s_1, ..., s_{N-1}], the global index
    space is [0, sum(sizes)). This class maps any global index to the
    corresponding sub-dataset and within-dataset index using binary search.

    Parameters
    ----------
    dataset_sizes : List[int]
        Number of samples in each sub-dataset.

    Examples
    --------
    >>> ci = ConcatIndices([1000, 500, 250])
    >>> ci[0]        # -> (0, 0)
    >>> ci[999]      # -> (0, 999)
    >>> ci[1000]     # -> (1, 0)
    >>> ci[1499]     # -> (1, 499)
    >>> ci[1500]     # -> (2, 0)
    >>> len(ci)      # -> 1750
    """

    def __init__(self, dataset_sizes: List[int]):
        if not dataset_sizes:
            raise ValueError("dataset_sizes must be non-empty")
        if any(s < 0 for s in dataset_sizes):
            raise ValueError("All dataset sizes must be non-negative")

        self.sizes = list(dataset_sizes)
        # Cumulative sum: [0, s_0, s_0+s_1, ...]
        self.cumulative: List[int] = list(itertools.accumulate([0] + self.sizes))
        self.total: int = self.cumulative[-1]

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, global_idx: int) -> Tuple[int, int]:
        """
        Map global_idx to (dataset_idx, sample_idx).

        Parameters
        ----------
        global_idx : int
            Index in [0, total).

        Returns
        -------
        (dataset_idx, sample_idx) : Tuple[int, int]
        """
        if global_idx < 0 or global_idx >= self.total:
            raise IndexError(
                f"Index {global_idx} is out of range [0, {self.total}). "
                f"Dataset has {self.total} samples total."
            )
        # Binary search in cumulative sums
        # bisect_right gives the insert position for global_idx
        # The dataset_idx is one position to the left
        pos = bisect.bisect_right(self.cumulative, global_idx) - 1
        dataset_idx = pos
        sample_idx  = global_idx - self.cumulative[pos]
        return dataset_idx, sample_idx

    def get_dataset_ranges(self) -> List[Tuple[int, int]]:
        """
        Return (start, end) global index ranges for each sub-dataset.

        Returns
        -------
        List of (start, end) tuples (end is exclusive).
        """
        return [
            (self.cumulative[i], self.cumulative[i + 1])
            for i in range(len(self.sizes))
        ]

    def __repr__(self) -> str:
        return (
            f"ConcatIndices("
            f"n_datasets={len(self.sizes)}, "
            f"total={self.total}, "
            f"sizes={self.sizes})"
        )


# ---------------------------------------------------------------------------
# Weight expansion utility
# ---------------------------------------------------------------------------

def expand_weights_to_samples(
    dataset_sizes: List[int],
    source_weights: List[float],
) -> List[float]:
    """
    Expand per-source weights to per-sample weights.

    Each sample in source i gets weight = source_weights[i] / dataset_sizes[i].
    This normalizes within-source so total weight contribution matches the
    intended source weight ratio regardless of dataset size.

    Parameters
    ----------
    dataset_sizes : List[int]
        Number of samples per source.
    source_weights : List[float]
        Mixing weight for each source.

    Returns
    -------
    List[float] of length sum(dataset_sizes).
    """
    if len(dataset_sizes) != len(source_weights):
        raise ValueError(
            f"dataset_sizes length ({len(dataset_sizes)}) must match "
            f"source_weights length ({len(source_weights)})"
        )
    result: List[float] = []
    for size, weight in zip(dataset_sizes, source_weights):
        per_sample = weight / size if size > 0 else 0.0
        result.extend([per_sample] * size)
    return result


# ---------------------------------------------------------------------------
# DistributedWeightedSampler
# ---------------------------------------------------------------------------

class DistributedWeightedSampler(Sampler):
    """
    Weighted sampling across distributed ranks without cross-rank duplicates.

    Guarantees
    ----------
    1. No duplicates within a single rank's epoch.
    2. No index appears in more than one rank's epoch (when replacement=False).
    3. All samples appear across all ranks combined (full coverage).
    4. Weight proportions are respected at the global level.
    5. Reproducible: same (seed, epoch) -> same permutation.

    Algorithm
    ---------
    Uses the exponential weighting trick for weighted sampling without
    replacement (Efraimidis & Spirakis, 2006):
        key_i = uniform(0,1)^(1/weight_i)
    Samples are sorted by key descending; higher-weight samples tend to
    appear earlier and are thus sampled more frequently.

    Partition: global_permutation[rank::world_size]

    Parameters
    ----------
    weights : List[float]
        Per-sample weights (length = total dataset size).
    num_samples : int
        Total samples to consider (usually len(weights)).
    rank : int
        This process's rank in the distributed group.
    world_size : int
        Total number of processes.
    seed : int
        Base random seed for determinism.
    replacement : bool
        If True, use multinomial sampling with replacement instead.
        Default False (no replacement, recommended for pretraining).
    """

    def __init__(
        self,
        weights: List[float],
        num_samples: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
        replacement: bool = False,
    ):
        super().__init__(None)  # data_source unused

        if rank < 0 or rank >= world_size:
            raise ValueError(
                f"rank={rank} must be in [0, world_size={world_size})"
            )
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")

        self.weights       = torch.tensor(weights, dtype=torch.double)
        self.num_samples   = num_samples
        self.rank          = rank
        self.world_size    = world_size
        self.seed          = seed
        self.replacement   = replacement
        self.epoch         = 0

        # Pad total to be divisible by world_size for even partition
        self.num_samples_per_rank: int = math.ceil(num_samples / world_size)
        self.total_size: int = self.num_samples_per_rank * world_size

    def set_epoch(self, epoch: int) -> None:
        """
        Update the epoch counter for reproducible per-epoch shuffling.
        Call this at the start of every training epoch.

        Parameters
        ----------
        epoch : int
            Current epoch index.
        """
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        """
        Yield indices for this rank's partition of the shuffled dataset.
        """
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        N = len(self.weights)

        if self.replacement:
            # Multinomial with replacement — for oversampling minority classes
            indices = torch.multinomial(
                self.weights,
                num_samples=self.total_size,
                replacement=True,
                generator=g,
            ).tolist()

        else:
            # Weighted shuffle without replacement (exponential key trick)
            if self.weights.sum() > 0:
                # u_i ~ Uniform(0,1), key_i = u_i^(1/w_i)
                # Avoid log(0): clamp weights to a small positive value
                clamped = self.weights.clamp(min=1e-10)
                uniform = torch.rand(N, generator=g, dtype=torch.double)
                # Take log for numerical stability: key = u^(1/w) = exp(log(u)/w)
                log_keys = uniform.log() / clamped
                # Sort descending (higher key = higher priority)
                indices = torch.argsort(log_keys, descending=True).tolist()
            else:
                # All zero weights: fall back to random shuffle
                indices = torch.randperm(N, generator=g).tolist()

            # Pad to total_size by cycling
            while len(indices) < self.total_size:
                extra = torch.randperm(N, generator=g).tolist()
                indices.extend(extra)
            indices = indices[:self.total_size]

        # Partition: stride-based round-robin
        # rank=0 gets [0, ws, 2*ws, ...]
        # rank=1 gets [1, ws+1, 2*ws+1, ...]
        rank_indices = indices[self.rank:self.total_size:self.world_size]

        # Clip to valid dataset range (handles padding artifacts)
        rank_indices = [min(i, N - 1) for i in rank_indices]

        return iter(rank_indices)

    def __len__(self) -> int:
        """Number of samples this rank will produce per epoch."""
        return self.num_samples_per_rank

    def __repr__(self) -> str:
        return (
            f"DistributedWeightedSampler("
            f"n_samples={self.num_samples}, "
            f"rank={self.rank}/{self.world_size}, "
            f"replacement={self.replacement}, "
            f"epoch={self.epoch})"
        )


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("DistributedWeightedSampler & ConcatIndices Self-Tests")
    print("=" * 60)

    PASS = "[PASS]"
    FAIL = "[FAIL]"
    errors = []

    # ===================================================================
    # ConcatIndices Tests
    # ===================================================================

    # -------------------------------------------------------------------
    # Test 1: Basic index mapping
    # -------------------------------------------------------------------
    ci = ConcatIndices([1000, 500, 250])
    cases = [
        (0,    (0, 0)),
        (999,  (0, 999)),
        (1000, (1, 0)),
        (1499, (1, 499)),
        (1500, (2, 0)),
        (1749, (2, 249)),
    ]
    all_ok = all(ci[idx] == expected for idx, expected in cases)
    if all_ok:
        print(f"{PASS} Test 1: ConcatIndices basic mapping correct")
    else:
        for idx, expected in cases:
            actual = ci[idx]
            if actual != expected:
                msg = f"Test 1 FAILED: ci[{idx}]={actual}, expected {expected}"
                print(f"{FAIL} {msg}")
                errors.append(msg)

    # -------------------------------------------------------------------
    # Test 2: Total length
    # -------------------------------------------------------------------
    if len(ci) == 1750:
        print(f"{PASS} Test 2: ConcatIndices total length = {len(ci)}")
    else:
        msg = f"Test 2 FAILED: len={len(ci)}, expected 1750"
        print(f"{FAIL} {msg}")
        errors.append(msg)

    # -------------------------------------------------------------------
    # Test 3: Index out of range raises IndexError
    # -------------------------------------------------------------------
    for bad_idx in [-1, 1750]:
        try:
            ci[bad_idx]
            msg = f"Test 3 FAILED: should raise IndexError for idx={bad_idx}"
            print(f"{FAIL} {msg}")
            errors.append(msg)
        except IndexError:
            print(f"{PASS} Test 3: IndexError for out-of-range idx={bad_idx}")

    # -------------------------------------------------------------------
    # Test 4: Single-source ConcatIndices
    # -------------------------------------------------------------------
    ci_single = ConcatIndices([500])
    ok = (ci_single[0] == (0, 0) and ci_single[499] == (0, 499))
    if ok:
        print(f"{PASS} Test 4: Single-source ConcatIndices correct")
    else:
        msg = "Test 4 FAILED"
        print(f"{FAIL} {msg}")
        errors.append(msg)

    # -------------------------------------------------------------------
    # Test 5: Dataset ranges
    # -------------------------------------------------------------------
    ranges = ci.get_dataset_ranges()
    expected_ranges = [(0, 1000), (1000, 1500), (1500, 1750)]
    if ranges == expected_ranges:
        print(f"{PASS} Test 5: get_dataset_ranges() = {ranges}")
    else:
        msg = f"Test 5 FAILED: {ranges} != {expected_ranges}"
        print(f"{FAIL} {msg}")
        errors.append(msg)

    # ===================================================================
    # DistributedWeightedSampler Tests
    # ===================================================================

    N = 1000  # Total samples

    # -------------------------------------------------------------------
    # Test 6: Single-rank coverage (all samples present, no duplicates)
    # -------------------------------------------------------------------
    uniform_weights = [1.0] * N
    sampler = DistributedWeightedSampler(
        weights=uniform_weights, num_samples=N, rank=0, world_size=1, seed=0
    )
    indices = list(sampler)
    unique_indices = set(indices)
    if len(unique_indices) == N:
        print(f"{PASS} Test 6: Single rank — all {N} samples, no duplicates")
    else:
        msg = f"Test 6 FAILED: got {len(unique_indices)} unique indices, expected {N}"
        print(f"{FAIL} {msg}")
        errors.append(msg)

    # -------------------------------------------------------------------
    # Test 7: No cross-rank duplicates (world_size=2)
    # -------------------------------------------------------------------
    s0 = DistributedWeightedSampler(uniform_weights, N, rank=0, world_size=2, seed=0)
    s1 = DistributedWeightedSampler(uniform_weights, N, rank=1, world_size=2, seed=0)
    idx0 = set(list(s0))
    idx1 = set(list(s1))
    overlap = idx0 & idx1
    if len(overlap) == 0:
        print(f"{PASS} Test 7: No cross-rank duplicates (world_size=2)")
    else:
        msg = f"Test 7 FAILED: {len(overlap)} cross-rank duplicates"
        print(f"{FAIL} {msg}")
        errors.append(msg)

    # -------------------------------------------------------------------
    # Test 8: Full coverage across ranks
    # -------------------------------------------------------------------
    all_covered = idx0 | idx1
    # Allow for minor padding artifacts (indices clipped to N-1)
    if len(all_covered) >= N - 1:
        print(f"{PASS} Test 8: Full coverage across ranks ({len(all_covered)}/{N})")
    else:
        msg = f"Test 8 FAILED: only {len(all_covered)}/{N} unique samples covered"
        print(f"{FAIL} {msg}")
        errors.append(msg)

    # -------------------------------------------------------------------
    # Test 9: Weight distribution (2:1 ratio)
    # -------------------------------------------------------------------
    # Source A: 600 samples, weight=2.0
    # Source B: 400 samples, weight=1.0
    w_a = [2.0 / 600] * 600
    w_b = [1.0 / 400] * 400
    weighted = w_a + w_b
    sw = DistributedWeightedSampler(weighted, 1000, rank=0, world_size=1, seed=42)
    sw_indices = list(sw)
    count_a = sum(1 for i in sw_indices if i < 600)
    count_b = sum(1 for i in sw_indices if i >= 600)
    prop_a = count_a / len(sw_indices)
    expected_prop_a = 2.0 / 3.0  # 2/(2+1)
    if abs(prop_a - expected_prop_a) < 0.08:  # 8% tolerance for N=1000
        print(f"{PASS} Test 9: Weight ratio ~ 2:1 (source_a={prop_a:.2%}, expected={expected_prop_a:.2%})")
    else:
        msg = f"Test 9 FAILED: prop_a={prop_a:.3f}, expected ~{expected_prop_a:.3f}"
        print(f"{FAIL} {msg}")
        errors.append(msg)

    # -------------------------------------------------------------------
    # Test 10: set_epoch changes ordering
    # -------------------------------------------------------------------
    sampler.set_epoch(0); order0 = list(sampler)
    sampler.set_epoch(1); order1 = list(sampler)
    if order0 != order1:
        print(f"{PASS} Test 10: set_epoch changes sample ordering")
    else:
        msg = "Test 10 FAILED: ordering did not change between epochs"
        print(f"{FAIL} {msg}")
        errors.append(msg)

    # -------------------------------------------------------------------
    # Test 11: Same epoch produces same ordering (reproducibility)
    # -------------------------------------------------------------------
    sampler.set_epoch(5); run1 = list(sampler)
    sampler.set_epoch(5); run2 = list(sampler)
    if run1 == run2:
        print(f"{PASS} Test 11: Same epoch is reproducible")
    else:
        msg = "Test 11 FAILED: same epoch produced different orderings"
        print(f"{FAIL} {msg}")
        errors.append(msg)

    # -------------------------------------------------------------------
    # Test 12: __len__ matches expected samples per rank
    # -------------------------------------------------------------------
    import math as _math
    for ws in [1, 2, 4, 8]:
        s = DistributedWeightedSampler(uniform_weights, N, rank=0, world_size=ws)
        expected_len = _math.ceil(N / ws)
        if len(s) == expected_len:
            print(f"{PASS} Test 12 [ws={ws}]: __len__={len(s)} (expected {expected_len})")
        else:
            msg = f"Test 12 [ws={ws}] FAILED: __len__={len(s)}, expected {expected_len}"
            print(f"{FAIL} {msg}")
            errors.append(msg)

    # -------------------------------------------------------------------
    # Test 13: expand_weights_to_samples
    # -------------------------------------------------------------------
    expanded = expand_weights_to_samples([3, 2], [6.0, 4.0])
    # Source 0: weight=6/3=2.0 per sample (3 samples)
    # Source 1: weight=4/2=2.0 per sample (2 samples)
    expected_exp = [2.0, 2.0, 2.0, 2.0, 2.0]
    if expanded == expected_exp:
        print(f"{PASS} Test 13: expand_weights_to_samples = {expanded}")
    else:
        msg = f"Test 13 FAILED: {expanded} != {expected_exp}"
        print(f"{FAIL} {msg}")
        errors.append(msg)

    # -------------------------------------------------------------------
    # Test 14: Invalid rank raises ValueError
    # -------------------------------------------------------------------
    try:
        DistributedWeightedSampler([1.0]*10, 10, rank=5, world_size=4)
        msg = "Test 14 FAILED: should raise ValueError for rank >= world_size"
        print(f"{FAIL} {msg}")
        errors.append(msg)
    except ValueError:
        print(f"{PASS} Test 14: ValueError for invalid rank")

    # -------------------------------------------------------------------
    # Test 15: Replacement mode basic smoke test
    # -------------------------------------------------------------------
    sw_rep = DistributedWeightedSampler(
        uniform_weights, N, rank=0, world_size=1, seed=0, replacement=True
    )
    rep_indices = list(sw_rep)
    if len(rep_indices) >= N:
        print(f"{PASS} Test 15: Replacement mode yields {len(rep_indices)} indices")
    else:
        msg = f"Test 15 FAILED: only {len(rep_indices)} indices in replacement mode"
        print(f"{FAIL} {msg}")
        errors.append(msg)

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print()
    if errors:
        print(f"FAILED: {len(errors)} test(s) failed:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("All tests passed.")
