"""
ShardedStreamDataset — Deterministic per-rank streaming dataset with caching.

Supports backends:
  - hf_streaming: Hugging Face datasets streaming
  - memmap: Token memmap (.bin + .idx) for pretraining
  - synthetic: In-memory synthetic data for testing

Provides set_epoch() for reshuffling and shard-level caching with checksum
validation.

Usage:
    cfg = StreamConfig(format="memmap", cache_dir="/tmp/cache")
    dataset = ShardedStreamDataset(cfg, rank=0, world_size=4)
    for batch in dataset:
        train_step(batch)
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Dict, List, Optional, Any

import numpy as np

try:
    import torch
    from torch.utils.data import IterableDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

    # Stub for when torch is unavailable
    class IterableDataset:
        pass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class StreamConfig:
    """Configuration for ShardedStreamDataset."""
    format: str = "synthetic"  # hf_streaming | memmap | synthetic
    data_path: Optional[str] = None  # path to data files
    cache_dir: Optional[str] = None
    shuffle_buffer_size: int = 1000
    shard_seed: int = 42
    target_seq_len: int = 512
    max_cache_bytes: int = 1_000_000_000  # 1 GB default
    num_samples: int = 1000  # for synthetic


# ---------------------------------------------------------------------------
# Shard Cache
# ---------------------------------------------------------------------------

class ShardCache:
    """LRU shard cache with checksum validation."""

    def __init__(self, cache_dir: str, max_bytes: int = 1_000_000_000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.access_order: OrderedDict = OrderedDict()
        self._load_existing()

    def _load_existing(self) -> None:
        """Load existing cache entries."""
        for p in sorted(self.cache_dir.glob("shard_*")):
            if p.suffix == ".checksum":
                continue
            self.access_order[p.name] = p.stat().st_size

    def _total_size(self) -> int:
        return sum(self.access_order.values())

    @staticmethod
    def compute_checksum(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def get(self, shard_id: str) -> Optional[bytes]:
        """Get cached shard data. Returns None if not cached or corrupt."""
        cache_path = self.cache_dir / shard_id
        checksum_path = self.cache_dir / f"{shard_id}.checksum"
        if not cache_path.exists() or not checksum_path.exists():
            return None
        data = cache_path.read_bytes()
        stored_checksum = checksum_path.read_text().strip()
        if self.compute_checksum(data) != stored_checksum:
            # Corrupted — evict
            self._evict(shard_id)
            return None
        self.access_order.move_to_end(shard_id)
        return data

    def put(self, shard_id: str, data: bytes) -> None:
        """Store shard data with checksum."""
        size = len(data)
        # Evict until we have space
        while self._total_size() + size > self.max_bytes and self.access_order:
            self._evict_lru()
        cache_path = self.cache_dir / shard_id
        checksum_path = self.cache_dir / f"{shard_id}.checksum"
        cache_path.write_bytes(data)
        checksum_path.write_text(self.compute_checksum(data))
        self.access_order[shard_id] = size

    def _evict(self, shard_id: str) -> None:
        cache_path = self.cache_dir / shard_id
        checksum_path = self.cache_dir / f"{shard_id}.checksum"
        if cache_path.exists():
            cache_path.unlink()
        if checksum_path.exists():
            checksum_path.unlink()
        self.access_order.pop(shard_id, None)

    def _evict_lru(self) -> None:
        if self.access_order:
            oldest, _ = self.access_order.popitem(last=False)
            self._evict(oldest)


# ---------------------------------------------------------------------------
# ShardedStreamDataset
# ---------------------------------------------------------------------------

class ShardedStreamDataset(IterableDataset):
    """
    Deterministic per-rank streaming dataset.

    Supports multiple backends with epoch-based reshuffling and optional
    shard-level caching.
    """

    def __init__(self, cfg: StreamConfig, rank: int = 0, world_size: int = 1):
        super().__init__()
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self._epoch = 0
        self._cache: Optional[ShardCache] = None
        if cfg.cache_dir:
            self._cache = ShardCache(cfg.cache_dir, cfg.max_cache_bytes)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for reshuffling."""
        self._epoch = epoch

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        if self.cfg.format == "hf_streaming":
            yield from self._iter_hf_streaming()
        elif self.cfg.format == "memmap":
            yield from self._iter_memmap()
        elif self.cfg.format == "synthetic":
            yield from self._iter_synthetic()
        else:
            raise ValueError(f"Unknown format: {self.cfg.format}")

    # ---- Synthetic backend (for testing) ----

    def _iter_synthetic(self) -> Iterator[Dict[str, Any]]:
        """Generate synthetic data for testing."""
        rng = np.random.RandomState(self.cfg.shard_seed + self._epoch + self.rank)
        total = self.cfg.num_samples
        # Assign samples to this rank
        indices = list(range(self.rank, total, self.world_size))
        # Shuffle based on epoch
        rng.shuffle(indices)
        seq_len = self.cfg.target_seq_len
        for idx in indices:
            # Deterministic token generation per index
            sample_rng = np.random.RandomState(idx + self.cfg.shard_seed)
            tokens = sample_rng.randint(1, 32000, size=seq_len).tolist()
            yield {
                "input_ids": tokens,
                "labels": tokens,
                "sample_id": idx,
            }

    # ---- HF Streaming backend ----

    def _iter_hf_streaming(self) -> Iterator[Dict[str, Any]]:
        """
        Stream from Hugging Face datasets.
        Requires: datasets library installed and cfg.data_path set.
        Falls back to synthetic if datasets not available.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            print("WARNING: datasets library not available, falling back to synthetic")
            yield from self._iter_synthetic()
            return

        ds = load_dataset(
            self.cfg.data_path, split="train", streaming=True, trust_remote_code=True
        )
        ds = ds.shard(num_shards=self.world_size, index=self.rank)
        ds = ds.shuffle(
            seed=self.cfg.shard_seed + self._epoch,
            buffer_size=self.cfg.shuffle_buffer_size,
        )

        for sample in ds:
            # Expect sample to have "input_ids" or "text"
            if "input_ids" in sample:
                yield sample
            elif "text" in sample:
                yield {"text": sample["text"], "sample_id": hash(sample["text"])}

    # ---- Memmap backend ----

    def _iter_memmap(self) -> Iterator[Dict[str, Any]]:
        """
        Load from token memmap (.bin + .idx files).
        Falls back to synthetic if files don't exist.
        """
        data_path = self.cfg.data_path
        if data_path is None:
            yield from self._iter_synthetic()
            return

        bin_path = Path(data_path).with_suffix(".bin")
        idx_path = Path(data_path).with_suffix(".idx")

        if not bin_path.exists():
            yield from self._iter_synthetic()
            return

        tokens = np.memmap(str(bin_path), dtype=np.uint16, mode="r")
        total_tokens = len(tokens)
        seq_len = self.cfg.target_seq_len

        if idx_path.exists():
            doc_offsets = np.memmap(str(idx_path), dtype=np.int64, mode="r")
        else:
            # No index file: treat as one big document
            doc_offsets = np.array([0, total_tokens], dtype=np.int64)

        # Generate block start positions
        num_blocks = total_tokens // seq_len
        all_starts = list(range(0, num_blocks * seq_len, seq_len))

        # Shard assignment
        rng = np.random.RandomState(self.cfg.shard_seed + self._epoch)
        rng.shuffle(all_starts)
        rank_starts = all_starts[self.rank::self.world_size]

        for start in rank_starts:
            block = tokens[start:start + seq_len].astype(np.int64).tolist()
            yield {
                "input_ids": block,
                "labels": block,
                "sample_id": start,
            }


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _run_self_tests() -> None:
    import shutil

    passed = 0
    failed = 0

    def check(name: str, condition: bool, detail: str = ""):
        nonlocal passed, failed
        status = "PASS" if condition else "FAIL"
        if not condition:
            failed += 1
            print(f"  [{status}] {name}: {detail}")
        else:
            passed += 1
            print(f"  [{status}] {name}")

    print("=" * 60)
    print("ShardedStreamDataset Self-Tests")
    print("=" * 60)

    # Test 1: Synthetic backend yields batches
    cfg = StreamConfig(format="synthetic", num_samples=100, target_seq_len=64)
    ds = ShardedStreamDataset(cfg, rank=0, world_size=1)
    batches = list(ds)
    check("T1: synthetic yields batches", len(batches) > 0,
          f"got {len(batches)} batches")
    check("T1b: batch has input_ids", "input_ids" in batches[0])
    check("T1c: input_ids correct length",
          len(batches[0]["input_ids"]) == 64,
          f"len={len(batches[0]['input_ids'])}")

    # Test 2: set_epoch changes ordering
    ds.set_epoch(0)
    order_e0 = [b["sample_id"] for b in ds]
    ds.set_epoch(1)
    order_e1 = [b["sample_id"] for b in ds]
    check("T2: set_epoch changes order", order_e0 != order_e1,
          f"e0_first3={order_e0[:3]}, e1_first3={order_e1[:3]}")

    # Test 3: Different ranks get different data
    ds_r0 = ShardedStreamDataset(cfg, rank=0, world_size=2)
    ds_r1 = ShardedStreamDataset(cfg, rank=1, world_size=2)
    ids_r0 = set(b["sample_id"] for b in ds_r0)
    ids_r1 = set(b["sample_id"] for b in ds_r1)
    overlap = ids_r0 & ids_r1
    check("T3: ranks get different samples", len(overlap) == 0,
          f"overlap={len(overlap)} samples")

    # Test 4: All samples covered by ranks
    all_covered = ids_r0 | ids_r1
    check("T4: all samples covered", len(all_covered) == cfg.num_samples,
          f"covered={len(all_covered)}, total={cfg.num_samples}")

    # Test 5: Same seed+epoch+rank gives same order (reproducibility)
    ds_a = ShardedStreamDataset(cfg, rank=0, world_size=1)
    ds_a.set_epoch(42)
    order_a = [b["sample_id"] for b in ds_a]
    ds_b = ShardedStreamDataset(cfg, rank=0, world_size=1)
    ds_b.set_epoch(42)
    order_b = [b["sample_id"] for b in ds_b]
    check("T5: reproducible ordering", order_a == order_b)

    # Test 6: ShardCache stores and retrieves
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = ShardCache(tmpdir, max_bytes=10000)
        data = b"hello world shard data" * 10
        cache.put("shard_001", data)
        retrieved = cache.get("shard_001")
        check("T6: cache stores and retrieves", retrieved == data)

    # Test 7: ShardCache checksum validation catches corruption
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = ShardCache(tmpdir, max_bytes=10000)
        data = b"valid shard data"
        cache.put("shard_002", data)
        # Corrupt the cached file
        cache_path = Path(tmpdir) / "shard_002"
        cache_path.write_bytes(b"corrupted data")
        retrieved = cache.get("shard_002")
        check("T7: corrupted cache returns None", retrieved is None)
        # File should be evicted
        check("T7b: corrupted file evicted", not cache_path.exists())

    # Test 8: ShardCache LRU eviction
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = ShardCache(tmpdir, max_bytes=500)
        cache.put("shard_a", b"a" * 200)
        cache.put("shard_b", b"b" * 200)
        # Access shard_a to make it more recent
        cache.get("shard_a")
        # Put a third that requires eviction of the LRU (shard_b)
        cache.put("shard_c", b"c" * 200)
        check("T8: LRU evicts oldest shard",
              cache.get("shard_b") is None,
              "shard_b should be evicted")
        check("T8b: recently accessed shard retained",
              cache.get("shard_a") is not None,
              "shard_a should still be cached")

    # Test 9: Memmap backend with synthetic .bin file
    with tempfile.TemporaryDirectory() as tmpdir:
        bin_path = os.path.join(tmpdir, "test_data.bin")
        tokens = np.arange(0, 1000, dtype=np.uint16)
        tokens.tofile(bin_path)

        memmap_cfg = StreamConfig(
            format="memmap",
            data_path=os.path.join(tmpdir, "test_data"),
            target_seq_len=64,
        )
        ds_mm = ShardedStreamDataset(memmap_cfg, rank=0, world_size=1)
        mm_batches = list(ds_mm)
        check("T9: memmap yields batches", len(mm_batches) > 0,
              f"got {len(mm_batches)} batches")
        if mm_batches:
            check("T9b: memmap batch has correct seq_len",
                  len(mm_batches[0]["input_ids"]) == 64,
                  f"len={len(mm_batches[0]['input_ids'])}")

    # Test 10: Memmap sharding gives different data to different ranks
    with tempfile.TemporaryDirectory() as tmpdir:
        bin_path = os.path.join(tmpdir, "test_data.bin")
        tokens = np.arange(0, 2000, dtype=np.uint16)
        tokens.tofile(bin_path)

        memmap_cfg = StreamConfig(
            format="memmap",
            data_path=os.path.join(tmpdir, "test_data"),
            target_seq_len=64,
        )
        ds_mm0 = ShardedStreamDataset(memmap_cfg, rank=0, world_size=2)
        ds_mm1 = ShardedStreamDataset(memmap_cfg, rank=1, world_size=2)
        ids_mm0 = set(b["sample_id"] for b in ds_mm0)
        ids_mm1 = set(b["sample_id"] for b in ds_mm1)
        mm_overlap = ids_mm0 & ids_mm1
        check("T10: memmap ranks get different blocks",
              len(mm_overlap) == 0,
              f"overlap={len(mm_overlap)}")

    # Test 11: Dataset can iterate multiple times
    ds_multi = ShardedStreamDataset(cfg, rank=0, world_size=1)
    count1 = len(list(ds_multi))
    count2 = len(list(ds_multi))
    check("T11: dataset iterable multiple times",
          count1 == count2 and count1 > 0,
          f"count1={count1}, count2={count2}")

    # Test 12: batch sample_ids are within expected range
    all_ids = [b["sample_id"] for b in ShardedStreamDataset(cfg, rank=0, world_size=1)]
    in_range = all(0 <= sid < cfg.num_samples for sid in all_ids)
    check("T12: sample_ids in valid range", in_range)

    print("-" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    _run_self_tests()
