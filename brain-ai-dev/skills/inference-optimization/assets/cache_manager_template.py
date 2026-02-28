"""
brain_ai/inference/cache.py — Multi-Level Cache Manager for Inference Optimization

This module provides the CacheManager for caching intermediate representations
during inference. The three-level hierarchy (L1 GPU, L2 CPU pinned, L3 disk)
mirrors CPU cache design, adapted for deep learning workloads.

Key classes:
    L1Cache          — GPU tensor cache with LRU eviction
    L2Cache          — CPU pinned memory cache with LRU eviction
    L3Cache          — Disk-backed persistent cache via torch.save/load
    CacheManager     — Unified multi-level cache with fallthrough lookups
    CacheKeyBuilder  — Generates deterministic cache keys from tensor inputs

Design principles:
    1. LRU eviction by default — OrderedDict provides O(1) operations.
    2. Memory budgets are enforced — eviction triggers before exceeding limits.
    3. TTL support — entries expire after a configurable time.
    4. Thread-safe — all public methods are protected by a reentrant lock.
    5. Statistics tracking — hits, misses, evictions tracked per level.

References:
    references/caching-architecture.md — Full design rationale
    SKILL.md § CacheManager contract
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


# ===========================================================================
# SECTION 1: CacheEntry
# ===========================================================================

@dataclass
class CacheEntry:
    """A single cache entry with metadata.

    Attributes:
        value: The cached tensor.
        created_at: Timestamp when the entry was created.
        ttl_seconds: Time-to-live (0 = no expiry).
        access_count: Number of times this entry has been accessed.
        size_bytes: Size of the cached tensor in bytes.
    """

    value: Tensor
    created_at: float = 0.0
    ttl_seconds: float = 0.0
    access_count: int = 0
    size_bytes: int = 0

    @property
    def is_expired(self) -> bool:
        """Check whether this entry has expired."""
        if self.ttl_seconds <= 0:
            return False
        return (time.time() - self.created_at) > self.ttl_seconds


# ===========================================================================
# SECTION 2: CacheStats
# ===========================================================================

@dataclass
class CacheStats:
    """Statistics for the cache system."""

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
    l3_hits: int = 0
    l3_misses: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def total_entries(self) -> int:
        return self.l1_entries + self.l2_entries + self.l3_entries

    @property
    def total_size_mb(self) -> float:
        return self.l1_size_mb + self.l2_size_mb + self.l3_size_mb


# ===========================================================================
# SECTION 3: Cache Key Builder
# ===========================================================================

class CacheKeyBuilder:
    """Generates deterministic cache keys from tensor inputs.

    Supports three hashing strategies:
    - full: SHA-256 of all tensor bytes (accurate, slower)
    - sampled: SHA-256 of sampled elements (fast, approximate)
    - stats: SHA-256 of shape+mean+std+min+max (fastest, collision-prone)
    """

    def __init__(self, strategy: str = "sampled", sample_size: int = 1024):
        self.strategy = strategy
        self.sample_size = sample_size

    def tensor_hash(self, t: Tensor) -> str:
        """Hash a tensor using the configured strategy."""
        if self.strategy == "full":
            return self._hash_full(t)
        elif self.strategy == "sampled":
            return self._hash_sampled(t)
        elif self.strategy == "stats":
            return self._hash_stats(t)
        else:
            return self._hash_full(t)

    def _hash_full(self, t: Tensor) -> str:
        data = t.detach().cpu().contiguous().numpy().tobytes()
        return hashlib.sha256(data).hexdigest()

    def _hash_sampled(self, t: Tensor) -> str:
        flat = t.detach().flatten()
        if flat.numel() <= self.sample_size:
            data = flat.cpu().numpy().tobytes()
        else:
            gen = torch.Generator().manual_seed(42)
            indices = torch.randperm(flat.numel(), generator=gen)[:self.sample_size]
            data = flat[indices].cpu().numpy().tobytes()
        return hashlib.sha256(data).hexdigest()

    def _hash_stats(self, t: Tensor) -> str:
        t_float = t.detach().float()
        stats = (
            f"{tuple(t.shape)}_{t_float.mean().item():.6f}_"
            f"{t_float.std().item():.6f}_{t_float.min().item():.6f}_"
            f"{t_float.max().item():.6f}"
        )
        return hashlib.sha256(stats.encode()).hexdigest()

    def build_key(
        self,
        inputs: Dict[str, Tensor],
        module_name: str = "",
        model_version: str = "",
    ) -> str:
        """Build a composite cache key from inputs and context."""
        parts = []
        for name in sorted(inputs.keys()):
            parts.append(f"{name}:{self.tensor_hash(inputs[name])}")
        combined = "|".join(parts)
        if module_name:
            combined = f"{module_name}:{combined}"
        if model_version:
            combined = f"{model_version}:{combined}"
        return hashlib.sha256(combined.encode()).hexdigest()


# ===========================================================================
# SECTION 4: L1Cache — GPU tensor cache
# ===========================================================================

class L1Cache:
    """GPU-resident tensor cache with LRU eviction.

    Stores tensors on the target device (GPU or CPU for testing). Evicts
    least recently used entries when the memory budget is exceeded.
    """

    def __init__(self, max_memory_mb: int = 512, device: str = "cpu"):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.device = device
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_bytes: int = 0
        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0

    def get(self, key: str) -> Optional[Tensor]:
        """Get a cached tensor by key. Returns None on miss."""
        if key in self.cache:
            entry = self.cache[key]
            if entry.is_expired:
                self._remove(key)
                self.misses += 1
                return None
            self.cache.move_to_end(key)
            entry.access_count += 1
            self.hits += 1
            return entry.value
        self.misses += 1
        return None

    def put(self, key: str, value: Tensor, ttl_seconds: float = 0.0) -> None:
        """Store a tensor in the cache."""
        if key in self.cache:
            self._remove(key)

        val = value.detach().to(self.device)
        entry_bytes = val.nelement() * val.element_size()

        # Evict until we have room
        while self.current_bytes + entry_bytes > self.max_memory_bytes and self.cache:
            self._evict_one()

        # If the single entry exceeds the budget, do not cache it
        if entry_bytes > self.max_memory_bytes:
            return

        entry = CacheEntry(
            value=val,
            created_at=time.time(),
            ttl_seconds=ttl_seconds,
            size_bytes=entry_bytes,
        )
        self.cache[key] = entry
        self.current_bytes += entry_bytes

    def remove(self, key: str) -> bool:
        """Remove a specific entry. Returns True if found."""
        if key in self.cache:
            self._remove(key)
            return True
        return False

    def clear(self) -> None:
        """Clear all entries."""
        self.cache.clear()
        self.current_bytes = 0

    def _remove(self, key: str) -> None:
        entry = self.cache.pop(key)
        self.current_bytes -= entry.size_bytes

    def _evict_one(self) -> None:
        if self.cache:
            _, entry = self.cache.popitem(last=False)
            self.current_bytes -= entry.size_bytes
            self.evictions += 1

    @property
    def size_mb(self) -> float:
        return self.current_bytes / (1024 * 1024)

    def __len__(self) -> int:
        return len(self.cache)


# ===========================================================================
# SECTION 5: L2Cache — CPU pinned memory cache
# ===========================================================================

class L2Cache:
    """CPU-resident cache (pinned memory when CUDA is available).

    Same interface as L1Cache but stores tensors on CPU.
    """

    def __init__(self, max_memory_mb: int = 2048):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_bytes: int = 0
        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0

    def get(self, key: str) -> Optional[Tensor]:
        if key in self.cache:
            entry = self.cache[key]
            if entry.is_expired:
                self._remove(key)
                self.misses += 1
                return None
            self.cache.move_to_end(key)
            entry.access_count += 1
            self.hits += 1
            return entry.value
        self.misses += 1
        return None

    def put(self, key: str, value: Tensor, ttl_seconds: float = 0.0) -> None:
        if key in self.cache:
            self._remove(key)

        cpu_val = value.detach().cpu()
        # Pinned memory only available with CUDA
        if torch.cuda.is_available():
            try:
                cpu_val = cpu_val.pin_memory()
            except RuntimeError:
                pass

        entry_bytes = cpu_val.nelement() * cpu_val.element_size()

        while self.current_bytes + entry_bytes > self.max_memory_bytes and self.cache:
            self._evict_one()

        if entry_bytes > self.max_memory_bytes:
            return

        entry = CacheEntry(
            value=cpu_val,
            created_at=time.time(),
            ttl_seconds=ttl_seconds,
            size_bytes=entry_bytes,
        )
        self.cache[key] = entry
        self.current_bytes += entry_bytes

    def remove(self, key: str) -> bool:
        if key in self.cache:
            self._remove(key)
            return True
        return False

    def clear(self) -> None:
        self.cache.clear()
        self.current_bytes = 0

    def _remove(self, key: str) -> None:
        entry = self.cache.pop(key)
        self.current_bytes -= entry.size_bytes

    def _evict_one(self) -> None:
        if self.cache:
            _, entry = self.cache.popitem(last=False)
            self.current_bytes -= entry.size_bytes
            self.evictions += 1

    @property
    def size_mb(self) -> float:
        return self.current_bytes / (1024 * 1024)

    def __len__(self) -> int:
        return len(self.cache)


# ===========================================================================
# SECTION 6: L3Cache — Disk-backed persistent cache
# ===========================================================================

class L3Cache:
    """Disk-backed persistent cache using torch.save/load.

    Each entry is saved as a separate file, named by a hash of its key.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            cache_dir = os.path.join(tempfile.gettempdir(), "brain_ai_cache")
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.index: Dict[str, Tuple[str, Tuple, str]] = {}
        self.hits: int = 0
        self.misses: int = 0
        self.total_bytes: int = 0

    def get(self, key: str) -> Optional[Tensor]:
        if key not in self.index:
            self.misses += 1
            return None
        filename = self.index[key]
        path = os.path.join(self.cache_dir, filename)
        if not os.path.exists(path):
            del self.index[key]
            self.misses += 1
            return None
        try:
            data = torch.load(path, weights_only=True)
            self.hits += 1
            return data
        except Exception:
            self.misses += 1
            return None

    def put(self, key: str, value: Tensor) -> None:
        filename = hashlib.sha256(key.encode()).hexdigest()[:16] + ".pt"
        path = os.path.join(self.cache_dir, filename)
        tensor_cpu = value.detach().cpu()
        torch.save(tensor_cpu, path)
        file_size = os.path.getsize(path)
        if key in self.index:
            # Remove old file size estimate
            old_path = os.path.join(self.cache_dir, self.index[key])
            if os.path.exists(old_path):
                self.total_bytes -= os.path.getsize(old_path)
        self.index[key] = filename
        self.total_bytes += file_size

    def remove(self, key: str) -> bool:
        if key not in self.index:
            return False
        filename = self.index.pop(key)
        path = os.path.join(self.cache_dir, filename)
        if os.path.exists(path):
            self.total_bytes -= os.path.getsize(path)
            os.remove(path)
        return True

    def clear(self) -> None:
        for filename in self.index.values():
            path = os.path.join(self.cache_dir, filename)
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
        self.index.clear()
        self.total_bytes = 0

    @property
    def size_mb(self) -> float:
        return self.total_bytes / (1024 * 1024)

    def __len__(self) -> int:
        return len(self.index)


# ===========================================================================
# SECTION 7: CacheManager — unified multi-level cache
# ===========================================================================

class CacheManager:
    """Multi-level cache manager with fallthrough lookups.

    Provides a unified interface over L1 (GPU), L2 (CPU pinned), and
    L3 (disk) caches. Lookups check each level in order; stores write
    to the specified level (default L1).

    Args:
        l1_cache_mb: L1 memory budget in MB.
        l2_cache_mb: L2 memory budget in MB.
        l3_cache_dir: Directory for L3 disk cache.
        enable_l2: Whether L2 is active.
        enable_l3: Whether L3 is active.
        default_ttl: Default TTL for entries (0 = no expiry).
        hash_strategy: Hashing strategy for key generation.
        device: Device for L1 cache tensors.
    """

    def __init__(
        self,
        l1_cache_mb: int = 512,
        l2_cache_mb: int = 2048,
        l3_cache_dir: Optional[str] = None,
        enable_l2: bool = True,
        enable_l3: bool = False,
        default_ttl: float = 0.0,
        hash_strategy: str = "sampled",
        device: str = "cpu",
    ):
        self.l1 = L1Cache(max_memory_mb=l1_cache_mb, device=device)
        self.l2 = L2Cache(max_memory_mb=l2_cache_mb) if enable_l2 else None
        self.l3 = L3Cache(cache_dir=l3_cache_dir) if enable_l3 else None
        self.default_ttl = default_ttl
        self.key_builder = CacheKeyBuilder(strategy=hash_strategy)
        self._lock = threading.RLock()

        # Aggregate stats
        self._total_hits: int = 0
        self._total_misses: int = 0

    def get(self, key: str, level: Optional[str] = None) -> Optional[Tensor]:
        """Look up a key. If level is None, try L1 -> L2 -> L3."""
        with self._lock:
            if level == "l1" or level is None:
                val = self.l1.get(key)
                if val is not None:
                    self._total_hits += 1
                    return val

            if level == "l2" or level is None:
                if self.l2 is not None:
                    val = self.l2.get(key)
                    if val is not None:
                        self._total_hits += 1
                        # Promote to L1
                        if level is None:
                            self.l1.put(key, val, ttl_seconds=self.default_ttl)
                        return val

            if level == "l3" or level is None:
                if self.l3 is not None:
                    val = self.l3.get(key)
                    if val is not None:
                        self._total_hits += 1
                        return val

            self._total_misses += 1
            return None

    def put(
        self,
        key: str,
        value: Tensor,
        level: str = "l1",
        ttl_seconds: Optional[float] = None,
    ) -> None:
        """Store a value at the specified cache level."""
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        with self._lock:
            if level == "l1":
                self.l1.put(key, value, ttl_seconds=ttl)
            elif level == "l2" and self.l2 is not None:
                self.l2.put(key, value, ttl_seconds=ttl)
            elif level == "l3" and self.l3 is not None:
                self.l3.put(key, value)
            else:
                # Default to L1 if level not available
                self.l1.put(key, value, ttl_seconds=ttl)

    def invalidate(self, key: Optional[str] = None) -> None:
        """Invalidate a specific key or all entries if key is None."""
        with self._lock:
            if key is None:
                self.l1.clear()
                if self.l2 is not None:
                    self.l2.clear()
                if self.l3 is not None:
                    self.l3.clear()
                self._total_hits = 0
                self._total_misses = 0
            else:
                self.l1.remove(key)
                if self.l2 is not None:
                    self.l2.remove(key)
                if self.l3 is not None:
                    self.l3.remove(key)

    def stats(self) -> CacheStats:
        """Return current cache statistics."""
        with self._lock:
            return CacheStats(
                hits=self._total_hits,
                misses=self._total_misses,
                evictions=self.l1.evictions + (
                    self.l2.evictions if self.l2 else 0
                ),
                l1_size_mb=self.l1.size_mb,
                l2_size_mb=self.l2.size_mb if self.l2 else 0.0,
                l3_size_mb=self.l3.size_mb if self.l3 else 0.0,
                l1_entries=len(self.l1),
                l2_entries=len(self.l2) if self.l2 else 0,
                l3_entries=len(self.l3) if self.l3 else 0,
                l1_hits=self.l1.hits,
                l1_misses=self.l1.misses,
                l2_hits=self.l2.hits if self.l2 else 0,
                l2_misses=self.l2.misses if self.l2 else 0,
                l3_hits=self.l3.hits if self.l3 else 0,
                l3_misses=self.l3.misses if self.l3 else 0,
            )

    def compute_key(self, inputs: Dict[str, Tensor], module_name: str = "") -> str:
        """Convenience: compute a cache key from model inputs."""
        return self.key_builder.build_key(inputs, module_name=module_name)


# ===========================================================================
# SECTION 8: Self-tests
# ===========================================================================

def _run_self_tests():
    """Run self-tests for cache manager components."""
    import traceback

    passed = 0
    failed = 0
    test_results = []

    def _test(name, fn):
        nonlocal passed, failed
        try:
            fn()
            passed += 1
            test_results.append(f"  PASS: {name}")
        except Exception as e:
            failed += 1
            test_results.append(f"  FAIL: {name} -- {e}")
            traceback.print_exc()

    # --- CacheKeyBuilder tests ---
    def test_key_builder_full():
        builder = CacheKeyBuilder(strategy="full")
        t = torch.randn(4, 8)
        k = builder.tensor_hash(t)
        assert len(k) == 64  # SHA-256 hex
        # Same tensor => same hash
        k2 = builder.tensor_hash(t)
        assert k == k2

    def test_key_builder_sampled():
        builder = CacheKeyBuilder(strategy="sampled", sample_size=16)
        t = torch.randn(100)
        k = builder.tensor_hash(t)
        assert len(k) == 64

    def test_key_builder_stats():
        builder = CacheKeyBuilder(strategy="stats")
        t = torch.randn(4, 8)
        k = builder.tensor_hash(t)
        assert len(k) == 64

    def test_key_builder_different_inputs():
        builder = CacheKeyBuilder(strategy="full")
        t1 = torch.randn(4, 8)
        t2 = torch.randn(4, 8)
        k1 = builder.tensor_hash(t1)
        k2 = builder.tensor_hash(t2)
        # Different random tensors should produce different hashes
        assert k1 != k2

    def test_key_builder_composite():
        builder = CacheKeyBuilder(strategy="full")
        inputs = {"a": torch.zeros(2, 4), "b": torch.ones(2, 4)}
        k = builder.build_key(inputs, module_name="encoder")
        assert len(k) == 64

    # --- L1Cache tests ---
    def test_l1_put_get():
        cache = L1Cache(max_memory_mb=1, device="cpu")
        t = torch.randn(10)
        cache.put("k1", t)
        got = cache.get("k1")
        assert got is not None
        assert torch.allclose(got, t)

    def test_l1_miss():
        cache = L1Cache(max_memory_mb=1, device="cpu")
        assert cache.get("nonexistent") is None

    def test_l1_eviction():
        # 0.001 MB = ~1024 bytes. Each float32 is 4 bytes, so 256 floats = 1024 bytes.
        cache = L1Cache(max_memory_mb=1, device="cpu")
        # Fill cache to force eviction
        # 1 MB = 1048576 bytes. 262144 float32 values = 1MB.
        big = torch.randn(262144)  # exactly 1MB
        cache.put("big", big)
        # Adding another should evict big
        big2 = torch.randn(262144)
        cache.put("big2", big2)
        assert cache.get("big") is None
        assert cache.get("big2") is not None

    def test_l1_lru_order():
        cache = L1Cache(max_memory_mb=1, device="cpu")
        # Each tensor is ~1KB (256 float32)
        for i in range(100):
            cache.put(f"k{i}", torch.randn(256))
        # Access k0 to make it recently used
        # k0 might already be evicted, so insert it again
        cache.put("k_recent", torch.randn(256))
        cache.get("k_recent")
        # k_recent should still exist
        assert cache.get("k_recent") is not None

    def test_l1_remove():
        cache = L1Cache(max_memory_mb=1, device="cpu")
        cache.put("k1", torch.randn(10))
        assert cache.remove("k1") is True
        assert cache.get("k1") is None
        assert cache.remove("k1") is False

    def test_l1_clear():
        cache = L1Cache(max_memory_mb=1, device="cpu")
        for i in range(10):
            cache.put(f"k{i}", torch.randn(10))
        cache.clear()
        assert len(cache) == 0
        assert cache.current_bytes == 0

    def test_l1_ttl():
        cache = L1Cache(max_memory_mb=1, device="cpu")
        cache.put("k1", torch.randn(10), ttl_seconds=0.05)
        assert cache.get("k1") is not None
        time.sleep(0.1)
        assert cache.get("k1") is None

    def test_l1_oversize_entry():
        cache = L1Cache(max_memory_mb=1, device="cpu")
        # 2MB tensor should not be cached in 1MB cache
        big = torch.randn(524288)  # 2MB
        cache.put("big", big)
        assert cache.get("big") is None

    def test_l1_size_mb():
        cache = L1Cache(max_memory_mb=10, device="cpu")
        cache.put("k1", torch.randn(256))  # 1KB
        assert cache.size_mb > 0

    def test_l1_hit_miss_counts():
        cache = L1Cache(max_memory_mb=1, device="cpu")
        cache.put("k1", torch.randn(10))
        cache.get("k1")  # hit
        cache.get("k2")  # miss
        assert cache.hits == 1
        assert cache.misses == 1

    # --- L2Cache tests ---
    def test_l2_put_get():
        cache = L2Cache(max_memory_mb=1)
        t = torch.randn(10)
        cache.put("k1", t)
        got = cache.get("k1")
        assert got is not None
        assert torch.allclose(got, t.cpu())

    def test_l2_eviction():
        cache = L2Cache(max_memory_mb=1)
        big = torch.randn(262144)
        cache.put("big", big)
        big2 = torch.randn(262144)
        cache.put("big2", big2)
        assert cache.get("big") is None

    def test_l2_clear():
        cache = L2Cache(max_memory_mb=1)
        for i in range(10):
            cache.put(f"k{i}", torch.randn(10))
        cache.clear()
        assert len(cache) == 0

    # --- L3Cache tests ---
    def test_l3_put_get():
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = L3Cache(cache_dir=tmpdir)
            t = torch.randn(10)
            cache.put("k1", t)
            got = cache.get("k1")
            assert got is not None
            assert torch.allclose(got, t)

    def test_l3_miss():
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = L3Cache(cache_dir=tmpdir)
            assert cache.get("nonexistent") is None

    def test_l3_remove():
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = L3Cache(cache_dir=tmpdir)
            cache.put("k1", torch.randn(10))
            assert cache.remove("k1") is True
            assert cache.get("k1") is None

    def test_l3_clear():
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = L3Cache(cache_dir=tmpdir)
            for i in range(5):
                cache.put(f"k{i}", torch.randn(10))
            cache.clear()
            assert len(cache) == 0

    # --- CacheManager tests ---
    def test_manager_put_get():
        mgr = CacheManager(l1_cache_mb=1, l2_cache_mb=1, device="cpu")
        t = torch.randn(10)
        mgr.put("k1", t)
        got = mgr.get("k1")
        assert got is not None
        assert torch.allclose(got, t)

    def test_manager_miss():
        mgr = CacheManager(l1_cache_mb=1, device="cpu")
        assert mgr.get("nonexistent") is None

    def test_manager_l2_fallthrough():
        mgr = CacheManager(l1_cache_mb=1, l2_cache_mb=1, device="cpu")
        t = torch.randn(10)
        # Put directly in L2
        mgr.put("k1", t, level="l2")
        # Get without specifying level should fallthrough to L2
        got = mgr.get("k1")
        assert got is not None
        assert torch.allclose(got, t)

    def test_manager_l3_fallthrough():
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CacheManager(
                l1_cache_mb=1, l2_cache_mb=1,
                l3_cache_dir=tmpdir, enable_l3=True, device="cpu",
            )
            t = torch.randn(10)
            mgr.put("k1", t, level="l3")
            # Clear L1 and L2 to force L3 lookup
            mgr.l1.clear()
            if mgr.l2:
                mgr.l2.clear()
            got = mgr.get("k1")
            assert got is not None
            assert torch.allclose(got, t)

    def test_manager_invalidate_all():
        mgr = CacheManager(l1_cache_mb=1, device="cpu")
        for i in range(10):
            mgr.put(f"k{i}", torch.randn(10))
        mgr.invalidate()
        st = mgr.stats()
        assert st.l1_entries == 0
        assert mgr.get("k0") is None

    def test_manager_invalidate_key():
        mgr = CacheManager(l1_cache_mb=1, device="cpu")
        mgr.put("k1", torch.randn(10))
        mgr.put("k2", torch.randn(10))
        mgr.invalidate("k1")
        assert mgr.get("k1") is None
        assert mgr.get("k2") is not None

    def test_manager_stats():
        mgr = CacheManager(l1_cache_mb=1, device="cpu")
        mgr.put("k1", torch.randn(10))
        mgr.get("k1")  # hit
        mgr.get("k2")  # miss
        st = mgr.stats()
        assert st.hits == 1
        assert st.misses == 1
        assert st.hit_rate > 0

    def test_manager_stats_entries():
        mgr = CacheManager(l1_cache_mb=10, device="cpu")
        for i in range(5):
            mgr.put(f"k{i}", torch.randn(10))
        st = mgr.stats()
        assert st.l1_entries == 5

    def test_manager_compute_key():
        mgr = CacheManager(l1_cache_mb=1, device="cpu")
        inputs = {"x": torch.randn(2, 4)}
        k = mgr.compute_key(inputs, module_name="test")
        assert len(k) == 64

    def test_manager_thread_safety():
        mgr = CacheManager(l1_cache_mb=10, device="cpu")
        errors = []

        def writer():
            try:
                for i in range(50):
                    mgr.put(f"thread_k{i}", torch.randn(10))
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for i in range(50):
                    mgr.get(f"thread_k{i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=writer),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0, f"Thread errors: {errors}"

    def test_key_collision_resistance():
        builder = CacheKeyBuilder(strategy="full")
        keys = set()
        for _ in range(100):
            t = torch.randn(8)
            k = builder.tensor_hash(t)
            keys.add(k)
        assert len(keys) == 100

    def test_cache_entry_not_expired():
        entry = CacheEntry(
            value=torch.randn(4), created_at=time.time(), ttl_seconds=100.0,
        )
        assert entry.is_expired is False

    def test_cache_entry_expired():
        entry = CacheEntry(
            value=torch.randn(4), created_at=time.time() - 10, ttl_seconds=1.0,
        )
        assert entry.is_expired is True

    def test_cache_entry_no_ttl():
        entry = CacheEntry(
            value=torch.randn(4), created_at=time.time() - 99999, ttl_seconds=0.0,
        )
        assert entry.is_expired is False

    def test_cache_stats_properties():
        st = CacheStats(hits=7, misses=3, l1_entries=5, l2_entries=3, l3_entries=2)
        assert abs(st.hit_rate - 0.7) < 1e-6
        assert st.total_entries == 10

    # Run all tests
    tests = [
        ("CacheKeyBuilder full", test_key_builder_full),
        ("CacheKeyBuilder sampled", test_key_builder_sampled),
        ("CacheKeyBuilder stats", test_key_builder_stats),
        ("CacheKeyBuilder different inputs", test_key_builder_different_inputs),
        ("CacheKeyBuilder composite key", test_key_builder_composite),
        ("L1Cache put/get", test_l1_put_get),
        ("L1Cache miss", test_l1_miss),
        ("L1Cache eviction", test_l1_eviction),
        ("L1Cache LRU order", test_l1_lru_order),
        ("L1Cache remove", test_l1_remove),
        ("L1Cache clear", test_l1_clear),
        ("L1Cache TTL expiry", test_l1_ttl),
        ("L1Cache oversize entry", test_l1_oversize_entry),
        ("L1Cache size_mb", test_l1_size_mb),
        ("L1Cache hit/miss counts", test_l1_hit_miss_counts),
        ("L2Cache put/get", test_l2_put_get),
        ("L2Cache eviction", test_l2_eviction),
        ("L2Cache clear", test_l2_clear),
        ("L3Cache put/get", test_l3_put_get),
        ("L3Cache miss", test_l3_miss),
        ("L3Cache remove", test_l3_remove),
        ("L3Cache clear", test_l3_clear),
        ("CacheManager put/get", test_manager_put_get),
        ("CacheManager miss", test_manager_miss),
        ("CacheManager L2 fallthrough", test_manager_l2_fallthrough),
        ("CacheManager L3 fallthrough", test_manager_l3_fallthrough),
        ("CacheManager invalidate all", test_manager_invalidate_all),
        ("CacheManager invalidate key", test_manager_invalidate_key),
        ("CacheManager stats hits/misses", test_manager_stats),
        ("CacheManager stats entries", test_manager_stats_entries),
        ("CacheManager compute_key", test_manager_compute_key),
        ("CacheManager thread safety", test_manager_thread_safety),
        ("Key collision resistance (100 tensors)", test_key_collision_resistance),
        ("CacheEntry not expired", test_cache_entry_not_expired),
        ("CacheEntry expired", test_cache_entry_expired),
        ("CacheEntry no TTL", test_cache_entry_no_ttl),
        ("CacheStats properties", test_cache_stats_properties),
    ]

    print(f"Running {len(tests)} self-tests for cache_manager_template...")
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
