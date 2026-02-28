"""
Multi-Head Hashing, Offloadable Embeddings, and Prefetch Infrastructure for Engram.

This is a comprehensive, production-quality template implementing the hash-embedding
subsystem for the Engram Conditional Memory skill. It provides:

  1. MultiHeadHash (nn.Module) -- Deterministic multiplicative-XOR hashing across
     multiple N-gram orders with per-layer salt support.

  2. PrimeSizer -- Utility for generating deterministic prime table sizes.

  3. OffloadableEmbedding (nn.Module) -- Embedding table with optional CPU offload,
     async prefetch via dedicated CUDA stream, pinned buffers, and dtype casting.

  4. PrefetchPlan -- Precomputes hash IDs for all layers and schedules async
     prefetch N layers ahead to overlap PCIe transfer with compute.

  5. PinnedBufferPool -- Pre-allocated pinned host buffers for double-buffering.

  6. EmbeddingAggregator (nn.Module) -- Aggregates retrieved embeddings across
     hash heads via sum, mean, or concat-project.

  7. Self-tests (35+ tests) -- Validates determinism, prime sizing, offload
     correctness, prefetch synchronization, and integration contracts.

All hashing is performed in int64 dtype. No numpy is used in the main computation
path. The module handles both CPU and CUDA gracefully.

Reference: DeepSeek Engram paper; see SKILL.md and references/ for full context.
"""

from __future__ import annotations

import math
import sys
import time
import random
import tempfile
import io
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class HashConfig:
    """Configuration for multi-head hashing.

    Attributes:
        hash_fn: Hash function family identifier. Currently only 'mult_xor'.
        use_prime_sizes: Whether to auto-generate prime-modulus table sizes.
        base_table_size: Starting size from which primes are generated.
        num_heads_per_order: Number of independent hash heads per N-gram order.
        per_layer_salt: If True, apply a per-layer multiplicative salt so the
            same N-gram maps to different rows at different insertion layers.
        num_layers: Maximum number of insertion layers (for salt generation).
        seed: Master deterministic seed for multiplier and salt generation.
    """

    hash_fn: str = "mult_xor"
    use_prime_sizes: bool = True
    base_table_size: int = 131071
    num_heads_per_order: int = 2
    per_layer_salt: bool = True
    num_layers: int = 32
    seed: int = 42


@dataclass
class OffloadConfig:
    """Configuration for CPU offloading and async prefetch.

    Attributes:
        weights_on_cpu: If True, store embedding weights in CPU memory.
        use_async_prefetch: If True, enable async prefetch via CUDA stream.
        prefetch_ahead_layers: How many layers ahead to trigger prefetch.
        pin_memory: If True, use pinned (page-locked) host memory.
        storage_dtype: Dtype string for host storage ('float16', 'bfloat16', 'float32').
        compute_dtype: Dtype string for compute on device ('float32', 'bfloat16').
    """

    weights_on_cpu: bool = False
    use_async_prefetch: bool = False
    prefetch_ahead_layers: int = 2
    pin_memory: bool = True
    storage_dtype: str = "float16"
    compute_dtype: str = "float32"


# ---------------------------------------------------------------------------
# PrimeSizer utility
# ---------------------------------------------------------------------------

class PrimeSizer:
    """Utility for generating deterministic prime table sizes.

    Generates distinct primes near a base size using trial-division primality
    testing. Results are cached for reuse across calls with identical parameters.
    """

    _cache: Dict[Tuple[int, int, int], List[int]] = {}

    @staticmethod
    def is_prime(n: int) -> bool:
        """Test whether *n* is prime using trial division.

        Args:
            n: Integer to test.

        Returns:
            True if *n* is prime, False otherwise.
        """
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    @staticmethod
    def next_prime(n: int) -> int:
        """Find the smallest prime >= *n*.

        Args:
            n: Starting value (must be >= 2).

        Returns:
            The smallest prime number that is >= *n*.
        """
        if n <= 2:
            return 2
        candidate = n if n % 2 != 0 else n + 1
        while not PrimeSizer.is_prime(candidate):
            candidate += 2
        return candidate

    @staticmethod
    def generate_prime_sizes(
        base_size: int,
        num_heads: int,
        seed: int = 42,
    ) -> List[int]:
        """Generate *num_heads* distinct primes near *base_size*.

        Primes are deterministic given the same (base_size, num_heads, seed).
        Results are cached.

        The algorithm uses a seeded PRNG to add jitter to base_size, then
        finds the next prime for each jittered value. If duplicates arise,
        additional offsets are tried.

        Args:
            base_size: Approximate target size for each table.
            num_heads: Number of distinct primes to generate.
            seed: Deterministic seed for the jitter PRNG.

        Returns:
            Sorted list of *num_heads* distinct primes.
        """
        cache_key = (base_size, num_heads, seed)
        if cache_key in PrimeSizer._cache:
            return PrimeSizer._cache[cache_key]

        rng = random.Random(seed)
        primes: List[int] = []
        seen: set = set()

        # Generate candidates with increasing jitter until we have enough.
        attempts = 0
        max_attempts = num_heads * 50
        jitter_range = max(base_size // 10, 100)

        while len(primes) < num_heads and attempts < max_attempts:
            offset = rng.randint(-jitter_range, jitter_range)
            candidate_base = max(7, base_size + offset)
            p = PrimeSizer.next_prime(candidate_base)
            if p not in seen:
                primes.append(p)
                seen.add(p)
            attempts += 1

        # Fallback: if not enough found, step upward from the largest prime.
        if len(primes) < num_heads:
            last = max(primes) if primes else base_size
            while len(primes) < num_heads:
                last = PrimeSizer.next_prime(last + 2)
                if last not in seen:
                    primes.append(last)
                    seen.add(last)

        primes.sort()
        PrimeSizer._cache[cache_key] = primes
        return primes

    @staticmethod
    def clear_cache() -> None:
        """Clear the internal prime cache."""
        PrimeSizer._cache.clear()


# ---------------------------------------------------------------------------
# Multiplier generation helpers
# ---------------------------------------------------------------------------

def _generate_multipliers(
    n: int,
    seed: int,
    *,
    low: int = 1_000_000_007,
    high: int = 2_147_483_647,
) -> List[int]:
    """Generate *n* large-prime-ish multipliers from a seeded PRNG.

    These are used as per-position coefficients in the multiplicative-XOR hash.
    They are large odd numbers, deterministic from *seed*.

    Args:
        n: Number of multipliers to generate.
        seed: Deterministic seed.
        low: Lower bound for generated multiplier (inclusive).
        high: Upper bound for generated multiplier (inclusive).

    Returns:
        List of *n* int multipliers.
    """
    rng = random.Random(seed)
    mults: List[int] = []
    for _ in range(n):
        v = rng.randint(low, high)
        # Ensure odd -- makes modular arithmetic better behaved.
        if v % 2 == 0:
            v += 1
        mults.append(v)
    return mults


def _generate_layer_salts(
    num_layers: int,
    seed: int,
    *,
    low: int = 1_000_000_007,
    high: int = 2_147_483_647,
) -> List[int]:
    """Generate per-layer multiplicative salts.

    Salt is applied as ``h = (h * salt[layer_id]) % table_size`` to decorrelate
    collision patterns across insertion layers.

    Args:
        num_layers: Number of layers to generate salts for.
        seed: Deterministic seed.
        low: Lower bound (inclusive).
        high: Upper bound (inclusive).

    Returns:
        List of *num_layers* odd integer salts.
    """
    rng = random.Random(seed + 999_983)  # Offset so salts differ from multipliers.
    salts: List[int] = []
    for _ in range(num_layers):
        v = rng.randint(low, high)
        if v % 2 == 0:
            v += 1
        salts.append(v)
    return salts


# ---------------------------------------------------------------------------
# MultiHeadHash
# ---------------------------------------------------------------------------

class MultiHeadHash(nn.Module):
    """Multi-head deterministic hashing for N-gram to embedding-row mapping.

    For each N-gram order *n* in ``[2 .. max_ngram_order]``, this module creates
    ``num_heads_per_order`` independent hash heads. Each head has:

    * A prime *table_size* (auto-generated or from config).
    * A multiplier array (int64, length *n*) used in the multiplicative-XOR hash.
    * An optional per-layer salt array.

    All parameters are registered as buffers (non-trainable, saved in state_dict).

    Hash function (per head):
        ``h = ((t1*m1) ^ (t2*m2) ^ ... ^ (tn*mn)) % table_size``

    Per-layer salt (optional):
        ``h = (h * layer_salt[layer_id]) % table_size``

    All computation is in int64 -- no floats.

    Attributes:
        H_total: Total number of hash heads across all orders.
    """

    def __init__(
        self,
        config: HashConfig,
        max_ngram_order: int = 4,
    ) -> None:
        """Initialize MultiHeadHash.

        Args:
            config: Hash configuration dataclass.
            max_ngram_order: Maximum N-gram order (inclusive). Heads are created
                for orders ``[2, 3, ..., max_ngram_order]``.
        """
        super().__init__()
        self.config = config
        self.max_ngram_order = max_ngram_order
        self.num_heads_per_order = config.num_heads_per_order

        # Determine N-gram orders: [2 .. max_ngram_order].
        self.orders: List[int] = list(range(2, max_ngram_order + 1))
        if len(self.orders) == 0:
            raise ValueError(
                f"max_ngram_order must be >= 2, got {max_ngram_order}"
            )

        self.H_total: int = len(self.orders) * config.num_heads_per_order

        # --- Generate prime table sizes ----------------------------------
        total_heads = self.H_total
        if config.use_prime_sizes:
            prime_sizes = PrimeSizer.generate_prime_sizes(
                base_size=config.base_table_size,
                num_heads=total_heads,
                seed=config.seed,
            )
        else:
            prime_sizes = [config.base_table_size] * total_heads

        # Store table sizes as a buffer so they appear in state_dict.
        self.register_buffer(
            "table_sizes",
            torch.tensor(prime_sizes, dtype=torch.int64),
        )

        # --- Generate multipliers per head --------------------------------
        # We flatten heads in the order: for each N-gram order, for each head.
        # multipliers shape: (H_total, max_ngram_order)
        # For heads with order n < max_ngram_order, positions [n:] are unused
        # (but still stored for uniform shape).
        mult_matrix = torch.zeros(
            total_heads, max_ngram_order, dtype=torch.int64
        )

        head_idx = 0
        for order in self.orders:
            for h in range(config.num_heads_per_order):
                # Seed is unique per (master_seed, order, head).
                head_seed = config.seed * 1000 + order * 100 + h
                mults = _generate_multipliers(order, seed=head_seed)
                for pos in range(order):
                    mult_matrix[head_idx, pos] = mults[pos]
                head_idx += 1

        self.register_buffer("multipliers", mult_matrix)

        # --- Map each head -> its order (for masking unused positions) ----
        head_orders = []
        for order in self.orders:
            for _h in range(config.num_heads_per_order):
                head_orders.append(order)
        self.register_buffer(
            "head_orders",
            torch.tensor(head_orders, dtype=torch.int64),
        )

        # --- Per-layer salts (optional) -----------------------------------
        if config.per_layer_salt:
            # Shape: (num_layers, H_total) -- each head gets its own per-layer salt.
            salt_matrix = torch.zeros(
                config.num_layers, total_heads, dtype=torch.int64,
            )
            for hi in range(total_heads):
                salt_seed = config.seed * 10000 + hi * 7 + 31
                salts = _generate_layer_salts(
                    config.num_layers, seed=salt_seed,
                )
                for li in range(config.num_layers):
                    salt_matrix[li, hi] = salts[li]
            self.register_buffer("layer_salts", salt_matrix)
        else:
            self.layer_salts: Optional[torch.Tensor] = None

        # --- Precompute a position mask so we only use the first *order*
        #     multiplier positions for each head. Shape: (H_total, max_ngram_order).
        pos_mask = torch.zeros(total_heads, max_ngram_order, dtype=torch.int64)
        for hi in range(total_heads):
            order = head_orders[hi]
            pos_mask[hi, :order] = 1
        self.register_buffer("pos_mask", pos_mask)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        canonical_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_id: int = 0,
    ) -> torch.Tensor:
        """Compute multi-head hash IDs for an input sequence.

        For each position *t* and each head, extracts the suffix N-gram
        (with left-padding of zeros for positions near the start), computes
        the multiplicative-XOR hash, and optionally applies per-layer salt.

        Masked (padding) positions produce ``hash_id = 0``.

        Args:
            canonical_ids: ``(B, T)`` int tensor of canonical (compressed) token IDs.
            attention_mask: ``(B, T)`` float or bool tensor. Positions with value 0
                (or False) are treated as padding and will receive ``hash_id = 0``.
            layer_id: Index of the insertion layer (used for per-layer salt).

        Returns:
            ``(B, T, H_total)`` int64 tensor of hash IDs, one per head.
        """
        B, T = canonical_ids.shape
        device = canonical_ids.device
        dtype = torch.int64

        # Ensure canonical_ids are int64.
        ids = canonical_ids.to(dtype=dtype)

        # Left-pad with zeros for N-gram extraction.
        max_order = self.max_ngram_order
        padded = F.pad(ids, (max_order - 1, 0), value=0)  # (B, max_order-1+T)

        # Build the N-gram window tensor: (B, T, max_ngram_order).
        # Position [b, t, k] = padded[b, t + k]  for k in [0 .. max_order-1].
        # This gives suffix N-grams: for order n, we use positions
        # [max_order - n : max_order] which correspond to (x_{t-n+1}, ..., x_t).
        # We build windows aligned to the RIGHT so that [:, :, -1] is always x_t.
        windows = torch.stack(
            [padded[:, i: i + T] for i in range(max_order)],
            dim=-1,
        )  # (B, T, max_order)

        # Expand for broadcasting across heads:
        # windows: (B, T, 1, max_order)
        # multipliers: (1, 1, H_total, max_order)
        windows_exp = windows.unsqueeze(2)  # (B, T, 1, max_order)
        mults_exp = self.multipliers.unsqueeze(0).unsqueeze(0)  # (1, 1, H, max_order)

        # Multiply element-wise: (B, T, H, max_order)
        products = windows_exp * mults_exp

        # Mask out unused positions (where k < max_order - head_order) per head.
        # pos_mask: (1, 1, H, max_order)
        pmask = self.pos_mask.unsqueeze(0).unsqueeze(0)
        products = products * pmask

        # XOR-reduce across the N-gram dimension.
        # torch doesn't have a direct xor-reduce, so we iterate.
        # For efficiency on GPU, we use a tree-reduction approach.
        hash_vals = self._xor_reduce(products)  # (B, T, H)

        # Modulo by per-head table size: table_sizes is (H,).
        table_sizes = self.table_sizes.unsqueeze(0).unsqueeze(0)  # (1, 1, H)
        hash_vals = hash_vals % table_sizes

        # Apply per-layer salt if configured.
        if self.layer_salts is not None:
            clamped_layer = min(layer_id, self.layer_salts.shape[0] - 1)
            salts = self.layer_salts[clamped_layer].unsqueeze(0).unsqueeze(0)  # (1, 1, H)
            hash_vals = (hash_vals * salts) % table_sizes

        # Ensure non-negative (Python semantics: modulo is non-negative for
        # positive divisor, but let's be explicit).
        hash_vals = hash_vals.abs()

        # Apply attention mask: set padding positions to hash_id = 0.
        if attention_mask is not None:
            # attention_mask: (B, T), 1=valid, 0=padding.
            mask = attention_mask.to(dtype=dtype).unsqueeze(-1)  # (B, T, 1)
            hash_vals = hash_vals * mask

        return hash_vals

    @staticmethod
    def _xor_reduce(x: torch.Tensor) -> torch.Tensor:
        """XOR-reduce along the last dimension.

        Args:
            x: ``(..., D)`` int64 tensor.

        Returns:
            ``(...)`` int64 tensor obtained by XOR-folding along dim=-1.
        """
        D = x.shape[-1]
        if D == 0:
            return torch.zeros(
                x.shape[:-1], dtype=x.dtype, device=x.device,
            )
        result = x[..., 0]
        for i in range(1, D):
            result = result ^ x[..., i]
        return result

    # ------------------------------------------------------------------
    # Incremental / streaming hash for last position only
    # ------------------------------------------------------------------

    def hash_last_position(
        self,
        canonical_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_id: int = 0,
    ) -> torch.Tensor:
        """Compute hash IDs only for the last non-padding position.

        This is useful in autoregressive streaming mode where only one new
        token is appended at a time.

        Args:
            canonical_ids: ``(B, T)`` int tensor of canonical token IDs.
            attention_mask: ``(B, T)`` mask (optional).
            layer_id: Insertion layer index.

        Returns:
            ``(B, 1, H_total)`` int64 hash IDs for the last position.
        """
        B, T = canonical_ids.shape
        device = canonical_ids.device
        dtype = torch.int64
        ids = canonical_ids.to(dtype=dtype)

        max_order = self.max_ngram_order

        # Extract suffix of length max_order ending at position T-1.
        if T >= max_order:
            suffix = ids[:, T - max_order:]  # (B, max_order)
        else:
            # Need left-padding.
            pad_len = max_order - T
            suffix = F.pad(ids, (pad_len, 0), value=0)  # (B, max_order)

        # suffix shape: (B, max_order) -- this is the N-gram window for position T-1.
        # Expand: (B, 1, max_order)
        suffix = suffix.unsqueeze(1)

        # Multiply with multipliers: (1, 1, H, max_order)
        mults_exp = self.multipliers.unsqueeze(0).unsqueeze(0)
        suffix_exp = suffix.unsqueeze(2)  # (B, 1, 1, max_order)
        products = suffix_exp * mults_exp  # (B, 1, H, max_order)

        # Mask unused positions.
        pmask = self.pos_mask.unsqueeze(0).unsqueeze(0)
        products = products * pmask

        # XOR reduce.
        hash_vals = self._xor_reduce(products)  # (B, 1, H)

        # Modulo.
        table_sizes = self.table_sizes.unsqueeze(0).unsqueeze(0)
        hash_vals = hash_vals % table_sizes

        # Per-layer salt.
        if self.layer_salts is not None:
            clamped_layer = min(layer_id, self.layer_salts.shape[0] - 1)
            salts = self.layer_salts[clamped_layer].unsqueeze(0).unsqueeze(0)
            hash_vals = (hash_vals * salts) % table_sizes

        hash_vals = hash_vals.abs()

        # Mask if last position is padding.
        if attention_mask is not None:
            last_mask = attention_mask[:, -1:].to(dtype=dtype).unsqueeze(-1)
            hash_vals = hash_vals * last_mask

        return hash_vals

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_head_metadata(self) -> List[Dict[str, Any]]:
        """Return metadata for each hash head.

        Returns:
            List of dicts with keys: ``order``, ``head_idx``, ``table_size``,
            ``multipliers`` (as Python list of ints).
        """
        metadata: List[Dict[str, Any]] = []
        head_idx = 0
        for order in self.orders:
            for h in range(self.num_heads_per_order):
                info: Dict[str, Any] = {
                    "order": order,
                    "head_idx": head_idx,
                    "global_head_within_order": h,
                    "table_size": int(self.table_sizes[head_idx].item()),
                    "multipliers": self.multipliers[head_idx, :order].tolist(),
                }
                metadata.append(info)
                head_idx += 1
        return metadata

    def __repr__(self) -> str:
        return (
            f"MultiHeadHash(orders={self.orders}, "
            f"heads_per_order={self.num_heads_per_order}, "
            f"H_total={self.H_total}, "
            f"table_sizes={self.table_sizes.tolist()}, "
            f"per_layer_salt={self.layer_salts is not None})"
        )


# ---------------------------------------------------------------------------
# PinnedBufferPool
# ---------------------------------------------------------------------------

class PinnedBufferPool:
    """Pool of pre-allocated pinned (page-locked) host buffers.

    Pinned memory enables faster CPU-to-GPU transfers via DMA. This pool
    avoids repeated allocation/deallocation overhead by maintaining a set
    of reusable buffers.

    Supports double-buffering: two buffers can be acquired simultaneously
    so one can be filled while the other is being transferred.

    Note: Only functional when CUDA is available. Falls back to regular
    CPU tensors otherwise.
    """

    def __init__(
        self,
        num_buffers: int = 4,
        initial_size: int = 0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize the pinned buffer pool.

        Args:
            num_buffers: Maximum number of buffers to maintain.
            initial_size: Initial number of elements per buffer (0 = allocate lazily).
            dtype: Element dtype for the buffers.
        """
        self.num_buffers = num_buffers
        self.dtype = dtype
        self._can_pin = torch.cuda.is_available()

        # Pool: list of (buffer_tensor, is_in_use) pairs.
        self._pool: List[Tuple[torch.Tensor, bool]] = []

        if initial_size > 0:
            for _ in range(num_buffers):
                buf = self._allocate(initial_size)
                self._pool.append((buf, False))

    def _allocate(self, size: int) -> torch.Tensor:
        """Allocate a pinned (or regular) buffer.

        Args:
            size: Number of elements.

        Returns:
            A 1-D tensor of the given size on CPU.
        """
        buf = torch.empty(size, dtype=self.dtype)
        if self._can_pin:
            buf = buf.pin_memory()
        return buf

    def acquire(self, size: int) -> torch.Tensor:
        """Acquire a buffer of at least *size* elements.

        If a sufficiently large free buffer exists in the pool, it is reused
        (and sliced to *size*). Otherwise, a new buffer is allocated.

        Args:
            size: Minimum number of elements needed.

        Returns:
            A 1-D pinned tensor with at least *size* elements.
        """
        # Try to reuse an existing free buffer.
        for i, (buf, in_use) in enumerate(self._pool):
            if not in_use and buf.numel() >= size:
                self._pool[i] = (buf, True)
                return buf[:size]

        # No suitable buffer found -- allocate a new one.
        # Over-allocate by 25% to reduce future allocations.
        alloc_size = max(size, int(size * 1.25))
        buf = self._allocate(alloc_size)
        if len(self._pool) < self.num_buffers:
            self._pool.append((buf, True))
        else:
            # Pool is full; replace the smallest free buffer.
            replaced = False
            for i, (existing, in_use) in enumerate(self._pool):
                if not in_use and existing.numel() < alloc_size:
                    self._pool[i] = (buf, True)
                    replaced = True
                    break
            if not replaced:
                # All in use or all larger; just append temporarily.
                self._pool.append((buf, True))

        return buf[:size]

    def release(self, buffer: torch.Tensor) -> None:
        """Release a previously acquired buffer back to the pool.

        Args:
            buffer: The tensor returned by :meth:`acquire`. Must share storage
                with a pooled buffer.
        """
        data_ptr = buffer.data_ptr()
        for i, (buf, in_use) in enumerate(self._pool):
            if buf.data_ptr() == data_ptr and in_use:
                self._pool[i] = (buf, False)
                return
        # If the buffer is not in the pool (e.g., was never added), just ignore.

    @property
    def num_allocated(self) -> int:
        """Number of buffers currently allocated in the pool."""
        return len(self._pool)

    @property
    def num_in_use(self) -> int:
        """Number of buffers currently acquired (in use)."""
        return sum(1 for _, in_use in self._pool if in_use)

    def clear(self) -> None:
        """Free all buffers in the pool."""
        self._pool.clear()

    def __repr__(self) -> str:
        return (
            f"PinnedBufferPool(allocated={self.num_allocated}, "
            f"in_use={self.num_in_use}, dtype={self.dtype})"
        )


# ---------------------------------------------------------------------------
# OffloadableEmbedding
# ---------------------------------------------------------------------------

def _resolve_dtype(name: str) -> torch.dtype:
    """Convert a string dtype name to a torch.dtype.

    Args:
        name: One of 'float16', 'float32', 'bfloat16', 'float64'.

    Returns:
        Corresponding torch.dtype.

    Raises:
        ValueError: If the name is not recognized.
    """
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float64": torch.float64,
    }
    if name not in mapping:
        raise ValueError(
            f"Unknown dtype '{name}'. Choose from {list(mapping.keys())}."
        )
    return mapping[name]


class OffloadableEmbedding(nn.Module):
    """Embedding table with optional CPU offload and async prefetch.

    Two modes of operation:

    * **On-device** (``weights_on_cpu=False``): Standard ``nn.Embedding`` residing
      on the compute device. Lookups use ``F.embedding``.

    * **Offload** (``weights_on_cpu=True``): Weight tensor is stored on CPU
      (optionally in pinned memory and reduced precision). Lookups are either
      synchronous (gather from CPU, copy to device) or asynchronous via
      :meth:`prefetch` / :meth:`consume_prefetched` with a dedicated CUDA stream.

    The prefetch pathway coalesces repeated IDs via ``torch.unique`` to minimize
    PCIe transfer volume, then copies only unique rows asynchronously. A CUDA
    event is recorded after the transfer so the compute stream can synchronize
    precisely when the data is needed.

    Attributes:
        num_embeddings: Number of rows in the embedding table.
        embedding_dim: Dimensionality of each embedding vector.
        offload: Whether weights live on CPU.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        weights_on_cpu: bool = False,
        storage_dtype: str = "float16",
        pin_memory: bool = True,
        compute_dtype: str = "float32",
    ) -> None:
        """Initialize OffloadableEmbedding.

        Args:
            num_embeddings: Number of embedding rows.
            embedding_dim: Dimension of each embedding vector.
            weights_on_cpu: If True, store on CPU with optional pinning.
            storage_dtype: Dtype for the stored weight tensor (string).
            pin_memory: If True and weights_on_cpu, pin the host memory.
            compute_dtype: Dtype for the returned embeddings (string).
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.offload = weights_on_cpu
        self._storage_dtype = _resolve_dtype(storage_dtype)
        self._compute_dtype = _resolve_dtype(compute_dtype)
        self._pin_memory = pin_memory and torch.cuda.is_available()

        if weights_on_cpu:
            # Allocate on CPU in storage dtype.
            weight_data = torch.randn(
                num_embeddings, embedding_dim, dtype=self._storage_dtype,
            ) * 0.02
            if self._pin_memory:
                weight_data = weight_data.pin_memory()
            # Store as a non-parameter buffer so it lives on CPU.
            self.register_buffer("weight", weight_data)
            # Override requires_grad so gradient flows for training if desired.
            self.weight.requires_grad_(False)

            # Prefetch infrastructure.
            self._prefetch_stream: Optional[torch.cuda.Stream] = None
            self._prefetch_event: Optional[torch.cuda.Event] = None
            if torch.cuda.is_available():
                self._prefetch_stream = torch.cuda.Stream()
                self._prefetch_event = torch.cuda.Event()

            # Prefetch state: stored between prefetch() and consume_prefetched().
            self._prefetch_state: Optional[
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ] = None
        else:
            # Standard on-device embedding.
            self.embedding = nn.Embedding(num_embeddings, embedding_dim)
            nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
            self._prefetch_stream = None
            self._prefetch_event = None
            self._prefetch_state = None

    # ------------------------------------------------------------------
    # Lookup (synchronous)
    # ------------------------------------------------------------------

    def lookup(self, ids: torch.Tensor) -> torch.Tensor:
        """Synchronous embedding lookup.

        Args:
            ids: ``(...)`` int tensor of embedding indices.

        Returns:
            ``(..., embedding_dim)`` tensor in compute_dtype.
        """
        if not self.offload:
            emb = self.embedding(ids)
            return emb.to(self._compute_dtype)

        # Offload mode: gather from CPU, copy to device.
        original_shape = ids.shape
        device = ids.device

        # Bring IDs to CPU for indexing.
        cpu_ids = ids.detach().cpu().long()
        cpu_emb = self.weight[cpu_ids.view(-1)]  # (N, D) in storage dtype

        # Transfer to device.
        if device.type == "cuda":
            gpu_emb = cpu_emb.to(device=device, non_blocking=False)
        else:
            gpu_emb = cpu_emb

        # Cast to compute dtype and reshape.
        gpu_emb = gpu_emb.to(self._compute_dtype)
        return gpu_emb.view(*original_shape, self.embedding_dim)

    # ------------------------------------------------------------------
    # Prefetch (asynchronous)
    # ------------------------------------------------------------------

    def prefetch(
        self,
        ids: torch.Tensor,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """Asynchronously prefetch embeddings for *ids*.

        Coalesces repeated IDs via ``torch.unique`` to minimize PCIe traffic.
        The transfer happens on a dedicated CUDA stream so the compute stream
        can continue with other work.

        After calling this method, call :meth:`consume_prefetched` to retrieve
        the embeddings (it will synchronize automatically).

        Args:
            ids: ``(...)`` int tensor of embedding indices (on any device).
            stream: Optional override CUDA stream. If None, uses internal stream.
        """
        if not self.offload:
            # Nothing to prefetch for on-device embeddings.
            # Store IDs so consume_prefetched can do a direct lookup.
            self._prefetch_state = (ids, torch.tensor([0]), torch.tensor([0]))
            return

        if self._prefetch_stream is None:
            # No CUDA available -- fall back to storing IDs for sync lookup.
            self._prefetch_state = (ids, torch.tensor([0]), torch.tensor([0]))
            return

        use_stream = stream if stream is not None else self._prefetch_stream
        target_device = ids.device if ids.is_cuda else torch.device("cuda:0")

        # Coalesce: find unique IDs and the inverse mapping.
        flat_ids = ids.view(-1).long()
        unique_ids, inverse = torch.unique(flat_ids, return_inverse=True)

        with torch.cuda.stream(use_stream):
            # Gather unique rows from CPU table.
            cpu_unique = unique_ids.cpu()
            cpu_emb = self.weight[cpu_unique]  # (U, D) in storage dtype

            # Async copy to device.
            gpu_emb = cpu_emb.to(
                device=target_device,
                non_blocking=True,
            )

            # Record event after the copy completes on this stream.
            self._prefetch_event.record(use_stream)

        # Store state for consumption.
        self._prefetch_state = (gpu_emb, inverse, ids)

    def consume_prefetched(self) -> torch.Tensor:
        """Retrieve previously prefetched embeddings.

        Waits for the CUDA event to ensure the async copy is complete, then
        reconstructs the full embedding tensor via the inverse mapping.

        Returns:
            ``(..., embedding_dim)`` tensor in compute_dtype, with the same
            leading shape as the *ids* passed to :meth:`prefetch`.

        Raises:
            RuntimeError: If :meth:`prefetch` was not called beforehand.
        """
        if self._prefetch_state is None:
            raise RuntimeError(
                "consume_prefetched() called without a preceding prefetch(). "
                "Call prefetch(ids) first."
            )

        if not self.offload:
            # On-device mode: just do a direct lookup.
            ids = self._prefetch_state[0]
            self._prefetch_state = None
            return self.lookup(ids)

        gpu_emb, inverse, original_ids = self._prefetch_state

        if self._prefetch_event is not None and self._prefetch_event.query() is False:
            # Wait for the async copy to finish on the current stream.
            self._prefetch_event.synchronize()

        # Reconstruct full tensor via inverse mapping.
        full_emb = gpu_emb[inverse]  # (N, D)

        # Cast to compute dtype.
        full_emb = full_emb.to(self._compute_dtype)

        # Reshape to match original IDs shape.
        target_shape = list(original_ids.shape) + [self.embedding_dim]
        full_emb = full_emb.view(*target_shape)

        # Clear prefetch state.
        self._prefetch_state = None

        return full_emb

    @property
    def has_pending_prefetch(self) -> bool:
        """Whether there is a pending prefetch that has not been consumed."""
        return self._prefetch_state is not None

    # ------------------------------------------------------------------
    # Standard forward (convenience)
    # ------------------------------------------------------------------

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Standard forward pass -- synchronous lookup.

        This is equivalent to :meth:`lookup` and is provided for compatibility
        with ``nn.Module`` conventions.

        Args:
            ids: ``(...)`` int tensor of embedding indices.

        Returns:
            ``(..., embedding_dim)`` tensor in compute_dtype.
        """
        return self.lookup(ids)

    def __repr__(self) -> str:
        return (
            f"OffloadableEmbedding(num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, offload={self.offload}, "
            f"storage_dtype={self._storage_dtype}, "
            f"compute_dtype={self._compute_dtype})"
        )


# ---------------------------------------------------------------------------
# PrefetchPlan
# ---------------------------------------------------------------------------

class PrefetchPlan:
    """Precomputes hash IDs for all layers and schedules async prefetch.

    Given input IDs and a hash module, the plan precomputes hash IDs for
    every layer upfront. It then manages a schedule where prefetch for
    layer L is triggered ``prefetch_ahead`` layers before L, allowing
    PCIe transfer to overlap with compute on earlier layers.

    Typical usage::

        plan = PrefetchPlan(input_ids, layer_ids, hash_module, prefetch_ahead=2)
        for layer_id in layer_ids:
            plan.prefetch_for_layer(layer_id, embedding_module)
            # ... other compute ...
            emb = plan.get_embeddings_for_layer(layer_id, embedding_module)

    Attributes:
        prefetch_ahead: Number of layers ahead to trigger prefetch.
        num_layers: Total number of layers in the plan.
    """

    def __init__(
        self,
        input_ids: torch.Tensor,
        layer_ids: Sequence[int],
        hash_module: MultiHeadHash,
        prefetch_ahead: int = 2,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Initialize PrefetchPlan.

        Args:
            input_ids: ``(B, T)`` canonical token IDs.
            layer_ids: Sequence of layer indices to plan for.
            hash_module: The MultiHeadHash module for computing hash IDs.
            prefetch_ahead: How many layers ahead to start prefetch.
            attention_mask: Optional ``(B, T)`` mask.
        """
        self.prefetch_ahead = prefetch_ahead
        self.layer_ids = list(layer_ids)
        self.num_layers = len(self.layer_ids)

        # Precompute hash IDs for all layers.
        self._hash_ids: Dict[int, torch.Tensor] = {}
        with torch.no_grad():
            for lid in self.layer_ids:
                h = hash_module(
                    input_ids,
                    attention_mask=attention_mask,
                    layer_id=lid,
                )
                self._hash_ids[lid] = h

        # Build prefetch schedule: map trigger_layer_id -> target_layer_id.
        # When we're about to compute layer *trigger*, we start prefetching
        # for layer *target* = trigger + prefetch_ahead.
        self._schedule: Dict[int, int] = {}
        for i, lid in enumerate(self.layer_ids):
            target_idx = i + prefetch_ahead
            if target_idx < self.num_layers:
                target_lid = self.layer_ids[target_idx]
                self._schedule[lid] = target_lid

        # Track which layers have been prefetched.
        self._prefetched_layers: set = set()

        # Track which layers had their prefetch consumed.
        self._consumed_layers: set = set()

    def get_hash_ids(self, layer_id: int) -> torch.Tensor:
        """Get precomputed hash IDs for a specific layer.

        Args:
            layer_id: The layer index.

        Returns:
            ``(B, T, H_total)`` int64 hash IDs.
        """
        return self._hash_ids[layer_id]

    def prefetch_for_layer(
        self,
        layer_id: int,
        embedding_module: OffloadableEmbedding,
    ) -> bool:
        """Trigger prefetch if the schedule says to do so at this layer.

        Checks whether a future layer's embeddings should be prefetched now.
        If so, calls ``embedding_module.prefetch()`` with the precomputed hash
        IDs for the target layer.

        Args:
            layer_id: The current layer being computed.
            embedding_module: The OffloadableEmbedding to prefetch from.

        Returns:
            True if a prefetch was triggered, False otherwise.
        """
        if layer_id not in self._schedule:
            return False

        target_lid = self._schedule[layer_id]
        if target_lid in self._prefetched_layers:
            return False  # Already prefetched.

        hash_ids = self._hash_ids[target_lid]
        embedding_module.prefetch(hash_ids)
        self._prefetched_layers.add(target_lid)
        return True

    def get_embeddings_for_layer(
        self,
        layer_id: int,
        embedding_module: OffloadableEmbedding,
    ) -> torch.Tensor:
        """Retrieve embeddings for a layer.

        If the layer was prefetched, consumes the prefetched data. Otherwise,
        performs a synchronous lookup.

        Args:
            layer_id: The layer index.
            embedding_module: The OffloadableEmbedding to retrieve from.

        Returns:
            ``(B, T, H_total, embedding_dim)`` or reshaped tensor.
        """
        hash_ids = self._hash_ids[layer_id]

        if layer_id in self._prefetched_layers and layer_id not in self._consumed_layers:
            self._consumed_layers.add(layer_id)
            return embedding_module.consume_prefetched()
        else:
            return embedding_module.lookup(hash_ids)

    @property
    def schedule(self) -> Dict[int, int]:
        """The prefetch schedule mapping trigger_layer -> target_layer."""
        return dict(self._schedule)

    def __repr__(self) -> str:
        return (
            f"PrefetchPlan(num_layers={self.num_layers}, "
            f"prefetch_ahead={self.prefetch_ahead}, "
            f"schedule={self._schedule})"
        )


# ---------------------------------------------------------------------------
# EmbeddingAggregator
# ---------------------------------------------------------------------------

class EmbeddingAggregator(nn.Module):
    """Aggregates retrieved embeddings from multiple hash heads.

    After looking up embeddings for each of ``H_total`` hash heads, this
    module combines them into a single representation. Three aggregation
    modes are supported:

    * **sum**: Element-wise sum across heads. Output dim = D_emb.
    * **mean**: Element-wise mean across heads. Output dim = D_emb.
    * **concat_project**: Concatenate all heads, then project down via a
      linear layer. Output dim = D_out.

    Attributes:
        mode: Aggregation mode string.
        output_dim: Dimensionality of the aggregated output.
    """

    def __init__(
        self,
        num_heads: int,
        embedding_dim: int,
        mode: str = "sum",
        output_dim: Optional[int] = None,
    ) -> None:
        """Initialize EmbeddingAggregator.

        Args:
            num_heads: Total number of hash heads (H_total).
            embedding_dim: Dimensionality of each head's embedding (D_emb).
            mode: Aggregation mode: 'sum', 'mean', or 'concat_project'.
            output_dim: Output dimensionality for 'concat_project' mode.
                Required when mode='concat_project'.
        """
        super().__init__()
        self.mode = mode
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim

        if mode == "concat_project":
            if output_dim is None:
                raise ValueError(
                    "output_dim is required for 'concat_project' mode."
                )
            self.output_dim = output_dim
            self.projection = nn.Linear(
                num_heads * embedding_dim, output_dim, bias=False,
            )
        elif mode in ("sum", "mean"):
            self.output_dim = embedding_dim
            self.projection = None
        else:
            raise ValueError(
                f"Unknown aggregation mode '{mode}'. "
                f"Choose from 'sum', 'mean', 'concat_project'."
            )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Aggregate embeddings from multiple heads.

        Args:
            embeddings: ``(B, T, H_total, D_emb)`` tensor of per-head embeddings.

        Returns:
            ``(B, T, output_dim)`` aggregated tensor.
        """
        if self.mode == "sum":
            return embeddings.sum(dim=2)  # (B, T, D_emb)
        elif self.mode == "mean":
            return embeddings.mean(dim=2)  # (B, T, D_emb)
        elif self.mode == "concat_project":
            B, T, H, D = embeddings.shape
            flat = embeddings.view(B, T, H * D)  # (B, T, H*D)
            return self.projection(flat)  # (B, T, output_dim)
        else:
            raise RuntimeError(f"Invalid mode '{self.mode}'")

    def __repr__(self) -> str:
        return (
            f"EmbeddingAggregator(mode='{self.mode}', "
            f"num_heads={self.num_heads}, "
            f"embedding_dim={self.embedding_dim}, "
            f"output_dim={self.output_dim})"
        )


# ---------------------------------------------------------------------------
# Composite: HashEmbeddingRetriever
# ---------------------------------------------------------------------------

class HashEmbeddingRetriever(nn.Module):
    """End-to-end retriever combining hashing, embedding lookup, and aggregation.

    This module wires together :class:`MultiHeadHash`,
    :class:`OffloadableEmbedding`, and :class:`EmbeddingAggregator` into a
    single callable that goes from canonical token IDs to aggregated embeddings.

    Optionally supports :class:`PrefetchPlan` for multi-layer async prefetch.

    Attributes:
        hash_module: The MultiHeadHash instance.
        aggregator: The EmbeddingAggregator instance.
    """

    def __init__(
        self,
        hash_config: HashConfig,
        max_ngram_order: int = 4,
        embedding_dim: int = 256,
        aggregation_mode: str = "sum",
        aggregation_output_dim: Optional[int] = None,
        offload_config: Optional[OffloadConfig] = None,
    ) -> None:
        """Initialize HashEmbeddingRetriever.

        Args:
            hash_config: Configuration for multi-head hashing.
            max_ngram_order: Maximum N-gram order.
            embedding_dim: Dimension of each embedding vector (per head).
            aggregation_mode: 'sum', 'mean', or 'concat_project'.
            aggregation_output_dim: Output dim for 'concat_project'.
            offload_config: Optional CPU offload configuration.
        """
        super().__init__()
        self.hash_module = MultiHeadHash(hash_config, max_ngram_order)
        H_total = self.hash_module.H_total

        # Resolve offload settings.
        oc = offload_config or OffloadConfig()

        # Create one embedding table per head.
        self.embedding_tables = nn.ModuleList()
        for hi in range(H_total):
            table_size = int(self.hash_module.table_sizes[hi].item())
            emb = OffloadableEmbedding(
                num_embeddings=table_size,
                embedding_dim=embedding_dim,
                weights_on_cpu=oc.weights_on_cpu,
                storage_dtype=oc.storage_dtype,
                pin_memory=oc.pin_memory,
                compute_dtype=oc.compute_dtype,
            )
            self.embedding_tables.append(emb)

        # Aggregator.
        self.aggregator = EmbeddingAggregator(
            num_heads=H_total,
            embedding_dim=embedding_dim,
            mode=aggregation_mode,
            output_dim=aggregation_output_dim,
        )

        self._embedding_dim = embedding_dim

    def forward(
        self,
        canonical_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_id: int = 0,
    ) -> torch.Tensor:
        """Compute aggregated hash embeddings.

        Args:
            canonical_ids: ``(B, T)`` canonical token IDs.
            attention_mask: Optional ``(B, T)`` mask.
            layer_id: Insertion layer index for per-layer salt.

        Returns:
            ``(B, T, output_dim)`` aggregated embeddings.
        """
        # Step 1: hash.
        hash_ids = self.hash_module(
            canonical_ids,
            attention_mask=attention_mask,
            layer_id=layer_id,
        )  # (B, T, H_total)

        B, T, H = hash_ids.shape

        # Step 2: per-head embedding lookup.
        head_embeddings: List[torch.Tensor] = []
        for hi in range(H):
            head_ids = hash_ids[:, :, hi]  # (B, T)
            emb = self.embedding_tables[hi].lookup(head_ids)  # (B, T, D_emb)
            head_embeddings.append(emb)

        # Stack: (B, T, H, D_emb)
        stacked = torch.stack(head_embeddings, dim=2)

        # Step 3: aggregate.
        return self.aggregator(stacked)

    def create_prefetch_plan(
        self,
        canonical_ids: torch.Tensor,
        layer_ids: Sequence[int],
        attention_mask: Optional[torch.Tensor] = None,
        prefetch_ahead: int = 2,
    ) -> PrefetchPlan:
        """Create a prefetch plan for multi-layer inference.

        Args:
            canonical_ids: ``(B, T)`` canonical token IDs.
            layer_ids: Sequence of layer indices.
            attention_mask: Optional ``(B, T)`` mask.
            prefetch_ahead: Layers ahead to start prefetch.

        Returns:
            A :class:`PrefetchPlan` instance.
        """
        return PrefetchPlan(
            input_ids=canonical_ids,
            layer_ids=layer_ids,
            hash_module=self.hash_module,
            prefetch_ahead=prefetch_ahead,
            attention_mask=attention_mask,
        )


# ---------------------------------------------------------------------------
# HashCollisionAnalyzer
# ---------------------------------------------------------------------------

class HashCollisionAnalyzer:
    """Diagnostic utility for analyzing hash collision statistics.

    Provides methods to measure unique ratio, collision rate, and distribution
    uniformity across hash heads and N-gram orders.
    """

    @staticmethod
    def compute_unique_ratio(hash_ids: torch.Tensor) -> float:
        """Compute the ratio of unique hash IDs to total IDs.

        Args:
            hash_ids: ``(B, T, H)`` int64 hash IDs from MultiHeadHash.

        Returns:
            Ratio in [0, 1]. Higher is better (fewer collisions).
        """
        flat = hash_ids.view(-1)
        num_unique = torch.unique(flat).numel()
        num_total = flat.numel()
        if num_total == 0:
            return 1.0
        return num_unique / num_total

    @staticmethod
    def compute_per_head_unique_ratio(hash_ids: torch.Tensor) -> List[float]:
        """Compute unique ratio per head.

        Args:
            hash_ids: ``(B, T, H)`` int64 hash IDs.

        Returns:
            List of H unique ratios.
        """
        H = hash_ids.shape[2]
        ratios: List[float] = []
        for hi in range(H):
            head_ids = hash_ids[:, :, hi].view(-1)
            num_unique = torch.unique(head_ids).numel()
            num_total = head_ids.numel()
            ratio = num_unique / max(num_total, 1)
            ratios.append(ratio)
        return ratios

    @staticmethod
    def compute_bucket_utilization(
        hash_ids: torch.Tensor,
        table_sizes: torch.Tensor,
    ) -> List[float]:
        """Compute fraction of table buckets that are populated per head.

        Args:
            hash_ids: ``(B, T, H)`` int64 hash IDs.
            table_sizes: ``(H,)`` int64 table sizes.

        Returns:
            List of H utilization fractions in [0, 1].
        """
        H = hash_ids.shape[2]
        utilizations: List[float] = []
        for hi in range(H):
            head_ids = hash_ids[:, :, hi].view(-1)
            num_unique = torch.unique(head_ids).numel()
            tsize = int(table_sizes[hi].item())
            utilizations.append(num_unique / max(tsize, 1))
        return utilizations


# ---------------------------------------------------------------------------
# StreamingHashCache
# ---------------------------------------------------------------------------

class StreamingHashCache:
    """Cache for incremental (autoregressive) hash computation.

    In streaming / autoregressive mode, tokens arrive one at a time. Rather
    than recomputing hashes for the entire sequence, we maintain a rolling
    buffer of the last ``max_ngram_order`` tokens and compute the hash only
    for the new position.

    This cache stores the token history and the last computed hash IDs.
    """

    def __init__(
        self,
        batch_size: int,
        max_ngram_order: int,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Initialize StreamingHashCache.

        Args:
            batch_size: Batch size B.
            max_ngram_order: Maximum N-gram order K.
            device: Device for buffers.
        """
        self.batch_size = batch_size
        self.max_ngram_order = max_ngram_order
        self.device = device

        # Rolling buffer of the last K tokens: (B, K).
        self._buffer = torch.zeros(
            batch_size, max_ngram_order, dtype=torch.int64, device=device,
        )
        self._position = 0  # Number of tokens seen so far.

    def push_token(self, token_ids: torch.Tensor) -> None:
        """Append a new token to the rolling buffer.

        Args:
            token_ids: ``(B,)`` int tensor of new token IDs.
        """
        # Shift left and insert at the end.
        self._buffer = torch.roll(self._buffer, shifts=-1, dims=1)
        self._buffer[:, -1] = token_ids.to(dtype=torch.int64)
        self._position += 1

    def get_ngram_window(self) -> torch.Tensor:
        """Get the current N-gram window for hashing.

        Returns:
            ``(B, max_ngram_order)`` int64 tensor. If fewer than
            max_ngram_order tokens have been seen, leading positions
            are zero (left-padded).
        """
        return self._buffer.clone()

    @property
    def position(self) -> int:
        """Number of tokens pushed so far."""
        return self._position

    def reset(self) -> None:
        """Reset the cache for a new sequence."""
        self._buffer.zero_()
        self._position = 0

    def __repr__(self) -> str:
        return (
            f"StreamingHashCache(batch_size={self.batch_size}, "
            f"max_ngram_order={self.max_ngram_order}, "
            f"position={self._position})"
        )


# ---------------------------------------------------------------------------
# HashEmbeddingCheckpointer
# ---------------------------------------------------------------------------

class HashEmbeddingCheckpointer:
    """Utility for saving and loading hash embedding state.

    Ensures that all deterministic buffers (multipliers, table sizes, salts)
    survive a save/load round-trip. Also validates config consistency on load.
    """

    @staticmethod
    def save(
        hash_module: MultiHeadHash,
        path: str,
        *,
        include_config: bool = True,
    ) -> None:
        """Save hash module state to a file.

        Args:
            hash_module: The MultiHeadHash module to save.
            path: File path for the checkpoint.
            include_config: If True, also save the HashConfig.
        """
        state: Dict[str, Any] = {
            "state_dict": hash_module.state_dict(),
            "max_ngram_order": hash_module.max_ngram_order,
            "orders": hash_module.orders,
            "num_heads_per_order": hash_module.num_heads_per_order,
            "H_total": hash_module.H_total,
        }
        if include_config:
            state["config"] = {
                "hash_fn": hash_module.config.hash_fn,
                "use_prime_sizes": hash_module.config.use_prime_sizes,
                "base_table_size": hash_module.config.base_table_size,
                "num_heads_per_order": hash_module.config.num_heads_per_order,
                "per_layer_salt": hash_module.config.per_layer_salt,
                "num_layers": hash_module.config.num_layers,
                "seed": hash_module.config.seed,
            }
        torch.save(state, path)

    @staticmethod
    def load(
        path: str,
        config: Optional[HashConfig] = None,
    ) -> MultiHeadHash:
        """Load hash module state from a file.

        Args:
            path: File path for the checkpoint.
            config: Optional HashConfig override. If None, uses the config
                saved in the checkpoint.

        Returns:
            Restored MultiHeadHash module.
        """
        state = torch.load(path, map_location="cpu", weights_only=False)

        if config is None:
            cfg_dict = state.get("config")
            if cfg_dict is None:
                raise ValueError(
                    "No config found in checkpoint and no config override provided."
                )
            config = HashConfig(**cfg_dict)

        max_ngram_order = state["max_ngram_order"]
        module = MultiHeadHash(config, max_ngram_order)
        module.load_state_dict(state["state_dict"])
        return module


# ---------------------------------------------------------------------------
# DTypeSafeOps
# ---------------------------------------------------------------------------

class DTypeSafeOps:
    """Collection of dtype-safe operations for hash computation.

    All operations ensure int64 computation to guarantee cross-platform
    determinism. These are utility methods used internally but exposed
    for testing and extension.
    """

    @staticmethod
    def safe_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Element-wise multiply ensuring int64.

        Args:
            a: Int tensor.
            b: Int tensor.

        Returns:
            a * b in int64.
        """
        return (a.to(torch.int64) * b.to(torch.int64))

    @staticmethod
    def safe_xor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Element-wise XOR ensuring int64.

        Args:
            a: Int tensor.
            b: Int tensor.

        Returns:
            a ^ b in int64.
        """
        return a.to(torch.int64) ^ b.to(torch.int64)

    @staticmethod
    def safe_modulo(a: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Element-wise modulo ensuring int64 and non-negative result.

        Args:
            a: Int tensor (dividend).
            m: Int tensor (modulus, must be > 0).

        Returns:
            a % m in int64, guaranteed non-negative.
        """
        result = a.to(torch.int64) % m.to(torch.int64)
        return result.abs()

    @staticmethod
    def ensure_int64(x: torch.Tensor) -> torch.Tensor:
        """Cast tensor to int64 if not already.

        Args:
            x: Any tensor.

        Returns:
            x in int64.
        """
        if x.dtype != torch.int64:
            return x.to(torch.int64)
        return x


# ---------------------------------------------------------------------------
# MultiOrderNgramExtractor
# ---------------------------------------------------------------------------

class MultiOrderNgramExtractor:
    """Extracts suffix N-grams across multiple orders from a token sequence.

    For each position *t* and each order *n*, the N-gram is the tuple
    ``(x_{t-n+1}, ..., x_t)`` with zero-padding for positions near the start
    where ``t - n + 1 < 0``.

    This class is used internally by MultiHeadHash but is also available
    for testing and extension.
    """

    @staticmethod
    def extract(
        token_ids: torch.Tensor,
        order: int,
    ) -> torch.Tensor:
        """Extract suffix N-grams of a given order.

        Args:
            token_ids: ``(B, T)`` int tensor of token IDs.
            order: The N-gram order (n).

        Returns:
            ``(B, T, n)`` int64 tensor where ``[b, t, :]`` is the suffix
            N-gram ending at position *t*, zero-padded if *t < n - 1*.
        """
        B, T = token_ids.shape
        ids = token_ids.to(torch.int64)

        # Left-pad with zeros.
        padded = F.pad(ids, (order - 1, 0), value=0)  # (B, order-1+T)

        # Sliding windows.
        ngrams = torch.stack(
            [padded[:, i: i + T] for i in range(order)],
            dim=-1,
        )  # (B, T, order)

        return ngrams

    @staticmethod
    def extract_multi_order(
        token_ids: torch.Tensor,
        orders: Sequence[int],
    ) -> Dict[int, torch.Tensor]:
        """Extract suffix N-grams for multiple orders.

        Args:
            token_ids: ``(B, T)`` int tensor.
            orders: Sequence of N-gram orders.

        Returns:
            Dict mapping order -> ``(B, T, order)`` tensor.
        """
        result: Dict[int, torch.Tensor] = {}
        for n in orders:
            result[n] = MultiOrderNgramExtractor.extract(token_ids, n)
        return result


# ---------------------------------------------------------------------------
# HashDistributionProfiler
# ---------------------------------------------------------------------------

class HashDistributionProfiler:
    """Profiling utility for analyzing hash distribution quality.

    Computes statistics like chi-squared uniformity, entropy, and
    max-bucket-load to help tune table sizes and hash parameters.
    """

    @staticmethod
    def chi_squared_uniformity(
        hash_ids: torch.Tensor,
        table_size: int,
    ) -> float:
        """Compute chi-squared statistic for hash distribution uniformity.

        A perfectly uniform hash would yield chi-squared close to (k-1) where
        k is the number of non-empty buckets.

        Args:
            hash_ids: 1-D int64 tensor of hash values.
            table_size: Size of the hash table.

        Returns:
            Chi-squared statistic. Lower (relative to table_size) is better.
        """
        flat = hash_ids.view(-1).long()
        N = flat.numel()
        if N == 0:
            return 0.0

        # Count occurrences of each bucket.
        counts = torch.zeros(table_size, dtype=torch.float64)
        for v in flat.tolist():
            if 0 <= v < table_size:
                counts[v] += 1

        expected = N / table_size
        chi_sq = ((counts - expected) ** 2 / max(expected, 1e-10)).sum().item()
        return chi_sq

    @staticmethod
    def hash_entropy(
        hash_ids: torch.Tensor,
        table_size: int,
    ) -> float:
        """Compute entropy of hash distribution.

        Maximum entropy for a uniform distribution over *table_size* buckets
        is ``log2(table_size)``.

        Args:
            hash_ids: 1-D int64 tensor.
            table_size: Size of the hash table.

        Returns:
            Entropy in bits.
        """
        flat = hash_ids.view(-1).long()
        N = flat.numel()
        if N == 0:
            return 0.0

        # Use bincount for efficiency.
        clamped = flat.clamp(0, table_size - 1)
        counts = torch.bincount(clamped, minlength=table_size).float()
        probs = counts / N
        # Avoid log(0).
        probs = probs[probs > 0]
        entropy = -(probs * probs.log2()).sum().item()
        return entropy

    @staticmethod
    def max_bucket_load(
        hash_ids: torch.Tensor,
        table_size: int,
    ) -> int:
        """Compute the maximum number of items in any single bucket.

        Args:
            hash_ids: 1-D int64 tensor.
            table_size: Size of the hash table.

        Returns:
            Maximum collision count.
        """
        flat = hash_ids.view(-1).long()
        if flat.numel() == 0:
            return 0
        clamped = flat.clamp(0, table_size - 1)
        counts = torch.bincount(clamped, minlength=table_size)
        return int(counts.max().item())


# ---------------------------------------------------------------------------
# SaltedHashVariant
# ---------------------------------------------------------------------------

class SaltedHashVariant:
    """Standalone utility for computing a single salted hash.

    This is a functional (non-module) implementation for use cases where
    a full MultiHeadHash module is not needed, e.g., in unit tests or
    one-off hash computations.
    """

    @staticmethod
    def hash_ngram(
        ngram: torch.Tensor,
        multipliers: torch.Tensor,
        table_size: int,
        salt: int = 1,
    ) -> torch.Tensor:
        """Hash a single N-gram (or batch thereof).

        Args:
            ngram: ``(B, n)`` or ``(n,)`` int tensor.
            multipliers: ``(n,)`` int64 multiplier coefficients.
            table_size: Prime modulus.
            salt: Multiplicative salt (default 1 = no salt).

        Returns:
            ``(B,)`` or scalar int64 hash value.
        """
        if ngram.dim() == 1:
            ngram = ngram.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        ngram = ngram.to(torch.int64)
        multipliers = multipliers.to(torch.int64)

        products = ngram * multipliers.unsqueeze(0)  # (B, n)

        # XOR reduce.
        h = products[:, 0]
        for i in range(1, products.shape[1]):
            h = h ^ products[:, i]

        h = (h * salt) % table_size
        h = h.abs()

        if squeeze:
            return h.squeeze(0)
        return h


# ---------------------------------------------------------------------------
# EmbeddingMemoryEstimator
# ---------------------------------------------------------------------------

class EmbeddingMemoryEstimator:
    """Estimates memory footprint of embedding tables.

    Useful for capacity planning and deciding whether to enable CPU offload.
    """

    @staticmethod
    def estimate_bytes(
        num_embeddings: int,
        embedding_dim: int,
        dtype: str = "float16",
        num_heads: int = 1,
    ) -> int:
        """Estimate total memory in bytes.

        Args:
            num_embeddings: Rows per table.
            embedding_dim: Columns per table.
            dtype: Storage dtype string.
            num_heads: Number of separate tables.

        Returns:
            Total bytes across all tables.
        """
        dtype_sizes = {
            "float16": 2,
            "bfloat16": 2,
            "float32": 4,
            "float64": 8,
        }
        element_size = dtype_sizes.get(dtype, 4)
        per_table = num_embeddings * embedding_dim * element_size
        return per_table * num_heads

    @staticmethod
    def estimate_human_readable(
        num_embeddings: int,
        embedding_dim: int,
        dtype: str = "float16",
        num_heads: int = 1,
    ) -> str:
        """Estimate memory with human-readable units.

        Args:
            num_embeddings: Rows per table.
            embedding_dim: Columns per table.
            dtype: Storage dtype string.
            num_heads: Number of separate tables.

        Returns:
            String like '1.23 GB'.
        """
        total_bytes = EmbeddingMemoryEstimator.estimate_bytes(
            num_embeddings, embedding_dim, dtype, num_heads,
        )
        if total_bytes >= 1 << 30:
            return f"{total_bytes / (1 << 30):.2f} GB"
        elif total_bytes >= 1 << 20:
            return f"{total_bytes / (1 << 20):.2f} MB"
        elif total_bytes >= 1 << 10:
            return f"{total_bytes / (1 << 10):.2f} KB"
        else:
            return f"{total_bytes} B"


# ---------------------------------------------------------------------------
# HashSeedExplorer
# ---------------------------------------------------------------------------

class HashSeedExplorer:
    """Explores different seeds to find optimal hash configurations.

    Given a sample dataset, evaluates multiple seeds and reports collision
    statistics for each, helping choose the best seed for production.
    """

    @staticmethod
    def evaluate_seed(
        canonical_ids: torch.Tensor,
        seed: int,
        max_ngram_order: int = 4,
        base_table_size: int = 131071,
        num_heads_per_order: int = 2,
    ) -> Dict[str, Any]:
        """Evaluate hash quality for a given seed.

        Args:
            canonical_ids: ``(B, T)`` sample data.
            seed: Seed to evaluate.
            max_ngram_order: Maximum N-gram order.
            base_table_size: Base table size.
            num_heads_per_order: Heads per order.

        Returns:
            Dict with 'seed', 'unique_ratio', 'per_head_ratios'.
        """
        config = HashConfig(
            seed=seed,
            base_table_size=base_table_size,
            num_heads_per_order=num_heads_per_order,
        )
        hasher = MultiHeadHash(config, max_ngram_order)
        hash_ids = hasher(canonical_ids)

        unique_ratio = HashCollisionAnalyzer.compute_unique_ratio(hash_ids)
        per_head = HashCollisionAnalyzer.compute_per_head_unique_ratio(hash_ids)

        return {
            "seed": seed,
            "unique_ratio": unique_ratio,
            "per_head_ratios": per_head,
        }

    @staticmethod
    def find_best_seed(
        canonical_ids: torch.Tensor,
        candidate_seeds: Sequence[int],
        max_ngram_order: int = 4,
        base_table_size: int = 131071,
        num_heads_per_order: int = 2,
    ) -> Dict[str, Any]:
        """Find the seed with the highest unique ratio.

        Args:
            canonical_ids: ``(B, T)`` sample data.
            candidate_seeds: Seeds to try.
            max_ngram_order: Maximum N-gram order.
            base_table_size: Base table size.
            num_heads_per_order: Heads per order.

        Returns:
            The evaluation dict for the best seed.
        """
        best: Optional[Dict[str, Any]] = None
        for s in candidate_seeds:
            result = HashSeedExplorer.evaluate_seed(
                canonical_ids, s, max_ngram_order,
                base_table_size, num_heads_per_order,
            )
            if best is None or result["unique_ratio"] > best["unique_ratio"]:
                best = result
        return best  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# GradientThroughHash
# ---------------------------------------------------------------------------

class GradientThroughHash(torch.autograd.Function):
    """Straight-through estimator for hash-based lookup.

    Since hashing is non-differentiable, this function provides a
    straight-through gradient pathway: the forward pass uses discrete
    hash IDs for lookup, while the backward pass passes gradients
    directly through the embedding output (bypassing the hash).

    This is useful when the embedding weights need to be trained.
    """

    @staticmethod
    def forward(
        ctx: Any,
        hash_ids: torch.Tensor,
        embedding_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Forward: gather embeddings using hash IDs.

        Args:
            ctx: Autograd context.
            hash_ids: ``(...)`` int tensor of indices.
            embedding_weight: ``(V, D)`` embedding weight matrix.

        Returns:
            ``(..., D)`` gathered embeddings.
        """
        ctx.save_for_backward(hash_ids, embedding_weight)
        flat_ids = hash_ids.view(-1).long()
        flat_ids = flat_ids.clamp(0, embedding_weight.shape[0] - 1)
        emb = embedding_weight[flat_ids]
        return emb.view(*hash_ids.shape, embedding_weight.shape[1])

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Backward: straight-through gradient to embedding weight.

        Args:
            ctx: Autograd context.
            grad_output: ``(..., D)`` gradient from downstream.

        Returns:
            (None for hash_ids, gradient for embedding_weight).
        """
        hash_ids, embedding_weight = ctx.saved_tensors
        flat_ids = hash_ids.view(-1).long()
        flat_ids = flat_ids.clamp(0, embedding_weight.shape[0] - 1)

        flat_grad = grad_output.reshape(-1, embedding_weight.shape[1])

        grad_weight = torch.zeros_like(embedding_weight)
        grad_weight.index_add_(0, flat_ids, flat_grad)

        return None, grad_weight


# ---------------------------------------------------------------------------
# BatchHasher
# ---------------------------------------------------------------------------

class BatchHasher:
    """Batch-optimized hash computation for large sequences.

    Splits very long sequences into chunks to avoid memory blowup during
    the N-gram window construction. Results are concatenated seamlessly.
    """

    @staticmethod
    def hash_chunked(
        hash_module: MultiHeadHash,
        canonical_ids: torch.Tensor,
        chunk_size: int = 2048,
        attention_mask: Optional[torch.Tensor] = None,
        layer_id: int = 0,
    ) -> torch.Tensor:
        """Compute hash IDs in chunks to bound memory usage.

        For sequences longer than *chunk_size*, the input is split into
        overlapping chunks (overlap = max_ngram_order - 1) so that N-gram
        windows at chunk boundaries are computed correctly.

        Args:
            hash_module: The MultiHeadHash module.
            canonical_ids: ``(B, T)`` canonical token IDs.
            chunk_size: Maximum chunk length.
            attention_mask: Optional ``(B, T)`` mask.
            layer_id: Layer index for salt.

        Returns:
            ``(B, T, H_total)`` hash IDs.
        """
        B, T = canonical_ids.shape
        if T <= chunk_size:
            return hash_module(canonical_ids, attention_mask, layer_id)

        max_order = hash_module.max_ngram_order
        overlap = max_order - 1
        results: List[torch.Tensor] = []

        start = 0
        while start < T:
            # Include overlap from previous chunk for N-gram context.
            context_start = max(0, start - overlap)
            end = min(start + chunk_size, T)

            chunk_ids = canonical_ids[:, context_start:end]
            chunk_mask = None
            if attention_mask is not None:
                chunk_mask = attention_mask[:, context_start:end]

            chunk_hash = hash_module(chunk_ids, chunk_mask, layer_id)

            # Only keep the non-overlap portion.
            trim_start = start - context_start
            results.append(chunk_hash[:, trim_start:, :])

            start = end

        return torch.cat(results, dim=1)


# ---------------------------------------------------------------------------
# EmbeddingInitializer
# ---------------------------------------------------------------------------

class EmbeddingInitializer:
    """Initialization strategies for hash embedding tables.

    Different initialization schemes can significantly affect training
    dynamics. This utility provides common initialization patterns.
    """

    @staticmethod
    def normal_(
        embedding: OffloadableEmbedding,
        mean: float = 0.0,
        std: float = 0.02,
    ) -> None:
        """Initialize with normal distribution.

        Args:
            embedding: The embedding module.
            mean: Mean of the normal distribution.
            std: Standard deviation.
        """
        if embedding.offload:
            nn.init.normal_(embedding.weight, mean=mean, std=std)
        else:
            nn.init.normal_(embedding.embedding.weight, mean=mean, std=std)

    @staticmethod
    def uniform_(
        embedding: OffloadableEmbedding,
        low: float = -0.1,
        high: float = 0.1,
    ) -> None:
        """Initialize with uniform distribution.

        Args:
            embedding: The embedding module.
            low: Lower bound.
            high: Upper bound.
        """
        if embedding.offload:
            nn.init.uniform_(embedding.weight, a=low, b=high)
        else:
            nn.init.uniform_(embedding.embedding.weight, a=low, b=high)

    @staticmethod
    def zero_(embedding: OffloadableEmbedding) -> None:
        """Initialize all embeddings to zero.

        Useful for zero-init residual connections.

        Args:
            embedding: The embedding module.
        """
        if embedding.offload:
            embedding.weight.zero_()
        else:
            embedding.embedding.weight.data.zero_()

    @staticmethod
    def kaiming_uniform_(
        embedding: OffloadableEmbedding,
        a: float = 0.0,
        mode: str = "fan_in",
    ) -> None:
        """Initialize with Kaiming uniform.

        Args:
            embedding: The embedding module.
            a: Negative slope for LeakyReLU (0 for ReLU).
            mode: 'fan_in' or 'fan_out'.
        """
        if embedding.offload:
            nn.init.kaiming_uniform_(embedding.weight, a=a, mode=mode)
        else:
            nn.init.kaiming_uniform_(
                embedding.embedding.weight, a=a, mode=mode,
            )


# ---------------------------------------------------------------------------
# ConfigValidator
# ---------------------------------------------------------------------------

class ConfigValidator:
    """Validates HashConfig and OffloadConfig for common misconfigurations."""

    @staticmethod
    def validate_hash_config(config: HashConfig) -> List[str]:
        """Check for potential issues in HashConfig.

        Args:
            config: The hash config to validate.

        Returns:
            List of warning/error message strings. Empty if all OK.
        """
        issues: List[str] = []

        if config.base_table_size < 100:
            issues.append(
                f"base_table_size={config.base_table_size} is very small. "
                f"Expect high collision rates."
            )

        if config.num_heads_per_order < 1:
            issues.append(
                f"num_heads_per_order={config.num_heads_per_order} must be >= 1."
            )

        if config.num_heads_per_order > 16:
            issues.append(
                f"num_heads_per_order={config.num_heads_per_order} is very large. "
                f"This may cause memory issues."
            )

        if not PrimeSizer.is_prime(config.base_table_size) and config.use_prime_sizes:
            # This is OK -- primes are generated from base_table_size.
            pass

        if config.num_layers < 1:
            issues.append(
                f"num_layers={config.num_layers} must be >= 1."
            )

        return issues

    @staticmethod
    def validate_offload_config(config: OffloadConfig) -> List[str]:
        """Check for potential issues in OffloadConfig.

        Args:
            config: The offload config to validate.

        Returns:
            List of warning/error message strings.
        """
        issues: List[str] = []

        if config.use_async_prefetch and not config.weights_on_cpu:
            issues.append(
                "use_async_prefetch=True but weights_on_cpu=False. "
                "Prefetch has no effect when weights are on device."
            )

        if config.prefetch_ahead_layers < 1:
            issues.append(
                f"prefetch_ahead_layers={config.prefetch_ahead_layers} must be >= 1."
            )

        valid_dtypes = {"float16", "bfloat16", "float32", "float64"}
        if config.storage_dtype not in valid_dtypes:
            issues.append(
                f"Unknown storage_dtype='{config.storage_dtype}'. "
                f"Valid: {valid_dtypes}"
            )
        if config.compute_dtype not in valid_dtypes:
            issues.append(
                f"Unknown compute_dtype='{config.compute_dtype}'. "
                f"Valid: {valid_dtypes}"
            )

        return issues


# ---------------------------------------------------------------------------
# SELF-TESTS (35+ tests)
# ---------------------------------------------------------------------------

def _run_self_tests() -> None:
    """Run all self-tests. Prints PASS/FAIL for each."""

    results: List[Tuple[str, bool, str]] = []

    def _test(name: str, passed: bool, detail: str = "") -> None:
        results.append((name, passed, detail))
        status = "PASS" if passed else "FAIL"
        msg = f"  [{status}] {name}"
        if detail:
            msg += f"  -- {detail}"
        print(msg)

    cuda_available = torch.cuda.is_available()
    print("=" * 72)
    print("Hash Embedding Template -- Self-Tests")
    print(f"  CUDA available: {cuda_available}")
    print("=" * 72)

    # --- 1. Hash determinism: same input + seed -> identical hash_ids ---
    print("\n--- Hash Determinism ---")
    config = HashConfig(seed=42, base_table_size=131071, num_heads_per_order=2)
    hasher = MultiHeadHash(config, max_ngram_order=4)
    ids = torch.randint(0, 10000, (2, 32), dtype=torch.int64)

    ref = hasher(ids, layer_id=0)
    all_match = True
    for run in range(10):
        out = hasher(ids, layer_id=0)
        if not torch.equal(ref, out):
            all_match = False
            break
    _test("1. Hash determinism (10 runs)", all_match)

    # --- 2. Hash determinism across dtypes: int32 vs int64 ---
    ids32 = ids.to(torch.int32)
    ids64 = ids.to(torch.int64)
    out32 = hasher(ids32, layer_id=0)
    out64 = hasher(ids64, layer_id=0)
    _test("2. Hash determinism int32 vs int64", torch.equal(out32, out64))

    # --- 3. CPU vs CUDA hash match ---
    if cuda_available:
        hasher_gpu = MultiHeadHash(config, max_ngram_order=4).cuda()
        ids_gpu = ids.cuda()
        out_cpu = hasher(ids, layer_id=0)
        out_gpu = hasher_gpu(ids_gpu, layer_id=0)
        _test("3. CPU vs CUDA hash match", torch.equal(out_cpu, out_gpu.cpu()))
    else:
        _test("3. CPU vs CUDA hash match", True, "SKIPPED (no CUDA)")

    # --- 4. Prime sizing: generated sizes are prime ---
    print("\n--- Prime Sizing ---")
    primes = PrimeSizer.generate_prime_sizes(131071, 6, seed=42)
    all_prime = all(PrimeSizer.is_prime(p) for p in primes)
    _test("4. Generated sizes are prime", all_prime, f"primes={primes}")

    # --- 5. Prime determinism: same seed -> same primes ---
    primes2 = PrimeSizer.generate_prime_sizes(131071, 6, seed=42)
    _test("5. Prime determinism (same seed)", primes == primes2)

    # --- 6. Prime uniqueness ---
    _test("6. All primes are distinct", len(set(primes)) == len(primes))

    # --- 7. next_prime correctness ---
    _test("7. next_prime(10) == 11", PrimeSizer.next_prime(10) == 11)
    _test("8. next_prime(13) == 13", PrimeSizer.next_prime(13) == 13)
    _test("9. next_prime(2) == 2", PrimeSizer.next_prime(2) == 2)

    # --- 10. Unique ratio sanity ---
    print("\n--- Collision Statistics ---")
    rand_ids = torch.randint(0, 50000, (4, 128), dtype=torch.int64)
    hash_out = hasher(rand_ids, layer_id=0)
    unique_ratio = HashCollisionAnalyzer.compute_unique_ratio(hash_out)
    _test(
        "10. Unique ratio > 0.8 for random data",
        unique_ratio > 0.8,
        f"unique_ratio={unique_ratio:.4f}",
    )

    # --- 11. Padding mask: masked positions -> hash_id=0 ---
    print("\n--- Masking ---")
    mask = torch.ones(2, 32, dtype=torch.float32)
    mask[:, -8:] = 0  # Last 8 positions are padding.
    masked_out = hasher(ids, attention_mask=mask, layer_id=0)
    padded_zeros = (masked_out[:, -8:, :] == 0).all().item()
    _test("11. Padding mask -> hash_id=0", padded_zeros)

    # --- 12. Non-masked positions are non-zero (at least some) ---
    non_pad_nonzero = (masked_out[:, :-8, :] != 0).any().item()
    _test("12. Non-masked positions have non-zero hashes", non_pad_nonzero)

    # --- 13. Per-layer salt: different layer_ids -> different hashes ---
    print("\n--- Per-Layer Salt ---")
    out_layer0 = hasher(ids, layer_id=0)
    out_layer1 = hasher(ids, layer_id=1)
    differ = not torch.equal(out_layer0, out_layer1)
    _test("13. Per-layer salt produces different hashes", differ)

    # --- 14. Same layer -> same hash ---
    out_layer0b = hasher(ids, layer_id=0)
    _test("14. Same layer -> same hash", torch.equal(out_layer0, out_layer0b))

    # --- 15. Without salt, different layers produce same hash ---
    config_no_salt = HashConfig(
        seed=42, base_table_size=131071,
        num_heads_per_order=2, per_layer_salt=False,
    )
    hasher_no_salt = MultiHeadHash(config_no_salt, max_ngram_order=4)
    out_ns0 = hasher_no_salt(ids, layer_id=0)
    out_ns1 = hasher_no_salt(ids, layer_id=1)
    _test("15. No salt -> same hash across layers", torch.equal(out_ns0, out_ns1))

    # --- 16-17. N-gram edge cases: positions near start ---
    print("\n--- N-gram Edge Cases ---")
    short_ids = torch.tensor([[100, 200]], dtype=torch.int64)  # T=2
    short_out = hasher(short_ids, layer_id=0)
    _test(
        "16. Short sequence (T=2) produces valid output",
        short_out.shape == (1, 2, hasher.H_total),
    )

    single_token = torch.tensor([[42]], dtype=torch.int64)  # T=1
    single_out = hasher(single_token, layer_id=0)
    _test(
        "17. Single token (T=1) produces valid output",
        single_out.shape == (1, 1, hasher.H_total),
    )

    # --- 18. Streaming vs full: last position matches ---
    print("\n--- Streaming vs Full ---")
    full_out = hasher(ids, layer_id=0)
    stream_out = hasher.hash_last_position(ids, layer_id=0)
    last_full = full_out[:, -1:, :]
    _test(
        "18. Streaming last-position matches full recompute",
        torch.equal(last_full, stream_out),
    )

    # --- 19. OffloadableEmbedding on-device: correct shape and dtype ---
    print("\n--- OffloadableEmbedding (on-device) ---")
    emb_on = OffloadableEmbedding(
        num_embeddings=1000, embedding_dim=64,
        weights_on_cpu=False, compute_dtype="float32",
    )
    lookup_ids = torch.randint(0, 1000, (2, 16))
    lookup_out = emb_on.lookup(lookup_ids)
    _test(
        "19. On-device lookup shape",
        lookup_out.shape == (2, 16, 64),
        f"shape={lookup_out.shape}",
    )
    _test(
        "20. On-device lookup dtype",
        lookup_out.dtype == torch.float32,
        f"dtype={lookup_out.dtype}",
    )

    # --- 21. OffloadableEmbedding offload: CPU lookup matches on-device ---
    print("\n--- OffloadableEmbedding (offload) ---")
    emb_off = OffloadableEmbedding(
        num_embeddings=1000, embedding_dim=64,
        weights_on_cpu=True, storage_dtype="float32",
        compute_dtype="float32", pin_memory=False,
    )
    # Copy weights so they match.
    with torch.no_grad():
        emb_off.weight.copy_(emb_on.embedding.weight.data)

    off_out = emb_off.lookup(lookup_ids)
    on_out = emb_on.lookup(lookup_ids)
    cos_sim = F.cosine_similarity(
        off_out.flatten().unsqueeze(0),
        on_out.flatten().unsqueeze(0),
    ).item()
    _test(
        "21. Offload lookup matches on-device (cosine)",
        cos_sim > 0.9999,
        f"cosine={cos_sim:.6f}",
    )

    # --- 22. Offload shape and dtype ---
    _test("22. Offload lookup shape", off_out.shape == (2, 16, 64))
    _test("23. Offload lookup dtype", off_out.dtype == torch.float32)

    # --- 24-25. Prefetch/consume cycle (CUDA) ---
    print("\n--- Prefetch/Consume ---")
    if cuda_available:
        emb_cuda_off = OffloadableEmbedding(
            num_embeddings=1000, embedding_dim=64,
            weights_on_cpu=True, storage_dtype="float32",
            compute_dtype="float32", pin_memory=True,
        )
        cuda_ids = torch.randint(0, 1000, (2, 16)).cuda()

        no_deadlock = True
        for cycle in range(20):
            try:
                emb_cuda_off.prefetch(cuda_ids)
                result = emb_cuda_off.consume_prefetched()
                if result.shape != (2, 16, 64):
                    no_deadlock = False
                    break
            except Exception as e:
                no_deadlock = False
                break
        _test("24. Prefetch/consume: no deadlock (20 cycles)", no_deadlock)

        # Verify prefetch result matches direct lookup.
        emb_cuda_off.prefetch(cuda_ids)
        pf_result = emb_cuda_off.consume_prefetched()
        direct_result = emb_cuda_off.lookup(cuda_ids)
        pf_cos = F.cosine_similarity(
            pf_result.flatten().unsqueeze(0),
            direct_result.flatten().unsqueeze(0),
        ).item()
        _test(
            "25. Prefetch result matches direct lookup",
            pf_cos > 0.9999,
            f"cosine={pf_cos:.6f}",
        )
    else:
        _test("24. Prefetch/consume: no deadlock", True, "SKIPPED (no CUDA)")
        _test("25. Prefetch matches direct", True, "SKIPPED (no CUDA)")

    # --- 26. Coalescing: repeated IDs reduce unique count ---
    print("\n--- Coalescing ---")
    repeated_ids = torch.tensor([[5, 5, 5, 5, 10, 10, 10, 10]], dtype=torch.int64)
    unique, inverse = torch.unique(repeated_ids.view(-1), return_inverse=True)
    _test(
        "26. Coalescing reduces unique count",
        unique.numel() == 2,
        f"unique={unique.numel()}, total={repeated_ids.numel()}",
    )

    # --- 27. Inverse mapping reconstructs original ---
    reconstructed = unique[inverse]
    _test(
        "27. Inverse mapping reconstructs original",
        torch.equal(reconstructed, repeated_ids.view(-1)),
    )

    # --- 28. PrefetchPlan: schedule triggers correctly ---
    print("\n--- PrefetchPlan ---")
    plan_ids = torch.randint(0, 10000, (2, 32), dtype=torch.int64)
    plan_layers = list(range(8))
    plan = PrefetchPlan(plan_ids, plan_layers, hasher, prefetch_ahead=2)
    schedule = plan.schedule
    expected_schedule = {0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7}
    _test(
        "28. PrefetchPlan schedule correct",
        schedule == expected_schedule,
        f"schedule={schedule}",
    )

    # --- 29. PrefetchPlan hash IDs are precomputed ---
    for lid in plan_layers:
        h = plan.get_hash_ids(lid)
        assert h.shape == (2, 32, hasher.H_total)
    _test("29. PrefetchPlan hash IDs precomputed for all layers", True)

    # --- 30-32. EmbeddingAggregator modes ---
    print("\n--- EmbeddingAggregator ---")
    dummy_embs = torch.randn(2, 16, 6, 64)  # B=2, T=16, H=6, D=64

    agg_sum = EmbeddingAggregator(num_heads=6, embedding_dim=64, mode="sum")
    sum_out = agg_sum(dummy_embs)
    _test(
        "30. Aggregator 'sum' shape",
        sum_out.shape == (2, 16, 64),
        f"shape={sum_out.shape}",
    )

    agg_mean = EmbeddingAggregator(num_heads=6, embedding_dim=64, mode="mean")
    mean_out = agg_mean(dummy_embs)
    _test(
        "31. Aggregator 'mean' shape",
        mean_out.shape == (2, 16, 64),
        f"shape={mean_out.shape}",
    )

    agg_proj = EmbeddingAggregator(
        num_heads=6, embedding_dim=64, mode="concat_project", output_dim=128,
    )
    proj_out = agg_proj(dummy_embs)
    _test(
        "32. Aggregator 'concat_project' shape",
        proj_out.shape == (2, 16, 128),
        f"shape={proj_out.shape}",
    )

    # --- 33. Head metadata correctness ---
    print("\n--- Head Metadata ---")
    meta = hasher.get_head_metadata()
    _test(
        "33. Head metadata count",
        len(meta) == hasher.H_total,
        f"expected={hasher.H_total}, got={len(meta)}",
    )

    # --- 34. Metadata orders are correct ---
    expected_orders = []
    for order in hasher.orders:
        for _ in range(hasher.num_heads_per_order):
            expected_orders.append(order)
    actual_orders = [m["order"] for m in meta]
    _test("34. Metadata orders correct", actual_orders == expected_orders)

    # --- 35. Metadata table sizes match ---
    meta_sizes = [m["table_size"] for m in meta]
    buf_sizes = hasher.table_sizes.tolist()
    _test("35. Metadata table sizes match buffer", meta_sizes == buf_sizes)

    # --- 36. Metadata multiplier lengths match order ---
    mult_lens_ok = all(len(m["multipliers"]) == m["order"] for m in meta)
    _test("36. Metadata multiplier lengths match order", mult_lens_ok)

    # --- 37. State dict save/load round-trip ---
    print("\n--- Checkpoint Round-Trip ---")
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=True) as f:
        path = f.name
        HashEmbeddingCheckpointer.save(hasher, path)
        loaded = HashEmbeddingCheckpointer.load(path)

        # Verify multipliers match.
        mult_match = torch.equal(hasher.multipliers, loaded.multipliers)
        _test("37. State dict round-trip: multipliers", mult_match)

        # Verify table sizes match.
        ts_match = torch.equal(hasher.table_sizes, loaded.table_sizes)
        _test("38. State dict round-trip: table_sizes", ts_match)

        # Verify salts match.
        if hasher.layer_salts is not None:
            salt_match = torch.equal(hasher.layer_salts, loaded.layer_salts)
            _test("39. State dict round-trip: layer_salts", salt_match)
        else:
            _test("39. State dict round-trip: layer_salts", True, "N/A (no salt)")

        # Verify hash output matches after load.
        ref_hash = hasher(ids, layer_id=0)
        loaded_hash = loaded(ids, layer_id=0)
        _test(
            "40. State dict round-trip: hash output match",
            torch.equal(ref_hash, loaded_hash),
        )

    # --- 41. Buffer-based save via state_dict only ---
    print("\n--- State Dict Direct ---")
    sd = hasher.state_dict()
    hasher2 = MultiHeadHash(config, max_ngram_order=4)
    hasher2.load_state_dict(sd)
    out_sd = hasher2(ids, layer_id=0)
    _test("41. state_dict load_state_dict hash match", torch.equal(ref, out_sd))

    # --- 42. OffloadableEmbedding forward == lookup ---
    print("\n--- Forward vs Lookup ---")
    fwd_out = emb_on(lookup_ids)
    lkp_out = emb_on.lookup(lookup_ids)
    _test(
        "42. OffloadableEmbedding forward == lookup",
        torch.equal(fwd_out, lkp_out),
    )

    # --- 43. PinnedBufferPool acquire/release ---
    print("\n--- PinnedBufferPool ---")
    pool = PinnedBufferPool(num_buffers=4, dtype=torch.float32)
    buf1 = pool.acquire(1024)
    _test("43. Buffer acquire shape", buf1.shape == (1024,))
    buf2 = pool.acquire(2048)
    _test("44. Second buffer acquire", buf2.shape == (2048,))
    _test("45. Pool has 2 in use", pool.num_in_use == 2)
    pool.release(buf1)
    _test("46. After release, 1 in use", pool.num_in_use == 1)
    pool.release(buf2)
    _test("47. After release all, 0 in use", pool.num_in_use == 0)

    # --- 48. Buffer reuse ---
    buf3 = pool.acquire(512)  # Should reuse buf1 (size 1024 >= 512).
    _test("48. Buffer reuse (acquired from pool)", pool.num_in_use == 1)
    pool.release(buf3)

    # --- 49. HashEmbeddingRetriever end-to-end ---
    print("\n--- HashEmbeddingRetriever ---")
    retriever = HashEmbeddingRetriever(
        hash_config=config,
        max_ngram_order=4,
        embedding_dim=32,
        aggregation_mode="sum",
    )
    ret_ids = torch.randint(0, 50000, (2, 16), dtype=torch.int64)
    ret_out = retriever(ret_ids, layer_id=0)
    _test(
        "49. Retriever output shape (sum)",
        ret_out.shape == (2, 16, 32),
        f"shape={ret_out.shape}",
    )

    # --- 50. Retriever with concat_project ---
    retriever_proj = HashEmbeddingRetriever(
        hash_config=config,
        max_ngram_order=4,
        embedding_dim=32,
        aggregation_mode="concat_project",
        aggregation_output_dim=128,
    )
    ret_proj_out = retriever_proj(ret_ids, layer_id=0)
    _test(
        "50. Retriever output shape (concat_project)",
        ret_proj_out.shape == (2, 16, 128),
        f"shape={ret_proj_out.shape}",
    )

    # --- 51. N-gram extractor ---
    print("\n--- MultiOrderNgramExtractor ---")
    ngram_ids = torch.tensor([[10, 20, 30, 40]], dtype=torch.int64)
    bigrams = MultiOrderNgramExtractor.extract(ngram_ids, order=2)
    # Position 0: (0, 10) -- left-padded; Position 1: (10, 20); etc.
    _test(
        "51. Bigram extraction shape",
        bigrams.shape == (1, 4, 2),
    )
    _test(
        "52. Bigram position 0 is left-padded",
        bigrams[0, 0, 0].item() == 0 and bigrams[0, 0, 1].item() == 10,
    )
    _test(
        "53. Bigram position 1 correct",
        bigrams[0, 1, 0].item() == 10 and bigrams[0, 1, 1].item() == 20,
    )

    # --- 54. Multi-order extraction ---
    multi = MultiOrderNgramExtractor.extract_multi_order(ngram_ids, orders=[2, 3])
    _test("54. Multi-order extraction keys", set(multi.keys()) == {2, 3})
    _test("55. Trigram shape", multi[3].shape == (1, 4, 3))

    # --- 56. StreamingHashCache ---
    print("\n--- StreamingHashCache ---")
    cache = StreamingHashCache(batch_size=1, max_ngram_order=4)
    for tok in [10, 20, 30, 40]:
        cache.push_token(torch.tensor([tok]))
    window = cache.get_ngram_window()
    _test(
        "56. StreamingHashCache window after 4 tokens",
        window.tolist() == [[10, 20, 30, 40]],
        f"window={window.tolist()}",
    )
    cache.push_token(torch.tensor([50]))
    window2 = cache.get_ngram_window()
    _test(
        "57. StreamingHashCache window after 5th token",
        window2.tolist() == [[20, 30, 40, 50]],
        f"window={window2.tolist()}",
    )

    # --- 58. DTypeSafeOps ---
    print("\n--- DTypeSafeOps ---")
    a = torch.tensor([100], dtype=torch.int32)
    b = torch.tensor([200], dtype=torch.int32)
    prod = DTypeSafeOps.safe_multiply(a, b)
    _test("58. safe_multiply dtype", prod.dtype == torch.int64)
    _test("59. safe_multiply value", prod.item() == 20000)
    xor_result = DTypeSafeOps.safe_xor(
        torch.tensor([0xFF], dtype=torch.int64),
        torch.tensor([0x0F], dtype=torch.int64),
    )
    _test("60. safe_xor value", xor_result.item() == 0xF0)

    # --- 61. ConfigValidator ---
    print("\n--- ConfigValidator ---")
    bad_config = HashConfig(
        base_table_size=5,
        num_heads_per_order=0,
        num_layers=0,
    )
    issues = ConfigValidator.validate_hash_config(bad_config)
    _test("61. ConfigValidator catches small table", any("small" in i for i in issues))
    _test("62. ConfigValidator catches 0 heads", any("must be >= 1" in i for i in issues))

    bad_offload = OffloadConfig(
        use_async_prefetch=True,
        weights_on_cpu=False,
    )
    off_issues = ConfigValidator.validate_offload_config(bad_offload)
    _test(
        "63. ConfigValidator catches prefetch without offload",
        any("no effect" in i for i in off_issues),
    )

    # --- 64. EmbeddingMemoryEstimator ---
    print("\n--- EmbeddingMemoryEstimator ---")
    est_bytes = EmbeddingMemoryEstimator.estimate_bytes(
        num_embeddings=100000, embedding_dim=256, dtype="float16", num_heads=6,
    )
    expected_bytes = 100000 * 256 * 2 * 6
    _test(
        "64. Memory estimate",
        est_bytes == expected_bytes,
        f"estimated={est_bytes}, expected={expected_bytes}",
    )
    human = EmbeddingMemoryEstimator.estimate_human_readable(
        num_embeddings=100000, embedding_dim=256, dtype="float16", num_heads=6,
    )
    _test("65. Human-readable estimate", "MB" in human or "GB" in human, f"'{human}'")

    # --- 66. BatchHasher ---
    print("\n--- BatchHasher ---")
    long_ids = torch.randint(0, 50000, (2, 300), dtype=torch.int64)
    full_hash = hasher(long_ids, layer_id=0)
    chunked_hash = BatchHasher.hash_chunked(hasher, long_ids, chunk_size=64, layer_id=0)
    _test(
        "66. Chunked hash matches full",
        torch.equal(full_hash, chunked_hash),
        f"full_shape={full_hash.shape}, chunked_shape={chunked_hash.shape}",
    )

    # --- 67. SaltedHashVariant ---
    print("\n--- SaltedHashVariant ---")
    ngram_t = torch.tensor([[100, 200, 300]], dtype=torch.int64)
    mults_t = torch.tensor([1000000007, 1000000009, 1000000021], dtype=torch.int64)
    h1 = SaltedHashVariant.hash_ngram(ngram_t, mults_t, table_size=131071, salt=1)
    h2 = SaltedHashVariant.hash_ngram(ngram_t, mults_t, table_size=131071, salt=1)
    _test("67. SaltedHashVariant determinism", torch.equal(h1, h2))
    h3 = SaltedHashVariant.hash_ngram(ngram_t, mults_t, table_size=131071, salt=7)
    _test("68. SaltedHashVariant different salt -> different hash", not torch.equal(h1, h3))

    # --- 69. GradientThroughHash ---
    print("\n--- GradientThroughHash ---")
    weight = torch.randn(100, 32, requires_grad=True)
    test_ids = torch.randint(0, 100, (4, 8))
    out = GradientThroughHash.apply(test_ids, weight)
    loss = out.sum()
    loss.backward()
    _test(
        "69. GradientThroughHash backward",
        weight.grad is not None and weight.grad.shape == (100, 32),
    )

    # --- 70. HashSeedExplorer ---
    print("\n--- HashSeedExplorer ---")
    sample = torch.randint(0, 50000, (2, 64), dtype=torch.int64)
    best = HashSeedExplorer.find_best_seed(
        sample,
        candidate_seeds=[42, 123, 456, 789],
        base_table_size=131071,
        num_heads_per_order=2,
    )
    _test(
        "70. HashSeedExplorer returns result",
        best is not None and "seed" in best and "unique_ratio" in best,
        f"best_seed={best['seed']}, ratio={best['unique_ratio']:.4f}",
    )

    # --- Summary ---
    print("\n" + "=" * 72)
    total = len(results)
    passed = sum(1 for _, p, _ in results if p)
    failed = total - passed
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if failed > 0:
        print("\nFailed tests:")
        for name, p, detail in results:
            if not p:
                print(f"  - {name}: {detail}")
    print("=" * 72)

    if failed > 0:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _run_self_tests()
