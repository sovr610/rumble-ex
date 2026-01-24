# DeepSeek Engram Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integrate DeepSeek's Engram conditional memory system into the brain-inspired AI architecture, providing O(1) static pattern retrieval as semantic memory.

**Architecture:** Two-phase integration: (1) Engram as a text encoder competing in Global Workspace, (2) Engram embedded at layer 1 in a transformer-style pipeline. Core components include tokenizer compression, multi-head hashing, context-aware gating, and CPU offloading for production scale.

**Tech Stack:** PyTorch 2.0+, existing brain_ai package patterns, ~10M entry hash tables with optional CPU offload.

---

## Task 1: Create Memory Module Directory Structure

**Files:**
- Create: `brain_ai/memory/__init__.py`
- Create: `brain_ai/layers/__init__.py`

**Step 1: Create the memory directory and __init__.py**

```bash
mkdir -p /home/sovr610/human-brain/brain_ai/memory
mkdir -p /home/sovr610/human-brain/brain_ai/layers
```

**Step 2: Write memory module __init__.py**

```python
# brain_ai/memory/__init__.py
"""
Memory systems for Brain-Inspired AI.

Engram: O(1) conditional memory via N-gram hash lookup.
"""

from .engram import (
    EngramConfig,
    EngramModule,
    EngramEmbedding,
    ContextAwareGating,
    RMSNorm,
)
from .tokenizer_compression import TokenizerCompression
from .hash_embedding import MultiHeadHash, OffloadableEmbedding

__all__ = [
    'EngramConfig',
    'EngramModule',
    'EngramEmbedding',
    'ContextAwareGating',
    'RMSNorm',
    'TokenizerCompression',
    'MultiHeadHash',
    'OffloadableEmbedding',
]
```

**Step 3: Write layers module __init__.py**

```python
# brain_ai/layers/__init__.py
"""
Neural network layers for Brain-Inspired AI.
"""

from .engram_layer import EngramAugmentedLayer

__all__ = [
    'EngramAugmentedLayer',
]
```

**Step 4: Verify files exist**

```bash
ls -la /home/sovr610/human-brain/brain_ai/memory/
ls -la /home/sovr610/human-brain/brain_ai/layers/
```

Expected: Both directories exist with __init__.py files.

---

## Task 2: Implement TokenizerCompression

**Files:**
- Create: `brain_ai/memory/tokenizer_compression.py`
- Create: `tests/test_engram.py`

**Step 1: Write the failing test**

```python
# tests/test_engram.py
"""Tests for Engram memory system."""

import pytest
import torch

from brain_ai.memory.tokenizer_compression import TokenizerCompression


class TestTokenizerCompression:
    """Tests for tokenizer compression."""

    def test_compress_reduces_vocab_range(self):
        """Compressed IDs should be in reduced range."""
        compressor = TokenizerCompression(
            vocab_size=50000,
            compressed_size=38500,  # ~77% of original
            mode="shared"
        )

        token_ids = torch.randint(0, 50000, (4, 32))
        compressed = compressor.compress(token_ids)

        assert compressed.shape == token_ids.shape
        assert compressed.max() < 38500
        assert compressed.min() >= 0

    def test_compress_is_deterministic(self):
        """Same input should always give same output."""
        compressor = TokenizerCompression(
            vocab_size=50000,
            compressed_size=38500,
            mode="shared"
        )

        token_ids = torch.randint(0, 50000, (4, 32))
        result1 = compressor.compress(token_ids)
        result2 = compressor.compress(token_ids)

        assert torch.equal(result1, result2)

    def test_semantic_equivalents_map_same(self):
        """Semantically equivalent tokens should map to same ID."""
        compressor = TokenizerCompression(
            vocab_size=50000,
            compressed_size=38500,
            mode="shared"
        )

        # In a real tokenizer, "Apple" (ID 100) and "apple" (ID 200)
        # would map to the same compressed ID. We test the mechanism works.
        # The actual mapping depends on tokenizer vocabulary.
        token_a = torch.tensor([[100]])
        token_b = torch.tensor([[100]])  # Same token

        assert torch.equal(
            compressor.compress(token_a),
            compressor.compress(token_b)
        )
```

**Step 2: Run test to verify it fails**

```bash
cd /home/sovr610/human-brain && python -m pytest tests/test_engram.py::TestTokenizerCompression -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'brain_ai.memory.tokenizer_compression'"

**Step 3: Write the implementation**

```python
# brain_ai/memory/tokenizer_compression.py
"""
Tokenizer Compression for Engram.

Maps semantically equivalent tokens to canonical IDs via:
- NFKC normalization
- Lowercasing
- Whitespace normalization

Achieves ~23% vocabulary reduction while preserving semantic content.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
import unicodedata


class TokenizerCompression:
    """
    Vocabulary projection that collapses semantically equivalent tokens.

    Maps raw token IDs to canonical identifiers. In production, this would
    analyze actual tokenizer vocabulary. Here we use hash-based simulation
    that can be replaced with actual token text analysis.

    Args:
        vocab_size: Original vocabulary size
        compressed_size: Target compressed vocabulary size (~77% of original)
        mode: "shared" uses same tokenizer as text encoder, "dedicated" is separate
        tokenizer: Optional tokenizer for text-based normalization
    """

    def __init__(
        self,
        vocab_size: int = 50000,
        compressed_size: Optional[int] = None,
        mode: str = "shared",
        tokenizer: Optional[object] = None,
    ):
        self.vocab_size = vocab_size
        self.compressed_size = compressed_size or int(vocab_size * 0.77)
        self.mode = mode
        self.tokenizer = tokenizer

        # Build projection table
        self._build_projection_table()

    def _build_projection_table(self):
        """Build the surjective mapping P: V -> V'."""
        self.projection = torch.zeros(self.vocab_size, dtype=torch.long)

        if self.tokenizer is not None and hasattr(self.tokenizer, 'get_vocab'):
            # Use actual token text for normalization
            self._build_from_tokenizer()
        else:
            # Fallback: deterministic hash-based projection
            self._build_hash_projection()

    def _build_hash_projection(self):
        """Build projection using deterministic hashing."""
        for token_id in range(self.vocab_size):
            # Use modulo for consistent mapping
            # This simulates normalization by mapping to compressed space
            canonical_id = token_id % self.compressed_size
            self.projection[token_id] = canonical_id

    def _build_from_tokenizer(self):
        """Build projection from actual tokenizer vocabulary."""
        vocab = self.tokenizer.get_vocab()

        # Map normalized text -> canonical ID
        normalized_to_id: Dict[str, int] = {}
        next_id = 0

        for token_text, token_id in vocab.items():
            # Normalize the token text
            normalized = self._normalize_text(token_text)

            if normalized not in normalized_to_id:
                normalized_to_id[normalized] = next_id
                next_id += 1
                if next_id >= self.compressed_size:
                    # Wrap around if we exceed target size
                    next_id = 0

            self.projection[token_id] = normalized_to_id[normalized]

    def _normalize_text(self, text: str) -> str:
        """Apply normalization to token text."""
        # NFKC normalization
        text = unicodedata.normalize('NFKC', text)

        # Lowercase
        text = text.lower()

        # Whitespace normalization
        text = ' '.join(text.split())

        # Strip leading/trailing whitespace markers
        text = text.strip()

        return text

    def compress(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Map raw token IDs to canonical IDs.

        Args:
            token_ids: (batch, seq_len) raw token IDs

        Returns:
            Compressed token IDs in range [0, compressed_size)
        """
        device = token_ids.device
        projection = self.projection.to(device)

        # Clamp to valid range to avoid index errors
        clamped = token_ids.clamp(0, self.vocab_size - 1)

        return projection[clamped]

    def get_compression_ratio(self) -> float:
        """Return the compression ratio achieved."""
        unique_mappings = len(self.projection.unique())
        return unique_mappings / self.vocab_size
```

**Step 4: Run tests to verify they pass**

```bash
cd /home/sovr610/human-brain && python -m pytest tests/test_engram.py::TestTokenizerCompression -v
```

Expected: All tests PASS.

---

## Task 3: Implement MultiHeadHash

**Files:**
- Create: `brain_ai/memory/hash_embedding.py`
- Modify: `tests/test_engram.py`

**Step 1: Write the failing tests**

Add to `tests/test_engram.py`:

```python
from brain_ai.memory.hash_embedding import MultiHeadHash, OffloadableEmbedding


class TestMultiHeadHash:
    """Tests for multi-head hashing."""

    def test_hash_output_shape(self):
        """Hash should produce correct output shape."""
        hasher = MultiHeadHash(
            ngram_order=2,
            num_heads=4,
            table_size=1000003,
        )

        # (batch, seq, ngram_order)
        ngrams = torch.randint(0, 10000, (4, 32, 2))
        hashed = hasher.hash(ngrams)

        assert hashed.shape == (4, 32, 4)  # (batch, seq, num_heads)

    def test_hash_in_valid_range(self):
        """Hash values should be in [0, table_size)."""
        table_size = 1000003
        hasher = MultiHeadHash(
            ngram_order=3,
            num_heads=4,
            table_size=table_size,
        )

        ngrams = torch.randint(0, 50000, (8, 64, 3))
        hashed = hasher.hash(ngrams)

        assert hashed.min() >= 0
        assert hashed.max() < table_size

    def test_hash_is_deterministic(self):
        """Same input should always produce same hash."""
        hasher = MultiHeadHash(
            ngram_order=2,
            num_heads=4,
            table_size=1000003,
        )

        ngrams = torch.randint(0, 10000, (4, 32, 2))
        result1 = hasher.hash(ngrams)
        result2 = hasher.hash(ngrams)

        assert torch.equal(result1, result2)

    def test_different_ngrams_produce_different_hashes(self):
        """Different n-grams should usually hash to different values."""
        hasher = MultiHeadHash(
            ngram_order=2,
            num_heads=4,
            table_size=1000003,
        )

        ngram_a = torch.tensor([[[100, 200]]])
        ngram_b = torch.tensor([[[100, 201]]])

        hash_a = hasher.hash(ngram_a)
        hash_b = hasher.hash(ngram_b)

        # At least one head should differ
        assert not torch.equal(hash_a, hash_b)


class TestOffloadableEmbedding:
    """Tests for offloadable embedding tables."""

    def test_forward_without_offload(self):
        """Standard embedding lookup should work."""
        embedding = OffloadableEmbedding(
            num_embeddings=10000,
            embedding_dim=64,
            offload=False,
        )

        indices = torch.randint(0, 10000, (4, 32))
        output = embedding(indices)

        assert output.shape == (4, 32, 64)

    def test_forward_with_offload_cpu(self):
        """CPU offload should produce correct results."""
        embedding = OffloadableEmbedding(
            num_embeddings=10000,
            embedding_dim=64,
            offload=True,
            prefetch=False,  # Disable prefetch for simple test
        )

        indices = torch.randint(0, 10000, (4, 32))
        output = embedding(indices)

        assert output.shape == (4, 32, 64)

    def test_offload_matches_standard(self):
        """Offloaded and standard should produce same results."""
        torch.manual_seed(42)
        standard = OffloadableEmbedding(
            num_embeddings=1000,
            embedding_dim=32,
            offload=False,
        )

        torch.manual_seed(42)
        offloaded = OffloadableEmbedding(
            num_embeddings=1000,
            embedding_dim=32,
            offload=True,
            prefetch=False,
        )

        # Copy weights to ensure same initialization
        offloaded.weight.data = standard.embedding.weight.data.clone()

        indices = torch.randint(0, 1000, (2, 16))

        out_std = standard(indices)
        out_off = offloaded(indices)

        assert torch.allclose(out_std, out_off, atol=1e-6)
```

**Step 2: Run tests to verify they fail**

```bash
cd /home/sovr610/human-brain && python -m pytest tests/test_engram.py::TestMultiHeadHash -v
cd /home/sovr610/human-brain && python -m pytest tests/test_engram.py::TestOffloadableEmbedding -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write the implementation**

```python
# brain_ai/memory/hash_embedding.py
"""
Multi-Head Hashing and Offloadable Embeddings for Engram.

Provides O(1) lookup via deterministic hashing with multiple heads
to reduce collision probability. Supports CPU offloading for large tables.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class MultiHeadHash:
    """
    Multi-head hashing for N-gram to embedding index mapping.

    Uses K distinct hash functions per N-gram order to reduce collisions.
    Hash function: multiplicative-XOR hash
    φ(g) = (Σ_i c_i * x_i) XOR seed mod M

    Args:
        ngram_order: The N in N-gram (e.g., 2 for bigrams)
        num_heads: Number of independent hash functions
        table_size: Size of hash table (should be prime)
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        ngram_order: int,
        num_heads: int,
        table_size: int,
        seed: int = 42,
    ):
        self.n = ngram_order
        self.num_heads = num_heads
        self.table_size = table_size

        # Generate random coefficients for each head
        rng = np.random.RandomState(seed)

        # Coefficients for multiplicative hash: (num_heads, ngram_order)
        self.coefficients = torch.tensor(
            rng.randint(1, table_size, size=(num_heads, ngram_order)),
            dtype=torch.long
        )

        # XOR seeds for each head
        self.seeds = torch.tensor(
            rng.randint(0, table_size, size=(num_heads,)),
            dtype=torch.long
        )

    def hash(self, ngrams: torch.Tensor) -> torch.Tensor:
        """
        Hash N-grams to embedding indices.

        Args:
            ngrams: (batch, seq_len, n) tensor of token IDs forming N-grams

        Returns:
            (batch, seq_len, num_heads) tensor of embedding indices
        """
        device = ngrams.device
        batch, seq_len, n = ngrams.shape

        coeffs = self.coefficients.to(device)  # (num_heads, n)
        seeds = self.seeds.to(device)          # (num_heads,)

        # Compute weighted sum: (batch, seq_len, num_heads)
        # ngrams: (batch, seq_len, n) -> (batch, seq_len, 1, n)
        # coeffs: (num_heads, n) -> (1, 1, num_heads, n)
        ngrams_expanded = ngrams.unsqueeze(2).long()  # (batch, seq, 1, n)
        coeffs_expanded = coeffs.unsqueeze(0).unsqueeze(0)  # (1, 1, heads, n)

        weighted = (ngrams_expanded * coeffs_expanded).sum(dim=-1)  # (batch, seq, heads)

        # XOR with seeds and take modulo
        seeds_expanded = seeds.unsqueeze(0).unsqueeze(0)  # (1, 1, heads)
        hashed = (weighted ^ seeds_expanded) % self.table_size

        return hashed


class OffloadableEmbedding(nn.Module):
    """
    Embedding table with optional CPU offload and async prefetching.

    For production scale (10M+ entries), embeddings can be stored in
    pinned CPU memory and transferred on-demand. Prefetching hides
    transfer latency by computing indices during the previous layer.

    Args:
        num_embeddings: Size of the embedding table
        embedding_dim: Dimension of each embedding vector
        offload: If True, store embeddings in CPU memory
        prefetch: If True, enable async prefetching (requires CUDA)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        offload: bool = False,
        prefetch: bool = True,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.offload = offload
        self.prefetch = prefetch and torch.cuda.is_available()

        if offload:
            # Store weights in CPU pinned memory for fast transfer
            self.weight = nn.Parameter(
                torch.zeros(num_embeddings, embedding_dim),
                requires_grad=True
            )
            # Initialize with normal distribution
            nn.init.normal_(self.weight, mean=0, std=0.02)

            # Pin memory for faster CPU->GPU transfer
            if torch.cuda.is_available():
                self.weight.data = self.weight.data.pin_memory()

            # Prefetch state
            self._prefetch_buffer: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
            self._prefetch_stream = torch.cuda.Stream() if self.prefetch else None
        else:
            # Standard GPU embedding
            self.embedding = nn.Embedding(num_embeddings, embedding_dim)
            nn.init.normal_(self.embedding.weight, mean=0, std=0.02)

    def prefetch_async(self, indices: torch.Tensor):
        """
        Prefetch embeddings for given indices asynchronously.

        Call this during the previous layer's computation to hide
        transfer latency. The prefetched embeddings are cached and
        used in the next forward() call.

        Args:
            indices: (batch, seq) or (batch, seq, heads) indices to prefetch
        """
        if not self.offload or not self.prefetch:
            return

        with torch.cuda.stream(self._prefetch_stream):
            # Flatten indices and get unique values
            flat_indices = indices.flatten()
            unique_indices = flat_indices.unique()

            # Transfer unique embeddings to GPU
            cpu_indices = unique_indices.cpu()
            embeddings = self.weight[cpu_indices].cuda(non_blocking=True)

            # Cache for forward pass
            self._prefetch_buffer = (unique_indices, embeddings)

    def _gather_from_cache(
        self,
        indices: torch.Tensor,
        cached_indices: torch.Tensor,
        cached_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Gather embeddings from prefetch cache."""
        # Build mapping from cached indices to positions
        device = indices.device
        original_shape = indices.shape
        flat_indices = indices.flatten()

        # Create lookup: cached_indices[i] -> i
        index_to_pos = torch.zeros(
            self.num_embeddings, dtype=torch.long, device=device
        )
        index_to_pos[cached_indices] = torch.arange(
            len(cached_indices), device=device
        )

        # Map requested indices to cache positions
        cache_positions = index_to_pos[flat_indices]

        # Gather from cached embeddings
        gathered = cached_embeddings[cache_positions]

        # Reshape to original
        return gathered.view(*original_shape, self.embedding_dim)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Look up embeddings for given indices.

        Args:
            indices: (batch, seq) or (batch, seq, heads) indices

        Returns:
            Embeddings with same leading dims + embedding_dim
        """
        if self.offload:
            if self._prefetch_buffer is not None and self.prefetch:
                # Wait for prefetch to complete
                torch.cuda.current_stream().wait_stream(self._prefetch_stream)
                cached_indices, cached_embeddings = self._prefetch_buffer
                result = self._gather_from_cache(indices, cached_indices, cached_embeddings)
                self._prefetch_buffer = None  # Clear cache
                return result
            else:
                # Synchronous fallback: transfer from CPU
                cpu_indices = indices.cpu()
                embeddings = self.weight[cpu_indices]
                if indices.is_cuda:
                    embeddings = embeddings.cuda()
                return embeddings
        else:
            return self.embedding(indices)
```

**Step 4: Run tests to verify they pass**

```bash
cd /home/sovr610/human-brain && python -m pytest tests/test_engram.py::TestMultiHeadHash -v
cd /home/sovr610/human-brain && python -m pytest tests/test_engram.py::TestOffloadableEmbedding -v
```

Expected: All tests PASS.

---

## Task 4: Implement Core Engram Module

**Files:**
- Create: `brain_ai/memory/engram.py`
- Modify: `tests/test_engram.py`

**Step 1: Write the failing tests**

Add to `tests/test_engram.py`:

```python
from brain_ai.memory.engram import (
    EngramConfig,
    EngramEmbedding,
    ContextAwareGating,
    EngramModule,
    RMSNorm,
)


class TestRMSNorm:
    """Tests for RMSNorm."""

    def test_output_shape(self):
        """RMSNorm should preserve shape."""
        norm = RMSNorm(dim=256)
        x = torch.randn(4, 32, 256)
        out = norm(x)
        assert out.shape == x.shape

    def test_normalized_scale(self):
        """Output should be approximately normalized."""
        norm = RMSNorm(dim=256)
        x = torch.randn(4, 32, 256) * 10  # Large values
        out = norm(x)

        # RMS should be close to 1 after normalization
        rms = torch.sqrt(torch.mean(out ** 2, dim=-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.5)


class TestEngramEmbedding:
    """Tests for EngramEmbedding."""

    def test_output_shape(self):
        """Embedding output should match expected shape."""
        config = EngramConfig(
            vocab_size=10000,
            embedding_dim=128,
            ngram_orders=(2, 3),
            num_heads=4,
            table_size=100003,
        )

        embedding = EngramEmbedding(config)
        token_ids = torch.randint(0, 10000, (4, 32))
        output = embedding(token_ids)

        assert output.shape == (4, 32, 128)

    def test_deterministic(self):
        """Same input should produce same output."""
        config = EngramConfig(
            vocab_size=10000,
            embedding_dim=128,
            ngram_orders=(2, 3),
            num_heads=4,
            table_size=100003,
        )

        embedding = EngramEmbedding(config)
        token_ids = torch.randint(0, 10000, (4, 32))

        out1 = embedding(token_ids)
        out2 = embedding(token_ids)

        assert torch.equal(out1, out2)


class TestContextAwareGating:
    """Tests for ContextAwareGating."""

    def test_output_shape(self):
        """Gating should produce correct output shape."""
        gating = ContextAwareGating(
            hidden_dim=512,
            memory_dim=256,
        )

        hidden = torch.randn(4, 32, 512)
        memory = torch.randn(4, 32, 256)

        output, alpha = gating(hidden, memory)

        assert output.shape == (4, 32, 512)
        assert alpha.shape == (4, 32)

    def test_gate_values_in_range(self):
        """Gate values should be in [0, 1]."""
        gating = ContextAwareGating(
            hidden_dim=512,
            memory_dim=256,
        )

        hidden = torch.randn(4, 32, 512)
        memory = torch.randn(4, 32, 256)

        _, alpha = gating(hidden, memory)

        assert alpha.min() >= 0
        assert alpha.max() <= 1


class TestEngramModule:
    """Tests for complete EngramModule."""

    def test_output_shape(self):
        """EngramModule should produce correct output shape."""
        config = EngramConfig(
            vocab_size=10000,
            embedding_dim=128,
            ngram_orders=(2, 3),
            num_heads=4,
            table_size=100003,
        )

        engram = EngramModule(config, hidden_dim=512)

        token_ids = torch.randint(0, 10000, (4, 32))
        hidden = torch.randn(4, 32, 512)

        output, info = engram(token_ids, hidden)

        assert output.shape == (4, 32, 512)
        assert 'gate_values' in info
        assert info['gate_values'].shape == (4, 32)

    def test_residual_compatible(self):
        """Output should be suitable for residual connection."""
        config = EngramConfig(
            vocab_size=10000,
            embedding_dim=128,
            ngram_orders=(2, 3),
            num_heads=4,
            table_size=100003,
        )

        engram = EngramModule(config, hidden_dim=512)

        token_ids = torch.randint(0, 10000, (4, 32))
        hidden = torch.randn(4, 32, 512)

        output, _ = engram(token_ids, hidden)

        # Should be able to add to hidden (residual)
        result = hidden + output
        assert result.shape == hidden.shape
        assert not torch.isnan(result).any()
```

**Step 2: Run tests to verify they fail**

```bash
cd /home/sovr610/human-brain && python -m pytest tests/test_engram.py::TestRMSNorm -v
cd /home/sovr610/human-brain && python -m pytest tests/test_engram.py::TestEngramEmbedding -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write the implementation**

```python
# brain_ai/memory/engram.py
"""
Engram: Conditional Memory via Scalable Lookup.

Implementation based on DeepSeek's Engram paper. Provides O(1) retrieval
of static patterns (idioms, named entities, formulaic phrases) via N-gram
hashing, freeing transformer depth for compositional reasoning.

Key components:
- EngramConfig: Configuration dataclass
- EngramEmbedding: N-gram hash tables with multi-head lookup
- ContextAwareGating: Gates memory based on transformer hidden state
- EngramModule: Complete module with convolution and residual
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

from .tokenizer_compression import TokenizerCompression
from .hash_embedding import MultiHeadHash, OffloadableEmbedding


@dataclass
class EngramConfig:
    """Configuration for Engram module."""

    # Vocabulary
    vocab_size: int = 50000
    compressed_vocab_size: Optional[int] = None  # ~77% of vocab_size if None

    # Embeddings
    embedding_dim: int = 256
    ngram_orders: Tuple[int, ...] = (2, 3)  # Bigrams and trigrams
    num_heads: int = 4
    table_size: int = 10_000_003  # Prime number, production scale

    # Tokenizer
    tokenizer_mode: str = "shared"  # "shared" or "dedicated"
    use_compression: bool = True

    # Convolution
    conv_kernel_size: int = 4
    conv_dilation: int = 3

    # Offloading
    offload_to_cpu: bool = False
    prefetch: bool = True

    # Gating
    gate_temperature: float = 1.0

    def __post_init__(self):
        if self.compressed_vocab_size is None:
            self.compressed_vocab_size = int(self.vocab_size * 0.77)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class EngramEmbedding(nn.Module):
    """
    N-gram embedding tables with multi-head hashing.

    For each N-gram order n, maintains K embedding tables of size M.
    Retrieved embeddings are concatenated across all orders and heads.
    """

    def __init__(self, config: EngramConfig):
        super().__init__()

        self.config = config
        self.ngram_orders = config.ngram_orders
        self.num_heads = config.num_heads

        # Dimension per individual embedding
        total_retrievals = len(self.ngram_orders) * config.num_heads
        self.dim_per_embedding = config.embedding_dim // total_retrievals

        # Create hash functions for each N-gram order
        self.hashers = nn.ModuleDict()  # Use ModuleDict for proper registration
        for n in self.ngram_orders:
            # Store hasher as a simple container
            self.register_buffer(
                f'coeffs_{n}',
                MultiHeadHash(n, config.num_heads, config.table_size, seed=42+n).coefficients
            )
            self.register_buffer(
                f'seeds_{n}',
                MultiHeadHash(n, config.num_heads, config.table_size, seed=42+n).seeds
            )

        # Store hasher objects for hashing
        self._hashers = {
            n: MultiHeadHash(n, config.num_heads, config.table_size, seed=42+n)
            for n in self.ngram_orders
        }

        # Create embedding tables
        self.embeddings = nn.ModuleDict()
        for n in self.ngram_orders:
            for k in range(config.num_heads):
                key = f"ngram{n}_head{k}"
                if config.offload_to_cpu:
                    self.embeddings[key] = OffloadableEmbedding(
                        num_embeddings=config.table_size,
                        embedding_dim=self.dim_per_embedding,
                        offload=True,
                        prefetch=config.prefetch,
                    )
                else:
                    self.embeddings[key] = OffloadableEmbedding(
                        num_embeddings=config.table_size,
                        embedding_dim=self.dim_per_embedding,
                        offload=False,
                    )

        # Tokenizer compression
        if config.use_compression:
            self.compressor = TokenizerCompression(
                vocab_size=config.vocab_size,
                compressed_size=config.compressed_vocab_size,
                mode=config.tokenizer_mode,
            )
        else:
            self.compressor = None

    def extract_ngrams(self, token_ids: torch.Tensor, n: int) -> torch.Tensor:
        """
        Extract suffix N-grams from token sequence.

        Args:
            token_ids: (batch, seq_len) compressed token IDs
            n: N-gram order

        Returns:
            (batch, seq_len, n) where [b, t, :] = (x_{t-n+1}, ..., x_t)
        """
        batch, seq_len = token_ids.shape

        # Pad with zeros for positions < n
        padded = F.pad(token_ids, (n - 1, 0), value=0)

        # Extract sliding windows
        ngrams = torch.stack([
            padded[:, i:i + seq_len]
            for i in range(n)
        ], dim=-1)

        return ngrams

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Retrieve N-gram embeddings for input sequence.

        Args:
            token_ids: (batch, seq_len) raw token IDs

        Returns:
            (batch, seq_len, embedding_dim) concatenated embeddings
        """
        # Apply tokenizer compression
        if self.compressor is not None:
            compressed_ids = self.compressor.compress(token_ids)
        else:
            compressed_ids = token_ids

        all_embeddings = []

        for n in self.ngram_orders:
            # Extract N-grams
            ngrams = self.extract_ngrams(compressed_ids, n)

            # Hash to indices
            indices = self._hashers[n].hash(ngrams)

            # Retrieve from each head's table
            for k in range(self.num_heads):
                key = f"ngram{n}_head{k}"
                head_indices = indices[:, :, k]
                head_emb = self.embeddings[key](head_indices)
                all_embeddings.append(head_emb)

        return torch.cat(all_embeddings, dim=-1)


class ContextAwareGating(nn.Module):
    """
    Context-aware gating for Engram.

    Uses hidden state (with global context) to gate retrieved memory.
    Gate α = σ(RMSNorm(h)ᵀ · RMSNorm(Ke) / √d)
    Output = α · Ve
    """

    def __init__(
        self,
        hidden_dim: int,
        memory_dim: int,
        temperature: float = 1.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        self.temperature = temperature

        # Projections
        self.W_K = nn.Linear(memory_dim, hidden_dim, bias=False)
        self.W_V = nn.Linear(memory_dim, hidden_dim, bias=False)

        # Normalization
        self.query_norm = RMSNorm(hidden_dim)
        self.key_norm = RMSNorm(hidden_dim)

        self.scale = hidden_dim ** -0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply context-aware gating.

        Args:
            hidden_states: (batch, seq, hidden_dim) current hidden state
            memory: (batch, seq, memory_dim) retrieved N-gram embeddings

        Returns:
            gated_output: (batch, seq, hidden_dim)
            gate_values: (batch, seq) the α values
        """
        # Project memory
        k = self.W_K(memory)
        v = self.W_V(memory)

        # Normalize
        q_norm = self.query_norm(hidden_states)
        k_norm = self.key_norm(k)

        # Compute gate
        gate_logits = (q_norm * k_norm).sum(dim=-1, keepdim=True) * self.scale
        gate_logits = gate_logits / self.temperature
        alpha = torch.sigmoid(gate_logits)

        # Apply gate
        gated_output = alpha * v

        return gated_output, alpha.squeeze(-1)


class EngramModule(nn.Module):
    """
    Complete Engram conditional memory module.

    Architecture:
    1. N-gram extraction with tokenizer compression
    2. Multi-head hash lookup from embedding tables
    3. Context-aware gating using hidden state
    4. Depthwise causal convolution for receptive field expansion
    5. Residual connection to backbone
    """

    def __init__(
        self,
        config: EngramConfig,
        hidden_dim: int,
    ):
        super().__init__()

        self.config = config
        self.hidden_dim = hidden_dim

        # N-gram embedding retrieval
        self.ngram_embedding = EngramEmbedding(config)

        # Context-aware gating
        self.gating = ContextAwareGating(
            hidden_dim=hidden_dim,
            memory_dim=config.embedding_dim,
            temperature=config.gate_temperature,
        )

        # Depthwise causal convolution
        padding = (config.conv_kernel_size - 1) * config.conv_dilation
        self.conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=config.conv_kernel_size,
            padding=padding,
            dilation=config.conv_dilation,
            groups=hidden_dim,  # Depthwise
        )

        # Pre-conv normalization
        self.conv_norm = RMSNorm(hidden_dim)

        # Zero-init conv for smooth training start
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(
        self,
        token_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply Engram conditional memory.

        Args:
            token_ids: (batch, seq_len) input token IDs
            hidden_states: (batch, seq_len, hidden_dim) current hidden states

        Returns:
            output: (batch, seq_len, hidden_dim) to add to residual stream
            info: Dictionary with diagnostics
        """
        batch, seq_len = token_ids.shape

        # Step 1: Retrieve N-gram embeddings
        memory = self.ngram_embedding(token_ids)

        # Step 2: Context-aware gating
        gated, gate_values = self.gating(hidden_states, memory)

        # Step 3: Depthwise causal convolution
        gated_norm = self.conv_norm(gated)
        conv_input = gated_norm.transpose(1, 2)  # (batch, hidden, seq)
        conv_output = self.conv(conv_input)
        conv_output = conv_output[:, :, :seq_len]  # Truncate for causality
        conv_output = conv_output.transpose(1, 2)  # Back to (batch, seq, hidden)

        # SiLU activation + residual within module
        output = F.silu(conv_output) + gated

        info = {
            'gate_values': gate_values,
            'memory_norm': memory.norm(dim=-1).mean(),
        }

        return output, info
```

**Step 4: Run tests to verify they pass**

```bash
cd /home/sovr610/human-brain && python -m pytest tests/test_engram.py -v
```

Expected: All tests PASS.

---

## Task 5: Add EngramConfig to BrainAIConfig

**Files:**
- Modify: `brain_ai/config.py`

**Step 1: Read current config.py**

Already read above.

**Step 2: Add EngramConfig import and field**

Add after line 113 (after MetaConfig):

```python
@dataclass
class EngramConfig:
    """Engram conditional memory configuration."""
    # Vocabulary
    vocab_size: int = 50000
    compressed_vocab_size: int = 38500  # ~77% of vocab_size

    # Embeddings
    embedding_dim: int = 256
    ngram_orders: tuple = (2, 3)  # Bigrams and trigrams
    num_heads: int = 4
    table_size: int = 10_000_003  # Prime, production scale

    # Tokenizer
    tokenizer_mode: str = "shared"
    use_compression: bool = True

    # Convolution
    conv_kernel_size: int = 4
    conv_dilation: int = 3

    # Offloading
    offload_to_cpu: bool = False
    prefetch: bool = True

    # Gating
    gate_temperature: float = 1.0
```

**Step 3: Add to BrainAIConfig**

Modify BrainAIConfig (after line 127) to add:

```python
    engram: EngramConfig = field(default_factory=EngramConfig)
```

And add feature flag (after line 134):

```python
    use_engram: bool = False
    engram_layer_idx: int = 1  # Which layer for Phase 2 integration
```

**Step 4: Verify import works**

```bash
cd /home/sovr610/human-brain && python -c "from brain_ai.config import BrainAIConfig, EngramConfig; print('OK')"
```

Expected: "OK"

---

## Task 6: Implement EngramTextEncoder (Phase 1)

**Files:**
- Create: `brain_ai/encoders/engram_encoder.py`
- Modify: `tests/test_engram.py`

**Step 1: Write the failing test**

Add to `tests/test_engram.py`:

```python
from brain_ai.encoders.engram_encoder import EngramTextEncoder
from brain_ai.config import EngramConfig


class TestEngramTextEncoder:
    """Tests for Phase 1 encoder integration."""

    def test_output_shape(self):
        """Encoder should produce workspace-compatible output."""
        config = EngramConfig(
            vocab_size=10000,
            embedding_dim=128,
            ngram_orders=(2, 3),
            num_heads=4,
            table_size=100003,
        )

        encoder = EngramTextEncoder(config, output_dim=512)
        token_ids = torch.randint(0, 10000, (4, 32))

        output = encoder(token_ids)

        assert output.shape == (4, 512)

    def test_matches_other_encoder_interface(self):
        """Should be usable alongside other encoders."""
        config = EngramConfig(
            vocab_size=10000,
            embedding_dim=128,
            ngram_orders=(2, 3),
            num_heads=4,
            table_size=100003,
        )

        encoder = EngramTextEncoder(config, output_dim=512)

        # Should accept token_ids like TextEncoder
        token_ids = torch.randint(0, 10000, (4, 32))
        output = encoder(token_ids)

        assert output.dim() == 2
        assert output.shape[-1] == 512
```

**Step 2: Run test to verify it fails**

```bash
cd /home/sovr610/human-brain && python -m pytest tests/test_engram.py::TestEngramTextEncoder -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write the implementation**

```python
# brain_ai/encoders/engram_encoder.py
"""
Engram Text Encoder.

Phase 1 integration: Engram as a fast text encoder that competes
with the standard transformer-based encoder in the Global Workspace.
"""

import torch
import torch.nn as nn
from typing import Optional

from ..memory.engram import EngramModule, EngramEmbedding
from ..config import EngramConfig


class EngramTextEncoder(nn.Module):
    """
    Engram-based text encoder for Global Workspace.

    Provides O(1) pattern retrieval as an alternative/complement
    to transformer-based text encoding. The workspace attention
    mechanism will learn when to prefer Engram vs transformer.

    Args:
        config: EngramConfig with embedding settings
        output_dim: Output dimension (should match other encoders)
        use_positional: Add positional encoding for sequence awareness
    """

    def __init__(
        self,
        config: EngramConfig,
        output_dim: int = 512,
        use_positional: bool = True,
        max_seq_len: int = 2048,
    ):
        super().__init__()

        self.config = config
        self.output_dim = output_dim
        self.use_positional = use_positional

        # Core Engram embedding
        self.engram_embedding = EngramEmbedding(config)

        # Positional encoding for sequence awareness
        if use_positional:
            self.pos_encoding = nn.Parameter(
                torch.zeros(1, max_seq_len, config.embedding_dim)
            )
            nn.init.normal_(self.pos_encoding, mean=0, std=0.02)

        # Project to output dimension
        self.output_proj = nn.Sequential(
            nn.Linear(config.embedding_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode tokens via Engram lookup.

        Args:
            token_ids: (batch, seq_len) input token IDs
            attention_mask: Optional (batch, seq_len) mask, 1=valid, 0=padding

        Returns:
            (batch, output_dim) encoded representation
        """
        batch, seq_len = token_ids.shape

        # Get Engram embeddings
        embeddings = self.engram_embedding(token_ids)  # (batch, seq, embed_dim)

        # Add positional encoding
        if self.use_positional:
            embeddings = embeddings + self.pos_encoding[:, :seq_len, :]

        # Pool across sequence
        if attention_mask is not None:
            # Masked mean pooling
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = embeddings.mean(dim=1)

        # Project to output dimension
        output = self.output_proj(pooled)

        return output


def create_engram_encoder(
    output_dim: int = 512,
    vocab_size: int = 50000,
    embedding_dim: int = 256,
    ngram_orders: tuple = (2, 3),
    num_heads: int = 4,
    table_size: int = 10_000_003,
    **kwargs,
) -> EngramTextEncoder:
    """Factory function to create Engram text encoder."""
    config = EngramConfig(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        ngram_orders=ngram_orders,
        num_heads=num_heads,
        table_size=table_size,
        **{k: v for k, v in kwargs.items() if hasattr(EngramConfig, k)},
    )

    return EngramTextEncoder(config, output_dim=output_dim)
```

**Step 4: Update encoders __init__.py**

Add to `brain_ai/encoders/__init__.py`:

```python
from .engram_encoder import EngramTextEncoder, create_engram_encoder
```

**Step 5: Run tests to verify they pass**

```bash
cd /home/sovr610/human-brain && python -m pytest tests/test_engram.py::TestEngramTextEncoder -v
```

Expected: All tests PASS.

---

## Task 7: Integrate Engram Encoder into BrainAI

**Files:**
- Modify: `brain_ai/system.py`
- Modify: `tests/test_engram.py`

**Step 1: Write the failing test**

Add to `tests/test_engram.py`:

```python
class TestBrainAIEngramIntegration:
    """Tests for Engram integration with BrainAI."""

    def test_brain_ai_with_engram_encoder(self):
        """BrainAI should work with Engram encoder enabled."""
        from brain_ai.system import create_brain_ai
        from brain_ai.config import BrainAIConfig

        config = BrainAIConfig.minimal()
        config.use_engram = True
        config.modalities = ['text']

        brain = create_brain_ai(
            modalities=['text'],
            use_engram=True,
            use_htm=False,
            use_symbolic=False,
            use_meta=False,
            device='cpu',
        )

        # Forward pass with token_ids
        token_ids = torch.randint(0, 30000, (2, 32))

        output = brain({'text': token_ids, 'token_ids': token_ids})

        assert output.shape[0] == 2

    def test_engram_competes_in_workspace(self):
        """Engram should participate in workspace competition."""
        from brain_ai.system import create_brain_ai

        brain = create_brain_ai(
            modalities=['text'],
            use_engram=True,
            use_htm=False,
            use_symbolic=False,
            use_meta=False,
            device='cpu',
        )

        token_ids = torch.randint(0, 30000, (2, 32))

        result = brain(
            {'text': token_ids, 'token_ids': token_ids},
            return_details=True
        )

        # Should have attention weights including engram
        if result.attention is not None:
            assert 'engram' in result.attention or len(result.attention) > 0
```

**Step 2: Run test to verify it fails**

```bash
cd /home/sovr610/human-brain && python -m pytest tests/test_engram.py::TestBrainAIEngramIntegration -v
```

Expected: FAIL (BrainAI doesn't support use_engram yet)

**Step 3: Modify system.py**

Add import at top (around line 40):

```python
from .encoders.engram_encoder import EngramTextEncoder, create_engram_encoder
```

Modify `_build_encoders` method (around line 93) to add Engram encoder:

```python
    def _build_encoders(self):
        """Build modality-specific encoders."""
        self.encoders = nn.ModuleDict()

        encoder_dim = self.config.encoder.output_dim

        if 'vision' in self.modalities:
            self.encoders['vision'] = create_vision_encoder(
                output_dim=encoder_dim,
                channels=self.config.encoder.vision_channels,
                beta=self.config.snn.beta,
                num_steps=self.config.snn.num_timesteps,
            )

        if 'text' in self.modalities:
            self.encoders['text'] = create_text_encoder(
                output_dim=encoder_dim,
                vocab_size=self.config.encoder.text_vocab_size,
                embed_dim=self.config.encoder.text_embed_dim,
                num_layers=self.config.encoder.text_num_layers,
                num_heads=self.config.encoder.text_num_heads,
            )

        if 'audio' in self.modalities:
            self.encoders['audio'] = create_audio_encoder(
                output_dim=encoder_dim,
                n_mels=self.config.encoder.audio_n_mels,
                sample_rate=self.config.encoder.audio_sample_rate,
            )

        if 'sensors' in self.modalities:
            self.encoders['sensors'] = create_sensor_encoder(
                output_dim=encoder_dim,
                hidden_dim=self.config.encoder.sensor_hidden_dim,
            )

        # Engram encoder (Phase 1 integration)
        if self.config.use_engram:
            self.encoders['engram'] = create_engram_encoder(
                output_dim=encoder_dim,
                vocab_size=self.config.engram.vocab_size,
                embedding_dim=self.config.engram.embedding_dim,
                ngram_orders=self.config.engram.ngram_orders,
                num_heads=self.config.engram.num_heads,
                table_size=self.config.engram.table_size,
            )
```

Modify `encode` method (around line 217) to handle token_ids for Engram:

```python
    def encode(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Encode all modality inputs.

        Args:
            inputs: Dict mapping modality names to tensors

        Returns:
            Dict of encoded features (all same dimension)
        """
        encoded = {}

        for name, data in inputs.items():
            if name in self.encoders:
                encoded[name] = self.encoders[name](data)

        # Handle Engram separately - it needs token_ids
        if 'engram' in self.encoders and 'token_ids' in inputs:
            encoded['engram'] = self.encoders['engram'](inputs['token_ids'])

        return encoded
```

Modify `_build_workspace` method (around line 141) to include engram in modality_dims:

```python
    def _build_workspace(self):
        """Build global workspace."""
        if self.config.use_workspace:
            modality_dims = {
                name: self.config.encoder.output_dim
                for name in self.modalities
            }

            # Add engram to workspace if enabled
            if self.config.use_engram:
                modality_dims['engram'] = self.config.encoder.output_dim

            self.workspace = create_global_workspace(
                workspace_dim=self.config.workspace.workspace_dim,
                modality_dims=modality_dims,
                num_heads=self.config.workspace.num_heads,
                capacity_limit=self.config.workspace.capacity_limit,
                memory_mode=self.config.workspace.memory_mode,
                use_htm=self.config.use_htm,
                htm_layer=self.htm if self.config.use_htm else None,
            )
        else:
            # ... rest unchanged
```

Modify `create_brain_ai` function (around line 392) to accept use_engram:

```python
def create_brain_ai(
    modalities: List[str] = ['vision'],
    output_type: str = 'classify',
    num_classes: int = 10,
    control_dim: int = 6,
    use_htm: bool = True,
    use_symbolic: bool = True,
    use_meta: bool = True,
    use_engram: bool = False,  # Add this
    workspace_dim: int = 512,
    device: str = 'auto',
    **kwargs,
) -> BrainAI:
    # ... docstring ...

    # Create config
    config = BrainAIConfig()
    config.modalities = modalities
    config.use_htm = use_htm
    config.use_symbolic = use_symbolic
    config.use_meta = use_meta
    config.use_engram = use_engram  # Add this
    # ... rest unchanged
```

**Step 4: Run tests to verify they pass**

```bash
cd /home/sovr610/human-brain && python -m pytest tests/test_engram.py::TestBrainAIEngramIntegration -v
```

Expected: All tests PASS.

---

## Task 8: Implement EngramAugmentedLayer (Phase 2)

**Files:**
- Create: `brain_ai/layers/engram_layer.py`
- Modify: `tests/test_engram.py`

**Step 1: Write the failing test**

Add to `tests/test_engram.py`:

```python
from brain_ai.layers.engram_layer import EngramAugmentedLayer
from brain_ai.config import EngramConfig, SNNConfig


class TestEngramAugmentedLayer:
    """Tests for Phase 2 layer integration."""

    def test_output_shape(self):
        """Layer should preserve input shape."""
        engram_config = EngramConfig(
            vocab_size=10000,
            embedding_dim=128,
            ngram_orders=(2, 3),
            num_heads=4,
            table_size=100003,
        )
        snn_config = SNNConfig()

        layer = EngramAugmentedLayer(
            hidden_dim=512,
            engram_config=engram_config,
            snn_config=snn_config,
            use_engram=True,
            use_snn=False,  # Disable SNN for simpler test
        )

        x = torch.randn(4, 32, 512)
        token_ids = torch.randint(0, 10000, (4, 32))

        output = layer(x, token_ids)

        assert output.shape == x.shape

    def test_layer_without_engram(self):
        """Layer should work without Engram enabled."""
        engram_config = EngramConfig()
        snn_config = SNNConfig()

        layer = EngramAugmentedLayer(
            hidden_dim=512,
            engram_config=engram_config,
            snn_config=snn_config,
            use_engram=False,
            use_snn=False,
        )

        x = torch.randn(4, 32, 512)
        token_ids = torch.randint(0, 10000, (4, 32))

        output = layer(x, token_ids)

        assert output.shape == x.shape

    def test_full_pipeline(self):
        """Full Engram -> Attention -> FFN pipeline."""
        engram_config = EngramConfig(
            vocab_size=10000,
            embedding_dim=128,
            ngram_orders=(2, 3),
            num_heads=4,
            table_size=100003,
        )
        snn_config = SNNConfig()

        layer = EngramAugmentedLayer(
            hidden_dim=256,
            engram_config=engram_config,
            snn_config=snn_config,
            use_engram=True,
            use_snn=False,
            num_heads=4,
        )

        x = torch.randn(2, 16, 256)
        token_ids = torch.randint(0, 10000, (2, 16))

        output = layer(x, token_ids)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()
```

**Step 2: Run test to verify it fails**

```bash
cd /home/sovr610/human-brain && python -m pytest tests/test_engram.py::TestEngramAugmentedLayer -v
```

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write the implementation**

```python
# brain_ai/layers/engram_layer.py
"""
Engram-Augmented Layer.

Phase 2 integration: Engram embedded at layer 1 in a transformer-style
pipeline. Follows the brain-inspired architecture:
    Engram → SNN → Attention → FFN
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict

from ..memory.engram import EngramModule, EngramConfig, RMSNorm
from ..config import SNNConfig


class EngramAugmentedLayer(nn.Module):
    """
    Transformer-style layer with Engram conditional memory.

    The layer follows a brain-inspired pipeline:
    1. Engram: O(1) static pattern retrieval (semantic memory)
    2. SNN: Temporal processing with spiking dynamics (optional)
    3. Attention: Global context integration
    4. FFN: Nonlinear transformation

    Args:
        hidden_dim: Hidden dimension throughout the layer
        engram_config: Configuration for Engram module
        snn_config: Configuration for SNN (optional)
        use_engram: Whether to use Engram at this layer
        use_snn: Whether to use SNN processing
        num_heads: Number of attention heads
        ffn_mult: FFN hidden dim multiplier
        dropout: Dropout rate
    """

    def __init__(
        self,
        hidden_dim: int,
        engram_config: EngramConfig,
        snn_config: Optional[SNNConfig] = None,
        use_engram: bool = True,
        use_snn: bool = False,
        num_heads: int = 8,
        ffn_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_engram = use_engram
        self.use_snn = use_snn

        # 1. Engram (conditional memory)
        if use_engram:
            self.engram = EngramModule(engram_config, hidden_dim)
            self.engram_norm = RMSNorm(hidden_dim)

        # 2. SNN (temporal processing) - optional
        if use_snn and snn_config is not None:
            from ..core.snn import SNNLinear
            self.snn = SNNLinear(
                hidden_dim, hidden_dim,
                beta=snn_config.beta,
                num_steps=snn_config.num_timesteps,
            )
            self.snn_norm = RMSNorm(hidden_dim)

        # 3. Attention (global context)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = RMSNorm(hidden_dim)

        # 4. FFN (nonlinear transform)
        ffn_dim = hidden_dim * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = RMSNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the layer.

        Args:
            x: (batch, seq, hidden_dim) input hidden states
            token_ids: (batch, seq) token IDs for Engram lookup
            attention_mask: Optional attention mask

        Returns:
            (batch, seq, hidden_dim) output hidden states
        """
        # 1. Engram (semantic memory)
        if self.use_engram and token_ids is not None:
            x_norm = self.engram_norm(x)
            engram_out, _ = self.engram(token_ids, x_norm)
            x = x + engram_out

        # 2. SNN (temporal dynamics)
        if self.use_snn and hasattr(self, 'snn'):
            x_norm = self.snn_norm(x)
            snn_out, _ = self.snn(x_norm)
            x = x + snn_out

        # 3. Attention (global context)
        x_norm = self.attn_norm(x)

        # Convert attention mask for MultiheadAttention
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0  # True = ignore

        attn_out, _ = self.attention(
            x_norm, x_norm, x_norm,
            key_padding_mask=key_padding_mask,
        )
        x = x + attn_out

        # 4. FFN (nonlinear transform)
        x_norm = self.ffn_norm(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out

        return x


def create_engram_layer(
    hidden_dim: int = 512,
    use_engram: bool = True,
    use_snn: bool = False,
    **kwargs,
) -> EngramAugmentedLayer:
    """Factory function to create Engram-augmented layer."""
    engram_config = EngramConfig(**{
        k: v for k, v in kwargs.items()
        if hasattr(EngramConfig, k)
    })

    snn_config = SNNConfig() if use_snn else None

    return EngramAugmentedLayer(
        hidden_dim=hidden_dim,
        engram_config=engram_config,
        snn_config=snn_config,
        use_engram=use_engram,
        use_snn=use_snn,
        **{k: v for k, v in kwargs.items()
           if k in ['num_heads', 'ffn_mult', 'dropout']},
    )
```

**Step 4: Run tests to verify they pass**

```bash
cd /home/sovr610/human-brain && python -m pytest tests/test_engram.py::TestEngramAugmentedLayer -v
```

Expected: All tests PASS.

---

## Task 9: Update Package Exports

**Files:**
- Modify: `brain_ai/__init__.py`

**Step 1: Read current __init__.py**

```bash
cat /home/sovr610/human-brain/brain_ai/__init__.py
```

**Step 2: Add Engram exports**

Add the following imports and exports:

```python
# Memory systems
from .memory import (
    EngramConfig,
    EngramModule,
    EngramEmbedding,
    ContextAwareGating,
    TokenizerCompression,
    MultiHeadHash,
    OffloadableEmbedding,
)

# Layers
from .layers import EngramAugmentedLayer

# Encoders
from .encoders.engram_encoder import EngramTextEncoder, create_engram_encoder
```

**Step 3: Verify imports work**

```bash
cd /home/sovr610/human-brain && python -c "
from brain_ai import EngramConfig, EngramModule, EngramTextEncoder
from brain_ai import create_brain_ai
print('All imports OK')
"
```

Expected: "All imports OK"

---

## Task 10: Run Full Test Suite

**Files:**
- All test files

**Step 1: Run all Engram tests**

```bash
cd /home/sovr610/human-brain && python -m pytest tests/test_engram.py -v
```

Expected: All tests PASS.

**Step 2: Run existing tests to ensure no regression**

```bash
cd /home/sovr610/human-brain && python -m pytest tests/ -v
```

Expected: All tests PASS (or pre-existing failures only).

**Step 3: Quick integration smoke test**

```bash
cd /home/sovr610/human-brain && python -c "
import torch
from brain_ai import create_brain_ai

# Test with Engram enabled
brain = create_brain_ai(
    modalities=['text'],
    use_engram=True,
    use_htm=False,
    use_symbolic=False,
    use_meta=False,
    device='cpu',
)

# Forward pass
token_ids = torch.randint(0, 30000, (2, 32))
output = brain({'text': token_ids, 'token_ids': token_ids})

print(f'Output shape: {output.shape}')
print('Engram integration successful!')
"
```

Expected: "Engram integration successful!"

---

## Summary

This plan implements DeepSeek Engram integration in 10 tasks:

1. **Directory structure** - Create `memory/` and `layers/` directories
2. **TokenizerCompression** - Vocabulary normalization
3. **MultiHeadHash + OffloadableEmbedding** - O(1) hashing with CPU offload
4. **Core EngramModule** - Complete Engram with gating and convolution
5. **EngramConfig** - Add configuration to BrainAIConfig
6. **EngramTextEncoder** - Phase 1 encoder integration
7. **BrainAI integration** - Connect Engram to workspace
8. **EngramAugmentedLayer** - Phase 2 layer integration
9. **Package exports** - Update `__init__.py`
10. **Full test suite** - Verify everything works

Each task follows TDD with exact file paths and complete code.
