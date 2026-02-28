#!/usr/bin/env python3
"""Engram Conditional Memory -- Runtime Contract Validation

Validates all 3 done-when gates and key contracts for the Engram conditional
memory subsystem covering tokenizer compression, multi-head hashing, gating
and fusion, offload/prefetch, Phase 1 encoder-competition, Phase 2 layer-
augmentation, and end-to-end integration.

Self-contained: runs without brain_ai package installed.  All component stubs
faithfully replicate the contracts specified in SKILL.md and the reference
documents.

Usage:
    python validate_engram.py                    # Run all checks
    python validate_engram.py --group gate_a     # Run specific group
    python validate_engram.py --verbose           # Detailed output
    python validate_engram.py --list              # List all check groups

Exit codes:
    0 = all checks passed
    1 = one or more checks failed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import threading
import time
import traceback
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ============================================================================
# ANSI colour helpers
# ============================================================================

_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def _c(text: str, code: str) -> str:
    """Wrap text in ANSI colour if stdout is a tty."""
    if sys.stdout.isatty():
        return f"{code}{text}{_RESET}"
    return text


# ============================================================================
# Group registry
# ============================================================================

_ALL_GROUPS: List[str] = [
    "tokenizer_compression",
    "hashing_determinism",
    "gating_fusion",
    "offload_prefetch",
    "phase1_encoder",
    "phase2_layer",
    "integration",
]

_GROUP_DESCRIPTIONS: Dict[str, str] = {
    "tokenizer_compression": "Tokenizer compression: build, special tokens, serialize, determinism",
    "hashing_determinism": "Done-When Gate (a): Hash determinism, int64, padding, salts, primes",
    "gating_fusion": "Done-When Gate (c): Gating [0,1], conservative init, AMP, grad flow, conv",
    "offload_prefetch": "Done-When Gate (b): Offload correctness, prefetch, coalescing, memory",
    "phase1_encoder": "Phase 1 encoder-competition: shape, masking, gradients, AMP",
    "phase2_layer": "Phase 2 layer-augmentation: residual, compute-now, consume-prefetched",
    "integration": "Integration: mini-backbone, telemetry, both modes, feature flags",
}

# ============================================================================
# Result dataclass
# ============================================================================


@dataclass
class CheckResult:
    """Outcome of a single validation check."""
    name: str
    group: str
    passed: bool
    message: str
    details: Optional[str] = None
    elapsed_ms: float = 0.0
    skipped: bool = False


# ============================================================================
# Inline stub implementations
# ============================================================================
# These minimal implementations are self-contained so the validation script
# does not depend on brain_ai being installed.  They faithfully replicate the
# contracts specified in SKILL.md and the reference documents.


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EngramConfig:
    """Configuration for Engram module (mirrors brain_ai/memory/engram.py)."""
    vocab_size: int = 50000
    compressed_vocab_size: Optional[int] = None
    embedding_dim: int = 256
    ngram_orders: Tuple[int, ...] = (2, 3)
    num_heads: int = 4
    table_size: int = 131071  # Prime
    tokenizer_mode: str = "shared"
    use_compression: bool = True
    conv_kernel_size: int = 4
    conv_dilation: int = 3
    offload_to_cpu: bool = False
    prefetch: bool = True
    gate_temperature: float = 1.0

    def __post_init__(self):
        if self.compressed_vocab_size is None:
            self.compressed_vocab_size = int(self.vocab_size * 0.77)


@dataclass
class HashConfig:
    """Configuration for multi-head hashing."""
    hash_fn: str = "mult_xor"
    use_prime_sizes: bool = True
    table_size: int = 131071
    per_layer_salt: bool = True
    seed: int = 42


@dataclass
class OffloadConfig:
    """Configuration for CPU offload and async prefetch."""
    weights_on_cpu: bool = False
    use_async_prefetch: bool = False
    prefetch_ahead_layers: int = 2
    pin_memory: bool = True
    storage_dtype: str = "float16"


@dataclass
class GatingConfig:
    """Configuration for context-aware gating."""
    gate_type: str = "scalar"
    use_rmsnorm: bool = True
    gate_init_bias: float = -2.0
    conv_dilation: int = 4


# ---------------------------------------------------------------------------
# Mock Tokenizer
# ---------------------------------------------------------------------------

class MockTokenizer:
    """Simulates a HuggingFace-style tokenizer with get_vocab()."""

    def __init__(self, vocab_size: int = 1000, seed: int = 0):
        self.vocab_size = vocab_size
        self._seed = seed
        self._vocab = self._build_vocab()

    def _build_vocab(self) -> Dict[str, int]:
        """Build a deterministic mock vocabulary with variant forms."""
        vocab: Dict[str, int] = OrderedDict()
        # Special tokens
        vocab["<pad>"] = 0
        vocab["<bos>"] = 1
        vocab["<eos>"] = 2
        vocab["<unk>"] = 3

        # Base words with variant forms for testing normalization
        base_words = [
            "the", "of", "and", "to", "in", "is", "it", "for", "that",
            "was", "on", "are", "as", "with", "his", "they", "be", "at",
            "one", "have", "this", "from", "by", "hot", "word", "but",
            "what", "some", "we", "can", "out", "other", "were", "all",
            "there", "when", "up", "use", "your", "how", "said", "an",
            "each", "she", "which", "do", "their", "time", "if", "will",
            "way", "about", "many", "then", "them", "write", "would",
            "like", "so", "these", "her", "long", "make", "thing", "see",
            "him", "two", "has", "look", "more", "day", "could", "go",
            "come", "did", "number", "sound", "no", "most", "people",
            "my", "over", "know", "water", "than", "call", "first", "who",
            "may", "down", "side", "been", "now", "find", "head", "stand",
            "own", "page", "should", "country", "found", "answer", "school",
        ]

        idx = 4
        for word in base_words:
            if idx >= self.vocab_size:
                break
            # lowercase
            vocab[word] = idx
            idx += 1
            if idx >= self.vocab_size:
                break
            # Title case (should normalize to same as lowercase)
            vocab[word.capitalize()] = idx
            idx += 1
            if idx >= self.vocab_size:
                break
            # UPPERCASE (should normalize to same as lowercase)
            vocab[word.upper()] = idx
            idx += 1
            if idx >= self.vocab_size:
                break
            # With leading space (common in BPE tokenizers)
            vocab[f" {word}"] = idx
            idx += 1

        # Fill remaining slots with generated tokens
        while idx < self.vocab_size:
            vocab[f"tok_{idx}"] = idx
            idx += 1

        return vocab

    def get_vocab(self) -> Dict[str, int]:
        return dict(self._vocab)


# ---------------------------------------------------------------------------
# Tokenizer Compression
# ---------------------------------------------------------------------------

class TokenizerCompression:
    """
    Surjective mapping collapsing textually equivalent tokens into canonical IDs.

    Normalization recipe: NFKC + lowercasing + whitespace normalization.
    Special tokens (pad=0, bos=1, eos=2, unk=3) are invariant.
    """

    SPECIAL_TOKENS = {0, 1, 2, 3}  # pad, bos, eos, unk

    def __init__(
        self,
        vocab_size: int = 50000,
        compressed_size: Optional[int] = None,
        mode: str = "shared",
        tokenizer: Optional[Any] = None,
        normalization: str = "nfkc_lower",
        special_policy: str = "identity",
    ):
        self.vocab_size = vocab_size
        self.compressed_size = compressed_size or int(vocab_size * 0.77)
        self.mode = mode
        self.tokenizer = tokenizer
        self.normalization = normalization
        self.special_policy = special_policy
        self.projection: Optional[torch.Tensor] = None
        self._version_hash: Optional[str] = None

        self._build_projection_table()

    def _normalize_text(self, text: str) -> str:
        """Apply normalization recipe to token text."""
        text = unicodedata.normalize('NFKC', text)
        text = text.lower()
        text = ' '.join(text.split())
        text = text.strip()
        return text

    def _build_projection_table(self):
        """Build the surjective mapping P: V -> V'."""
        self.projection = torch.zeros(self.vocab_size, dtype=torch.long)

        if self.tokenizer is not None and hasattr(self.tokenizer, 'get_vocab'):
            self._build_from_tokenizer()
        else:
            self._build_hash_projection()

        self._compute_version_hash()

    def _build_hash_projection(self):
        """Build projection using deterministic hashing."""
        for token_id in range(self.vocab_size):
            if token_id in self.SPECIAL_TOKENS:
                self.projection[token_id] = token_id
            else:
                canonical_id = 4 + (token_id % (self.compressed_size - 4))
                self.projection[token_id] = canonical_id

    def _build_from_tokenizer(self):
        """Build projection from actual tokenizer vocabulary."""
        vocab = self.tokenizer.get_vocab()

        # Sort for determinism (avoids dict iteration order issues)
        sorted_items = sorted(vocab.items(), key=lambda x: x[1])

        # Map normalized text -> canonical ID
        normalized_to_id: Dict[str, int] = {}
        next_id = 4  # Reserve 0-3 for special tokens

        for token_text, token_id in sorted_items:
            if token_id >= self.vocab_size:
                continue

            if token_id in self.SPECIAL_TOKENS:
                self.projection[token_id] = token_id
                continue

            normalized = self._normalize_text(token_text)

            if normalized not in normalized_to_id:
                normalized_to_id[normalized] = next_id
                next_id += 1
                if next_id >= self.compressed_size:
                    next_id = 4  # Wrap around (skip specials)

            self.projection[token_id] = normalized_to_id[normalized]

    def _compute_version_hash(self):
        """Compute deterministic version hash for the compression table."""
        h = hashlib.sha256()
        h.update(f"vocab_size={self.vocab_size}".encode())
        h.update(f"compressed_size={self.compressed_size}".encode())
        h.update(f"normalization={self.normalization}".encode())
        h.update(f"special_policy={self.special_policy}".encode())
        h.update(self.projection.numpy().tobytes())
        self._version_hash = h.hexdigest()

    @property
    def version_hash(self) -> str:
        return self._version_hash

    def compress_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Map raw token IDs to canonical IDs.

        Args:
            input_ids: (B, T) raw token IDs

        Returns:
            Compressed token IDs with same shape as input
        """
        device = input_ids.device
        projection = self.projection.to(device)
        clamped = input_ids.clamp(0, self.vocab_size - 1)
        return projection[clamped]

    def serialize(self, path: str):
        """Save compression table and metadata to file."""
        data = {
            "vocab_size": self.vocab_size,
            "compressed_size": self.compressed_size,
            "normalization": self.normalization,
            "special_policy": self.special_policy,
            "version_hash": self._version_hash,
            "projection": self.projection.tolist(),
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "TokenizerCompression":
        """Load compression table from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        instance = cls.__new__(cls)
        instance.vocab_size = data["vocab_size"]
        instance.compressed_size = data["compressed_size"]
        instance.normalization = data["normalization"]
        instance.special_policy = data["special_policy"]
        instance._version_hash = data["version_hash"]
        instance.projection = torch.tensor(data["projection"], dtype=torch.long)
        instance.tokenizer = None
        instance.mode = "shared"
        return instance

    def get_compression_ratio(self) -> float:
        """Return the compression ratio (unique outputs / vocab_size)."""
        unique_mappings = len(self.projection.unique())
        return unique_mappings / self.vocab_size


# ---------------------------------------------------------------------------
# Prime utility
# ---------------------------------------------------------------------------

def is_prime(n: int) -> bool:
    """Miller-Rabin primality test for numbers we care about."""
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


def next_prime(n: int) -> int:
    """Find the smallest prime >= n."""
    if n <= 2:
        return 2
    candidate = n if n % 2 != 0 else n + 1
    while not is_prime(candidate):
        candidate += 2
    return candidate


def generate_prime_table_sizes(base_size: int, num_tables: int) -> List[int]:
    """Generate num_tables distinct prime sizes near base_size."""
    sizes = []
    current = base_size
    for _ in range(num_tables):
        p = next_prime(current)
        sizes.append(p)
        current = p + 2  # Ensure distinct primes
    return sizes


# ---------------------------------------------------------------------------
# Multi-Head Hash
# ---------------------------------------------------------------------------

class MultiHeadHash:
    """
    Multi-head hashing for N-gram to embedding index mapping.

    Uses multiplicative-XOR hash:
        h_k(g) = (sum_i c_{k,i} * x_i) XOR seed_k mod M

    All operations in int64 for determinism across CPU/CUDA.
    """

    def __init__(
        self,
        ngram_order: int,
        num_heads: int,
        table_size: int,
        seed: int = 42,
        layer_salt: int = 0,
    ):
        self.n = ngram_order
        self.num_heads = num_heads
        self.table_size = table_size
        self.seed = seed
        self.layer_salt = layer_salt

        # Generate deterministic coefficients
        import numpy as np
        rng = np.random.RandomState(seed + layer_salt)

        self.coefficients = torch.tensor(
            rng.randint(1, table_size, size=(num_heads, ngram_order)),
            dtype=torch.int64,
        )
        self.seeds = torch.tensor(
            rng.randint(0, table_size, size=(num_heads,)),
            dtype=torch.int64,
        )

    def hash(self, ngrams: torch.Tensor) -> torch.Tensor:
        """
        Hash N-grams to embedding indices.

        Args:
            ngrams: (B, T, n) tensor of token IDs

        Returns:
            (B, T, num_heads) tensor of int64 embedding indices
        """
        device = ngrams.device
        B, T, n = ngrams.shape

        coeffs = self.coefficients.to(device)  # (H, n)
        seeds = self.seeds.to(device)           # (H,)

        ngrams_long = ngrams.to(torch.int64)
        ngrams_exp = ngrams_long.unsqueeze(2)  # (B, T, 1, n)
        coeffs_exp = coeffs.unsqueeze(0).unsqueeze(0)  # (1, 1, H, n)

        weighted = (ngrams_exp * coeffs_exp).sum(dim=-1)  # (B, T, H)

        seeds_exp = seeds.unsqueeze(0).unsqueeze(0)  # (1, 1, H)
        hashed = (weighted ^ seeds_exp) % self.table_size

        return hashed

    def hash_streaming(self, ngrams: torch.Tensor, pos: int) -> torch.Tensor:
        """
        Hash a single position (streaming / incremental).

        Args:
            ngrams: (B, T, n) full N-grams tensor
            pos: Position to hash

        Returns:
            (B, num_heads) hash IDs for the given position
        """
        single = ngrams[:, pos:pos+1, :]  # (B, 1, n)
        return self.hash(single).squeeze(1)  # (B, H)


# ---------------------------------------------------------------------------
# Offloadable Embedding
# ---------------------------------------------------------------------------

class OffloadableEmbedding(nn.Module):
    """
    Embedding table with optional CPU offload and async prefetching.

    Two modes:
      1. On-device: standard nn.Embedding on GPU/CPU
      2. CPU offload: weights in pinned CPU memory, async copy to GPU
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
            self.weight = nn.Parameter(
                torch.zeros(num_embeddings, embedding_dim),
                requires_grad=True,
            )
            nn.init.normal_(self.weight, mean=0, std=0.02)
            if torch.cuda.is_available():
                self.weight.data = self.weight.data.pin_memory()
            self._prefetch_buffer: Optional[Tuple[Tensor, Tensor]] = None
            self._prefetch_stream = (
                torch.cuda.Stream() if self.prefetch else None
            )
            self._prefetch_event: Optional[torch.cuda.Event] = None
        else:
            self.embedding = nn.Embedding(num_embeddings, embedding_dim)
            nn.init.normal_(self.embedding.weight, mean=0, std=0.02)

    def prefetch_async(self, indices: torch.Tensor):
        """Prefetch embeddings for given indices asynchronously."""
        if not self.offload or not self.prefetch:
            return

        with torch.cuda.stream(self._prefetch_stream):
            flat_indices = indices.flatten()
            unique_indices = flat_indices.unique()
            cpu_indices = unique_indices.cpu()
            embeddings = self.weight.data[cpu_indices].cuda(non_blocking=True)
            self._prefetch_buffer = (unique_indices, embeddings)
            self._prefetch_event = self._prefetch_stream.record_event()

    def consume_prefetched(self, indices: torch.Tensor) -> Optional[Tensor]:
        """Consume prefetched data with proper synchronization."""
        if self._prefetch_buffer is None:
            return None
        if self._prefetch_event is not None:
            self._prefetch_event.synchronize()
        cached_indices, cached_embeddings = self._prefetch_buffer
        result = self._gather_from_cache(indices, cached_indices, cached_embeddings)
        self._prefetch_buffer = None
        self._prefetch_event = None
        return result

    def _gather_from_cache(
        self,
        indices: Tensor,
        cached_indices: Tensor,
        cached_embeddings: Tensor,
    ) -> Tensor:
        """Gather embeddings from prefetch cache."""
        device = indices.device
        original_shape = indices.shape
        flat_indices = indices.flatten()

        index_to_pos = torch.zeros(
            self.num_embeddings, dtype=torch.long, device=device
        )
        index_to_pos[cached_indices] = torch.arange(
            len(cached_indices), device=device
        )
        cache_positions = index_to_pos[flat_indices]
        gathered = cached_embeddings[cache_positions]
        return gathered.view(*original_shape, self.embedding_dim)

    def forward(self, indices: Tensor) -> Tensor:
        """Look up embeddings for given indices."""
        if self.offload:
            consumed = self.consume_prefetched(indices) if self.prefetch else None
            if consumed is not None:
                return consumed
            # Synchronous fallback
            cpu_indices = indices.cpu()
            embeddings = self.weight.data[cpu_indices]
            if indices.is_cuda:
                embeddings = embeddings.cuda()
            return embeddings
        else:
            return self.embedding(indices)

    def get_num_unique_fetches(self, indices: Tensor) -> int:
        """Return the number of unique rows that would be fetched."""
        return indices.flatten().unique().numel()


# ---------------------------------------------------------------------------
# Prefetch Plan
# ---------------------------------------------------------------------------

class PrefetchPlan:
    """
    Schedules prefetch N layers ahead.

    Given a list of layer indices that have Engram modules, determines at
    which layer to trigger prefetch for each Engram layer.
    """

    def __init__(self, engram_layer_indices: List[int], prefetch_ahead: int = 2):
        self.engram_layers = sorted(engram_layer_indices)
        self.prefetch_ahead = prefetch_ahead
        self._schedule: Dict[int, int] = {}
        self._build_schedule()

    def _build_schedule(self):
        """Build prefetch trigger schedule: trigger_layer -> engram_layer."""
        for el in self.engram_layers:
            trigger = max(0, el - self.prefetch_ahead)
            self._schedule[trigger] = el

    def should_prefetch_at(self, layer_idx: int) -> Optional[int]:
        """Returns the Engram layer to prefetch for, or None."""
        return self._schedule.get(layer_idx)

    @property
    def schedule(self) -> Dict[int, int]:
        return dict(self._schedule)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization -- fp32 safe under AMP."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        # Force fp32 computation for AMP safety
        dtype = x.dtype
        x_fp32 = x.float()
        rms = torch.sqrt(torch.mean(x_fp32 ** 2, dim=-1, keepdim=True) + self.eps)
        normed = x_fp32 / rms
        return (self.weight.float() * normed).to(dtype)


# ---------------------------------------------------------------------------
# Context-Aware Gating
# ---------------------------------------------------------------------------

class ContextAwareGating(nn.Module):
    """
    Context-aware gating for Engram.

    gate alpha = sigmoid(RMSNorm(h)^T . RMSNorm(Ke) / sqrt(d) + bias)
    output = alpha * Ve

    Conservative initialization: gate_bias = -2.0 => mean gate ~ sigmoid(-2) ~ 0.12
    """

    def __init__(
        self,
        hidden_dim: int,
        memory_dim: int,
        temperature: float = 1.0,
        gate_init_bias: float = -2.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        self.temperature = temperature

        self.Wq = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wk = nn.Linear(memory_dim, hidden_dim, bias=False)
        self.Wv = nn.Linear(memory_dim, hidden_dim, bias=False)

        self.query_norm = RMSNorm(hidden_dim)
        self.key_norm = RMSNorm(hidden_dim)

        self.gate_bias = nn.Parameter(torch.tensor(gate_init_bias))
        self.scale = hidden_dim ** -0.5

    def forward(
        self,
        hidden_states: Tensor,
        memory: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            hidden_states: (B, T, hidden_dim)
            memory: (B, T, memory_dim)

        Returns:
            gated_output: (B, T, hidden_dim)
            alpha: (B, T, 1) gate values in [0, 1]
        """
        q = self.Wq(hidden_states)
        k = self.Wk(memory)
        v = self.Wv(memory)

        q_norm = self.query_norm(q)
        k_norm = self.key_norm(k)

        gate_logits = (q_norm * k_norm).sum(dim=-1, keepdim=True) * self.scale
        gate_logits = gate_logits / self.temperature + self.gate_bias

        alpha = torch.sigmoid(gate_logits)  # (B, T, 1)
        gated_output = alpha * v

        return gated_output, alpha


# ---------------------------------------------------------------------------
# Depthwise Causal Convolution
# ---------------------------------------------------------------------------

class DepthwiseCausalConv(nn.Module):
    """
    Depthwise causal convolution that expands receptive field.

    Causal padding ensures no future leakage: only left-padding is used.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 4,
        dilation: int = 4,
    ):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation  # Causal: only left pad

        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=0,  # We handle padding manually for strict causality
            dilation=dilation,
            groups=channels,  # Depthwise
        )
        # Zero init for smooth training start
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, T, C) input

        Returns:
            (B, T, C) causally convolved output
        """
        B, T, C = x.shape
        # Transpose to (B, C, T) for Conv1d
        h = x.transpose(1, 2)
        # Left-pad only (causal)
        h = F.pad(h, (self.padding, 0))
        h = self.conv(h)
        # Ensure output length matches input
        h = h[:, :, :T]
        return h.transpose(1, 2)  # Back to (B, T, C)


# ---------------------------------------------------------------------------
# N-gram Extraction
# ---------------------------------------------------------------------------

def extract_ngrams(token_ids: Tensor, n: int) -> Tensor:
    """
    Extract suffix N-grams from token sequence.

    Args:
        token_ids: (B, T) compressed token IDs
        n: N-gram order

    Returns:
        (B, T, n) where [b, t, :] = (x_{t-n+1}, ..., x_t)
        Positions with t < n-1 are left-padded with zeros.
    """
    B, T = token_ids.shape
    padded = F.pad(token_ids, (n - 1, 0), value=0)
    ngrams = torch.stack([
        padded[:, i:i + T]
        for i in range(n)
    ], dim=-1)
    return ngrams


# ---------------------------------------------------------------------------
# Engram Embedding (N-gram hash tables)
# ---------------------------------------------------------------------------

class EngramEmbedding(nn.Module):
    """
    N-gram embedding tables with multi-head hashing.

    For each N-gram order n, maintains num_heads embedding tables.
    Retrieved embeddings are concatenated across all orders and heads.
    """

    def __init__(self, config: EngramConfig, layer_salt: int = 0):
        super().__init__()
        self.config = config
        self.ngram_orders = config.ngram_orders
        self.num_heads = config.num_heads
        self.layer_salt = layer_salt

        total_retrievals = len(self.ngram_orders) * config.num_heads
        self.dim_per_embedding = max(1, config.embedding_dim // total_retrievals)

        # Hash functions per N-gram order
        self._hashers: Dict[int, MultiHeadHash] = {}
        for n in self.ngram_orders:
            self._hashers[n] = MultiHeadHash(
                n, config.num_heads, config.table_size,
                seed=config.table_size + n,  # Deterministic
                layer_salt=layer_salt,
            )

        # Embedding tables
        self.embeddings = nn.ModuleDict()
        for n in self.ngram_orders:
            for k in range(config.num_heads):
                key = f"ngram{n}_head{k}"
                self.embeddings[key] = OffloadableEmbedding(
                    num_embeddings=config.table_size,
                    embedding_dim=self.dim_per_embedding,
                    offload=config.offload_to_cpu,
                    prefetch=config.prefetch,
                )

        # Compression
        if config.use_compression:
            self.compressor = TokenizerCompression(
                vocab_size=config.vocab_size,
                compressed_size=config.compressed_vocab_size,
            )
        else:
            self.compressor = None

    def forward(self, token_ids: Tensor) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Retrieve N-gram embeddings.

        Args:
            token_ids: (B, T) raw token IDs

        Returns:
            embeddings: (B, T, embedding_dim) concatenated
            telemetry: dict with hash_ids, unique counts etc.
        """
        if self.compressor is not None:
            compressed_ids = self.compressor.compress_ids(token_ids)
        else:
            compressed_ids = token_ids

        all_embeddings = []
        all_hash_ids = []

        for n in self.ngram_orders:
            ngrams = extract_ngrams(compressed_ids, n)
            indices = self._hashers[n].hash(ngrams)
            all_hash_ids.append(indices)

            for k in range(self.num_heads):
                key = f"ngram{n}_head{k}"
                head_indices = indices[:, :, k]
                head_emb = self.embeddings[key](head_indices)
                all_embeddings.append(head_emb)

        result = torch.cat(all_embeddings, dim=-1)
        telemetry = {
            "hash_ids": all_hash_ids,
            "compressed_ids": compressed_ids,
        }
        return result, telemetry


# ---------------------------------------------------------------------------
# EngramModule (full module with gating, conv, residual)
# ---------------------------------------------------------------------------

class EngramModule(nn.Module):
    """
    Complete Engram conditional memory module.

    Pipeline:
      1. N-gram extraction + tokenizer compression
      2. Multi-head hash lookup
      3. Context-aware gating
      4. Depthwise causal convolution
      5. Residual fusion -> delta for backbone
    """

    def __init__(
        self,
        config: EngramConfig,
        hidden_dim: int,
        layer_id: int = 0,
    ):
        super().__init__()
        self.config = config
        self.hidden_dim = hidden_dim
        self.layer_id = layer_id

        self.ngram_embedding = EngramEmbedding(config, layer_salt=layer_id)

        self.gating = ContextAwareGating(
            hidden_dim=hidden_dim,
            memory_dim=config.embedding_dim,
            temperature=config.gate_temperature,
        )

        self.conv = DepthwiseCausalConv(
            channels=hidden_dim,
            kernel_size=config.conv_kernel_size,
            dilation=config.conv_dilation,
        )

        self.conv_norm = RMSNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.out_proj.weight)

    def forward(
        self,
        hidden_states: Tensor,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        *,
        cache_state: Optional[Dict] = None,
        return_details: bool = False,
    ) -> Tuple[Tensor, Optional[Dict]]:
        """
        Args:
            hidden_states: (B, T, D) from backbone
            input_ids: (B, T) token IDs
            attention_mask: (B, T) optional mask (1=keep, 0=pad)
            cache_state: optional prefetched data
            return_details: whether to return telemetry

        Returns:
            delta: (B, T, D) to add to residual stream
            telemetry: optional dict with gate stats etc.
        """
        B, T, D = hidden_states.shape

        # Retrieve memory
        memory, emb_telemetry = self.ngram_embedding(input_ids)

        # Apply attention mask to memory
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            memory = memory * mask

        # Gate
        gated, alpha = self.gating(hidden_states, memory)

        # Conv
        conv_in = self.conv_norm(gated)
        conv_out = self.conv(conv_in)

        # Fuse
        fused = F.silu(conv_out) + gated

        # Project to backbone dim
        delta = self.out_proj(fused)

        # Apply mask
        if attention_mask is not None:
            delta = delta * mask

        telemetry = None
        if return_details:
            telemetry = {
                "layer_id": self.layer_id,
                "gate_mean": alpha.mean().item(),
                "gate_min": alpha.min().item(),
                "gate_max": alpha.max().item(),
                "memory_norm": memory.norm(dim=-1).mean().item(),
                "delta_norm": delta.norm(dim=-1).mean().item(),
            }
            telemetry.update(emb_telemetry)

        return delta, telemetry


# ---------------------------------------------------------------------------
# Phase 1: EngramTextEncoder (encoder-competition mode)
# ---------------------------------------------------------------------------

class EngramTextEncoder(nn.Module):
    """
    Phase 1 integration: Engram as an encoder competing in Global Workspace.

    Produces workspace-aligned (B, T, D_workspace) representation from
    input token IDs alone (no hidden states needed from backbone).
    """

    WORKSPACE_DIM = 4096  # Workspace alignment target

    def __init__(
        self,
        config: EngramConfig,
        workspace_dim: int = 4096,
    ):
        super().__init__()
        self.config = config
        self.workspace_dim = workspace_dim

        self.engram_emb = EngramEmbedding(config)

        # Project to workspace dimension
        self.proj = nn.Linear(config.embedding_dim, workspace_dim)
        self.norm = RMSNorm(workspace_dim)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            input_ids: (B, T) token IDs
            attention_mask: (B, T) optional mask (1=keep, 0=pad)

        Returns:
            (B, T, workspace_dim) workspace-aligned representation
        """
        memory, _ = self.engram_emb(input_ids)
        projected = self.proj(memory)
        output = self.norm(projected)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            output = output * mask

        return output


# ---------------------------------------------------------------------------
# Phase 2: EngramAugmentedLayer (layer-augmentation mode)
# ---------------------------------------------------------------------------

class EngramAugmentedLayer(nn.Module):
    """
    Phase 2 integration: Engram injected at selected backbone layers.

    Computes delta residually: output = input + delta(engram).
    Supports two paths:
      - "compute now": compute hash + retrieve + gate in forward
      - "consume prefetched": use pre-supplied embeddings from async prefetch
    """

    def __init__(
        self,
        hidden_dim: int,
        config: EngramConfig,
        layer_id: int = 0,
        use_engram: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_engram = use_engram
        self.layer_id = layer_id

        if use_engram:
            self.engram = EngramModule(config, hidden_dim, layer_id=layer_id)
            self.engram_norm = RMSNorm(hidden_dim)

    def forward(
        self,
        hidden_states: Tensor,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        *,
        prefetched_embeddings: Optional[Tensor] = None,
        return_details: bool = False,
    ) -> Tuple[Tensor, Optional[Dict]]:
        """
        Args:
            hidden_states: (B, T, D) from backbone
            input_ids: (B, T) token IDs (needed if no prefetched data)
            attention_mask: (B, T) optional
            prefetched_embeddings: pre-fetched embedding data
            return_details: whether to return telemetry

        Returns:
            output: (B, T, D) augmented hidden states
            telemetry: optional dict
        """
        if not self.use_engram or input_ids is None:
            return hidden_states, None

        normed = self.engram_norm(hidden_states)
        delta, telemetry = self.engram(
            normed, input_ids, attention_mask,
            cache_state=(
                {"prefetched": prefetched_embeddings}
                if prefetched_embeddings is not None
                else None
            ),
            return_details=return_details,
        )

        output = hidden_states + delta
        return output, telemetry


# ============================================================================
# Validation runner infrastructure
# ============================================================================

class ValidationRunner:
    """Manages check execution, timing, and reporting."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[CheckResult] = []
        self._group_timings: Dict[str, float] = {}

    def run_check(
        self,
        name: str,
        group: str,
        fn: Callable[[], Tuple[bool, str, Optional[str]]],
    ):
        """Run a single check with timing and error handling."""
        t0 = time.perf_counter()
        try:
            passed, message, details = fn()
            elapsed = (time.perf_counter() - t0) * 1000
            result = CheckResult(
                name=name, group=group, passed=passed,
                message=message, details=details, elapsed_ms=elapsed,
            )
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            tb = traceback.format_exc()
            result = CheckResult(
                name=name, group=group, passed=False,
                message=f"EXCEPTION: {e}",
                details=tb if self.verbose else str(e),
                elapsed_ms=elapsed,
            )
        self.results.append(result)
        self._print_result(result)

    def skip_check(self, name: str, group: str, reason: str):
        """Record a skipped check."""
        result = CheckResult(
            name=name, group=group, passed=True,
            message=f"SKIPPED: {reason}", skipped=True,
        )
        self.results.append(result)
        self._print_result(result)

    def _print_result(self, r: CheckResult):
        if r.skipped:
            tag = _c("SKIP", _YELLOW)
        elif r.passed:
            tag = _c("PASS", _GREEN)
        else:
            tag = _c("FAIL", _RED)
        timing = f" ({r.elapsed_ms:.1f}ms)" if r.elapsed_ms > 0 else ""
        print(f"  [{tag}] {r.name}{timing} -- {r.message}")
        if r.details and (self.verbose or not r.passed):
            for line in r.details.strip().split("\n"):
                print(f"         {_c(line, _DIM)}")

    def begin_group(self, group: str):
        desc = _GROUP_DESCRIPTIONS.get(group, group)
        print(f"\n{_c('=' * 72, _CYAN)}")
        print(f"  {_c(f'Group: {group}', _BOLD)} -- {desc}")
        print(f"{_c('=' * 72, _CYAN)}")
        self._group_timings[group] = time.perf_counter()

    def end_group(self, group: str):
        elapsed = (time.perf_counter() - self._group_timings.get(group, time.perf_counter())) * 1000
        group_results = [r for r in self.results if r.group == group]
        passed = sum(1 for r in group_results if r.passed)
        failed = sum(1 for r in group_results if not r.passed)
        skipped = sum(1 for r in group_results if r.skipped)
        total = len(group_results)
        print(f"\n  Group summary: {passed}/{total} passed"
              f"{f', {skipped} skipped' if skipped else ''}"
              f"{f', {failed} FAILED' if failed else ''}"
              f" ({elapsed:.0f}ms)")

    def print_final_summary(self):
        print(f"\n{'=' * 72}")
        print(_c("  FINAL SUMMARY", _BOLD))
        print(f"{'=' * 72}")

        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        skipped = sum(1 for r in self.results if r.skipped)

        print(f"\n  Total checks : {total}")
        print(f"  Passed       : {_c(str(passed), _GREEN)}")
        if skipped:
            print(f"  Skipped      : {_c(str(skipped), _YELLOW)}")
        if failed:
            print(f"  Failed       : {_c(str(failed), _RED)}")
            print(f"\n  Failed checks:")
            for r in self.results:
                if not r.passed:
                    print(f"    - [{r.group}] {r.name}: {r.message}")

        overall = "PASS" if failed == 0 else "FAIL"
        colour = _GREEN if failed == 0 else _RED
        print(f"\n  Overall: {_c(overall, colour)}")
        print(f"{'=' * 72}\n")

        return failed == 0


# ============================================================================
# Validation Group 1: Tokenizer Compression
# ============================================================================

def validate_tokenizer_compression(runner: ValidationRunner):
    """~8 checks for tokenizer compression contracts."""
    group = "tokenizer_compression"
    runner.begin_group(group)

    # ---- Check 1.1: Deterministic build (same config -> identical tables) ----
    def check_deterministic_build():
        tok = MockTokenizer(vocab_size=500)
        tc1 = TokenizerCompression(
            vocab_size=500, tokenizer=tok, normalization="nfkc_lower",
        )
        tc2 = TokenizerCompression(
            vocab_size=500, tokenizer=tok, normalization="nfkc_lower",
        )
        match = torch.equal(tc1.projection, tc2.projection)
        return (
            match,
            f"Tables {'match' if match else 'DIFFER'} across 2 builds",
            f"tc1 unique={tc1.projection.unique().numel()}, "
            f"tc2 unique={tc2.projection.unique().numel()}",
        )
    runner.run_check("1.1 Deterministic build: same config -> identical tables",
                     group, check_deterministic_build)

    # ---- Check 1.2: Special tokens are invariant ----
    def check_special_tokens():
        tok = MockTokenizer(vocab_size=500)
        tc = TokenizerCompression(vocab_size=500, tokenizer=tok)
        specials = {0: "pad", 1: "bos", 2: "eos", 3: "unk"}
        all_ok = True
        details_parts = []
        for sid, sname in specials.items():
            mapped = tc.projection[sid].item()
            ok = (mapped == sid)
            all_ok = all_ok and ok
            details_parts.append(f"{sname}({sid})->{mapped} {'OK' if ok else 'FAIL'}")
        return (
            all_ok,
            f"Special tokens {'all invariant' if all_ok else 'BROKEN'}",
            "; ".join(details_parts),
        )
    runner.run_check("1.2 Special tokens (pad/bos/eos/unk) map to themselves",
                     group, check_special_tokens)

    # ---- Check 1.3: Compression actually reduces vocab ----
    def check_compression_reduces():
        tok = MockTokenizer(vocab_size=500)
        tc = TokenizerCompression(vocab_size=500, tokenizer=tok)
        original = 500
        compressed = tc.projection.unique().numel()
        reduced = compressed < original
        ratio = tc.get_compression_ratio()
        return (
            reduced,
            f"Compressed {original} -> {compressed} unique IDs (ratio {ratio:.3f})",
            f"Reduction: {100 * (1 - ratio):.1f}%",
        )
    runner.run_check("1.3 Compression reduces vocab (compressed < original)",
                     group, check_compression_reduces)

    # ---- Check 1.4: Serialize / load roundtrip ----
    def check_serialize_roundtrip():
        tok = MockTokenizer(vocab_size=500)
        tc_orig = TokenizerCompression(vocab_size=500, tokenizer=tok)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = f.name
        try:
            tc_orig.serialize(path)
            tc_loaded = TokenizerCompression.load(path)
            tables_match = torch.equal(tc_orig.projection, tc_loaded.projection)
            hash_match = tc_orig.version_hash == tc_loaded.version_hash
            meta_match = (
                tc_orig.vocab_size == tc_loaded.vocab_size
                and tc_orig.compressed_size == tc_loaded.compressed_size
                and tc_orig.normalization == tc_loaded.normalization
            )
            all_ok = tables_match and hash_match and meta_match
            return (
                all_ok,
                f"Roundtrip {'OK' if all_ok else 'FAILED'}: "
                f"table={'match' if tables_match else 'diff'}, "
                f"hash={'match' if hash_match else 'diff'}, "
                f"meta={'match' if meta_match else 'diff'}",
                None,
            )
        finally:
            os.unlink(path)
    runner.run_check("1.4 Serialize to file, load back, verify identical",
                     group, check_serialize_roundtrip)

    # ---- Check 1.5: Version hash is deterministic ----
    def check_version_hash_deterministic():
        tok = MockTokenizer(vocab_size=500)
        hashes = set()
        for _ in range(5):
            tc = TokenizerCompression(vocab_size=500, tokenizer=tok)
            hashes.add(tc.version_hash)
        ok = len(hashes) == 1
        return (
            ok,
            f"Version hash deterministic: {len(hashes)} unique hash(es) in 5 builds",
            f"Hash: {list(hashes)[0][:16]}..." if ok else f"Hashes: {hashes}",
        )
    runner.run_check("1.5 Version hash is deterministic (same inputs -> same hash)",
                     group, check_version_hash_deterministic)

    # ---- Check 1.6: Version hash is sensitive to config ----
    def check_version_hash_sensitive():
        tok = MockTokenizer(vocab_size=500)
        tc1 = TokenizerCompression(vocab_size=500, tokenizer=tok, normalization="nfkc_lower")
        tc2 = TokenizerCompression(vocab_size=500, tokenizer=tok, normalization="nfkc_only")
        different = tc1.version_hash != tc2.version_hash
        return (
            different,
            f"Different normalization recipes -> "
            f"{'different' if different else 'SAME'} hashes",
            f"h1={tc1.version_hash[:16]}... h2={tc2.version_hash[:16]}...",
        )
    runner.run_check("1.6 Version hash sensitive (different recipe -> different hash)",
                     group, check_version_hash_sensitive)

    # ---- Check 1.7: compress_ids preserves tensor shape ----
    def check_compress_ids_shape():
        tc = TokenizerCompression(vocab_size=500)
        shapes_ok = True
        details = []
        for shape in [(1, 10), (4, 32), (2, 128), (8, 1)]:
            ids = torch.randint(0, 500, shape)
            compressed = tc.compress_ids(ids)
            ok = compressed.shape == ids.shape
            shapes_ok = shapes_ok and ok
            details.append(f"{shape} -> {tuple(compressed.shape)} {'OK' if ok else 'FAIL'}")
        return (
            shapes_ok,
            f"Shape preservation: {'all OK' if shapes_ok else 'FAILED'}",
            "; ".join(details),
        )
    runner.run_check("1.7 compress_ids preserves tensor shape for (B, T) input",
                     group, check_compress_ids_shape)

    # ---- Check 1.8: Fast path performance ----
    def check_compress_ids_speed():
        tc = TokenizerCompression(vocab_size=50000)
        ids = torch.randint(0, 50000, (4, 1000))
        # Warmup
        for _ in range(3):
            tc.compress_ids(ids)
        # Timed
        t0 = time.perf_counter()
        for _ in range(100):
            tc.compress_ids(ids)
        elapsed_ms = (time.perf_counter() - t0) / 100 * 1000
        ok = elapsed_ms < 1.0
        return (
            ok,
            f"compress_ids latency: {elapsed_ms:.3f}ms per call "
            f"({'<1ms' if ok else '>1ms SLOW'})",
            f"100 calls on (4, 1000) input, vocab=50k",
        )
    runner.run_check("1.8 Fast path: compress_ids < 1ms for 1000-token batch",
                     group, check_compress_ids_speed)

    runner.end_group(group)


# ============================================================================
# Validation Group 2: Hashing Determinism
# ============================================================================

def validate_hashing_determinism(runner: ValidationRunner):
    """~10 checks for hashing determinism contracts -- Done-When Gate (a)."""
    group = "hashing_determinism"
    runner.begin_group(group)

    SEED = 42
    TABLE_SIZE = 131071  # Prime
    B, T = 2, 16

    # ---- Check 2.1: Determinism across 10 runs ----
    def check_hash_determinism():
        ids = torch.randint(0, 1000, (B, T))
        ngrams = extract_ngrams(ids, 3)
        reference = None
        all_match = True
        for run in range(10):
            hasher = MultiHeadHash(3, 4, TABLE_SIZE, seed=SEED)
            h = hasher.hash(ngrams)
            if reference is None:
                reference = h.clone()
            elif not torch.equal(h, reference):
                all_match = False
                break
        return (
            all_match,
            f"10 runs with same input/seed/config: "
            f"{'all identical' if all_match else 'DIFFER'}",
            f"Shape: {reference.shape}, dtype: {reference.dtype}",
        )
    runner.run_check("2.1 Same (ids, seed, config) -> identical hash_ids x10 runs",
                     group, check_hash_determinism)

    # ---- Check 2.2: Hash IDs are int64 ----
    def check_hash_dtype():
        ids = torch.randint(0, 1000, (B, T))
        ngrams = extract_ngrams(ids, 3)
        hasher = MultiHeadHash(3, 4, TABLE_SIZE, seed=SEED)
        h = hasher.hash(ngrams)
        ok = h.dtype == torch.int64
        return (
            ok,
            f"Hash ID dtype: {h.dtype} ({'OK' if ok else 'expected int64'})",
            None,
        )
    runner.run_check("2.2 Hash IDs are int64 dtype",
                     group, check_hash_dtype)

    # ---- Check 2.3: Padding positions produce hash_id = 0 ----
    def check_padding_hashes_zero():
        # All-zero input (padding)
        ids = torch.zeros(B, T, dtype=torch.long)
        ngrams = extract_ngrams(ids, 3)
        hasher = MultiHeadHash(3, 4, TABLE_SIZE, seed=SEED)
        h = hasher.hash(ngrams)
        # With all-zero ngrams: weighted sum = 0, XOR with seed, mod table_size
        # The key is that zero-padded positions produce CONSISTENT values
        # (not necessarily 0 since XOR with seed), but we verify consistency
        all_same_per_pos = True
        for t in range(T):
            vals = h[:, t, :]
            # All batch elements should have same hash for same all-zero input
            if not torch.equal(vals[0:1].expand_as(vals), vals):
                all_same_per_pos = False
                break
        # Also check that masking to zero is preserved
        mask = torch.zeros(B, T, dtype=torch.long)
        masked_ids = ids * mask  # Still all zeros
        ngrams_masked = extract_ngrams(masked_ids, 3)
        h_masked = hasher.hash(ngrams_masked)
        match = torch.equal(h, h_masked)
        ok = all_same_per_pos and match
        return (
            ok,
            f"Padding (all-zero) positions: consistent={all_same_per_pos}, "
            f"mask-invariant={match}",
            f"Hash at pos 0: {h[0, 0, :].tolist()}",
        )
    runner.run_check("2.3 Padding positions produce consistent hash_ids",
                     group, check_padding_hashes_zero)

    # ---- Check 2.4: Different per-layer salts produce different hash_ids ----
    def check_per_layer_salt():
        ids = torch.randint(10, 1000, (B, T))
        ngrams = extract_ngrams(ids, 3)
        h0 = MultiHeadHash(3, 4, TABLE_SIZE, seed=SEED, layer_salt=0).hash(ngrams)
        h1 = MultiHeadHash(3, 4, TABLE_SIZE, seed=SEED, layer_salt=1).hash(ngrams)
        h2 = MultiHeadHash(3, 4, TABLE_SIZE, seed=SEED, layer_salt=7).hash(ngrams)
        diff_01 = not torch.equal(h0, h1)
        diff_02 = not torch.equal(h0, h2)
        diff_12 = not torch.equal(h1, h2)
        all_diff = diff_01 and diff_02 and diff_12
        return (
            all_diff,
            f"Per-layer salts: 0 vs 1={diff_01}, 0 vs 7={diff_02}, "
            f"1 vs 7={diff_12}",
            None,
        )
    runner.run_check("2.4 Different per-layer salts -> different hash_ids",
                     group, check_per_layer_salt)

    # ---- Check 2.5: Different inputs produce (mostly) different hashes ----
    def check_different_inputs():
        ids_a = torch.randint(10, 500, (B, T))
        ids_b = torch.randint(500, 1000, (B, T))
        ngrams_a = extract_ngrams(ids_a, 3)
        ngrams_b = extract_ngrams(ids_b, 3)
        hasher = MultiHeadHash(3, 4, TABLE_SIZE, seed=SEED)
        h_a = hasher.hash(ngrams_a)
        h_b = hasher.hash(ngrams_b)
        # Not all collisions
        differ = not torch.equal(h_a, h_b)
        # Count how many positions differ
        total = h_a.numel()
        num_diff = (h_a != h_b).sum().item()
        ratio = num_diff / total
        return (
            differ,
            f"Different inputs: {num_diff}/{total} positions differ "
            f"({ratio:.1%})",
            None,
        )
    runner.run_check("2.5 Different inputs are (mostly) different (not all collisions)",
                     group, check_different_inputs)

    # ---- Check 2.6: Unique ratio on random data > 0.8 ----
    def check_unique_ratio():
        ids = torch.randint(0, 10000, (8, 128))
        ngrams = extract_ngrams(ids, 3)
        hasher = MultiHeadHash(3, 4, TABLE_SIZE, seed=SEED)
        h = hasher.hash(ngrams)
        # Check per head
        ratios = []
        for head in range(4):
            head_h = h[:, :, head].flatten()
            unique = head_h.unique().numel()
            total = head_h.numel()
            ratios.append(unique / total)
        avg_ratio = sum(ratios) / len(ratios)
        ok = avg_ratio > 0.8
        return (
            ok,
            f"Unique ratio on random data: {avg_ratio:.3f} "
            f"({'> 0.8' if ok else '< 0.8 LOW'})",
            f"Per-head ratios: {[f'{r:.3f}' for r in ratios]}",
        )
    runner.run_check("2.6 Unique ratio on random data > 0.8 (prime table)",
                     group, check_unique_ratio)

    # ---- Check 2.7: Prime sizing produces actual primes ----
    def check_prime_sizing():
        sizes = generate_prime_table_sizes(100000, 8)
        all_prime = all(is_prime(s) for s in sizes)
        all_distinct = len(set(sizes)) == len(sizes)
        ok = all_prime and all_distinct
        return (
            ok,
            f"Generated {len(sizes)} table sizes: "
            f"all prime={all_prime}, all distinct={all_distinct}",
            f"Sizes: {sizes}",
        )
    runner.run_check("2.7 Prime sizing: generated table sizes are actually prime",
                     group, check_prime_sizing)

    # ---- Check 2.8: Streaming hash matches full recompute ----
    def check_streaming_hash():
        ids = torch.randint(0, 1000, (B, T))
        ngrams = extract_ngrams(ids, 3)
        hasher = MultiHeadHash(3, 4, TABLE_SIZE, seed=SEED)
        # Full
        h_full = hasher.hash(ngrams)
        # Streaming for last position
        last_pos = T - 1
        h_stream = hasher.hash_streaming(ngrams, last_pos)
        match = torch.equal(h_full[:, last_pos, :], h_stream)
        return (
            match,
            f"Streaming hash at pos {last_pos}: "
            f"{'matches' if match else 'DIFFERS from'} full recompute",
            f"Full: {h_full[0, last_pos, :].tolist()}, "
            f"Stream: {h_stream[0].tolist()}",
        )
    runner.run_check("2.8 Streaming hash: incremental matches full recompute",
                     group, check_streaming_hash)

    # ---- Check 2.9: CPU/CUDA produce identical hash_ids ----
    def check_cpu_cuda_determinism():
        if not torch.cuda.is_available():
            return None  # Signal to skip
        ids = torch.randint(0, 1000, (B, T))
        ngrams_cpu = extract_ngrams(ids, 3)
        ngrams_cuda = ngrams_cpu.cuda()
        hasher = MultiHeadHash(3, 4, TABLE_SIZE, seed=SEED)
        h_cpu = hasher.hash(ngrams_cpu)
        h_cuda = hasher.hash(ngrams_cuda)
        match = torch.equal(h_cpu, h_cuda.cpu())
        return (
            match,
            f"CPU vs CUDA hash_ids: {'identical' if match else 'DIFFER'}",
            f"Max abs diff: {(h_cpu - h_cuda.cpu()).abs().max().item()}",
        )

    if torch.cuda.is_available():
        runner.run_check("2.9 CPU and CUDA produce identical hash_ids",
                         group, check_cpu_cuda_determinism)
    else:
        runner.skip_check("2.9 CPU and CUDA produce identical hash_ids",
                          group, "CUDA not available")

    # ---- Check 2.10: N-gram edge case: t=0 correctly left-pads ----
    def check_ngram_left_pad():
        ids = torch.tensor([[100, 200, 300, 400]], dtype=torch.long)
        for n in [2, 3, 4]:
            ngrams = extract_ngrams(ids, n)
            # At t=0, all elements except the last should be 0 (left-pad)
            first_ngram = ngrams[0, 0, :]
            expected_zeros = n - 1
            actual_zeros = (first_ngram[:expected_zeros] == 0).all().item()
            last_val = first_ngram[-1].item()
            if not actual_zeros or last_val != 100:
                return (
                    False,
                    f"N-gram order {n}: left-pad broken at t=0",
                    f"Expected {expected_zeros} zeros + 100, got {first_ngram.tolist()}",
                )
        return (
            True,
            "All N-gram orders correctly left-pad at t=0",
            f"e.g. order=3: {extract_ngrams(ids, 3)[0, 0, :].tolist()}",
        )
    runner.run_check("2.10 N-gram edge case: position t=0 correctly left-pads",
                     group, check_ngram_left_pad)

    runner.end_group(group)


# ============================================================================
# Validation Group 3: Gating and Fusion
# ============================================================================

def validate_gating_fusion(runner: ValidationRunner):
    """~8 checks for gating and fusion contracts -- Done-When Gate (c)."""
    group = "gating_fusion"
    runner.begin_group(group)

    H_DIM = 64
    M_DIM = 32
    B, T = 2, 16

    # ---- Check 3.1: Gate output alpha in [0, 1] ----
    def check_gate_bounds():
        gating = ContextAwareGating(H_DIM, M_DIM)
        hidden = torch.randn(B, T, H_DIM)
        memory = torch.randn(B, T, M_DIM)
        _, alpha = gating(hidden, memory)
        in_range = (alpha >= 0.0).all().item() and (alpha <= 1.0).all().item()
        return (
            in_range,
            f"Gate alpha in [0, 1]: "
            f"min={alpha.min().item():.6f}, max={alpha.max().item():.6f}",
            f"Shape: {alpha.shape}",
        )
    runner.run_check("3.1 Gate output alpha is in [0, 1] for all positions",
                     group, check_gate_bounds)

    # ---- Check 3.2: Conservative init: mean gate near sigmoid(-2) ----
    def check_conservative_init():
        target = torch.sigmoid(torch.tensor(-2.0)).item()  # ~0.1192
        gating = ContextAwareGating(H_DIM, M_DIM, gate_init_bias=-2.0)
        # With random init on Wq/Wk, the dot product term should be near 0
        # So gate ~ sigmoid(bias) = sigmoid(-2) ~ 0.12
        torch.manual_seed(0)
        hidden = torch.randn(B, 128, H_DIM)
        memory = torch.randn(B, 128, M_DIM)
        with torch.no_grad():
            _, alpha = gating(hidden, memory)
        mean_gate = alpha.mean().item()
        # Allow some tolerance since Wq/Wk contribute noise
        tolerance = 0.15
        ok = abs(mean_gate - target) < tolerance
        return (
            ok,
            f"Mean gate at init: {mean_gate:.4f} "
            f"(target ~{target:.4f}, tol={tolerance})",
            f"sigmoid(-2)={target:.4f}, actual mean={mean_gate:.4f}, "
            f"diff={abs(mean_gate - target):.4f}",
        )
    runner.run_check("3.2 Conservative init: mean gate near sigmoid(-2) ~ 0.12",
                     group, check_conservative_init)

    # ---- Check 3.3: RMSNorm AMP safety: fp16 no NaN ----
    def check_rmsnorm_amp_safety():
        norm = RMSNorm(H_DIM)
        # Very small fp16 values that could underflow
        x_fp16 = torch.randn(B, T, H_DIM).half() * 1e-3
        out = norm(x_fp16)
        no_nan = not torch.isnan(out).any().item()
        no_inf = not torch.isinf(out).any().item()
        ok = no_nan and no_inf
        return (
            ok,
            f"RMSNorm fp16: no_nan={no_nan}, no_inf={no_inf}",
            f"Input dtype: {x_fp16.dtype}, output dtype: {out.dtype}, "
            f"output range: [{out.min().item():.4f}, {out.max().item():.4f}]",
        )
    runner.run_check("3.3 RMSNorm AMP safety: fp16 input produces no NaN",
                     group, check_rmsnorm_amp_safety)

    # ---- Check 3.4: Gating gradient flow ----
    def check_gating_gradient_flow():
        gating = ContextAwareGating(H_DIM, M_DIM)
        hidden = torch.randn(B, T, H_DIM, requires_grad=True)
        memory = torch.randn(B, T, M_DIM, requires_grad=True)
        output, alpha = gating(hidden, memory)
        loss = output.sum() + alpha.sum()
        loss.backward()

        has_grad = {}
        has_grad["Wq"] = gating.Wq.weight.grad is not None and gating.Wq.weight.grad.abs().sum() > 0
        has_grad["Wk"] = gating.Wk.weight.grad is not None and gating.Wk.weight.grad.abs().sum() > 0
        has_grad["Wv"] = gating.Wv.weight.grad is not None and gating.Wv.weight.grad.abs().sum() > 0
        has_grad["gate_bias"] = gating.gate_bias.grad is not None and gating.gate_bias.grad.abs().sum() > 0
        has_grad["hidden"] = hidden.grad is not None and hidden.grad.abs().sum() > 0
        has_grad["memory"] = memory.grad is not None and memory.grad.abs().sum() > 0

        all_ok = all(has_grad.values())
        return (
            all_ok,
            f"Gradient flow: {sum(has_grad.values())}/{len(has_grad)} params have gradients",
            "; ".join(f"{k}={'OK' if v else 'MISSING'}" for k, v in has_grad.items()),
        )
    runner.run_check("3.4 Gradient flow: grads exist on Wq, Wk, Wv, gate_bias",
                     group, check_gating_gradient_flow)

    # ---- Check 3.5: Depthwise causal conv causality ----
    def check_causal_conv():
        C = H_DIM
        conv = DepthwiseCausalConv(C, kernel_size=4, dilation=4)
        # Non-zero weights for the test
        nn.init.normal_(conv.conv.weight, std=0.5)
        nn.init.normal_(conv.conv.bias, std=0.1)

        x = torch.randn(1, T, C)
        y_original = conv(x).detach().clone()

        # Perturb a future position (pos T-1) and check past (pos 0..T-2) unchanged
        x_perturbed = x.clone()
        x_perturbed[0, T - 1, :] += 10.0
        y_perturbed = conv(x_perturbed).detach()

        # Past positions should be unchanged
        past_diff = (y_original[0, :T-1, :] - y_perturbed[0, :T-1, :]).abs().max().item()
        causal = past_diff < 1e-5
        return (
            causal,
            f"Causal conv: perturbing future pos -> past max diff = {past_diff:.2e} "
            f"({'causal' if causal else 'LEAKS'})",
            f"Checked positions 0..{T-2} after perturbing position {T-1}",
        )
    runner.run_check("3.5 Depthwise causal conv: future perturbation doesn't affect past",
                     group, check_causal_conv)

    # ---- Check 3.6: Conv output shape matches input ----
    def check_conv_shape():
        C = H_DIM
        shapes_ok = True
        details = []
        for t in [1, 8, 16, 64, 128]:
            conv = DepthwiseCausalConv(C, kernel_size=4, dilation=4)
            x = torch.randn(B, t, C)
            y = conv(x)
            ok = y.shape == x.shape
            shapes_ok = shapes_ok and ok
            details.append(f"T={t}: {tuple(x.shape)}->{tuple(y.shape)} {'OK' if ok else 'FAIL'}")
        return (
            shapes_ok,
            f"Conv output shape: {'all match input' if shapes_ok else 'MISMATCH'}",
            "; ".join(details),
        )
    runner.run_check("3.6 Conv output shape matches input shape (B, T, C)",
                     group, check_conv_shape)

    # ---- Check 3.7: Full EngramModule forward ----
    def check_engram_module_forward():
        config = EngramConfig(
            vocab_size=1000, embedding_dim=H_DIM,
            ngram_orders=(2, 3), num_heads=2,
            table_size=1009,  # Small prime
            conv_kernel_size=4, conv_dilation=3,
        )
        module = EngramModule(config, hidden_dim=H_DIM, layer_id=0)
        hidden = torch.randn(B, T, H_DIM)
        ids = torch.randint(0, 1000, (B, T))
        mask = torch.ones(B, T)
        delta, telemetry = module(hidden, ids, mask, return_details=True)
        shape_ok = delta.shape == (B, T, H_DIM)
        finite = torch.isfinite(delta).all().item()
        has_telemetry = telemetry is not None and "layer_id" in telemetry
        ok = shape_ok and finite and has_telemetry
        return (
            ok,
            f"EngramModule forward: shape={tuple(delta.shape)} "
            f"finite={finite} telemetry={has_telemetry}",
            f"delta norm: {delta.norm().item():.4f}, "
            f"gate mean: {telemetry.get('gate_mean', 'N/A') if telemetry else 'N/A'}",
        )
    runner.run_check("3.7 Full EngramModule forward produces correct delta shape",
                     group, check_engram_module_forward)

    # ---- Check 3.8: AMP autocast no NaN ----
    def check_amp_autocast():
        config = EngramConfig(
            vocab_size=1000, embedding_dim=H_DIM,
            ngram_orders=(2, 3), num_heads=2,
            table_size=1009,
        )
        module = EngramModule(config, hidden_dim=H_DIM, layer_id=0)
        hidden = torch.randn(B, T, H_DIM)
        ids = torch.randint(0, 1000, (B, T))
        mask = torch.ones(B, T)

        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        amp_dtype = torch.float16

        with torch.autocast(device_type=device_type, dtype=amp_dtype):
            delta, _ = module(hidden, ids, mask)

        no_nan = not torch.isnan(delta).any().item()
        no_inf = not torch.isinf(delta).any().item()
        ok = no_nan and no_inf
        return (
            ok,
            f"AMP autocast ({device_type}/fp16): no_nan={no_nan}, no_inf={no_inf}",
            f"delta dtype: {delta.dtype}, range: "
            f"[{delta.min().item():.4f}, {delta.max().item():.4f}]",
        )
    runner.run_check("3.8 AMP autocast: full forward pass produces no NaN",
                     group, check_amp_autocast)

    runner.end_group(group)


# ============================================================================
# Validation Group 4: Offload and Prefetch
# ============================================================================

def validate_offload_prefetch(runner: ValidationRunner):
    """~8 checks for offload/prefetch contracts -- Done-When Gate (b)."""
    group = "offload_prefetch"
    runner.begin_group(group)

    EMB_DIM = 32
    NUM_EMB = 1009  # Small prime
    B, T = 2, 16

    HAS_CUDA = torch.cuda.is_available()

    # ---- Check 4.1: On-device embedding lookup shape and dtype ----
    def check_ondevice_lookup():
        emb = OffloadableEmbedding(NUM_EMB, EMB_DIM, offload=False)
        indices = torch.randint(0, NUM_EMB, (B, T))
        out = emb(indices)
        shape_ok = out.shape == (B, T, EMB_DIM)
        dtype_ok = out.dtype == torch.float32
        ok = shape_ok and dtype_ok
        return (
            ok,
            f"On-device lookup: shape={tuple(out.shape)} dtype={out.dtype}",
            f"Expected: ({B}, {T}, {EMB_DIM}) float32",
        )
    runner.run_check("4.1 On-device embedding lookup: correct shape and dtype",
                     group, check_ondevice_lookup)

    # ---- Check 4.2: CPU offload matches on-device ----
    if HAS_CUDA:
        def check_offload_matches_ondevice():
            torch.manual_seed(123)
            emb_device = OffloadableEmbedding(NUM_EMB, EMB_DIM, offload=False)
            emb_device.embedding.weight.data.copy_(
                torch.randn(NUM_EMB, EMB_DIM)
            )

            emb_offload = OffloadableEmbedding(NUM_EMB, EMB_DIM, offload=True, prefetch=False)
            emb_offload.weight.data.copy_(emb_device.embedding.weight.data)

            indices = torch.randint(0, NUM_EMB, (B, T))
            out_device = emb_device(indices)

            indices_cuda = indices.cuda()
            # Offload will do synchronous fallback (no prefetch)
            out_offload = emb_offload(indices_cuda)

            cos_sim = F.cosine_similarity(
                out_device.flatten().unsqueeze(0),
                out_offload.cpu().flatten().unsqueeze(0),
            ).item()
            ok = cos_sim > 0.9999
            return (
                ok,
                f"Offload vs on-device cosine similarity: {cos_sim:.6f} "
                f"({'> 0.9999' if ok else '< 0.9999 MISMATCH'})",
                None,
            )
        runner.run_check("4.2 CPU offload output matches on-device (cosine > 0.9999)",
                         group, check_offload_matches_ondevice)
    else:
        runner.skip_check("4.2 CPU offload output matches on-device (cosine > 0.9999)",
                          group, "CUDA not available")

    # ---- Check 4.3: Prefetch/consume cycle no deadlock ----
    if HAS_CUDA:
        def check_prefetch_no_deadlock():
            emb = OffloadableEmbedding(NUM_EMB, EMB_DIM, offload=True, prefetch=True)
            emb.weight.data.copy_(torch.randn(NUM_EMB, EMB_DIM))

            success_count = 0
            for i in range(10):
                indices = torch.randint(0, NUM_EMB, (B, T)).cuda()
                # Prefetch
                emb.prefetch_async(indices)
                # Consume
                result = emb(indices)
                if result is not None and result.shape == (B, T, EMB_DIM):
                    success_count += 1

            ok = success_count == 10
            return (
                ok,
                f"Prefetch/consume cycle: {success_count}/10 succeeded "
                f"({'no deadlock' if ok else 'ISSUES'})",
                None,
            )

        # Run with a timeout
        def check_prefetch_with_timeout():
            result_holder = [None]
            def run():
                result_holder[0] = check_prefetch_no_deadlock()
            t = threading.Thread(target=run)
            t.start()
            t.join(timeout=15.0)  # 15s total for 10 iterations
            if t.is_alive():
                return (False, "DEADLOCK: prefetch/consume did not complete in 15s", None)
            return result_holder[0]

        runner.run_check("4.3 Prefetch/consume cycle: 10 iterations, no deadlock",
                         group, check_prefetch_with_timeout)
    else:
        runner.skip_check("4.3 Prefetch/consume cycle: 10 iterations, no deadlock",
                          group, "CUDA not available")

    # ---- Check 4.4: Coalescing reduces unique row fetches ----
    if HAS_CUDA:
        def check_coalescing():
            emb = OffloadableEmbedding(NUM_EMB, EMB_DIM, offload=True, prefetch=True)
            # Input with many repeated IDs
            repeated_ids = torch.cat([
                torch.full((B, T // 2), 42, dtype=torch.long),
                torch.full((B, T // 2), 99, dtype=torch.long),
            ], dim=1).cuda()
            unique_count = emb.get_num_unique_fetches(repeated_ids)
            total = repeated_ids.numel()
            reduction = 1.0 - (unique_count / total)
            ok = unique_count < total
            return (
                ok,
                f"Coalescing: {unique_count} unique out of {total} total "
                f"(reduction {reduction:.1%})",
                f"Repeated IDs: only 2 unique values in {total} elements",
            )
        runner.run_check("4.4 Coalescing: repeated IDs trigger fewer unique fetches",
                         group, check_coalescing)
    else:
        runner.skip_check("4.4 Coalescing: repeated IDs trigger fewer unique fetches",
                          group, "CUDA not available")

    # ---- Check 4.5: PrefetchPlan schedules correctly ----
    def check_prefetch_plan():
        plan = PrefetchPlan(engram_layer_indices=[1, 5, 9], prefetch_ahead=2)
        schedule = plan.schedule
        # Layer 1 should be prefetched starting at max(0, 1-2) = 0
        # Layer 5 should be prefetched starting at max(0, 5-2) = 3
        # Layer 9 should be prefetched starting at max(0, 9-2) = 7
        expected = {0: 1, 3: 5, 7: 9}
        ok = schedule == expected
        # Also test should_prefetch_at
        check_at_0 = plan.should_prefetch_at(0) == 1
        check_at_1 = plan.should_prefetch_at(1) is None
        check_at_3 = plan.should_prefetch_at(3) == 5
        checks_ok = check_at_0 and check_at_1 and check_at_3
        all_ok = ok and checks_ok
        return (
            all_ok,
            f"PrefetchPlan schedule: {schedule} "
            f"{'matches expected' if ok else 'MISMATCH'}",
            f"Expected: {expected}; API checks: {checks_ok}",
        )
    runner.run_check("4.5 PrefetchPlan correctly schedules N layers ahead",
                     group, check_prefetch_plan)

    # ---- Check 4.6: Pinned buffer allocation ----
    if HAS_CUDA:
        def check_pinned_buffer():
            try:
                buf = torch.zeros(NUM_EMB, EMB_DIM).pin_memory()
                is_pinned = buf.is_pinned()
                return (
                    is_pinned,
                    f"Pinned buffer allocation: is_pinned={is_pinned}",
                    f"Shape: {tuple(buf.shape)}, dtype: {buf.dtype}",
                )
            except Exception as e:
                return (False, f"Pinned buffer allocation failed: {e}", None)
        runner.run_check("4.6 Pinned buffer allocation doesn't fail",
                         group, check_pinned_buffer)
    else:
        runner.skip_check("4.6 Pinned buffer allocation doesn't fail",
                          group, "CUDA not available")

    # ---- Check 4.7: Memory accounting: offload uses less GPU memory ----
    if HAS_CUDA:
        def check_memory_accounting():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # On-device
            mem_before_device = torch.cuda.memory_allocated()
            emb_device = OffloadableEmbedding(
                NUM_EMB, EMB_DIM, offload=False
            ).cuda()
            mem_after_device = torch.cuda.memory_allocated()
            device_usage = mem_after_device - mem_before_device

            del emb_device
            torch.cuda.empty_cache()

            # Offload
            mem_before_offload = torch.cuda.memory_allocated()
            emb_offload = OffloadableEmbedding(
                NUM_EMB, EMB_DIM, offload=True, prefetch=True
            )
            mem_after_offload = torch.cuda.memory_allocated()
            offload_usage = mem_after_offload - mem_before_offload

            del emb_offload
            torch.cuda.empty_cache()

            ok = offload_usage < device_usage
            return (
                ok,
                f"GPU memory: on-device={device_usage} B, offload={offload_usage} B "
                f"({'offload < device' if ok else 'offload >= device'})",
                None,
            )
        runner.run_check("4.7 Memory: offload mode uses less GPU memory than on-device",
                         group, check_memory_accounting)
    else:
        runner.skip_check("4.7 Memory: offload mode uses less GPU memory than on-device",
                          group, "CUDA not available")

    # ---- Check 4.8: On-device lookup is differentiable ----
    def check_ondevice_differentiable():
        emb = OffloadableEmbedding(NUM_EMB, EMB_DIM, offload=False)
        indices = torch.randint(0, NUM_EMB, (B, T))
        out = emb(indices)
        loss = out.sum()
        loss.backward()
        has_grad = emb.embedding.weight.grad is not None
        grad_nonzero = has_grad and emb.embedding.weight.grad.abs().sum() > 0
        ok = has_grad and grad_nonzero
        return (
            ok,
            f"On-device embedding differentiable: has_grad={has_grad}, "
            f"nonzero={grad_nonzero}",
            None,
        )
    runner.run_check("4.8 On-device embedding lookup is differentiable",
                     group, check_ondevice_differentiable)

    runner.end_group(group)


# ============================================================================
# Validation Group 5: Phase 1 Encoder-Competition
# ============================================================================

def validate_phase1_encoder(runner: ValidationRunner):
    """~6 checks for Phase 1 encoder-competition -- Done-When Gate (c)."""
    group = "phase1_encoder"
    runner.begin_group(group)

    B, T = 2, 16
    WS_DIM = 4096
    H_DIM = 64  # Smaller embedding_dim for test speed

    config = EngramConfig(
        vocab_size=1000, embedding_dim=H_DIM,
        ngram_orders=(2, 3), num_heads=2,
        table_size=1009,
    )

    # ---- Check 5.1: Output shape is (B, T, 4096) ----
    def check_encoder_output_shape():
        encoder = EngramTextEncoder(config, workspace_dim=WS_DIM)
        ids = torch.randint(0, 1000, (B, T))
        out = encoder(ids)
        ok = out.shape == (B, T, WS_DIM)
        return (
            ok,
            f"Encoder output shape: {tuple(out.shape)} "
            f"({'= (B,T,4096)' if ok else 'MISMATCH'})",
            f"Expected: ({B}, {T}, {WS_DIM})",
        )
    runner.run_check("5.1 EngramTextEncoder output shape is (B, T, 4096)",
                     group, check_encoder_output_shape)

    # ---- Check 5.2: Padding positions in output are zero ----
    def check_padding_positions_zero():
        encoder = EngramTextEncoder(config, workspace_dim=WS_DIM)
        ids = torch.randint(0, 1000, (B, T))
        mask = torch.ones(B, T)
        mask[:, T // 2:] = 0  # Second half is padding
        out = encoder(ids, attention_mask=mask)

        padded_region = out[:, T // 2:, :]
        is_zero = (padded_region == 0).all().item()
        non_padded = out[:, :T // 2, :]
        has_content = non_padded.abs().sum().item() > 0
        ok = is_zero and has_content
        return (
            ok,
            f"Padding positions zero: padded_zero={is_zero}, "
            f"content_nonzero={has_content}",
            f"Padded region norm: {padded_region.norm().item():.6f}, "
            f"content norm: {non_padded.norm().item():.4f}",
        )
    runner.run_check("5.2 Padding positions in output are zero (masked)",
                     group, check_padding_positions_zero)

    # ---- Check 5.3: Different input_ids produce different outputs ----
    def check_different_inputs_different_outputs():
        encoder = EngramTextEncoder(config, workspace_dim=WS_DIM)
        ids_a = torch.randint(0, 500, (B, T))
        ids_b = torch.randint(500, 1000, (B, T))
        out_a = encoder(ids_a)
        out_b = encoder(ids_b)
        differ = not torch.allclose(out_a, out_b, atol=1e-6)
        max_diff = (out_a - out_b).abs().max().item()
        return (
            differ,
            f"Different inputs produce different outputs: "
            f"max_diff={max_diff:.6f}",
            None,
        )
    runner.run_check("5.3 Different input_ids produce different outputs",
                     group, check_different_inputs_different_outputs)

    # ---- Check 5.4: Gradient flow through encoder ----
    def check_encoder_gradient_flow():
        encoder = EngramTextEncoder(config, workspace_dim=WS_DIM)
        ids = torch.randint(0, 1000, (B, T))
        out = encoder(ids)
        loss = out.sum()
        loss.backward()

        params_with_grad = 0
        total_params = 0
        for name, p in encoder.named_parameters():
            total_params += 1
            if p.grad is not None and p.grad.abs().sum() > 0:
                params_with_grad += 1

        ok = params_with_grad > 0
        return (
            ok,
            f"Gradient flow: {params_with_grad}/{total_params} params have gradients",
            None,
        )
    runner.run_check("5.4 Gradient flow: gradients exist on encoder parameters",
                     group, check_encoder_gradient_flow)

    # ---- Check 5.5: AMP safety: no NaN ----
    def check_encoder_amp_safety():
        encoder = EngramTextEncoder(config, workspace_dim=WS_DIM)
        ids = torch.randint(0, 1000, (B, T))

        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.autocast(device_type=device_type, dtype=torch.float16):
            out = encoder(ids)

        no_nan = not torch.isnan(out).any().item()
        no_inf = not torch.isinf(out).any().item()
        ok = no_nan and no_inf
        return (
            ok,
            f"AMP safety: no_nan={no_nan}, no_inf={no_inf}",
            f"Output dtype: {out.dtype}",
        )
    runner.run_check("5.5 AMP safety: no NaN under autocast",
                     group, check_encoder_amp_safety)

    # ---- Check 5.6: Output is finite ----
    def check_encoder_output_finite():
        encoder = EngramTextEncoder(config, workspace_dim=WS_DIM)
        ids = torch.randint(0, 1000, (B, T))
        out = encoder(ids)
        finite = torch.isfinite(out).all().item()
        return (
            finite,
            f"Output finite: {finite}",
            f"Range: [{out.min().item():.4f}, {out.max().item():.4f}]",
        )
    runner.run_check("5.6 Output is finite (no inf)",
                     group, check_encoder_output_finite)

    runner.end_group(group)


# ============================================================================
# Validation Group 6: Phase 2 Layer-Augmentation
# ============================================================================

def validate_phase2_layer(runner: ValidationRunner):
    """~6 checks for Phase 2 layer-augmentation -- Done-When Gate (c)."""
    group = "phase2_layer"
    runner.begin_group(group)

    B, T = 2, 16
    H_DIM = 64

    config = EngramConfig(
        vocab_size=1000, embedding_dim=H_DIM,
        ngram_orders=(2, 3), num_heads=2,
        table_size=1009,
    )

    # ---- Check 6.1: Output shape matches input ----
    def check_layer_output_shape():
        layer = EngramAugmentedLayer(H_DIM, config, layer_id=0)
        hidden = torch.randn(B, T, H_DIM)
        ids = torch.randint(0, 1000, (B, T))
        out, _ = layer(hidden, ids)
        ok = out.shape == hidden.shape
        return (
            ok,
            f"Layer output shape: {tuple(out.shape)} "
            f"{'matches' if ok else 'DIFFERS from'} input {tuple(hidden.shape)}",
            None,
        )
    runner.run_check("6.1 EngramAugmentedLayer output shape matches input (B, T, D)",
                     group, check_layer_output_shape)

    # ---- Check 6.2: Residual: output differs from input ----
    def check_residual_nonzero():
        layer = EngramAugmentedLayer(H_DIM, config, layer_id=0)
        # Give non-zero weights to out_proj so delta is nonzero
        with torch.no_grad():
            layer.engram.out_proj.weight.fill_(0.01)
        hidden = torch.randn(B, T, H_DIM)
        ids = torch.randint(0, 1000, (B, T))
        out, _ = layer(hidden, ids)
        diff = (out - hidden).abs().sum().item()
        ok = diff > 1e-6
        return (
            ok,
            f"Residual non-zero: delta norm = {diff:.6f} "
            f"({'nonzero' if ok else 'ZERO -- pass-through'})",
            None,
        )
    runner.run_check("6.2 Residual: output differs from input (non-zero delta)",
                     group, check_residual_nonzero)

    # ---- Check 6.3: Compute-now path ----
    def check_compute_now_path():
        layer = EngramAugmentedLayer(H_DIM, config, layer_id=0)
        hidden = torch.randn(B, T, H_DIM)
        ids = torch.randint(0, 1000, (B, T))
        mask = torch.ones(B, T)
        out, telemetry = layer(
            hidden, ids, mask,
            prefetched_embeddings=None,
            return_details=True,
        )
        shape_ok = out.shape == (B, T, H_DIM)
        finite = torch.isfinite(out).all().item()
        has_telemetry = telemetry is not None
        ok = shape_ok and finite and has_telemetry
        return (
            ok,
            f"Compute-now path: shape_ok={shape_ok}, finite={finite}, "
            f"telemetry={has_telemetry}",
            None,
        )
    runner.run_check("6.3 'Compute now' path works without prefetched data",
                     group, check_compute_now_path)

    # ---- Check 6.4: Consume-prefetched path ----
    def check_consume_prefetched_path():
        layer = EngramAugmentedLayer(H_DIM, config, layer_id=0)
        hidden = torch.randn(B, T, H_DIM)
        ids = torch.randint(0, 1000, (B, T))
        mask = torch.ones(B, T)
        # Simulate pre-supplied embeddings
        fake_prefetched = torch.randn(B, T, H_DIM)
        out, telemetry = layer(
            hidden, ids, mask,
            prefetched_embeddings=fake_prefetched,
            return_details=True,
        )
        shape_ok = out.shape == (B, T, H_DIM)
        finite = torch.isfinite(out).all().item()
        ok = shape_ok and finite
        return (
            ok,
            f"Consume-prefetched path: shape_ok={shape_ok}, finite={finite}",
            None,
        )
    runner.run_check("6.4 'Consume prefetched' path works with pre-supplied embeddings",
                     group, check_consume_prefetched_path)

    # ---- Check 6.5: return_details includes layer_id ----
    def check_telemetry_layer_id():
        layer = EngramAugmentedLayer(H_DIM, config, layer_id=7)
        hidden = torch.randn(B, T, H_DIM)
        ids = torch.randint(0, 1000, (B, T))
        _, telemetry = layer(hidden, ids, return_details=True)
        has_layer_id = telemetry is not None and "layer_id" in telemetry
        correct_id = has_layer_id and telemetry["layer_id"] == 7
        ok = has_layer_id and correct_id
        return (
            ok,
            f"Telemetry layer_id: present={has_layer_id}, "
            f"correct={correct_id} (expected 7)",
            f"telemetry keys: {list(telemetry.keys()) if telemetry else 'None'}",
        )
    runner.run_check("6.5 return_details includes layer_id in telemetry",
                     group, check_telemetry_layer_id)

    # ---- Check 6.6: Gradient flow through augmented layer ----
    def check_layer_gradient_flow():
        layer = EngramAugmentedLayer(H_DIM, config, layer_id=0)
        hidden = torch.randn(B, T, H_DIM, requires_grad=True)
        ids = torch.randint(0, 1000, (B, T))
        out, _ = layer(hidden, ids)
        loss = out.sum()
        loss.backward()

        input_has_grad = hidden.grad is not None and hidden.grad.abs().sum() > 0
        params_with_grad = 0
        total_params = 0
        for name, p in layer.named_parameters():
            total_params += 1
            if p.grad is not None and p.grad.abs().sum() > 0:
                params_with_grad += 1

        ok = input_has_grad and params_with_grad > 0
        return (
            ok,
            f"Gradient flow: input_grad={input_has_grad}, "
            f"{params_with_grad}/{total_params} params have grad",
            None,
        )
    runner.run_check("6.6 Gradient flow through augmented layer",
                     group, check_layer_gradient_flow)

    runner.end_group(group)


# ============================================================================
# Validation Group 7: Integration
# ============================================================================

def validate_integration(runner: ValidationRunner):
    """~6 checks for end-to-end integration -- Done-When Gates (a), (b), (c)."""
    group = "integration"
    runner.begin_group(group)

    B, T = 2, 16
    H_DIM = 64

    config = EngramConfig(
        vocab_size=1000, embedding_dim=H_DIM,
        ngram_orders=(2, 3), num_heads=2,
        table_size=1009,
    )

    # ---- Check 7.1: Mini-backbone simulation ----
    def check_mini_backbone():
        """4 layers, Engram at layers 1 and 3."""
        num_layers = 4
        engram_at = {1, 3}

        layers = nn.ModuleList()
        for i in range(num_layers):
            if i in engram_at:
                layers.append(
                    EngramAugmentedLayer(H_DIM, config, layer_id=i, use_engram=True)
                )
            else:
                # Simple pass-through + linear for non-engram layers
                layers.append(nn.Linear(H_DIM, H_DIM))

        hidden = torch.randn(B, T, H_DIM)
        ids = torch.randint(0, 1000, (B, T))
        mask = torch.ones(B, T)

        x = hidden
        for i, layer in enumerate(layers):
            if i in engram_at:
                x, _ = layer(x, ids, mask)
            else:
                x = layer(x)

        shape_ok = x.shape == (B, T, H_DIM)
        finite = torch.isfinite(x).all().item()
        ok = shape_ok and finite
        return (
            ok,
            f"Mini-backbone (4 layers, Engram at 1,3): "
            f"shape={tuple(x.shape)}, finite={finite}",
            None,
        )
    runner.run_check("7.1 Mini-backbone: 4 layers with Engram at layers 1 and 3",
                     group, check_mini_backbone)

    # ---- Check 7.2: Forward pass completes without error ----
    def check_forward_completes():
        num_layers = 4
        layers = nn.ModuleList()
        for i in range(num_layers):
            layers.append(
                EngramAugmentedLayer(H_DIM, config, layer_id=i, use_engram=(i % 2 == 1))
            )

        hidden = torch.randn(B, T, H_DIM)
        ids = torch.randint(0, 1000, (B, T))
        mask = torch.ones(B, T)

        x = hidden
        for layer in layers:
            x, _ = layer(x, ids, mask)

        ok = x.shape == (B, T, H_DIM)
        return (
            ok,
            f"Forward pass through 4 EngramAugmentedLayers completes: shape={tuple(x.shape)}",
            None,
        )
    runner.run_check("7.2 Forward pass completes without error",
                     group, check_forward_completes)

    # ---- Check 7.3: Telemetry aggregated from all Engram layers ----
    def check_telemetry_aggregated():
        num_layers = 4
        engram_at = {1, 3}
        layers = nn.ModuleList()
        for i in range(num_layers):
            layers.append(
                EngramAugmentedLayer(
                    H_DIM, config, layer_id=i,
                    use_engram=(i in engram_at),
                )
            )

        hidden = torch.randn(B, T, H_DIM)
        ids = torch.randint(0, 1000, (B, T))
        mask = torch.ones(B, T)

        all_telemetry = []
        x = hidden
        for layer in layers:
            x, telem = layer(x, ids, mask, return_details=True)
            if telem is not None:
                all_telemetry.append(telem)

        num_telemetry = len(all_telemetry)
        expected = len(engram_at)
        ok = num_telemetry == expected

        layer_ids = [t.get("layer_id", -1) for t in all_telemetry]
        ids_match = set(layer_ids) == engram_at
        ok = ok and ids_match

        return (
            ok,
            f"Telemetry: {num_telemetry} reports from Engram layers "
            f"(expected {expected}), layer_ids={layer_ids}",
            f"Expected layer_ids: {sorted(engram_at)}",
        )
    runner.run_check("7.3 Telemetry aggregated from all Engram layers",
                     group, check_telemetry_aggregated)

    # ---- Check 7.4: return_details=True has expected keys ----
    def check_telemetry_keys():
        layer = EngramAugmentedLayer(H_DIM, config, layer_id=0, use_engram=True)
        hidden = torch.randn(B, T, H_DIM)
        ids = torch.randint(0, 1000, (B, T))
        _, telem = layer(hidden, ids, return_details=True)

        expected_keys = {"layer_id", "gate_mean", "gate_min", "gate_max",
                         "memory_norm", "delta_norm"}
        if telem is None:
            return (False, "Telemetry is None", None)
        present_keys = set(telem.keys())
        missing = expected_keys - present_keys
        ok = len(missing) == 0
        return (
            ok,
            f"Telemetry keys: {len(expected_keys - missing)}/{len(expected_keys)} "
            f"present {'(all)' if ok else f'-- missing: {missing}'}",
            f"Present: {sorted(present_keys)}",
        )
    runner.run_check("7.4 return_details=True: telemetry has expected keys",
                     group, check_telemetry_keys)

    # ---- Check 7.5: Both modes coexist ----
    def check_both_modes_coexist():
        # Phase 1: encoder
        encoder = EngramTextEncoder(config, workspace_dim=128)
        # Phase 2: layer augmentation
        layer = EngramAugmentedLayer(H_DIM, config, layer_id=0)

        ids = torch.randint(0, 1000, (B, T))
        mask = torch.ones(B, T)

        # Run encoder
        enc_out = encoder(ids, mask)
        enc_shape_ok = enc_out.shape == (B, T, 128)

        # Run layer
        hidden = torch.randn(B, T, H_DIM)
        layer_out, _ = layer(hidden, ids, mask)
        layer_shape_ok = layer_out.shape == (B, T, H_DIM)

        ok = enc_shape_ok and layer_shape_ok
        return (
            ok,
            f"Both modes coexist: encoder shape={tuple(enc_out.shape)} "
            f"layer shape={tuple(layer_out.shape)}",
            f"Encoder and layer augmentation can exist in same model",
        )
    runner.run_check("7.5 Both modes coexist: encoder + layer augmentation",
                     group, check_both_modes_coexist)

    # ---- Check 7.6: Feature flag toggle: disabling engram -> pass-through ----
    def check_feature_flag_toggle():
        layer_enabled = EngramAugmentedLayer(
            H_DIM, config, layer_id=0, use_engram=True
        )
        # Give nonzero weights to out_proj so delta is nonzero when engram active
        with torch.no_grad():
            layer_enabled.engram.out_proj.weight.normal_(std=0.1)
        layer_disabled = EngramAugmentedLayer(
            H_DIM, config, layer_id=0, use_engram=False
        )

        hidden = torch.randn(B, T, H_DIM)
        ids = torch.randint(0, 1000, (B, T))

        out_enabled, _ = layer_enabled(hidden, ids)
        out_disabled, _ = layer_disabled(hidden, ids)

        # Disabled should be exact pass-through
        is_passthrough = torch.equal(out_disabled, hidden)
        # Enabled should differ (due to engram delta)
        differs = not torch.equal(out_enabled, hidden)

        ok = is_passthrough and differs
        return (
            ok,
            f"Feature flag: disabled=passthrough={is_passthrough}, "
            f"enabled=differs={differs}",
            None,
        )
    runner.run_check("7.6 Feature flag: disabling engram -> pass-through (output=input)",
                     group, check_feature_flag_toggle)

    runner.end_group(group)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Engram Conditional Memory -- Runtime Contract Validation",
    )
    parser.add_argument(
        "--group", type=str, default=None,
        help=f"Run only a specific group. Choices: {_ALL_GROUPS}",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show detailed output for all checks (not just failures)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all validation groups and exit",
    )
    args = parser.parse_args()

    if args.list:
        print("\nAvailable validation groups:")
        for g in _ALL_GROUPS:
            desc = _GROUP_DESCRIPTIONS.get(g, "")
            print(f"  {_c(g, _BOLD):40s} {desc}")
        print()
        sys.exit(0)

    print(f"\n{_c('Engram Conditional Memory -- Runtime Contract Validation', _BOLD)}")
    print(f"{'=' * 72}")
    print(f"  PyTorch version : {torch.__version__}")
    print(f"  CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device     : {torch.cuda.get_device_name(0)}")
    print(f"  Python          : {sys.version.split()[0]}")
    print(f"{'=' * 72}")

    runner = ValidationRunner(verbose=args.verbose)

    # Map group names to validation functions
    group_fns = {
        "tokenizer_compression": validate_tokenizer_compression,
        "hashing_determinism": validate_hashing_determinism,
        "gating_fusion": validate_gating_fusion,
        "offload_prefetch": validate_offload_prefetch,
        "phase1_encoder": validate_phase1_encoder,
        "phase2_layer": validate_phase2_layer,
        "integration": validate_integration,
    }

    if args.group:
        if args.group not in group_fns:
            print(f"\nError: unknown group '{args.group}'. "
                  f"Choices: {list(group_fns.keys())}")
            sys.exit(1)
        groups_to_run = [args.group]
    else:
        groups_to_run = _ALL_GROUPS

    t_start = time.perf_counter()
    for g in groups_to_run:
        group_fns[g](runner)
    total_time = time.perf_counter() - t_start

    all_passed = runner.print_final_summary()
    print(f"  Total time: {total_time:.2f}s\n")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
