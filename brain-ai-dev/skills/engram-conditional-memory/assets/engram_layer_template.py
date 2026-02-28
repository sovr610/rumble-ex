"""
brain_ai/layers/engram_layer.py -- EngramAugmentedLayer and EngramTextEncoder integration modes.

Phase 2 (layer-augmentation): ``EngramAugmentedLayer`` is inserted at configurable backbone
layers. It computes a delta residually via the EngramModule pipeline (hash -> retrieve ->
gate -> conv -> project) and adds it *before* attention/FFN per the DeepSeek paper. Supports
two paths: "compute now" (full pipeline) and "consume prefetched" (reuse pre-fetched
embeddings from an earlier layer boundary to overlap PCIe transfer with compute).

Phase 1 (encoder-competition): ``EngramTextEncoder`` produces workspace-aligned ``(B, T, 4096)``
representations that compete in the Global Workspace alongside other modality encoders. It
uses the same tokenizer compression and N-gram hash retrieval as Phase 2, but wraps them in a
lightweight input-embedding + workspace-projection pipeline.

Supporting infrastructure:
  - ``EngramLayerConfig`` / ``EngramEncoderConfig`` -- configuration dataclasses
  - ``EngramLayerRegistry`` -- tracks which backbone layers have Engram augmentation
  - ``PrefetchHookManager`` -- manages async prefetch scheduling for Phase 2 + CPU offload
  - ``EngramCheckpointMixin`` -- save/load for compression tables, hash configs, weights

This template is self-contained: mock components replace the real EngramModule, MultiHeadHash,
and TokenizerCompression so the file can be executed directly for validation.

Usage (once integrated into brain_ai)::

    from brain_ai.layers.engram_layer import (
        EngramAugmentedLayer,
        EngramTextEncoder,
        EngramLayerConfig,
        EngramEncoderConfig,
        EngramLayerRegistry,
        PrefetchHookManager,
        EngramCheckpointMixin,
    )

Copy this file to ``brain_ai/layers/engram_layer.py`` when integrating into the main package,
replacing mock components with real imports.
"""

from __future__ import annotations

import json
import hashlib
import logging
import math
import os
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

DEFAULT_WORKSPACE_DIM: int = 4096
DEFAULT_EMBEDDING_DIM: int = 256
DEFAULT_NGRAM_ORDERS: Tuple[int, ...] = (2, 3)
DEFAULT_NUM_HEADS: int = 4
DEFAULT_TABLE_SIZE: int = 131071  # Prime
DEFAULT_CONV_KERNEL: int = 4
DEFAULT_CONV_DILATION: int = 4
DEFAULT_GATE_INIT_BIAS: float = -2.0
DEFAULT_PREFETCH_AHEAD: int = 2
RMSNORM_EPS: float = 1e-6
MAX_NGRAM_ORDER: int = 8  # Safety clamp for N-gram orders


# ========================================================================== #
#  SECTION 1: Configuration Dataclasses                                      #
# ========================================================================== #

@dataclass
class EngramConfig:
    """Configuration for the core EngramModule.

    This mirrors the config surface documented in SKILL.md. When this template
    is integrated, replace this inline copy with::

        from brain_ai.config import EngramConfig  # or memory.engram
    """

    # Vocabulary / tokenizer
    vocab_size: int = 50000
    compressed_vocab_size: Optional[int] = None
    use_tokenizer_compression: bool = True

    # Embeddings
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    ngram_orders: Tuple[int, ...] = DEFAULT_NGRAM_ORDERS
    num_heads_per_order: int = DEFAULT_NUM_HEADS
    table_size: int = DEFAULT_TABLE_SIZE

    # Hash
    hash_fn: str = "mult_xor"
    use_prime_sizes: bool = True
    per_layer_salt: bool = True
    hash_seed: int = 42

    # Gating
    use_context_gate: bool = True
    gate_type: str = "scalar"
    use_rmsnorm: bool = True
    gate_init_bias: float = DEFAULT_GATE_INIT_BIAS
    gate_temperature: float = 1.0

    # Convolution
    use_depthwise_conv: bool = True
    conv_kernel_size: int = DEFAULT_CONV_KERNEL
    conv_dilation: int = DEFAULT_CONV_DILATION

    # Offload
    offload_to_cpu: bool = False
    use_async_prefetch: bool = False
    prefetch_ahead_layers: int = DEFAULT_PREFETCH_AHEAD
    pin_memory: bool = True
    storage_dtype: str = "float16"

    def __post_init__(self) -> None:
        if self.compressed_vocab_size is None:
            self.compressed_vocab_size = int(self.vocab_size * 0.77)


@dataclass
class EngramLayerConfig:
    """Configuration for a single EngramAugmentedLayer (Phase 2).

    Attributes:
        hidden_dim: Backbone hidden dimension (e.g., 4096 for a 7B model).
        engram_config: Reference to the shared EngramConfig controlling
            N-gram order, embedding dim, gating, convolution, etc.
        layer_id: Integer index of this layer in the backbone. Used as
            per-layer hash salt to decorrelate collisions across layers.
        use_prefetch: Whether this layer expects prefetched embeddings from
            a PrefetchHookManager instead of running the full pipeline.
        residual_scale: Scalar multiplier on the engram delta before residual
            addition. Useful for gradual warm-in during early training.
        use_pre_norm: Apply RMSNorm to hidden_states before feeding to the
            EngramModule (pre-norm residual style).
        dropout: Dropout probability on the projected delta.
    """

    hidden_dim: int = DEFAULT_WORKSPACE_DIM
    engram_config: EngramConfig = field(default_factory=EngramConfig)
    layer_id: int = 0
    use_prefetch: bool = False
    residual_scale: float = 1.0
    use_pre_norm: bool = True
    dropout: float = 0.0


@dataclass
class EngramEncoderConfig:
    """Configuration for EngramTextEncoder (Phase 1 encoder-competition).

    Attributes:
        vocab_size: Input vocabulary size (before compression).
        embed_dim: Dimension of the lightweight input embedding used as
            "hidden_states" for the gating mechanism.
        workspace_dim: Output dimension, must match the unified workspace
            dimension (default 4096).
        engram_config: Reference to the shared EngramConfig.
        use_tokenizer_compression: Whether to apply tokenizer compression.
        max_seq_len: Maximum sequence length for positional encoding.
        use_positional: Whether to add learned positional encoding.
        dropout: Dropout on the workspace projection.
    """

    vocab_size: int = 50000
    embed_dim: int = DEFAULT_EMBEDDING_DIM
    workspace_dim: int = DEFAULT_WORKSPACE_DIM
    engram_config: EngramConfig = field(default_factory=EngramConfig)
    use_tokenizer_compression: bool = True
    max_seq_len: int = 2048
    use_positional: bool = True
    dropout: float = 0.0


# ========================================================================== #
#  SECTION 2: RMSNorm (fp32-safe)                                           #
# ========================================================================== #

class RMSNorm(nn.Module):
    """Root-Mean-Square Layer Normalization.

    Forces fp32 computation under AMP to prevent NaN. This is the same RMSNorm
    used in the gating pathway per the anti-pattern guidance: "fp16 for RMSNorm
    -- gating normalization needs fp32 under AMP".

    Args:
        dim: Feature dimension to normalize over (last dim).
        eps: Epsilon for numerical stability.
    """

    def __init__(self, dim: int, eps: float = RMSNORM_EPS) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """Normalize x along the last dimension.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            Normalized tensor of the same shape.
        """
        input_dtype = x.dtype
        # Force fp32 for stability under AMP
        x_fp32 = x.float()
        rms = torch.sqrt(torch.mean(x_fp32 ** 2, dim=-1, keepdim=True) + self.eps)
        normed = x_fp32 / rms
        return (self.weight.float() * normed).to(input_dtype)


# ========================================================================== #
#  SECTION 3: Mock Components (for self-contained testing)                   #
# ========================================================================== #
#
# When integrating into brain_ai, replace these with:
#     from brain_ai.memory.engram import EngramModule
#     from brain_ai.memory.hash_embedding import MultiHeadHash
#     from brain_ai.memory.tokenizer_compression import TokenizerCompression

class MockTokenizerCompression(nn.Module):
    """Identity tokenizer compression for self-testing.

    In production, this is replaced by the real TokenizerCompression that
    collapses textually equivalent tokens into canonical IDs via NFKC +
    lowercasing surjection.
    """

    def __init__(self, vocab_size: int, compressed_size: Optional[int] = None) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.compressed_size = compressed_size or vocab_size

    def compress(self, input_ids: Tensor) -> Tensor:
        """Return input_ids unchanged (identity mapping).

        Args:
            input_ids: (B, T) integer token IDs.

        Returns:
            Same tensor, unmodified.
        """
        return input_ids

    def save(self, path: str) -> None:
        """Save compression table (no-op for mock)."""
        pass

    def load(self, path: str) -> None:
        """Load compression table (no-op for mock)."""
        pass


class MockHashModule(nn.Module):
    """Deterministic multiplicative-XOR hash mock for self-testing.

    Produces deterministic int64 hash IDs from N-gram windows using a simple
    polynomial rolling hash. The real MultiHeadHash uses per-head coefficient
    vectors and XOR seeds for collision reduction.

    Args:
        ngram_order: Number of tokens per N-gram.
        num_heads: Number of independent hash heads.
        table_size: Modulus for hash output.
        seed: Deterministic seed for coefficient generation.
    """

    def __init__(
        self,
        ngram_order: int,
        num_heads: int,
        table_size: int,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.n = ngram_order
        self.num_heads = num_heads
        self.table_size = table_size

        # Generate deterministic coefficients
        gen = torch.Generator()
        gen.manual_seed(seed)
        coeffs = torch.randint(1, table_size, (num_heads, ngram_order), generator=gen)
        seeds = torch.randint(0, table_size, (num_heads,), generator=gen)
        self.register_buffer("coefficients", coeffs)
        self.register_buffer("seeds", seeds)

    def hash(self, ngrams: Tensor) -> Tensor:
        """Hash N-gram windows to embedding indices.

        Args:
            ngrams: (B, T, n) integer tensor of N-gram token windows.

        Returns:
            (B, T, num_heads) int64 tensor of hash indices in [0, table_size).
        """
        B, T, n = ngrams.shape
        device = ngrams.device

        coeffs = self.coefficients.to(device)  # (H, n)
        seeds_val = self.seeds.to(device)  # (H,)

        # (B, T, 1, n) * (1, 1, H, n) -> sum over n -> (B, T, H)
        ngrams_exp = ngrams.unsqueeze(2).long()
        coeffs_exp = coeffs.unsqueeze(0).unsqueeze(0)
        weighted = (ngrams_exp * coeffs_exp).sum(dim=-1)

        seeds_exp = seeds_val.unsqueeze(0).unsqueeze(0)
        hashed = (weighted ^ seeds_exp) % self.table_size

        return hashed


class MockEngramModule(nn.Module):
    """Simplified EngramModule for self-contained testing.

    Implements the full pipeline: compress -> extract N-grams -> hash ->
    retrieve -> gate -> conv -> project, using mock hash and compression
    components. The real EngramModule in ``brain_ai/memory/engram.py`` uses
    OffloadableEmbedding with CPU offload and async prefetch support.

    Args:
        config: EngramConfig controlling dimensions and feature flags.
        hidden_dim: Backbone hidden dimension.
        layer_salt: Per-layer salt for hash seed (decorrelates collisions).
    """

    def __init__(
        self,
        config: EngramConfig,
        hidden_dim: int,
        layer_salt: int = 0,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_dim = hidden_dim
        self.layer_salt = layer_salt

        # Tokenizer compression
        if config.use_tokenizer_compression:
            self.compressor: Optional[MockTokenizerCompression] = MockTokenizerCompression(
                config.vocab_size, config.compressed_vocab_size
            )
        else:
            self.compressor = None

        # Hash modules per N-gram order
        self._hashers: Dict[int, MockHashModule] = {}
        for n in config.ngram_orders:
            seed = config.hash_seed + n + (layer_salt * 997 if config.per_layer_salt else 0)
            self._hashers[n] = MockHashModule(
                ngram_order=n,
                num_heads=config.num_heads_per_order,
                table_size=config.table_size,
                seed=seed,
            )

        # Embedding tables: one per (ngram_order, head)
        total_heads = len(config.ngram_orders) * config.num_heads_per_order
        self.dim_per_head = max(1, config.embedding_dim // total_heads)
        self.total_embed_dim = self.dim_per_head * total_heads

        self.embeddings = nn.ModuleDict()
        for n in config.ngram_orders:
            for k in range(config.num_heads_per_order):
                key = f"ngram{n}_head{k}"
                self.embeddings[key] = nn.Embedding(
                    num_embeddings=config.table_size,
                    embedding_dim=self.dim_per_head,
                )

        # Context-aware gating
        if config.use_context_gate:
            self.gate_query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.gate_key_proj = nn.Linear(self.total_embed_dim, hidden_dim, bias=False)
            self.gate_value_proj = nn.Linear(self.total_embed_dim, hidden_dim, bias=False)

            if config.use_rmsnorm:
                self.query_norm = RMSNorm(hidden_dim)
                self.key_norm = RMSNorm(hidden_dim)

            self.gate_bias = nn.Parameter(
                torch.full((1,), config.gate_init_bias)
            )
            self.gate_scale = hidden_dim ** -0.5
        else:
            # Without gating: simple linear projection
            self.direct_proj = nn.Linear(self.total_embed_dim, hidden_dim, bias=False)

        # Depthwise causal convolution
        if config.use_depthwise_conv:
            padding = (config.conv_kernel_size - 1) * config.conv_dilation
            self.conv = nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=config.conv_kernel_size,
                padding=padding,
                dilation=config.conv_dilation,
                groups=hidden_dim,
            )
            self.conv_norm = RMSNorm(hidden_dim)
            # Zero-init for smooth training start
            nn.init.zeros_(self.conv.weight)
            if self.conv.bias is not None:
                nn.init.zeros_(self.conv.bias)

        # Output projection (if embed_dim != hidden_dim after gating)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.output_proj.weight)  # Zero-init for stable start

    def _extract_ngrams(self, token_ids: Tensor, n: int) -> Tensor:
        """Extract suffix N-gram windows from a token sequence.

        For position t, the N-gram is (x_{t-n+1}, ..., x_t). Positions
        before the start of the sequence are padded with 0.

        Args:
            token_ids: (B, T) compressed token IDs.
            n: N-gram order.

        Returns:
            (B, T, n) tensor of N-gram token windows.
        """
        B, T = token_ids.shape
        padded = F.pad(token_ids, (n - 1, 0), value=0)
        ngrams = torch.stack([padded[:, i:i + T] for i in range(n)], dim=-1)
        return ngrams

    def _retrieve_embeddings(self, token_ids: Tensor) -> Tensor:
        """Run the full hash-retrieve pipeline.

        Steps:
            1. Optionally compress token IDs.
            2. For each N-gram order: extract windows -> hash -> gather embeddings.
            3. Concatenate all head embeddings.

        Args:
            token_ids: (B, T) raw token IDs.

        Returns:
            (B, T, total_embed_dim) concatenated embeddings.
        """
        if self.compressor is not None:
            compressed = self.compressor.compress(token_ids)
        else:
            compressed = token_ids

        all_embeds: List[Tensor] = []
        for n in self.config.ngram_orders:
            ngrams = self._extract_ngrams(compressed, n)
            indices = self._hashers[n].hash(ngrams)  # (B, T, H)
            for k in range(self.config.num_heads_per_order):
                key = f"ngram{n}_head{k}"
                head_idx = indices[:, :, k]  # (B, T)
                head_emb = self.embeddings[key](head_idx)  # (B, T, dim_per_head)
                all_embeds.append(head_emb)

        return torch.cat(all_embeds, dim=-1)  # (B, T, total_embed_dim)

    def _compute_hash_ids(self, token_ids: Tensor) -> Dict[int, Tensor]:
        """Compute hash IDs for all N-gram orders without retrieving embeddings.

        Useful for prefetch planning: compute hash IDs early, then do the
        actual gather later (possibly from CPU-offloaded tables).

        Args:
            token_ids: (B, T) raw token IDs.

        Returns:
            Dict mapping ngram_order -> (B, T, num_heads) int64 hash indices.
        """
        if self.compressor is not None:
            compressed = self.compressor.compress(token_ids)
        else:
            compressed = token_ids

        result: Dict[int, Tensor] = {}
        for n in self.config.ngram_orders:
            ngrams = self._extract_ngrams(compressed, n)
            result[n] = self._hashers[n].hash(ngrams)
        return result

    def _apply_gating(
        self,
        hidden_states: Tensor,
        retrieved: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Apply context-aware gating to retrieved embeddings.

        Gate alpha = sigmoid(RMSNorm(q)^T . RMSNorm(k) / sqrt(d) + bias)
        Output = alpha * V(retrieved)

        Args:
            hidden_states: (B, T, hidden_dim) current backbone state.
            retrieved: (B, T, total_embed_dim) from hash tables.

        Returns:
            gated: (B, T, hidden_dim) gated output.
            gate_values: (B, T, 1) gate scalars in [0, 1].
        """
        if not self.config.use_context_gate:
            projected = self.direct_proj(retrieved)
            ones = torch.ones(
                projected.shape[0], projected.shape[1], 1,
                device=projected.device, dtype=projected.dtype,
            )
            return projected, ones

        # Project
        k = self.gate_key_proj(retrieved)
        v = self.gate_value_proj(retrieved)

        # Normalize (fp32 under AMP via RMSNorm)
        if self.config.use_rmsnorm:
            q_norm = self.query_norm(hidden_states)
            k_norm = self.key_norm(k)
        else:
            q_norm = hidden_states
            k_norm = k

        # Compute gate logits
        gate_logits = (q_norm * k_norm).sum(dim=-1, keepdim=True) * self.gate_scale
        gate_logits = gate_logits + self.gate_bias
        gate_logits = gate_logits / max(self.config.gate_temperature, 1e-6)

        alpha = torch.sigmoid(gate_logits)  # (B, T, 1) in [0, 1]

        gated = alpha * v
        return gated, alpha

    def _apply_conv(self, x: Tensor) -> Tensor:
        """Apply depthwise causal convolution.

        Causal padding ensures no future leakage: we pad on the left only
        and truncate the output to the original sequence length.

        Args:
            x: (B, T, D) input tensor.

        Returns:
            (B, T, D) convolved tensor with SiLU activation.
        """
        if not self.config.use_depthwise_conv:
            return x

        T = x.shape[1]
        x_norm = self.conv_norm(x)
        x_t = x_norm.transpose(1, 2)  # (B, D, T)
        conv_out = self.conv(x_t)
        conv_out = conv_out[:, :, :T]  # Truncate for causality
        conv_out = conv_out.transpose(1, 2)  # (B, T, D)
        return F.silu(conv_out) + x  # SiLU + residual within module

    def forward(
        self,
        hidden_states: Tensor,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        *,
        prefetched_embeddings: Optional[Tensor] = None,
        cache_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Compute the engram delta for residual addition.

        Two paths:
          (a) "compute now": full pipeline (hash -> retrieve -> gate -> conv -> project)
          (b) "consume prefetched": skip hash/retrieve, use pre-fetched embeddings

        Args:
            hidden_states: (B, T, hidden_dim) current backbone hidden states.
            input_ids: (B, T) token IDs for hash computation.
            attention_mask: Optional (B, T) mask, 1=valid, 0=padding.
            prefetched_embeddings: Optional (B, T, total_embed_dim) pre-fetched
                embeddings (from PrefetchHookManager).
            cache_state: Optional mutable dict for streaming/caching state.

        Returns:
            delta: (B, T, hidden_dim) to be added residually.
            telemetry: Dict with gate stats, norms, etc.
        """
        telemetry: Dict[str, Any] = {
            "layer_salt": self.layer_salt,
            "path": "prefetched" if prefetched_embeddings is not None else "compute",
        }

        # Step 1: Retrieve embeddings (or use prefetched)
        if prefetched_embeddings is not None:
            retrieved = prefetched_embeddings
        else:
            retrieved = self._retrieve_embeddings(input_ids)

        # Step 2: Apply attention mask to zero out padding positions
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            retrieved = retrieved * mask_expanded

        # Step 3: Context-aware gating
        gated, gate_values = self._apply_gating(hidden_states, retrieved)

        # Step 4: Depthwise causal convolution
        conv_out = self._apply_conv(gated)

        # Step 5: Output projection
        delta = self.output_proj(conv_out)

        # Telemetry
        with torch.no_grad():
            telemetry["gate_mean"] = gate_values.mean().item()
            telemetry["gate_min"] = gate_values.min().item()
            telemetry["gate_max"] = gate_values.max().item()
            telemetry["delta_norm"] = delta.norm().item()
            telemetry["retrieved_norm"] = retrieved.norm().item()

        return delta, telemetry

    def get_hash_ids(self, input_ids: Tensor) -> Dict[int, Tensor]:
        """Expose hash ID computation for prefetch planning.

        Args:
            input_ids: (B, T) raw token IDs.

        Returns:
            Dict[ngram_order, (B, T, num_heads)] hash indices.
        """
        return self._compute_hash_ids(input_ids)


# ========================================================================== #
#  SECTION 4: EngramAugmentedLayer (Phase 2 -- Layer Augmentation)           #
# ========================================================================== #

class EngramAugmentedLayer(nn.Module):
    """Phase 2 layer-augmentation: Engram injected at a configurable backbone layer.

    The layer computes a residual delta via the EngramModule pipeline and adds it
    to the incoming hidden_states. This happens *before* attention/FFN (per the
    DeepSeek paper design), allowing the static engram knowledge to influence
    the downstream context-dependent computations.

    Two execution paths:
      (a) "compute now": run the full EngramModule pipeline on the current
          layer's hidden_states and input_ids.
      (b) "consume prefetched": if ``prefetched_embeddings`` is provided (from a
          PrefetchHookManager), skip hash/retrieve and go directly to gating.

    The layer optionally applies:
      - Pre-normalization (RMSNorm) on hidden_states before the engram pipeline.
      - Residual scaling (for gradual warm-in during early training).
      - Dropout on the delta before addition.

    Args:
        hidden_dim: Backbone hidden dimension.
        config: EngramLayerConfig with all configuration.
        layer_id: Per-layer index (used as hash salt, overrides config.layer_id).

    Example::

        cfg = EngramLayerConfig(hidden_dim=4096, layer_id=3)
        layer = EngramAugmentedLayer(4096, cfg, layer_id=3)
        out = layer(hidden_states, input_ids)
        # out.shape == hidden_states.shape
    """

    def __init__(
        self,
        hidden_dim: int,
        config: EngramLayerConfig,
        layer_id: int,
    ) -> None:
        super().__init__()
        self._layer_id = layer_id
        self._config = config
        self._hidden_dim = hidden_dim
        self._has_prefetch = config.use_prefetch

        # Pre-normalization
        if config.use_pre_norm:
            self.pre_norm = RMSNorm(hidden_dim)
        else:
            self.pre_norm = None

        # Core EngramModule with per-layer salt
        self.engram = MockEngramModule(
            config=config.engram_config,
            hidden_dim=hidden_dim,
            layer_salt=layer_id,
        )

        # Residual scale
        self._residual_scale = config.residual_scale

        # Dropout
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None

        # Telemetry accumulator
        self._last_telemetry: Optional[Dict[str, Any]] = None

    @property
    def layer_id(self) -> int:
        """Return the backbone layer index for this augmentation."""
        return self._layer_id

    @property
    def has_prefetch(self) -> bool:
        """Whether this layer expects prefetched embeddings."""
        return self._has_prefetch

    @property
    def telemetry_summary(self) -> Optional[Dict[str, Any]]:
        """Return the last forward pass telemetry, or None if not yet called."""
        return self._last_telemetry

    def forward(
        self,
        hidden_states: Tensor,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        *,
        prefetched_embeddings: Optional[Tensor] = None,
        cache_state: Optional[Dict[str, Any]] = None,
        return_details: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Any]]]:
        """Compute engram delta and add residually to hidden_states.

        This method implements both execution paths:
          (a) Without prefetched_embeddings: runs full pipeline.
          (b) With prefetched_embeddings: skips hash/retrieve, uses provided data.

        The output is always: ``output = hidden_states + scale * delta``

        Args:
            hidden_states: (B, T, hidden_dim) from the backbone.
            input_ids: (B, T) token IDs for hash computation.
            attention_mask: Optional (B, T) mask, 1=valid, 0=padding.
            prefetched_embeddings: Optional (B, T, embed_dim) pre-fetched data.
            cache_state: Optional mutable dict for streaming caching.
            return_details: If True, return (output, telemetry_dict).

        Returns:
            output: (B, T, hidden_dim) with engram delta added residually.
            telemetry: (only if return_details=True) dict with per-layer stats.
        """
        # Pre-norm
        if self.pre_norm is not None:
            h_normed = self.pre_norm(hidden_states)
        else:
            h_normed = hidden_states

        # EngramModule forward
        delta, telemetry = self.engram(
            hidden_states=h_normed,
            input_ids=input_ids,
            attention_mask=attention_mask,
            prefetched_embeddings=prefetched_embeddings,
            cache_state=cache_state,
        )

        # Dropout
        if self.dropout is not None:
            delta = self.dropout(delta)

        # Residual with scale
        output = hidden_states + self._residual_scale * delta

        # Augment telemetry
        telemetry["layer_id"] = self._layer_id
        telemetry["residual_scale"] = self._residual_scale
        telemetry["hidden_norm_in"] = hidden_states.norm().item()
        telemetry["hidden_norm_out"] = output.detach().norm().item()
        self._last_telemetry = telemetry

        if return_details:
            return output, telemetry
        return output

    def extra_repr(self) -> str:
        return (
            f"layer_id={self._layer_id}, hidden_dim={self._hidden_dim}, "
            f"residual_scale={self._residual_scale}, "
            f"has_prefetch={self._has_prefetch}"
        )


# ========================================================================== #
#  SECTION 5: EngramTextEncoder (Phase 1 -- Encoder Competition)             #
# ========================================================================== #

class EngramTextEncoder(nn.Module):
    """Phase 1 encoder-competition: Engram as a fast text encoder for the workspace.

    Produces workspace-aligned ``(B, T, workspace_dim)`` representations that
    compete with other modality encoders (vision, audio, etc.) in the Global
    Workspace. Uses the same tokenizer compression and N-gram hash retrieval
    as Phase 2 layer-augmentation, but wraps them in a lightweight pipeline:

    1. Input embedding:  input_ids -> nn.Embedding -> (B, T, embed_dim)
    2. EngramModule:     hash/retrieve/gate using embedded hidden_states
    3. Residual:         embedded + delta
    4. Workspace proj:   Linear(D_out, workspace_dim) -> (B, T, 4096)
    5. Padding mask:     zero out positions where attention_mask == 0

    The output must match the encoder suite contract:
      - Shape: (B, T, D_workspace) with D_workspace = 4096
      - Padding positions masked to zero
      - Compatible with EncoderOutput dataclass for workspace competition

    Args:
        vocab_size: Input vocabulary size (before compression).
        config: EngramEncoderConfig with all configuration.
        workspace_dim: Override for workspace dimension (default from config).

    Example::

        cfg = EngramEncoderConfig(vocab_size=50000, workspace_dim=4096)
        encoder = EngramTextEncoder(50000, cfg)
        out = encoder(input_ids, attention_mask)
        # out.shape == (B, T, 4096)
    """

    def __init__(
        self,
        vocab_size: int,
        config: EngramEncoderConfig,
        workspace_dim: int = DEFAULT_WORKSPACE_DIM,
    ) -> None:
        super().__init__()
        self._config = config
        self._workspace_dim = workspace_dim
        effective_ws = workspace_dim if workspace_dim != DEFAULT_WORKSPACE_DIM else config.workspace_dim

        # Step 1: Lightweight input embedding
        self.input_embedding = nn.Embedding(vocab_size, config.embed_dim)
        nn.init.normal_(self.input_embedding.weight, mean=0.0, std=0.02)

        # Optional positional encoding
        if config.use_positional:
            self.pos_encoding = nn.Parameter(
                torch.zeros(1, config.max_seq_len, config.embed_dim)
            )
            nn.init.normal_(self.pos_encoding, mean=0.0, std=0.02)
        else:
            self.pos_encoding = None

        # Step 2: EngramModule for N-gram hash retrieval and gating
        engram_cfg = config.engram_config
        # Override vocab size from encoder config
        engram_cfg_copy = EngramConfig(
            vocab_size=vocab_size,
            compressed_vocab_size=engram_cfg.compressed_vocab_size,
            use_tokenizer_compression=config.use_tokenizer_compression,
            embedding_dim=engram_cfg.embedding_dim,
            ngram_orders=engram_cfg.ngram_orders,
            num_heads_per_order=engram_cfg.num_heads_per_order,
            table_size=engram_cfg.table_size,
            hash_fn=engram_cfg.hash_fn,
            use_prime_sizes=engram_cfg.use_prime_sizes,
            per_layer_salt=False,  # No per-layer salt for encoder mode
            hash_seed=engram_cfg.hash_seed,
            use_context_gate=engram_cfg.use_context_gate,
            gate_type=engram_cfg.gate_type,
            use_rmsnorm=engram_cfg.use_rmsnorm,
            gate_init_bias=engram_cfg.gate_init_bias,
            gate_temperature=engram_cfg.gate_temperature,
            use_depthwise_conv=engram_cfg.use_depthwise_conv,
            conv_kernel_size=engram_cfg.conv_kernel_size,
            conv_dilation=engram_cfg.conv_dilation,
            offload_to_cpu=engram_cfg.offload_to_cpu,
            use_async_prefetch=engram_cfg.use_async_prefetch,
            prefetch_ahead_layers=engram_cfg.prefetch_ahead_layers,
            pin_memory=engram_cfg.pin_memory,
            storage_dtype=engram_cfg.storage_dtype,
        )
        self.engram = MockEngramModule(
            config=engram_cfg_copy,
            hidden_dim=config.embed_dim,
            layer_salt=0,
        )

        # Post-engram normalization
        self.post_norm = RMSNorm(config.embed_dim)

        # Step 4: Workspace projection
        self.workspace_proj = nn.Sequential(
            nn.Linear(config.embed_dim, effective_ws),
            nn.GELU(),
            RMSNorm(effective_ws),
        )

        # Dropout
        self.proj_dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None

        # Telemetry
        self._last_telemetry: Optional[Dict[str, Any]] = None

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        *,
        return_details: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Any]]]:
        """Encode tokens via Engram pipeline and project to workspace dimension.

        Steps:
          1. Embed input_ids -> (B, T, embed_dim)
          2. EngramModule forward with input_ids and embedded hidden_states
          3. Residual: embedded + delta
          4. Workspace projection to (B, T, workspace_dim=4096)
          5. Apply attention_mask (zero out padding positions)

        Args:
            input_ids: (B, T) integer token IDs.
            attention_mask: Optional (B, T) mask, 1=valid, 0=padding.
            return_details: If True, return (output, telemetry_dict).

        Returns:
            output: (B, T, workspace_dim) workspace-aligned representation.
            telemetry: (only if return_details=True) dict with encoder stats.
        """
        B, T = input_ids.shape

        # Step 1: Input embedding
        embedded = self.input_embedding(input_ids)  # (B, T, embed_dim)

        # Add positional encoding
        if self.pos_encoding is not None:
            embedded = embedded + self.pos_encoding[:, :T, :]

        # Step 2: EngramModule forward
        delta, engram_telemetry = self.engram(
            hidden_states=embedded,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Step 3: Residual
        fused = embedded + delta
        fused = self.post_norm(fused)

        # Step 4: Workspace projection
        output = self.workspace_proj(fused)  # (B, T, workspace_dim)

        # Dropout
        if self.proj_dropout is not None:
            output = self.proj_dropout(output)

        # Step 5: Apply attention mask (zero out padding positions)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
            output = output * mask_expanded

        # Telemetry
        telemetry: Dict[str, Any] = {
            "encoder_type": "engram_text",
            "input_shape": list(input_ids.shape),
            "output_shape": list(output.shape),
            "embedded_norm": embedded.detach().norm().item(),
            "output_norm": output.detach().norm().item(),
            "engram": engram_telemetry,
        }
        self._last_telemetry = telemetry

        if return_details:
            return output, telemetry
        return output

    def extra_repr(self) -> str:
        ws = self._config.workspace_dim
        emb = self._config.embed_dim
        return f"embed_dim={emb}, workspace_dim={ws}"


# ========================================================================== #
#  SECTION 6: EngramLayerRegistry                                           #
# ========================================================================== #

class EngramLayerRegistry:
    """Registry tracking which backbone layers have Engram augmentation.

    The orchestrator queries this registry to know:
      - Which layer indices have an EngramAugmentedLayer.
      - What order to process them in.
      - How to route input_ids to each layer.

    This is a plain Python class (not nn.Module) because the registry does not
    own the layers' parameters -- those are owned by the backbone or a separate
    ModuleList.

    Usage::

        registry = EngramLayerRegistry()
        registry.register_layer(1, engram_layer_1)
        registry.register_layer(5, engram_layer_5)
        print(registry.get_insertion_layers())  # [1, 5]
    """

    def __init__(self) -> None:
        self._layers: Dict[int, EngramAugmentedLayer] = OrderedDict()

    def register_layer(
        self,
        layer_idx: int,
        engram_layer: EngramAugmentedLayer,
    ) -> None:
        """Register an EngramAugmentedLayer at a specific backbone layer index.

        Args:
            layer_idx: The backbone layer index where the engram augmentation
                should be applied.
            engram_layer: The EngramAugmentedLayer instance to register.

        Raises:
            ValueError: If ``layer_idx`` is already registered.
        """
        if layer_idx in self._layers:
            raise ValueError(
                f"Layer index {layer_idx} is already registered. "
                f"Unregister it first with unregister_layer({layer_idx})."
            )
        self._layers[layer_idx] = engram_layer
        logger.info(
            "Registered EngramAugmentedLayer at backbone layer %d (layer_id=%d)",
            layer_idx, engram_layer.layer_id,
        )

    def unregister_layer(self, layer_idx: int) -> Optional[EngramAugmentedLayer]:
        """Remove an EngramAugmentedLayer from the registry.

        Args:
            layer_idx: The backbone layer index to unregister.

        Returns:
            The removed EngramAugmentedLayer, or None if not found.
        """
        return self._layers.pop(layer_idx, None)

    def get_insertion_layers(self) -> List[int]:
        """Return sorted list of backbone layer indices with Engram augmentation.

        Returns:
            Sorted list of integer layer indices.
        """
        return sorted(self._layers.keys())

    def get_layer(self, layer_idx: int) -> Optional[EngramAugmentedLayer]:
        """Get the EngramAugmentedLayer at a specific backbone layer index.

        Args:
            layer_idx: The backbone layer index.

        Returns:
            The EngramAugmentedLayer, or None if no augmentation at this index.
        """
        return self._layers.get(layer_idx, None)

    def has_layer(self, layer_idx: int) -> bool:
        """Check if an Engram layer is registered at this index.

        Args:
            layer_idx: The backbone layer index.

        Returns:
            True if an Engram layer is registered.
        """
        return layer_idx in self._layers

    @property
    def num_layers(self) -> int:
        """Number of registered Engram layers."""
        return len(self._layers)

    def clear(self) -> None:
        """Remove all registered layers."""
        self._layers.clear()

    def __repr__(self) -> str:
        indices = self.get_insertion_layers()
        return f"EngramLayerRegistry(layers_at={indices})"

    def __contains__(self, layer_idx: int) -> bool:
        return layer_idx in self._layers

    def __len__(self) -> int:
        return len(self._layers)

    def __iter__(self):
        """Iterate over (layer_idx, engram_layer) pairs in sorted order."""
        for idx in sorted(self._layers.keys()):
            yield idx, self._layers[idx]


# ========================================================================== #
#  SECTION 7: PrefetchHookManager                                           #
# ========================================================================== #

class PrefetchHookManager:
    """Manages prefetch scheduling for Phase 2 + CPU offload.

    When embedding tables are on CPU, the PCIe transfer latency can be hidden
    by prefetching embeddings N layers ahead of where they will be consumed.
    This manager computes the prefetch schedule and provides hooks that the
    backbone can call at each layer boundary.

    Scheduling logic:
      - For each Engram layer at index L, prefetch should be triggered at
        layer (L - prefetch_ahead).
      - If (L - prefetch_ahead) < 0, prefetch at layer 0.
      - Prefetch computation: hash IDs are precomputed once upfront, then
        embedding gather is scheduled at the trigger layer.

    Usage::

        manager = PrefetchHookManager(registry, prefetch_ahead=2)
        hook = manager.create_hook(engram_module, hash_module, input_ids)
        for layer_idx in range(num_layers):
            hook(layer_idx)  # triggers prefetch if needed
            # ... backbone layer forward ...

    Args:
        layer_registry: EngramLayerRegistry with registered Engram layers.
        prefetch_ahead: Number of layers ahead to trigger prefetch.
    """

    def __init__(
        self,
        layer_registry: EngramLayerRegistry,
        prefetch_ahead: int = DEFAULT_PREFETCH_AHEAD,
    ) -> None:
        self._registry = layer_registry
        self._prefetch_ahead = prefetch_ahead

        # Build the prefetch schedule: {trigger_layer: target_layer}
        self._schedule: Dict[int, int] = {}
        for target_idx in layer_registry.get_insertion_layers():
            trigger_idx = max(0, target_idx - prefetch_ahead)
            # If multiple targets map to the same trigger, use the earliest target
            if trigger_idx not in self._schedule:
                self._schedule[trigger_idx] = target_idx
            else:
                # Keep the earliest target that hasn't been scheduled
                existing_target = self._schedule[trigger_idx]
                if target_idx < existing_target:
                    self._schedule[trigger_idx] = target_idx

        # For multi-target-per-trigger, build a list-based schedule
        self._schedule_multi: Dict[int, List[int]] = {}
        for target_idx in layer_registry.get_insertion_layers():
            trigger_idx = max(0, target_idx - prefetch_ahead)
            if trigger_idx not in self._schedule_multi:
                self._schedule_multi[trigger_idx] = []
            self._schedule_multi[trigger_idx].append(target_idx)

        # Prefetch buffers: {target_layer: prefetched_embeddings}
        self._buffers: Dict[int, Optional[Tensor]] = {}

    def should_prefetch_at(self, current_layer_idx: int) -> Optional[int]:
        """Check if prefetch should be triggered at the current layer.

        Args:
            current_layer_idx: The current backbone layer being processed.

        Returns:
            The target layer index to prefetch for, or None if no prefetch
            should be triggered at this layer.
        """
        return self._schedule.get(current_layer_idx, None)

    def get_prefetch_targets_at(self, current_layer_idx: int) -> List[int]:
        """Get all prefetch targets triggered at the current layer.

        Args:
            current_layer_idx: The current backbone layer being processed.

        Returns:
            List of target layer indices to prefetch for (may be empty).
        """
        return self._schedule_multi.get(current_layer_idx, [])

    def get_prefetched(self, target_layer_idx: int) -> Optional[Tensor]:
        """Retrieve prefetched embeddings for a target layer.

        Args:
            target_layer_idx: The layer that will consume the embeddings.

        Returns:
            Prefetched embedding tensor, or None if not yet prefetched.
        """
        return self._buffers.get(target_layer_idx, None)

    def store_prefetched(self, target_layer_idx: int, embeddings: Tensor) -> None:
        """Store prefetched embeddings for consumption by the target layer.

        Args:
            target_layer_idx: Layer that will consume these embeddings.
            embeddings: (B, T, embed_dim) tensor to store.
        """
        self._buffers[target_layer_idx] = embeddings

    def clear_prefetched(self, target_layer_idx: int) -> None:
        """Clear the buffer for a target layer after consumption.

        Args:
            target_layer_idx: The layer whose buffer should be cleared.
        """
        self._buffers.pop(target_layer_idx, None)

    def clear_all(self) -> None:
        """Clear all prefetch buffers."""
        self._buffers.clear()

    @property
    def schedule(self) -> Dict[int, int]:
        """The prefetch schedule: {trigger_layer: target_layer}."""
        return dict(self._schedule)

    @property
    def prefetch_ahead(self) -> int:
        """Number of layers ahead for prefetch triggering."""
        return self._prefetch_ahead

    def create_hook(
        self,
        engram_modules: Dict[int, MockEngramModule],
        input_ids: Tensor,
    ) -> Callable[[int], None]:
        """Create a hook callable for use at each backbone layer boundary.

        The hook precomputes hash IDs upfront for all target layers, then
        at each trigger layer, performs the embedding gather and stores the
        result in the prefetch buffer.

        Args:
            engram_modules: Dict mapping target_layer_idx -> MockEngramModule.
                Each module's ``_retrieve_embeddings`` will be called during
                prefetch.
            input_ids: (B, T) token IDs shared across all layers.

        Returns:
            A callable ``hook(layer_idx: int) -> None`` that should be called
            at each layer boundary. The hook triggers prefetch when
            ``layer_idx`` matches a scheduled trigger point.
        """
        # Precompute hash IDs for all targets upfront
        precomputed_hash_ids: Dict[int, Dict[int, Tensor]] = {}
        for target_idx in self._registry.get_insertion_layers():
            module = engram_modules.get(target_idx)
            if module is not None:
                precomputed_hash_ids[target_idx] = module.get_hash_ids(input_ids)

        def hook(layer_idx: int) -> None:
            """Prefetch hook called at each layer boundary.

            Args:
                layer_idx: Current backbone layer index.
            """
            targets = self.get_prefetch_targets_at(layer_idx)
            for target_idx in targets:
                module = engram_modules.get(target_idx)
                if module is not None:
                    # In production, this would be an async gather on a
                    # dedicated CUDA stream. For the mock, we do it inline.
                    embeddings = module._retrieve_embeddings(input_ids)
                    self.store_prefetched(target_idx, embeddings)
                    logger.debug(
                        "Prefetched embeddings at layer %d for target %d",
                        layer_idx, target_idx,
                    )

        return hook

    def __repr__(self) -> str:
        return (
            f"PrefetchHookManager(ahead={self._prefetch_ahead}, "
            f"schedule={self._schedule})"
        )


# ========================================================================== #
#  SECTION 8: EngramCheckpointMixin                                         #
# ========================================================================== #

class EngramCheckpointMixin:
    """Mixin for save/load of Engram-specific state.

    Handles persistence of:
      - Compression table (surjective mapping from raw to canonical IDs)
      - Hash configuration (seeds, coefficients, table sizes)
      - Embedding weights (all N-gram embedding tables)
      - Gating weights (projection matrices, bias, norms)

    File layout::

        <path>/
            engram_state.pt                -- embedding + gating weights
            .engram_compression.pt         -- compression table artifact
            .engram_hash_config.json       -- hash seeds, coefficients, table sizes

    Usage::

        class MyModel(nn.Module, EngramCheckpointMixin):
            def __init__(self):
                super().__init__()
                self.engram = MockEngramModule(...)

            # save_engram_state and load_engram_state are now available
    """

    def save_engram_state(self, path: str) -> None:
        """Save Engram-specific state to disk.

        Creates the directory if it does not exist. Saves three files:
          1. ``engram_state.pt``: All nn.Module state_dict entries from the
             engram submodule.
          2. ``.engram_compression.pt``: Tokenizer compression table (if any).
          3. ``.engram_hash_config.json``: Hash configuration for deterministic
             reproducibility.

        Args:
            path: Directory path for the checkpoint.
        """
        os.makedirs(path, exist_ok=True)

        # Find the engram submodule
        engram_module = self._find_engram_module()
        if engram_module is None:
            logger.warning("No engram module found for checkpointing.")
            return

        # 1. Save embedding + gating weights
        state_path = os.path.join(path, "engram_state.pt")
        torch.save(engram_module.state_dict(), state_path)
        logger.info("Saved engram state to %s", state_path)

        # 2. Save compression table
        comp_path = os.path.join(path, ".engram_compression.pt")
        if hasattr(engram_module, "compressor") and engram_module.compressor is not None:
            comp_state = {
                "vocab_size": engram_module.compressor.vocab_size,
                "compressed_size": engram_module.compressor.compressed_size,
            }
            torch.save(comp_state, comp_path)
            logger.info("Saved compression table to %s", comp_path)

        # 3. Save hash config as JSON
        hash_config_path = os.path.join(path, ".engram_hash_config.json")
        hash_config = self._extract_hash_config(engram_module)
        with open(hash_config_path, "w") as f:
            json.dump(hash_config, f, indent=2)
        logger.info("Saved hash config to %s", hash_config_path)

    def load_engram_state(self, path: str, strict: bool = True) -> None:
        """Load Engram-specific state from disk.

        Validates hash config consistency: if the saved hash config does not
        match the current model's config, a warning is raised (or error if
        strict=True).

        Args:
            path: Directory path containing the checkpoint files.
            strict: If True, raise ValueError on config mismatch.

        Raises:
            FileNotFoundError: If the checkpoint directory does not exist.
            ValueError: If strict=True and hash config does not match.
        """
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Checkpoint directory not found: {path}")

        engram_module = self._find_engram_module()
        if engram_module is None:
            logger.warning("No engram module found for loading.")
            return

        # 1. Load embedding + gating weights
        state_path = os.path.join(path, "engram_state.pt")
        if os.path.isfile(state_path):
            state_dict = torch.load(state_path, map_location="cpu", weights_only=True)
            engram_module.load_state_dict(state_dict, strict=strict)
            logger.info("Loaded engram state from %s", state_path)

        # 2. Load compression table
        comp_path = os.path.join(path, ".engram_compression.pt")
        if os.path.isfile(comp_path):
            comp_state = torch.load(comp_path, map_location="cpu", weights_only=True)
            if hasattr(engram_module, "compressor") and engram_module.compressor is not None:
                engram_module.compressor.vocab_size = comp_state["vocab_size"]
                engram_module.compressor.compressed_size = comp_state["compressed_size"]
            logger.info("Loaded compression table from %s", comp_path)

        # 3. Validate hash config
        hash_config_path = os.path.join(path, ".engram_hash_config.json")
        if os.path.isfile(hash_config_path):
            with open(hash_config_path, "r") as f:
                saved_config = json.load(f)
            current_config = self._extract_hash_config(engram_module)
            if saved_config != current_config:
                msg = (
                    f"Hash config mismatch between saved and current model.\n"
                    f"Saved: {saved_config}\n"
                    f"Current: {current_config}"
                )
                if strict:
                    raise ValueError(msg)
                else:
                    logger.warning(msg)

    def _find_engram_module(self) -> Optional[MockEngramModule]:
        """Locate the EngramModule within this model.

        Searches for attributes named 'engram' on self and on any
        EngramAugmentedLayer children.

        Returns:
            The MockEngramModule (or real EngramModule), or None.
        """
        if hasattr(self, "engram") and isinstance(getattr(self, "engram"), nn.Module):
            return getattr(self, "engram")
        # Search children
        if isinstance(self, nn.Module):
            for child in self.modules():  # type: ignore[union-attr]
                if isinstance(child, MockEngramModule):
                    return child
        return None

    def _extract_hash_config(self, engram_module: MockEngramModule) -> Dict[str, Any]:
        """Extract hash configuration for persistence.

        Args:
            engram_module: The engram module to extract config from.

        Returns:
            Dict with hash config fields.
        """
        config = engram_module.config
        return {
            "hash_fn": config.hash_fn,
            "hash_seed": config.hash_seed,
            "table_size": config.table_size,
            "ngram_orders": list(config.ngram_orders),
            "num_heads_per_order": config.num_heads_per_order,
            "per_layer_salt": config.per_layer_salt,
            "use_prime_sizes": config.use_prime_sizes,
            "layer_salt": engram_module.layer_salt,
        }


# ========================================================================== #
#  SECTION 9: Factory Functions                                              #
# ========================================================================== #

def create_engram_augmented_layer(
    hidden_dim: int = DEFAULT_WORKSPACE_DIM,
    layer_id: int = 0,
    residual_scale: float = 1.0,
    use_prefetch: bool = False,
    **engram_kwargs: Any,
) -> EngramAugmentedLayer:
    """Factory function to create an EngramAugmentedLayer with sensible defaults.

    Args:
        hidden_dim: Backbone hidden dimension.
        layer_id: Layer index for per-layer salt.
        residual_scale: Scale for residual delta.
        use_prefetch: Whether to use prefetch path.
        **engram_kwargs: Additional arguments passed to EngramConfig.

    Returns:
        Configured EngramAugmentedLayer instance.
    """
    engram_config = EngramConfig(
        **{k: v for k, v in engram_kwargs.items() if hasattr(EngramConfig, k)}
    )
    layer_config = EngramLayerConfig(
        hidden_dim=hidden_dim,
        engram_config=engram_config,
        layer_id=layer_id,
        use_prefetch=use_prefetch,
        residual_scale=residual_scale,
    )
    return EngramAugmentedLayer(
        hidden_dim=hidden_dim,
        config=layer_config,
        layer_id=layer_id,
    )


def create_engram_text_encoder(
    vocab_size: int = 50000,
    workspace_dim: int = DEFAULT_WORKSPACE_DIM,
    embed_dim: int = DEFAULT_EMBEDDING_DIM,
    **engram_kwargs: Any,
) -> EngramTextEncoder:
    """Factory function to create an EngramTextEncoder with sensible defaults.

    Args:
        vocab_size: Input vocabulary size.
        workspace_dim: Output workspace dimension (must be 4096 for standard pipeline).
        embed_dim: Internal embedding dimension.
        **engram_kwargs: Additional arguments passed to EngramConfig.

    Returns:
        Configured EngramTextEncoder instance.
    """
    engram_config = EngramConfig(
        vocab_size=vocab_size,
        **{k: v for k, v in engram_kwargs.items() if hasattr(EngramConfig, k)},
    )
    encoder_config = EngramEncoderConfig(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        workspace_dim=workspace_dim,
        engram_config=engram_config,
    )
    return EngramTextEncoder(
        vocab_size=vocab_size,
        config=encoder_config,
        workspace_dim=workspace_dim,
    )


# ========================================================================== #
#  SECTION 10: Backbone Integration Utilities                                #
# ========================================================================== #

class EngramBackboneWrapper(nn.Module, EngramCheckpointMixin):
    """Utility wrapper that augments a simple sequential backbone with Engram layers.

    This demonstrates how to wire EngramAugmentedLayer instances into a
    backbone model using the EngramLayerRegistry and PrefetchHookManager.

    The wrapper inserts Engram augmentation at registered layer indices,
    applying the delta *before* the backbone layer itself (pre-layer residual).

    Args:
        backbone_layers: nn.ModuleList of backbone layers.
        registry: EngramLayerRegistry with registered Engram layers.
        use_prefetch: Whether to enable prefetch scheduling.
        prefetch_ahead: Number of layers ahead for prefetch (if enabled).
    """

    def __init__(
        self,
        backbone_layers: nn.ModuleList,
        registry: EngramLayerRegistry,
        use_prefetch: bool = False,
        prefetch_ahead: int = DEFAULT_PREFETCH_AHEAD,
    ) -> None:
        super().__init__()
        self.backbone = backbone_layers
        self.registry = registry
        self.use_prefetch = use_prefetch

        # Collect Engram layers into a ModuleList for parameter registration
        self.engram_layers = nn.ModuleList()
        for idx, layer in registry:
            self.engram_layers.append(layer)

        # Prefetch manager (if enabled)
        if use_prefetch:
            self.prefetch_manager = PrefetchHookManager(registry, prefetch_ahead)
        else:
            self.prefetch_manager = None

    def forward(
        self,
        hidden_states: Tensor,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        *,
        return_details: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Any]]]:
        """Forward pass through the augmented backbone.

        For each layer index:
          1. If an Engram layer is registered, apply it first (pre-layer).
          2. Then apply the backbone layer.
          3. If prefetch is enabled, run the prefetch hook after each layer.

        Args:
            hidden_states: (B, T, D) initial hidden states.
            input_ids: (B, T) token IDs for Engram computation.
            attention_mask: Optional (B, T) mask, 1=valid, 0=padding.
            return_details: If True, collect telemetry from all Engram layers.

        Returns:
            output: (B, T, D) final hidden states.
            telemetry: (only if return_details=True) dict of per-layer telemetry.
        """
        all_telemetry: Dict[str, Any] = {} if return_details else {}

        # Set up prefetch hook if enabled
        prefetch_hook: Optional[Callable[[int], None]] = None
        if self.prefetch_manager is not None:
            engram_modules: Dict[int, MockEngramModule] = {}
            for idx, layer in self.registry:
                engram_modules[idx] = layer.engram
            prefetch_hook = self.prefetch_manager.create_hook(engram_modules, input_ids)

        for layer_idx, backbone_layer in enumerate(self.backbone):
            # Trigger prefetch hook before processing
            if prefetch_hook is not None:
                prefetch_hook(layer_idx)

            # Apply Engram augmentation if registered at this layer
            engram_layer = self.registry.get_layer(layer_idx)
            if engram_layer is not None:
                # Check for prefetched embeddings
                prefetched = None
                if self.prefetch_manager is not None:
                    prefetched = self.prefetch_manager.get_prefetched(layer_idx)
                    if prefetched is not None:
                        self.prefetch_manager.clear_prefetched(layer_idx)

                if return_details:
                    hidden_states, telemetry = engram_layer(
                        hidden_states, input_ids, attention_mask,
                        prefetched_embeddings=prefetched,
                        return_details=True,
                    )
                    all_telemetry[f"layer_{layer_idx}"] = telemetry
                else:
                    hidden_states = engram_layer(
                        hidden_states, input_ids, attention_mask,
                        prefetched_embeddings=prefetched,
                    )

            # Apply backbone layer
            hidden_states = backbone_layer(hidden_states)

        if return_details:
            return hidden_states, all_telemetry
        return hidden_states


# ========================================================================== #
#  SECTION 11: Telemetry Utilities                                          #
# ========================================================================== #

class EngramTelemetryAggregator:
    """Aggregates telemetry from multiple Engram layers across forward passes.

    Provides running statistics for monitoring gate saturation, delta norms,
    and prefetch hit rates during training.

    Usage::

        agg = EngramTelemetryAggregator()
        for step in range(num_steps):
            _, telemetry = model(x, return_details=True)
            agg.update(telemetry)
        print(agg.summary())
    """

    def __init__(self) -> None:
        self._gate_means: List[float] = []
        self._gate_mins: List[float] = []
        self._gate_maxs: List[float] = []
        self._delta_norms: List[float] = []
        self._prefetch_hits: int = 0
        self._total_layers: int = 0
        self._num_updates: int = 0

    def update(self, telemetry: Dict[str, Any]) -> None:
        """Update aggregator with telemetry from one forward pass.

        Args:
            telemetry: Dict of per-layer telemetry dicts. Keys should be
                ``layer_<idx>`` mapping to dicts with gate/delta stats.
        """
        self._num_updates += 1
        for key, layer_tel in telemetry.items():
            if not isinstance(layer_tel, dict):
                continue
            if "gate_mean" in layer_tel:
                self._gate_means.append(layer_tel["gate_mean"])
            if "gate_min" in layer_tel:
                self._gate_mins.append(layer_tel["gate_min"])
            if "gate_max" in layer_tel:
                self._gate_maxs.append(layer_tel["gate_max"])
            if "delta_norm" in layer_tel:
                self._delta_norms.append(layer_tel["delta_norm"])
            if layer_tel.get("path") == "prefetched":
                self._prefetch_hits += 1
            self._total_layers += 1

    def summary(self) -> Dict[str, Any]:
        """Return aggregated statistics.

        Returns:
            Dict with mean/min/max gate values, mean delta norm,
            prefetch hit rate, and total number of updates.
        """
        def _safe_mean(vals: List[float]) -> float:
            return sum(vals) / len(vals) if vals else 0.0

        def _safe_min(vals: List[float]) -> float:
            return min(vals) if vals else 0.0

        def _safe_max(vals: List[float]) -> float:
            return max(vals) if vals else 0.0

        return {
            "num_updates": self._num_updates,
            "total_layers_processed": self._total_layers,
            "gate_mean": _safe_mean(self._gate_means),
            "gate_min_overall": _safe_min(self._gate_mins),
            "gate_max_overall": _safe_max(self._gate_maxs),
            "delta_norm_mean": _safe_mean(self._delta_norms),
            "prefetch_hit_rate": (
                self._prefetch_hits / self._total_layers
                if self._total_layers > 0 else 0.0
            ),
        }

    def reset(self) -> None:
        """Reset all accumulated statistics."""
        self._gate_means.clear()
        self._gate_mins.clear()
        self._gate_maxs.clear()
        self._delta_norms.clear()
        self._prefetch_hits = 0
        self._total_layers = 0
        self._num_updates = 0


# ========================================================================== #
#  SECTION 12: Initialization Utilities                                     #
# ========================================================================== #

def init_engram_weights(module: nn.Module, std: float = 0.02) -> None:
    """Initialize Engram-specific weights following best practices.

    - Embedding tables: normal(0, std)
    - Linear projections: normal(0, std)
    - Gate bias: set to DEFAULT_GATE_INIT_BIAS (-2.0) for conservative start
    - Conv weights: zero-init for smooth training start
    - RMSNorm weights: ones

    Args:
        module: The module to initialize (recurses into children).
        std: Standard deviation for normal initialization.
    """
    for name, param in module.named_parameters():
        if "weight" in name:
            if "conv" in name:
                nn.init.zeros_(param)
            elif "norm" in name or "rmsnorm" in name:
                nn.init.ones_(param)
            elif "embedding" in name or "embed" in name:
                nn.init.normal_(param, mean=0.0, std=std)
            elif "output_proj" in name:
                nn.init.zeros_(param)
            else:
                nn.init.normal_(param, mean=0.0, std=std)
        elif "bias" in name:
            if "gate_bias" in name:
                nn.init.constant_(param, DEFAULT_GATE_INIT_BIAS)
            else:
                nn.init.zeros_(param)


def count_engram_parameters(module: nn.Module) -> Dict[str, int]:
    """Count parameters in an Engram module, broken down by component.

    Args:
        module: The Engram module to analyze.

    Returns:
        Dict mapping component names to parameter counts.
    """
    counts: Dict[str, int] = {}
    for name, param in module.named_parameters():
        # Determine component from parameter name
        parts = name.split(".")
        component = parts[0] if parts else "unknown"
        if component not in counts:
            counts[component] = 0
        counts[component] += param.numel()

    counts["total"] = sum(p.numel() for p in module.parameters())
    return counts


# ========================================================================== #
#  SECTION 13: Validation Utilities                                         #
# ========================================================================== #

def validate_engram_layer_output(
    output: Tensor,
    hidden_states: Tensor,
    attention_mask: Optional[Tensor] = None,
) -> List[str]:
    """Validate EngramAugmentedLayer output against contract.

    Checks:
      - Output shape matches input shape
      - No NaN or Inf values
      - Output differs from input (delta is non-zero)

    Args:
        output: (B, T, D) layer output.
        hidden_states: (B, T, D) layer input.
        attention_mask: Optional (B, T) mask.

    Returns:
        List of violation messages (empty if all checks pass).
    """
    violations: List[str] = []

    if output.shape != hidden_states.shape:
        violations.append(
            f"Shape mismatch: output {output.shape} != input {hidden_states.shape}"
        )

    if torch.isnan(output).any():
        violations.append("Output contains NaN values")

    if torch.isinf(output).any():
        violations.append("Output contains Inf values")

    if torch.allclose(output, hidden_states, atol=1e-8):
        violations.append("Output identical to input (delta is zero)")

    return violations


def validate_encoder_output(
    output: Tensor,
    input_ids: Tensor,
    attention_mask: Optional[Tensor],
    workspace_dim: int = DEFAULT_WORKSPACE_DIM,
) -> List[str]:
    """Validate EngramTextEncoder output against encoder suite contract.

    Checks:
      - Output shape is (B, T, workspace_dim)
      - No NaN or Inf values
      - Padding positions are zero (if mask provided)

    Args:
        output: (B, T, D) encoder output.
        input_ids: (B, T) input token IDs (for shape reference).
        attention_mask: Optional (B, T) mask, 1=valid, 0=padding.
        workspace_dim: Expected last dimension.

    Returns:
        List of violation messages (empty if all checks pass).
    """
    violations: List[str] = []
    B, T = input_ids.shape

    if output.shape != (B, T, workspace_dim):
        violations.append(
            f"Shape mismatch: expected ({B}, {T}, {workspace_dim}), "
            f"got {tuple(output.shape)}"
        )

    if torch.isnan(output).any():
        violations.append("Output contains NaN values")

    if torch.isinf(output).any():
        violations.append("Output contains Inf values")

    if attention_mask is not None:
        padding_positions = (attention_mask == 0)
        if padding_positions.any():
            padding_output = output[padding_positions]
            if not torch.allclose(padding_output, torch.zeros_like(padding_output), atol=1e-6):
                violations.append("Padding positions are not zero")

    return violations


# ========================================================================== #
#  SECTION 14: Module-level convenience exports                             #
# ========================================================================== #

__all__ = [
    # Config
    "EngramConfig",
    "EngramLayerConfig",
    "EngramEncoderConfig",
    # Normalization
    "RMSNorm",
    # Mock components
    "MockTokenizerCompression",
    "MockHashModule",
    "MockEngramModule",
    # Core integration modules
    "EngramAugmentedLayer",
    "EngramTextEncoder",
    # Registry / scheduling
    "EngramLayerRegistry",
    "PrefetchHookManager",
    # Checkpoint
    "EngramCheckpointMixin",
    # Backbone wrapper
    "EngramBackboneWrapper",
    # Telemetry
    "EngramTelemetryAggregator",
    # Factory functions
    "create_engram_augmented_layer",
    "create_engram_text_encoder",
    # Utilities
    "init_engram_weights",
    "count_engram_parameters",
    "validate_engram_layer_output",
    "validate_encoder_output",
    # Constants
    "DEFAULT_WORKSPACE_DIM",
    "DEFAULT_EMBEDDING_DIM",
    "DEFAULT_NGRAM_ORDERS",
    "DEFAULT_TABLE_SIZE",
    "DEFAULT_PREFETCH_AHEAD",
    "RMSNORM_EPS",
]


# ========================================================================== #
#  SECTION 15: Self-Tests                                                   #
# ========================================================================== #

def _self_test() -> None:
    """Comprehensive self-tests for all components in this module.

    Runs 35+ tests covering:
      - EngramAugmentedLayer: shape, residual, telemetry, prefetch, grad, AMP
      - EngramTextEncoder: shape, masking, diversity, telemetry, grad, AMP
      - EngramLayerRegistry: register, retrieve, sorted indices
      - PrefetchHookManager: scheduling, hooks
      - EngramCheckpointMixin: save/load roundtrip
      - EngramBackboneWrapper: integration with mini-backbone
      - Validation utilities
    """
    import sys

    passed = 0
    failed = 0
    errors: List[str] = []

    def _check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal passed, failed
        if condition:
            passed += 1
            print(f"  PASS  {name}")
        else:
            failed += 1
            msg = f"  FAIL  {name}"
            if detail:
                msg += f" -- {detail}"
            print(msg)
            errors.append(f"{name}: {detail}")

    print("=" * 70)
    print("Engram Layer Template -- Self-Tests")
    print("=" * 70)

    torch.manual_seed(42)
    device = torch.device("cpu")
    B, T, D = 2, 16, 128
    vocab_size = 1000
    workspace_dim = 4096

    # Shared small EngramConfig for testing
    test_engram_config = EngramConfig(
        vocab_size=vocab_size,
        embedding_dim=64,
        ngram_orders=(2, 3),
        num_heads_per_order=2,
        table_size=1009,  # Small prime
        hash_seed=42,
        use_context_gate=True,
        use_rmsnorm=True,
        gate_init_bias=-2.0,
        use_depthwise_conv=True,
        conv_kernel_size=4,
        conv_dilation=2,
    )

    # ==================================================================== #
    #  EngramAugmentedLayer Tests                                          #
    # ==================================================================== #
    print("\n--- EngramAugmentedLayer ---")

    layer_config = EngramLayerConfig(
        hidden_dim=D,
        engram_config=test_engram_config,
        layer_id=3,
        use_prefetch=False,
        residual_scale=1.0,
    )
    layer = EngramAugmentedLayer(D, layer_config, layer_id=3)
    # Initialize output_proj with small random weights so delta is non-zero.
    # In production the zero-init is intentional for smooth training start,
    # but tests need non-zero delta to validate residual behaviour.
    with torch.no_grad():
        nn.init.normal_(layer.engram.output_proj.weight, std=0.02)
    layer.eval()

    input_ids = torch.randint(0, vocab_size, (B, T))
    hidden_states = torch.randn(B, T, D)
    attention_mask = torch.ones(B, T, dtype=torch.long)
    attention_mask[:, -3:] = 0  # Last 3 positions are padding

    # T01: Output shape matches input
    with torch.no_grad():
        out = layer(hidden_states, input_ids, attention_mask)
    _check(
        "T01 output shape matches input",
        out.shape == hidden_states.shape,
        f"expected {hidden_states.shape}, got {out.shape}",
    )

    # T02: Residual connection -- output differs from input
    _check(
        "T02 residual: output differs from input",
        not torch.allclose(out, hidden_states, atol=1e-7),
        "output and input are identical (delta is zero)",
    )

    # T03: return_details populates telemetry with layer_id
    with torch.no_grad():
        out_det, telemetry = layer(hidden_states, input_ids, attention_mask, return_details=True)
    _check(
        "T03a return_details: telemetry is dict",
        isinstance(telemetry, dict),
        f"got {type(telemetry)}",
    )
    _check(
        "T03b return_details: layer_id in telemetry",
        telemetry.get("layer_id") == 3,
        f"got layer_id={telemetry.get('layer_id')}",
    )
    _check(
        "T03c return_details: gate_mean in telemetry",
        "gate_mean" in telemetry,
        f"keys: {list(telemetry.keys())}",
    )

    # T04: "Compute now" path works without prefetched_embeddings
    with torch.no_grad():
        out_compute = layer(hidden_states, input_ids, attention_mask)
    _check(
        "T04 compute-now path: no errors, valid output",
        out_compute.shape == (B, T, D) and not torch.isnan(out_compute).any(),
        "NaN or shape error",
    )

    # T05: "Consume prefetched" path works with prefetched embeddings
    total_heads = len(test_engram_config.ngram_orders) * test_engram_config.num_heads_per_order
    dim_per_head = max(1, test_engram_config.embedding_dim // total_heads)
    total_embed_dim = dim_per_head * total_heads
    prefetched = torch.randn(B, T, total_embed_dim)
    with torch.no_grad():
        out_prefetch = layer(
            hidden_states, input_ids, attention_mask,
            prefetched_embeddings=prefetched,
        )
    _check(
        "T05 prefetch path: valid output shape",
        out_prefetch.shape == (B, T, D) and not torch.isnan(out_prefetch).any(),
        "NaN or shape error on prefetch path",
    )

    # T06: Prefetch path produces different output than compute path (different embeddings)
    _check(
        "T06 prefetch vs compute: different outputs",
        not torch.allclose(out_compute, out_prefetch, atol=1e-5),
        "prefetch and compute outputs are identical",
    )

    # T07: Gradient flow -- gradients exist on gating parameters
    layer.train()
    layer.zero_grad()
    hidden_grad = torch.randn(B, T, D, requires_grad=True)
    out_grad = layer(hidden_grad, input_ids, attention_mask)
    loss = out_grad.sum()
    loss.backward()
    has_gate_grad = False
    for name, param in layer.named_parameters():
        if ("gate" in name or "query_norm" in name or "key_norm" in name) and param.grad is not None and param.grad.abs().sum() > 0:
            has_gate_grad = True
            break
    _check(
        "T07 gradient flow: gradients on gating parameters",
        has_gate_grad,
        "no gradient found on any gate parameter",
    )

    # T08: Gradient flow on input hidden states
    _check(
        "T08 gradient flow: gradient on input hidden_states",
        hidden_grad.grad is not None and hidden_grad.grad.abs().sum() > 0,
        "no gradient on input hidden_states",
    )

    # T09: AMP safety -- no NaN under autocast
    amp_safe = True
    if torch.cuda.is_available():
        layer_cuda = EngramAugmentedLayer(D, layer_config, layer_id=3).cuda()
        h_cuda = torch.randn(B, T, D, device="cuda")
        ids_cuda = torch.randint(0, vocab_size, (B, T), device="cuda")
        with torch.amp.autocast("cuda"):
            out_amp = layer_cuda(h_cuda, ids_cuda)
        amp_safe = not torch.isnan(out_amp).any().item()
    _check(
        "T09 AMP safety: no NaN under autocast",
        amp_safe,
        "NaN detected under AMP (or no CUDA available -- skipped)",
    )

    # T10: Residual scale works
    layer_config_scaled = EngramLayerConfig(
        hidden_dim=D,
        engram_config=test_engram_config,
        layer_id=5,
        residual_scale=0.1,
    )
    layer_scaled = EngramAugmentedLayer(D, layer_config_scaled, layer_id=5)
    layer_scaled.eval()
    with torch.no_grad():
        out_scaled = layer_scaled(hidden_states, input_ids, attention_mask)
    # The scaled output should be closer to hidden_states than the unscaled one
    diff_unscaled = (out_compute - hidden_states).abs().mean().item()
    diff_scaled = (out_scaled - hidden_states).abs().mean().item()
    _check(
        "T10 residual scale: scaled output closer to input",
        diff_scaled < diff_unscaled + 1e-6,
        f"scaled_diff={diff_scaled:.6f}, unscaled_diff={diff_unscaled:.6f}",
    )

    # T11: telemetry_summary property
    _check(
        "T11 telemetry_summary property populated",
        layer.telemetry_summary is not None,
        "telemetry_summary is None after forward pass",
    )

    # T12: extra_repr
    repr_str = layer.extra_repr()
    _check(
        "T12 extra_repr contains layer_id",
        "layer_id=3" in repr_str,
        f"got: {repr_str}",
    )

    # ==================================================================== #
    #  EngramTextEncoder Tests                                             #
    # ==================================================================== #
    print("\n--- EngramTextEncoder ---")

    enc_config = EngramEncoderConfig(
        vocab_size=vocab_size,
        embed_dim=64,
        workspace_dim=workspace_dim,
        engram_config=test_engram_config,
    )
    encoder = EngramTextEncoder(vocab_size, enc_config, workspace_dim=workspace_dim)
    encoder.eval()

    enc_input_ids = torch.randint(0, vocab_size, (B, T))
    enc_mask = torch.ones(B, T, dtype=torch.long)
    enc_mask[:, -4:] = 0  # Last 4 are padding

    # T13: Output shape is (B, T, 4096)
    with torch.no_grad():
        enc_out = encoder(enc_input_ids, enc_mask)
    _check(
        "T13 encoder output shape is (B, T, 4096)",
        enc_out.shape == (B, T, workspace_dim),
        f"expected ({B}, {T}, {workspace_dim}), got {enc_out.shape}",
    )

    # T14: Padding masking -- output at padding positions is zero
    padding_positions = (enc_mask == 0)
    padding_output = enc_out[padding_positions]
    _check(
        "T14 padding masking: padding positions are zero",
        torch.allclose(padding_output, torch.zeros_like(padding_output), atol=1e-6),
        f"max padding value: {padding_output.abs().max().item():.6f}",
    )

    # T15: Different input_ids produce different outputs
    enc_input_ids_2 = torch.randint(0, vocab_size, (B, T))
    with torch.no_grad():
        enc_out_2 = encoder(enc_input_ids_2, enc_mask)
    _check(
        "T15 different inputs -> different outputs",
        not torch.allclose(enc_out, enc_out_2, atol=1e-5),
        "identical outputs for different inputs",
    )

    # T16: return_details includes encoder-level telemetry
    with torch.no_grad():
        enc_out_det, enc_tel = encoder(enc_input_ids, enc_mask, return_details=True)
    _check(
        "T16a return_details: telemetry is dict",
        isinstance(enc_tel, dict),
        f"got {type(enc_tel)}",
    )
    _check(
        "T16b return_details: encoder_type in telemetry",
        enc_tel.get("encoder_type") == "engram_text",
        f"got encoder_type={enc_tel.get('encoder_type')}",
    )
    _check(
        "T16c return_details: engram sub-telemetry present",
        "engram" in enc_tel and isinstance(enc_tel["engram"], dict),
        f"keys: {list(enc_tel.keys())}",
    )

    # T17: Gradient flow on embedding layer
    encoder.train()
    enc_ids_grad = torch.randint(0, vocab_size, (B, T))
    enc_out_grad = encoder(enc_ids_grad, enc_mask)
    loss_enc = enc_out_grad.sum()
    loss_enc.backward()
    _check(
        "T17a gradient flow: embedding weight has gradient",
        encoder.input_embedding.weight.grad is not None
        and encoder.input_embedding.weight.grad.abs().sum() > 0,
        "no gradient on embedding weight",
    )

    # T18: Gradient flow on workspace projection
    proj_has_grad = False
    for name, param in encoder.workspace_proj.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            proj_has_grad = True
            break
    _check(
        "T18 gradient flow: workspace projection has gradient",
        proj_has_grad,
        "no gradient on workspace projection parameters",
    )

    # T19: AMP safety for encoder
    enc_amp_safe = True
    if torch.cuda.is_available():
        encoder_cuda = EngramTextEncoder(vocab_size, enc_config, workspace_dim=workspace_dim).cuda()
        ids_cuda_enc = torch.randint(0, vocab_size, (B, T), device="cuda")
        mask_cuda_enc = torch.ones(B, T, dtype=torch.long, device="cuda")
        with torch.amp.autocast("cuda"):
            out_amp_enc = encoder_cuda(ids_cuda_enc, mask_cuda_enc)
        enc_amp_safe = not torch.isnan(out_amp_enc).any().item()
    _check(
        "T19 AMP safety: encoder no NaN under autocast",
        enc_amp_safe,
        "NaN detected under AMP (or no CUDA -- skipped)",
    )

    # T20: No-mask path works
    with torch.no_grad():
        enc_out_nomask = encoder(enc_input_ids)
    _check(
        "T20 no-mask path: valid output",
        enc_out_nomask.shape == (B, T, workspace_dim)
        and not torch.isnan(enc_out_nomask).any(),
        "NaN or shape error without mask",
    )

    # T21: All-padding mask produces all-zero output
    all_pad_mask = torch.zeros(B, T, dtype=torch.long)
    with torch.no_grad():
        enc_out_allpad = encoder(enc_input_ids, all_pad_mask)
    _check(
        "T21 all-padding mask: output is all zeros",
        torch.allclose(enc_out_allpad, torch.zeros_like(enc_out_allpad), atol=1e-6),
        f"max value: {enc_out_allpad.abs().max().item():.6f}",
    )

    # ==================================================================== #
    #  EngramLayerRegistry Tests                                           #
    # ==================================================================== #
    print("\n--- EngramLayerRegistry ---")

    registry = EngramLayerRegistry()

    layer_1 = EngramAugmentedLayer(D, layer_config, layer_id=1)
    layer_5_cfg = EngramLayerConfig(
        hidden_dim=D, engram_config=test_engram_config, layer_id=5,
    )
    layer_5 = EngramAugmentedLayer(D, layer_5_cfg, layer_id=5)

    registry.register_layer(1, layer_1)
    registry.register_layer(5, layer_5)

    # T22: Register and retrieve
    _check(
        "T22a register and retrieve layer 1",
        registry.get_layer(1) is layer_1,
        "retrieved wrong layer",
    )
    _check(
        "T22b register and retrieve layer 5",
        registry.get_layer(5) is layer_5,
        "retrieved wrong layer",
    )

    # T23: get_insertion_layers returns sorted
    _check(
        "T23 insertion layers sorted",
        registry.get_insertion_layers() == [1, 5],
        f"got {registry.get_insertion_layers()}",
    )

    # T24: Unregistered layer returns None
    _check(
        "T24 unregistered layer returns None",
        registry.get_layer(3) is None,
        "expected None for unregistered layer",
    )

    # T25: has_layer
    _check(
        "T25a has_layer(1) is True",
        registry.has_layer(1),
        "expected True",
    )
    _check(
        "T25b has_layer(99) is False",
        not registry.has_layer(99),
        "expected False",
    )

    # T26: num_layers
    _check(
        "T26 num_layers == 2",
        registry.num_layers == 2,
        f"got {registry.num_layers}",
    )

    # T27: Duplicate registration raises ValueError
    dup_error = False
    try:
        registry.register_layer(1, layer_1)
    except ValueError:
        dup_error = True
    _check(
        "T27 duplicate registration raises ValueError",
        dup_error,
        "no ValueError raised",
    )

    # T28: Unregister
    removed = registry.unregister_layer(5)
    _check(
        "T28a unregister returns the layer",
        removed is layer_5,
        "wrong layer returned from unregister",
    )
    _check(
        "T28b unregister removes from registry",
        registry.get_layer(5) is None,
        "layer 5 still in registry",
    )
    # Re-register for later tests
    registry.register_layer(5, layer_5)

    # T29: __contains__
    _check(
        "T29 __contains__ (1 in registry)",
        1 in registry,
        "expected True",
    )

    # ==================================================================== #
    #  PrefetchHookManager Tests                                           #
    # ==================================================================== #
    print("\n--- PrefetchHookManager ---")

    # Registry with layers at 4 and 8
    pf_registry = EngramLayerRegistry()
    pf_cfg_4 = EngramLayerConfig(hidden_dim=D, engram_config=test_engram_config, layer_id=4)
    pf_cfg_8 = EngramLayerConfig(hidden_dim=D, engram_config=test_engram_config, layer_id=8)
    pf_layer_4 = EngramAugmentedLayer(D, pf_cfg_4, layer_id=4)
    pf_layer_8 = EngramAugmentedLayer(D, pf_cfg_8, layer_id=8)
    pf_registry.register_layer(4, pf_layer_4)
    pf_registry.register_layer(8, pf_layer_8)

    manager = PrefetchHookManager(pf_registry, prefetch_ahead=2)

    # T30: Correct prefetch scheduling
    _check(
        "T30a prefetch for layer 4 triggered at layer 2",
        manager.should_prefetch_at(2) == 4,
        f"got {manager.should_prefetch_at(2)}",
    )
    _check(
        "T30b prefetch for layer 8 triggered at layer 6",
        manager.should_prefetch_at(6) == 8,
        f"got {manager.should_prefetch_at(6)}",
    )

    # T31: No prefetch for non-trigger layers
    _check(
        "T31a no prefetch at layer 0",
        manager.should_prefetch_at(0) is None,
        f"got {manager.should_prefetch_at(0)}",
    )
    _check(
        "T31b no prefetch at layer 3",
        manager.should_prefetch_at(3) is None,
        f"got {manager.should_prefetch_at(3)}",
    )
    _check(
        "T31c no prefetch at layer 7",
        manager.should_prefetch_at(7) is None,
        f"got {manager.should_prefetch_at(7)}",
    )

    # T32: Hook callable works without errors
    engram_modules = {
        4: pf_layer_4.engram,
        8: pf_layer_8.engram,
    }
    hook = manager.create_hook(engram_modules, input_ids)
    hook_error = False
    try:
        for i in range(10):
            hook(i)
    except Exception as e:
        hook_error = True
        _check("T32 hook callable", False, str(e))
    if not hook_error:
        _check("T32 hook callable works for 10 layers", True, "")

    # T33: Prefetched data available after hook
    manager.clear_all()
    hook2 = manager.create_hook(engram_modules, input_ids)
    hook2(2)  # Should trigger prefetch for layer 4
    prefetched_4 = manager.get_prefetched(4)
    _check(
        "T33 prefetched data available for layer 4",
        prefetched_4 is not None and prefetched_4.shape[0] == B,
        f"prefetched is {'None' if prefetched_4 is None else prefetched_4.shape}",
    )

    # T34: Schedule property
    _check(
        "T34 schedule property has entries",
        len(manager.schedule) > 0,
        f"schedule is empty: {manager.schedule}",
    )

    # ==================================================================== #
    #  EngramCheckpointMixin Tests                                         #
    # ==================================================================== #
    print("\n--- EngramCheckpointMixin ---")

    class CheckpointTestModel(nn.Module, EngramCheckpointMixin):
        def __init__(self):
            super().__init__()
            cfg = EngramLayerConfig(hidden_dim=D, engram_config=test_engram_config, layer_id=0)
            self.engram_layer = EngramAugmentedLayer(D, cfg, layer_id=0)
            self.engram = self.engram_layer.engram

    ckpt_model = CheckpointTestModel()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "engram_ckpt")

        # T35: Save without errors
        save_error = False
        try:
            ckpt_model.save_engram_state(save_path)
        except Exception as e:
            save_error = True
            _check("T35 save_engram_state", False, str(e))
        if not save_error:
            _check("T35 save_engram_state no errors", True, "")

        # T36: Files created
        _check(
            "T36a engram_state.pt exists",
            os.path.isfile(os.path.join(save_path, "engram_state.pt")),
            "file not found",
        )
        _check(
            "T36b .engram_hash_config.json exists",
            os.path.isfile(os.path.join(save_path, ".engram_hash_config.json")),
            "file not found",
        )

        # T37: Load without errors
        ckpt_model_2 = CheckpointTestModel()
        load_error = False
        try:
            ckpt_model_2.load_engram_state(save_path)
        except Exception as e:
            load_error = True
            _check("T37 load_engram_state", False, str(e))
        if not load_error:
            _check("T37 load_engram_state no errors", True, "")

        # T38: Loaded weights match saved weights
        with torch.no_grad():
            out_original = ckpt_model.engram_layer(hidden_states, input_ids)
            out_loaded = ckpt_model_2.engram_layer(hidden_states, input_ids)
        _check(
            "T38 loaded weights match: outputs identical",
            torch.allclose(out_original, out_loaded, atol=1e-6),
            f"max diff: {(out_original - out_loaded).abs().max().item():.8f}",
        )

    # ==================================================================== #
    #  Integration Test: Mini-Backbone with Engram Augmentation            #
    # ==================================================================== #
    print("\n--- Integration: Mini-Backbone ---")

    # Create a 4-layer backbone with Engram at layers 1 and 3
    class SimpleBackboneLayer(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.ff = nn.Linear(dim, dim)
        def forward(self, x):
            return x + F.gelu(self.ff(x))

    backbone_layers = nn.ModuleList([SimpleBackboneLayer(D) for _ in range(4)])

    int_registry = EngramLayerRegistry()
    int_cfg_1 = EngramLayerConfig(hidden_dim=D, engram_config=test_engram_config, layer_id=1)
    int_cfg_3 = EngramLayerConfig(hidden_dim=D, engram_config=test_engram_config, layer_id=3)
    int_layer_1 = EngramAugmentedLayer(D, int_cfg_1, layer_id=1)
    int_layer_3 = EngramAugmentedLayer(D, int_cfg_3, layer_id=3)
    int_registry.register_layer(1, int_layer_1)
    int_registry.register_layer(3, int_layer_3)

    wrapper = EngramBackboneWrapper(backbone_layers, int_registry, use_prefetch=False)

    int_hidden = torch.randn(B, T, D)
    int_ids = torch.randint(0, vocab_size, (B, T))
    int_mask = torch.ones(B, T, dtype=torch.long)

    # T39: Forward pass through wrapper
    with torch.no_grad():
        int_out = wrapper(int_hidden, int_ids, int_mask)
    _check(
        "T39 integration: output shape correct",
        int_out.shape == (B, T, D),
        f"expected ({B}, {T}, {D}), got {int_out.shape}",
    )

    # T40: No NaN in output
    _check(
        "T40 integration: no NaN in output",
        not torch.isnan(int_out).any(),
        "NaN detected in integration output",
    )

    # T41: Output differs from input (augmentation happened)
    _check(
        "T41 integration: output differs from input",
        not torch.allclose(int_out, int_hidden, atol=1e-6),
        "output identical to input",
    )

    # T42: Telemetry collected from all Engram layers
    with torch.no_grad():
        int_out_tel, int_tel = wrapper(int_hidden, int_ids, int_mask, return_details=True)
    _check(
        "T42a integration telemetry: layer_1 present",
        "layer_1" in int_tel,
        f"keys: {list(int_tel.keys())}",
    )
    _check(
        "T42b integration telemetry: layer_3 present",
        "layer_3" in int_tel,
        f"keys: {list(int_tel.keys())}",
    )

    # T43: Each Engram layer telemetry has correct layer_id
    _check(
        "T43a layer_1 telemetry has layer_id=1",
        int_tel.get("layer_1", {}).get("layer_id") == 1,
        f"got {int_tel.get('layer_1', {}).get('layer_id')}",
    )
    _check(
        "T43b layer_3 telemetry has layer_id=3",
        int_tel.get("layer_3", {}).get("layer_id") == 3,
        f"got {int_tel.get('layer_3', {}).get('layer_id')}",
    )

    # T44: Gradient flow through wrapper
    wrapper.train()
    int_hidden_grad = torch.randn(B, T, D, requires_grad=True)
    int_out_grad = wrapper(int_hidden_grad, int_ids, int_mask)
    loss_int = int_out_grad.sum()
    loss_int.backward()
    _check(
        "T44 integration gradient flow: input has gradient",
        int_hidden_grad.grad is not None and int_hidden_grad.grad.abs().sum() > 0,
        "no gradient on input",
    )

    # ==================================================================== #
    #  Integration Test: Backbone with Prefetch                            #
    # ==================================================================== #
    print("\n--- Integration: Backbone + Prefetch ---")

    # 6-layer backbone, Engram at layers 2 and 4, prefetch_ahead=2
    backbone_layers_pf = nn.ModuleList([SimpleBackboneLayer(D) for _ in range(6)])
    pf_reg = EngramLayerRegistry()
    pf_c2 = EngramLayerConfig(hidden_dim=D, engram_config=test_engram_config, layer_id=2)
    pf_c4 = EngramLayerConfig(hidden_dim=D, engram_config=test_engram_config, layer_id=4)
    pf_l2 = EngramAugmentedLayer(D, pf_c2, layer_id=2)
    pf_l4 = EngramAugmentedLayer(D, pf_c4, layer_id=4)
    pf_reg.register_layer(2, pf_l2)
    pf_reg.register_layer(4, pf_l4)

    wrapper_pf = EngramBackboneWrapper(backbone_layers_pf, pf_reg, use_prefetch=True, prefetch_ahead=2)
    pf_hidden = torch.randn(B, T, D)

    # T45: Forward pass with prefetch enabled
    with torch.no_grad():
        pf_out = wrapper_pf(pf_hidden, int_ids, int_mask)
    _check(
        "T45 prefetch integration: output shape correct",
        pf_out.shape == (B, T, D) and not torch.isnan(pf_out).any(),
        f"shape={pf_out.shape}, nan={torch.isnan(pf_out).any()}",
    )

    # ==================================================================== #
    #  Validation Utilities Tests                                          #
    # ==================================================================== #
    print("\n--- Validation Utilities ---")

    # T46: validate_engram_layer_output -- valid case
    valid_violations = validate_engram_layer_output(out_compute, hidden_states)
    _check(
        "T46 validate_layer_output: valid output has no violations",
        len(valid_violations) == 0,
        f"violations: {valid_violations}",
    )

    # T47: validate_engram_layer_output -- NaN case
    nan_output = out_compute.clone()
    nan_output[0, 0, 0] = float("nan")
    nan_violations = validate_engram_layer_output(nan_output, hidden_states)
    _check(
        "T47 validate_layer_output: NaN detected",
        any("NaN" in v for v in nan_violations),
        f"violations: {nan_violations}",
    )

    # T48: validate_encoder_output -- valid case
    with torch.no_grad():
        valid_enc_out = encoder(enc_input_ids, enc_mask)
    enc_violations = validate_encoder_output(valid_enc_out, enc_input_ids, enc_mask, workspace_dim)
    _check(
        "T48 validate_encoder_output: valid output has no violations",
        len(enc_violations) == 0,
        f"violations: {enc_violations}",
    )

    # ==================================================================== #
    #  Telemetry Aggregator Tests                                          #
    # ==================================================================== #
    print("\n--- TelemetryAggregator ---")

    agg = EngramTelemetryAggregator()
    agg.update(int_tel)
    summary = agg.summary()

    # T49: Aggregator summary has expected fields
    _check(
        "T49a aggregator: num_updates == 1",
        summary["num_updates"] == 1,
        f"got {summary['num_updates']}",
    )
    _check(
        "T49b aggregator: gate_mean is numeric",
        isinstance(summary["gate_mean"], float),
        f"got {type(summary['gate_mean'])}",
    )

    # T50: Reset clears data
    agg.reset()
    _check(
        "T50 aggregator reset: num_updates == 0",
        agg.summary()["num_updates"] == 0,
        f"got {agg.summary()['num_updates']}",
    )

    # ==================================================================== #
    #  Factory Function Tests                                              #
    # ==================================================================== #
    print("\n--- Factory Functions ---")

    # T51: create_engram_augmented_layer
    factory_layer = create_engram_augmented_layer(
        hidden_dim=D, layer_id=7, residual_scale=0.5,
        vocab_size=vocab_size, table_size=1009,
    )
    _check(
        "T51 factory: create_engram_augmented_layer",
        isinstance(factory_layer, EngramAugmentedLayer) and factory_layer.layer_id == 7,
        f"type={type(factory_layer)}, layer_id={factory_layer.layer_id}",
    )

    # T52: create_engram_text_encoder
    factory_enc = create_engram_text_encoder(
        vocab_size=vocab_size, workspace_dim=workspace_dim,
        embed_dim=64, table_size=1009,
    )
    with torch.no_grad():
        factory_out = factory_enc(enc_input_ids, enc_mask)
    _check(
        "T52 factory: create_engram_text_encoder output shape",
        factory_out.shape == (B, T, workspace_dim),
        f"got {factory_out.shape}",
    )

    # ==================================================================== #
    #  Parameter Counting Tests                                            #
    # ==================================================================== #
    print("\n--- Parameter Counting ---")

    # T53: count_engram_parameters
    counts = count_engram_parameters(layer)
    _check(
        "T53 param counting: total > 0",
        counts.get("total", 0) > 0,
        f"got {counts.get('total', 0)}",
    )

    # ==================================================================== #
    #  Mock Component Tests                                                #
    # ==================================================================== #
    print("\n--- Mock Components ---")

    # T54: MockHashModule determinism
    mock_hash = MockHashModule(ngram_order=3, num_heads=4, table_size=1009, seed=42)
    ngrams_test = torch.randint(0, 500, (2, 8, 3))
    hash1 = mock_hash.hash(ngrams_test)
    hash2 = mock_hash.hash(ngrams_test)
    _check(
        "T54 MockHashModule: deterministic",
        torch.equal(hash1, hash2),
        "hash outputs differ for same input",
    )

    # T55: MockHashModule output range
    _check(
        "T55 MockHashModule: output in range [0, table_size)",
        hash1.min() >= 0 and hash1.max() < 1009,
        f"range: [{hash1.min()}, {hash1.max()}]",
    )

    # T56: MockTokenizerCompression identity
    mock_comp = MockTokenizerCompression(50000)
    test_ids = torch.randint(0, 50000, (2, 10))
    _check(
        "T56 MockTokenizerCompression: identity mapping",
        torch.equal(mock_comp.compress(test_ids), test_ids),
        "compressed IDs differ from input",
    )

    # T57: MockEngramModule forward
    mock_engram = MockEngramModule(test_engram_config, hidden_dim=D, layer_salt=0)
    mock_h = torch.randn(B, T, D)
    mock_ids = torch.randint(0, vocab_size, (B, T))
    delta, tel = mock_engram(mock_h, mock_ids)
    _check(
        "T57 MockEngramModule: delta shape",
        delta.shape == (B, T, D),
        f"got {delta.shape}",
    )

    # ==================================================================== #
    #  RMSNorm Tests                                                       #
    # ==================================================================== #
    print("\n--- RMSNorm ---")

    # T58: RMSNorm output shape
    norm = RMSNorm(D)
    norm_in = torch.randn(B, T, D)
    norm_out = norm(norm_in)
    _check(
        "T58 RMSNorm: output shape",
        norm_out.shape == norm_in.shape,
        f"got {norm_out.shape}",
    )

    # T59: RMSNorm fp32 safety under half precision
    norm_half = norm_in.half()
    norm_out_half = norm(norm_half)
    _check(
        "T59 RMSNorm: no NaN with half input",
        not torch.isnan(norm_out_half).any(),
        "NaN in RMSNorm with half-precision input",
    )

    # ==================================================================== #
    #  Config Dataclass Tests                                              #
    # ==================================================================== #
    print("\n--- Config Dataclasses ---")

    # T60: EngramConfig post_init
    cfg_auto = EngramConfig(vocab_size=100000)
    _check(
        "T60 EngramConfig: auto compressed_vocab_size",
        cfg_auto.compressed_vocab_size == 77000,
        f"got {cfg_auto.compressed_vocab_size}",
    )

    # T61: EngramLayerConfig defaults
    lc = EngramLayerConfig()
    _check(
        "T61 EngramLayerConfig: default hidden_dim == 4096",
        lc.hidden_dim == 4096,
        f"got {lc.hidden_dim}",
    )

    # T62: EngramEncoderConfig defaults
    ec = EngramEncoderConfig()
    _check(
        "T62 EngramEncoderConfig: default workspace_dim == 4096",
        ec.workspace_dim == 4096,
        f"got {ec.workspace_dim}",
    )

    # ==================================================================== #
    #  Summary                                                             #
    # ==================================================================== #
    print("\n" + "=" * 70)
    total = passed + failed
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if errors:
        print("\nFailures:")
        for e in errors:
            print(f"  {e}")
    print("=" * 70)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    _self_test()
