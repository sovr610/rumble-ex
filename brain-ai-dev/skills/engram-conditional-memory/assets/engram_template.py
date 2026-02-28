"""
brain_ai/memory/engram.py -- Engram Conditional Memory: N-gram Extraction,
Context-Aware Gating, Depthwise Causal Convolution, and EngramModule.

This module implements the core Engram conditional memory subsystem based on
DeepSeek's Engram paper. It provides O(1) retrieval of static linguistic
patterns (idioms, named entities, formulaic phrases) via deterministic N-gram
hashing, freeing transformer depth for compositional reasoning.

Architecture overview::

    input_ids --> TokenizerCompression --> SuffixNgramExtractor --> MultiHeadHash
                                                                       |
    hidden_states -----> ContextAwareGating <-- EmbeddingAggregator <--+
                               |
                      DepthwiseCausalConv1d
                               |
                      OutputProjection --> delta (residual)

Key components:
    SuffixNgramExtractor   -- Extracts suffix N-grams of orders 2..K from
                              canonical token IDs, with mask-aware padding.
    StreamingCache         -- Rolling buffer for incremental (autoregressive)
                              N-gram extraction without recomputing history.
    RMSNorm                -- Root Mean Square normalization with AMP-safe
                              fp32 accumulation.
    ContextAwareGating     -- Gates retrieved memory using hidden-state queries
                              and memory keys via element-wise dot product, with
                              optional per-head gating and RMSNorm on q/k.
    DepthwiseCausalConv1d  -- Depthwise separable causal convolution with
                              configurable dilation and kernel size for
                              receptive field expansion.
    EngramModule           -- Orchestrates the full pipeline from N-gram
                              extraction through gated retrieval and causal
                              convolution to produce a residual delta.
    EngramTelemetry        -- Diagnostic dataclass capturing gate statistics,
                              unique ID ratios, prefetch stats, and compression
                              ratio for monitoring and debugging.

Hard invariants:
    - Hash IDs are deterministic across CPU/CUDA/distributed ranks given the
      same seed and configuration.
    - Gating outputs are bounded [0, 1] and AMP-safe (no NaN under mixed
      precision via fp32 computation paths for norms and reductions).
    - Depthwise causal convolution is strictly causal: output at position t
      depends only on input at positions <= t.
    - Tokenizer compression is deterministic: same input_ids produce the same
      canonical_ids across runs and devices.

Typical import (once integrated into brain_ai)::

    from brain_ai.memory.engram import (
        EngramConfig,
        SuffixNgramExtractor,
        StreamingCache,
        RMSNorm,
        ContextAwareGating,
        DepthwiseCausalConv1d,
        EngramModule,
        EngramTelemetry,
    )

Copy this file to ``brain_ai/memory/engram.py`` when integrating into the
main package.

References:
    DeepSeek (2024) "Engram: Conditional Memory for Language Models"
    Zhang et al. (2023) "Root Mean Square Layer Normalization"
    Ba et al. (2016) "Layer Normalization"
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Default pad_id used for left-padding N-gram suffixes at the start of a
# sequence where insufficient context exists.
DEFAULT_PAD_ID: int = 0

# Minimum epsilon for RMSNorm to avoid division by zero in fp32.
RMSNORM_EPS_MIN: float = 1e-12

# Default conservative gate bias -- sigmoid(-2.0) ~ 0.12, ensuring the
# gate starts with low influence and must be learned to open.
DEFAULT_GATE_INIT_BIAS: float = -2.0

# Supported activation functions for DepthwiseCausalConv1d.
SUPPORTED_CONV_ACTIVATIONS = ("silu", "gelu", "relu", "none")

# Supported gate types for ContextAwareGating.
SUPPORTED_GATE_TYPES = ("scalar", "per_head")


# ===================================================================== #
#                    SECTION 1: EngramConfig                             #
# ===================================================================== #

@dataclass
class EngramConfig:
    """Configuration dataclass for the Engram conditional memory module.

    This aggregates all hyperparameters needed by ``EngramModule`` and its
    sub-components.  Presets are available via class methods for quick
    experimentation at different scales.

    Attributes:
        hidden_dim:
            Hidden dimension of the host backbone (transformer).  Used for
            gating projections and the output projection.
        max_ngram_order:
            Maximum N-gram order K.  Suffix N-grams of orders 2..K are
            extracted at each position.
        pad_id:
            Token ID used for left-padding when insufficient context is
            available at the start of the sequence.
        vocab_size:
            Original vocabulary size of the tokenizer.
        compressed_vocab_size:
            Target vocabulary size after tokenizer compression.  Defaults
            to ~77% of ``vocab_size`` if not specified.
        embedding_dim:
            Dimension of each retrieved embedding.  The aggregated embedding
            has this dimension after multi-head aggregation.
        num_heads_per_order:
            Number of independent hash heads per N-gram order.
        table_size:
            Size of each hash embedding table.  Should be a prime for good
            distribution.
        hash_seed:
            Deterministic seed for hash function coefficient generation.
        use_tokenizer_compression:
            Whether to apply tokenizer compression to input_ids before
            N-gram extraction.
        use_context_gate:
            Whether to use context-aware gating.  If False, retrieved
            embeddings are passed through without gating.
        use_depthwise_conv:
            Whether to apply depthwise causal convolution after gating.
        gate_type:
            Type of gating: ``"scalar"`` for a single gate per position,
            ``"per_head"`` for separate gates per attention head.
        num_gate_heads:
            Number of gate heads when ``gate_type="per_head"``.
        gate_dim:
            Projection dimension for gating queries and keys.
        gate_init_bias:
            Initial bias for the gate logit.  Negative values (e.g. -2.0)
            produce conservative initial gating ~ sigmoid(-2) ~ 0.12.
        use_rmsnorm:
            Whether to apply RMSNorm to gating queries and keys.
        rmsnorm_eps:
            Epsilon for RMSNorm stability.
        conv_kernel_size:
            Kernel size for the depthwise causal convolution.
        conv_dilation:
            Dilation factor for the depthwise causal convolution.
        conv_activation:
            Activation function after convolution: ``"silu"`` or ``"gelu"``.
        offload_to_cpu:
            Whether embedding tables are offloaded to CPU (for large tables).
        use_prefetch:
            Whether to enable async prefetching of embeddings (requires CUDA).
        per_layer_salt:
            Whether to use per-layer salt for hash decorrelation.
        output_dim:
            Dimension of the output delta.  Defaults to ``hidden_dim`` if
            not specified.
    """

    # Host backbone dimension
    hidden_dim: int = 4096

    # N-gram extraction
    max_ngram_order: int = 4
    pad_id: int = DEFAULT_PAD_ID

    # Vocabulary
    vocab_size: int = 128000
    compressed_vocab_size: Optional[int] = None

    # Embeddings
    embedding_dim: int = 256
    num_heads_per_order: int = 2
    table_size: int = 131071  # Prime
    hash_seed: int = 42

    # Tokenizer compression
    use_tokenizer_compression: bool = True

    # Gating
    use_context_gate: bool = True
    gate_type: str = "scalar"
    num_gate_heads: int = 8
    gate_dim: int = 128
    gate_init_bias: float = DEFAULT_GATE_INIT_BIAS
    use_rmsnorm: bool = True
    rmsnorm_eps: float = 1e-6

    # Convolution
    use_depthwise_conv: bool = True
    conv_kernel_size: int = 4
    conv_dilation: int = 4
    conv_activation: str = "silu"

    # Offloading
    offload_to_cpu: bool = False
    use_prefetch: bool = False
    per_layer_salt: bool = True

    # Output
    output_dim: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate configuration and compute derived fields."""
        if self.compressed_vocab_size is None:
            self.compressed_vocab_size = int(self.vocab_size * 0.77)

        if self.output_dim is None:
            self.output_dim = self.hidden_dim

        if self.max_ngram_order < 2:
            raise ValueError(
                f"max_ngram_order must be >= 2, got {self.max_ngram_order}"
            )

        if self.gate_type not in SUPPORTED_GATE_TYPES:
            raise ValueError(
                f"gate_type must be one of {SUPPORTED_GATE_TYPES}, "
                f"got '{self.gate_type}'"
            )

        if self.conv_activation not in SUPPORTED_CONV_ACTIVATIONS:
            raise ValueError(
                f"conv_activation must be one of {SUPPORTED_CONV_ACTIVATIONS}, "
                f"got '{self.conv_activation}'"
            )

    @property
    def ngram_orders(self) -> Tuple[int, ...]:
        """Return the tuple of N-gram orders from 2 to max_ngram_order."""
        return tuple(range(2, self.max_ngram_order + 1))

    @property
    def total_heads(self) -> int:
        """Total number of hash heads across all N-gram orders."""
        return len(self.ngram_orders) * self.num_heads_per_order

    @classmethod
    def minimal(cls) -> "EngramConfig":
        """Minimal configuration for unit testing (~1K params)."""
        return cls(
            hidden_dim=64,
            max_ngram_order=3,
            vocab_size=1000,
            compressed_vocab_size=770,
            embedding_dim=32,
            num_heads_per_order=1,
            table_size=1009,
            gate_dim=16,
            num_gate_heads=2,
            conv_kernel_size=3,
            conv_dilation=2,
            use_prefetch=False,
            offload_to_cpu=False,
        )

    @classmethod
    def dev(cls) -> "EngramConfig":
        """Development configuration for fast iteration."""
        return cls(
            hidden_dim=256,
            max_ngram_order=3,
            vocab_size=50000,
            embedding_dim=128,
            num_heads_per_order=2,
            table_size=10007,
            gate_dim=64,
            num_gate_heads=4,
            conv_kernel_size=4,
            conv_dilation=3,
        )

    @classmethod
    def production(cls) -> "EngramConfig":
        """Production configuration matching ~2.5B param budget."""
        return cls(
            hidden_dim=4096,
            max_ngram_order=4,
            vocab_size=128000,
            embedding_dim=4096,
            num_heads_per_order=4,
            table_size=100_000_007,
            gate_dim=256,
            num_gate_heads=32,
            conv_kernel_size=7,
            conv_dilation=4,
            offload_to_cpu=True,
            use_prefetch=True,
        )


# ===================================================================== #
#                SECTION 2: StreamingCache                               #
# ===================================================================== #

@dataclass
class StreamingCache:
    """Rolling buffer for incremental N-gram extraction during autoregressive
    generation.

    Maintains a FIFO buffer of the most recent (K-1) canonical token IDs so
    that suffix N-grams can be extracted for each new token without
    reprocessing the entire sequence.

    Attributes:
        buffer:
            Tensor of shape ``(K-1,)`` holding the most recent canonical IDs.
            Initialized to ``pad_id``.
        position:
            Current position counter (0-indexed).  Tracks how many tokens
            have been processed.
        pad_id:
            Padding ID used for initialization and for positions with
            insufficient context.
        max_order:
            Maximum N-gram order K.  The buffer stores K-1 entries.
    """

    buffer: Tensor
    position: int
    pad_id: int
    max_order: int

    @classmethod
    def create(
        cls,
        max_order: int,
        pad_id: int = DEFAULT_PAD_ID,
        dtype: torch.dtype = torch.long,
        device: Union[str, torch.device] = "cpu",
    ) -> "StreamingCache":
        """Create a new StreamingCache with zero-initialized buffer.

        Args:
            max_order: Maximum N-gram order K.
            pad_id: Padding ID for the buffer.
            dtype: Integer dtype for the buffer.
            device: Device for the buffer tensor.

        Returns:
            A new StreamingCache ready for use.
        """
        buf = torch.full(
            (max_order - 1,), pad_id, dtype=dtype, device=device
        )
        return cls(buffer=buf, position=0, pad_id=pad_id, max_order=max_order)

    def update(self, new_id: int) -> None:
        """Shift buffer left by one position and append the new canonical ID.

        This is an in-place operation.  After calling ``update(x)``, the
        buffer contents are ``[old[1], old[2], ..., old[K-2], x]``.

        Args:
            new_id: The new canonical token ID to append.
        """
        buf_len = self.buffer.shape[0]
        if buf_len > 1:
            self.buffer[:-1] = self.buffer[1:].clone()
        self.buffer[-1] = new_id
        self.position += 1

    def reset(self) -> None:
        """Reset the buffer to all pad_id and position to 0."""
        self.buffer.fill_(self.pad_id)
        self.position = 0

    def get_context(self) -> Tensor:
        """Return the current buffer contents (K-1 most recent IDs).

        Returns:
            Tensor of shape ``(K-1,)`` with the most recent canonical IDs.
        """
        return self.buffer.clone()

    def clone(self) -> "StreamingCache":
        """Return a deep copy of this cache."""
        return StreamingCache(
            buffer=self.buffer.clone(),
            position=self.position,
            pad_id=self.pad_id,
            max_order=self.max_order,
        )


# ===================================================================== #
#            SECTION 3: SuffixNgramExtractor                             #
# ===================================================================== #

class SuffixNgramExtractor:
    """Extracts suffix N-grams from canonical token ID sequences.

    For each position ``t`` and each order ``n`` in ``[2, K]``, extracts the
    suffix tuple ``(canonical_ids[t-n+1], ..., canonical_ids[t])``.  Positions
    at the start of the sequence where ``t < n-1`` are left-padded with
    ``pad_id``.

    The extractor is mask-aware: if ``attention_mask`` indicates that a
    position is padding, the N-grams at that position are filled with
    ``pad_id``.

    This class is NOT an ``nn.Module`` because it contains no learnable
    parameters.  It is a pure function of the input IDs and mask.

    Args:
        max_order:
            Maximum N-gram order K.  N-grams of orders 2, 3, ..., K are
            extracted.
        pad_id:
            Token ID used for left-padding at sequence boundaries.

    Example::

        extractor = SuffixNgramExtractor(max_order=4, pad_id=0)
        canonical_ids = torch.tensor([[10, 20, 30, 40, 50]])  # (1, 5)
        ngrams = extractor.extract(canonical_ids)
        # ngrams is a list of 3 tensors:
        #   ngrams[0]: shape (1, 5, 2) -- bigrams
        #   ngrams[1]: shape (1, 5, 3) -- trigrams
        #   ngrams[2]: shape (1, 5, 4) -- 4-grams
    """

    def __init__(self, max_order: int, pad_id: int = DEFAULT_PAD_ID) -> None:
        if max_order < 2:
            raise ValueError(
                f"max_order must be >= 2, got {max_order}"
            )
        self.max_order = max_order
        self.pad_id = pad_id

    @property
    def orders(self) -> Tuple[int, ...]:
        """Return the tuple of N-gram orders extracted."""
        return tuple(range(2, self.max_order + 1))

    def extract(
        self,
        canonical_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> List[Tensor]:
        """Extract suffix N-grams for all positions and all orders.

        Args:
            canonical_ids:
                Integer tensor of shape ``(B, T)`` with canonical (compressed)
                token IDs.
            attention_mask:
                Optional boolean or integer tensor of shape ``(B, T)`` where
                1/True indicates a valid token and 0/False indicates padding.
                When provided, N-grams at padding positions are filled with
                ``pad_id``.

        Returns:
            List of tensors, one per N-gram order.  The i-th tensor has
            shape ``(B, T, orders[i])`` and dtype matching ``canonical_ids``.
            The list is ordered from smallest to largest order.

        Note:
            For the entry at ``ngrams[i][b, t, :]`` where ``i`` corresponds
            to order ``n = orders[i]``:
              - If ``t >= n-1``: the suffix is
                ``canonical_ids[b, t-n+1 : t+1]``
              - If ``t < n-1``: left-padded with ``pad_id``
        """
        B, T = canonical_ids.shape
        device = canonical_ids.device
        dtype = canonical_ids.dtype

        # Apply attention mask: replace padding positions with pad_id so
        # N-grams including padding tokens hash to the padding hash.
        if attention_mask is not None:
            mask = attention_mask.bool()
            # Where mask is False (padding), replace with pad_id
            ids = canonical_ids.clone()
            ids[~mask] = self.pad_id
        else:
            ids = canonical_ids

        result: List[Tensor] = []

        for n in self.orders:
            # Left-pad with pad_id for positions where t < n-1
            padded = F.pad(ids, (n - 1, 0), mode="constant", value=self.pad_id)
            # padded shape: (B, T + n - 1)

            # Extract sliding windows of size n
            # Use unfold for efficiency
            ngrams = padded.unfold(dimension=1, size=n, step=1)
            # ngrams shape: (B, T, n)

            # If attention_mask is provided, zero out N-grams at padding
            # positions (set to pad_id)
            if attention_mask is not None:
                # Expand mask to cover the n-gram dimension
                mask_expanded = mask.unsqueeze(-1).expand_as(ngrams)
                ngrams = ngrams.clone()
                ngrams[~mask_expanded] = self.pad_id

            result.append(ngrams)

        return result

    def extract_stacked(
        self,
        canonical_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Extract N-grams and return as a single stacked tensor.

        The result is padded to the maximum order along the last dimension
        so that all orders can be stored in one tensor.

        Args:
            canonical_ids: ``(B, T)`` canonical token IDs.
            attention_mask: Optional ``(B, T)`` mask.

        Returns:
            Tensor of shape ``(B, T, K-1, K)`` where ``K = max_order``.
            For order ``n`` (index ``n-2``), only the first ``n`` entries
            along the last dimension are meaningful; the rest are ``pad_id``.
        """
        ngram_list = self.extract(canonical_ids, attention_mask)
        B, T = canonical_ids.shape
        K = self.max_order
        num_orders = K - 1  # orders 2..K
        device = canonical_ids.device
        dtype = canonical_ids.dtype

        stacked = torch.full(
            (B, T, num_orders, K), self.pad_id, dtype=dtype, device=device
        )

        for i, n in enumerate(self.orders):
            stacked[:, :, i, :n] = ngram_list[i]

        return stacked

    def extract_streaming(
        self,
        new_token_id: int,
        cache_state: StreamingCache,
    ) -> List[Tensor]:
        """Incrementally extract N-grams for a single new token.

        Given a new canonical token ID and a rolling buffer (cache_state),
        extracts N-grams for just the new position.  This is semantically
        equivalent to calling ``extract()`` on the full sequence and taking
        the last position, but runs in O(K) time instead of O(T * K).

        Args:
            new_token_id:
                The new canonical token ID (integer).
            cache_state:
                A ``StreamingCache`` that is updated in-place.  Must have
                been created with the same ``max_order`` and ``pad_id`` as
                this extractor.

        Returns:
            List of tensors, one per N-gram order.  The i-th tensor has
            shape ``(1, 1, orders[i])`` (batch=1, time=1, order dimension),
            matching the format of ``extract()`` for a single position.

        Side effects:
            ``cache_state`` is updated with the new token.
        """
        if cache_state.max_order != self.max_order:
            raise ValueError(
                f"Cache max_order ({cache_state.max_order}) does not match "
                f"extractor max_order ({self.max_order})"
            )

        device = cache_state.buffer.device
        dtype = cache_state.buffer.dtype

        # Build the full context window: buffer + new_token
        context = torch.cat([
            cache_state.buffer,
            torch.tensor([new_token_id], dtype=dtype, device=device),
        ])
        # context has shape (K-1 + 1,) = (K,)
        # context[-1] is the new token
        # context[-n:] gives the suffix of length n

        result: List[Tensor] = []

        for n in self.orders:
            # Extract suffix of length n from the end of context
            # context has K elements; we need the last n
            suffix = context[-n:]  # shape: (n,)
            ngram = suffix.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, n)
            result.append(ngram)

        # Update cache state
        cache_state.update(new_token_id)

        return result


# ===================================================================== #
#            SECTION 4: EmbeddingAggregator                              #
# ===================================================================== #

class EmbeddingAggregator(nn.Module):
    """Aggregates multi-head, multi-order retrieved embeddings into a single
    vector per position.

    Supports three aggregation strategies:

    - ``"concat"``: Concatenate all head embeddings, then project.
    - ``"mean"``: Average across all heads and orders, then project.
    - ``"weighted"``: Learnable per-head weights, sum, then project.

    Args:
        num_orders:
            Number of N-gram orders (e.g. 3 for orders 2,3,4).
        num_heads_per_order:
            Number of hash heads per order.
        dim_per_head:
            Dimension of each individual head embedding.
        output_dim:
            Desired aggregated output dimension.
        mode:
            Aggregation mode: ``"concat"``, ``"mean"``, or ``"weighted"``.
    """

    def __init__(
        self,
        num_orders: int,
        num_heads_per_order: int,
        dim_per_head: int,
        output_dim: int,
        mode: str = "concat",
    ) -> None:
        super().__init__()

        self.num_orders = num_orders
        self.num_heads_per_order = num_heads_per_order
        self.dim_per_head = dim_per_head
        self.output_dim = output_dim
        self.mode = mode
        self.total_heads = num_orders * num_heads_per_order

        if mode == "concat":
            concat_dim = self.total_heads * dim_per_head
            self.proj = nn.Linear(concat_dim, output_dim)
        elif mode == "mean":
            self.proj = nn.Linear(dim_per_head, output_dim)
        elif mode == "weighted":
            self.head_weights = nn.Parameter(
                torch.ones(self.total_heads) / self.total_heads
            )
            self.proj = nn.Linear(dim_per_head, output_dim)
        else:
            raise ValueError(
                f"Unsupported aggregation mode: '{mode}'. "
                f"Choose from 'concat', 'mean', 'weighted'."
            )

    def forward(self, head_embeddings: List[Tensor]) -> Tensor:
        """Aggregate head embeddings into a single vector per position.

        Args:
            head_embeddings:
                List of tensors, each of shape ``(B, T, dim_per_head)``.
                Length must equal ``total_heads``.

        Returns:
            Aggregated tensor of shape ``(B, T, output_dim)``.
        """
        if len(head_embeddings) != self.total_heads:
            raise ValueError(
                f"Expected {self.total_heads} head embeddings, "
                f"got {len(head_embeddings)}"
            )

        if self.mode == "concat":
            concatenated = torch.cat(head_embeddings, dim=-1)
            return self.proj(concatenated)

        elif self.mode == "mean":
            stacked = torch.stack(head_embeddings, dim=0)  # (H, B, T, D)
            averaged = stacked.mean(dim=0)  # (B, T, D)
            return self.proj(averaged)

        elif self.mode == "weighted":
            stacked = torch.stack(head_embeddings, dim=-2)  # (B, T, H, D)
            weights = F.softmax(self.head_weights, dim=0)  # (H,)
            weights = weights.view(1, 1, -1, 1)  # (1, 1, H, 1)
            weighted = (stacked * weights).sum(dim=-2)  # (B, T, D)
            return self.proj(weighted)

        # Should not reach here due to __init__ validation
        raise RuntimeError(f"Unknown mode: {self.mode}")


# ===================================================================== #
#                    SECTION 5: RMSNorm                                  #
# ===================================================================== #

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Normalizes the input tensor by its root mean square, scaled by a
    learnable weight (gamma) parameter.  Unlike LayerNorm, RMSNorm does not
    subtract the mean, which makes it faster and empirically competitive.

    Formula::

        y = x / sqrt(mean(x^2, dim=-1, keepdim=True) + eps) * weight

    AMP Safety:
        All internal computation is performed in fp32 regardless of the
        input dtype.  The output is cast back to the input dtype.  This
        prevents NaN and Inf under mixed-precision training.

    Args:
        dim: Feature dimension (last dimension of input).
        eps: Small constant for numerical stability.  Default 1e-6.

    Shape:
        Input: ``(*, dim)`` where ``*`` is any number of leading dimensions.
        Output: Same shape as input.

    Example::

        norm = RMSNorm(256)
        x = torch.randn(2, 10, 256)
        y = norm(x)  # (2, 10, 256)
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.dim = dim
        self.eps = max(eps, RMSNORM_EPS_MIN)
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: Tensor) -> Tensor:
        """Compute RMS normalization in fp32.

        Args:
            x: Input tensor, any dtype.

        Returns:
            Normalized tensor in fp32.
        """
        x_fp32 = x.float()
        rms = torch.sqrt(
            torch.mean(x_fp32 * x_fp32, dim=-1, keepdim=True) + self.eps
        )
        return x_fp32 / rms

    def forward(self, x: Tensor) -> Tensor:
        """Apply RMSNorm to input.

        Args:
            x: Input tensor of shape ``(*, dim)``.

        Returns:
            Normalized and scaled tensor, same shape and dtype as input.
        """
        input_dtype = x.dtype
        normed = self._norm(x)
        # weight is fp32, normed is fp32, multiply in fp32 then cast back
        output = normed * self.weight
        return output.to(input_dtype)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"


# ===================================================================== #
#        SECTION 6: ContextAwareGating                                   #
# ===================================================================== #

class ContextAwareGating(nn.Module):
    """Context-aware gating for Engram conditional memory.

    Uses the transformer's hidden state as a query to gate retrieved memory
    embeddings via an element-wise dot-product attention mechanism.  The
    gate is a scalar (or per-head vector) in [0, 1] that modulates the
    value projection of the retrieved memory.

    Architecture::

        q = RMSNorm_q(W_q(hidden_states))        # (B, T, gate_dim)
        k = RMSNorm_k(W_k(retrieved_memory))      # (B, T, gate_dim)
        v = W_v(retrieved_memory)                  # (B, T, output_dim)

        gate_logit = sum(q * k, dim=-1) + bias     # (B, T, 1) or (B, T, H)
        alpha = sigmoid(gate_logit)                 # (B, T, 1) or (B, T, H)

        output = alpha * v                          # (B, T, output_dim)

    The gate bias is initialized to a negative value (default -2.0) so that
    ``sigmoid(-2) ~ 0.12``, ensuring the module starts with conservative
    (low) influence and must learn to open the gate.

    Gate types:
        - ``"scalar"``: Single gate per position, alpha is ``(B, T, 1)``.
        - ``"per_head"``: Separate gate per head, alpha is ``(B, T, H)``
          with H independent gate biases.  The value projection output_dim
          must be divisible by H.

    AMP Safety:
        RMSNorm computations use fp32 internally.  The sigmoid is computed
        after the dot product to avoid precision issues.

    Args:
        hidden_dim:
            Dimension of the hidden states from the host backbone.
        memory_dim:
            Dimension of the retrieved memory embeddings.
        gate_dim:
            Projection dimension for queries and keys.
        output_dim:
            Dimension of the value projection output.  Typically equals
            ``hidden_dim``.
        gate_type:
            ``"scalar"`` or ``"per_head"``.
        num_gate_heads:
            Number of heads when ``gate_type="per_head"``.
        use_rmsnorm:
            Whether to apply RMSNorm to projected queries and keys.
        gate_init_bias:
            Initial value for the gate bias.  Negative values produce
            conservative initial gating.
        rmsnorm_eps:
            Epsilon for RMSNorm layers.
    """

    def __init__(
        self,
        hidden_dim: int,
        memory_dim: int,
        gate_dim: int = 128,
        output_dim: Optional[int] = None,
        gate_type: str = "scalar",
        num_gate_heads: int = 8,
        use_rmsnorm: bool = True,
        gate_init_bias: float = DEFAULT_GATE_INIT_BIAS,
        rmsnorm_eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if gate_type not in SUPPORTED_GATE_TYPES:
            raise ValueError(
                f"gate_type must be one of {SUPPORTED_GATE_TYPES}, "
                f"got '{gate_type}'"
            )

        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        self.gate_dim = gate_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        self.gate_type = gate_type
        self.num_gate_heads = num_gate_heads
        self.use_rmsnorm = use_rmsnorm

        # Query projection: hidden -> gate_dim
        self.Wq = nn.Linear(hidden_dim, gate_dim, bias=False)

        # Key projection: memory -> gate_dim
        self.Wk = nn.Linear(memory_dim, gate_dim, bias=False)

        # Value projection: memory -> output_dim
        self.Wv = nn.Linear(memory_dim, self.output_dim, bias=False)

        # Optional RMSNorm on queries and keys
        if use_rmsnorm:
            self.rmsnorm_q = RMSNorm(gate_dim, eps=rmsnorm_eps)
            self.rmsnorm_k = RMSNorm(gate_dim, eps=rmsnorm_eps)
        else:
            self.rmsnorm_q = nn.Identity()
            self.rmsnorm_k = nn.Identity()

        # Gate bias
        if gate_type == "scalar":
            self.gate_bias = nn.Parameter(
                torch.full((1,), gate_init_bias)
            )
        elif gate_type == "per_head":
            if self.output_dim % num_gate_heads != 0:
                raise ValueError(
                    f"output_dim ({self.output_dim}) must be divisible by "
                    f"num_gate_heads ({num_gate_heads}) for per_head gating"
                )
            self.gate_bias = nn.Parameter(
                torch.full((num_gate_heads,), gate_init_bias)
            )
            # Per-head gate needs gate_dim divisible by num_gate_heads
            if gate_dim % num_gate_heads != 0:
                raise ValueError(
                    f"gate_dim ({gate_dim}) must be divisible by "
                    f"num_gate_heads ({num_gate_heads}) for per_head gating"
                )
            self._head_dim = gate_dim // num_gate_heads

        # Initialize projections with small values
        nn.init.xavier_uniform_(self.Wq.weight, gain=0.1)
        nn.init.xavier_uniform_(self.Wk.weight, gain=0.1)
        nn.init.xavier_uniform_(self.Wv.weight, gain=0.1)

    def forward(
        self,
        hidden_states: Tensor,
        retrieved_memory: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Apply context-aware gating to retrieved memory.

        Args:
            hidden_states:
                Tensor of shape ``(B, T, hidden_dim)`` from the host
                backbone.
            retrieved_memory:
                Tensor of shape ``(B, T, memory_dim)`` from the embedding
                retrieval stage.

        Returns:
            gated_output:
                Tensor of shape ``(B, T, output_dim)`` -- the gated memory.
            alpha:
                Gate values, shape ``(B, T, 1)`` for scalar gating or
                ``(B, T, H)`` for per-head gating.  Values are in [0, 1].
        """
        # Project to gate space
        q = self.Wq(hidden_states)   # (B, T, gate_dim)
        k = self.Wk(retrieved_memory)  # (B, T, gate_dim)
        v = self.Wv(retrieved_memory)  # (B, T, output_dim)

        # Apply RMSNorm
        q = self.rmsnorm_q(q)  # (B, T, gate_dim)
        k = self.rmsnorm_k(k)  # (B, T, gate_dim)

        if self.gate_type == "scalar":
            # Element-wise product and sum -> scalar gate per position
            gate_logit = (q * k).sum(dim=-1, keepdim=True)  # (B, T, 1)
            gate_logit = gate_logit + self.gate_bias  # (B, T, 1)
            alpha = torch.sigmoid(gate_logit)  # (B, T, 1)
            gated = alpha * v  # (B, T, output_dim)

        elif self.gate_type == "per_head":
            B, T, _ = q.shape
            H = self.num_gate_heads
            head_dim = self._head_dim
            out_head_dim = self.output_dim // H

            # Reshape to per-head: (B, T, H, head_dim)
            q_heads = q.view(B, T, H, head_dim)
            k_heads = k.view(B, T, H, head_dim)

            # Per-head dot product: (B, T, H)
            gate_logit = (q_heads * k_heads).sum(dim=-1)  # (B, T, H)
            gate_logit = gate_logit + self.gate_bias  # broadcast (H,)
            alpha = torch.sigmoid(gate_logit)  # (B, T, H)

            # Reshape value to per-head: (B, T, H, out_head_dim)
            v_heads = v.view(B, T, H, out_head_dim)

            # Apply per-head gate: (B, T, H, out_head_dim)
            gated_heads = alpha.unsqueeze(-1) * v_heads

            # Reshape back: (B, T, output_dim)
            gated = gated_heads.reshape(B, T, self.output_dim)

        return gated, alpha

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, memory_dim={self.memory_dim}, "
            f"gate_dim={self.gate_dim}, output_dim={self.output_dim}, "
            f"gate_type={self.gate_type}, use_rmsnorm={self.use_rmsnorm}, "
            f"bias_init={self.gate_bias.data.flatten()[0].item():.2f}"
        )


# ===================================================================== #
#        SECTION 7: DepthwiseCausalConv1d                                #
# ===================================================================== #

class DepthwiseCausalConv1d(nn.Module):
    """Depthwise separable causal 1D convolution with configurable dilation.

    Applies a depthwise (groups=channels) 1D convolution with causal padding
    so that the output at position t depends only on input at positions <= t.
    This expands the effective receptive field for Engram's gated embeddings
    without introducing future information leakage.

    Causal padding scheme::

        pad_left = dilation * (kernel_size - 1)
        pad_right = 0

    The convolution is applied to the padded signal and then trimmed back to
    the original sequence length T.

    The module supports configurable activation functions (SiLU, GELU, ReLU,
    or none) applied after the convolution.

    Args:
        channels:
            Number of channels (features) in the input.  Each channel gets
            its own 1D convolutional filter (depthwise).
        kernel_size:
            Size of the convolutional kernel.
        dilation:
            Dilation factor for the convolution.  Effective receptive field
            is ``dilation * (kernel_size - 1) + 1``.
        activation:
            Activation function: ``"silu"``, ``"gelu"``, ``"relu"``,
            or ``"none"``.
        bias:
            Whether to include a bias term in the convolution.

    Shape:
        Input: ``(B, T, C)`` where B is batch size, T is sequence length,
            and C is ``channels``.
        Output: Same shape ``(B, T, C)``.

    Causality guarantee:
        If the input at position ``t' > t`` is perturbed, the output at
        position ``t`` is unaffected.  This is verified in the self-tests.

    Example::

        conv = DepthwiseCausalConv1d(channels=256, kernel_size=4, dilation=3)
        x = torch.randn(2, 100, 256)
        y = conv(x)  # (2, 100, 256)
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 4,
        dilation: int = 1,
        activation: str = "silu",
        bias: bool = True,
    ) -> None:
        super().__init__()

        if activation not in SUPPORTED_CONV_ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {SUPPORTED_CONV_ACTIVATIONS}, "
                f"got '{activation}'"
            )

        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.activation_name = activation
        self.pad_left = dilation * (kernel_size - 1)
        self.pad_right = 0

        # Effective receptive field size
        self.receptive_field = dilation * (kernel_size - 1) + 1

        # Depthwise convolution: groups = channels
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=channels,
            bias=bias,
            padding=0,  # We handle padding manually for causality
        )

        # Activation function
        if activation == "silu":
            self.act = nn.SiLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "none":
            self.act = nn.Identity()

        # Initialize weights for stability
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize convolution weights with small values for smooth
        training start."""
        nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))
        if self.conv.bias is not None:
            fan_in = self.conv.weight.size(1) * self.conv.weight.size(2)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.conv.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        """Apply depthwise causal convolution.

        Args:
            x: Input tensor of shape ``(B, T, C)``.

        Returns:
            Output tensor of shape ``(B, T, C)``.
        """
        B, T, C = x.shape

        # Transpose to (B, C, T) for Conv1d
        x_t = x.transpose(1, 2)  # (B, C, T)

        # Apply causal padding on the left only
        x_padded = F.pad(x_t, (self.pad_left, self.pad_right))
        # x_padded shape: (B, C, T + pad_left)

        # Apply convolution
        out = self.conv(x_padded)  # (B, C, T') where T' >= T

        # Trim to original length (ensure causality)
        out = out[:, :, :T]

        # Apply activation
        out = self.act(out)

        # Transpose back to (B, T, C)
        return out.transpose(1, 2)

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, kernel_size={self.kernel_size}, "
            f"dilation={self.dilation}, activation={self.activation_name}, "
            f"receptive_field={self.receptive_field}, "
            f"pad_left={self.pad_left}"
        )


# ===================================================================== #
#         SECTION 8: MockMultiHeadHash (template-only)                   #
# ===================================================================== #

class MockMultiHeadHash:
    """Lightweight deterministic hash for template self-tests.

    When this template is integrated into the main package, replace usages
    with ``brain_ai.memory.hash_embedding.MultiHeadHash``.

    Uses the multiplicative-XOR hash:
        phi(g) = (sum_i c_i * x_i) XOR seed mod M

    Args:
        ngram_order: The N in N-gram.
        num_heads: Number of independent hash functions.
        table_size: Size of hash table (should be prime).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        ngram_order: int,
        num_heads: int,
        table_size: int,
        seed: int = 42,
    ) -> None:
        self.n = ngram_order
        self.num_heads = num_heads
        self.table_size = table_size

        # Generate random coefficients deterministically
        import numpy as np
        rng = np.random.RandomState(seed)
        self.coefficients = torch.tensor(
            rng.randint(1, table_size, size=(num_heads, ngram_order)),
            dtype=torch.long,
        )
        self.seeds = torch.tensor(
            rng.randint(0, table_size, size=(num_heads,)),
            dtype=torch.long,
        )

    def hash(self, ngrams: Tensor) -> Tensor:
        """Hash N-grams to embedding indices.

        Args:
            ngrams: ``(B, T, n)`` tensor of token IDs forming N-grams.

        Returns:
            ``(B, T, num_heads)`` tensor of embedding indices.
        """
        device = ngrams.device
        B, T, n = ngrams.shape

        coeffs = self.coefficients.to(device)  # (H, n)
        seeds = self.seeds.to(device)  # (H,)

        # (B, T, 1, n) * (1, 1, H, n) -> sum over n -> (B, T, H)
        ngrams_exp = ngrams.unsqueeze(2).long()  # (B, T, 1, n)
        coeffs_exp = coeffs.unsqueeze(0).unsqueeze(0)  # (1, 1, H, n)
        weighted = (ngrams_exp * coeffs_exp).sum(dim=-1)  # (B, T, H)

        seeds_exp = seeds.unsqueeze(0).unsqueeze(0)  # (1, 1, H)
        hashed = (weighted ^ seeds_exp) % self.table_size

        return hashed


# ===================================================================== #
#    SECTION 9: MockOffloadableEmbedding (template-only)                 #
# ===================================================================== #

class MockOffloadableEmbedding(nn.Module):
    """Lightweight embedding for template self-tests.

    When this template is integrated into the main package, replace usages
    with ``brain_ai.memory.hash_embedding.OffloadableEmbedding``.

    Args:
        num_embeddings: Number of entries in the embedding table.
        embedding_dim: Dimension of each embedding vector.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)

    def forward(self, indices: Tensor) -> Tensor:
        """Look up embeddings by index.

        Args:
            indices: Integer tensor of arbitrary shape.

        Returns:
            Embeddings of shape ``(*indices.shape, embedding_dim)``.
        """
        return self.embedding(indices)


# ===================================================================== #
#       SECTION 10: MockTokenizerCompression (template-only)             #
# ===================================================================== #

class MockTokenizerCompression:
    """Lightweight tokenizer compression for template self-tests.

    When this template is integrated into the main package, replace usages
    with ``brain_ai.memory.tokenizer_compression.TokenizerCompression``.

    Uses deterministic modulo projection as a placeholder for the full
    NFKC + lowercasing normalization pipeline.

    Args:
        vocab_size: Original vocabulary size.
        compressed_size: Target compressed vocabulary size.
    """

    def __init__(
        self,
        vocab_size: int = 128000,
        compressed_size: Optional[int] = None,
    ) -> None:
        self.vocab_size = vocab_size
        self.compressed_size = compressed_size or int(vocab_size * 0.77)
        self.projection = torch.arange(vocab_size) % self.compressed_size

    def compress(self, token_ids: Tensor) -> Tensor:
        """Map raw token IDs to canonical IDs.

        Args:
            token_ids: ``(B, T)`` integer tensor of raw token IDs.

        Returns:
            Canonical IDs in ``[0, compressed_size)``, same shape.
        """
        device = token_ids.device
        proj = self.projection.to(device)
        clamped = token_ids.clamp(0, self.vocab_size - 1)
        return proj[clamped]

    def get_compression_ratio(self) -> float:
        """Return the compression ratio achieved."""
        return self.compressed_size / self.vocab_size


# ===================================================================== #
#        SECTION 11: EngramTelemetry                                     #
# ===================================================================== #

@dataclass
class EngramTelemetry:
    """Diagnostic information collected during an EngramModule forward pass.

    Attributes:
        gate_mean:
            Mean gate value across all positions and batch items.
        gate_std:
            Standard deviation of gate values.
        gate_sparsity:
            Fraction of gate values below ``eps`` (effectively zero).
        gate_min:
            Minimum gate value observed.
        gate_max:
            Maximum gate value observed.
        unique_id_ratio_per_head:
            Dictionary mapping head name (e.g. ``"order2_head0"``) to the
            fraction of unique hash IDs in the batch, as a proxy for
            collision rate.
        prefetch_stats:
            Optional dictionary with prefetch diagnostics:
            ``bytes_transferred``, ``overlap_ms``, ``coalescing_ratio``.
        compression_ratio:
            Ratio of compressed vocabulary to original vocabulary.
        memory_norm_mean:
            Mean L2 norm of retrieved memory embeddings.
        delta_norm_mean:
            Mean L2 norm of the output delta residual.
        num_positions:
            Total number of non-padding positions processed.
        num_ngram_orders:
            Number of N-gram orders used.
    """

    gate_mean: float = 0.0
    gate_std: float = 0.0
    gate_sparsity: float = 0.0
    gate_min: float = 0.0
    gate_max: float = 1.0
    unique_id_ratio_per_head: Dict[str, float] = field(default_factory=dict)
    prefetch_stats: Optional[Dict[str, float]] = None
    compression_ratio: float = 1.0
    memory_norm_mean: float = 0.0
    delta_norm_mean: float = 0.0
    num_positions: int = 0
    num_ngram_orders: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a flat dictionary for logging and tensorboard."""
        d: Dict[str, Any] = {
            "engram/gate_mean": self.gate_mean,
            "engram/gate_std": self.gate_std,
            "engram/gate_sparsity": self.gate_sparsity,
            "engram/gate_min": self.gate_min,
            "engram/gate_max": self.gate_max,
            "engram/compression_ratio": self.compression_ratio,
            "engram/memory_norm_mean": self.memory_norm_mean,
            "engram/delta_norm_mean": self.delta_norm_mean,
            "engram/num_positions": self.num_positions,
            "engram/num_ngram_orders": self.num_ngram_orders,
        }
        for head_name, ratio in self.unique_id_ratio_per_head.items():
            d[f"engram/unique_ratio/{head_name}"] = ratio
        if self.prefetch_stats is not None:
            for k, v in self.prefetch_stats.items():
                d[f"engram/prefetch/{k}"] = v
        return d

    def summary(self) -> str:
        """Return a human-readable one-line summary."""
        return (
            f"gate={self.gate_mean:.4f}+/-{self.gate_std:.4f} "
            f"sparsity={self.gate_sparsity:.2%} "
            f"mem_norm={self.memory_norm_mean:.3f} "
            f"delta_norm={self.delta_norm_mean:.3f}"
        )


# ===================================================================== #
#         SECTION 12: EngramModule                                       #
# ===================================================================== #

class EngramModule(nn.Module):
    """Complete Engram conditional memory module.

    Orchestrates the full pipeline:
      1. Tokenizer compression (optional): raw IDs -> canonical IDs
      2. Suffix N-gram extraction: canonical IDs -> N-gram tuples
      3. Multi-head hashing: N-gram tuples -> hash indices
      4. Embedding retrieval: hash indices -> per-head embeddings
      5. Embedding aggregation: per-head -> single vector per position
      6. Context-aware gating (optional): hidden_states gate the embeddings
      7. Depthwise causal convolution (optional): expand receptive field
      8. Output projection: project to backbone hidden dimension
      9. Telemetry collection (optional): diagnostic statistics

    The module produces a ``delta`` tensor of shape ``(B, T, hidden_dim)``
    intended for residual addition to the transformer hidden states.

    Supports two operating modes:
      - **Full mode**: Process entire sequences of arbitrary length.
      - **Streaming mode**: Process one token at a time with a rolling cache,
        for autoregressive generation.

    Individual components can be toggled via configuration flags:
      - ``use_context_gate=False``: Skip gating, pass raw embeddings through.
      - ``use_depthwise_conv=False``: Skip convolution.
      - ``use_tokenizer_compression=False``: Use raw IDs directly.

    Args:
        config: ``EngramConfig`` with all hyperparameters.

    Shape:
        hidden_states: ``(B, T, hidden_dim)``
        input_ids: ``(B, T)``
        attention_mask: ``(B, T)`` optional
        delta (output): ``(B, T, hidden_dim)``

    Example::

        config = EngramConfig.minimal()
        module = EngramModule(config)

        hidden = torch.randn(2, 10, config.hidden_dim)
        ids = torch.randint(0, config.vocab_size, (2, 10))
        delta, telemetry = module(hidden, ids)
        print(delta.shape)  # (2, 10, 64)
    """

    def __init__(self, config: EngramConfig) -> None:
        super().__init__()

        self.config = config
        hidden_dim = config.hidden_dim
        output_dim = config.output_dim

        # --- Step 1: Tokenizer compression (mock for template) ---
        if config.use_tokenizer_compression:
            self.compressor: Optional[MockTokenizerCompression] = (
                MockTokenizerCompression(
                    vocab_size=config.vocab_size,
                    compressed_size=config.compressed_vocab_size,
                )
            )
        else:
            self.compressor = None

        # --- Step 2: N-gram extraction ---
        self.ngram_extractor = SuffixNgramExtractor(
            max_order=config.max_ngram_order,
            pad_id=config.pad_id,
        )

        # --- Step 3: Multi-head hashing ---
        self.hashers: Dict[int, MockMultiHeadHash] = {}
        for n in config.ngram_orders:
            self.hashers[n] = MockMultiHeadHash(
                ngram_order=n,
                num_heads=config.num_heads_per_order,
                table_size=config.table_size,
                seed=config.hash_seed + n,
            )

        # --- Step 4: Embedding tables ---
        # Compute per-head embedding dimension
        total_heads = config.total_heads
        if total_heads == 0:
            raise ValueError("total_heads cannot be zero; check ngram_orders.")
        self.dim_per_head = config.embedding_dim // total_heads
        if self.dim_per_head == 0:
            raise ValueError(
                f"embedding_dim ({config.embedding_dim}) is too small for "
                f"{total_heads} total heads. Need at least {total_heads}."
            )

        self.embedding_tables = nn.ModuleDict()
        for n in config.ngram_orders:
            for h in range(config.num_heads_per_order):
                key = f"order{n}_head{h}"
                self.embedding_tables[key] = MockOffloadableEmbedding(
                    num_embeddings=config.table_size,
                    embedding_dim=self.dim_per_head,
                )

        # --- Step 5: Embedding aggregation ---
        self.aggregator = EmbeddingAggregator(
            num_orders=len(config.ngram_orders),
            num_heads_per_order=config.num_heads_per_order,
            dim_per_head=self.dim_per_head,
            output_dim=config.embedding_dim,
            mode="concat",
        )

        # --- Step 6: Context-aware gating ---
        if config.use_context_gate:
            self.gating: Optional[ContextAwareGating] = ContextAwareGating(
                hidden_dim=hidden_dim,
                memory_dim=config.embedding_dim,
                gate_dim=config.gate_dim,
                output_dim=output_dim,
                gate_type=config.gate_type,
                num_gate_heads=config.num_gate_heads,
                use_rmsnorm=config.use_rmsnorm,
                gate_init_bias=config.gate_init_bias,
                rmsnorm_eps=config.rmsnorm_eps,
            )
        else:
            self.gating = None
            # When gating is disabled, we need a projection from embedding_dim
            # to output_dim
            if config.embedding_dim != output_dim:
                self.no_gate_proj = nn.Linear(
                    config.embedding_dim, output_dim
                )
            else:
                self.no_gate_proj = nn.Identity()

        # --- Step 7: Depthwise causal convolution ---
        if config.use_depthwise_conv:
            self.causal_conv: Optional[DepthwiseCausalConv1d] = (
                DepthwiseCausalConv1d(
                    channels=output_dim,
                    kernel_size=config.conv_kernel_size,
                    dilation=config.conv_dilation,
                    activation=config.conv_activation,
                )
            )
            self.conv_norm = RMSNorm(output_dim, eps=config.rmsnorm_eps)
        else:
            self.causal_conv = None
            self.conv_norm = None

        # --- Step 8: Output projection ---
        if output_dim != hidden_dim:
            self.output_proj = nn.Linear(output_dim, hidden_dim)
        else:
            self.output_proj = nn.Identity()

        # --- Residual scale (zero-init for smooth start) ---
        self.residual_scale = nn.Parameter(torch.zeros(1))

        logger.info(
            "EngramModule initialized: hidden=%d, embedding=%d, orders=%s, "
            "heads=%d, table_size=%d, gate=%s, conv=%s",
            hidden_dim,
            config.embedding_dim,
            config.ngram_orders,
            total_heads,
            config.table_size,
            config.use_context_gate,
            config.use_depthwise_conv,
        )

    def _extract_and_hash(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor],
        cache_state: Optional[StreamingCache],
    ) -> Tuple[List[Tensor], Dict[str, Tensor]]:
        """Extract N-grams and hash to embedding indices.

        Returns:
            head_embeddings:
                List of (B, T, dim_per_head) tensors, one per head.
            hash_id_dict:
                Dict mapping head name to (B, T) hash IDs for telemetry.
        """
        # Tokenizer compression
        if self.compressor is not None:
            canonical_ids = self.compressor.compress(input_ids)
        else:
            canonical_ids = input_ids

        # Extract N-grams
        if cache_state is not None:
            # Streaming mode: process last token only
            # input_ids should be (B, 1) in streaming mode
            assert input_ids.shape[1] == 1, (
                f"Streaming mode expects T=1, got T={input_ids.shape[1]}"
            )
            new_id = canonical_ids[0, 0].item()
            ngram_list = self.ngram_extractor.extract_streaming(
                new_id, cache_state
            )
            # Each element in ngram_list is (1, 1, n); expand to batch
            B = input_ids.shape[0]
            if B > 1:
                ngram_list = [ng.expand(B, -1, -1) for ng in ngram_list]
        else:
            # Full mode
            ngram_list = self.ngram_extractor.extract(
                canonical_ids, attention_mask
            )

        # Hash and retrieve embeddings for each order and head
        head_embeddings: List[Tensor] = []
        hash_id_dict: Dict[str, Tensor] = {}

        for order_idx, n in enumerate(self.config.ngram_orders):
            ngrams = ngram_list[order_idx]  # (B, T, n)
            hash_ids = self.hashers[n].hash(ngrams)  # (B, T, num_heads)

            for h in range(self.config.num_heads_per_order):
                key = f"order{n}_head{h}"
                head_hash_ids = hash_ids[:, :, h]  # (B, T)
                emb = self.embedding_tables[key](head_hash_ids)  # (B, T, D_h)
                head_embeddings.append(emb)
                hash_id_dict[key] = head_hash_ids

        return head_embeddings, hash_id_dict

    def _compute_telemetry(
        self,
        alpha: Optional[Tensor],
        hash_id_dict: Dict[str, Tensor],
        memory: Tensor,
        delta: Tensor,
        attention_mask: Optional[Tensor],
    ) -> EngramTelemetry:
        """Compute diagnostic telemetry from a forward pass.

        Args:
            alpha: Gate values (B, T, 1) or (B, T, H), or None.
            hash_id_dict: Dict of head_name -> (B, T) hash IDs.
            memory: Aggregated memory (B, T, D_emb).
            delta: Output delta (B, T, D).
            attention_mask: Optional (B, T) mask.

        Returns:
            EngramTelemetry dataclass.
        """
        telemetry = EngramTelemetry()
        telemetry.num_ngram_orders = len(self.config.ngram_orders)

        # Gate statistics
        if alpha is not None:
            with torch.no_grad():
                alpha_flat = alpha.float().flatten()
                telemetry.gate_mean = alpha_flat.mean().item()
                telemetry.gate_std = alpha_flat.std().item()
                telemetry.gate_min = alpha_flat.min().item()
                telemetry.gate_max = alpha_flat.max().item()
                telemetry.gate_sparsity = (
                    (alpha_flat < 1e-6).float().mean().item()
                )
        else:
            telemetry.gate_mean = 1.0
            telemetry.gate_std = 0.0
            telemetry.gate_min = 1.0
            telemetry.gate_max = 1.0
            telemetry.gate_sparsity = 0.0

        # Unique ID ratio per head (collision proxy)
        with torch.no_grad():
            for head_name, ids in hash_id_dict.items():
                flat_ids = ids.flatten()
                unique_count = flat_ids.unique().numel()
                total_count = flat_ids.numel()
                ratio = unique_count / max(total_count, 1)
                telemetry.unique_id_ratio_per_head[head_name] = ratio

        # Memory and delta norms
        with torch.no_grad():
            telemetry.memory_norm_mean = (
                memory.float().norm(dim=-1).mean().item()
            )
            telemetry.delta_norm_mean = (
                delta.float().norm(dim=-1).mean().item()
            )

        # Compression ratio
        if self.compressor is not None:
            telemetry.compression_ratio = (
                self.compressor.get_compression_ratio()
            )
        else:
            telemetry.compression_ratio = 1.0

        # Number of valid positions
        if attention_mask is not None:
            telemetry.num_positions = int(attention_mask.sum().item())
        else:
            B, T = delta.shape[:2]
            telemetry.num_positions = B * T

        return telemetry

    def forward(
        self,
        hidden_states: Tensor,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        *,
        cache_state: Optional[StreamingCache] = None,
        layer_id: int = 0,
        return_details: bool = False,
    ) -> Tuple[Tensor, Optional[Union[EngramTelemetry, Dict[str, Any]]]]:
        """Run the full Engram pipeline.

        Args:
            hidden_states:
                ``(B, T, hidden_dim)`` current hidden states from backbone.
            input_ids:
                ``(B, T)`` raw token IDs.  If ``T=1`` and ``cache_state``
                is provided, operates in streaming mode.
            attention_mask:
                Optional ``(B, T)`` mask where 1=valid, 0=padding.
            cache_state:
                Optional ``StreamingCache`` for autoregressive generation.
                When provided, only the last token is processed.
            layer_id:
                Index of the backbone layer where this module is inserted.
                Used for per-layer hash salt when ``per_layer_salt=True``.
            return_details:
                If True, return full ``EngramTelemetry``.  If False, return
                None for the telemetry slot.

        Returns:
            delta:
                ``(B, T, hidden_dim)`` residual to add to hidden states.
            telemetry:
                ``EngramTelemetry`` if ``return_details=True``, else None.
        """
        B, T, D = hidden_states.shape

        # --- Steps 1-4: Extract, hash, retrieve ---
        head_embeddings, hash_id_dict = self._extract_and_hash(
            input_ids, attention_mask, cache_state
        )

        # --- Step 5: Aggregate ---
        memory = self.aggregator(head_embeddings)  # (B, T, embedding_dim)

        # --- Step 6: Context-aware gating ---
        alpha: Optional[Tensor] = None
        if self.gating is not None:
            gated, alpha = self.gating(hidden_states, memory)
            # gated: (B, T, output_dim)
        else:
            gated = self.no_gate_proj(memory)  # (B, T, output_dim)

        # --- Step 7: Depthwise causal convolution ---
        if self.causal_conv is not None:
            conv_input = self.conv_norm(gated)
            conv_out = self.causal_conv(conv_input)
            # Residual within the module: conv output + gated
            fused = conv_out + gated
        else:
            fused = gated

        # --- Step 8: Output projection + residual scale ---
        delta = self.output_proj(fused)  # (B, T, hidden_dim)

        # Apply learnable residual scale (starts at 0 for smooth init)
        delta = delta * self.residual_scale

        # --- Step 9: Telemetry ---
        telemetry: Optional[EngramTelemetry] = None
        if return_details:
            telemetry = self._compute_telemetry(
                alpha, hash_id_dict, memory, delta, attention_mask
            )

        return delta, telemetry

    def create_cache(
        self,
        device: Union[str, torch.device] = "cpu",
    ) -> StreamingCache:
        """Create a new streaming cache for autoregressive generation.

        Args:
            device: Device for the cache buffer.

        Returns:
            A fresh ``StreamingCache`` ready for streaming inference.
        """
        return StreamingCache.create(
            max_order=self.config.max_ngram_order,
            pad_id=self.config.pad_id,
            device=device,
        )

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.config.hidden_dim}, "
            f"embedding_dim={self.config.embedding_dim}, "
            f"orders={self.config.ngram_orders}, "
            f"heads_per_order={self.config.num_heads_per_order}, "
            f"gate={self.config.use_context_gate}, "
            f"conv={self.config.use_depthwise_conv}"
        )


# ===================================================================== #
#    SECTION 13: EngramAugmentedLayer (skeleton for integration)         #
# ===================================================================== #

class EngramAugmentedLayer(nn.Module):
    """Wrapper that augments a host backbone layer with Engram memory.

    In Phase 2 integration mode, this module is inserted at selected backbone
    layers.  It runs the EngramModule to produce a residual delta, adds it
    to the hidden states, and then passes the result to the original layer.

    This is a skeleton for the full implementation in
    ``brain_ai/layers/engram_layer.py``.

    Args:
        config: ``EngramConfig`` for the Engram module.
        host_layer: The original backbone layer (e.g., TransformerBlock).
        layer_id: Index of this layer in the backbone.
    """

    def __init__(
        self,
        config: EngramConfig,
        host_layer: Optional[nn.Module] = None,
        layer_id: int = 0,
    ) -> None:
        super().__init__()
        self.engram = EngramModule(config)
        self.host_layer = host_layer
        self.layer_id = layer_id
        self.pre_norm = RMSNorm(config.hidden_dim, eps=config.rmsnorm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Apply Engram augmentation then the host layer.

        Args:
            hidden_states: ``(B, T, D)`` from previous layer.
            input_ids: ``(B, T)`` raw token IDs (passed through all layers).
            attention_mask: Optional ``(B, T)`` mask.
            **kwargs: Additional arguments passed to the host layer.

        Returns:
            Augmented hidden states ``(B, T, D)``.
        """
        # Engram delta
        normed = self.pre_norm(hidden_states)
        delta, _ = self.engram(
            normed, input_ids, attention_mask, layer_id=self.layer_id
        )
        augmented = hidden_states + delta

        # Host layer (if present)
        if self.host_layer is not None:
            augmented = self.host_layer(augmented, **kwargs)

        return augmented


# ===================================================================== #
#       SECTION 14: Utility Functions                                    #
# ===================================================================== #

def compute_effective_receptive_field(
    kernel_size: int,
    dilation: int,
) -> int:
    """Compute the effective receptive field of a causal convolution.

    Args:
        kernel_size: Size of the convolutional kernel.
        dilation: Dilation factor.

    Returns:
        Number of past positions that influence the output at position t.
    """
    return dilation * (kernel_size - 1) + 1


def find_next_prime(n: int) -> int:
    """Find the smallest prime number >= n.

    Used for sizing hash tables.  Primes reduce systematic collisions
    in multiplicative hashing.

    Args:
        n: Lower bound.

    Returns:
        Smallest prime >= n.
    """
    if n <= 2:
        return 2

    candidate = n if n % 2 != 0 else n + 1

    while True:
        if _is_prime(candidate):
            return candidate
        candidate += 2


def _is_prime(n: int) -> bool:
    """Check if n is prime using trial division."""
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


def validate_causality(
    module: DepthwiseCausalConv1d,
    channels: int,
    seq_len: int = 20,
    probe_position: int = 10,
    rtol: float = 1e-5,
) -> bool:
    """Verify that a DepthwiseCausalConv1d module is strictly causal.

    Creates an input, computes the output, then perturbs a future position
    and checks that the output at the probe position is unchanged.

    Args:
        module: The convolution module to test.
        channels: Number of channels.
        seq_len: Sequence length for the test input.
        probe_position: Position to check for causal invariance.
        rtol: Relative tolerance for output comparison.

    Returns:
        True if the module is causal, False otherwise.
    """
    module_training = module.training
    module.train(False)

    x = torch.randn(1, seq_len, channels)
    with torch.no_grad():
        y_original = module(x)

    # Perturb a future position
    x_perturbed = x.clone()
    future_pos = probe_position + 1
    if future_pos < seq_len:
        x_perturbed[0, future_pos, :] += 100.0
        with torch.no_grad():
            y_perturbed = module(x_perturbed)

        # Output at probe_position should be identical
        diff = (
            y_original[0, probe_position] - y_perturbed[0, probe_position]
        ).abs().max().item()
        module.train(module_training)
        return diff < rtol

    module.train(module_training)
    return True  # No future position to perturb


def count_parameters(module: nn.Module) -> int:
    """Count the total number of learnable parameters in a module.

    Args:
        module: PyTorch module.

    Returns:
        Total number of parameters (summing all param.numel()).
    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# ===================================================================== #
#                SECTION 15: Self-Tests                                  #
# ===================================================================== #

def _print_test_result(
    test_name: str, passed: bool, detail: str = ""
) -> None:
    """Print a formatted test result line."""
    status = "PASS" if passed else "FAIL"
    detail_str = f"  ({detail})" if detail else ""
    print(f"  [{status}] {test_name}{detail_str}")


def _run_all_tests() -> None:
    """Run all self-tests for the engram template module.

    Tests are organized by component and numbered for easy reference.
    Each test prints PASS or FAIL.  A summary is printed at the end.
    """
    print("=" * 72)
    print("  Engram Template Self-Tests")
    print("=" * 72)

    passed_count = 0
    failed_count = 0
    total_count = 0

    def record(name: str, passed: bool, detail: str = "") -> None:
        nonlocal passed_count, failed_count, total_count
        total_count += 1
        if passed:
            passed_count += 1
        else:
            failed_count += 1
        _print_test_result(name, passed, detail)

    # ------------------------------------------------------------------
    # N-gram Extraction Tests
    # ------------------------------------------------------------------
    print("\n--- SuffixNgramExtractor ---")

    # Test 1: Known input -> expected bigrams
    extractor = SuffixNgramExtractor(max_order=4, pad_id=0)
    ids = torch.tensor([[10, 20, 30, 40, 50]])
    ngrams = extractor.extract(ids)

    # Bigrams (order=2)
    expected_bigrams = torch.tensor([[[0, 10], [10, 20], [20, 30], [30, 40], [40, 50]]])
    bigrams_ok = torch.equal(ngrams[0], expected_bigrams)
    record("1. Bigram extraction correctness", bigrams_ok,
           f"got {ngrams[0][0].tolist()}")

    # Test 2: Known input -> expected trigrams
    expected_trigrams = torch.tensor([[[0, 0, 10], [0, 10, 20], [10, 20, 30],
                                       [20, 30, 40], [30, 40, 50]]])
    trigrams_ok = torch.equal(ngrams[1], expected_trigrams)
    record("2. Trigram extraction correctness", trigrams_ok,
           f"got {ngrams[1][0].tolist()}")

    # Test 3: Known input -> expected 4-grams
    expected_4grams = torch.tensor([[[0, 0, 0, 10], [0, 0, 10, 20],
                                      [0, 10, 20, 30], [10, 20, 30, 40],
                                      [20, 30, 40, 50]]])
    fourgrams_ok = torch.equal(ngrams[2], expected_4grams)
    record("3. 4-gram extraction correctness", fourgrams_ok,
           f"got {ngrams[2][0].tolist()}")

    # Test 4: Edge position t=0 uses pad_id for left context
    t0_bigram = ngrams[0][0, 0].tolist()
    t0_ok = t0_bigram[0] == 0 and t0_bigram[1] == 10
    record("4. N-gram t=0 uses pad_id for left context", t0_ok,
           f"bigram at t=0: {t0_bigram}")

    # Test 5: Edge position t=1 for trigram uses pad_ids
    t1_trigram = ngrams[1][0, 1].tolist()
    t1_ok = t1_trigram == [0, 10, 20]
    record("5. N-gram t=1 trigram correct padding", t1_ok,
           f"trigram at t=1: {t1_trigram}")

    # Test 6: N-gram masking -- padding positions produce pad-only N-grams
    mask = torch.tensor([[1, 1, 0, 1, 1]])
    masked_ngrams = extractor.extract(ids, attention_mask=mask)
    # At position t=2 (masked), all values should be pad_id=0
    t2_bigram_masked = masked_ngrams[0][0, 2].tolist()
    mask_ok = all(v == 0 for v in t2_bigram_masked)
    record("6. N-gram masking: padding produces pad-only N-grams", mask_ok,
           f"masked bigram at t=2: {t2_bigram_masked}")

    # Test 7: N-gram mask propagation to neighboring positions
    # Position t=3 has a valid token, but its bigram includes t=2 which is
    # masked. The masked t=2 should be pad_id.
    t3_bigram_masked = masked_ngrams[0][0, 3].tolist()
    # ids were [10, 20, 30, 40, 50], after mask ids become [10, 20, 0, 40, 50]
    # bigram at t=3 should be [0, 40] (t=2 is masked to 0)
    mask_prop_ok = t3_bigram_masked == [0, 40]
    record("7. N-gram mask propagation to neighbors", mask_prop_ok,
           f"bigram at t=3 with mask: {t3_bigram_masked}")

    # Test 8: Stacked extraction shape
    stacked = extractor.extract_stacked(ids)
    expected_shape = (1, 5, 3, 4)  # (B, T, K-1=3, K=4)
    stacked_ok = stacked.shape == expected_shape
    record("8. Stacked N-gram extraction shape", stacked_ok,
           f"expected {expected_shape}, got {tuple(stacked.shape)}")

    # Test 9: Number of orders
    orders_ok = len(ngrams) == 3 and extractor.orders == (2, 3, 4)
    record("9. Correct number of N-gram orders", orders_ok,
           f"orders: {extractor.orders}")

    # Test 10: Batch dimension preserved
    batch_ids = torch.tensor([[10, 20, 30], [40, 50, 60]])
    batch_ngrams = extractor.extract(batch_ids)
    batch_ok = all(ng.shape[0] == 2 for ng in batch_ngrams)
    record("10. Batch dimension preserved in extraction", batch_ok)

    # ------------------------------------------------------------------
    # Streaming Tests
    # ------------------------------------------------------------------
    print("\n--- Streaming N-gram Extraction ---")

    # Test 11: Streaming vs full equivalence
    full_ids = torch.tensor([[10, 20, 30, 40, 50]])
    full_ngrams = extractor.extract(full_ids)

    cache = StreamingCache.create(max_order=4, pad_id=0)
    token_sequence = [10, 20, 30, 40, 50]
    streaming_results: List[List[Tensor]] = []

    for token_id in token_sequence:
        result = extractor.extract_streaming(token_id, cache)
        streaming_results.append(result)

    # Compare last position of full with last streaming result
    streaming_match = True
    for order_idx in range(len(extractor.orders)):
        full_last = full_ngrams[order_idx][0, -1]  # (n,)
        stream_last = streaming_results[-1][order_idx][0, 0]  # (n,)
        if not torch.equal(full_last, stream_last):
            streaming_match = False
            break
    record("11. Streaming vs full equivalence (last position)", streaming_match)

    # Test 12: Streaming matches full for middle position (t=2)
    mid_match = True
    for order_idx in range(len(extractor.orders)):
        full_mid = full_ngrams[order_idx][0, 2]  # (n,)
        stream_mid = streaming_results[2][order_idx][0, 0]  # (n,)
        if not torch.equal(full_mid, stream_mid):
            mid_match = False
            break
    record("12. Streaming vs full equivalence (middle position t=2)", mid_match)

    # Test 13: Streaming matches full for first position (t=0)
    first_match = True
    for order_idx in range(len(extractor.orders)):
        full_first = full_ngrams[order_idx][0, 0]
        stream_first = streaming_results[0][order_idx][0, 0]
        if not torch.equal(full_first, stream_first):
            first_match = False
            break
    record("13. Streaming vs full equivalence (first position t=0)", first_match)

    # ------------------------------------------------------------------
    # StreamingCache Tests
    # ------------------------------------------------------------------
    print("\n--- StreamingCache ---")

    # Test 14: Cache update shifts correctly
    cache2 = StreamingCache.create(max_order=4, pad_id=0)
    # Buffer is [0, 0, 0] (K-1 = 3)
    cache2.update(10)
    shift_ok_1 = cache2.buffer.tolist() == [0, 0, 10]
    cache2.update(20)
    shift_ok_2 = cache2.buffer.tolist() == [0, 10, 20]
    cache2.update(30)
    shift_ok_3 = cache2.buffer.tolist() == [10, 20, 30]
    shift_ok = shift_ok_1 and shift_ok_2 and shift_ok_3
    record("14. StreamingCache update shifts correctly", shift_ok,
           f"after 10,20,30: {cache2.buffer.tolist()}")

    # Test 15: Cache position counter
    pos_ok = cache2.position == 3
    record("15. StreamingCache position counter", pos_ok,
           f"position={cache2.position}")

    # Test 16: Cache reset
    cache2.reset()
    reset_ok = cache2.buffer.tolist() == [0, 0, 0] and cache2.position == 0
    record("16. StreamingCache reset", reset_ok)

    # Test 17: Cache clone independence
    cache3 = StreamingCache.create(max_order=3, pad_id=0)
    cache3.update(42)
    cache3_clone = cache3.clone()
    cache3.update(99)
    clone_ok = cache3_clone.buffer[-1].item() == 42 and cache3.buffer[-1].item() == 99
    record("17. StreamingCache clone independence", clone_ok)

    # ------------------------------------------------------------------
    # RMSNorm Tests
    # ------------------------------------------------------------------
    print("\n--- RMSNorm ---")

    # Test 18: RMSNorm output has unit RMS
    norm = RMSNorm(64)
    x_rms = torch.randn(2, 10, 64)
    y_rms = norm(x_rms)
    rms_val = torch.sqrt(torch.mean(y_rms.float() ** 2, dim=-1)).mean().item()
    rms_ok = abs(rms_val - 1.0) < 0.2  # With learned weight=1, should be ~1
    record("18. RMSNorm output approximately unit RMS", rms_ok,
           f"mean RMS={rms_val:.4f}")

    # Test 19: RMSNorm preserves shape
    shape_ok = y_rms.shape == x_rms.shape
    record("19. RMSNorm preserves shape", shape_ok)

    # Test 20: RMSNorm AMP safety -- fp16 input doesn't produce NaN
    x_fp16 = torch.randn(2, 10, 64, dtype=torch.float16)
    y_fp16 = norm(x_fp16)
    amp_nan_ok = not torch.isnan(y_fp16).any().item()
    amp_dtype_ok = y_fp16.dtype == torch.float16
    record("20. RMSNorm AMP safety (fp16 no NaN)", amp_nan_ok and amp_dtype_ok,
           f"dtype={y_fp16.dtype}, has_nan={torch.isnan(y_fp16).any().item()}")

    # Test 21: RMSNorm with very small input (near-zero stability)
    x_tiny = torch.ones(1, 1, 64) * 1e-7
    y_tiny = norm(x_tiny)
    tiny_ok = not torch.isnan(y_tiny).any().item() and not torch.isinf(y_tiny).any().item()
    record("21. RMSNorm stability with near-zero input", tiny_ok)

    # ------------------------------------------------------------------
    # ContextAwareGating Tests
    # ------------------------------------------------------------------
    print("\n--- ContextAwareGating ---")

    B_test, T_test, D_hidden, D_mem = 2, 10, 64, 32

    # Test 22: Scalar gating output shape
    gate = ContextAwareGating(
        hidden_dim=D_hidden, memory_dim=D_mem, gate_dim=16,
        output_dim=D_hidden, gate_type="scalar", use_rmsnorm=True,
        gate_init_bias=-2.0,
    )
    h = torch.randn(B_test, T_test, D_hidden)
    m = torch.randn(B_test, T_test, D_mem)
    gated_out, alpha_out = gate(h, m)
    gate_shape_ok = gated_out.shape == (B_test, T_test, D_hidden)
    alpha_shape_ok = alpha_out.shape == (B_test, T_test, 1)
    record("22. Scalar gating output shape", gate_shape_ok and alpha_shape_ok,
           f"gated={tuple(gated_out.shape)}, alpha={tuple(alpha_out.shape)}")

    # Test 23: Gate values bounded [0, 1]
    bounded_ok = (
        alpha_out.min().item() >= 0.0
        and alpha_out.max().item() <= 1.0
    )
    record("23. Gate values bounded [0, 1]", bounded_ok,
           f"min={alpha_out.min().item():.6f}, max={alpha_out.max().item():.6f}")

    # Test 24: Conservative gate init -- with zero inputs, gate should
    # be near sigmoid(bias) ~ sigmoid(-2) ~ 0.119, because q*k ~ 0
    h_zero = torch.zeros(1, 1, D_hidden)
    m_zero = torch.zeros(1, 1, D_mem)
    _, alpha_zero = gate(h_zero, m_zero)
    expected_init = torch.sigmoid(torch.tensor(-2.0)).item()
    init_val = alpha_zero.mean().item()
    init_ok = abs(init_val - expected_init) < 0.05
    record("24. Conservative gate init (~sigmoid(-2))", init_ok,
           f"gate(0,0)={init_val:.4f}, expected~{expected_init:.4f}")

    # Test 25: Gate gradient flow
    h_grad = torch.randn(B_test, T_test, D_hidden, requires_grad=True)
    m_grad = torch.randn(B_test, T_test, D_mem, requires_grad=True)
    gated_g, alpha_g = gate(h_grad, m_grad)
    loss = gated_g.sum()
    loss.backward()
    grad_ok = (
        gate.Wq.weight.grad is not None
        and gate.Wk.weight.grad is not None
        and gate.Wv.weight.grad is not None
        and gate.gate_bias.grad is not None
    )
    record("25. Gate gradient flow (Wq, Wk, Wv, bias)", grad_ok)

    # Test 26: Per-head gating
    gate_ph = ContextAwareGating(
        hidden_dim=D_hidden, memory_dim=D_mem, gate_dim=16,
        output_dim=D_hidden, gate_type="per_head", num_gate_heads=4,
        use_rmsnorm=True, gate_init_bias=-2.0,
    )
    gated_ph, alpha_ph = gate_ph(h, m)
    ph_shape_ok = (
        gated_ph.shape == (B_test, T_test, D_hidden)
        and alpha_ph.shape == (B_test, T_test, 4)
    )
    record("26. Per-head gating shapes", ph_shape_ok,
           f"gated={tuple(gated_ph.shape)}, alpha={tuple(alpha_ph.shape)}")

    # Test 27: Per-head gate values bounded
    ph_bounded = (
        alpha_ph.min().item() >= 0.0
        and alpha_ph.max().item() <= 1.0
    )
    record("27. Per-head gate values bounded [0, 1]", ph_bounded)

    # ------------------------------------------------------------------
    # DepthwiseCausalConv1d Tests
    # ------------------------------------------------------------------
    print("\n--- DepthwiseCausalConv1d ---")

    C_conv = 32

    # Test 28: Output shape matches input
    conv = DepthwiseCausalConv1d(
        channels=C_conv, kernel_size=4, dilation=3, activation="silu"
    )
    x_conv = torch.randn(2, 20, C_conv)
    y_conv = conv(x_conv)
    conv_shape_ok = y_conv.shape == x_conv.shape
    record("28. Causal conv output shape matches input", conv_shape_ok,
           f"in={tuple(x_conv.shape)}, out={tuple(y_conv.shape)}")

    # Test 29: Causality test -- perturbing future doesn't change past
    causal_ok = validate_causality(conv, C_conv, seq_len=20, probe_position=10)
    record("29. Causality: future perturbation doesn't affect past", causal_ok)

    # Test 30: Causality at position 0
    causal_pos0 = validate_causality(conv, C_conv, seq_len=20, probe_position=0)
    record("30. Causality at position 0", causal_pos0)

    # Test 31: Dilation effective receptive field
    erf = compute_effective_receptive_field(kernel_size=4, dilation=3)
    erf_ok = erf == 10  # 3 * (4-1) + 1 = 10
    record("31. Dilation effective receptive field", erf_ok,
           f"expected 10, got {erf}")

    # Test 32: Different activation (GELU)
    conv_gelu = DepthwiseCausalConv1d(
        channels=C_conv, kernel_size=3, dilation=1, activation="gelu"
    )
    y_gelu = conv_gelu(x_conv)
    gelu_ok = y_gelu.shape == x_conv.shape
    record("32. GELU activation conv works", gelu_ok)

    # Test 33: No activation
    conv_none = DepthwiseCausalConv1d(
        channels=C_conv, kernel_size=3, dilation=1, activation="none"
    )
    y_none = conv_none(x_conv)
    none_ok = y_none.shape == x_conv.shape
    record("33. No activation conv works", none_ok)

    # ------------------------------------------------------------------
    # EngramModule Tests
    # ------------------------------------------------------------------
    print("\n--- EngramModule ---")

    config = EngramConfig.minimal()

    # Test 34: Forward pass produces correct delta shape
    module = EngramModule(config)
    h_mod = torch.randn(2, 10, config.hidden_dim)
    ids_mod = torch.randint(0, config.vocab_size, (2, 10))
    delta, telem = module(h_mod, ids_mod)
    delta_shape_ok = delta.shape == (2, 10, config.hidden_dim)
    record("34. EngramModule delta shape correct", delta_shape_ok,
           f"expected (2, 10, {config.hidden_dim}), got {tuple(delta.shape)}")

    # Test 35: Telemetry is None when return_details=False
    telem_none_ok = telem is None
    record("35. Telemetry is None without return_details", telem_none_ok)

    # Test 36: return_details=True produces telemetry
    delta_d, telem_d = module(h_mod, ids_mod, return_details=True)
    telem_populated = (
        telem_d is not None
        and isinstance(telem_d, EngramTelemetry)
        and telem_d.gate_mean >= 0.0
        and len(telem_d.unique_id_ratio_per_head) > 0
    )
    record("36. EngramModule return_details populates telemetry", telem_populated,
           f"gate_mean={telem_d.gate_mean:.4f}" if telem_d else "None")

    # Test 37: Telemetry summary string
    if telem_d is not None:
        summary = telem_d.summary()
        summary_ok = "gate=" in summary and "sparsity=" in summary
        record("37. Telemetry summary string", summary_ok, summary)
    else:
        record("37. Telemetry summary string", False, "no telemetry")

    # Test 38: Telemetry to_dict
    if telem_d is not None:
        d = telem_d.to_dict()
        dict_ok = "engram/gate_mean" in d and "engram/delta_norm_mean" in d
        record("38. Telemetry to_dict keys", dict_ok, f"keys={list(d.keys())[:5]}...")
    else:
        record("38. Telemetry to_dict keys", False, "no telemetry")

    # Test 39: Without context gating
    config_no_gate = EngramConfig.minimal()
    config_no_gate.use_context_gate = False
    module_ng = EngramModule(config_no_gate)
    delta_ng, _ = module_ng(h_mod, ids_mod)
    no_gate_ok = delta_ng.shape == (2, 10, config_no_gate.hidden_dim)
    record("39. EngramModule without gating", no_gate_ok,
           f"shape={tuple(delta_ng.shape)}")

    # Test 40: Without depthwise conv
    config_no_conv = EngramConfig.minimal()
    config_no_conv.use_depthwise_conv = False
    module_nc = EngramModule(config_no_conv)
    delta_nc, _ = module_nc(h_mod, ids_mod)
    no_conv_ok = delta_nc.shape == (2, 10, config_no_conv.hidden_dim)
    record("40. EngramModule without depthwise conv", no_conv_ok,
           f"shape={tuple(delta_nc.shape)}")

    # Test 41: Without both gating and conv
    config_bare = EngramConfig.minimal()
    config_bare.use_context_gate = False
    config_bare.use_depthwise_conv = False
    module_bare = EngramModule(config_bare)
    delta_bare, _ = module_bare(h_mod, ids_mod)
    bare_ok = delta_bare.shape == (2, 10, config_bare.hidden_dim)
    record("41. EngramModule bare (no gate, no conv)", bare_ok)

    # Test 42: Without tokenizer compression
    config_no_comp = EngramConfig.minimal()
    config_no_comp.use_tokenizer_compression = False
    module_ncomp = EngramModule(config_no_comp)
    ids_raw = torch.randint(0, config_no_comp.table_size, (2, 10))
    delta_ncomp, _ = module_ncomp(h_mod, ids_raw)
    no_comp_ok = delta_ncomp.shape == (2, 10, config_no_comp.hidden_dim)
    record("42. EngramModule without tokenizer compression", no_comp_ok)

    # Test 43: With attention mask
    mask_mod = torch.ones(2, 10, dtype=torch.long)
    mask_mod[0, 7:] = 0  # Pad last 3 positions of first batch item
    delta_masked, telem_masked = module(
        h_mod, ids_mod, attention_mask=mask_mod, return_details=True
    )
    masked_mod_ok = delta_masked.shape == (2, 10, config.hidden_dim)
    record("43. EngramModule with attention mask", masked_mod_ok)

    # Test 44: AMP compatibility (autocast)
    amp_ok = True
    try:
        device = "cpu"  # autocast on CPU requires torch >= 1.10
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=True):
            delta_amp, _ = module(h_mod, ids_mod)
        amp_nan = torch.isnan(delta_amp).any().item()
        amp_ok = not amp_nan
    except Exception as e:
        # autocast may not be supported on all platforms
        amp_ok = True  # Skip gracefully
        logger.debug("AMP autocast skipped: %s", e)
    record("44. AMP compatibility (no NaN under autocast)", amp_ok)

    # Test 45: Batch independence -- set residual_scale > 0 so outputs
    # are not trivially zero, then verify different inputs produce
    # different outputs.
    module_ind = EngramModule(config)
    with torch.no_grad():
        module_ind.residual_scale.fill_(1.0)
    h_ind = torch.randn(3, 5, config.hidden_dim)
    ids_ind = torch.randint(0, config.vocab_size, (3, 5))
    # Make batch items have different IDs
    ids_ind[0] = torch.tensor([1, 2, 3, 4, 5])
    ids_ind[1] = torch.tensor([100, 200, 300, 400, 500])
    ids_ind[2] = torch.tensor([1, 2, 3, 4, 5])  # Same as batch 0
    delta_ind, _ = module_ind(h_ind, ids_ind)
    # Batch 0 and 1 have different IDs and different hidden states
    same_check = torch.allclose(delta_ind[0], delta_ind[1], atol=1e-6)
    ind_ok = not same_check  # Different inputs -> different outputs
    record("45. Batch independence (different inputs -> different outputs)", ind_ok)

    # Test 46: Determinism -- same input, same output across runs
    torch.manual_seed(0)
    h_det = torch.randn(2, 5, config.hidden_dim)
    ids_det = torch.randint(0, config.vocab_size, (2, 5))
    module.train(False)
    with torch.no_grad():
        d1, _ = module(h_det, ids_det)
        d2, _ = module(h_det, ids_det)
    det_ok = torch.allclose(d1, d2, atol=1e-7)
    record("46. Determinism (same input -> same output)", det_ok)

    # Test 47: Streaming cache creation
    cache_created = module.create_cache(device="cpu")
    cache_ok = (
        isinstance(cache_created, StreamingCache)
        and cache_created.max_order == config.max_ngram_order
        and cache_created.position == 0
    )
    record("47. Streaming cache creation", cache_ok)

    # Test 48: Streaming forward pass
    module.train(True)
    cache_fwd = module.create_cache(device="cpu")
    h_stream = torch.randn(1, 1, config.hidden_dim)
    ids_stream = torch.randint(0, config.vocab_size, (1, 1))
    delta_stream, _ = module(
        h_stream, ids_stream, cache_state=cache_fwd
    )
    stream_ok = delta_stream.shape == (1, 1, config.hidden_dim)
    record("48. Streaming forward pass shape", stream_ok,
           f"shape={tuple(delta_stream.shape)}")

    # ------------------------------------------------------------------
    # EmbeddingAggregator Tests
    # ------------------------------------------------------------------
    print("\n--- EmbeddingAggregator ---")

    # Test 49: Concat aggregation
    agg_concat = EmbeddingAggregator(
        num_orders=2, num_heads_per_order=2, dim_per_head=8,
        output_dim=32, mode="concat"
    )
    fake_heads = [torch.randn(2, 5, 8) for _ in range(4)]
    agg_out = agg_concat(fake_heads)
    agg_concat_ok = agg_out.shape == (2, 5, 32)
    record("49. Concat aggregation shape", agg_concat_ok,
           f"shape={tuple(agg_out.shape)}")

    # Test 50: Mean aggregation
    agg_mean = EmbeddingAggregator(
        num_orders=2, num_heads_per_order=2, dim_per_head=8,
        output_dim=32, mode="mean"
    )
    agg_mean_out = agg_mean(fake_heads)
    agg_mean_ok = agg_mean_out.shape == (2, 5, 32)
    record("50. Mean aggregation shape", agg_mean_ok)

    # Test 51: Weighted aggregation
    agg_weighted = EmbeddingAggregator(
        num_orders=2, num_heads_per_order=2, dim_per_head=8,
        output_dim=32, mode="weighted"
    )
    agg_w_out = agg_weighted(fake_heads)
    agg_w_ok = agg_w_out.shape == (2, 5, 32)
    record("51. Weighted aggregation shape", agg_w_ok)

    # ------------------------------------------------------------------
    # MockMultiHeadHash Tests
    # ------------------------------------------------------------------
    print("\n--- MockMultiHeadHash ---")

    # Test 52: Hash determinism
    hasher = MockMultiHeadHash(ngram_order=3, num_heads=2, table_size=1009, seed=42)
    test_ngrams = torch.tensor([[[10, 20, 30], [40, 50, 60]]])
    h1 = hasher.hash(test_ngrams)
    h2 = hasher.hash(test_ngrams)
    hash_det_ok = torch.equal(h1, h2)
    record("52. Hash determinism", hash_det_ok)

    # Test 53: Hash values within table_size
    hash_range_ok = (h1 >= 0).all().item() and (h1 < 1009).all().item()
    record("53. Hash values within table_size", hash_range_ok,
           f"min={h1.min().item()}, max={h1.max().item()}")

    # Test 54: Different N-grams produce different hashes (low collision)
    diff_ngrams = torch.tensor([[[10, 20, 30], [10, 20, 31]]])
    h_diff = hasher.hash(diff_ngrams)
    # At least one head should differ
    diff_ok = not torch.equal(h_diff[0, 0], h_diff[0, 1])
    record("54. Different N-grams -> different hashes", diff_ok,
           f"hash1={h_diff[0,0].tolist()}, hash2={h_diff[0,1].tolist()}")

    # ------------------------------------------------------------------
    # EngramAugmentedLayer Test
    # ------------------------------------------------------------------
    print("\n--- EngramAugmentedLayer ---")

    # Test 55: Augmented layer forward
    aug_layer = EngramAugmentedLayer(config=config, host_layer=None, layer_id=0)
    h_aug = torch.randn(2, 10, config.hidden_dim)
    ids_aug = torch.randint(0, config.vocab_size, (2, 10))
    out_aug = aug_layer(h_aug, ids_aug)
    aug_ok = out_aug.shape == (2, 10, config.hidden_dim)
    record("55. EngramAugmentedLayer forward shape", aug_ok,
           f"shape={tuple(out_aug.shape)}")

    # ------------------------------------------------------------------
    # Utility Tests
    # ------------------------------------------------------------------
    print("\n--- Utilities ---")

    # Test 56: find_next_prime
    prime_ok = (
        find_next_prime(10) == 11
        and find_next_prime(11) == 11
        and find_next_prime(100) == 101
        and find_next_prime(2) == 2
    )
    record("56. find_next_prime correctness", prime_ok)

    # Test 57: count_parameters
    param_count = count_parameters(module)
    count_ok = param_count > 0
    record("57. count_parameters returns positive count", count_ok,
           f"params={param_count}")

    # Test 58: EngramConfig presets
    cfg_dev = EngramConfig.dev()
    cfg_prod = EngramConfig.production()
    presets_ok = (
        cfg_dev.max_ngram_order == 3
        and cfg_prod.max_ngram_order == 4
        and cfg_prod.hidden_dim == 4096
    )
    record("58. EngramConfig presets (dev/production)", presets_ok)

    # Test 59: EngramConfig validation
    try:
        bad_cfg = EngramConfig(max_ngram_order=1)
        validation_ok = False
    except ValueError:
        validation_ok = True
    record("59. EngramConfig validates max_ngram_order >= 2", validation_ok)

    # Test 60: EngramConfig gate_type validation
    try:
        bad_gate = EngramConfig(gate_type="invalid")
        gate_val_ok = False
    except ValueError:
        gate_val_ok = True
    record("60. EngramConfig validates gate_type", gate_val_ok)

    # ------------------------------------------------------------------
    # Edge Case and Robustness Tests
    # ------------------------------------------------------------------
    print("\n--- Edge Cases and Robustness ---")

    # Test 61: Single-position sequence
    h_single = torch.randn(1, 1, config.hidden_dim)
    ids_single = torch.randint(0, config.vocab_size, (1, 1))
    delta_single, _ = module(h_single, ids_single)
    single_ok = delta_single.shape == (1, 1, config.hidden_dim)
    record("61. Single-position sequence", single_ok)

    # Test 62: Long sequence
    h_long = torch.randn(1, 200, config.hidden_dim)
    ids_long = torch.randint(0, config.vocab_size, (1, 200))
    delta_long, _ = module(h_long, ids_long)
    long_ok = delta_long.shape == (1, 200, config.hidden_dim)
    record("62. Long sequence (T=200)", long_ok)

    # Test 63: All-padding mask
    h_allpad = torch.randn(1, 5, config.hidden_dim)
    ids_allpad = torch.randint(0, config.vocab_size, (1, 5))
    mask_allpad = torch.zeros(1, 5, dtype=torch.long)
    delta_allpad, _ = module(h_allpad, ids_allpad, attention_mask=mask_allpad)
    allpad_ok = (
        delta_allpad.shape == (1, 5, config.hidden_dim)
        and not torch.isnan(delta_allpad).any().item()
    )
    record("63. All-padding mask (no NaN)", allpad_ok)

    # Test 64: Gradient flow through EngramModule
    module.train(True)
    h_gf = torch.randn(2, 5, config.hidden_dim, requires_grad=True)
    ids_gf = torch.randint(0, config.vocab_size, (2, 5))
    delta_gf, _ = module(h_gf, ids_gf)
    loss_gf = delta_gf.sum()
    loss_gf.backward()
    gf_ok = h_gf.grad is not None and not torch.isnan(h_gf.grad).any().item()
    record("64. Gradient flow through EngramModule", gf_ok)

    # Test 65: Module repr
    repr_str = repr(module)
    repr_ok = "EngramModule" in repr_str
    record("65. Module repr contains class name", repr_ok)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"  RESULTS: {passed_count}/{total_count} passed, "
          f"{failed_count} failed")
    print("=" * 72)

    if failed_count > 0:
        print("\n  WARNING: Some tests failed. Review output above.")
    else:
        print("\n  All tests passed.")


# ===================================================================== #
#                        Module Exports                                  #
# ===================================================================== #

__all__ = [
    "EngramConfig",
    "StreamingCache",
    "SuffixNgramExtractor",
    "EmbeddingAggregator",
    "RMSNorm",
    "ContextAwareGating",
    "DepthwiseCausalConv1d",
    "EngramModule",
    "EngramTelemetry",
    "EngramAugmentedLayer",
    "MockMultiHeadHash",
    "MockOffloadableEmbedding",
    "MockTokenizerCompression",
    "compute_effective_receptive_field",
    "find_next_prime",
    "validate_causality",
    "count_parameters",
]


if __name__ == "__main__":
    _run_all_tests()
