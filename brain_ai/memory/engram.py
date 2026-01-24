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
