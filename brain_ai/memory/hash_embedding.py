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
            # Store weights in CPU memory
            self.weight = nn.Parameter(
                torch.zeros(num_embeddings, embedding_dim),
                requires_grad=True
            )
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
        """Prefetch embeddings for given indices asynchronously."""
        if not self.offload or not self.prefetch:
            return

        with torch.cuda.stream(self._prefetch_stream):
            flat_indices = indices.flatten()
            unique_indices = flat_indices.unique()
            cpu_indices = unique_indices.cpu()
            embeddings = self.weight[cpu_indices].cuda(non_blocking=True)
            self._prefetch_buffer = (unique_indices, embeddings)

    def _gather_from_cache(
        self,
        indices: torch.Tensor,
        cached_indices: torch.Tensor,
        cached_embeddings: torch.Tensor,
    ) -> torch.Tensor:
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

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Look up embeddings for given indices."""
        if self.offload:
            if self._prefetch_buffer is not None and self.prefetch:
                torch.cuda.current_stream().wait_stream(self._prefetch_stream)
                cached_indices, cached_embeddings = self._prefetch_buffer
                result = self._gather_from_cache(indices, cached_indices, cached_embeddings)
                self._prefetch_buffer = None
                return result
            else:
                # Synchronous fallback
                cpu_indices = indices.cpu()
                embeddings = self.weight[cpu_indices]
                if indices.is_cuda:
                    embeddings = embeddings.cuda()
                return embeddings
        else:
            return self.embedding(indices)
