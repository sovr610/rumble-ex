# brain_ai/encoders/engram_encoder.py
"""
Engram Text Encoder.

Phase 1 integration: Engram as a fast text encoder that competes
with the standard transformer-based encoder in the Global Workspace.
"""

import torch
import torch.nn as nn
from typing import Optional

from ..memory.engram import EngramEmbedding, EngramConfig


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
