"""
Text Encoder

Transformer-based text encoder with spike output for Global Workspace.
Supports both learned embeddings and pretrained models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TextEncoder(nn.Module):
    """
    Transformer-based Text Encoder.

    Encodes text sequences into fixed-size representations
    for the Global Workspace.

    Architecture:
        Token IDs -> Embedding -> Positional Encoding ->
        Transformer Encoder -> Pooling -> Projection -> Output

    Args:
        vocab_size: Vocabulary size
        embed_dim: Embedding dimension
        output_dim: Output feature dimension (default 512)
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        ff_dim: Feedforward hidden dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout rate
        pooling: Pooling strategy ('cls', 'mean', 'max')
    """

    def __init__(
        self,
        vocab_size: int = 30000,
        embed_dim: int = 256,
        output_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 4,
        ff_dim: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pooling: str = "mean",
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.pooling = pooling

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # CLS token for classification pooling
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_len, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output projection
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

        # Layer norm
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len), 1 for valid, 0 for padding

        Returns:
            features: (batch, output_dim)
        """
        batch_size, seq_len = input_ids.shape

        # Embed tokens
        x = self.embedding(input_ids)  # (batch, seq, embed_dim)

        # Add CLS token if using CLS pooling
        if self.pooling == "cls":
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            if attention_mask is not None:
                cls_mask = torch.ones(batch_size, 1, device=attention_mask.device)
                attention_mask = torch.cat([cls_mask, attention_mask], dim=1)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Create attention mask for transformer (True = ignore)
        if attention_mask is not None:
            # Transformer expects (batch, seq) with True for positions to mask
            src_key_padding_mask = attention_mask == 0
        else:
            src_key_padding_mask = None

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Pooling
        if self.pooling == "cls":
            pooled = x[:, 0, :]  # CLS token
        elif self.pooling == "mean":
            if attention_mask is not None:
                # Masked mean pooling
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                pooled = x.mean(dim=1)
        elif self.pooling == "max":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                x = x.masked_fill(mask == 0, float('-inf'))
            pooled = x.max(dim=1).values
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        # Project to output dimension
        features = self.projection(pooled)
        features = self.layer_norm(features)

        return features


class SpikeTextEncoder(nn.Module):
    """
    Text encoder with spike-based output.

    Wraps TextEncoder and adds rate-coded spike generation
    for compatibility with SNN downstream processing.
    """

    def __init__(
        self,
        vocab_size: int = 30000,
        embed_dim: int = 256,
        output_dim: int = 512,
        num_steps: int = 25,
        **kwargs
    ):
        super().__init__()

        self.num_steps = num_steps
        self.encoder = TextEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            output_dim=output_dim,
            **kwargs
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_spikes: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Optional attention mask
            return_spikes: If True, return spike train instead of features

        Returns:
            features: (batch, output_dim) or spikes (time, batch, output_dim)
        """
        features = self.encoder(input_ids, attention_mask)

        if return_spikes:
            # Convert to spike train via rate coding
            # Normalize to [0, 1] for spike probability
            probs = torch.sigmoid(features)
            spikes = torch.rand(
                self.num_steps, *probs.shape, device=probs.device
            ) < probs
            return spikes.float()

        return features


class CharacterTextEncoder(nn.Module):
    """
    Character-level text encoder.

    Processes raw characters instead of tokens.
    More robust to typos and unknown words.
    """

    def __init__(
        self,
        output_dim: int = 512,
        embed_dim: int = 128,
        num_chars: int = 256,  # ASCII + extended
        kernel_sizes: list = [3, 4, 5],
        num_filters: int = 100,
        max_seq_len: int = 512,
    ):
        super().__init__()

        self.embedding = nn.Embedding(num_chars, embed_dim)

        # Parallel conv filters with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k)
            for k in kernel_sizes
        ])

        # Projection
        self.projection = nn.Sequential(
            nn.Linear(len(kernel_sizes) * num_filters, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Character codes (batch, seq_len)

        Returns:
            features: (batch, output_dim)
        """
        # Embed characters
        embedded = self.embedding(x)  # (batch, seq, embed)
        embedded = embedded.transpose(1, 2)  # (batch, embed, seq)

        # Apply convolutions and pool
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))  # (batch, filters, seq')
            pooled = F.adaptive_max_pool1d(conv_out, 1).squeeze(-1)
            conv_outputs.append(pooled)

        # Concatenate and project
        combined = torch.cat(conv_outputs, dim=1)
        return self.projection(combined)


class HybridTextEncoder(nn.Module):
    """
    Hybrid encoder combining token-level and character-level representations.

    Provides robustness of character-level with efficiency of token-level.
    """

    def __init__(
        self,
        vocab_size: int = 30000,
        output_dim: int = 512,
        token_dim: int = 256,
        char_dim: int = 128,
    ):
        super().__init__()

        self.token_encoder = TextEncoder(
            vocab_size=vocab_size,
            embed_dim=token_dim,
            output_dim=output_dim // 2,
        )

        self.char_encoder = CharacterTextEncoder(
            output_dim=output_dim // 2,
            embed_dim=char_dim,
        )

        self.fusion = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        char_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: Token IDs (batch, seq_len)
            char_ids: Character codes (batch, char_seq_len)
            attention_mask: Token attention mask
        """
        token_features = self.token_encoder(input_ids, attention_mask)
        char_features = self.char_encoder(char_ids)

        combined = torch.cat([token_features, char_features], dim=-1)
        return self.fusion(combined)


# Factory function
def create_text_encoder(
    encoder_type: str = "standard",
    **kwargs
) -> nn.Module:
    """
    Create text encoder by type.

    Args:
        encoder_type: 'standard', 'spike', 'character', or 'hybrid'
    """
    if encoder_type == "standard":
        return TextEncoder(**kwargs)
    elif encoder_type == "spike":
        return SpikeTextEncoder(**kwargs)
    elif encoder_type == "character":
        return CharacterTextEncoder(**kwargs)
    elif encoder_type == "hybrid":
        return HybridTextEncoder(**kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
