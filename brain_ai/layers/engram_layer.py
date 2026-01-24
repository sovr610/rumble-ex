# brain_ai/layers/engram_layer.py
"""
Engram-Augmented Layer.

Phase 2 integration: Engram embedded at layer 1 in a transformer-style
pipeline. Follows the brain-inspired architecture:
    Engram -> SNN -> Attention -> FFN
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
