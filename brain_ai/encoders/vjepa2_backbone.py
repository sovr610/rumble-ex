"""
V-JEPA 2 Vision Encoder Backbone

Wraps the frozen V-JEPA 2 ViT-g encoder as a drop-in replacement
for VisionEncoder, producing (B, output_dim) features compatible
with the Global Workspace pipeline.

When pretrained=False, uses a lightweight mock backbone for testing.
When pretrained=True, loads weights via HuggingFace transformers.
"""

import torch
import torch.nn as nn
from typing import Tuple

from ..config import VJEPA2EncoderConfig


class MockViTBackbone(nn.Module):
    """Lightweight ViT mock for testing without pretrained weights."""

    def __init__(self, encoder_dim: int = 64, patch_size: int = 16):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            3, encoder_dim, kernel_size=patch_size, stride=patch_size,
        )
        self.norm = nn.LayerNorm(encoder_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.patch_embed(x)
        B, D, H, W = tokens.shape
        tokens = tokens.flatten(2).transpose(1, 2)
        tokens = self.norm(tokens)
        positions = torch.zeros(tokens.shape[1], dtype=torch.long, device=x.device)
        return tokens, positions


class VJEPA2Backbone(nn.Module):
    """Frozen V-JEPA 2 ViT-g backbone. All parameters frozen."""

    def __init__(self, cfg: VJEPA2EncoderConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.pretrained:
            self._load_pretrained(cfg.model_name)
        else:
            self.model = MockViTBackbone(encoder_dim=cfg.encoder_dim)

        for p in self.parameters():
            p.requires_grad = False

    def _load_pretrained(self, model_name: str):
        try:
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(f"facebook/{model_name}")
        except (ImportError, OSError):
            import warnings
            warnings.warn(f"Could not load pretrained {model_name}. Falling back to mock.")
            self.model = MockViTBackbone(encoder_dim=self.cfg.encoder_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.model(x)


class AttentivePooler(nn.Module):
    """Cross-attention pooler with learnable query tokens."""

    def __init__(self, input_dim: int, num_queries: int = 16,
                 num_layers: int = 4, num_heads: int = 16):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, input_dim) * 0.02)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=input_dim, nhead=num_heads,
                dim_feedforward=input_dim * 4,
                batch_first=True, norm_first=True,
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, encoder_tokens: torch.Tensor) -> torch.Tensor:
        B = encoder_tokens.shape[0]
        queries = self.queries.expand(B, -1, -1)
        for layer in self.layers:
            queries = layer(queries, encoder_tokens)
        return self.norm(queries)


class VJEPA2VisionEncoder(nn.Module):
    """V-JEPA 2 vision encoder adapter for BrainAI pipeline.
    Drop-in replacement for VisionEncoder."""

    def __init__(self, cfg: VJEPA2EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.output_dim = cfg.output_dim
        self.backbone = VJEPA2Backbone(cfg)
        self.pooler = AttentivePooler(
            input_dim=cfg.encoder_dim, num_queries=cfg.num_query_tokens,
            num_layers=cfg.probe_layers, num_heads=cfg.probe_heads,
        )
        self.projection = nn.Sequential(
            nn.Linear(cfg.encoder_dim, cfg.output_dim),
            nn.GELU(),
            nn.Linear(cfg.output_dim, cfg.output_dim),
        )

    def get_raw_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.backbone(x)

    def forward(self, x: torch.Tensor, temporal_input: bool = False) -> torch.Tensor:
        if temporal_input:
            T, B = x.shape[:2]
            x = x[T // 2]
        tokens, positions = self.backbone(x)
        pooled = self.pooler(tokens)
        pooled = pooled.mean(dim=1)
        features = self.projection(pooled)
        return features
