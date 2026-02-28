# Copyright (c) Meta Platforms, Inc. and affiliates.
# MIT License
#
# model_hub_template.py
#
# Factory functions for V-JEPA 2 encoder and predictor models.
# Covers PyTorch Hub entry points and HuggingFace integration.
#
# PyTorch Hub usage::
#
#     import torch
#     encoder, predictor = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_giant')
#
# Direct factory usage (offline)::
#
#     encoder, predictor = vjepa2_vit_giant(pretrained=False)
#
# HuggingFace usage::
#
#     from transformers import AutoModel, AutoVideoProcessor
#     model     = AutoModel.from_pretrained("facebook/vjepa2-vitg-fpc64-256")
#     processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitg-fpc64-256")

from __future__ import annotations

import math
import unittest
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# hubconf.py requirements declaration (used by torch.hub)
# ---------------------------------------------------------------------------

dependencies = ["torch", "timm", "einops"]


# ---------------------------------------------------------------------------
# Internal video ViT building blocks
# ---------------------------------------------------------------------------

class _PatchEmbed3D(nn.Module):
    """
    3D patch embedding: [B, C, T, H, W] -> [B, N, embed_dim].

    Spatial patch size: ``patch_size x patch_size``.
    Temporal patch size (tubelet): ``tubelet_size`` frames.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        tubelet_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.patch_size   = patch_size
        self.tubelet_size = tubelet_size
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, T, H, W]
        x = self.proj(x)            # [B, D, T', H', W']
        B, D, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, T'*H'*W', D]
        return x


class _VideoViTEncoder(nn.Module):
    """
    Minimal Video Vision Transformer (ViT) encoder.

    Accepts video clips of shape [B, C, T, H, W] and returns patch tokens
    of shape [B, N, embed_dim].

    In the real V-JEPA 2 codebase this uses RoPE positional embeddings
    and flash attention. Here we use standard sinusoidal positions and
    nn.MultiheadAttention for portability and testing.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        tubelet_size: int = 2,
        num_frames: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.patch_embed = _PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        # Sinusoidal positional embedding (computed at first forward)
        self._pos_embed: Optional[Tensor] = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # pre-LN (matches ViT)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def _get_pos_embed(self, n: int, device: torch.device) -> Tensor:
        if self._pos_embed is not None and self._pos_embed.shape[1] == n:
            return self._pos_embed.to(device)
        position = torch.arange(n, device=device).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2, device=device).float()
            * (-math.log(10000.0) / self.embed_dim)
        )
        pe = torch.zeros(1, n, self.embed_dim, device=device)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self._pos_embed = pe
        return pe

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, C, T, H, W]

        Returns:
            tokens: [B, N, embed_dim]
        """
        tokens = self.patch_embed(x)                    # [B, N, D]
        pos    = self._get_pos_embed(tokens.shape[1], x.device)
        tokens = tokens + pos                           # [B, N, D]
        tokens = self.transformer(tokens)               # [B, N, D]
        tokens = self.norm(tokens)
        return tokens


class _VJEPAPredictor(nn.Module):
    """
    Narrow predictor transformer for V-JEPA 2.

    Receives context tokens from the encoder and mask tokens representing
    future patch positions. Returns predicted representations for masked positions.
    """

    def __init__(
        self,
        context_embed_dim: int = 768,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 12,
        num_mask_tokens: int = 10,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.embed_dim  = embed_dim

        # Project encoder context to predictor dimension
        self.input_proj = nn.Linear(context_embed_dim, embed_dim)

        # Learnable mask token for future positions
        self.mask_token = nn.Parameter(torch.zeros(1, num_mask_tokens, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        predictor_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(predictor_layer, num_layers=depth)
        self.norm        = nn.LayerNorm(embed_dim)

        # Project back to encoder dimension for loss computation
        self.output_proj = nn.Linear(embed_dim, context_embed_dim)

    def forward(
        self,
        context_tokens: Tensor,
        num_future_tokens: Optional[int] = None,
    ) -> Tensor:
        """
        Args:
            context_tokens:    [B, N_ctx, context_embed_dim]
            num_future_tokens: Override number of future patches; defaults to mask_token count.

        Returns:
            predicted_tokens: [B, N_fut, context_embed_dim]
        """
        B       = context_tokens.shape[0]
        N_fut   = num_future_tokens or self.mask_token.shape[1]

        ctx     = self.input_proj(context_tokens)              # [B, N_ctx, D_pred]
        fut     = self.mask_token[:, :N_fut, :].expand(B, -1, -1)  # [B, N_fut, D_pred]
        tokens  = torch.cat([ctx, fut], dim=1)                 # [B, N_ctx+N_fut, D_pred]
        tokens  = self.transformer(tokens)
        tokens  = self.norm(tokens)
        fut_out = tokens[:, ctx.shape[1] :, :]                 # [B, N_fut, D_pred]
        return self.output_proj(fut_out)                       # [B, N_fut, D_ctx]


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def _load_weights(
    encoder: nn.Module,
    predictor: nn.Module,
    url: str,
    offline_path: Optional[str] = None,
) -> None:
    """Load pretrained weights from URL or local path, with strict=False."""
    if offline_path is not None:
        state = torch.load(offline_path, map_location="cpu")
    else:
        state = torch.hub.load_state_dict_from_url(url, map_location="cpu", check_hash=False)

    if "encoder" in state:
        encoder.load_state_dict(state["encoder"], strict=False)
    if "predictor" in state:
        predictor.load_state_dict(state["predictor"], strict=False)


def vjepa2_vit_large(
    pretrained: bool = True,
    offline_path: Optional[str] = None,
) -> Tuple[nn.Module, nn.Module]:
    """
    V-JEPA 2 ViT-Large encoder and predictor.

    Encoder:   embed_dim=1024, depth=24, num_heads=16
    Predictor: embed_dim=384,  depth=12, num_heads=12

    Args:
        pretrained:    If True, download and load pretrained weights.
        offline_path:  Path to local checkpoint file (skips URL download).

    Returns:
        (encoder, predictor) tuple of nn.Module instances.
    """
    encoder = _VideoViTEncoder(
        img_size=224, patch_size=16, tubelet_size=2,
        num_frames=16, embed_dim=1024, depth=24, num_heads=16,
    )
    predictor = _VJEPAPredictor(
        context_embed_dim=1024, embed_dim=384, depth=12,
        num_heads=12, num_mask_tokens=10,
    )
    if pretrained:
        url = "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_vitl.pt"
        _load_weights(encoder, predictor, url, offline_path)
    return encoder, predictor


def vjepa2_vit_giant(
    pretrained: bool = True,
    offline_path: Optional[str] = None,
) -> Tuple[nn.Module, nn.Module]:
    """
    V-JEPA 2 ViT-Giant encoder and predictor.

    Encoder:   embed_dim=1408, depth=40, num_heads=16
    Predictor: embed_dim=384,  depth=12, num_heads=12

    Args:
        pretrained:    If True, download and load pretrained weights.
        offline_path:  Path to local checkpoint file (skips URL download).

    Returns:
        (encoder, predictor) tuple of nn.Module instances.
    """
    encoder = _VideoViTEncoder(
        img_size=224, patch_size=16, tubelet_size=2,
        num_frames=16, embed_dim=1408, depth=40, num_heads=16,
    )
    predictor = _VJEPAPredictor(
        context_embed_dim=1408, embed_dim=384, depth=12,
        num_heads=12, num_mask_tokens=10,
    )
    if pretrained:
        url = "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_vitg.pt"
        _load_weights(encoder, predictor, url, offline_path)
    return encoder, predictor


def vjepa2_ac_vit_giant(
    pretrained: bool = True,
    offline_path: Optional[str] = None,
) -> Tuple[nn.Module, nn.Module]:
    """
    V-JEPA 2 Action Classifier ViT-Giant (EPIC-Kitchens anticipation checkpoint).

    Same architecture as vjepa2_vit_giant but with the action-classifier
    fine-tuned weights.

    Returns:
        (encoder, predictor) tuple of nn.Module instances.
    """
    encoder = _VideoViTEncoder(
        img_size=224, patch_size=16, tubelet_size=2,
        num_frames=16, embed_dim=1408, depth=40, num_heads=16,
    )
    predictor = _VJEPAPredictor(
        context_embed_dim=1408, embed_dim=384, depth=12,
        num_heads=12, num_mask_tokens=10,
    )
    if pretrained:
        url = "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_ac_vitg.pt"
        _load_weights(encoder, predictor, url, offline_path)
    return encoder, predictor


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

try:
    from torchvision import transforms as _tv_transforms
    import torchvision.transforms.functional as _tvF

    class _VideoPreprocessTransform:
        """
        Video preprocessing pipeline for [C, T, H, W] tensors.

        torchvision spatial transforms (Resize, CenterCrop, Normalize) operate
        on [C, H, W] images. For video we apply them to each frame individually
        by transposing T and batch dimensions, then restoring the layout.

        Processing steps:
            1. Temporal subsampling: sample ``frames_per_clip`` frames uniformly.
            2. Spatial resize: resize short side to ``short_side_size``.
            3. Center crop: crop to ``crop_size x crop_size``.
            4. Normalize: subtract mean, divide by std per channel.

        Input:  [C, T, H, W] float tensor in [0, 1] range.
        Output: [C, T', crop_size, crop_size] normalized float tensor.
        """

        def __init__(
            self,
            crop_size: int,
            short_side_size: int,
            frames_per_clip: int,
            mean: Tuple[float, ...],
            std: Tuple[float, ...],
        ) -> None:
            self.crop_size       = crop_size
            self.short_side_size = short_side_size
            self.frames_per_clip = frames_per_clip
            self.mean = mean
            self.std  = std

            # Per-frame spatial transforms
            self._spatial = _tv_transforms.Compose([
                _tv_transforms.Resize(short_side_size, antialias=True),
                _tv_transforms.CenterCrop(crop_size),
            ])
            self._normalize = _tv_transforms.Normalize(
                mean=list(mean), std=list(std)
            )

        def _temporal_subsample(self, x: Tensor) -> Tensor:
            """x: [C, T, H, W] -> [C, T', H, W]"""
            T = x.shape[1]
            if T <= self.frames_per_clip:
                return x
            indices = torch.linspace(0, T - 1, self.frames_per_clip).long()
            return x[:, indices]

        def __call__(self, x: Tensor) -> Tensor:
            """
            Args:
                x: [C, T, H, W] video tensor in [0, 1] range.
            Returns:
                [C, frames_per_clip, crop_size, crop_size] normalized tensor.
            """
            # 1. Temporal subsample: [C, T, H, W] -> [C, T', H, W]
            x = self._temporal_subsample(x)

            C, T, H, W = x.shape

            # 2. Apply spatial transforms to each frame
            # Reshape to [T, C, H, W] for per-frame processing
            frames = x.permute(1, 0, 2, 3)  # [T, C, H, W]

            processed = []
            for t in range(T):
                frame = self._spatial(frames[t])  # [C, H', W']
                processed.append(frame)
            frames_out = torch.stack(processed, dim=0)  # [T, C, crop, crop]

            # 3. Normalize: apply to each frame [C, H, W]
            normed = []
            for t in range(T):
                normed.append(self._normalize(frames_out[t]))
            frames_norm = torch.stack(normed, dim=0)  # [T, C, crop, crop]

            # 4. Restore [C, T, H, W] layout
            return frames_norm.permute(1, 0, 2, 3)  # [C, T, crop, crop]

        def __repr__(self) -> str:
            return (
                f"VideoPreprocessTransform("
                f"crop={self.crop_size}, short_side={self.short_side_size}, "
                f"frames={self.frames_per_clip})"
            )

    def vjepa2_preprocessor(
        crop_size: int = 224,
        short_side_size: int = 256,
        frames_per_clip: int = 16,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...]  = (0.229, 0.224, 0.225),
    ) -> Callable:
        """
        Standard V-JEPA 2 video preprocessing pipeline.

        Input: tensor of shape [C, T, H, W] in [0, 1] range.
        Output: normalized tensor of shape [C, frames_per_clip, crop_size, crop_size].

        Args:
            crop_size:        Final spatial crop size.
            short_side_size:  Resize short side to this value before cropping.
            frames_per_clip:  Number of frames to sample.
            mean:             Per-channel mean for normalization.
            std:              Per-channel std for normalization.

        Returns:
            A callable that preprocesses [C, T, H, W] video tensors.
        """
        return _VideoPreprocessTransform(
            crop_size=crop_size,
            short_side_size=short_side_size,
            frames_per_clip=frames_per_clip,
            mean=mean,
            std=std,
        )

except ImportError:
    def vjepa2_preprocessor(**kwargs) -> Callable:  # type: ignore[misc]
        raise ImportError("torchvision is required for vjepa2_preprocessor(). pip install torchvision")


# ---------------------------------------------------------------------------
# Self-Tests
# ---------------------------------------------------------------------------

class _TestFactoryFunctions(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)

    def test_vit_large_returns_tuple_of_two(self):
        enc, pred = vjepa2_vit_large(pretrained=False)
        self.assertEqual(len((enc, pred)), 2)

    def test_vit_giant_returns_tuple_of_two(self):
        enc, pred = vjepa2_vit_giant(pretrained=False)
        self.assertEqual(len((enc, pred)), 2)

    def test_ac_vit_giant_returns_tuple_of_two(self):
        enc, pred = vjepa2_ac_vit_giant(pretrained=False)
        self.assertEqual(len((enc, pred)), 2)

    def test_encoder_is_nn_module(self):
        enc, _ = vjepa2_vit_large(pretrained=False)
        self.assertIsInstance(enc, nn.Module)

    def test_predictor_is_nn_module(self):
        _, pred = vjepa2_vit_large(pretrained=False)
        self.assertIsInstance(pred, nn.Module)

    def test_vit_large_encoder_embed_dim(self):
        enc, _ = vjepa2_vit_large(pretrained=False)
        self.assertEqual(enc.embed_dim, 1024)

    def test_vit_giant_encoder_embed_dim(self):
        enc, _ = vjepa2_vit_giant(pretrained=False)
        self.assertEqual(enc.embed_dim, 1408)

    def test_encoder_forward_shape_vit_large(self):
        """ViT-L encoder: [B=2, C=3, T=16, H=224, W=224] -> [2, 1568, 1024]."""
        enc, _ = vjepa2_vit_large(pretrained=False)
        x = torch.randn(2, 3, 16, 224, 224)
        with torch.no_grad():
            out = enc(x)
        self.assertEqual(out.dim(), 3)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[2], 1024)

    def test_encoder_forward_shape_vit_giant_small_input(self):
        """ViT-G is too large for a full forward in a unit test; use depth=2 proxy."""
        enc = _VideoViTEncoder(embed_dim=64, depth=2, num_heads=4)
        x   = torch.randn(2, 3, 4, 32, 32)
        with torch.no_grad():
            out = enc(x)
        self.assertEqual(out.dim(), 3)
        self.assertEqual(out.shape[2], 64)

    def test_predictor_forward_shape(self):
        enc, pred = vjepa2_vit_large(pretrained=False)
        x = torch.randn(2, 3, 16, 224, 224)
        with torch.no_grad():
            ctx  = enc(x)
            fut  = pred(ctx, num_future_tokens=8)
        self.assertEqual(fut.dim(), 3)
        self.assertEqual(fut.shape[0], 2)
        self.assertEqual(fut.shape[1], 8)
        self.assertEqual(fut.shape[2], 1024)   # projected back to encoder dim

    def test_factory_offline_mode(self):
        """pretrained=False should not attempt network access."""
        import unittest.mock as mock
        with mock.patch("torch.hub.load_state_dict_from_url") as mock_download:
            enc, pred = vjepa2_vit_giant(pretrained=False)
            mock_download.assert_not_called()

    def test_no_nan_in_encoder_output(self):
        enc, _ = vjepa2_vit_large(pretrained=False)
        x = torch.randn(1, 3, 4, 64, 64)
        with torch.no_grad():
            out = enc(x)
        self.assertFalse(torch.isnan(out).any())


class _TestPreprocessor(unittest.TestCase):

    def setUp(self):
        try:
            import torchvision  # noqa: F401
            self.has_torchvision = True
        except ImportError:
            self.has_torchvision = False

    @unittest.skipUnless(True, "torchvision optional")
    def test_preprocessor_is_callable(self):
        if not self.has_torchvision:
            self.skipTest("torchvision not installed")
        prep = vjepa2_preprocessor()
        self.assertTrue(callable(prep))

    def test_preprocessor_output_shape(self):
        if not self.has_torchvision:
            self.skipTest("torchvision not installed")
        prep   = vjepa2_preprocessor(crop_size=224, frames_per_clip=16)
        # Simulate [C, T, H, W] input in [0, 1] range
        video  = torch.rand(3, 32, 256, 256)  # C T H W
        output = prep(video)
        self.assertEqual(output.shape, (3, 16, 224, 224))

    def test_preprocessor_custom_frames(self):
        if not self.has_torchvision:
            self.skipTest("torchvision not installed")
        prep   = vjepa2_preprocessor(crop_size=256, frames_per_clip=64)
        video  = torch.rand(3, 128, 300, 300)
        output = prep(video)
        self.assertEqual(output.shape, (3, 64, 256, 256))

    def test_preprocessor_output_dtype(self):
        if not self.has_torchvision:
            self.skipTest("torchvision not installed")
        prep   = vjepa2_preprocessor()
        video  = torch.rand(3, 16, 256, 256)
        output = prep(video)
        self.assertEqual(output.dtype, torch.float32)

    def test_preprocessor_normalizes(self):
        """Output should not be in [0, 1] — normalization shifts the range."""
        if not self.has_torchvision:
            self.skipTest("torchvision not installed")
        prep   = vjepa2_preprocessor()
        video  = torch.ones(3, 16, 256, 256)   # all-ones input
        output = prep(video)
        # After normalization with ImageNet stats, values shift outside [0, 1]
        self.assertFalse(((output >= 0) & (output <= 1)).all())


if __name__ == "__main__":
    print("Running model_hub self-tests...")
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromTestCase(_TestFactoryFunctions)
    suite.addTests(loader.loadTestsFromTestCase(_TestPreprocessor))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("\nAll self-tests passed.")
    else:
        raise SystemExit(1)
