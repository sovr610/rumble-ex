"""
patch_embed_template.py
=======================
PatchEmbed (2-D, images) and PatchEmbed3D (3-D video tubelet tokenization).

References:
    - references/tubelet-tokenization.md
    - SKILL.md § PatchEmbed3D, § Key Concepts / Tubelet Tokenization

Done-when Gate 1 (Tokenization):
    PatchEmbed3D converts [B, 3, 16, 224, 224] -> [B, 1568, D]
    Token count must equal (T/t) * (H/P) * (W/P).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# 2-D Image Patch Embedding
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """
    Standard 2-D image patch embedding for Vision Transformers.

    Applies a Conv2d with kernel_size == stride == patch_size to produce
    non-overlapping spatial patches which are then flattened into a sequence.

    Args:
        img_size:   Input image height = width (pixels). Must be divisible by
                    patch_size.
        patch_size: Side length (pixels) of each square patch.
        in_chans:   Number of input channels (3 for RGB).
        embed_dim:  Output embedding dimension D.

    Input:  Tensor[B, in_chans, H, W]
    Output: Tensor[B, N, D]  where N = (H/patch_size) * (W/patch_size)
    """

    def __init__(
        self,
        img_size:   int = 224,
        patch_size: int = 16,
        in_chans:   int = 3,
        embed_dim:  int = 768,
    ) -> None:
        super().__init__()

        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size ({img_size}) must be divisible by "
                f"patch_size ({patch_size})."
            )

        self.img_size   = img_size
        self.patch_size = patch_size
        self.embed_dim  = embed_dim

        # Grid of patches: (H_grid, W_grid)
        self.grid_size: Tuple[int, int] = (
            img_size // patch_size,
            img_size // patch_size,
        )
        self._num_patches: int = self.grid_size[0] * self.grid_size[1]

        # Single Conv2d performs the tokenisation
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Optional learnable norm (disabled by default for drop-in use)
        self.norm: Optional[nn.Module] = None

    @property
    def num_patches(self) -> int:
        """Total number of spatial patches per image."""
        return self._num_patches

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor[B, C, H, W]
        Returns:
            Tensor[B, N, D]  (N = H_grid * W_grid)
        """
        B, C, H, W = x.shape

        if H != self.img_size or W != self.img_size:
            raise ValueError(
                f"Input image size ({H}x{W}) does not match expected "
                f"({self.img_size}x{self.img_size})."
            )

        # [B, D, H_grid, W_grid]
        x = self.proj(x)
        # [B, D, N] -> [B, N, D]
        x = x.flatten(2).transpose(1, 2)

        if self.norm is not None:
            x = self.norm(x)

        return x

    def extra_repr(self) -> str:
        return (
            f"img_size={self.img_size}, patch_size={self.patch_size}, "
            f"num_patches={self._num_patches}, embed_dim={self.embed_dim}"
        )


# ---------------------------------------------------------------------------
# 3-D Video Tubelet Embedding
# ---------------------------------------------------------------------------

class PatchEmbed3D(nn.Module):
    """
    3-D tubelet patch embedding for video using a single Conv3d.

    Tokenizes a video tensor [B, C, T, H, W] into spatiotemporal patch
    embeddings [B, N, D] where N = (T/tubelet_size) * (H/patch_size) *
    (W/patch_size).

    The Conv3d kernel covers exactly one tubelet: (tubelet_size, patch_size,
    patch_size) pixels.  Setting stride == kernel_size produces non-overlapping
    tubelets.

    Args:
        img_size:     Spatial height = width (pixels). Must be divisible by
                      patch_size.
        frames:       Number of input frames T. Must be divisible by
                      tubelet_size.
        patch_size:   Spatial patch side length (pixels).
        tubelet_size: Number of frames per tubelet (temporal kernel depth).
        in_chans:     Input channels (3 for RGB).
        embed_dim:    Output embedding dimension D.

    Input:  Tensor[B, in_chans, T, H, W]
    Output: Tensor[B, N, D]
    """

    def __init__(
        self,
        img_size:     int = 224,
        frames:       int = 16,
        patch_size:   int = 16,
        tubelet_size: int = 2,
        in_chans:     int = 3,
        embed_dim:    int = 1024,
    ) -> None:
        super().__init__()

        if frames % tubelet_size != 0:
            raise ValueError(
                f"frames ({frames}) must be divisible by "
                f"tubelet_size ({tubelet_size})."
            )
        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size ({img_size}) must be divisible by "
                f"patch_size ({patch_size})."
            )

        self.img_size     = img_size
        self.frames       = frames
        self.patch_size   = patch_size
        self.tubelet_size = tubelet_size
        self.embed_dim    = embed_dim

        # Grid dimensions: (depth, height, width)
        self.grid_size: Tuple[int, int, int] = (
            frames   // tubelet_size,   # temporal depth
            img_size // patch_size,     # spatial height
            img_size // patch_size,     # spatial width
        )
        self._num_patches: int = (
            self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        )

        # Core tokenisation: one Conv3d maps a tubelet to an embedding vector
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

        # Optional layer norm after projection (e.g. for stable pre-training)
        self.norm: Optional[nn.Module] = None

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier-uniform init matching ViT convention."""
        fan_in  = self.proj.in_channels * math.prod(self.proj.kernel_size)
        fan_out = self.proj.out_channels
        std = math.sqrt(2.0 / (fan_in + fan_out))
        nn.init.normal_(self.proj.weight, mean=0.0, std=std)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    @property
    def num_patches(self) -> int:
        """Total number of spatiotemporal tokens per clip."""
        return self._num_patches

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor[B, C, T, H, W]  — raw video clip.
        Returns:
            Tensor[B, N, D]  — sequence of tubelet embeddings.

        Raises:
            ValueError: if T, H, or W are inconsistent with the configuration.
        """
        B, C, T, H, W = x.shape

        if T != self.frames:
            raise ValueError(
                f"Input has {T} frames but model expects {self.frames}."
            )
        if H != self.img_size or W != self.img_size:
            raise ValueError(
                f"Input spatial size ({H}x{W}) does not match expected "
                f"({self.img_size}x{self.img_size})."
            )

        # [B, D, T/t, H/P, W/P]
        x = self.proj(x)
        # [B, D, N] -> [B, N, D]
        x = x.flatten(2).transpose(1, 2)

        if self.norm is not None:
            x = self.norm(x)

        return x

    def extra_repr(self) -> str:
        d, h, w = self.grid_size
        return (
            f"img_size={self.img_size}, frames={self.frames}, "
            f"patch_size={self.patch_size}, tubelet_size={self.tubelet_size}, "
            f"grid_size=({d},{h},{w}), num_patches={self._num_patches}, "
            f"embed_dim={self.embed_dim}"
        )


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_patch_embed(
    img_size:   int = 224,
    patch_size: int = 16,
    embed_dim:  int = 768,
    in_chans:   int = 3,
) -> PatchEmbed:
    """Convenience factory for 2-D PatchEmbed."""
    return PatchEmbed(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
    )


def make_patch_embed_3d(
    img_size:     int = 224,
    frames:       int = 16,
    patch_size:   int = 16,
    tubelet_size: int = 2,
    embed_dim:    int = 1024,
    in_chans:     int = 3,
) -> PatchEmbed3D:
    """Convenience factory for 3-D PatchEmbed3D."""
    return PatchEmbed3D(
        img_size=img_size,
        frames=frames,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
    )


# ---------------------------------------------------------------------------
# Self-tests  (python patch_embed_template.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    PASSED = 0
    FAILED = 0

    def check(name: str, condition: bool, msg: str = "") -> None:
        global PASSED, FAILED
        if condition:
            print(f"  PASS  {name}")
            PASSED += 1
        else:
            print(f"  FAIL  {name}  {msg}")
            FAILED += 1

    print("=" * 60)
    print("PatchEmbed (2-D) tests")
    print("=" * 60)

    # --- 2-D tests -----------------------------------------------------------
    pe = PatchEmbed(img_size=224, patch_size=16, embed_dim=768)
    x2d = torch.randn(2, 3, 224, 224)
    out2d = pe(x2d)

    check("PE2D output shape [2,196,768]",
          out2d.shape == (2, 196, 768),
          f"got {out2d.shape}")
    check("PE2D num_patches == 196",
          pe.num_patches == 196)
    check("PE2D grid_size == (14,14)",
          pe.grid_size == (14, 14))

    # Batch size 4
    out_b4 = pe(torch.randn(4, 3, 224, 224))
    check("PE2D batch=4 output shape",
          out_b4.shape == (4, 196, 768),
          f"got {out_b4.shape}")

    # Larger patch
    pe32 = PatchEmbed(img_size=224, patch_size=32, embed_dim=1024)
    out32 = pe32(torch.randn(1, 3, 224, 224))
    check("PE2D patch32 shape [1,49,1024]",
          out32.shape == (1, 49, 1024),
          f"got {out32.shape}")
    check("PE2D patch32 num_patches == 49",
          pe32.num_patches == 49)

    # Error on wrong size
    try:
        pe_bad = PatchEmbed(img_size=225, patch_size=16)
        check("PE2D bad img_size raises ValueError", False, "no error raised")
    except ValueError:
        check("PE2D bad img_size raises ValueError", True)

    print()
    print("=" * 60)
    print("PatchEmbed3D tests")
    print("=" * 60)

    # --- 3-D tests -----------------------------------------------------------
    pe3d = PatchEmbed3D(
        img_size=224, frames=16, patch_size=16, tubelet_size=2, embed_dim=1024
    )
    x3d = torch.randn(2, 3, 16, 224, 224)
    out3d = pe3d(x3d)

    check("PE3D output shape [2,1568,1024]",
          out3d.shape == (2, 1568, 1024),
          f"got {out3d.shape}")
    check("PE3D num_patches == 1568",
          pe3d.num_patches == 1568)
    check("PE3D grid_size == (8,14,14)",
          pe3d.grid_size == (8, 14, 14))

    # 256-pixel test (canonical: 2048 tokens)
    pe3d_256 = PatchEmbed3D(
        img_size=256, frames=16, patch_size=16, tubelet_size=2, embed_dim=1024
    )
    out256 = pe3d_256(torch.randn(1, 3, 16, 256, 256))
    check("PE3D 256px shape [1,2048,1024]",
          out256.shape == (1, 2048, 1024),
          f"got {out256.shape}")
    check("PE3D 256px num_patches == 2048",
          pe3d_256.num_patches == 2048)

    # 8-frame clip
    pe3d_8f = PatchEmbed3D(
        img_size=224, frames=8, patch_size=16, tubelet_size=2, embed_dim=768
    )
    out8f = pe3d_8f(torch.randn(4, 3, 8, 224, 224))
    check("PE3D 8-frame shape [4,784,768]",
          out8f.shape == (4, 784, 768),
          f"got {out8f.shape}")

    # tubelet=4
    pe3d_t4 = PatchEmbed3D(
        img_size=224, frames=16, patch_size=16, tubelet_size=4, embed_dim=1024
    )
    out_t4 = pe3d_t4(torch.randn(1, 3, 16, 224, 224))
    check("PE3D tubelet=4 shape [1,784,1024]",
          out_t4.shape == (1, 784, 1024),
          f"got {out_t4.shape}")

    # Error: frames not divisible by tubelet
    try:
        pe3d_bad = PatchEmbed3D(frames=15, tubelet_size=2)
        check("PE3D bad frames raises ValueError", False, "no error raised")
    except ValueError:
        check("PE3D bad frames raises ValueError", True)

    # Error: img_size not divisible by patch
    try:
        pe3d_bad2 = PatchEmbed3D(img_size=225, patch_size=16)
        check("PE3D bad img_size raises ValueError", False, "no error raised")
    except ValueError:
        check("PE3D bad img_size raises ValueError", True)

    # Wrong frame count at forward time
    try:
        pe3d(torch.randn(1, 3, 8, 224, 224))
        check("PE3D wrong T at forward raises ValueError", False, "no error raised")
    except ValueError:
        check("PE3D wrong T at forward raises ValueError", True)

    print()
    print("=" * 60)
    print(f"Results: {PASSED} passed, {FAILED} failed")
    print("=" * 60)

    sys.exit(0 if FAILED == 0 else 1)
