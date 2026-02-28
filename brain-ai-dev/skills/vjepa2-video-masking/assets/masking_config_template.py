"""
masking_config_template.py
==========================
MaskConfig and VideoConfig dataclasses.
apply_masks convenience wrapper.
make_mask_generators factory.

References:
    - SKILL.md § Configuration Surface, § MaskGenerator
    - references/masking-algorithm.md

All public names are importable:
    from masking_config_template import (
        MaskConfig, VideoConfig, apply_masks, make_mask_generators
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MaskConfig:
    """
    Configuration for spatiotemporal block mask generation.

    Attributes:
        spatial_scale_small:         (min, max) fraction of H*W for small blocks.
        spatial_scale_large:         (min, max) fraction of H*W for large blocks.
        temporal_scale:              (min, max) fraction of T covered per block.
        aspect_ratio:                (min, max) block_h / block_w.
        npred_small:                 Number of small prediction blocks per sample.
        npred_large:                 Number of large prediction blocks per sample.
        max_context_frames_ratio:    Limit encoder visibility to this fraction of T.
        max_keep:                    Cap encoder-visible tokens (None = no cap).
    """

    spatial_scale_small:          Tuple[float, float] = (0.15, 0.15)
    spatial_scale_large:          Tuple[float, float] = (0.70, 0.70)
    temporal_scale:               Tuple[float, float] = (1.0, 1.0)
    aspect_ratio:                 Tuple[float, float] = (0.75, 1.5)
    npred_small:                  int = 8
    npred_large:                  int = 2
    max_context_frames_ratio:     float = 1.0
    max_keep:                     Optional[int] = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        def _check_range(name: str, lo: float, hi: float) -> None:
            if lo > hi:
                raise ValueError(
                    f"MaskConfig.{name}: min ({lo}) must be <= max ({hi})."
                )
            if lo <= 0:
                raise ValueError(
                    f"MaskConfig.{name}: min ({lo}) must be > 0."
                )

        _check_range("spatial_scale_small",
                     *self.spatial_scale_small)
        _check_range("spatial_scale_large",
                     *self.spatial_scale_large)
        _check_range("temporal_scale",
                     *self.temporal_scale)
        _check_range("aspect_ratio",
                     *self.aspect_ratio)

        if self.npred_small < 0:
            raise ValueError(
                f"MaskConfig.npred_small must be >= 0, got {self.npred_small}."
            )
        if self.npred_large < 0:
            raise ValueError(
                f"MaskConfig.npred_large must be >= 0, got {self.npred_large}."
            )
        if not (0.0 < self.max_context_frames_ratio <= 1.0):
            raise ValueError(
                f"MaskConfig.max_context_frames_ratio must be in (0, 1], "
                f"got {self.max_context_frames_ratio}."
            )
        if self.max_keep is not None and self.max_keep < 0:
            raise ValueError(
                f"MaskConfig.max_keep must be >= 0, got {self.max_keep}."
            )

    @classmethod
    def default(cls) -> "MaskConfig":
        """V-JEPA 2 default masking config."""
        return cls()

    @classmethod
    def causal(cls, ratio: float = 0.5) -> "MaskConfig":
        """Causal variant: encoder sees only the first `ratio` of T frames."""
        return cls(max_context_frames_ratio=ratio)

    @classmethod
    def ablation_random(cls) -> "MaskConfig":
        """Near-random masking (wide spatial scale range) for ablation."""
        return cls(
            spatial_scale_small=(0.1, 0.9),
            spatial_scale_large=(0.1, 0.9),
            aspect_ratio=(1.0, 1.0),  # square blocks
        )


@dataclass
class VideoConfig:
    """
    Configuration for video tokenization.

    Attributes:
        frames_per_clip:   Number of frames T per clip.
        patch_size:        Spatial patch side length P (pixels).
        tubelet_size:      Temporal kernel depth t (frames per tubelet).
        img_size:          Spatial height = width H = W (pixels).
        multi_fpc:         Enable heterogeneous FPC batching.
        in_chans:          Input channels (3 for RGB).
        embed_dim:         Token embedding dimension D.
    """

    frames_per_clip: int = 16
    patch_size:      int = 16
    tubelet_size:    int = 2
    img_size:        int = 224
    multi_fpc:       bool = False
    in_chans:        int = 3
    embed_dim:       int = 1024

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.frames_per_clip % self.tubelet_size != 0:
            raise ValueError(
                f"VideoConfig.frames_per_clip ({self.frames_per_clip}) must "
                f"be divisible by tubelet_size ({self.tubelet_size})."
            )
        if self.img_size % self.patch_size != 0:
            raise ValueError(
                f"VideoConfig.img_size ({self.img_size}) must be divisible "
                f"by patch_size ({self.patch_size})."
            )
        if self.frames_per_clip <= 0:
            raise ValueError("VideoConfig.frames_per_clip must be > 0.")
        if self.patch_size <= 0:
            raise ValueError("VideoConfig.patch_size must be > 0.")
        if self.tubelet_size <= 0:
            raise ValueError("VideoConfig.tubelet_size must be > 0.")
        if self.img_size <= 0:
            raise ValueError("VideoConfig.img_size must be > 0.")
        if self.embed_dim <= 0:
            raise ValueError("VideoConfig.embed_dim must be > 0.")
        if self.in_chans <= 0:
            raise ValueError("VideoConfig.in_chans must be > 0.")

    @property
    def grid_size(self) -> Tuple[int, int, int]:
        """(T_grid, H_grid, W_grid) token grid dimensions."""
        return (
            self.frames_per_clip // self.tubelet_size,
            self.img_size // self.patch_size,
            self.img_size // self.patch_size,
        )

    @property
    def num_patches(self) -> int:
        """Total number of tokens N = T_grid * H_grid * W_grid."""
        T, H, W = self.grid_size
        return T * H * W

    @classmethod
    def vjepa2_base(cls) -> "VideoConfig":
        """Canonical V-JEPA 2 configuration: 16-frame 224px clips."""
        return cls(
            frames_per_clip=16, patch_size=16, tubelet_size=2,
            img_size=224, embed_dim=1024,
        )

    @classmethod
    def vjepa2_256(cls) -> "VideoConfig":
        """High-resolution: 16-frame 256px clips (2048 tokens)."""
        return cls(
            frames_per_clip=16, patch_size=16, tubelet_size=2,
            img_size=256, embed_dim=1024,
        )

    @classmethod
    def small(cls) -> "VideoConfig":
        """Small config for unit tests: 8-frame 64px clips."""
        return cls(
            frames_per_clip=8, patch_size=16, tubelet_size=2,
            img_size=64, embed_dim=256,
        )


# ---------------------------------------------------------------------------
# apply_masks (module-level convenience wrapper)
# ---------------------------------------------------------------------------

def apply_masks(x: Tensor, masks: List[Tensor]) -> Tensor:
    """
    Gather encoder-visible tokens from a full token sequence.

    Thin wrapper around the canonical implementation in multi_sequence_template.
    Provided here so masking_config_template is self-contained and importable
    without multi_sequence_template.

    Args:
        x:     Tensor[B, N, D]
        masks: List[Tensor[N, bool]]  -- True = visible.

    Returns:
        Tensor[B, N_vis, D]
    """
    B, N, D = x.shape
    if len(masks) != B:
        raise ValueError(
            f"len(masks)={len(masks)} must equal batch size B={B}."
        )

    n_vis = int(masks[0].long().sum().item())
    for i, m in enumerate(masks):
        if m.shape != (N,):
            raise ValueError(f"masks[{i}].shape={m.shape}, expected ({N},).")
        nv = int(m.long().sum().item())
        if nv != n_vis:
            raise ValueError(
                f"masks[{i}] has {nv} visible tokens, masks[0] has {n_vis}."
            )

    out = torch.zeros(B, n_vis, D, dtype=x.dtype, device=x.device)
    for i, m in enumerate(masks):
        out[i] = x[i][m]
    return out


# ---------------------------------------------------------------------------
# make_mask_generators factory
# ---------------------------------------------------------------------------

def make_mask_generators(
    mask_config:  MaskConfig,
    video_config: VideoConfig,
) -> Tuple[Any, Any]:
    """
    Build (small_generator, large_generator) from config objects.

    Returns a tuple of two MaskGenerator instances: one configured for
    small blocks (fine-grained local prediction) and one for large blocks
    (holistic coarse prediction).

    Defers the import of MaskGenerator to avoid circular dependency when
    users import only masking_config_template.

    Args:
        mask_config:  Masking hyper-parameters.
        video_config: Video tokenization config (provides grid_size).

    Returns:
        (mg_small, mg_large)  -- both MaskGenerator instances.
    """
    # Lazy import to allow standalone use of config module
    try:
        from mask_generator_template import MaskGenerator
    except ImportError as exc:
        raise ImportError(
            "make_mask_generators requires mask_generator_template.py "
            "to be on the Python path."
        ) from exc

    grid = video_config.grid_size

    mg_small = MaskGenerator(
        spatial_scale=mask_config.spatial_scale_small,
        temporal_scale=mask_config.temporal_scale,
        aspect_ratio=mask_config.aspect_ratio,
        npred=mask_config.npred_small,
        grid_size=grid,
        max_context_frames_ratio=mask_config.max_context_frames_ratio,
        max_keep=mask_config.max_keep,
    )

    mg_large = MaskGenerator(
        spatial_scale=mask_config.spatial_scale_large,
        temporal_scale=mask_config.temporal_scale,
        aspect_ratio=mask_config.aspect_ratio,
        npred=mask_config.npred_large,
        grid_size=grid,
        max_context_frames_ratio=mask_config.max_context_frames_ratio,
        max_keep=mask_config.max_keep,
    )

    return mg_small, mg_large


# ---------------------------------------------------------------------------
# Self-tests  (python masking_config_template.py)
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

    # -------------------------------------------------------------------
    # MaskConfig validation
    # -------------------------------------------------------------------
    print("=" * 60)
    print("MaskConfig -- validation")
    print("=" * 60)

    mc = MaskConfig()
    check("MaskConfig default constructs without error", True)
    check("MaskConfig.npred_small == 8", mc.npred_small == 8)
    check("MaskConfig.npred_large == 2", mc.npred_large == 2)
    check("MaskConfig.max_context_frames_ratio == 1.0",
          mc.max_context_frames_ratio == 1.0)
    check("MaskConfig.max_keep is None", mc.max_keep is None)

    # Causal preset
    mc_causal = MaskConfig.causal(0.5)
    check("MaskConfig.causal(0.5).max_context_frames_ratio == 0.5",
          mc_causal.max_context_frames_ratio == 0.5)

    # Invalid: min > max
    try:
        MaskConfig(spatial_scale_small=(0.8, 0.2))
        check("MaskConfig raises on invalid spatial_scale_small", False, "no error")
    except ValueError:
        check("MaskConfig raises on invalid spatial_scale_small", True)

    # Invalid: max_context_frames_ratio out of range
    try:
        MaskConfig(max_context_frames_ratio=0.0)
        check("MaskConfig raises on max_context_frames_ratio=0", False, "no error")
    except ValueError:
        check("MaskConfig raises on max_context_frames_ratio=0", True)

    # Invalid: negative npred
    try:
        MaskConfig(npred_small=-1)
        check("MaskConfig raises on npred_small=-1", False, "no error")
    except ValueError:
        check("MaskConfig raises on npred_small=-1", True)

    # -------------------------------------------------------------------
    # VideoConfig validation
    # -------------------------------------------------------------------
    print()
    print("=" * 60)
    print("VideoConfig -- properties")
    print("=" * 60)

    vc = VideoConfig.vjepa2_base()
    check("VideoConfig vjepa2_base frames_per_clip == 16",
          vc.frames_per_clip == 16)
    check("VideoConfig vjepa2_base grid_size == (8,14,14)",
          vc.grid_size == (8, 14, 14))
    check("VideoConfig vjepa2_base num_patches == 1568",
          vc.num_patches == 1568)

    vc256 = VideoConfig.vjepa2_256()
    check("VideoConfig vjepa2_256 num_patches == 2048",
          vc256.num_patches == 2048)

    # Divisibility errors
    try:
        VideoConfig(frames_per_clip=15, tubelet_size=2)
        check("VideoConfig raises on frames not divisible by tubelet", False, "no error")
    except ValueError:
        check("VideoConfig raises on frames not divisible by tubelet", True)

    try:
        VideoConfig(img_size=225, patch_size=16)
        check("VideoConfig raises on img_size not divisible by patch", False, "no error")
    except ValueError:
        check("VideoConfig raises on img_size not divisible by patch", True)

    # -------------------------------------------------------------------
    # apply_masks correctness
    # -------------------------------------------------------------------
    print()
    print("=" * 60)
    print("apply_masks -- correctness")
    print("=" * 60)

    B, N, D = 3, 100, 32
    x  = torch.randn(B, N, D)
    ms = [torch.zeros(N, dtype=torch.bool) for _ in range(B)]
    for m in ms:
        m[::2] = True   # 50 visible tokens

    out = apply_masks(x, ms)
    check("apply_masks shape [3,50,32]",
          out.shape == (B, 50, D), f"got {out.shape}")

    for i, m in enumerate(ms):
        check(f"apply_masks[{i}] values match x[i][m]",
              torch.allclose(out[i], x[i][m]))

    # All-True mask
    ms_all = [torch.ones(N, dtype=torch.bool) for _ in range(B)]
    out_all = apply_masks(x, ms_all)
    check("apply_masks all-True == identity",
          torch.allclose(out_all, x))

    # Error: mismatched B
    try:
        apply_masks(x, ms[:2])
        check("apply_masks raises on mask count mismatch", False, "no error")
    except ValueError:
        check("apply_masks raises on mask count mismatch", True)

    # -------------------------------------------------------------------
    # make_mask_generators factory
    # -------------------------------------------------------------------
    print()
    print("=" * 60)
    print("make_mask_generators -- factory")
    print("=" * 60)

    mc2  = MaskConfig()
    vc2  = VideoConfig.vjepa2_base()
    mg_s, mg_l = make_mask_generators(mc2, vc2)

    enc_s, pred_s = mg_s(2, seed=0)
    enc_l, pred_l = mg_l(2, seed=0)

    for i in range(2):
        check(f"mg_small[{i}] full coverage",
              (enc_s[i] | pred_s[i]).all().item())
        check(f"mg_large[{i}] full coverage",
              (enc_l[i] | pred_l[i]).all().item())

    check("mg_small grid matches VideoConfig",
          mg_s.grid_size == vc2.grid_size)
    check("mg_large grid matches VideoConfig",
          mg_l.grid_size == vc2.grid_size)
    check("mg_small npred == 8", mg_s.npred == mc2.npred_small)
    check("mg_large npred == 2", mg_l.npred == mc2.npred_large)

    print()
    print("=" * 60)
    print(f"Results: {PASSED} passed, {FAILED} failed")
    print("=" * 60)

    sys.exit(0 if FAILED == 0 else 1)
