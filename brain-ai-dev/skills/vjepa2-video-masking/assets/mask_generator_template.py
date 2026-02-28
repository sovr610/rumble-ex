"""
mask_generator_template.py
==========================
MaskGenerator: produces multi-block 3-D spatiotemporal masks for V-JEPA 2
pretraining.

References:
    - references/masking-algorithm.md
    - SKILL.md § MaskGenerator, § Multi-Block 3D Masking Algorithm

Done-when Gate 2 (Mask Coverage):
    MaskGenerator produces (masks_enc, masks_pred) where:
    - masks_enc[i] | masks_pred[i]  is all True  (no gaps)
    - masks_enc[i] & masks_pred[i]  is all False (no overlaps)
    - Both have shape [N] = [T*H*W] boolean tensors
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sample_block_size(
    T: int, H: int, W: int,
    spatial_scale: Tuple[float, float],
    temporal_scale: Tuple[float, float],
    aspect_ratio: Tuple[float, float],
    rng: random.Random,
) -> Tuple[int, int, int]:
    """
    Sample (temporal_len, block_h, block_w) for one mask block.

    Uses the V-JEPA 2 algorithm:
        temporal_len = round(T * uniform(t_scale_min, t_scale_max))
        spatial_area = H * W * uniform(s_scale_min, s_scale_max)
        aspect       = uniform(aspect_min, aspect_max)
        block_h      = round(sqrt(spatial_area * aspect))
        block_w      = round(sqrt(spatial_area / aspect))

    All dimensions clamped to [1, grid_size] to prevent zero-size blocks or
    out-of-bounds placement.
    """
    # Temporal
    t_lo, t_hi = temporal_scale
    temporal_len = max(1, round(T * rng.uniform(t_lo, t_hi)))
    temporal_len = min(temporal_len, T)

    # Spatial
    s_lo, s_hi = spatial_scale
    spatial_area = H * W * rng.uniform(s_lo, s_hi)
    aspect = rng.uniform(*aspect_ratio)

    block_h = round(math.sqrt(spatial_area * aspect))
    block_w = round(math.sqrt(spatial_area / max(aspect, 1e-6)))

    block_h = max(1, min(block_h, H))
    block_w = max(1, min(block_w, W))

    return temporal_len, block_h, block_w


def _place_block(
    T: int, H: int, W: int,
    temporal_len: int, block_h: int, block_w: int,
    rng: random.Random,
) -> Tuple[int, int, int]:
    """
    Sample a random top-left corner (t0, h0, w0) so the block fits in the
    grid.  Raises ValueError if grid is smaller than block.
    """
    if T < temporal_len or H < block_h or W < block_w:
        raise ValueError(
            f"Block ({temporal_len},{block_h},{block_w}) does not fit in "
            f"grid ({T},{H},{W})."
        )
    t0 = rng.randint(0, T - temporal_len)
    h0 = rng.randint(0, H - block_h)
    w0 = rng.randint(0, W - block_w)
    return t0, h0, w0


def _cap_keep(
    mask: Tensor,
    max_keep: int,
    rng: random.Random,
) -> Tensor:
    """
    Randomly subsample encoder-visible tokens to at most max_keep.

    Args:
        mask:     Bool tensor of shape [N]; True = encoder visible.
        max_keep: Maximum number of True values to retain.
        rng:      Random number generator for reproducibility.
    Returns:
        Bool tensor [N] with at most max_keep True values.
    """
    visible = mask.nonzero(as_tuple=False).squeeze(1)  # [K]
    K = visible.numel()
    if K <= max_keep:
        return mask

    # Select max_keep indices from visible set
    perm = torch.randperm(K)[:max_keep]
    selected = visible[perm]

    new_mask = torch.zeros_like(mask)
    new_mask[selected] = True
    return new_mask


# ---------------------------------------------------------------------------
# MaskGenerator
# ---------------------------------------------------------------------------

class MaskGenerator:
    """
    Spatiotemporal multi-block mask generator for V-JEPA 2 pretraining.

    Places `npred` cuboid blocks randomly in the (T, H, W) token grid to
    form the prediction mask.  The encoder mask is the bitwise complement.

    The returned masks are flat boolean tensors of length N = T * H * W.

    Args:
        spatial_scale:             (min, max) fraction of H*W area per block.
        temporal_scale:            (min, max) fraction of T depth per block.
        aspect_ratio:              (min, max) block_h / block_w ratio.
        npred:                     Number of prediction blocks per sample.
        grid_size:                 (T, H, W) token grid dimensions.
        max_context_frames_ratio:  Cap visible encoder frames to this fraction.
        max_keep:                  Cap encoder-visible tokens (None = no cap).
    """

    def __init__(
        self,
        spatial_scale:            Tuple[float, float] = (0.15, 0.15),
        temporal_scale:           Tuple[float, float] = (1.0, 1.0),
        aspect_ratio:             Tuple[float, float] = (0.75, 1.5),
        npred:                    int = 8,
        grid_size:                Tuple[int, int, int] = (8, 14, 14),
        max_context_frames_ratio: float = 1.0,
        max_keep:                 Optional[int] = None,
    ) -> None:
        T, H, W = grid_size
        if T <= 0 or H <= 0 or W <= 0:
            raise ValueError(f"grid_size must be positive, got {grid_size}.")
        if not (0.0 < spatial_scale[0] <= spatial_scale[1]):
            raise ValueError(f"spatial_scale must satisfy 0 < min <= max, got {spatial_scale}.")
        if not (0.0 < temporal_scale[0] <= temporal_scale[1]):
            raise ValueError(f"temporal_scale must satisfy 0 < min <= max, got {temporal_scale}.")
        if not (0.0 < aspect_ratio[0] <= aspect_ratio[1]):
            raise ValueError(f"aspect_ratio must satisfy 0 < min <= max, got {aspect_ratio}.")
        if npred < 0:
            raise ValueError(f"npred must be >= 0, got {npred}.")
        if not (0.0 < max_context_frames_ratio <= 1.0):
            raise ValueError(
                f"max_context_frames_ratio must be in (0, 1], got {max_context_frames_ratio}."
            )

        self.spatial_scale            = spatial_scale
        self.temporal_scale           = temporal_scale
        self.aspect_ratio             = aspect_ratio
        self.npred                    = npred
        self.grid_size                = grid_size      # (T, H, W)
        self.max_context_frames_ratio = max_context_frames_ratio
        self.max_keep                 = max_keep
        self.N                        = T * H * W

    # ------------------------------------------------------------------
    # Internal: generate masks for one sample
    # ------------------------------------------------------------------

    def _generate_single(self, rng: random.Random) -> Tuple[Tensor, Tensor]:
        """
        Generate (mask_enc, mask_pred) for a single video clip.

        Returns:
            mask_enc:  Bool[N] -- True for encoder-visible tokens.
            mask_pred: Bool[N] -- True for prediction-target tokens.
        """
        T, H, W = self.grid_size

        pred_3d = torch.zeros(T, H, W, dtype=torch.bool)

        for _ in range(self.npred):
            tl, bh, bw = _sample_block_size(
                T, H, W,
                self.spatial_scale,
                self.temporal_scale,
                self.aspect_ratio,
                rng,
            )
            t0, h0, w0 = _place_block(T, H, W, tl, bh, bw, rng)
            pred_3d[t0:t0 + tl, h0:h0 + bh, w0:w0 + bw] = True

        enc_3d = ~pred_3d

        # Apply temporal context cap
        if self.max_context_frames_ratio < 1.0:
            max_frames = max(1, round(T * self.max_context_frames_ratio))
            enc_3d[max_frames:, :, :] = False

        mask_enc  = enc_3d.flatten()   # [N]
        mask_pred = pred_3d.flatten()  # [N]

        # Apply max_keep cap on encoder-visible tokens
        if self.max_keep is not None:
            mask_enc = _cap_keep(mask_enc, self.max_keep, rng)

        return mask_enc, mask_pred

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(
        self,
        batch_size: int,
        seed:       Optional[int] = None,
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Generate masks for a batch of clips.

        Args:
            batch_size: Number of clips in the batch.
            seed:       Optional integer seed for deterministic generation.
                        When None, uses the global Python random state.

        Returns:
            masks_enc:  List[Tensor[N, bool]]  -- one per sample.
            masks_pred: List[Tensor[N, bool]]  -- one per sample.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")

        rng = random.Random(seed) if seed is not None else random.Random()

        masks_enc:  List[Tensor] = []
        masks_pred: List[Tensor] = []

        for _ in range(batch_size):
            m_enc, m_pred = self._generate_single(rng)
            masks_enc.append(m_enc)
            masks_pred.append(m_pred)

        return masks_enc, masks_pred

    def __repr__(self) -> str:
        T, H, W = self.grid_size
        return (
            f"MaskGenerator(grid=({T},{H},{W}), N={self.N}, npred={self.npred}, "
            f"spatial_scale={self.spatial_scale}, "
            f"temporal_scale={self.temporal_scale}, "
            f"aspect_ratio={self.aspect_ratio}, "
            f"max_context_frames_ratio={self.max_context_frames_ratio}, "
            f"max_keep={self.max_keep})"
        )


# ---------------------------------------------------------------------------
# Preset factories
# ---------------------------------------------------------------------------

def make_small_mask_generator(
    grid_size: Tuple[int, int, int] = (8, 14, 14),
    npred:     int = 8,
    **kwargs,
) -> MaskGenerator:
    """Small-block generator: fine-grained local prediction targets."""
    return MaskGenerator(
        spatial_scale=(0.15, 0.15),
        temporal_scale=(1.0, 1.0),
        aspect_ratio=(0.75, 1.5),
        npred=npred,
        grid_size=grid_size,
        **kwargs,
    )


def make_large_mask_generator(
    grid_size: Tuple[int, int, int] = (8, 14, 14),
    npred:     int = 2,
    **kwargs,
) -> MaskGenerator:
    """Large-block generator: holistic coarse prediction targets."""
    return MaskGenerator(
        spatial_scale=(0.7, 0.7),
        temporal_scale=(1.0, 1.0),
        aspect_ratio=(0.75, 1.5),
        npred=npred,
        grid_size=grid_size,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Self-tests  (python mask_generator_template.py)
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
    # Baseline: standard 8x14x14 grid
    # -------------------------------------------------------------------
    print("=" * 60)
    print("MaskGenerator -- coverage invariants")
    print("=" * 60)

    GRID  = (8, 14, 14)
    N     = 8 * 14 * 14  # 1568
    BATCH = 4

    mg = MaskGenerator(
        spatial_scale=(0.15, 0.15),
        temporal_scale=(1.0, 1.0),
        aspect_ratio=(0.75, 1.5),
        npred=8,
        grid_size=GRID,
    )
    enc_list, pred_list = mg(BATCH, seed=42)

    # Shape checks
    check("Number of enc masks == batch_size",
          len(enc_list) == BATCH)
    check("Number of pred masks == batch_size",
          len(pred_list) == BATCH)
    check("enc mask shape == [N]",
          enc_list[0].shape == (N,),
          f"got {enc_list[0].shape}")
    check("pred mask shape == [N]",
          pred_list[0].shape == (N,),
          f"got {pred_list[0].shape}")
    check("enc dtype is bool",
          enc_list[0].dtype == torch.bool)
    check("pred dtype is bool",
          pred_list[0].dtype == torch.bool)

    # Coverage invariants for every sample
    all_covered    = True
    no_overlap     = True
    exact_sum      = True
    for i in range(BATCH):
        enc  = enc_list[i]
        pred = pred_list[i]
        if not (enc | pred).all().item():
            all_covered = False
        if (enc & pred).any().item():
            no_overlap = False
        if enc.sum().item() + pred.sum().item() != N:
            exact_sum = False

    check("No gaps: (enc OR pred) == all True", all_covered)
    check("No overlaps: (enc AND pred) == all False", no_overlap)
    check("enc.sum() + pred.sum() == N", exact_sum)

    # -------------------------------------------------------------------
    # Large block generator
    # -------------------------------------------------------------------
    print()
    print("=" * 60)
    print("MaskGenerator -- large blocks")
    print("=" * 60)

    mg_large = make_large_mask_generator(grid_size=GRID, npred=2)
    enc_l, pred_l = mg_large(BATCH, seed=7)

    for i in range(BATCH):
        enc  = enc_l[i]
        pred = pred_l[i]
        check(f"Large[{i}] full coverage",
              (enc | pred).all().item())
        check(f"Large[{i}] no overlap",
              not (enc & pred).any().item())

    # Large blocks should cover more area on average
    mean_pred_ratio = sum(p.float().mean().item() for p in pred_l) / BATCH
    check("Large blocks cover >50% on average",
          mean_pred_ratio >= 0.5,
          f"mean pred ratio = {mean_pred_ratio:.3f}")

    # -------------------------------------------------------------------
    # max_context_frames_ratio
    # -------------------------------------------------------------------
    print()
    print("=" * 60)
    print("MaskGenerator -- max_context_frames_ratio")
    print("=" * 60)

    T, H, W = GRID
    mg_half = MaskGenerator(
        spatial_scale=(0.15, 0.15),
        temporal_scale=(1.0, 1.0),
        aspect_ratio=(1.0, 1.0),
        npred=0,            # no pred blocks so enc = full complement
        grid_size=GRID,
        max_context_frames_ratio=0.5,
    )
    enc_h, pred_h = mg_half(1, seed=0)
    enc_3d = enc_h[0].view(T, H, W)
    max_enc_frame = enc_3d.any(dim=(1, 2)).nonzero(as_tuple=False)

    if max_enc_frame.numel() > 0:
        last_visible = max_enc_frame[-1, 0].item()
        check("max_context_frames_ratio=0.5 limits visible frames",
              last_visible < T // 2 + 1,
              f"last visible frame = {last_visible}, T = {T}")
    else:
        check("max_context_frames_ratio=0.5 some frames visible",
              False, "no visible frames at all")

    # -------------------------------------------------------------------
    # max_keep cap
    # -------------------------------------------------------------------
    print()
    print("=" * 60)
    print("MaskGenerator -- max_keep")
    print("=" * 60)

    MAX_KEEP = 300
    mg_mk = MaskGenerator(
        spatial_scale=(0.15, 0.15),
        temporal_scale=(1.0, 1.0),
        aspect_ratio=(1.0, 1.0),
        npred=1,
        grid_size=GRID,
        max_keep=MAX_KEEP,
    )
    enc_mk, pred_mk = mg_mk(BATCH, seed=99)
    all_capped = all(e.sum().item() <= MAX_KEEP for e in enc_mk)
    check(f"max_keep={MAX_KEEP} respected for all samples", all_capped,
          f"counts = {[e.sum().item() for e in enc_mk]}")

    # -------------------------------------------------------------------
    # Seed determinism
    # -------------------------------------------------------------------
    print()
    print("=" * 60)
    print("MaskGenerator -- seed determinism")
    print("=" * 60)

    mg_det = make_small_mask_generator(grid_size=GRID, npred=4)
    enc_a, _ = mg_det(BATCH, seed=1234)
    enc_b, _ = mg_det(BATCH, seed=1234)
    enc_c, _ = mg_det(BATCH, seed=9999)

    same_seed = all(torch.equal(a, b) for a, b in zip(enc_a, enc_b))
    diff_seed = any(not torch.equal(a, c) for a, c in zip(enc_a, enc_c))

    check("Same seed produces identical masks", same_seed)
    check("Different seeds produce different masks (probabilistic)", diff_seed)

    # -------------------------------------------------------------------
    # Edge cases
    # -------------------------------------------------------------------
    print()
    print("=" * 60)
    print("MaskGenerator -- edge cases")
    print("=" * 60)

    # npred=0 -> enc all True
    mg_zero = MaskGenerator(npred=0, grid_size=(4, 4, 4))
    enc_z, pred_z = mg_zero(1, seed=0)
    check("npred=0: enc all True", enc_z[0].all().item())
    check("npred=0: pred all False", not pred_z[0].any().item())

    # Single-token grid
    mg_one = MaskGenerator(npred=1, grid_size=(1, 1, 1))
    enc_o, pred_o = mg_one(1, seed=0)
    check("grid=(1,1,1) no error", True)
    check("grid=(1,1,1) masks have length 1",
          enc_o[0].numel() == 1 and pred_o[0].numel() == 1)

    # Invalid grid
    try:
        MaskGenerator(grid_size=(0, 14, 14))
        check("grid=(0,14,14) raises ValueError", False, "no error raised")
    except ValueError:
        check("grid=(0,14,14) raises ValueError", True)

    print()
    print("=" * 60)
    print(f"Results: {PASSED} passed, {FAILED} failed")
    print("=" * 60)

    sys.exit(0 if FAILED == 0 else 1)
