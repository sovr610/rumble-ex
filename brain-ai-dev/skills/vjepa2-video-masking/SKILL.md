---
name: V-JEPA 2 Video & Masking
description: >
  This skill should be used when the user asks to "implement tubelet tokenization",
  "create 3D masking strategy", "video patch embedding", "mask collator",
  "multi-block masking", "spatiotemporal masking", "frames per clip handling",
  "variable length video batching", "video temporal modeling",
  "multi-sequence wrapper", "context frames ratio", "max keep tokens",
  "mask generator for V-JEPA", or needs guidance on video tokenization,
  spatiotemporal masking strategies, mask collation, or variable-length
  video batch handling for V-JEPA 2.
version: 0.1.0
---

# V-JEPA 2 Video & Masking

## Overview

Guide implementation of video processing and masking infrastructure for V-JEPA 2. Cover tubelet tokenization (3D Conv patching), temporal modeling with variable frames-per-clip (FPC), multi-block 3D spatiotemporal mask generation, mask collation for DataLoader integration, multi-sequence wrappers for heterogeneous batch lengths, and the critical interplay between masking and encoder efficiency.

## Public Contract

### PatchEmbed3D

Tokenize video into spatiotemporal patches using 3D convolution.

```python
class PatchEmbed3D(nn.Module):
    def __init__(self, img_size=224, patch_size=16, tubelet_size=2,
                 in_chans=3, embed_dim=1024): ...
    def forward(self, x: Tensor) -> Tensor: ...  # [B, C, T, H, W] -> [B, N, D]
    def num_patches(self) -> int: ...
```

### MaskGenerator

Generate spatiotemporal block masks for JEPA pretraining.

```python
class MaskGenerator:
    def __init__(self, spatial_scale: Tuple[float, float],
                 temporal_scale: Tuple[float, float],
                 aspect_ratio: Tuple[float, float],
                 npred: int = 8, grid_size: Tuple[int, int, int] = (8, 16, 16)): ...
    def __call__(self, batch_size: int) -> Tuple[List[Tensor], List[Tensor]]: ...
```

### MaskCollator

Custom DataLoader collator integrating masking with batching.

```python
class MaskCollator:
    def __init__(self, mask_generators: List[MaskGenerator],
                 max_context_frames_ratio: float = 1.0,
                 max_keep: Optional[int] = None): ...
    def __call__(self, batch) -> List[Tuple[Tensor, List[Tensor], List[Tensor]]]: ...
```

### MultiSequenceWrapper

Handle variable-length sequences in a single batch.

```python
class MultiSequenceEncoder(nn.Module):
    def __init__(self, encoder: VisionTransformer): ...
    def forward(self, x_groups, masks_groups) -> List[Tensor]: ...

class MultiSequencePredictor(nn.Module):
    def __init__(self, predictor: VisionTransformerPredictor): ...
    def forward(self, context_groups, masks_enc_groups, masks_pred_groups) -> List[Tensor]: ...
```

## Key Concepts

### Tubelet Tokenization

Videos `[B, C, T, H, W]` tokenized via `Conv3d(kernel=(t, P, P))`:
- Default: `tubelet_size=2`, `patch_size=16`
- For 16-frame 256px video: `(16/2) * (256/16) * (256/16) = 8 * 16 * 16 = 2048` tokens
- Grid dimensions: `(T/t, H/P, W/P)` = `(depth, height, width)`

### Multi-Block 3D Masking Algorithm

```
1. Sample block size from configured range:
   - temporal_len = T * uniform(t_scale_min, t_scale_max)
   - spatial_area = H * W * uniform(s_scale_min, s_scale_max)
   - aspect = uniform(aspect_min, aspect_max)
   - block_h = sqrt(spatial_area * aspect)
   - block_w = sqrt(spatial_area / aspect)

2. For each sample in batch:
   - Randomly place npred blocks in (T, H, W) grid -> prediction targets
   - Encoder mask = complement (visible patches)

3. max_context_frames_ratio limits visible temporal extent
4. max_keep caps encoder-visible tokens for memory
```

### Masking Configuration

- **Spatial scales**: e.g., `[0.15, 0.15]` (small) and `[0.7, 0.7]` (large)
- **Temporal scales**: e.g., `[1.0, 1.0]` (full temporal extent)
- **Aspect ratios**: e.g., `[0.75, 1.5]` (non-square patches allowed)
- **npred**: Number of prediction blocks (default: 8 small + 2 large)

### Mask Collator Details

- Wraps `default_collate` with mask generation
- Groups batch items by FPC (frames-per-clip)
- Returns `[(collated_batch, masks_enc, masks_pred)]` per FPC group
- Uses `multiprocessing.Value` counter for deterministic seed stepping across workers

### Multi-Sequence Batching

When training with heterogeneous clip lengths:
1. Mask collator groups items by their FPC
2. Each FPC group processed separately through encoder
3. Results collected and loss computed per group
4. Enables mixed-resolution training in single batch

### Token Masking for Efficiency

`apply_masks(x, masks)` removes masked tokens before encoder forward:
- Reduces sequence length from full to visible-only
- Proportional FLOPs reduction (e.g., 75% masked = 4x reduction)
- Tokens re-inserted at original positions for predictor

## Configuration Surface

```python
@dataclass
class MaskConfig:
    spatial_scale_small: Tuple[float, float] = (0.15, 0.15)
    spatial_scale_large: Tuple[float, float] = (0.7, 0.7)
    temporal_scale: Tuple[float, float] = (1.0, 1.0)
    aspect_ratio: Tuple[float, float] = (0.75, 1.5)
    npred_small: int = 8
    npred_large: int = 2
    max_context_frames_ratio: float = 1.0
    max_keep: Optional[int] = None

@dataclass
class VideoConfig:
    frames_per_clip: int = 16
    patch_size: int = 16
    tubelet_size: int = 2
    img_size: int = 224
    multi_fpc: bool = False              # Enable heterogeneous FPC
```

## Done-When Gates

1. **Tokenization** — `PatchEmbed3D` converts `[B, 3, 16, 224, 224]` to `[B, 2048, D]`; token count matches `(T/t)*(H/P)*(W/P)`.
2. **Mask Coverage** — `MaskGenerator` produces masks where encoder+prediction masks together cover all tokens exactly once; no overlaps, no gaps.
3. **Collator Grouping** — `MaskCollator` correctly groups mixed-FPC batches; each group has consistent sequence length.

## Failure Modes

| Mode | Symptom | Fix |
|------|---------|-----|
| Mask overlap | Duplicate tokens in loss | Verify mask complement logic; encoder ∪ pred = all tokens |
| FPC mismatch | Shape error in collator | Ensure all items in FPC group have same frame count |
| Too few visible tokens | Degenerate context | Increase max_keep or reduce masking ratio |
| Worker seed divergence | Non-reproducible masks | Verify multiprocessing.Value counter shared correctly |

## Resources

### Reference Files
- **`references/tubelet-tokenization.md`** — 3D Conv patching, grid dimensions, temporal modeling
- **`references/masking-algorithm.md`** — Block sampling, spatial/temporal scales, aspect ratios
- **`references/mask-collation.md`** — Collator design, FPC grouping, worker seeding
- **`references/multi-sequence.md`** — Variable-length batching, wrapper architecture
- **`references/testing-matrix.md`** — Test scenarios for video and masking

### Asset Files
- **`assets/patch_embed_template.py`** — PatchEmbed, PatchEmbed3D with grid computation
- **`assets/mask_generator_template.py`** — MaskGenerator with block sampling algorithm
- **`assets/mask_collator_template.py`** — MaskCollator with FPC grouping, worker seeding
- **`assets/multi_sequence_template.py`** — MultiSequenceEncoder/Predictor wrappers
- **`assets/masking_config_template.py`** — MaskConfig, VideoConfig, apply_masks utility

### Scripts
- **`scripts/validate_masking.py`** — Validates done-when gates
- **`scripts/gen_masking_tests.py`** — Generates 100+ pytest test cases
- **`scripts/masking_benchmark.py`** — Mask generation throughput and coverage analysis
