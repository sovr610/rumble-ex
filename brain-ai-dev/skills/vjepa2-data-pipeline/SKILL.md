---
name: V-JEPA 2 Data Pipeline
description: >
  This skill should be used when the user asks to "load video dataset",
  "implement video transforms", "data augmentation for V-JEPA",
  "video decoding with decord", "clip sampling", "frame padding",
  "RandAugment for video", "motion shift augmentation",
  "random erasing", "video normalization", "YAML config parsing",
  "dataset registry", "distributed sampler", "weighted sampling",
  "multi-source dataset", "video DataLoader", "worker seeding",
  or needs guidance on video data loading, augmentation pipelines,
  configuration management, or dataset engineering for V-JEPA 2.
version: 0.1.0
---

# V-JEPA 2 Data Pipeline

## Overview

Guide implementation of the complete data pipeline for V-JEPA 2: video decoding (decord), clip sampling (fps/duration/frame_step modes), data augmentation (crop, flip, RandAugment, motion shift, random erasing), transform pipelines, dataset management (multi-source with weights), distributed sampling, YAML configuration, and DataLoader engineering with deterministic worker seeding.

## Public Contract

### VideoDataset

Core video dataset with configurable clip sampling.

```python
class VideoDataset(Dataset):
    def __init__(self, data_paths: List[str], clip_mode: str = "fps",
                 frames_per_clip: int = 16, target_fps: int = 10,
                 transform: Optional[Callable] = None): ...
    def __getitem__(self, idx) -> Dict[str, Tensor]: ...
```

### VideoTransformPipeline

Composable video augmentation pipeline.

```python
class VideoTransformPipeline:
    def __init__(self, config: AugConfig): ...
    def get_train_transform(self) -> Callable: ...
    def get_eval_transform(self) -> Callable: ...
```

### DataManager

Unified factory for building datasets and loaders.

```python
class DataManager:
    def __init__(self, config: DataConfig): ...
    def build_train_loader(self, mask_collator: Optional[MaskCollator] = None) -> DataLoader: ...
    def build_eval_loader(self) -> DataLoader: ...
```

### DistributedWeightedSampler

Weighted sampling supporting multi-source datasets across ranks.

```python
class DistributedWeightedSampler(Sampler):
    def __init__(self, weights: List[float], num_samples: int,
                 rank: int, world_size: int): ...
```

## Key Concepts

### Video Transform Pipeline

```
Video [T, H, W, C] -> RandomResizedCrop (+motion shift) -> HorizontalFlip
                    -> [Optional: RandAugment per-frame]
                    -> [Optional: RandomErasing]
                    -> ClipToTensor [C, T, H, W] -> Normalize (ImageNet mu/sigma)
```

### Clip Sampling Modes (mutually exclusive)

| Mode | Parameter | Description |
|------|-----------|-------------|
| `fps` | `target_fps=10` | Sample frames at target FPS |
| `duration` | `clip_duration_sec=3.2` | Fixed duration clip |
| `frame_step` | `frame_step=4` | Fixed step between frames |

### Frame Padding

`circulant` mode: wraps video cyclically for short clips (fewer frames than requested).

### Key Augmentation Operations

| Transform | Description |
|-----------|-------------|
| RandomResizedCrop | Spatial crop with scale/aspect jitter |
| Motion Shift | Temporal jittering of spatial crop position across frames |
| RandAugment | Per-frame augmentations (shear, translate, rotate, color) |
| Random Erasing | Cube mode for temporal consistency |
| ClipToTensor | `[T, H, W, C]` list -> `[C, T, H, W]` float tensor |

### Normalization

ImageNet defaults: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`
When auto-augment disabled: mean/std scaled to 0-255 range.

### Robotics-Specific Augmentation

- No horizontal flip (direction-sensitive)
- Fixed or near-fixed scale (minimal spatial jitter)
- Preserves spatial relationships critical for robot control

### Multi-Source Dataset

- Multiple data paths with per-source weights for mixing
- Variable FPC per source for heterogeneous training
- `ConcatIndices` maps global indices to `(dataset_idx, sample_idx)`

### Worker Seeding

- Deterministic via LCG algorithm for reproducibility
- Sets `torch`, `random`, `numpy` seeds per worker
- Optional resource monitoring thread (CPU/memory)

### YAML Configuration

Standard sections: `app`, `meta`, `mask`, `model`, `data`, `data_aug`, `loss`, `optimization`.
All parameters via `dict.get("key", default)` pattern.

## Configuration Surface

```python
@dataclass
class DataConfig:
    data_paths: List[str] = ()
    data_weights: List[float] = ()
    clip_mode: str = "fps"
    frames_per_clip: int = 16
    target_fps: int = 10
    img_size: int = 224
    num_workers: int = 8
    batch_size: int = 64

@dataclass
class AugConfig:
    crop_scale: Tuple[float, float] = (0.3, 1.0)
    crop_ratio: Tuple[float, float] = (0.75, 1.33)
    horizontal_flip: bool = True
    auto_augment: bool = False
    rand_augment_n: int = 2
    rand_augment_m: int = 9
    motion_shift: bool = True
    random_erasing: float = 0.0          # Probability
    normalize_mean: Tuple = (0.485, 0.456, 0.406)
    normalize_std: Tuple = (0.229, 0.224, 0.225)
```

## Done-When Gates

1. **Video Loading** — `VideoDataset.__getitem__()` returns correctly shaped tensor `[C, T, H, W]` from synthetic video data; all three clip modes produce valid frame counts.
2. **Transform Pipeline** — Train transform produces augmented tensors with correct shape and normalization; eval transform is deterministic (same input = same output).
3. **Multi-Source Sampling** — `DistributedWeightedSampler` respects weights; no duplicate samples across ranks; full coverage per epoch.

## Resources

### Reference Files
- **`references/video-decoding.md`** — decord VideoReader, clip modes, frame padding, GPU decoding
- **`references/augmentation-ops.md`** — Each transform operation, parameters, temporal consistency
- **`references/dataset-management.md`** — Multi-source mixing, ConcatIndices, weighted sampling
- **`references/yaml-config.md`** — Config schema, section descriptions, progressive training configs
- **`references/testing-matrix.md`** — Test scenarios

### Asset Files
- **`assets/video_dataset_template.py`** — VideoDataset with clip sampling, frame padding
- **`assets/video_transforms_template.py`** — All video transforms, pipeline composition
- **`assets/data_manager_template.py`** — DataManager factory, loader construction
- **`assets/distributed_sampler_template.py`** — DistributedWeightedSampler, ConcatIndices
- **`assets/data_config_template.py`** — DataConfig, AugConfig, YAML parsing utilities

### Scripts
- **`scripts/validate_data.py`** — Validates done-when gates
- **`scripts/gen_data_tests.py`** — Generates 100+ pytest test cases
- **`scripts/data_benchmark.py`** — Loading throughput and augmentation overhead benchmarks
