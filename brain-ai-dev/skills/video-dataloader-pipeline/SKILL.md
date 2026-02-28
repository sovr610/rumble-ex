---
name: Video DataLoader Pipeline (Decord + tf.data)
description: >
  This skill should be used when the user asks to "build a video dataloader",
  "create video dataset class", "load video frames with decord",
  "decord VideoReader dataset", "video DataLoader pipeline",
  "sparse frame sampling for training", "temporal stride video loading",
  "TFRecord video converter", "tf.data video pipeline for TPU",
  "video TFRecord sharded pipeline", "TPU video data pipeline",
  "video DataModule", "PyTorch Lightning video DataModule",
  "distributed video sampler", "video augmentation pipeline",
  "torchvision transforms v2 video", "random resized crop video",
  "video color jitter augmentation", "GPU video augmentation",
  "multi-backend video loader", "decord to torch bridge",
  "video worker configuration", "persistent workers video",
  "pin_memory video DataLoader", "video prefetch pipeline",
  or needs guidance on building production video ingestion pipelines
  with decord-based GPU training loaders and tf.data-based TPU pipelines.
version: 0.1.0
---

# Video DataLoader Pipeline (Decord + tf.data)

## Overview

Generate a complete multi-backend video ingestion pipeline supporting both PyTorch (GPU training via decord) and tf.data (TPU training via TFRecords). The PyTorch path uses `decord.VideoReader` for efficient random-access frame decoding with GPU-accelerated augmentations via `torchvision.transforms.v2`. The tf.data path converts videos to sharded TFRecords and builds a high-throughput `tf.data.Dataset` pipeline optimized for TPU pod training. A PyTorch Lightning `DataModule` wraps the PyTorch path with distributed sampler, worker scaling, and train/val/test split management.

Design principle: **decode only what is needed, augment on GPU, shard for TPU, configure by dataclass.**

## Public Contract

### VideoDataset (PyTorch)

```python
class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, manifest: List[VideoMeta], cfg: VideoDatasetConfig,
                 transform: Optional[Callable] = None): ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> Dict[str, Tensor]: ...
```

### VideoDataModule (Lightning)

```python
class VideoDataModule(L.LightningDataModule):
    def __init__(self, cfg: DataModuleConfig): ...
    def prepare_data(self) -> None: ...
    def setup(self, stage: Optional[str] = None) -> None: ...
    def train_dataloader(self) -> DataLoader: ...
    def val_dataloader(self) -> DataLoader: ...
    def test_dataloader(self) -> DataLoader: ...
```

### TFRecordConverter

```python
class TFRecordConverter:
    def __init__(self, cfg: TFRecordConfig): ...
    def convert_split(self, manifest: List[VideoMeta],
                      output_dir: str, num_shards: int) -> None: ...
```

### TFDataPipeline

```python
def build_tf_video_pipeline(
    shard_pattern: str, cfg: TFPipelineConfig,
    batch_size: int, is_training: bool,
) -> tf.data.Dataset: ...
```

## Key Concepts

### Decord Frame Loading

Use `decord.VideoReader` with `ctx=cpu(0)` for CPU decoding (GPU context has DataLoader compatibility issues). The critical API:

```
vr = VideoReader(path, ctx=cpu(0), num_threads=1)
frames = vr.get_batch(frame_indices)   # decode ONLY requested frames
```

Set `decord.bridge.set_bridge('torch')` once at module level to get tensors directly. `get_batch()` handles duplicate indices internally and optimizes seek patterns.

**Frame sampling procedure** per clip:
1. Read total frames: `total = len(vr)`
2. Compute span: `span = num_frames * stride`
3. Sample random start: `start = randint(0, max(0, total - span))`
4. Build indices: `[start + i * stride for i in range(num_frames)]`
5. Clamp to `[0, total - 1]`
6. Call `vr.get_batch(indices)` — shape `(T, H, W, C)` uint8

### Metadata Caching

On init, scan all video paths and cache `(path, num_frames, fps, duration)` in a list. Avoid opening `VideoReader` in `__init__` — open per-`__getitem__` call to prevent file handle leaks in multi-worker DataLoaders. Optionally serialize the manifest to JSON for fast restarts.

### GPU-Accelerated Augmentation

After decoding, convert `(T, H, W, C)` uint8 to `(T, C, H, W)` float32 `[0, 1]`, then apply `torchvision.transforms.v2`:

| Transform | Parameters | Notes |
|-----------|-----------|-------|
| `RandomResizedCrop` | size=224, scale=(0.5, 1.0), ratio=(3/4, 4/3) | Spatial crop per-clip (same crop all frames) |
| `RandomHorizontalFlip` | p=0.5 | Same flip all frames |
| `ColorJitter` | brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1 | Same jitter all frames |

Wrap the `(T, C, H, W)` tensor as `torchvision.tv_tensors.Video` before passing to transforms so all spatial transforms apply consistently across the temporal dimension.

### TFRecord Conversion

Convert videos to sharded TFRecords for TPU training:

1. For each video, decode frames using decord
2. Encode clip as raw uint8 bytes (or JPEG-encode per frame for compression)
3. Write `tf.train.Example` with features: `video_bytes`, `label`, `num_frames`, `height`, `width`
4. Distribute across shards — rule of thumb: **10x files per TPU host**, each **100MB+**

### tf.data Pipeline for TPU

```
files = tf.data.Dataset.list_files(shard_pattern, shuffle=True)
dataset = files.interleave(
    tf.data.TFRecordDataset,
    cycle_length=tf.data.AUTOTUNE,
    num_parallel_calls=tf.data.AUTOTUNE
)
dataset = dataset.shuffle(10000)
dataset = dataset.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(batch_size, drop_remainder=True)
dataset = dataset.prefetch(tf.data.AUTOTUNE)
```

The `parse_fn` decodes `tf.io.FixedLenFeature` bytes, reshapes to `(T, H, W, C)`, casts to float32, normalizes, and applies augmentations via `tf.image.*`.

For TPU pod training, use `tf.distribute.Strategy`'s auto-sharding or explicit per-replica shard assignment.

### DataModule Configuration

The `VideoDataModule` wraps train/val/test splits with:

| Setting | Value | Rationale |
|---------|-------|-----------|
| `num_workers` | `4 * num_gpus` | Overlap decode with training |
| `pin_memory` | `True` | Faster host-to-device transfer |
| `persistent_workers` | `True` | Avoid worker respawn overhead |
| `drop_last` | `True` (train) | Even batch sizes for distributed |
| `prefetch_factor` | `2` | Default, increase if IO-bound |

Lightning auto-injects `DistributedSampler` — do **not** add one manually. The `prepare_data()` method handles one-time manifest generation; `setup(stage)` creates dataset instances per split.

## Configuration Surface

```python
@dataclass
class VideoDatasetConfig:
    num_frames: int = 16               # Frames per clip
    stride: int = 4                    # Frame skip between samples
    crop_size: int = 224               # Spatial resolution after crop
    crop_scale: Tuple[float, float] = (0.5, 1.0)
    crop_ratio: Tuple[float, float] = (0.75, 1.333)
    hflip_prob: float = 0.5
    color_jitter: Tuple[float, ...] = (0.4, 0.4, 0.2, 0.1)
    normalize_mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, ...] = (0.229, 0.224, 0.225)

@dataclass
class DataModuleConfig:
    data_root: str = ""
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    batch_size: int = 8
    num_workers_per_gpu: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2

@dataclass
class TFRecordConfig:
    num_shards: int = 256
    num_frames: int = 16
    stride: int = 4
    crop_size: int = 224
    compression: str = "none"          # "none" | "jpeg"
    jpeg_quality: int = 95

@dataclass
class TFPipelineConfig:
    shuffle_buffer: int = 10000
    num_frames: int = 16
    crop_size: int = 224
    augment: bool = True
```

## Done-When Gates

1. **PyTorch Path Loads Clips** — `VideoDataset.__getitem__()` returns `(T, C, H, W)` float32 tensor in `[0, 1]` with correct shape. Augmentations apply consistently across temporal dimension. DataLoader with `num_workers > 0` produces batches without deadlock or file handle leaks.
2. **TFRecord Round-Trip Works** — `TFRecordConverter` writes sharded files. `build_tf_video_pipeline()` reads them back, producing `(batch, T, H, W, C)` float32 tensors matching the original clip dimensions and value range.
3. **DataModule Integrates with Trainer** — `VideoDataModule` provides train/val/test dataloaders. Lightning Trainer runs `fit()` for at least 2 steps without error. Distributed sampler is auto-injected (not manually added).

## Resources

### Reference Files
- **`references/decord-loading.md`** — VideoReader API, frame sampling strategies, bridge setup, metadata caching, multi-worker safety, performance pitfalls
- **`references/augmentation-pipeline.md`** — torchvision.transforms.v2 for video, tv_tensors.Video wrapping, per-clip consistent transforms, GPU augmentation, normalization
- **`references/tfrecord-pipeline.md`** — TFRecord schema, shard sizing rules, conversion procedure, tf.data pipeline construction, TPU pod interleaving, parse_fn implementation
- **`references/datamodule-distributed.md`** — Lightning DataModule lifecycle, distributed sampler auto-injection, worker scaling formula, pin_memory/persistent_workers, multi-GPU gotchas
- **`references/testing-matrix.md`** — Test scenarios for all components

### Asset Files
- **`assets/video_dataset_template.py`** — VideoDataset with decord loading, frame sampling, metadata caching, self-tests
- **`assets/augmentation_template.py`** — Video augmentation pipeline with torchvision.transforms.v2, tv_tensors.Video, self-tests
- **`assets/tfrecord_converter_template.py`** — TFRecordConverter with sharded writing, JPEG compression option, self-tests
- **`assets/tf_pipeline_template.py`** — build_tf_video_pipeline with interleaving, parse_fn, TPU-ready, self-tests
- **`assets/datamodule_template.py`** — VideoDataModule with worker scaling, distributed setup, self-tests
- **`assets/video_config_template.py`** — All config dataclasses, validation, serialization

### Scripts
- **`scripts/validate_video_pipeline.py`** — Validates done-when gates
- **`scripts/gen_video_tests.py`** — Generates pytest test cases
- **`scripts/throughput_benchmark.py`** — Measures frames/sec for decord vs tf.data paths
