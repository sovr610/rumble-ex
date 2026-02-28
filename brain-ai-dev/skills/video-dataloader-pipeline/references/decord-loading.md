# Decord VideoReader for Video Deep Learning

## Overview

Decord is a high-performance video loading library designed for deep learning workflows. It provides `VideoReader`, a random-access video decoder that integrates directly with PyTorch and NumPy via a bridge interface. The key design philosophy is decode-only-what-you-need: unlike sequential decoders, `VideoReader` can jump to arbitrary frames using keyframe seek, making it ideal for sparse temporal sampling that most video understanding models require.

This document covers the complete API surface, frame sampling strategies, metadata caching patterns, multi-worker safety, and performance considerations for production video pipelines.

---

## Installation and Imports

```python
# Install decord (CPU-only build recommended for DataLoader compatibility)
# pip install decord

import decord
from decord import VideoReader, cpu, gpu
import torch
import numpy as np
```

---

## Bridge Setup

The bridge controls what data type `get_batch()` and frame access return. Set the bridge once at module level — never per-call, as repeated bridge switching has overhead and can cause issues in multi-threaded contexts.

```python
# At the TOP of your module, before any VideoReader usage
decord.bridge.set_bridge('torch')
# Now get_batch() returns torch.Tensor directly instead of NDArray
```

Available bridges:
- `'native'` — returns `decord.ndarray.NDArray` (decord's own type)
- `'numpy'` — returns `numpy.ndarray`
- `'torch'` — returns `torch.Tensor` (recommended for PyTorch pipelines)
- `'mxnet'` — returns MXNet NDArray (rarely used)

Setting the bridge to `'torch'` at module load time is preferred over the `'numpy'` bridge even if you plan to convert to tensors afterward, because it avoids the intermediate NumPy allocation.

---

## VideoReader Constructor

```python
vr = VideoReader(
    path,                    # str or path-like: video file path
    ctx=cpu(0),              # decoding context: cpu(0) or gpu(0)
    num_threads=1,           # decoder threads (1 recommended in workers)
    width=-1,                # optional: resize width on decode (-1 = native)
    height=-1,               # optional: resize height on decode (-1 = native)
)
```

### Context Selection

**Use `ctx=cpu(0)` in DataLoader workers.** The GPU context (`ctx=gpu(0)`) uses CUDA for hardware decode but has known compatibility issues with PyTorch DataLoader's multi-process forking. Specifically:

- CUDA contexts cannot be cleanly forked across processes
- GPU decode is faster for high-resolution video but breaks with `num_workers > 0`
- Recommended pattern: CPU decode + GPU augmentation (move tensor after decoding)

```python
# CORRECT: CPU decode, GPU augmentation
vr = VideoReader(path, ctx=cpu(0), num_threads=1)
frames = vr.get_batch(indices)   # returns CPU tensor
frames = frames.to('cuda')       # move to GPU for augmentation
```

### Width/Height Resize

Decord can resize during decode, which is slightly faster than post-decode resize for large videos:

```python
# Resize to 256px height, maintain aspect ratio (width=-1 keeps proportional)
vr = VideoReader(path, ctx=cpu(0), num_threads=1, height=256)
```

This is useful for a two-stage decode: resize to 256 during decode, then random crop to 224 in augmentation. Avoids unnecessarily large intermediate tensors.

---

## Frame Access

### Single Frame Access

```python
frame = vr[i]   # returns (H, W, C) uint8 tensor (with torch bridge)
```

Single frame access is convenient but avoid it in loops — each call may trigger a separate seek. Use `get_batch()` for multiple frames.

### Batch Frame Access

```python
indices = [0, 4, 8, 12, 16, 20]   # list or array of frame indices
frames = vr.get_batch(indices)      # returns (T, H, W, C) uint8 tensor
```

`get_batch()` is the primary API for training. Key behaviors:
- Accepts any sequence of indices (lists, NumPy arrays, Python lists)
- Accepts duplicate indices (useful for zero-padding short videos)
- Internally sorts and seeks efficiently — random access is keyframe-based
- Returns shape `(T, H, W, C)` where T = len(indices)
- dtype is uint8 (pixel values 0-255)

### Metadata Access

```python
total_frames = len(vr)             # int: total frame count
fps = vr.get_avg_fps()            # float: average frames per second
duration = len(vr) / vr.get_avg_fps()  # float: duration in seconds
key_indices = vr.get_key_indices() # list: keyframe positions
```

---

## Temporal Sampling Strategies

### Strategy 1: Uniform Sampling with Fixed Stride

Sample every `stride` frames starting from frame 0. Deterministic, good for evaluation.

```python
def sample_uniform(total_frames, num_frames, stride):
    indices = list(range(0, total_frames, stride))[:num_frames]
    # Pad if video is too short
    while len(indices) < num_frames:
        indices.append(indices[-1])
    return indices
```

Use case: evaluation clips, where reproducibility matters more than diversity.

### Strategy 2: Random Start with Fixed Stride (Primary Training Strategy)

Sample `num_frames` frames with `stride` gap, but randomize the starting position. This is the standard approach for action recognition.

```python
import random

def sample_random_start(total_frames, num_frames, stride):
    span = num_frames * stride              # total temporal window needed
    max_start = max(0, total_frames - span)
    start = random.randint(0, max_start)   # uniform random start
    indices = [start + i * stride for i in range(num_frames)]
    # Clamp all indices to valid range [0, total_frames - 1]
    indices = [min(idx, total_frames - 1) for idx in indices]
    return indices
```

This provides temporal diversity: across multiple epochs, the model sees different segments of each video.

### Strategy 3: Clip-Level Random Sampling

For datasets with multiple clips per video (e.g., Kinetics-style), randomly sample a clip boundary and then apply random-start within it.

```python
def sample_clip(total_frames, num_frames, stride, clip_idx=None, num_clips=10):
    span = num_frames * stride
    clip_length = total_frames // num_clips
    clip_start = (clip_idx or random.randint(0, num_clips - 1)) * clip_length
    clip_end = clip_start + clip_length
    max_start = max(clip_start, clip_end - span)
    start = random.randint(clip_start, max_start)
    indices = [start + i * stride for i in range(num_frames)]
    return [min(max(idx, 0), total_frames - 1) for idx in indices]
```

### Complete Frame Sampling Algorithm (Production)

```python
def compute_frame_indices(total_frames: int, num_frames: int, stride: int) -> list:
    """
    Standard temporal sampling: random start with fixed stride.

    Args:
        total_frames: total number of frames in the video
        num_frames: number of frames to sample per clip
        stride: temporal distance between consecutive sampled frames

    Returns:
        list of frame indices, clamped to [0, total_frames - 1]
    """
    span = num_frames * stride
    max_start = max(0, total_frames - span)
    start = random.randint(0, max_start)
    indices = [start + i * stride for i in range(num_frames)]
    # Clamp to valid range — handles short videos gracefully
    indices = [min(idx, total_frames - 1) for idx in indices]
    return indices
```

Edge case handling:
- If `total_frames < num_frames * stride`: `max_start = 0`, so start is always 0. Clamping ensures all indices are valid.
- Very short videos (total_frames < num_frames): indices will repeat the last valid frame.
- Single-frame videos: all indices clamp to 0.

---

## Metadata Caching

Scanning a large video dataset at training startup is expensive. Cache metadata to a JSON manifest file.

### Scanning and Building the Manifest

```python
import os
import json
from dataclasses import dataclass, asdict
from typing import List

@dataclass
class VideoMeta:
    path: str
    num_frames: int
    fps: float
    label: int

def scan_manifest(data_root: str, split: str, cache_path: str = None) -> List[VideoMeta]:
    """
    Walk data_root/split, open each video briefly to get metadata,
    and cache to JSON for fast restart.
    """
    manifest_path = cache_path or os.path.join(data_root, f"{split}_manifest.json")

    # Load cached manifest if available
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            data = json.load(f)
        return [VideoMeta(**item) for item in data]

    manifest = []
    split_dir = os.path.join(data_root, split)

    for label_name in sorted(os.listdir(split_dir)):
        label_dir = os.path.join(split_dir, label_name)
        if not os.path.isdir(label_dir):
            continue
        label_id = sorted(os.listdir(split_dir)).index(label_name)

        for fname in sorted(os.listdir(label_dir)):
            if not fname.endswith(('.mp4', '.avi', '.mkv', '.webm')):
                continue
            fpath = os.path.join(label_dir, fname)
            try:
                vr = VideoReader(fpath, ctx=cpu(0), num_threads=1)
                meta = VideoMeta(
                    path=fpath,
                    num_frames=len(vr),
                    fps=float(vr.get_avg_fps()),
                    label=label_id,
                )
                manifest.append(meta)
                del vr  # important: release file handle immediately
            except Exception as e:
                print(f"Warning: skipping {fpath}: {e}")

    # Serialize to JSON for fast restart
    with open(manifest_path, 'w') as f:
        json.dump([asdict(m) for m in manifest], f)

    return manifest
```

**Why cache metadata separately?** Opening 100,000 video files at startup just to get frame counts takes 2-5 minutes even with fast NVMe. After caching, restart takes milliseconds. The JSON manifest also allows filtering (e.g., remove videos with < 32 frames) without re-scanning.

---

## Multi-Worker Safety

This is the most critical practical concern when using decord with PyTorch DataLoader.

### The File Handle Leak Problem

**Never create VideoReader in `__init__`.**

```python
# WRONG: Creates VideoReader in __init__
class BadDataset(Dataset):
    def __init__(self, manifest):
        self.readers = [VideoReader(m.path, ctx=cpu(0)) for m in manifest]
        # Each worker will have len(manifest) open file handles!
        # With num_workers=4, that's 4x your dataset size in open files.
```

```python
# CORRECT: Create VideoReader per __getitem__ call
class GoodDataset(Dataset):
    def __init__(self, manifest):
        self.manifest = manifest
        # No VideoReader here!

    def __getitem__(self, idx):
        meta = self.manifest[idx]
        # Create fresh reader per item — file handle opens and closes within this call
        vr = VideoReader(meta.path, ctx=cpu(0), num_threads=1)
        frames = vr.get_batch(indices)
        del vr  # explicit deletion (or use with context if available)
        return frames
```

### Thread Count in Workers

Each DataLoader worker is a separate process with its own Python interpreter. Inside a worker, decord can still use internal threads for I/O. However:

- `num_threads=1` is recommended in workers to avoid thread contention between workers
- `num_threads=4` in a single-process pipeline (no DataLoader workers) can be beneficial
- The bottleneck is typically disk I/O, not CPU decode, so more threads rarely helps in workers

```python
# In DataLoader workers: single-threaded decord
vr = VideoReader(path, ctx=cpu(0), num_threads=1)
```

### Worker Initialization Function

For one-time setup per worker (e.g., confirming bridge is set), use `worker_init_fn`:

```python
def worker_init_fn(worker_id):
    import decord
    decord.bridge.set_bridge('torch')
    # Each worker independently sets the bridge
    # numpy.random seed is automatically set differently per worker
    worker_seed = torch.initial_seed() % 2**32
    import random
    random.seed(worker_seed)

loader = DataLoader(dataset, num_workers=4, worker_init_fn=worker_init_fn)
```

---

## Performance Pitfalls

### Pitfall 1: Repeated VideoReader Initialization

After thousands of VideoReader opens, initialization latency increases (tracked in decord issue #280). This manifests as training throughput degrading over a long epoch.

Mitigation: Use persistent workers (`persistent_workers=True`). Each worker processes many items before the process is recycled, spreading the per-worker init cost.

### Pitfall 2: Setting Bridge Per-Call

```python
# WRONG: setting bridge inside __getitem__
def __getitem__(self, idx):
    decord.bridge.set_bridge('torch')  # called thousands of times!
    ...
```

The bridge is a global state. Set it once at module import time. Resetting it per-call adds overhead and is thread-unsafe.

### Pitfall 3: Single-Frame Loop Instead of get_batch

```python
# SLOW: individual frame access in a loop
frames = [vr[i] for i in indices]  # N separate seeks and decodes

# FAST: batch access
frames = vr.get_batch(indices)  # single optimized decode pass
```

`get_batch()` internally optimizes seek order and can reuse keyframe data across nearby indices. Using a loop for multi-frame access can be 5-10x slower.

### Pitfall 4: GPU Context in Workers

```python
# BREAKS: GPU context with num_workers > 0
vr = VideoReader(path, ctx=gpu(0))  # CUDA fork issue
```

The CUDA context cannot be safely forked. With `num_workers > 0`, this will either crash or produce corrupt frames. Always use `ctx=cpu(0)` in workers.

### Pitfall 5: Not Clamping Indices

Short videos (fewer frames than `num_frames * stride`) will produce out-of-bounds indices if not clamped. Decord raises an exception for out-of-bounds access.

```python
# Always clamp after computing indices
indices = [min(idx, total_frames - 1) for idx in indices]
```

---

## Output Format Reference

With `bridge='torch'`:

| Method | Output Shape | dtype | Range |
|--------|-------------|-------|-------|
| `vr[i]` | `(H, W, C)` | uint8 | 0-255 |
| `vr.get_batch(indices)` | `(T, H, W, C)` | uint8 | 0-255 |

Note the channel-last format. For PyTorch models expecting NCHW, you must permute after decoding:

```python
frames = vr.get_batch(indices)   # (T, H, W, C) uint8
frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W) uint8
frames = frames.float() / 255.0      # (T, C, H, W) float32 [0, 1]
```

---

## Complete Working Example

```python
import decord
from decord import VideoReader, cpu
import torch
import random

# Set bridge at module level
decord.bridge.set_bridge('torch')

def load_video_clip(path: str, num_frames: int = 16, stride: int = 4) -> torch.Tensor:
    """
    Load a temporally sampled clip from a video file.

    Returns:
        Tensor of shape (T, C, H, W), float32, range [0, 1]
    """
    vr = VideoReader(path, ctx=cpu(0), num_threads=1)
    total = len(vr)

    span = num_frames * stride
    max_start = max(0, total - span)
    start = random.randint(0, max_start)
    indices = [start + i * stride for i in range(num_frames)]
    indices = [min(idx, total - 1) for idx in indices]

    frames = vr.get_batch(indices)         # (T, H, W, C) uint8
    del vr

    frames = frames.permute(0, 3, 1, 2)   # (T, C, H, W) uint8
    frames = frames.float() / 255.0        # (T, C, H, W) float32 [0, 1]
    return frames
```

---

## Summary of Key Rules

1. Set `decord.bridge.set_bridge('torch')` once at module level
2. Use `ctx=cpu(0)` always in DataLoader workers (GPU context breaks fork)
3. Use `num_threads=1` inside worker processes
4. Never create `VideoReader` in `__init__` — only in `__getitem__`
5. Always clamp frame indices to `[0, total_frames - 1]`
6. Prefer `get_batch(indices)` over looping `vr[i]`
7. Delete the `VideoReader` object after use (or let it go out of scope)
8. Cache metadata to JSON to avoid expensive startup scans
