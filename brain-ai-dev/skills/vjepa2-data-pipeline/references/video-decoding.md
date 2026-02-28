# Video Decoding Reference

## decord VideoReader

decord is the primary video decoding library for V-JEPA 2. It provides efficient
random-access frame reading backed by FFmpeg, with optional GPU-accelerated decoding.

### Basic Usage

```python
import decord
from decord import VideoReader, cpu, gpu

# CPU decoding (default)
vr = VideoReader(video_path, ctx=cpu(0))

# GPU decoding (requires decord built with GPU support)
vr = VideoReader(video_path, ctx=gpu(0))

# Read specific frames by index
frames = vr.get_batch(frame_indices)  # returns NDArray [T, H, W, C] uint8

# Convert to numpy
frames = frames.asnumpy()  # shape [T, H, W, C], dtype uint8

# Metadata
total_frames = len(vr)
native_fps   = vr.get_avg_fps()
```

### GPU Decoding Option

GPU decoding can accelerate throughput when the GPU has available decode engines
(NVDEC on NVIDIA, VCN on AMD). The decoded frames land directly in GPU memory,
skipping the CPU->GPU transfer step.

```python
# Check GPU availability before using GPU context
import decord
ctx = decord.gpu(0) if decord.gpu(0) is not None else decord.cpu(0)
vr = VideoReader(video_path, ctx=ctx)
```

Caveats:
- Not all codecs are supported for GPU decode (H.264, HEVC are common)
- GPU decode contexts are not thread-safe; use one context per worker
- In DataLoader workers, always use CPU context to avoid CUDA re-initialization

---

## Clip Sampling Modes

Three mutually exclusive modes determine which frames are selected from a video.
Each mode produces exactly `frames_per_clip` (FPC) frame indices, applying
circulant padding when the video is shorter than required.

### Mode 1: fps (Target FPS Sampling)

Sample frames uniformly at `target_fps`, starting from a random offset.

**Parameters:**
- `target_fps` (int): Desired playback rate, e.g. 10
- `frames_per_clip` (int): Number of frames to return, e.g. 16

**Algorithm:**
```
clip_duration_sec = frames_per_clip / target_fps
# e.g., 16 frames at 10 fps = 1.6 seconds of video

native_fps = vr.get_avg_fps()
frame_step = max(1, round(native_fps / target_fps))

# Total frames needed in native fps
native_frames_needed = frame_step * frames_per_clip

# Random start within valid range
max_start = max(0, total_frames - native_frames_needed)
start = random.randint(0, max_start)

frame_indices = [start + i * frame_step for i in range(frames_per_clip)]
```

**Use case:** Standard pretraining where temporal coverage per clip is important.

### Mode 2: duration (Fixed Duration Sampling)

Sample a fixed-duration window from the video at uniform spacing.

**Parameters:**
- `clip_duration_sec` (float): Window length in seconds, e.g. 3.2
- `frames_per_clip` (int): Number of frames to sample from that window

**Algorithm:**
```
native_fps = vr.get_avg_fps()
clip_length_frames = int(clip_duration_sec * native_fps)

# Random start within valid range
max_start = max(0, total_frames - clip_length_frames)
start = random.randint(0, max_start)
end = start + clip_length_frames

# Uniformly sample FPC frames from [start, end)
frame_indices = np.linspace(start, end - 1, frames_per_clip, dtype=int).tolist()
```

**Use case:** Downstream tasks where temporal receptive field must match inference.

### Mode 3: frame_step (Fixed Step Sampling)

Sample frames with a fixed stride between consecutive frames.

**Parameters:**
- `frame_step` (int): Stride between frames, e.g. 4
- `frames_per_clip` (int): Number of frames to return

**Algorithm:**
```
native_frames_needed = frame_step * (frames_per_clip - 1) + 1

# Random start within valid range
max_start = max(0, total_frames - native_frames_needed)
start = random.randint(0, max_start)

frame_indices = [start + i * frame_step for i in range(frames_per_clip)]
```

**Use case:** Action recognition where motion speed normalization is not required.

---

## Circulant Frame Padding

When a video has fewer frames than required by the clip sampling mode, circulant
(cyclic) padding wraps the video indices to fill the remaining slots.

### Algorithm

```python
def circulant_pad(frame_indices: List[int], total_frames: int) -> List[int]:
    """
    Clamp indices into [0, total_frames-1] using modular arithmetic.
    This creates a looping effect for short videos.
    """
    return [idx % total_frames for idx in frame_indices]
```

### Example

Video has 5 frames, FPC=8, computed indices = [0, 2, 4, 6, 8, 10, 12, 14]

After circulant padding (mod 5): [0, 2, 4, 1, 3, 0, 2, 4]

The video loops seamlessly without duplicate static frames at boundaries.

### Implementation Notes

- Apply circulant padding AFTER computing frame_indices, before calling `vr.get_batch()`
- Clamp indices to [0, total_frames - 1] first; if total_frames is 0, return zeros tensor
- For empty videos (0 frames), return a zero tensor of the correct shape

```python
def safe_pad_indices(frame_indices: List[int], total_frames: int) -> List[int]:
    if total_frames == 0:
        return [0] * len(frame_indices)
    return [max(0, min(idx, total_frames - 1)) if idx < total_frames
            else idx % total_frames
            for idx in frame_indices]
```

---

## Frame Indices Computation

Complete reference implementation combining mode dispatch and padding:

```python
import random
import numpy as np
from typing import List

def compute_frame_indices(
    total_frames: int,
    frames_per_clip: int,
    clip_mode: str,
    target_fps: float = 10.0,
    native_fps: float = 30.0,
    clip_duration_sec: float = 3.2,
    frame_step: int = 4,
) -> List[int]:
    """
    Compute frame indices for a single clip, with circulant padding for
    videos shorter than required.

    Returns a list of length `frames_per_clip`.
    """
    if clip_mode == "fps":
        step = max(1, round(native_fps / target_fps))
        needed = step * frames_per_clip
        max_start = max(0, total_frames - needed)
        start = random.randint(0, max_start)
        indices = [start + i * step for i in range(frames_per_clip)]

    elif clip_mode == "duration":
        clip_len = max(1, int(clip_duration_sec * native_fps))
        max_start = max(0, total_frames - clip_len)
        start = random.randint(0, max_start)
        end = start + clip_len
        indices = np.linspace(start, end - 1, frames_per_clip, dtype=int).tolist()

    elif clip_mode == "frame_step":
        needed = frame_step * (frames_per_clip - 1) + 1
        max_start = max(0, total_frames - needed)
        start = random.randint(0, max_start)
        indices = [start + i * frame_step for i in range(frames_per_clip)]

    else:
        raise ValueError(f"Unknown clip_mode: {clip_mode!r}. "
                         f"Expected one of 'fps', 'duration', 'frame_step'.")

    # Apply circulant padding for short videos
    if total_frames > 0:
        indices = [idx % total_frames for idx in indices]
    else:
        indices = [0] * frames_per_clip

    return indices
```

---

## Performance Considerations

### Worker Count
- Use `num_workers >= 4` for GPU training; 8-16 is typical
- Each worker maintains its own `VideoReader` instance
- Avoid sharing `VideoReader` across processes (not fork-safe)

### Prefetch Factor
- `prefetch_factor=2` is the DataLoader default
- For high-resolution video, increase to 4 to hide I/O latency

### Video Caching
- decord caches decoded frames internally per VideoReader instance
- For small datasets that fit in RAM, consider caching frames as numpy arrays

### File Format Recommendations
- MP4 with H.264 encoding provides best decord compatibility
- Avoid variable frame rate (VFR) videos; prefer constant frame rate (CFR)
- Pre-segment long videos into shorter clips at preprocessing time

### Thread Safety
- `VideoReader` is NOT thread-safe; always create per-worker
- Use `worker_init_fn` to initialize per-worker state

```python
def worker_init_fn(worker_id: int):
    worker_info = torch.utils.data.get_worker_info()
    # Each worker gets its own random state
    seed = worker_info.seed % (2**31)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
```
