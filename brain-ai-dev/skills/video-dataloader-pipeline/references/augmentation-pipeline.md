# Video Augmentation with torchvision.transforms.v2

## Overview

`torchvision.transforms.v2` is the modern augmentation API introduced in torchvision 0.15+. It extends the original `transforms` module with structured tensor types (`tv_tensors`) that carry semantic meaning — the transforms can then apply spatially consistent operations across all frames of a video clip simultaneously. This is the correct way to augment video data with PyTorch: wrap your clip as `tv_tensors.Video`, apply the transform pipeline once, and get back a consistently augmented clip.

---

## Why transforms.v2 for Video?

The original `torchvision.transforms` module applies transforms frame-by-frame. To get temporally consistent augmentation (same crop, same flip, same color jitter across all T frames), you would need to extract and apply random parameters manually.

`transforms.v2` solves this via the `tv_tensors` type system:

- `tv_tensors.Image` — a 2D image (C, H, W)
- `tv_tensors.Video` — a temporal stack of frames (T, C, H, W)

When a transform receives a `tv_tensors.Video`, it generates random parameters once and applies them to all T frames. This gives you free temporal consistency without any extra code.

---

## Converting Decord Output to Video Tensor

Decord's `get_batch()` returns `(T, H, W, C)` uint8 when the torch bridge is active. The conversion to a training-ready tensor requires three steps:

```python
import torch
from torchvision import tv_tensors

def prepare_video_tensor(frames_thwc: torch.Tensor) -> tv_tensors.Video:
    """
    Convert decord output to a tv_tensors.Video ready for augmentation.

    Args:
        frames_thwc: Tensor of shape (T, H, W, C), dtype uint8, range [0, 255]

    Returns:
        tv_tensors.Video of shape (T, C, H, W), dtype float32, range [0.0, 1.0]
    """
    # Step 1: Channel-last to channel-first
    frames = frames_thwc.permute(0, 3, 1, 2)   # (T, H, W, C) -> (T, C, H, W)

    # Step 2: uint8 [0, 255] -> float32 [0.0, 1.0]
    frames = frames.float() / 255.0

    # Step 3: Wrap as Video type so transforms know it's a temporal sequence
    return tv_tensors.Video(frames)
```

The `tv_tensors.Video` wrapper is a subclass of `torch.Tensor` — it carries no extra data, just a type tag that transforms inspect. You can use it anywhere a regular tensor is expected.

---

## Training Transform Pipeline

```python
from torchvision.transforms import v2

def build_train_transform(cfg) -> v2.Compose:
    """
    Build augmentation pipeline for training.

    Applies random crop, flip, color jitter, and normalization.
    All spatial/color transforms are temporally consistent (same params for all frames).
    """
    return v2.Compose([
        v2.RandomResizedCrop(
            size=cfg.crop_size,
            scale=cfg.crop_scale,         # e.g., (0.5, 1.0)
            ratio=cfg.crop_ratio,          # e.g., (3/4, 4/3)
            interpolation=v2.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        v2.RandomHorizontalFlip(p=cfg.hflip_prob),  # e.g., 0.5
        v2.ColorJitter(
            brightness=cfg.color_jitter[0],   # 0.4
            contrast=cfg.color_jitter[1],     # 0.4
            saturation=cfg.color_jitter[2],   # 0.2
            hue=cfg.color_jitter[3],          # 0.1
        ),
        v2.Normalize(
            mean=list(cfg.normalize_mean),    # [0.485, 0.456, 0.406]
            std=list(cfg.normalize_std),      # [0.229, 0.224, 0.225]
        ),
    ])
```

### Evaluation Transform Pipeline

```python
def build_eval_transform(cfg) -> v2.Compose:
    """
    Build deterministic transform pipeline for validation and testing.

    No random augmentations: resize then center crop, normalize only.
    """
    return v2.Compose([
        v2.Resize(
            size=int(cfg.crop_size * 256 / 224),   # e.g., 256 for crop_size=224
            interpolation=v2.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        v2.CenterCrop(size=cfg.crop_size),
        v2.Normalize(
            mean=list(cfg.normalize_mean),
            std=list(cfg.normalize_std),
        ),
    ])
```

The evaluation pipeline is deterministic: same video always produces the same output. This is essential for reproducible validation metrics.

---

## Transform Behavior with tv_tensors.Video

### RandomResizedCrop

- Samples a random crop area (within `scale` range) with random aspect ratio (within `ratio` range)
- Resizes the crop to the target size
- When applied to `tv_tensors.Video` with shape `(T, C, H, W)`: generates one crop box, applies it to all T frames
- The crop box is the same for all frames — spatially consistent across time

```python
# Effective crop area spans 50%-100% of the frame area
# Aspect ratio spans 0.75 to 1.333 (landscape to portrait)
transform = v2.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(3/4, 4/3))
```

### RandomHorizontalFlip

- Flips the video left-right with probability `p`
- Same decision (flip or not flip) applied to all T frames
- Does NOT apply random vertical flip (horizontal flip is more natural for most video content)

### ColorJitter

- Randomly adjusts brightness, contrast, saturation, and hue
- Parameters are sampled once per clip, applied identically to all T frames
- This prevents the "flickering" artifact that would occur if color were jittered independently per frame
- Ranges: brightness/contrast can be float (symmetric range) or tuple (min, max)

```python
# Each jitter dimension is sampled from a symmetric range:
# brightness: [max(0, 1-0.4), 1+0.4] = [0.6, 1.4]
# contrast:   [max(0, 1-0.4), 1+0.4] = [0.6, 1.4]
# saturation: [max(0, 1-0.2), 1+0.2] = [0.8, 1.2]
# hue:        [-0.1, 0.1]
v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
```

### Normalize

- Subtracts mean, divides by std, per channel
- ImageNet statistics are standard for models pretrained on ImageNet
- Applied to all frames independently (normalization is per-pixel, no spatial dependencies)

```python
# ImageNet statistics
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
```

After normalization, pixel values will be roughly in range [-2.5, 2.5]. Mean should be approximately zero for ImageNet-similar data.

---

## Avoiding Common Mistakes

### Mistake 1: Using PIL-Based Transforms on Video Tensors

The legacy `torchvision.transforms` (v1) module expects PIL Images or 2D tensors. Using them on `(T, C, H, W)` tensors will either raise an error or apply transforms in an unexpected way.

```python
# WRONG: v1 transforms on video tensor
from torchvision import transforms
t = transforms.RandomCrop(224)  # expects 2D input
t(video_tensor)                  # will fail or misbehave

# CORRECT: v2 transforms with Video wrapper
from torchvision.transforms import v2
t = v2.RandomResizedCrop(224)
t(tv_tensors.Video(video_tensor))  # works correctly
```

### Mistake 2: Forgetting to Wrap as Video Type

Without the `tv_tensors.Video` wrapper, some v2 transforms may treat the first dimension as batch rather than time.

```python
# RISKY: may not apply transforms correctly across T
frames = frames.permute(0, 3, 1, 2).float() / 255.0
result = transform(frames)  # transform may treat (T, C, H, W) as batch of images

# SAFE: explicit Video wrapper
video = tv_tensors.Video(frames)
result = transform(video)   # guaranteed temporal consistency
```

### Mistake 3: Applying ColorJitter Per-Frame

If you apply transforms frame-by-frame in a loop, each frame gets different random parameters. This causes flickering that the model must learn to ignore.

```python
# WRONG: per-frame application breaks temporal consistency
frames_aug = torch.stack([transform(frame) for frame in frames_list])

# CORRECT: single-pass on the whole clip
video = tv_tensors.Video(torch.stack(frames_list, dim=0))
frames_aug = transform(video)
```

---

## GPU Augmentation

For large batches, moving the tensor to GPU before transforms can speed up augmentation significantly, particularly for large spatial operations like resizing.

```python
def augment_on_gpu(frames_thwc: torch.Tensor, transform, device: str = 'cuda') -> torch.Tensor:
    """
    Prepare and augment video on GPU.
    """
    video = prepare_video_tensor(frames_thwc)  # still on CPU
    video = video.to(device)                    # move to GPU
    video = transform(video)                    # transform on GPU
    return video  # (T, C, H, W) float32 on GPU
```

Note: GPU augmentation is most beneficial when:
- You have a fast GPU with free compute capacity
- Batch size is large enough to amortize GPU transfer overhead
- The augmentation pipeline includes compute-heavy ops (large resizes, many ColorJitter steps)

For small batches or inference, CPU augmentation is often comparable.

---

## Complete Integration Example

```python
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2
import decord
from decord import VideoReader, cpu

decord.bridge.set_bridge('torch')

MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

def build_train_transform():
    return v2.Compose([
        v2.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(3/4, 4/3), antialias=True),
        v2.RandomHorizontalFlip(0.5),
        v2.ColorJitter(0.4, 0.4, 0.2, 0.1),
        v2.Normalize(mean=list(MEAN), std=list(STD)),
    ])

def load_and_augment(path, indices, transform):
    vr = VideoReader(path, ctx=cpu(0), num_threads=1)
    raw = vr.get_batch(indices)          # (T, H, W, C) uint8
    del vr

    raw = raw.permute(0, 3, 1, 2)        # (T, C, H, W) uint8
    video = tv_tensors.Video(raw.float() / 255.0)  # (T, C, H, W) float32 [0,1]
    augmented = transform(video)          # (T, C, H, W) float32 normalized
    return augmented
```

---

## Summary Table

| Stage | Shape | dtype | Range |
|-------|-------|-------|-------|
| decord output | (T, H, W, C) | uint8 | [0, 255] |
| after permute | (T, C, H, W) | uint8 | [0, 255] |
| after /255.0 | (T, C, H, W) | float32 | [0.0, 1.0] |
| as tv_tensors.Video | (T, C, H, W) | float32 | [0.0, 1.0] |
| after RandomResizedCrop | (T, C, crop_size, crop_size) | float32 | [0.0, 1.0] |
| after ColorJitter | (T, C, crop_size, crop_size) | float32 | [0.0, 1.0] |
| after Normalize | (T, C, crop_size, crop_size) | float32 | ~[-2.5, 2.5] |
