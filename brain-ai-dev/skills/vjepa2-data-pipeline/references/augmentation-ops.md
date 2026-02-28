# Augmentation Operations Reference

## Transform Pipeline Order

```
Raw Video [T, H, W, C] uint8
    -> RandomResizedCrop  (spatial crop with scale/aspect jitter)
    -> MotionShift        (temporal jitter of crop position across frames)
    -> HorizontalFlip     (50% probability, disabled for robotics)
    -> [Optional] RandAugment (per-frame: shear/translate/rotate/color)
    -> [Optional] RandomErasing (cube mode for temporal consistency)
    -> ClipToTensor       ([T, H, W, C] list -> float [C, T, H, W])
    -> Normalize          (ImageNet mu/sigma)
```

Each transform operates on a list of PIL Images (one per frame) EXCEPT
`ClipToTensor` and `Normalize` which operate on tensors.

---

## RandomResizedCrop

Randomly crops and resizes each frame to a fixed spatial size using the same
spatial crop parameters for all frames in the clip (temporal consistency).

**Parameters:**
- `size` (int or tuple): Output spatial size, e.g. 224 or (224, 224)
- `scale` (tuple[float, float]): Fraction of area to crop, e.g. (0.3, 1.0)
- `ratio` (tuple[float, float]): Aspect ratio range, e.g. (0.75, 1.33)
- `interpolation`: PIL interpolation mode, default `Image.BILINEAR`

**Algorithm:**
```python
# Sample ONE set of crop parameters for the entire clip
i, j, h, w = torchvision.transforms.RandomResizedCrop.get_params(
    frames[0], scale=scale, ratio=ratio
)
# Apply the same crop to ALL frames
cropped = [F.resized_crop(frame, i, j, h, w, size, interpolation)
           for frame in frames]
```

**Key property:** The same (i, j, h, w) crop box is applied to every frame,
preserving temporal coherence. Each clip gets a different random crop.

**Scale guidance:**
- Pretraining: (0.3, 1.0) — aggressive crops encourage invariance
- Fine-tuning: (0.5, 1.0) — less aggressive
- Robotics: (0.9, 1.0) — near-fixed scale

---

## MotionShift

Applies a temporal jitter to the spatial crop position across frames, simulating
natural camera motion and improving robustness to viewpoint change.

**Parameters:**
- `max_shift_frac` (float): Maximum shift as fraction of frame size, e.g. 0.1
- Applied AFTER RandomResizedCrop, BEFORE HorizontalFlip

**Algorithm:**
```python
# For each frame t, compute a small spatial offset
shift_x = random.uniform(-max_shift_frac, max_shift_frac) * width
shift_y = random.uniform(-max_shift_frac, max_shift_frac) * height

# Apply affine translation to each frame independently
# The shift varies smoothly across the clip to mimic camera motion
shifts = np.linspace(
    (-shift_x, -shift_y),
    (shift_x, shift_y),
    num_frames
)
result = [F.affine(frame, angle=0, translate=shift, scale=1.0, shear=0)
          for frame, shift in zip(frames, shifts)]
```

**When to disable:** Robotics tasks where precise spatial alignment is critical.
Set `motion_shift=False` in `AugConfig`.

---

## HorizontalFlip

Flips all frames horizontally with probability p=0.5.

**Parameters:**
- `p` (float): Flip probability, default 0.5

**Algorithm:**
```python
if random.random() < p:
    frames = [F.hflip(frame) for frame in frames]
```

**Key property:** Same flip decision applied to ALL frames (temporal consistency).

**Robotics usage:** Set `horizontal_flip=False` in `AugConfig`. Direction-sensitive
tasks (robot arm control, navigation) break when left/right are swapped.

---

## RandAugment (Per-Frame)

Applies n random augmentation operations per frame, each with magnitude m.
Operations are applied INDEPENDENTLY per frame for temporal variety.

**Parameters:**
- `n` (int): Number of operations to apply, e.g. 2
- `m` (int): Magnitude of operations on scale 0-30, e.g. 9

**Available Operations:**

| Operation | Description | Magnitude Effect |
|-----------|-------------|-----------------|
| `ShearX` | Horizontal shear transform | Shear angle |
| `ShearY` | Vertical shear transform | Shear angle |
| `TranslateX` | Horizontal translation | Pixel offset |
| `TranslateY` | Vertical translation | Pixel offset |
| `Rotate` | Rotation around center | Degrees |
| `AutoContrast` | Maximize contrast | N/A (binary) |
| `Equalize` | Histogram equalization | N/A (binary) |
| `Solarize` | Invert pixels above threshold | Threshold |
| `Color` | Color saturation adjustment | Saturation factor |
| `Contrast` | Contrast adjustment | Contrast factor |
| `Brightness` | Brightness adjustment | Brightness factor |
| `Sharpness` | Sharpness adjustment | Sharpness factor |
| `Posterize` | Reduce color bits | Bits |
| `Cutout` | Random square masking | Square size |

**Implementation:**
```python
class RandAugmentVideo:
    def __init__(self, n: int = 2, m: int = 9):
        self.n = n
        self.m = m
        self.ops = self._build_ops()

    def __call__(self, frames: List[Image.Image]) -> List[Image.Image]:
        # Each frame gets its own independent augmentation
        return [self._augment_frame(f) for f in frames]

    def _augment_frame(self, frame: Image.Image) -> Image.Image:
        ops = random.sample(self.ops, self.n)
        for op in ops:
            frame = op(frame, self.m)
        return frame
```

**Note:** When `auto_augment=False`, normalization uses mean/std scaled to [0, 255]
range instead of the normalized [0,1] ImageNet values.

---

## RandomErasing

Erases a random rectangular region of the video. In "cube" mode, the same
(x, y, h, w) region is erased from ALL frames for temporal consistency.

**Parameters:**
- `p` (float): Probability of erasing, e.g. 0.25
- `scale` (tuple): Fraction of image area to erase, e.g. (0.02, 0.33)
- `ratio` (tuple): Aspect ratio of erased region, e.g. (0.3, 3.3)
- `value` (float or str): Fill value; 0 for black, 'random' for noise
- `cube_mode` (bool): Apply same region to all frames (default True)

**Cube Mode Algorithm:**
```python
# Compute erase box from FIRST frame
i, j, h, w, v = torchvision.transforms.RandomErasing.get_params(
    tensor[0], scale=scale, ratio=ratio, value=value
)
# Apply same box to ALL frames
tensor[:, :, i:i+h, j:j+w] = v
```

**Non-cube mode** (per-frame): Each frame gets independent erasing. Less common.

**Applied on tensor** (after ClipToTensor), operating on [C, T, H, W] input.

---

## ClipToTensor

Converts a list of PIL Images (or numpy arrays) to a float PyTorch tensor
with the channel dimension moved to first position.

**Input:** `List[PIL.Image]` of length T, each H x W x C uint8
**Output:** `torch.Tensor` of shape [C, T, H, W], float32, values in [0, 1]

**Implementation:**
```python
class ClipToTensor:
    def __call__(self, frames: List[Image.Image]) -> torch.Tensor:
        # Stack: [T, H, W, C] uint8
        arr = np.stack([np.array(f) for f in frames], axis=0)
        # To float [0, 1]
        arr = arr.astype(np.float32) / 255.0
        # [T, H, W, C] -> [C, T, H, W]
        tensor = torch.from_numpy(arr).permute(3, 0, 1, 2)
        return tensor.contiguous()
```

---

## Normalize

Normalizes a [C, T, H, W] tensor using per-channel mean and std.

**Standard ImageNet values:**
- `mean = [0.485, 0.456, 0.406]`
- `std  = [0.229, 0.224, 0.225]`

**When auto_augment=False (no RandAugment):**
Some implementations scale mean/std to [0, 255] range:
- `mean_255 = [m * 255 for m in mean]  # [123.7, 116.3, 103.5]`
- `std_255  = [s * 255 for s in std]   # [58.4,  57.1,  57.4]`

**Implementation:**
```python
class VideoNormalize:
    def __init__(self, mean, std):
        # Reshape for broadcasting over [C, T, H, W]
        self.mean = torch.tensor(mean).view(3, 1, 1, 1)
        self.std  = torch.tensor(std).view(3, 1, 1, 1)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std
```

---

## Eval Transform (Deterministic)

The evaluation pipeline uses center crop instead of random crop and
skips all stochastic augmentations.

```
Raw Video [T, H, W, C]
    -> CenterCrop (224x224 from 256x256 resize)
    -> ClipToTensor
    -> Normalize
```

**Implementation:**
```python
def get_eval_transform(img_size=224, mean=..., std=...) -> Callable:
    resize_size = int(img_size * 256 / 224)  # e.g., 256 for img_size=224
    return Compose([
        VideoResize(resize_size),        # Shortest edge = resize_size
        VideoCenterCrop(img_size),       # Center crop to img_size
        ClipToTensor(),
        VideoNormalize(mean, std),
    ])
```

**Determinism guarantee:** Given the same input frames, `get_eval_transform()`
must always produce the exact same output tensor. Verified in tests.

---

## Robotics-Specific Configuration

```python
robotics_aug = AugConfig(
    crop_scale=(0.9, 1.0),     # Near-fixed scale; minimal spatial jitter
    crop_ratio=(1.0, 1.0),     # Square crops only
    horizontal_flip=False,      # No flip; direction-sensitive
    auto_augment=False,         # No RandAugment; preserve color fidelity
    motion_shift=False,         # No motion shift; preserve spatial precision
    random_erasing=0.0,         # No erasing; preserve all spatial context
    normalize_mean=(0.485, 0.456, 0.406),
    normalize_std=(0.229, 0.224, 0.225),
)
```

---

## Complete Train Transform Assembly

```python
from torchvision.transforms import Compose

def get_train_transform(config: AugConfig, img_size: int = 224) -> Callable:
    transforms = []

    # 1. Spatial crop (same params all frames)
    transforms.append(
        VideoRandomResizedCrop(img_size, scale=config.crop_scale,
                               ratio=config.crop_ratio)
    )

    # 2. Temporal motion jitter
    if config.motion_shift:
        transforms.append(VideoMotionShift(max_shift_frac=0.1))

    # 3. Horizontal flip
    if config.horizontal_flip:
        transforms.append(VideoRandomHorizontalFlip(p=0.5))

    # 4. Per-frame RandAugment (optional)
    if config.auto_augment:
        transforms.append(
            RandAugmentVideo(n=config.rand_augment_n, m=config.rand_augment_m)
        )

    # 5. Convert to tensor [C, T, H, W]
    transforms.append(ClipToTensor())

    # 6. Random erasing (cube mode, on tensor)
    if config.random_erasing > 0:
        transforms.append(
            VideoRandomErasing(p=config.random_erasing, cube_mode=True)
        )

    # 7. Normalize
    transforms.append(VideoNormalize(config.normalize_mean, config.normalize_std))

    return Compose(transforms)
```
