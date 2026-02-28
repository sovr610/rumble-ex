"""
video_transforms_template.py
=============================
Video augmentation pipeline for V-JEPA 2.

All spatial augmentations (crop, flip) apply the SAME parameters to every
frame in the clip (temporal consistency). Per-frame augmentations (RandAugment)
apply independently to each frame.

Public API
----------
VideoTransformPipeline(config: AugConfig)
    .get_train_transform() -> Callable[[List[np.ndarray]], Tensor]
    .get_eval_transform()  -> Callable[[List[np.ndarray]], Tensor]

Individual transforms (also usable standalone):
    VideoRandomResizedCrop(size, scale, ratio)
    VideoMotionShift(max_shift_frac)
    VideoRandomHorizontalFlip(p)
    RandAugmentVideo(n, m)
    VideoRandomErasing(p, scale, ratio, cube_mode)
    ClipToTensor()
    VideoNormalize(mean, std)
    VideoResize(size)
    VideoCenterCrop(size)
"""

from __future__ import annotations

import math
import random
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


# Try PIL; fall back to minimal stubs for environments without Pillow
try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False
    Image = None


# ---------------------------------------------------------------------------
# Frame type alias
# ---------------------------------------------------------------------------
# Frames are passed as List[np.ndarray] of shape [H, W, C] uint8.
# After ClipToTensor they become Tensor[C, T, H, W] float32.
Frames = List[np.ndarray]


# ---------------------------------------------------------------------------
# Helper: numpy frame <-> PIL Image
# ---------------------------------------------------------------------------

def _to_pil(arr: np.ndarray) -> "Image.Image":
    """Convert [H, W, C] uint8 numpy array to PIL Image."""
    if _PIL_AVAILABLE:
        return Image.fromarray(arr.astype(np.uint8))
    return arr  # passthrough if PIL unavailable


def _from_pil(img: Union["Image.Image", np.ndarray]) -> np.ndarray:
    """Convert PIL Image (or passthrough numpy) to [H, W, C] uint8 array."""
    if _PIL_AVAILABLE and isinstance(img, Image.Image):
        return np.array(img)
    return img


# ---------------------------------------------------------------------------
# VideoRandomResizedCrop
# ---------------------------------------------------------------------------

class VideoRandomResizedCrop:
    """
    Randomly crops and resizes all frames to `size` using the SAME crop params.

    Parameters
    ----------
    size : int
        Output spatial size (square).
    scale : tuple[float, float]
        Min/max fraction of area to retain.
    ratio : tuple[float, float]
        Min/max aspect ratio of the crop.
    interpolation : int
        Interpolation mode for resizing (default: nearest-neighbor fallback).
    """

    def __init__(
        self,
        size: int,
        scale: Tuple[float, float] = (0.3, 1.0),
        ratio: Tuple[float, float] = (0.75, 1.33),
        interpolation: int = 2,  # Image.BILINEAR = 2
    ):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    def _get_params(self, h: int, w: int) -> Tuple[int, int, int, int]:
        """
        Compute (top, left, crop_h, crop_w) for a single crop.
        Mirrors torchvision.transforms.RandomResizedCrop.get_params logic.
        """
        area = h * w
        log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))

        for _ in range(10):
            target_area = area * random.uniform(*self.scale)
            aspect = math.exp(random.uniform(*log_ratio))
            crop_w = int(round(math.sqrt(target_area * aspect)))
            crop_h = int(round(math.sqrt(target_area / aspect)))
            if 0 < crop_w <= w and 0 < crop_h <= h:
                top  = random.randint(0, h - crop_h)
                left = random.randint(0, w - crop_w)
                return top, left, crop_h, crop_w

        # Fallback: center crop at largest scale
        in_ratio = w / h
        if in_ratio < self.ratio[0]:
            crop_w = w
            crop_h = int(round(w / self.ratio[0]))
        elif in_ratio > self.ratio[1]:
            crop_h = h
            crop_w = int(round(h * self.ratio[1]))
        else:
            crop_h, crop_w = h, w
        top  = (h - crop_h) // 2
        left = (w - crop_w) // 2
        return top, left, crop_h, crop_w

    def __call__(self, frames: Frames) -> Frames:
        h, w = frames[0].shape[:2]
        top, left, ch, cw = self._get_params(h, w)

        result = []
        for frame in frames:
            cropped = frame[top:top+ch, left:left+cw]
            # Resize to target size
            if _PIL_AVAILABLE:
                img = _to_pil(cropped)
                img = img.resize((self.size, self.size), self.interpolation)
                result.append(_from_pil(img))
            else:
                # Simple nearest-neighbor resize via numpy
                arr = cropped.astype(np.float32)
                t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
                t = F.interpolate(t, (self.size, self.size), mode="nearest")
                result.append(t.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8))
        return result


# ---------------------------------------------------------------------------
# VideoMotionShift
# ---------------------------------------------------------------------------

class VideoMotionShift:
    """
    Applies a temporally varying spatial shift across frames to simulate motion.

    The shift interpolates linearly from (-max_shift, -max_shift) to
    (max_shift, max_shift) across the temporal axis, then applies an affine
    translation to each frame independently.

    Parameters
    ----------
    max_shift_frac : float
        Maximum shift as fraction of spatial size, e.g. 0.1 = 10%.
    """

    def __init__(self, max_shift_frac: float = 0.1):
        self.max_shift_frac = max_shift_frac

    def __call__(self, frames: Frames) -> Frames:
        T = len(frames)
        if T == 0 or self.max_shift_frac <= 0:
            return frames

        h, w = frames[0].shape[:2]
        max_sx = int(self.max_shift_frac * w)
        max_sy = int(self.max_shift_frac * h)

        # Random end-point shift
        end_sx = random.randint(-max_sx, max_sx)
        end_sy = random.randint(-max_sy, max_sy)

        # Linearly interpolate shift across frames
        shifts_x = np.linspace(0, end_sx, T)
        shifts_y = np.linspace(0, end_sy, T)

        result = []
        for t, (frame, sx, sy) in enumerate(zip(frames, shifts_x, shifts_y)):
            sx_i, sy_i = int(round(sx)), int(round(sy))
            shifted = self._shift_frame(frame, sx_i, sy_i)
            result.append(shifted)
        return result

    @staticmethod
    def _shift_frame(frame: np.ndarray, sx: int, sy: int) -> np.ndarray:
        """Translate frame by (sx, sy) pixels with zero-padding at boundaries."""
        h, w, c = frame.shape
        result = np.zeros_like(frame)
        # Compute source and destination slices
        src_x0 = max(0, -sx);  src_x1 = min(w, w - sx)
        dst_x0 = max(0,  sx);  dst_x1 = min(w, w + sx)
        src_y0 = max(0, -sy);  src_y1 = min(h, h - sy)
        dst_y0 = max(0,  sy);  dst_y1 = min(h, h + sy)
        if src_x0 < src_x1 and src_y0 < src_y1:
            result[dst_y0:dst_y1, dst_x0:dst_x1] = frame[src_y0:src_y1, src_x0:src_x1]
        return result


# ---------------------------------------------------------------------------
# VideoRandomHorizontalFlip
# ---------------------------------------------------------------------------

class VideoRandomHorizontalFlip:
    """
    Flips all frames horizontally with probability p.
    Same flip decision for all frames (temporal consistency).

    Parameters
    ----------
    p : float
        Flip probability.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, frames: Frames) -> Frames:
        if random.random() < self.p:
            return [frame[:, ::-1, :].copy() for frame in frames]
        return frames


# ---------------------------------------------------------------------------
# RandAugmentVideo
# ---------------------------------------------------------------------------

class RandAugmentVideo:
    """
    Applies n randomly-selected augmentation operations to each frame
    INDEPENDENTLY (per-frame, not temporally consistent by design).

    Parameters
    ----------
    n : int
        Number of augmentation operations per frame.
    m : int
        Magnitude of each operation (0-30 scale).
    """

    def __init__(self, n: int = 2, m: int = 9):
        self.n = n
        self.m = m

    def __call__(self, frames: Frames) -> Frames:
        return [self._augment_frame(f) for f in frames]

    def _augment_frame(self, frame: np.ndarray) -> np.ndarray:
        if _PIL_AVAILABLE:
            img = _to_pil(frame)
            ops = random.sample(self._get_ops(), self.n)
            for op in ops:
                img = op(img, self.m)
            return _from_pil(img)
        else:
            # Fallback: no augmentation when PIL unavailable
            return frame

    def _get_ops(self) -> List[Callable]:
        """Build list of augmentation operation callables."""
        if not _PIL_AVAILABLE:
            return []
        return [
            self._shear_x,
            self._shear_y,
            self._translate_x,
            self._translate_y,
            self._rotate,
            self._auto_contrast,
            self._equalize,
            self._solarize,
            self._color,
            self._contrast,
            self._brightness,
            self._sharpness,
            self._posterize,
        ]

    # -- Individual operations -- each takes (img, magnitude) -> img

    @staticmethod
    def _magnitude_to_level(m: int, max_val: float, scale: int = 30) -> float:
        return (m / scale) * max_val

    def _shear_x(self, img: "Image.Image", m: int) -> "Image.Image":
        level = self._magnitude_to_level(m, 0.3)
        level = random.choice([-level, level])
        return img.transform(img.size, Image.AFFINE, (1, level, 0, 0, 1, 0))

    def _shear_y(self, img: "Image.Image", m: int) -> "Image.Image":
        level = self._magnitude_to_level(m, 0.3)
        level = random.choice([-level, level])
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, level, 1, 0))

    def _translate_x(self, img: "Image.Image", m: int) -> "Image.Image":
        level = self._magnitude_to_level(m, img.size[0] * 0.33)
        level = random.choice([-level, level])
        return img.transform(img.size, Image.AFFINE, (1, 0, level, 0, 1, 0))

    def _translate_y(self, img: "Image.Image", m: int) -> "Image.Image":
        level = self._magnitude_to_level(m, img.size[1] * 0.33)
        level = random.choice([-level, level])
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, level))

    def _rotate(self, img: "Image.Image", m: int) -> "Image.Image":
        degrees = self._magnitude_to_level(m, 30.0)
        degrees = random.choice([-degrees, degrees])
        return img.rotate(degrees)

    @staticmethod
    def _auto_contrast(img: "Image.Image", m: int) -> "Image.Image":
        return ImageOps.autocontrast(img)

    @staticmethod
    def _equalize(img: "Image.Image", m: int) -> "Image.Image":
        return ImageOps.equalize(img)

    def _solarize(self, img: "Image.Image", m: int) -> "Image.Image":
        threshold = int(self._magnitude_to_level(m, 256))
        return ImageOps.solarize(img, threshold)

    def _color(self, img: "Image.Image", m: int) -> "Image.Image":
        factor = 1.0 + self._magnitude_to_level(m, 0.9) * random.choice([-1, 1])
        factor = max(0.1, factor)
        return ImageEnhance.Color(img).enhance(factor)

    def _contrast(self, img: "Image.Image", m: int) -> "Image.Image":
        factor = 1.0 + self._magnitude_to_level(m, 0.9) * random.choice([-1, 1])
        factor = max(0.1, factor)
        return ImageEnhance.Contrast(img).enhance(factor)

    def _brightness(self, img: "Image.Image", m: int) -> "Image.Image":
        factor = 1.0 + self._magnitude_to_level(m, 0.9) * random.choice([-1, 1])
        factor = max(0.1, factor)
        return ImageEnhance.Brightness(img).enhance(factor)

    def _sharpness(self, img: "Image.Image", m: int) -> "Image.Image":
        factor = 1.0 + self._magnitude_to_level(m, 0.9) * random.choice([-1, 1])
        return ImageEnhance.Sharpness(img).enhance(factor)

    def _posterize(self, img: "Image.Image", m: int) -> "Image.Image":
        bits = max(1, int(8 - self._magnitude_to_level(m, 4)))
        return ImageOps.posterize(img, bits)


# ---------------------------------------------------------------------------
# ClipToTensor
# ---------------------------------------------------------------------------

class ClipToTensor:
    """
    Converts a list of numpy frames [H, W, C] uint8 to Tensor [C, T, H, W] float32.
    Values are scaled from [0, 255] to [0.0, 1.0].
    """

    def __call__(self, frames: Frames) -> torch.Tensor:
        # Stack: [T, H, W, C]
        arr = np.stack([np.asarray(f, dtype=np.uint8) for f in frames], axis=0)
        arr = arr.astype(np.float32) / 255.0
        # [T, H, W, C] -> [C, T, H, W]
        tensor = torch.from_numpy(arr).permute(3, 0, 1, 2)
        return tensor.contiguous()


# ---------------------------------------------------------------------------
# VideoRandomErasing (cube mode)
# ---------------------------------------------------------------------------

class VideoRandomErasing:
    """
    Randomly erases a rectangular region from all frames.
    In cube_mode=True (default), the same region is erased from all frames.

    Operates on a tensor [C, T, H, W] (after ClipToTensor).

    Parameters
    ----------
    p : float
        Probability of erasing.
    scale : tuple[float, float]
        Fraction of image area to erase.
    ratio : tuple[float, float]
        Aspect ratio of erased region.
    value : float
        Fill value (0 = black).
    cube_mode : bool
        Apply same region to all frames.
    """

    def __init__(
        self,
        p: float = 0.25,
        scale: Tuple[float, float] = (0.02, 0.33),
        ratio: Tuple[float, float] = (0.3, 3.3),
        value: float = 0.0,
        cube_mode: bool = True,
    ):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.cube_mode = cube_mode

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        tensor : Tensor[C, T, H, W]
        """
        if random.random() >= self.p:
            return tensor

        C, T, H, W = tensor.shape
        area = H * W

        for _ in range(10):
            erase_area = area * random.uniform(*self.scale)
            aspect = random.uniform(*self.ratio)
            erase_h = int(round(math.sqrt(erase_area / aspect)))
            erase_w = int(round(math.sqrt(erase_area * aspect)))
            if erase_h < H and erase_w < W:
                top  = random.randint(0, H - erase_h)
                left = random.randint(0, W - erase_w)
                tensor = tensor.clone()
                if self.cube_mode:
                    # Same region for ALL frames
                    tensor[:, :, top:top+erase_h, left:left+erase_w] = self.value
                else:
                    # Independent erasing per frame
                    for t in range(T):
                        top_t  = random.randint(0, H - erase_h)
                        left_t = random.randint(0, W - erase_w)
                        tensor[:, t, top_t:top_t+erase_h, left_t:left_t+erase_w] = self.value
                return tensor

        return tensor


# ---------------------------------------------------------------------------
# VideoNormalize
# ---------------------------------------------------------------------------

class VideoNormalize:
    """
    Normalizes a [C, T, H, W] tensor using per-channel mean and std.

    Parameters
    ----------
    mean : sequence of 3 floats
        Per-channel mean (ImageNet: 0.485, 0.456, 0.406).
    std : sequence of 3 floats
        Per-channel std (ImageNet: 0.229, 0.224, 0.225).
    """

    def __init__(
        self,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
    ):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1, 1)
        self.std  = torch.tensor(std,  dtype=torch.float32).view(3, 1, 1, 1)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std


# ---------------------------------------------------------------------------
# VideoResize + VideoCenterCrop (for eval pipeline)
# ---------------------------------------------------------------------------

class VideoResize:
    """
    Resize all frames so the shorter edge equals `size`.
    Maintains aspect ratio.
    """

    def __init__(self, size: int):
        self.size = size

    def __call__(self, frames: Frames) -> Frames:
        result = []
        for frame in frames:
            h, w = frame.shape[:2]
            if h < w:
                new_h = self.size
                new_w = int(round(w * self.size / h))
            else:
                new_w = self.size
                new_h = int(round(h * self.size / w))
            if _PIL_AVAILABLE:
                img = _to_pil(frame).resize((new_w, new_h), 2)  # BILINEAR
                result.append(_from_pil(img))
            else:
                t = torch.from_numpy(frame.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
                t = F.interpolate(t, (new_h, new_w), mode="nearest")
                result.append(t.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8))
        return result


class VideoCenterCrop:
    """
    Crops the center `size x size` region from all frames.
    """

    def __init__(self, size: int):
        self.size = size

    def __call__(self, frames: Frames) -> Frames:
        result = []
        for frame in frames:
            h, w = frame.shape[:2]
            top  = (h - self.size) // 2
            left = (w - self.size) // 2
            result.append(frame[top:top+self.size, left:left+self.size].copy())
        return result


# ---------------------------------------------------------------------------
# Compose (lightweight, avoids torchvision dependency at class level)
# ---------------------------------------------------------------------------

class Compose:
    """Sequentially applies a list of transforms."""

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self) -> str:
        parts = [f"  {t}" for t in self.transforms]
        return "Compose([\n" + ",\n".join(parts) + "\n])"


# ---------------------------------------------------------------------------
# VideoTransformPipeline
# ---------------------------------------------------------------------------

class VideoTransformPipeline:
    """
    Factory for building train and eval transform pipelines.

    Parameters
    ----------
    config : AugConfig-like object or dict
        Augmentation configuration with attributes matching AugConfig.
    img_size : int
        Target spatial size.
    """

    def __init__(self, config, img_size: int = 224):
        self.config = config
        self.img_size = img_size

    def get_train_transform(self) -> Callable:
        """
        Build the training augmentation pipeline.

        Pipeline:
            RandomResizedCrop -> [MotionShift] -> [HorizontalFlip]
            -> [RandAugment] -> ClipToTensor -> [RandomErasing] -> Normalize
        """
        cfg = self.config
        transforms: List[Callable] = []

        # 1. Spatial crop (same params all frames)
        transforms.append(
            VideoRandomResizedCrop(
                size=self.img_size,
                scale=_get(cfg, "crop_scale", (0.3, 1.0)),
                ratio=_get(cfg, "crop_ratio", (0.75, 1.33)),
            )
        )

        # 2. Temporal motion jitter
        if _get(cfg, "motion_shift", True):
            transforms.append(VideoMotionShift(max_shift_frac=0.1))

        # 3. Horizontal flip
        if _get(cfg, "horizontal_flip", True):
            transforms.append(VideoRandomHorizontalFlip(p=0.5))

        # 4. Per-frame RandAugment
        if _get(cfg, "auto_augment", False):
            transforms.append(
                RandAugmentVideo(
                    n=_get(cfg, "rand_augment_n", 2),
                    m=_get(cfg, "rand_augment_m", 9),
                )
            )

        # 5. Convert to tensor [C, T, H, W] float32
        transforms.append(ClipToTensor())

        # 6. Random erasing (cube mode, on tensor)
        erase_p = _get(cfg, "random_erasing", 0.0)
        if erase_p > 0:
            transforms.append(VideoRandomErasing(p=erase_p, cube_mode=True))

        # 7. Normalize
        transforms.append(
            VideoNormalize(
                mean=_get(cfg, "normalize_mean", (0.485, 0.456, 0.406)),
                std=_get(cfg, "normalize_std",  (0.229, 0.224, 0.225)),
            )
        )

        return Compose(transforms)

    def get_eval_transform(self) -> Callable:
        """
        Build the deterministic evaluation pipeline.

        Pipeline:
            VideoResize(resize_size) -> VideoCenterCrop(img_size)
            -> ClipToTensor -> Normalize
        """
        cfg = self.config
        resize_size = int(self.img_size * 256 / 224)  # e.g., 256 for 224

        transforms = [
            VideoResize(resize_size),
            VideoCenterCrop(self.img_size),
            ClipToTensor(),
            VideoNormalize(
                mean=_get(cfg, "normalize_mean", (0.485, 0.456, 0.406)),
                std=_get(cfg, "normalize_std",  (0.229, 0.224, 0.225)),
            ),
        ]
        return Compose(transforms)


def _get(obj, attr: str, default):
    """Get attribute from dataclass or dict with fallback default."""
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("VideoTransformPipeline Self-Tests")
    print("=" * 60)

    PASS = "[PASS]"
    FAIL = "[FAIL]"
    errors = []

    # Create synthetic frames [T, H, W, C] uint8
    def make_frames(T=8, H=256, W=256, C=3, seed=42) -> Frames:
        rng = np.random.RandomState(seed)
        arr = rng.randint(0, 256, (T, H, W, C), dtype=np.uint8)
        return [arr[t] for t in range(T)]

    T, H, W = 8, 256, 256
    IMG_SIZE = 112
    frames = make_frames(T=T, H=H, W=W)

    # Default config dict
    cfg = {
        "crop_scale": (0.3, 1.0),
        "crop_ratio": (0.75, 1.33),
        "horizontal_flip": True,
        "auto_augment": False,
        "rand_augment_n": 2,
        "rand_augment_m": 9,
        "motion_shift": True,
        "random_erasing": 0.0,
        "normalize_mean": (0.485, 0.456, 0.406),
        "normalize_std":  (0.229, 0.224, 0.225),
    }

    pipeline = VideoTransformPipeline(cfg, img_size=IMG_SIZE)

    # -------------------------------------------------------------------
    # Test 1: Train transform output shape
    # -------------------------------------------------------------------
    train_tfm = pipeline.get_train_transform()
    out = train_tfm(frames)
    expected = (3, T, IMG_SIZE, IMG_SIZE)
    if out.shape == expected:
        print(f"{PASS} Test 1: Train transform shape {out.shape}")
    else:
        msg = f"Test 1 FAILED: expected {expected}, got {out.shape}"
        print(f"{FAIL} {msg}")
        errors.append(msg)

    # -------------------------------------------------------------------
    # Test 2: Train transform dtype
    # -------------------------------------------------------------------
    if out.dtype == torch.float32:
        print(f"{PASS} Test 2: dtype is float32")
    else:
        msg = f"Test 2 FAILED: dtype={out.dtype}"
        print(f"{FAIL} {msg}")
        errors.append(msg)

    # -------------------------------------------------------------------
    # Test 3: Eval transform output shape
    # -------------------------------------------------------------------
    eval_tfm = pipeline.get_eval_transform()
    eval_out = eval_tfm(frames)
    if eval_out.shape == expected:
        print(f"{PASS} Test 3: Eval transform shape {eval_out.shape}")
    else:
        msg = f"Test 3 FAILED: expected {expected}, got {eval_out.shape}"
        print(f"{FAIL} {msg}")
        errors.append(msg)

    # -------------------------------------------------------------------
    # Test 4: Eval transform is deterministic
    # -------------------------------------------------------------------
    eval_out2 = eval_tfm(frames)
    if torch.allclose(eval_out, eval_out2):
        print(f"{PASS} Test 4: Eval transform is deterministic")
    else:
        msg = "Test 4 FAILED: eval transform not deterministic"
        print(f"{FAIL} {msg}")
        errors.append(msg)

    # -------------------------------------------------------------------
    # Test 5: RandomResizedCrop temporal consistency
    # -------------------------------------------------------------------
    identical_frames = [frames[0].copy() for _ in range(T)]
    rrc = VideoRandomResizedCrop(IMG_SIZE)
    rrc_out = ClipToTensor()(rrc(identical_frames))
    # All temporal slices should be identical
    t_consistent = all(
        torch.allclose(rrc_out[:, 0, :, :], rrc_out[:, t, :, :])
        for t in range(1, T)
    )
    if t_consistent:
        print(f"{PASS} Test 5: RandomResizedCrop applies same crop to all frames")
    else:
        msg = "Test 5 FAILED: RandomResizedCrop not temporally consistent"
        print(f"{FAIL} {msg}")
        errors.append(msg)

    # -------------------------------------------------------------------
    # Test 6: HorizontalFlip temporal consistency
    # -------------------------------------------------------------------
    flip = VideoRandomHorizontalFlip(p=1.0)  # Always flip
    flip_out = ClipToTensor()(flip(identical_frames))
    t_consistent_flip = all(
        torch.allclose(flip_out[:, 0, :, :], flip_out[:, t, :, :])
        for t in range(1, T)
    )
    if t_consistent_flip:
        print(f"{PASS} Test 6: HorizontalFlip same decision for all frames")
    else:
        msg = "Test 6 FAILED: HorizontalFlip not temporally consistent"
        print(f"{FAIL} {msg}")
        errors.append(msg)

    # -------------------------------------------------------------------
    # Test 7: Normalized values in reasonable range
    # -------------------------------------------------------------------
    v_min = out.min().item()
    v_max = out.max().item()
    if v_min > -10.0 and v_max < 10.0:
        print(f"{PASS} Test 7: Normalized values in [-10,10]: [{v_min:.3f}, {v_max:.3f}]")
    else:
        msg = f"Test 7 FAILED: values out of range: [{v_min:.3f}, {v_max:.3f}]"
        print(f"{FAIL} {msg}")
        errors.append(msg)

    # -------------------------------------------------------------------
    # Test 8: RandomErasing cube mode -- same region all frames
    # -------------------------------------------------------------------
    dummy_tensor = torch.ones(3, T, IMG_SIZE, IMG_SIZE)
    eraser = VideoRandomErasing(p=1.0, scale=(0.1, 0.3), cube_mode=True)
    erased = eraser(dummy_tensor)
    # Find zeroed region in frame 0
    zero_mask = (erased[:, 0, :, :] == 0)
    if zero_mask.any():
        all_frames_match = all(
            (erased[:, t, :, :][zero_mask] == 0).all()
            for t in range(1, T)
        )
        if all_frames_match:
            print(f"{PASS} Test 8: RandomErasing cube_mode erases same region all frames")
        else:
            msg = "Test 8 FAILED: cube mode erasing inconsistent across frames"
            print(f"{FAIL} {msg}")
            errors.append(msg)
    else:
        msg = "Test 8 FAILED: RandomErasing p=1.0 did not erase anything"
        print(f"{FAIL} {msg}")
        errors.append(msg)

    # -------------------------------------------------------------------
    # Test 9: RandAugment pipeline (with auto_augment=True)
    # -------------------------------------------------------------------
    if _PIL_AVAILABLE:
        cfg_ra = dict(cfg, auto_augment=True)
        pipeline_ra = VideoTransformPipeline(cfg_ra, img_size=IMG_SIZE)
        train_ra = pipeline_ra.get_train_transform()
        out_ra = train_ra(frames)
        if out_ra.shape == expected:
            print(f"{PASS} Test 9: RandAugment pipeline shape {out_ra.shape}")
        else:
            msg = f"Test 9 FAILED: RandAugment pipeline shape {out_ra.shape}"
            print(f"{FAIL} {msg}")
            errors.append(msg)
    else:
        print("[SKIP] Test 9: PIL not available, skipping RandAugment test")

    # -------------------------------------------------------------------
    # Test 10: Robotics config (no flip, fixed scale, no augment)
    # -------------------------------------------------------------------
    robotics_cfg = {
        "crop_scale": (0.9, 1.0),
        "crop_ratio": (1.0, 1.0),
        "horizontal_flip": False,
        "auto_augment": False,
        "motion_shift": False,
        "random_erasing": 0.0,
        "normalize_mean": (0.485, 0.456, 0.406),
        "normalize_std":  (0.229, 0.224, 0.225),
    }
    rob_pipeline = VideoTransformPipeline(robotics_cfg, img_size=IMG_SIZE)
    rob_out = rob_pipeline.get_train_transform()(frames)
    if rob_out.shape == expected:
        print(f"{PASS} Test 10: Robotics transform shape {rob_out.shape}")
    else:
        msg = f"Test 10 FAILED: robotics shape {rob_out.shape}"
        print(f"{FAIL} {msg}")
        errors.append(msg)

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print()
    if errors:
        print(f"FAILED: {len(errors)} test(s) failed:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("All tests passed.")
