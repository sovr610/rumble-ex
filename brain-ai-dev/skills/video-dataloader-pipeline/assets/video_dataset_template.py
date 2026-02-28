"""
VideoDataset template for video deep learning pipelines.

Uses decord.VideoReader for efficient random-access frame decoding.
Frame sampling: random-start + fixed stride pattern.
Multi-worker safe: VideoReader opened per __getitem__, never in __init__.

CRITICAL: set_bridge called once at module level.
CRITICAL: Never use bare .eval() on nn.Module — use module.train(False) instead.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

try:
    import decord
    from decord import VideoReader, cpu

    decord.bridge.set_bridge("torch")
    _DECORD_AVAILABLE = True
except ImportError:
    _DECORD_AVAILABLE = False
    print("[video_dataset_template] WARNING: decord not available — dataset will not function.")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class VideoMeta:
    """Lightweight metadata record for a single video clip."""

    path: str
    num_frames: int
    fps: float
    label: int

    def __post_init__(self) -> None:
        if self.num_frames < 1:
            raise ValueError(f"num_frames must be >= 1, got {self.num_frames}")
        if self.fps <= 0.0:
            raise ValueError(f"fps must be > 0, got {self.fps}")


@dataclass
class VideoDatasetConfig:
    """Configuration for VideoDataset frame sampling and preprocessing."""

    num_frames: int = 16
    stride: int = 4
    crop_size: int = 224
    crop_scale: Tuple[float, float] = (0.5, 1.0)
    crop_ratio: Tuple[float, float] = (0.75, 1.333)
    hflip_prob: float = 0.5
    color_jitter: Tuple[float, ...] = (0.4, 0.4, 0.2, 0.1)
    normalize_mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, ...] = (0.229, 0.224, 0.225)

    def __post_init__(self) -> None:
        if self.num_frames < 1:
            raise ValueError(f"num_frames must be >= 1, got {self.num_frames}")
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}")
        if self.crop_size < 1:
            raise ValueError(f"crop_size must be >= 1, got {self.crop_size}")
        if self.crop_scale[0] >= self.crop_scale[1]:
            raise ValueError(f"crop_scale[0] must be < crop_scale[1], got {self.crop_scale}")
        if self.crop_ratio[0] >= self.crop_ratio[1]:
            raise ValueError(f"crop_ratio[0] must be < crop_ratio[1], got {self.crop_ratio}")


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def scan_manifest(
    data_root: str,
    split: str,
    cache_path: Optional[str] = None,
) -> List[VideoMeta]:
    """
    Walk data_root/split directory tree, open each video briefly to collect
    metadata (num_frames, fps), and cache results to a JSON file.

    Directory layout assumed:
        data_root/
            split/
                class_name_0/
                    video_001.mp4
                class_name_1/
                    ...

    Args:
        data_root: root directory of the dataset
        split: subdirectory name, e.g. 'train', 'val', 'test'
        cache_path: optional path for the JSON manifest cache file.
                    Defaults to data_root/{split}_manifest.json.

    Returns:
        List of VideoMeta, one per discovered video file.
    """
    if not _DECORD_AVAILABLE:
        raise RuntimeError("decord is required for scan_manifest")

    manifest_path = cache_path or os.path.join(data_root, f"{split}_manifest.json")

    # Return cached manifest if it exists
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            data = json.load(f)
        return [VideoMeta(**item) for item in data]

    split_dir = os.path.join(data_root, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    class_names = sorted(
        name for name in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, name))
    )
    label_map = {name: idx for idx, name in enumerate(class_names)}
    video_extensions = {".mp4", ".avi", ".mkv", ".webm", ".mov"}
    manifest: List[VideoMeta] = []

    for class_name in class_names:
        class_dir = os.path.join(split_dir, class_name)
        label_id = label_map[class_name]
        for fname in sorted(os.listdir(class_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in video_extensions:
                continue
            fpath = os.path.join(class_dir, fname)
            try:
                vr = VideoReader(fpath, ctx=cpu(0), num_threads=1)
                meta = VideoMeta(
                    path=fpath,
                    num_frames=len(vr),
                    fps=float(vr.get_avg_fps()),
                    label=label_id,
                )
                manifest.append(meta)
                del vr
            except Exception as exc:
                print(f"[scan_manifest] WARNING: skipping {fpath}: {exc}")

    os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump([asdict(m) for m in manifest], f, indent=2)

    print(f"[scan_manifest] Scanned {len(manifest)} videos -> {manifest_path}")
    return manifest


# ---------------------------------------------------------------------------
# Frame index computation
# ---------------------------------------------------------------------------


def compute_frame_indices(
    total_frames: int,
    num_frames: int,
    stride: int,
) -> List[int]:
    """
    Compute frame indices using random-start + fixed-stride temporal sampling.

    Algorithm:
        span      = num_frames * stride
        max_start = max(0, total_frames - span)
        start     = randint(0, max_start)  [inclusive on both ends]
        indices   = [start + i * stride for i in range(num_frames)]
        clamp each index to [0, total_frames - 1]

    Args:
        total_frames: total number of frames in the video
        num_frames: number of frames to sample per clip
        stride: temporal gap between consecutive sampled frames

    Returns:
        List of frame indices, length == num_frames, each in [0, total_frames-1].
    """
    span = num_frames * stride
    max_start = max(0, total_frames - span)
    start = random.randint(0, max_start)
    indices = [start + i * stride for i in range(num_frames)]
    # Clamp — handles short videos that can't fill the full span
    clamped = [min(idx, total_frames - 1) for idx in indices]
    return clamped


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class VideoDataset(Dataset):
    """
    PyTorch Dataset for video clips using decord for frame decoding.

    Multi-worker safe: VideoReader is created inside __getitem__, not __init__.
    This prevents file handle leaks when used with num_workers > 0 in DataLoader.

    Returns dicts with keys:
        'video': Tensor (T, C, H, W), float32, range [0, 1] before transform
        'label': int label
        'index': original dataset index (useful for debugging)
    """

    def __init__(
        self,
        manifest: List[VideoMeta],
        cfg: VideoDatasetConfig,
        transform: Optional[Callable] = None,
    ) -> None:
        """
        Args:
            manifest: list of VideoMeta records
            cfg: VideoDatasetConfig controlling frame sampling and spatial config
            transform: optional callable applied to the video tensor before returning.
                       Receives a (T, C, H, W) float32 Tensor or tv_tensors.Video.
        """
        if not _DECORD_AVAILABLE:
            raise RuntimeError("decord is required for VideoDataset")
        self.manifest = manifest
        self.cfg = cfg
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        """
        Decode a video clip and optionally apply augmentation.

        Opens a new VideoReader per call — intentional for multi-worker safety.
        If decoding fails (corrupt file), retries with the next index up to 10 times.

        Returns:
            dict with 'video' (T,C,H,W float32), 'label' (int), 'index' (int)
        """
        max_retries = 10
        for attempt in range(max_retries):
            current_idx = (idx + attempt) % len(self.manifest)
            meta = self.manifest[current_idx]

            try:
                # Open fresh VideoReader — never stored on self to avoid handle leaks
                vr = VideoReader(meta.path, ctx=cpu(0), num_threads=1)
                total_frames = len(vr)

                indices = compute_frame_indices(
                    total_frames,
                    self.cfg.num_frames,
                    self.cfg.stride,
                )

                # get_batch returns (T, H, W, C) uint8 with torch bridge
                frames = vr.get_batch(indices)
                del vr  # release file handle immediately

                # (T, H, W, C) uint8 -> (T, C, H, W) float32 [0, 1]
                frames = frames.permute(0, 3, 1, 2).float() / 255.0

                if self.transform is not None:
                    frames = self.transform(frames)

                return {
                    "video": frames,
                    "label": meta.label,
                    "index": current_idx,
                }

            except Exception as exc:
                if attempt == 0:
                    print(
                        f"[VideoDataset] WARNING: decode failed for {meta.path}: {exc}. Retrying."
                    )

        # All retries exhausted — return zeros to avoid crashing the DataLoader
        print(
            f"[VideoDataset] ERROR: all {max_retries} retries failed starting at idx {idx}. "
            "Returning zero tensor."
        )
        dummy_shape = (self.cfg.num_frames, 3, self.cfg.crop_size, self.cfg.crop_size)
        return {
            "video": torch.zeros(dummy_shape, dtype=torch.float32),
            "label": 0,
            "index": idx,
        }


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    passed = 0
    failed = 0

    def check(condition: bool, name: str, detail: str = "") -> None:
        global passed, failed
        if condition:
            print(f"  PASS: {name}")
            passed += 1
        else:
            msg = f"  FAIL: {name}"
            if detail:
                msg += f" -- {detail}"
            print(msg)
            failed += 1

    print("=" * 60)
    print("VideoDataset Self-Tests")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Test 1: VideoMeta creation and field access
    # -----------------------------------------------------------------------
    print("\n[Test 1] VideoMeta creation and field access")
    meta = VideoMeta(path="/tmp/test.mp4", num_frames=100, fps=25.0, label=3)
    check(meta.path == "/tmp/test.mp4", "VideoMeta.path correct")
    check(meta.num_frames == 100, "VideoMeta.num_frames correct")
    check(meta.fps == 25.0, "VideoMeta.fps correct")
    check(meta.label == 3, "VideoMeta.label correct")

    # -----------------------------------------------------------------------
    # Test 2: VideoMeta validation rejects invalid values
    # -----------------------------------------------------------------------
    print("\n[Test 2] VideoMeta validation")
    try:
        VideoMeta(path="/tmp/x.mp4", num_frames=0, fps=25.0, label=0)
        check(False, "VideoMeta rejects num_frames=0", "no exception raised")
    except ValueError:
        check(True, "VideoMeta rejects num_frames=0")

    try:
        VideoMeta(path="/tmp/x.mp4", num_frames=10, fps=-1.0, label=0)
        check(False, "VideoMeta rejects fps=-1", "no exception raised")
    except ValueError:
        check(True, "VideoMeta rejects fps=-1")

    # -----------------------------------------------------------------------
    # Test 3: VideoDatasetConfig defaults
    # -----------------------------------------------------------------------
    print("\n[Test 3] VideoDatasetConfig defaults")
    cfg = VideoDatasetConfig()
    check(cfg.num_frames == 16, f"num_frames=16, got {cfg.num_frames}")
    check(cfg.stride == 4, f"stride=4, got {cfg.stride}")
    check(cfg.crop_size == 224, f"crop_size=224, got {cfg.crop_size}")
    check(len(cfg.normalize_mean) == 3, "normalize_mean has 3 values")
    check(len(cfg.normalize_std) == 3, "normalize_std has 3 values")
    check(cfg.crop_scale == (0.5, 1.0), f"crop_scale=(0.5,1.0), got {cfg.crop_scale}")
    check(cfg.hflip_prob == 0.5, f"hflip_prob=0.5, got {cfg.hflip_prob}")

    # -----------------------------------------------------------------------
    # Test 4: VideoDatasetConfig validation
    # -----------------------------------------------------------------------
    print("\n[Test 4] VideoDatasetConfig validation")
    try:
        VideoDatasetConfig(num_frames=0)
        check(False, "config rejects num_frames=0", "no exception raised")
    except ValueError:
        check(True, "config rejects num_frames=0")

    try:
        VideoDatasetConfig(stride=0)
        check(False, "config rejects stride=0", "no exception raised")
    except ValueError:
        check(True, "config rejects stride=0")

    try:
        VideoDatasetConfig(crop_size=0)
        check(False, "config rejects crop_size=0", "no exception raised")
    except ValueError:
        check(True, "config rejects crop_size=0")

    try:
        VideoDatasetConfig(crop_scale=(0.8, 0.5))
        check(False, "config rejects inverted crop_scale", "no exception raised")
    except ValueError:
        check(True, "config rejects inverted crop_scale")

    try:
        VideoDatasetConfig(crop_ratio=(2.0, 1.0))
        check(False, "config rejects inverted crop_ratio", "no exception raised")
    except ValueError:
        check(True, "config rejects inverted crop_ratio")

    # -----------------------------------------------------------------------
    # Test 5: Frame index calculation for various video lengths
    # -----------------------------------------------------------------------
    print("\n[Test 5] Frame index calculation")

    # Standard case: enough frames to fill span
    indices = compute_frame_indices(total_frames=64, num_frames=16, stride=4)
    check(len(indices) == 16, f"standard: len=16, got {len(indices)}")
    check(all(0 <= i <= 63 for i in indices), "standard: all in [0, 63]")
    check(indices == sorted(indices), "standard: non-decreasing")

    # Short video: fewer frames than span
    indices = compute_frame_indices(total_frames=5, num_frames=16, stride=4)
    check(len(indices) == 16, f"short video: len=16, got {len(indices)}")
    check(all(0 <= i <= 4 for i in indices), "short video: clamped to [0, 4]")

    # Single frame video
    indices = compute_frame_indices(total_frames=1, num_frames=8, stride=2)
    check(len(indices) == 8, f"1-frame video: 8 indices, got {len(indices)}")
    check(all(i == 0 for i in indices), "1-frame video: all indices are 0")

    # Stride=1
    indices = compute_frame_indices(total_frames=100, num_frames=10, stride=1)
    check(len(indices) == 10, f"stride=1: 10 indices, got {len(indices)}")
    check(all(0 <= i <= 99 for i in indices), "stride=1: all in [0, 99]")

    # Very large video — start should vary
    unique_starts = set()
    for _ in range(100):
        idxs = compute_frame_indices(total_frames=1000, num_frames=8, stride=4)
        unique_starts.add(idxs[0])
    check(len(unique_starts) > 5, f"randomness: >5 unique starts in 100 calls, got {len(unique_starts)}")

    # -----------------------------------------------------------------------
    # Test 6: Indices are clamped to valid range in all cases
    # -----------------------------------------------------------------------
    print("\n[Test 6] Index clamping")
    for total in [1, 3, 7, 16, 100]:
        idxs = compute_frame_indices(total_frames=total, num_frames=16, stride=4)
        all_valid = all(0 <= i < total for i in idxs)
        check(all_valid, f"clamped to [0, {total-1}] for total={total}")

    # -----------------------------------------------------------------------
    # Test 7: Mock dataset output structure and shapes
    # -----------------------------------------------------------------------
    print("\n[Test 7] Mock dataset output structure")

    class _MockVideoDataset(VideoDataset):
        """Bypasses decord for testing output structure."""

        def __getitem__(self, idx: int) -> Dict[str, object]:
            T = self.cfg.num_frames
            H = W = self.cfg.crop_size
            return {
                "video": torch.rand(T, 3, H, W, dtype=torch.float32),
                "label": self.manifest[idx].label,
                "index": idx,
            }

    mock_manifest = [
        VideoMeta(path="/fake/a.mp4", num_frames=64, fps=25.0, label=0),
        VideoMeta(path="/fake/b.mp4", num_frames=32, fps=30.0, label=1),
        VideoMeta(path="/fake/c.mp4", num_frames=100, fps=24.0, label=2),
    ]
    mock_cfg = VideoDatasetConfig(num_frames=8, stride=2, crop_size=112)
    ds = _MockVideoDataset(mock_manifest, mock_cfg) if _DECORD_AVAILABLE else _MockVideoDataset.__new__(_MockVideoDataset)
    if not _DECORD_AVAILABLE:
        # Manually init without calling super().__init__ that checks decord
        ds.manifest = mock_manifest
        ds.cfg = mock_cfg
        ds.transform = None

    check(len(ds) == 3, f"dataset len=3, got {len(ds)}")

    sample = ds[0]
    check("video" in sample, "sample has 'video' key")
    check("label" in sample, "sample has 'label' key")
    check("index" in sample, "sample has 'index' key")
    check(
        sample["video"].shape == (8, 3, 112, 112),
        f"video shape (8,3,112,112), got {tuple(sample['video'].shape)}",
    )
    check(
        sample["video"].dtype == torch.float32,
        f"dtype float32, got {sample['video'].dtype}",
    )
    check(
        0.0 <= sample["video"].min().item() and sample["video"].max().item() <= 1.0,
        "values in [0, 1]",
    )
    check(sample["label"] == 0, f"label=0, got {sample['label']}")
    check(sample["index"] == 0, f"index=0, got {sample['index']}")

    # -----------------------------------------------------------------------
    # Test 8: DataLoader compatibility
    # -----------------------------------------------------------------------
    print("\n[Test 8] DataLoader compatibility")
    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)
    batches = list(loader)
    check(len(batches) >= 1, f"DataLoader produces batches, got {len(batches)}")
    b0 = batches[0]
    check(b0["video"].ndim == 5, f"batch 'video' has 5 dims, got {b0['video'].ndim}")
    check(
        b0["video"].shape[1] == 8,
        f"batch T-dim=8, got {b0['video'].shape[1]}",
    )
    check(
        b0["video"].shape[2] == 3,
        f"batch C-dim=3, got {b0['video'].shape[2]}",
    )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Results: {passed} PASSED, {failed} FAILED out of {passed + failed} total")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
