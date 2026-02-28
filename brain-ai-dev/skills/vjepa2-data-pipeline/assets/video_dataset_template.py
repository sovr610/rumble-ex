"""
video_dataset_template.py
=========================
VideoDataset: decord-based video loading with three clip sampling modes.

Public API
----------
VideoDataset(data_paths, clip_mode, frames_per_clip, target_fps, transform)
    .__getitem__(idx) -> {"video": Tensor[C,T,H,W], "label": int, "path": str}

Clip Modes
----------
"fps"        -- Sample frames at target_fps from a random start offset
"duration"   -- Fixed duration window uniformly sampled to frames_per_clip
"frame_step" -- Fixed stride between consecutive frames

Frame Padding
-------------
Circulant (modular) padding for videos shorter than required.
"""

from __future__ import annotations

import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Optional decord import — falls back to synthetic frames in test/dev mode
# ---------------------------------------------------------------------------
try:
    import decord
    from decord import VideoReader, cpu, gpu
    _DECORD_AVAILABLE = True
except ImportError:
    _DECORD_AVAILABLE = False
    VideoReader = None


# ---------------------------------------------------------------------------
# Frame index computation
# ---------------------------------------------------------------------------

def compute_frame_indices(
    total_frames: int,
    frames_per_clip: int,
    clip_mode: str,
    native_fps: float = 30.0,
    target_fps: float = 10.0,
    clip_duration_sec: float = 3.2,
    frame_step: int = 4,
) -> List[int]:
    """
    Compute clip frame indices with circulant padding for short videos.

    Parameters
    ----------
    total_frames : int
        Number of frames in the video (len(VideoReader)).
    frames_per_clip : int
        Number of frames to return.
    clip_mode : str
        One of "fps", "duration", "frame_step".
    native_fps : float
        Native video frame rate from VideoReader.get_avg_fps().
    target_fps : float
        Target frames per second when clip_mode="fps".
    clip_duration_sec : float
        Duration in seconds when clip_mode="duration".
    frame_step : int
        Stride between frames when clip_mode="frame_step".

    Returns
    -------
    List[int]
        Frame indices of length `frames_per_clip`, all in [0, total_frames-1].
    """
    if total_frames <= 0:
        return [0] * frames_per_clip

    if clip_mode == "fps":
        step = max(1, round(native_fps / max(target_fps, 1.0)))
        native_needed = step * frames_per_clip
        max_start = max(0, total_frames - native_needed)
        start = random.randint(0, max_start)
        indices = [start + i * step for i in range(frames_per_clip)]

    elif clip_mode == "duration":
        clip_len = max(1, int(clip_duration_sec * native_fps))
        max_start = max(0, total_frames - clip_len)
        start = random.randint(0, max_start)
        end = start + clip_len
        indices = np.linspace(start, end - 1, frames_per_clip, dtype=int).tolist()

    elif clip_mode == "frame_step":
        native_needed = frame_step * (frames_per_clip - 1) + 1
        max_start = max(0, total_frames - native_needed)
        start = random.randint(0, max_start)
        indices = [start + i * frame_step for i in range(frames_per_clip)]

    else:
        raise ValueError(
            f"Unknown clip_mode={clip_mode!r}. "
            "Expected one of 'fps', 'duration', 'frame_step'."
        )

    # Circulant padding: wrap indices into [0, total_frames-1]
    indices = [int(idx) % total_frames for idx in indices]
    return indices


# ---------------------------------------------------------------------------
# Index file loading utilities
# ---------------------------------------------------------------------------

def load_index_file(path: str) -> List[Tuple[str, int, int]]:
    """
    Load a video index file.

    Each line: ``video_path num_frames label``
    If num_frames is missing, returns -1 (lazy length query).
    If label is missing, returns -1.

    Parameters
    ----------
    path : str
        Path to the index .txt file or a directory containing video files.

    Returns
    -------
    List of (video_path, num_frames, label) tuples.
    """
    records: List[Tuple[str, int, int]] = []

    if os.path.isfile(path) and path.endswith(".txt"):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                vpath = parts[0]
                nf = int(parts[1]) if len(parts) >= 2 else -1
                label = int(parts[2]) if len(parts) >= 3 else -1
                records.append((vpath, nf, label))

    elif os.path.isdir(path):
        # Scan directory for video files
        exts = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv"}
        for root, _dirs, files in os.walk(path):
            for fname in sorted(files):
                if os.path.splitext(fname)[1].lower() in exts:
                    full = os.path.join(root, fname)
                    records.append((full, -1, -1))

    else:
        # Treat as a single video path
        if os.path.isfile(path):
            records.append((path, -1, -1))

    return records


# ---------------------------------------------------------------------------
# Synthetic frame generation for testing (no video files required)
# ---------------------------------------------------------------------------

def _make_synthetic_frames(
    num_frames: int,
    height: int,
    width: int,
    seed: int = 0,
) -> np.ndarray:
    """
    Generate random uint8 frames [T, H, W, C] for testing without video files.
    """
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (num_frames, height, width, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# VideoDataset
# ---------------------------------------------------------------------------

class VideoDataset(Dataset):
    """
    Dataset for loading video clips with configurable sampling modes.

    Supports real video files via decord (when available) and falls back
    to synthetic frames for testing and development.

    Parameters
    ----------
    data_paths : List[str]
        List of paths to video files, index .txt files, or directories.
    clip_mode : str
        Frame sampling mode: "fps" | "duration" | "frame_step".
    frames_per_clip : int
        Number of frames per clip (T dimension).
    target_fps : int
        Target playback FPS for clip_mode="fps".
    clip_duration_sec : float
        Clip duration in seconds for clip_mode="duration".
    frame_step : int
        Frame stride for clip_mode="frame_step".
    transform : Optional[Callable]
        Transform applied to raw frames (list of PIL Images or numpy array).
        If None, returns raw numpy array.
    img_size : int
        Target spatial size (H and W). Used for synthetic frames only.
    use_gpu_decode : bool
        Use GPU-accelerated decoding via decord.gpu(). Requires CUDA decord.
    synthetic_num_frames : int
        Number of synthetic frames per video (for testing without real videos).
    """

    VALID_CLIP_MODES = {"fps", "duration", "frame_step"}

    def __init__(
        self,
        data_paths: List[str],
        clip_mode: str = "fps",
        frames_per_clip: int = 16,
        target_fps: int = 10,
        clip_duration_sec: float = 3.2,
        frame_step: int = 4,
        transform: Optional[Callable] = None,
        img_size: int = 224,
        use_gpu_decode: bool = False,
        synthetic_num_frames: int = 60,
    ):
        if clip_mode not in self.VALID_CLIP_MODES:
            raise ValueError(
                f"clip_mode={clip_mode!r} is not valid. "
                f"Choose from {sorted(self.VALID_CLIP_MODES)}."
            )

        self.clip_mode = clip_mode
        self.frames_per_clip = frames_per_clip
        self.target_fps = target_fps
        self.clip_duration_sec = clip_duration_sec
        self.frame_step = frame_step
        self.transform = transform
        self.img_size = img_size
        self.use_gpu_decode = use_gpu_decode
        self.synthetic_num_frames = synthetic_num_frames

        # Flatten all data_paths into a list of (video_path, num_frames, label)
        self._records: List[Tuple[str, int, int]] = []
        for p in data_paths:
            self._records.extend(load_index_file(p))

        # If no real files found and decord unavailable, enter synthetic mode
        self._synthetic_mode = (
            not _DECORD_AVAILABLE
            or len(self._records) == 0
        )
        if self._synthetic_mode and len(self._records) == 0:
            # Create a dummy record for synthetic mode
            self._records = [("<synthetic>", synthetic_num_frames, -1)]

        # Determine decord context (per-dataset; workers override per-call)
        if _DECORD_AVAILABLE and not self._synthetic_mode:
            try:
                self._ctx = gpu(0) if use_gpu_decode else cpu(0)
            except Exception:
                self._ctx = cpu(0)
        else:
            self._ctx = None

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load and return a single video clip.

        Returns
        -------
        dict with keys:
            "video" : Tensor[C, T, H, W] float32
            "label" : int (-1 if unavailable)
            "path"  : str (video file path)
        """
        video_path, known_num_frames, label = self._records[idx]

        if self._synthetic_mode or video_path == "<synthetic>":
            frames_np = _make_synthetic_frames(
                self.synthetic_num_frames, self.img_size, self.img_size, seed=idx
            )
            native_fps = 30.0
            total_frames = len(frames_np)
        else:
            frames_np, native_fps, total_frames = self._load_video_frames(
                video_path, known_num_frames
            )

        # Compute frame indices based on sampling mode
        frame_indices = compute_frame_indices(
            total_frames=total_frames,
            frames_per_clip=self.frames_per_clip,
            clip_mode=self.clip_mode,
            native_fps=native_fps,
            target_fps=self.target_fps,
            clip_duration_sec=self.clip_duration_sec,
            frame_step=self.frame_step,
        )

        # Extract clip frames [T, H, W, C]
        clip_np = frames_np[frame_indices]  # shape [T, H, W, C]

        # Convert to list of numpy frames for transform pipeline
        clip_frames = [clip_np[t] for t in range(self.frames_per_clip)]

        if self.transform is not None:
            video_tensor = self.transform(clip_frames)
        else:
            # Default: stack to tensor [C, T, H, W] float32
            video_tensor = self._default_to_tensor(clip_np)

        return {
            "video": video_tensor,
            "label": label,
            "path": video_path,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_video_frames(
        self,
        video_path: str,
        known_num_frames: int,
    ) -> Tuple[np.ndarray, float, int]:
        """
        Load all frames (or enough for one clip) using decord.

        Returns
        -------
        frames_np : np.ndarray [N, H, W, C] uint8
        native_fps : float
        total_frames : int
        """
        # Workers should always use cpu(0) to avoid CUDA re-init issues
        ctx = cpu(0)
        try:
            vr = VideoReader(video_path, ctx=ctx)
            native_fps = float(vr.get_avg_fps()) or 30.0
            total_frames = len(vr)

            if total_frames == 0:
                # Return synthetic fallback for corrupt/empty video
                dummy = _make_synthetic_frames(
                    self.frames_per_clip, self.img_size, self.img_size
                )
                return dummy, native_fps, self.frames_per_clip

            # Load all frames at once for small videos;
            # for large videos, compute indices first then load selectively
            if total_frames <= self.frames_per_clip * 4:
                all_indices = list(range(total_frames))
                frames = vr.get_batch(all_indices).asnumpy()
            else:
                # Precompute indices to avoid loading entire video
                frame_indices = compute_frame_indices(
                    total_frames=total_frames,
                    frames_per_clip=self.frames_per_clip,
                    clip_mode=self.clip_mode,
                    native_fps=native_fps,
                    target_fps=self.target_fps,
                    clip_duration_sec=self.clip_duration_sec,
                    frame_step=self.frame_step,
                )
                frames = vr.get_batch(frame_indices).asnumpy()
                # Return directly (indices already applied)
                return frames, native_fps, total_frames

            return frames, native_fps, total_frames

        except Exception as exc:
            # Graceful degradation: return synthetic frames
            import warnings
            warnings.warn(
                f"Failed to decode {video_path!r}: {exc}. "
                "Using synthetic frames as fallback.",
                RuntimeWarning,
            )
            dummy = _make_synthetic_frames(
                self.frames_per_clip, self.img_size, self.img_size
            )
            return dummy, 30.0, self.frames_per_clip

    @staticmethod
    def _default_to_tensor(clip_np: np.ndarray) -> torch.Tensor:
        """
        Convert [T, H, W, C] uint8 numpy array to [C, T, H, W] float32 tensor.
        Values scaled to [0, 1].
        """
        arr = clip_np.astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(3, 0, 1, 2)
        return tensor.contiguous()

    def __repr__(self) -> str:
        return (
            f"VideoDataset("
            f"n_videos={len(self)}, "
            f"clip_mode={self.clip_mode!r}, "
            f"frames_per_clip={self.frames_per_clip}, "
            f"target_fps={self.target_fps}, "
            f"synthetic={self._synthetic_mode})"
        )


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("VideoDataset Self-Tests")
    print("=" * 60)

    IMG_SIZE = 112
    FPC = 8
    SYNTHETIC_FRAMES = 30
    PASS = "[PASS]"
    FAIL = "[FAIL]"

    def make_dataset(clip_mode: str, **kwargs) -> VideoDataset:
        """Build a synthetic-mode dataset for testing."""
        return VideoDataset(
            data_paths=[],
            clip_mode=clip_mode,
            frames_per_clip=FPC,
            img_size=IMG_SIZE,
            synthetic_num_frames=SYNTHETIC_FRAMES,
            **kwargs,
        )

    errors: List[str] = []

    # -------------------------------------------------------------------
    # Test 1: Output shape is correct [C, T, H, W]
    # -------------------------------------------------------------------
    ds = make_dataset("fps")
    sample = ds[0]
    expected_shape = (3, FPC, IMG_SIZE, IMG_SIZE)
    if sample["video"].shape == expected_shape:
        print(f"{PASS} Test 1: Output shape {sample['video'].shape}")
    else:
        msg = f"Test 1 FAILED: expected {expected_shape}, got {sample['video'].shape}"
        print(f"{FAIL} {msg}")
        errors.append(msg)

    # -------------------------------------------------------------------
    # Test 2: dtype is float32
    # -------------------------------------------------------------------
    if sample["video"].dtype == torch.float32:
        print(f"{PASS} Test 2: dtype is float32")
    else:
        msg = f"Test 2 FAILED: dtype is {sample['video'].dtype}"
        print(f"{FAIL} {msg}")
        errors.append(msg)

    # -------------------------------------------------------------------
    # Test 3: Values in [0, 1] range
    # -------------------------------------------------------------------
    v_min, v_max = sample["video"].min().item(), sample["video"].max().item()
    if 0.0 <= v_min and v_max <= 1.0:
        print(f"{PASS} Test 3: Values in [0,1]: min={v_min:.4f} max={v_max:.4f}")
    else:
        msg = f"Test 3 FAILED: values out of range [0,1]: min={v_min:.4f} max={v_max:.4f}"
        print(f"{FAIL} {msg}")
        errors.append(msg)

    # -------------------------------------------------------------------
    # Test 4: All three clip modes produce FPC frames
    # -------------------------------------------------------------------
    for mode in ["fps", "duration", "frame_step"]:
        ds_m = make_dataset(mode)
        s = ds_m[0]
        if s["video"].shape[1] == FPC:
            print(f"{PASS} Test 4 [{mode}]: T dimension = {FPC}")
        else:
            msg = f"Test 4 [{mode}] FAILED: T = {s['video'].shape[1]}, expected {FPC}"
            print(f"{FAIL} {msg}")
            errors.append(msg)

    # -------------------------------------------------------------------
    # Test 5: Short video (fewer frames than FPC) uses circulant padding
    # -------------------------------------------------------------------
    ds_short = VideoDataset(
        data_paths=[],
        clip_mode="fps",
        frames_per_clip=16,
        img_size=64,
        synthetic_num_frames=3,  # Only 3 frames, need 16
    )
    s_short = ds_short[0]
    if s_short["video"].shape == (3, 16, 64, 64):
        print(f"{PASS} Test 5: Short video circulant padding -> shape {s_short['video'].shape}")
    else:
        msg = f"Test 5 FAILED: shape {s_short['video'].shape}"
        print(f"{FAIL} {msg}")
        errors.append(msg)

    # -------------------------------------------------------------------
    # Test 6: Circulant padding does not produce all-zero tensor
    # -------------------------------------------------------------------
    if s_short["video"].abs().sum() > 0:
        print(f"{PASS} Test 6: Short video output is non-zero (frames loaded)")
    else:
        msg = "Test 6 FAILED: short video tensor is all zeros"
        print(f"{FAIL} {msg}")
        errors.append(msg)

    # -------------------------------------------------------------------
    # Test 7: Return dict has required keys
    # -------------------------------------------------------------------
    required_keys = {"video", "label", "path"}
    missing = required_keys - set(sample.keys())
    if not missing:
        print(f"{PASS} Test 7: Return dict has all required keys")
    else:
        msg = f"Test 7 FAILED: missing keys {missing}"
        print(f"{FAIL} {msg}")
        errors.append(msg)

    # -------------------------------------------------------------------
    # Test 8: compute_frame_indices -- fps mode basic
    # -------------------------------------------------------------------
    indices = compute_frame_indices(
        total_frames=300, frames_per_clip=16,
        clip_mode="fps", native_fps=30.0, target_fps=10.0
    )
    if len(indices) == 16 and all(0 <= i < 300 for i in indices):
        print(f"{PASS} Test 8: compute_frame_indices fps -> 16 valid indices")
    else:
        msg = f"Test 8 FAILED: indices={indices}"
        print(f"{FAIL} {msg}")
        errors.append(msg)

    # -------------------------------------------------------------------
    # Test 9: compute_frame_indices -- circulant on very short video
    # -------------------------------------------------------------------
    indices_short = compute_frame_indices(
        total_frames=2, frames_per_clip=8,
        clip_mode="fps", native_fps=30.0, target_fps=10.0
    )
    if len(indices_short) == 8 and all(0 <= i < 2 for i in indices_short):
        print(f"{PASS} Test 9: Circulant pad short video -> {indices_short}")
    else:
        msg = f"Test 9 FAILED: indices_short={indices_short}"
        print(f"{FAIL} {msg}")
        errors.append(msg)

    # -------------------------------------------------------------------
    # Test 10: Invalid clip_mode raises ValueError
    # -------------------------------------------------------------------
    try:
        VideoDataset(data_paths=[], clip_mode="invalid")
        msg = "Test 10 FAILED: should have raised ValueError"
        print(f"{FAIL} {msg}")
        errors.append(msg)
    except ValueError:
        print(f"{PASS} Test 10: Invalid clip_mode raises ValueError")

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
        print(f"All tests passed.")
