"""
TFRecord conversion template for video datasets.

Converts video files (via decord) to sharded TFRecord files suitable for
tf.data pipelines on TPU or GPU training at scale.

Encoding options:
  - "none": raw uint8 bytes (larger files, fastest decode at training time)
  - "jpeg": per-frame JPEG compression (smaller files, adds decode cost)

Shard naming: {split}-{shard_id:05d}-of-{num_shards:05d}.tfrecord

CRITICAL: Never use bare .eval on nn.Module -- use module.train(False) instead.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

try:
    import decord
    from decord import VideoReader, cpu
    # Use 'torch' bridge for conversion path; get_batch output will be a Tensor.
    # We call .numpy() on the result where a numpy array is needed.
    decord.bridge.set_bridge("torch")
    _DECORD_AVAILABLE = True
except ImportError:
    _DECORD_AVAILABLE = False
    print("[tfrecord_converter_template] WARNING: decord not available.")

try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False
    print("[tfrecord_converter_template] WARNING: tensorflow not available -- TFRecord ops will be skipped.")

try:
    from PIL import Image as _PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TFRecordConfig:
    """Configuration for TFRecord conversion."""

    num_shards: int = 256
    num_frames: int = 16
    stride: int = 4
    crop_size: int = 224
    compression: str = "none"    # "none" | "jpeg"
    jpeg_quality: int = 95

    def __post_init__(self) -> None:
        if self.num_shards < 1:
            raise ValueError(f"num_shards must be >= 1, got {self.num_shards}")
        if self.num_frames < 1:
            raise ValueError(f"num_frames must be >= 1, got {self.num_frames}")
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}")
        if self.crop_size < 1:
            raise ValueError(f"crop_size must be >= 1, got {self.crop_size}")
        if self.compression not in ("none", "jpeg"):
            raise ValueError(
                f"compression must be 'none' or 'jpeg', got '{self.compression}'"
            )
        if not (1 <= self.jpeg_quality <= 100):
            raise ValueError(f"jpeg_quality must be 1-100, got {self.jpeg_quality}")


# ---------------------------------------------------------------------------
# Feature builder helpers
# ---------------------------------------------------------------------------

def _bytes_feature(value: bytes) -> "tf.train.Feature":
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value: int) -> "tf.train.Feature":
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value: float) -> "tf.train.Feature":
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


# ---------------------------------------------------------------------------
# Encoding utilities
# ---------------------------------------------------------------------------

def encode_raw(frames_np: np.ndarray) -> bytes:
    """
    Encode frames as a flat byte array (lossless).

    Args:
        frames_np: shape (T, H, W, C), dtype uint8

    Returns:
        Raw bytes of the array contents.
    """
    return frames_np.tobytes()


def encode_jpeg_per_frame(frames_np: np.ndarray, quality: int = 95) -> bytes:
    """
    Encode each frame as JPEG and concatenate with 4-byte length prefixes.

    Format: [4-byte LE length][JPEG bytes] repeated T times.

    Args:
        frames_np: shape (T, H, W, C), dtype uint8
        quality: JPEG quality (1-100)

    Returns:
        Concatenated bytes of all JPEG-encoded frames with length prefixes.
    """
    if not _PIL_AVAILABLE:
        raise RuntimeError("Pillow is required for JPEG encoding (pip install Pillow)")

    buffer = io.BytesIO()
    for frame in frames_np:
        img = _PILImage.fromarray(frame)
        frame_buf = io.BytesIO()
        img.save(frame_buf, format="JPEG", quality=quality)
        frame_bytes = frame_buf.getvalue()
        # 4-byte little-endian length prefix
        buffer.write(len(frame_bytes).to_bytes(4, byteorder="little"))
        buffer.write(frame_bytes)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Example builder
# ---------------------------------------------------------------------------

def make_example(
    video_bytes: bytes,
    label: int,
    num_frames: int,
    height: int,
    width: int,
    fps: float = 25.0,
) -> "tf.train.Example":
    """
    Build a tf.train.Example for a video clip.

    Features:
        video_bytes (bytes):  encoded clip data (raw or JPEG)
        label       (int64):  class label
        num_frames  (int64):  number of frames
        height      (int64):  frame height in pixels
        width       (int64):  frame width in pixels
        fps         (float):  frames per second
    """
    feature = {
        "video_bytes": _bytes_feature(video_bytes),
        "label":       _int64_feature(label),
        "num_frames":  _int64_feature(num_frames),
        "height":      _int64_feature(height),
        "width":       _int64_feature(width),
        "fps":         _float_feature(fps),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


# ---------------------------------------------------------------------------
# Main converter class
# ---------------------------------------------------------------------------

class TFRecordConverter:
    """
    Converts a video dataset to sharded TFRecord files.

    Usage:
        cfg = TFRecordConfig(num_shards=128, compression="jpeg")
        converter = TFRecordConverter(cfg)
        converter.convert_split(manifest, output_dir="/data/tfrecords", split="train")
    """

    def __init__(self, cfg: TFRecordConfig) -> None:
        self.cfg = cfg

    def convert_split(
        self,
        manifest: list,
        output_dir: str,
        split: str = "train",
        num_shards: Optional[int] = None,
    ) -> None:
        """
        Convert a list of VideoMeta records to sharded TFRecord files.

        Args:
            manifest: list with .path, .label, .num_frames, .fps attributes
            output_dir: directory where shard files will be written
            split: dataset split name used in shard file names
            num_shards: override self.cfg.num_shards if provided
        """
        if not _TF_AVAILABLE:
            raise RuntimeError("tensorflow is required for TFRecord conversion")
        if not _DECORD_AVAILABLE:
            raise RuntimeError("decord is required for TFRecord conversion")

        n_shards = num_shards or self.cfg.num_shards
        os.makedirs(output_dir, exist_ok=True)

        shard_paths = [
            os.path.join(output_dir, self._shard_name(split, i, n_shards))
            for i in range(n_shards)
        ]
        writers = [tf.io.TFRecordWriter(p) for p in shard_paths]

        converted = 0
        skipped = 0

        for idx, meta in enumerate(manifest):
            shard_id = idx % n_shards
            try:
                frames = self._decode_video(meta.path)
                T, H, W, C = frames.shape

                video_bytes = self._encode_video(frames)
                example = make_example(
                    video_bytes,
                    label=meta.label,
                    num_frames=T,
                    height=H,
                    width=W,
                    fps=getattr(meta, "fps", 25.0),
                )
                writers[shard_id].write(example.SerializeToString())
                converted += 1

                if (idx + 1) % 1000 == 0:
                    print(f"[TFRecordConverter] Converted {idx + 1}/{len(manifest)} clips...")

            except Exception as exc:
                print(f"[TFRecordConverter] WARNING: skipping {meta.path}: {exc}")
                skipped += 1

        for w in writers:
            w.close()

        print(
            f"[TFRecordConverter] Done. {converted} clips written to {n_shards} shards "
            f"in {output_dir}. {skipped} skipped."
        )

    def _shard_name(self, split: str, shard_id: int, num_shards: int) -> str:
        """Generate shard filename following the standard naming convention."""
        return f"{split}-{shard_id:05d}-of-{num_shards:05d}.tfrecord"

    def _decode_video(self, path: str) -> np.ndarray:
        """
        Decode a video clip using decord.

        Returns:
            numpy array of shape (T, H, W, C), dtype uint8.
        """
        vr = VideoReader(path, ctx=cpu(0), num_threads=1)
        total = len(vr)
        cfg = self.cfg
        span = cfg.num_frames * cfg.stride
        # Use center-start for deterministic TFRecord conversion
        start = max(0, (total - span) // 2)
        indices = [min(start + i * cfg.stride, total - 1) for i in range(cfg.num_frames)]
        frames = vr.get_batch(indices)  # returns Tensor with 'torch' bridge
        del vr
        # Convert to numpy array for encoding (works for both Tensor and ndarray)
        if hasattr(frames, "numpy"):
            frames = frames.numpy()
        return np.array(frames, dtype=np.uint8)

    def _encode_video(self, frames_np: np.ndarray) -> bytes:
        """Encode frames according to self.cfg.compression setting."""
        if self.cfg.compression == "jpeg":
            return encode_jpeg_per_frame(frames_np, quality=self.cfg.jpeg_quality)
        return encode_raw(frames_np)


# ---------------------------------------------------------------------------
# Shard distribution utility
# ---------------------------------------------------------------------------

def compute_shard_distribution(num_examples: int, num_shards: int) -> List[int]:
    """
    Compute how many examples go into each shard for balanced distribution.

    The standard distribution assigns example i to shard (i % num_shards).

    Args:
        num_examples: total number of examples in the dataset
        num_shards: number of output shards

    Returns:
        List of length num_shards, where entry j is the count of examples in shard j.
    """
    base = num_examples // num_shards
    remainder = num_examples % num_shards
    counts = [base + (1 if i < remainder else 0) for i in range(num_shards)]
    return counts


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
    print("TFRecordConverter Self-Tests")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Test 1: TFRecordConfig defaults
    # -----------------------------------------------------------------------
    print("\n[Test 1] TFRecordConfig defaults")
    cfg = TFRecordConfig()
    check(cfg.num_shards == 256, f"num_shards=256, got {cfg.num_shards}")
    check(cfg.num_frames == 16, f"num_frames=16, got {cfg.num_frames}")
    check(cfg.stride == 4, f"stride=4, got {cfg.stride}")
    check(cfg.crop_size == 224, f"crop_size=224, got {cfg.crop_size}")
    check(cfg.compression == "none", f"compression='none', got '{cfg.compression}'")
    check(cfg.jpeg_quality == 95, f"jpeg_quality=95, got {cfg.jpeg_quality}")

    # -----------------------------------------------------------------------
    # Test 2: TFRecordConfig validation
    # -----------------------------------------------------------------------
    print("\n[Test 2] TFRecordConfig validation")
    for kwargs, desc in [
        ({"num_shards": 0}, "rejects num_shards=0"),
        ({"num_frames": 0}, "rejects num_frames=0"),
        ({"stride": 0}, "rejects stride=0"),
        ({"crop_size": 0}, "rejects crop_size=0"),
        ({"compression": "lz4"}, "rejects unknown compression"),
        ({"jpeg_quality": 0}, "rejects jpeg_quality=0"),
        ({"jpeg_quality": 101}, "rejects jpeg_quality=101"),
    ]:
        try:
            TFRecordConfig(**kwargs)
            check(False, desc, "no exception raised")
        except ValueError:
            check(True, desc)

    # -----------------------------------------------------------------------
    # Test 3: Shard naming convention
    # -----------------------------------------------------------------------
    print("\n[Test 3] Shard naming convention")
    conv = TFRecordConverter(TFRecordConfig())
    name_0 = conv._shard_name("train", 0, 256)
    name_1 = conv._shard_name("train", 1, 256)
    name_last = conv._shard_name("val", 99, 100)

    check(name_0 == "train-00000-of-00256.tfrecord", f"shard 0: '{name_0}'")
    check(name_1 == "train-00001-of-00256.tfrecord", f"shard 1: '{name_1}'")
    check(name_last == "val-00099-of-00100.tfrecord", f"last val shard: '{name_last}'")

    names = [conv._shard_name("train", i, 10) for i in range(10)]
    check(names == sorted(names), "shard names sort lexicographically in correct order")

    # -----------------------------------------------------------------------
    # Test 4: Shard distribution logic
    # -----------------------------------------------------------------------
    print("\n[Test 4] Shard distribution logic")
    dist = compute_shard_distribution(100, 3)
    check(sum(dist) == 100, f"total examples preserved: {sum(dist)}")
    check(len(dist) == 3, f"3 shards produced")
    check(max(dist) - min(dist) <= 1, f"balanced distribution (max-min <= 1)")

    dist2 = compute_shard_distribution(200, 4)
    check(all(d == 50 for d in dist2), f"even split: all 50")

    dist3 = compute_shard_distribution(42, 1)
    check(dist3 == [42], f"single shard: [42], got {dist3}")

    dist4 = compute_shard_distribution(3, 10)
    check(sum(dist4) == 3, f"sparse total preserved")
    check(sum(1 for d in dist4 if d > 0) == 3, "sparse: exactly 3 non-empty shards")

    # -----------------------------------------------------------------------
    # Test 5: Raw encoding is lossless
    # -----------------------------------------------------------------------
    print("\n[Test 5] Raw encoding (lossless)")
    T, H, W, C = 4, 32, 32, 3
    frames = np.random.randint(0, 256, (T, H, W, C), dtype=np.uint8)
    raw_bytes = encode_raw(frames)

    check(isinstance(raw_bytes, bytes), "encode_raw returns bytes")
    expected_size = T * H * W * C
    check(len(raw_bytes) == expected_size, f"raw size={expected_size}, got {len(raw_bytes)}")

    decoded = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(T, H, W, C)
    check(np.array_equal(decoded, frames), "raw round-trip is lossless")

    # -----------------------------------------------------------------------
    # Test 6: JPEG encoding produces smaller output than raw
    # -----------------------------------------------------------------------
    print("\n[Test 6] JPEG encoding vs raw size")
    if _PIL_AVAILABLE:
        T_j, H_j, W_j, C_j = 8, 64, 64, 3
        frames_j = np.random.randint(0, 256, (T_j, H_j, W_j, C_j), dtype=np.uint8)
        raw_size = len(encode_raw(frames_j))
        jpeg_bytes = encode_jpeg_per_frame(frames_j, quality=85)
        jpeg_size = len(jpeg_bytes)
        check(jpeg_size < raw_size, f"JPEG ({jpeg_size}B) < raw ({raw_size}B)")
        check(isinstance(jpeg_bytes, bytes), "encode_jpeg_per_frame returns bytes")
    else:
        print("  SKIP: Pillow not available -- JPEG size test skipped")

    # -----------------------------------------------------------------------
    # Test 7: tf.train.Example feature structure
    # -----------------------------------------------------------------------
    print("\n[Test 7] tf.train.Example feature structure")
    if _TF_AVAILABLE:
        sample_bytes = b"fake_video_data"
        example = make_example(
            video_bytes=sample_bytes,
            label=42,
            num_frames=16,
            height=224,
            width=224,
            fps=30.0,
        )
        features = example.features.feature
        check("video_bytes" in features, "example has 'video_bytes' feature")
        check("label" in features, "example has 'label' feature")
        check("num_frames" in features, "example has 'num_frames' feature")
        check("height" in features, "example has 'height' feature")
        check("width" in features, "example has 'width' feature")
        check("fps" in features, "example has 'fps' feature")
        check(
            features["label"].int64_list.value[0] == 42,
            f"label=42, got {features['label'].int64_list.value[0]}",
        )
        check(
            features["num_frames"].int64_list.value[0] == 16,
            f"num_frames=16",
        )
        serialized = example.SerializeToString()
        check(len(serialized) > 0, f"serializes to {len(serialized)} bytes")

        parsed = tf.train.Example()
        parsed.ParseFromString(serialized)
        check(
            parsed.features.feature["label"].int64_list.value[0] == 42,
            "label survives serialization round-trip",
        )
    else:
        print("  SKIP: TensorFlow not available -- example structure tests skipped")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Results: {passed} PASSED, {failed} FAILED out of {passed + failed} total")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
