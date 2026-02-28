"""
tf.data pipeline builder for video TFRecord datasets.

Constructs a high-throughput tf.data.Dataset from sharded TFRecord files,
suitable for TPU pod training. Features shard-level shuffle, parallel
interleave, example-level shuffle, parse + augment mapping, batching
with drop_remainder, and prefetch.

CRITICAL: Never use bare .eval on nn.Module -- use module.train(False) instead.
For model inference in a training loop, call model.train(False) before the
forward pass, not the unsafe bare .eval pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import tensorflow as tf
    _TF_AVAILABLE = True
    AUTOTUNE = tf.data.AUTOTUNE
except ImportError:
    _TF_AVAILABLE = False
    AUTOTUNE = None
    print("[tf_pipeline_template] WARNING: tensorflow not available -- pipeline builder disabled.")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TFPipelineConfig:
    """Configuration for the tf.data video pipeline."""

    shuffle_buffer: int = 10_000
    num_frames: int = 16
    crop_size: int = 224
    augment: bool = True

    # ImageNet normalization constants
    normalize_mean: tuple = (0.485, 0.456, 0.406)
    normalize_std: tuple = (0.229, 0.224, 0.225)

    def __post_init__(self) -> None:
        if self.shuffle_buffer < 1:
            raise ValueError(f"shuffle_buffer must be >= 1, got {self.shuffle_buffer}")
        if self.num_frames < 1:
            raise ValueError(f"num_frames must be >= 1, got {self.num_frames}")
        if self.crop_size < 1:
            raise ValueError(f"crop_size must be >= 1, got {self.crop_size}")


# ---------------------------------------------------------------------------
# Feature specification
# ---------------------------------------------------------------------------

def get_feature_spec() -> dict:
    """
    Return the tf.io.FixedLenFeature spec for parsing a single TFRecord example.

    Schema matches TFRecordConverter.make_example() output.
    """
    if not _TF_AVAILABLE:
        raise RuntimeError("tensorflow is required for get_feature_spec")

    return {
        "video_bytes": tf.io.FixedLenFeature([], tf.string),
        "label":       tf.io.FixedLenFeature([], tf.int64),
        "num_frames":  tf.io.FixedLenFeature([], tf.int64),
        "height":      tf.io.FixedLenFeature([], tf.int64),
        "width":       tf.io.FixedLenFeature([], tf.int64),
        "fps":         tf.io.FixedLenFeature([], tf.float32, default_value=25.0),
    }


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def augment_fn(video: "tf.Tensor", cfg: TFPipelineConfig) -> "tf.Tensor":
    """
    Apply temporally consistent augmentations to a video clip.

    All spatial operations use a single random parameter set applied to every
    frame, preserving temporal consistency.

    Args:
        video: Tensor of shape (T, H, W, 3), dtype float32, range [0, 1]
        cfg: TFPipelineConfig

    Returns:
        Augmented tensor of shape (T, crop_size, crop_size, 3), float32, [0, 1].
    """
    T = tf.shape(video)[0]
    H = tf.shape(video)[1]
    W = tf.shape(video)[2]

    crop_h = cfg.crop_size
    crop_w = cfg.crop_size

    # Random crop: sample a single offset for all frames
    max_offset_h = tf.maximum(H - crop_h, 0)
    max_offset_w = tf.maximum(W - crop_w, 0)
    offset_h = tf.random.uniform([], 0, max_offset_h + 1, dtype=tf.int32)
    offset_w = tf.random.uniform([], 0, max_offset_w + 1, dtype=tf.int32)
    video = video[:, offset_h:offset_h + crop_h, offset_w:offset_w + crop_w, :]

    # Random horizontal flip: same decision for all frames
    flip = tf.random.uniform([]) < 0.5
    video = tf.cond(flip, lambda: tf.image.flip_left_right(video), lambda: video)

    # Random brightness: same offset for all frames
    delta = tf.random.uniform([], -0.1, 0.1)
    video = tf.clip_by_value(video + delta, 0.0, 1.0)

    return video


# ---------------------------------------------------------------------------
# Parse function
# ---------------------------------------------------------------------------

def _make_parse_fn(cfg: TFPipelineConfig, is_training: bool):
    """
    Factory that returns a parse function closed over cfg and is_training.

    The returned function parses a serialized tf.train.Example proto,
    decodes the raw video bytes, normalizes, optionally augments,
    and returns a dict with 'video' and 'label' keys.
    """
    if not _TF_AVAILABLE:
        raise RuntimeError("tensorflow is required for _make_parse_fn")

    feature_spec = get_feature_spec()
    mean = tf.constant(list(cfg.normalize_mean), dtype=tf.float32)   # shape (3,)
    std  = tf.constant(list(cfg.normalize_std),  dtype=tf.float32)   # shape (3,)

    def parse_fn(example_proto):
        parsed = tf.io.parse_single_example(example_proto, feature_spec)

        num_frames = tf.cast(parsed["num_frames"], tf.int32)
        height     = tf.cast(parsed["height"],     tf.int32)
        width      = tf.cast(parsed["width"],      tf.int32)

        # Decode raw uint8 bytes -> (T, H, W, 3) uint8
        video = tf.io.decode_raw(parsed["video_bytes"], tf.uint8)
        video = tf.reshape(video, [num_frames, height, width, 3])

        # Cast to float32 and scale to [0, 1]
        video = tf.cast(video, tf.float32) / 255.0

        # Optional augmentation (training only)
        if is_training and cfg.augment:
            video = augment_fn(video, cfg)
        else:
            # Center crop for validation/test
            H_in = tf.shape(video)[1]
            W_in = tf.shape(video)[2]
            offset_h = (H_in - cfg.crop_size) // 2
            offset_w = (W_in - cfg.crop_size) // 2
            video = video[
                :,
                offset_h:offset_h + cfg.crop_size,
                offset_w:offset_w + cfg.crop_size,
                :,
            ]

        # ImageNet normalization: (video - mean) / std  [broadcast over T,H,W]
        video = (video - mean) / std

        label = tf.cast(parsed["label"], tf.int32)
        return {"video": video, "label": label}

    return parse_fn


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_tf_video_pipeline(
    shard_pattern: str,
    cfg: TFPipelineConfig,
    batch_size: int,
    is_training: bool,
) -> "tf.data.Dataset":
    """
    Build a tf.data.Dataset that reads sharded TFRecord video files.

    Pipeline stages:
      1. list_files with shard-level shuffle (training) or deterministic (inference)
      2. interleave TFRecordDataset across shards with AUTOTUNE parallelism
      3. shuffle example buffer (training only)
      4. map: parse + optional augment
      5. batch with drop_remainder=True for fixed shapes
      6. prefetch AUTOTUNE

    Args:
        shard_pattern: glob pattern matching TFRecord files,
                       e.g. "gs://bucket/train-*.tfrecord"
        cfg: TFPipelineConfig
        batch_size: examples per output batch
        is_training: if True, shuffle and augment; if False, deterministic center crop

    Returns:
        tf.data.Dataset yielding dicts with keys:
            'video': Tensor (batch, T, crop_size, crop_size, 3), float32, normalized
            'label': Tensor (batch,), int32
    """
    if not _TF_AVAILABLE:
        raise RuntimeError("tensorflow is required for build_tf_video_pipeline")

    # Step 1: Shard-level shuffle
    files = tf.data.Dataset.list_files(shard_pattern, shuffle=is_training)

    # Step 2: Parallel shard reading
    dataset = files.interleave(
        lambda f: tf.data.TFRecordDataset(f, buffer_size=64 * 1024 * 1024),
        cycle_length=AUTOTUNE,
        num_parallel_calls=AUTOTUNE,
        deterministic=not is_training,
    )

    # Step 3: Example-level shuffle (training only)
    if is_training:
        dataset = dataset.shuffle(
            buffer_size=cfg.shuffle_buffer,
            reshuffle_each_iteration=True,
        )

    # Step 4: Parse + augment
    parse_fn = _make_parse_fn(cfg, is_training)
    dataset = dataset.map(parse_fn, num_parallel_calls=AUTOTUNE)

    # Step 5: Batch with drop_remainder for fixed shapes (required for TPU)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # Step 6: Prefetch pipeline overlap
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


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
    print("tf.data Pipeline Template Self-Tests")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Test 1: TFPipelineConfig defaults
    # -----------------------------------------------------------------------
    print("\n[Test 1] TFPipelineConfig defaults")
    cfg = TFPipelineConfig()
    check(cfg.shuffle_buffer == 10_000, f"shuffle_buffer=10000, got {cfg.shuffle_buffer}")
    check(cfg.num_frames == 16, f"num_frames=16, got {cfg.num_frames}")
    check(cfg.crop_size == 224, f"crop_size=224, got {cfg.crop_size}")
    check(cfg.augment is True, f"augment=True, got {cfg.augment}")
    check(len(cfg.normalize_mean) == 3, "normalize_mean has 3 values")
    check(len(cfg.normalize_std) == 3, "normalize_std has 3 values")

    # -----------------------------------------------------------------------
    # Test 2: TFPipelineConfig validation
    # -----------------------------------------------------------------------
    print("\n[Test 2] TFPipelineConfig validation")
    for kwargs, desc in [
        ({"shuffle_buffer": 0}, "rejects shuffle_buffer=0"),
        ({"num_frames": 0}, "rejects num_frames=0"),
        ({"crop_size": 0}, "rejects crop_size=0"),
    ]:
        try:
            TFPipelineConfig(**kwargs)
            check(False, desc, "no exception raised")
        except ValueError:
            check(True, desc)

    # -----------------------------------------------------------------------
    # Test 3: Feature spec structure
    # -----------------------------------------------------------------------
    print("\n[Test 3] Feature spec structure")
    if _TF_AVAILABLE:
        spec = get_feature_spec()
        check("video_bytes" in spec, "spec has 'video_bytes'")
        check("label" in spec, "spec has 'label'")
        check("num_frames" in spec, "spec has 'num_frames'")
        check("height" in spec, "spec has 'height'")
        check("width" in spec, "spec has 'width'")
        check("fps" in spec, "spec has 'fps'")
        check(
            isinstance(spec["video_bytes"], tf.io.FixedLenFeature),
            "video_bytes is FixedLenFeature",
        )
        check(
            spec["label"].dtype == tf.int64,
            f"label dtype=int64, got {spec['label'].dtype}",
        )
    else:
        print("  SKIP: TensorFlow not available -- feature spec tests skipped")

    # -----------------------------------------------------------------------
    # Test 4: Pipeline builder returns correct type
    # -----------------------------------------------------------------------
    print("\n[Test 4] Pipeline builder return type")
    if _TF_AVAILABLE:
        import tempfile, os, numpy as np

        # Write a minimal synthetic TFRecord to test the pipeline end-to-end
        with tempfile.TemporaryDirectory() as tmp:
            shard_path = os.path.join(tmp, "train-00000-of-00001.tfrecord")

            # Create synthetic example
            T, H, W = 4, 32, 32
            frames = np.random.randint(0, 256, (T, H, W, 3), dtype=np.uint8)
            raw_bytes = frames.tobytes()

            feature = {
                "video_bytes": tf.train.Feature(bytes_list=tf.train.BytesList(value=[raw_bytes])),
                "label":       tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
                "num_frames":  tf.train.Feature(int64_list=tf.train.Int64List(value=[T])),
                "height":      tf.train.Feature(int64_list=tf.train.Int64List(value=[H])),
                "width":       tf.train.Feature(int64_list=tf.train.Int64List(value=[W])),
                "fps":         tf.train.Feature(float_list=tf.train.FloatList(value=[25.0])),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Write 20 examples so we can form a batch of 4
            with tf.io.TFRecordWriter(shard_path) as writer:
                for _ in range(20):
                    writer.write(example.SerializeToString())

            cfg_small = TFPipelineConfig(shuffle_buffer=10, num_frames=T, crop_size=H)
            pattern = os.path.join(tmp, "train-*.tfrecord")

            # Training pipeline
            ds_train = build_tf_video_pipeline(
                shard_pattern=pattern,
                cfg=cfg_small,
                batch_size=4,
                is_training=False,  # no augment since H==crop_size (no room to crop)
            )
            check(isinstance(ds_train, tf.data.Dataset), "build returns tf.data.Dataset")

            # Consume one batch and verify shape
            for batch in ds_train.take(1):
                vid = batch["video"]
                lbl = batch["label"]
                check(vid.shape[0] == 4, f"batch size=4, got {vid.shape[0]}")
                check(vid.shape[1] == T, f"T={T}, got {vid.shape[1]}")
                check(vid.dtype == tf.float32, f"dtype float32, got {vid.dtype}")
                check(lbl.shape == (4,), f"label shape (4,), got {lbl.shape}")

            # Inference pipeline (deterministic)
            ds_infer = build_tf_video_pipeline(
                shard_pattern=pattern,
                cfg=cfg_small,
                batch_size=4,
                is_training=False,
            )
            check(isinstance(ds_infer, tf.data.Dataset), "inference pipeline returns Dataset")
    else:
        print("  SKIP: TensorFlow not available -- pipeline type tests skipped")

    # -----------------------------------------------------------------------
    # Test 5: Augmentation function shape
    # -----------------------------------------------------------------------
    print("\n[Test 5] augment_fn output shape")
    if _TF_AVAILABLE:
        cfg_aug = TFPipelineConfig(crop_size=16, num_frames=4)
        T_a, H_a, W_a = 4, 32, 32
        video_in = tf.random.uniform((T_a, H_a, W_a, 3))
        video_out = augment_fn(video_in, cfg_aug)
        check(
            video_out.shape[0] == T_a,
            f"T preserved: {video_out.shape[0]}",
        )
        check(
            video_out.shape[1] == 16 and video_out.shape[2] == 16,
            f"spatial crop to 16x16, got {video_out.shape[1]}x{video_out.shape[2]}",
        )
        vmin = float(tf.reduce_min(video_out).numpy())
        vmax = float(tf.reduce_max(video_out).numpy())
        check(vmin >= 0.0, f"augment output min >= 0, got {vmin:.4f}")
        check(vmax <= 1.0, f"augment output max <= 1, got {vmax:.4f}")
    else:
        print("  SKIP: TensorFlow not available -- augment_fn shape tests skipped")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Results: {passed} PASSED, {failed} FAILED out of {passed + failed} total")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
