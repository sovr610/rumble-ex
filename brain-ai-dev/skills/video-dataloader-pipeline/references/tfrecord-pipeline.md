# TFRecord Conversion and tf.data Pipeline for TPU

## Overview

TFRecord is TensorFlow's binary serialization format, optimized for sequential reads on distributed storage (GCS). For TPU pod training, video data must be pre-converted to sharded TFRecord files so the tf.data pipeline can stream data efficiently from Google Cloud Storage without seeking through large video files at training time. This document covers the schema design, sharding strategy, conversion procedure, and the complete tf.data pipeline construction with TPU-specific considerations.

---

## TFRecord Schema Design

A `tf.train.Example` is a protocol buffer mapping feature names to feature values. For video, the schema must encode the clip data and its metadata.

### Feature Schema

```python
import tensorflow as tf

# Schema for one TFRecord example (one video clip)
FEATURE_SCHEMA = {
    # Encoded video data: raw uint8 bytes or JPEG-per-frame bytes
    'video_bytes': tf.io.FixedLenFeature([], tf.string),

    # Class label integer
    'label': tf.io.FixedLenFeature([], tf.int64),

    # Metadata for reshaping on decode
    'num_frames': tf.io.FixedLenFeature([], tf.int64),
    'height':     tf.io.FixedLenFeature([], tf.int64),
    'width':      tf.io.FixedLenFeature([], tf.int64),

    # Optional metadata
    'fps':        tf.io.FixedLenFeature([], tf.float32, default_value=25.0),
}
```

### Helper: Building a Feature Dict

```python
def _bytes_feature(value: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value: int) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value: float) -> tf.train.Feature:
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
```

---

## Encoding Strategies

### Strategy 1: Raw uint8 Bytes

Encode the entire clip as a flat byte array. Fastest to decode (no JPEG decompression), largest files.

```python
def encode_raw(frames_np: np.ndarray) -> bytes:
    """
    frames_np: shape (T, H, W, C), dtype uint8
    Returns flat bytes of the array.
    """
    return frames_np.tobytes()

def decode_raw(video_bytes: tf.Tensor, num_frames, height, width) -> tf.Tensor:
    """Decode raw bytes back to (T, H, W, C) uint8."""
    video = tf.io.decode_raw(video_bytes, tf.uint8)
    video = tf.reshape(video, [num_frames, height, width, 3])
    return video
```

File size estimate: 16 frames * 224 * 224 * 3 bytes = ~2.4 MB per clip (uncompressed).

### Strategy 2: JPEG-Per-Frame

Encode each frame as JPEG, concatenate bytes with length prefixes. Smaller files, adds CPU decode cost at training time.

```python
import io
from PIL import Image

def encode_jpeg_per_frame(frames_np: np.ndarray, quality: int = 95) -> bytes:
    """
    frames_np: shape (T, H, W, C), dtype uint8
    Returns concatenated JPEG bytes with 4-byte length prefix per frame.
    """
    buffer = io.BytesIO()
    for frame in frames_np:
        img = Image.fromarray(frame)
        frame_buf = io.BytesIO()
        img.save(frame_buf, format='JPEG', quality=quality)
        frame_bytes = frame_buf.getvalue()
        # Write 4-byte length prefix then frame bytes
        buffer.write(len(frame_bytes).to_bytes(4, 'little'))
        buffer.write(frame_bytes)
    return buffer.getvalue()
```

JPEG at quality=95 typically achieves 10-15x compression over raw bytes. For `num_frames=16, crop_size=224`, expect ~200-400 KB per clip vs 2.4 MB raw.

### Which Strategy to Choose?

| Criterion | Raw uint8 | JPEG-per-frame |
|-----------|-----------|---------------|
| Decode speed | Faster | Slower (JPEG decompression) |
| File size | ~2.4 MB/clip | ~150-400 KB/clip |
| Storage cost | High | Low |
| TPU throughput | Better when GCS I/O is fast | Better when GCS I/O is slow |
| Recommended for | GPU/CPU training | Large-scale TPU pod with bandwidth constraints |

---

## Sharding Rules

Proper sharding is critical for TPU training throughput.

### Rule 1: 10x Files Per TPU Host

TPU pods have multiple hosts (each running a VM). Each host should read from at least 10 shards to ensure good random-access efficiency. Fewer shards per host creates a bottleneck where hosts wait for each other.

```
num_shards >= 10 * num_tpu_hosts
```

For an 8-host pod: `num_shards >= 80`

### Rule 2: Each Shard 100MB or Larger

GCS reads are most efficient for large sequential reads. Tiny shards increase per-file metadata overhead.

```
shard_size >= 100 MB  (raw) or >= 20 MB (JPEG)
```

### Rule 3: Balanced Shards

Ensure examples are distributed evenly across shards so no shard is significantly larger or smaller than others.

### Shard Calculation Example

- 1 million clips, JPEG encoding, ~300 KB per clip
- Total data: 300 GB
- Target shard size: 100 MB → 3000 shards minimum
- For 8 TPU hosts: 8 * 10 = 80 minimum → 3000 satisfies this
- Final choice: 3000 shards

### Shard Naming Convention

```python
shard_name = f"{split}-{shard_id:05d}-of-{num_shards:05d}.tfrecord"
# Examples:
# train-00000-of-03000.tfrecord
# train-00001-of-03000.tfrecord
# val-00000-of-00100.tfrecord
```

---

## Conversion Procedure

### Complete Converter Implementation

```python
import os
import numpy as np
import tensorflow as tf
from decord import VideoReader, cpu
import decord

decord.bridge.set_bridge('numpy')  # numpy is better for TFRecord conversion

def convert_split(manifest, output_dir, split, num_shards, cfg):
    """
    Convert a list of VideoMeta to sharded TFRecord files.

    Args:
        manifest: List[VideoMeta] — video paths, labels, frame counts
        output_dir: directory to write shards
        split: 'train', 'val', or 'test'
        num_shards: number of output shards
        cfg: TFRecordConfig
    """
    os.makedirs(output_dir, exist_ok=True)

    # Open all shard writers
    writers = []
    for shard_id in range(num_shards):
        path = os.path.join(output_dir, f"{split}-{shard_id:05d}-of-{num_shards:05d}.tfrecord")
        writers.append(tf.io.TFRecordWriter(path))

    for idx, meta in enumerate(manifest):
        shard_id = idx % num_shards
        try:
            # Decode frames
            vr = VideoReader(meta.path, ctx=cpu(0), num_threads=1)
            total = len(vr)
            span = cfg.num_frames * cfg.stride
            max_start = max(0, total - span)
            start = 0  # deterministic for TFRecord (use center crop)
            indices = [min(start + i * cfg.stride, total - 1) for i in range(cfg.num_frames)]
            frames = vr.get_batch(indices)  # numpy array (T, H, W, C)
            del vr

            # Optionally resize to crop_size
            if frames.shape[1] != cfg.crop_size or frames.shape[2] != cfg.crop_size:
                # Resize to square crop_size for storage
                import cv2
                resized = np.stack([
                    cv2.resize(frames[i], (cfg.crop_size, cfg.crop_size))
                    for i in range(cfg.num_frames)
                ])
                frames = resized

            T, H, W, C = frames.shape

            # Encode
            if cfg.compression == 'jpeg':
                video_bytes = _encode_jpeg_frames(frames, cfg.jpeg_quality)
            else:
                video_bytes = frames.tobytes()

            # Build Example
            example = _make_example(video_bytes, meta.label, T, H, W)
            writers[shard_id].write(example.SerializeToString())

        except Exception as e:
            print(f"Warning: skipping {meta.path}: {e}")

    for w in writers:
        w.close()

    print(f"Wrote {len(manifest)} examples to {num_shards} shards in {output_dir}")
```

---

## tf.data Pipeline Construction

### Step-by-Step Pipeline

```python
def build_tf_video_pipeline(
    shard_pattern: str,
    cfg,
    batch_size: int,
    is_training: bool,
) -> tf.data.Dataset:
    """
    Build a tf.data pipeline for video TFRecords.

    Args:
        shard_pattern: glob pattern for shard files, e.g., "gs://bucket/train-*.tfrecord"
        cfg: TFPipelineConfig
        batch_size: examples per batch
        is_training: if True, shuffle and augment; if False, deterministic

    Returns:
        tf.data.Dataset yielding dicts with 'video' and 'label' keys
    """
    AUTOTUNE = tf.data.AUTOTUNE

    # Step 1: Shard-level shuffle
    files = tf.data.Dataset.list_files(shard_pattern, shuffle=is_training)

    # Step 2: Parallel shard reading with interleave
    dataset = files.interleave(
        lambda f: tf.data.TFRecordDataset(f, buffer_size=64 * 1024 * 1024),
        cycle_length=AUTOTUNE,
        num_parallel_calls=AUTOTUNE,
        deterministic=not is_training,
    )

    # Step 3: Example-level shuffle (training only)
    if is_training:
        dataset = dataset.shuffle(buffer_size=cfg.shuffle_buffer, reshuffle_each_iteration=True)

    # Step 4: Parse + augment
    parse_fn = _make_parse_fn(cfg, is_training)
    dataset = dataset.map(parse_fn, num_parallel_calls=AUTOTUNE)

    # Step 5: Batch with drop_remainder for even batches
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # Step 6: Prefetch for pipeline overlap
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset
```

### Parse Function

```python
def _make_parse_fn(cfg, is_training: bool):
    """Factory that creates a parse function closed over cfg and is_training."""

    MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    STD  = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

    def parse_fn(example_proto):
        feature_spec = {
            'video_bytes': tf.io.FixedLenFeature([], tf.string),
            'label':       tf.io.FixedLenFeature([], tf.int64),
            'num_frames':  tf.io.FixedLenFeature([], tf.int64),
            'height':      tf.io.FixedLenFeature([], tf.int64),
            'width':       tf.io.FixedLenFeature([], tf.int64),
        }
        parsed = tf.io.parse_single_example(example_proto, feature_spec)

        num_frames = tf.cast(parsed['num_frames'], tf.int32)
        height     = tf.cast(parsed['height'],     tf.int32)
        width      = tf.cast(parsed['width'],      tf.int32)

        # Decode raw bytes
        video = tf.io.decode_raw(parsed['video_bytes'], tf.uint8)
        video = tf.reshape(video, [num_frames, height, width, 3])

        # Cast to float32 and normalize to [0, 1]
        video = tf.cast(video, tf.float32) / 255.0

        # Augment if training
        if is_training and cfg.augment:
            video = _augment_video(video, cfg)

        # ImageNet normalization
        video = (video - MEAN) / STD

        label = tf.cast(parsed['label'], tf.int32)
        return {'video': video, 'label': label}

    return parse_fn
```

### Augmentation Function for tf.data

```python
def _augment_video(video: tf.Tensor, cfg) -> tf.Tensor:
    """
    Apply temporally consistent augmentations to a video clip.

    video: shape (T, H, W, C), float32, range [0, 1]
    Returns: shape (T, crop_size, crop_size, C), float32, range [0, 1]
    """
    T = tf.shape(video)[0]
    H = tf.shape(video)[1]
    W = tf.shape(video)[2]
    C = tf.shape(video)[3]

    # Random crop: same crop box for all frames
    crop_h = cfg.crop_size
    crop_w = cfg.crop_size
    offset_h = tf.random.uniform([], 0, H - crop_h + 1, dtype=tf.int32)
    offset_w = tf.random.uniform([], 0, W - crop_w + 1, dtype=tf.int32)
    video = video[:, offset_h:offset_h + crop_h, offset_w:offset_w + crop_w, :]

    # Random horizontal flip: same decision for all frames
    flip = tf.random.uniform([]) < 0.5
    video = tf.cond(flip, lambda: tf.image.flip_left_right(video), lambda: video)

    # Random brightness/contrast (applied per-frame but with same seed is complex in tf)
    # Simpler: apply same random brightness offset to all frames
    delta = tf.random.uniform([], -0.1, 0.1)
    video = tf.clip_by_value(video + delta, 0.0, 1.0)

    return video
```

---

## TPU Pod Considerations

### Data Must Be on GCS

TPU pods cannot read from local disk. All TFRecord files must be on Google Cloud Storage:

```
gs://your-bucket/dataset/train-*.tfrecord
gs://your-bucket/dataset/val-*.tfrecord
```

Adjust the GCS URI format in your `shard_pattern` accordingly.

### Explicit Shard Assignment

For maximum control over data distribution across TPU hosts, use `tf.distribute.Strategy`:

```python
strategy = tf.distribute.TPUStrategy(resolver)

with strategy.scope():
    dataset = build_tf_video_pipeline(shard_pattern, cfg, batch_size // num_replicas, True)
    dist_dataset = strategy.experimental_distribute_dataset(dataset)
```

The `experimental_distribute_dataset` method auto-shards the TFRecord files across replicas. No manual shard assignment needed if using this API.

### drop_remainder=True is Required

TPU requires fixed tensor shapes. A batch that is smaller than `batch_size` (from the last incomplete batch) causes a shape mismatch error. Always set `drop_remainder=True`.

### Deterministic Mode for Reproducibility

```python
options = tf.data.Options()
options.deterministic = True
dataset = dataset.with_options(options)
```

Note: deterministic mode can reduce throughput because it prevents out-of-order completion of parallel ops. Use for debugging or exact result reproduction, not for production training.

---

## Performance Tuning

### Buffer Sizes

```python
# TFRecordDataset internal buffer (bytes)
tf.data.TFRecordDataset(filepath, buffer_size=64 * 1024 * 1024)  # 64 MB

# Shuffle buffer (examples, not bytes)
dataset.shuffle(buffer_size=10000)  # pre-load 10k examples for shuffle
```

### Prefetch and Parallel Calls

Always use `tf.data.AUTOTUNE` for `num_parallel_calls` and `cycle_length` — TensorFlow will tune these dynamically based on available resources.

```python
dataset = dataset.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.prefetch(tf.data.AUTOTUNE)
```

### GCS Throughput

Reading from GCS is latency-sensitive for small files. With large TFRecord shards (100MB+) and `interleave` with many parallel readers, you should achieve near-full bandwidth utilization. Monitor with TensorFlow Profiler's input pipeline analysis.

---

## Summary

| Aspect | Recommendation |
|--------|---------------|
| Encoding | Raw uint8 for speed, JPEG for storage efficiency |
| Shard count | 10x per TPU host, each 100MB+ |
| Shard naming | `{split}-{shard_id:05d}-of-{num_shards:05d}.tfrecord` |
| Shuffle | Shard-level + example-level (buffer 10000) |
| Parallelism | AUTOTUNE for all parallel_calls and cycle_length |
| Batching | drop_remainder=True always |
| Prefetch | AUTOTUNE |
| Data location | GCS always for TPU |
| Determinism | Options.deterministic=True only for debugging |
