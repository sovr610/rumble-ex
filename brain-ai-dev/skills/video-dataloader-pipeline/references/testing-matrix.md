# Testing Matrix for Video DataLoader Pipeline

## Overview

This document organizes test scenarios by component phase. Each phase builds on the previous — run them in order during development and CI. Every phase has unit tests (fast, no I/O) and integration tests (require video files or TFRecord files). Mark integration tests with `@pytest.mark.integration` and skip in fast CI runs.

---

## Phase 1: VideoDataset

### 1.1 Frame Shape Tests

| Test | What to verify | Expected |
|------|---------------|----------|
| `test_getitem_output_shape` | `__getitem__` returns dict with 'video' key | shape `(T, C, H, W)` |
| `test_getitem_video_ndim` | video tensor has 4 dimensions | ndim == 4 |
| `test_getitem_channel_dim` | channel dimension is 3 (RGB) | video.shape[1] == 3 |
| `test_getitem_spatial_dim` | spatial dims match crop_size | H == W == crop_size |
| `test_getitem_temporal_dim` | temporal dim matches num_frames | T == cfg.num_frames |
| `test_getitem_label_present` | dict contains 'label' key | isinstance(label, int) |

### 1.2 Value Range Tests

| Test | What to verify | Expected |
|------|---------------|----------|
| `test_getitem_dtype_float32` | video tensor is float32 | dtype == torch.float32 |
| `test_getitem_range_min` | minimum value before normalize >= 0 | min >= 0.0 |
| `test_getitem_range_max` | maximum value before normalize <= 1 | max <= 1.0 |
| `test_getitem_not_all_zeros` | tensor is not degenerate | mean > 0.001 |
| `test_getitem_not_all_ones` | tensor is not saturated | mean < 0.999 |

### 1.3 Temporal Sampling Correctness

| Test | What to verify | Expected |
|------|---------------|----------|
| `test_sampling_indices_count` | exactly num_frames indices produced | len(indices) == num_frames |
| `test_sampling_indices_clamped` | all indices within [0, total-1] | all(0 <= i < total) |
| `test_sampling_short_video` | short videos handled without exception | no IndexError |
| `test_sampling_stride_monotone` | indices are monotonically non-decreasing | indices[i] <= indices[i+1] |
| `test_sampling_single_frame_video` | 1-frame video: all indices are 0 | all(i == 0 for i in indices) |
| `test_sampling_random_start_varies` | repeated calls give different starts | at least 2 unique starts in 100 calls |

### 1.4 Metadata Caching

| Test | What to verify | Expected |
|------|---------------|----------|
| `test_manifest_json_written` | scan_manifest writes JSON file | os.path.exists(manifest_path) |
| `test_manifest_loaded_from_cache` | second call reads from JSON | scan called once, load called twice |
| `test_manifest_fields` | each VideoMeta has required fields | path, num_frames, fps, label |
| `test_manifest_num_frames_positive` | all videos have > 0 frames | all(m.num_frames > 0) |
| `test_manifest_corrupted_skip` | corrupted video files are skipped | no exception raised |

### 1.5 Multi-Worker Loading

| Test | What to verify | Expected |
|------|---------------|----------|
| `test_dataloader_num_workers_2` | DataLoader with num_workers=2 produces batches | 5 batches without deadlock |
| `test_dataloader_no_file_handle_leak` | after 100 iterations, no unclosed handles | psutil.num_fds() within 50 of start |
| `test_dataloader_worker_independence` | different workers produce different samples | not all batches identical |
| `test_dataloader_pin_memory` | pin_memory=True, tensors are pinned | batch['video'].is_pinned() == True |

---

## Phase 2: Augmentation

### 2.1 Output Shape Tests

| Test | What to verify | Expected |
|------|---------------|----------|
| `test_train_transform_shape` | train transform output shape | (T, 3, crop_size, crop_size) |
| `test_eval_transform_shape` | eval transform output shape | (T, 3, crop_size, crop_size) |
| `test_prepare_video_tensor_shape` | permute + float + Video wrapper | (T, C, H, W) float32 |

### 2.2 Value Range After Normalize

| Test | What to verify | Expected |
|------|---------------|----------|
| `test_normalize_zero_mean` | per-channel mean near 0 on ImageNet data | abs(mean) < 0.5 |
| `test_normalize_range` | values not strictly bounded post-normalize | min < 0 and max > 0 |
| `test_prepare_tensor_range` | before normalize: [0, 1] | 0.0 <= min and max <= 1.0 |
| `test_prepare_tensor_dtype` | output is float32 | dtype == torch.float32 |

### 2.3 Spatial Consistency Across T Frames

| Test | What to verify | Expected |
|------|---------------|----------|
| `test_flip_consistency` | all frames flipped or none | if flip: all frames mirror of original |
| `test_crop_consistency` | same spatial region cropped from all frames | crop region identical across T |
| `test_color_jitter_consistency` | same color transform across frames | per-channel stats differ from original |
| `test_jitter_same_params_all_frames` | jitter delta same for frame 0 and frame T-1 | delta_0 == delta_T-1 |

### 2.4 Eval vs Train Transforms

| Test | What to verify | Expected |
|------|---------------|----------|
| `test_eval_deterministic` | same input → same output for eval | output_1 == output_2 |
| `test_train_stochastic` | same input → different output for train | output_1 != output_2 (usually) |
| `test_eval_no_flip` | eval transform never flips | left != right markers preserved |
| `test_eval_center_crop` | eval crops center, not random | crop box == center of frame |

---

## Phase 3: TFRecordConverter

### 3.1 Shard Count and File Sizes

| Test | What to verify | Expected |
|------|---------------|----------|
| `test_shard_count` | correct number of files written | num_files == num_shards |
| `test_shard_names` | naming convention correct | f"{split}-{i:05d}-of-{n:05d}.tfrecord" |
| `test_shard_nonempty` | no empty shard files | all(os.path.getsize(f) > 0 for f in shards) |
| `test_shard_distribution_even` | examples distributed evenly | max_count - min_count <= 1 |

### 3.2 Round-Trip Decode Correctness

| Test | What to verify | Expected |
|------|---------------|----------|
| `test_roundtrip_shape` | decoded shape matches written shape | decoded.shape == original.shape |
| `test_roundtrip_values_raw` | raw encoding is lossless | decoded == original (byte-exact) |
| `test_roundtrip_values_jpeg` | JPEG encoding within tolerance | abs(decoded - original).mean() < 5.0 |
| `test_roundtrip_label` | label preserved correctly | decoded_label == original_label |

### 3.3 JPEG Compression Quality

| Test | What to verify | Expected |
|------|---------------|----------|
| `test_jpeg_smaller_than_raw` | JPEG bytes < raw bytes | len(jpeg) < len(raw) |
| `test_jpeg_quality_95` | at quality 95, PSNR > 35 dB | psnr > 35.0 |
| `test_jpeg_quality_effect` | lower quality → smaller size | size_80 < size_95 |

### 3.4 Example Feature Structure

| Test | What to verify | Expected |
|------|---------------|----------|
| `test_example_has_video_bytes` | video_bytes feature present | 'video_bytes' in features |
| `test_example_has_label` | label feature present | 'label' in features |
| `test_example_has_metadata` | num_frames, height, width present | all present |
| `test_example_serializable` | example serializes to bytes | len(example.SerializeToString()) > 0 |

---

## Phase 4: tf.data Pipeline

### 4.1 Batch Shapes

| Test | What to verify | Expected |
|------|---------------|----------|
| `test_pipeline_batch_shape_video` | video batch shape | (batch_size, T, H, W, C) |
| `test_pipeline_batch_shape_label` | label batch shape | (batch_size,) |
| `test_pipeline_drop_remainder` | no partial batches | all batches have batch_size examples |
| `test_pipeline_dtype_float32` | output is float32 | video.dtype == tf.float32 |

### 4.2 Value Range

| Test | What to verify | Expected |
|------|---------------|----------|
| `test_pipeline_range_normalized` | values span negative range (normalized) | min < 0.0 |
| `test_pipeline_mean_near_zero` | per-channel mean near 0 | abs(mean) < 1.0 |

### 4.3 Shuffle Randomness

| Test | What to verify | Expected |
|------|---------------|----------|
| `test_pipeline_shuffle_varies` | two training runs produce different batch order | batch_1_labels != batch_2_labels |
| `test_pipeline_eval_deterministic` | two eval runs produce same order | batch_1_labels == batch_2_labels |
| `test_pipeline_shard_shuffle` | shard order randomized between epochs | different shard read order |

### 4.4 Deterministic Mode

| Test | What to verify | Expected |
|------|---------------|----------|
| `test_deterministic_option` | deterministic=True gives same output twice | run_1 == run_2 |
| `test_deterministic_vs_stochastic` | with vs without deterministic differ | (may differ if no shuffle) |

---

## Phase 5: DataModule

### 5.1 DataLoader Iteration

| Test | What to verify | Expected |
|------|---------------|----------|
| `test_train_loader_iterates` | train_dataloader() yields batches | 5 batches without exception |
| `test_val_loader_iterates` | val_dataloader() yields batches | all val batches consumed |
| `test_test_loader_iterates` | test_dataloader() yields batches | all test batches consumed |
| `test_train_batch_shape` | train batch shape correct | (B, T, C, H, W) |

### 5.2 Worker Count

| Test | What to verify | Expected |
|------|---------------|----------|
| `test_num_workers_1gpu` | 1 GPU → num_workers_per_gpu workers | num_workers == cfg.num_workers_per_gpu |
| `test_num_workers_4gpu` | 4 GPUs → 4 * num_workers_per_gpu | num_workers == 4 * cfg.num_workers_per_gpu |
| `test_num_workers_0_allowed` | num_workers_per_gpu=0 → no workers | num_workers == 0 |

### 5.3 Distributed Sampler Presence

| Test | What to verify | Expected |
|------|---------------|----------|
| `test_no_manual_distributed_sampler` | DataModule does not create DistributedSampler | no DistributedSampler in dataloader kwargs |
| `test_lightning_injects_sampler` | after Trainer.fit, sampler is DistributedSampler | type(sampler).__name__ == 'DistributedSampler' |
| `test_train_loader_has_shuffle` | shuffle=True in train loader | loader.dataset is shuffled |
| `test_val_loader_no_shuffle` | shuffle=False in val loader | deterministic ordering |

### 5.4 Stage-Based Dataset Creation

| Test | What to verify | Expected |
|------|---------------|----------|
| `test_setup_fit_creates_train_val` | stage='fit' creates train + val | train_dataset and val_dataset not None |
| `test_setup_test_creates_test` | stage='test' creates test | test_dataset not None |
| `test_setup_validate_creates_val` | stage='validate' creates val | val_dataset not None |
| `test_setup_none_creates_all` | stage=None creates all | all three datasets not None |

---

## Phase 6: Integration

### 6.1 End-to-End PyTorch Path

```
video file → VideoDataset.__getitem__ → augmented tensor → DataLoader batch
```

| Test | What to verify | Expected |
|------|---------------|----------|
| `test_e2e_pytorch_shape` | batch from real video file has correct shape | (B, T, C, H, W) |
| `test_e2e_pytorch_dtype` | batch is float32 | dtype == torch.float32 |
| `test_e2e_pytorch_range` | batch values in normalized range | min < 0 (after normalize) |
| `test_e2e_pytorch_label` | label batch has correct length | len(labels) == batch_size |
| `test_e2e_pytorch_workers` | num_workers=2 produces correct batches | 5 batches, no deadlock |

### 6.2 End-to-End TFRecord Path

```
video file → TFRecordConverter → .tfrecord shards → tf.data pipeline → batch
```

| Test | What to verify | Expected |
|------|---------------|----------|
| `test_e2e_tf_shards_exist` | conversion produces expected shard files | num_shards files created |
| `test_e2e_tf_batch_shape` | pipeline batch shape correct | (B, T, H, W, C) or (B, T, C, H, W) |
| `test_e2e_tf_roundtrip_values` | decoded values close to original | max abs diff < 5.0 (JPEG) or 0.0 (raw) |
| `test_e2e_tf_n_batches` | pipeline yields multiple batches | at least 3 batches |

---

## Parametrize Targets

The following tests benefit from parametrization across multiple input configurations:

```python
@pytest.mark.parametrize("num_frames,stride,total_frames", [
    (8, 2, 16),
    (16, 4, 64),
    (32, 2, 60),       # stride overshoots end
    (16, 1, 10),       # very short video
    (4, 8, 200),       # long video, sparse sampling
])
def test_frame_sampling_shapes(num_frames, stride, total_frames):
    ...

@pytest.mark.parametrize("crop_size", [112, 224, 256])
def test_transform_output_shape(crop_size):
    ...

@pytest.mark.parametrize("num_shards", [1, 4, 16])
def test_shard_distribution(num_shards):
    ...

@pytest.mark.parametrize("compression", ["none", "jpeg"])
def test_encoding_roundtrip(compression):
    ...

@pytest.mark.parametrize("cfg_cls", [
    VideoDatasetConfig, DataModuleConfig, TFRecordConfig, TFPipelineConfig
])
def test_config_roundtrip(cfg_cls):
    ...
```

---

## CI Integration

Recommended pytest markers:

```python
# pytest.ini or pyproject.toml
[pytest]
markers =
    unit: fast unit tests, no I/O (< 1 second each)
    integration: requires video files or TFRecord files (10+ seconds each)
    gpu: requires CUDA GPU
    tf: requires TensorFlow installation
    slow: benchmark-level tests
```

CI commands:

```bash
# Fast unit tests only (no video files needed)
pytest -m "unit" tests/ -v

# Full suite with integration (needs sample video files)
pytest -m "unit or integration" tests/ -v

# TF tests (needs tensorflow installed)
pytest -m "tf" tests/ -v --tb=short
```
