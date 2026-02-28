# Testing Matrix Reference

## Overview

This matrix covers all test scenarios for the V-JEPA 2 data pipeline, organized
by component. Each scenario specifies inputs, expected outputs, and what property
is being verified. The `gen_data_tests.py` script generates pytest code for all
scenarios listed here.

---

## 1. Video Loading Shapes

### 1.1 clip_mode="fps" — Standard Case

| Parameter | Value |
|-----------|-------|
| total_frames | 300 |
| native_fps | 30.0 |
| frames_per_clip | 16 |
| target_fps | 10 |
| img_size | 224 |

Expected frame_step = round(30/10) = 3
Expected indices: 16 indices in [0, 300), each separated by step=3
Expected output shape: `[3, 16, 224, 224]` (C, T, H, W)

### 1.2 clip_mode="fps" — Low Native FPS

| Parameter | Value |
|-----------|-------|
| total_frames | 50 |
| native_fps | 8.0 |
| frames_per_clip | 8 |
| target_fps | 10 |
| img_size | 112 |

frame_step = max(1, round(8/10)) = 1 (clamp to 1)
Expected output shape: `[3, 8, 112, 112]`

### 1.3 clip_mode="duration"

| Parameter | Value |
|-----------|-------|
| total_frames | 900 |
| native_fps | 30.0 |
| clip_duration_sec | 3.0 |
| frames_per_clip | 16 |
| img_size | 224 |

clip_length_frames = int(3.0 * 30) = 90
indices = linspace(start, start+89, 16, dtype=int)
Expected output shape: `[3, 16, 224, 224]`

### 1.4 clip_mode="frame_step"

| Parameter | Value |
|-----------|-------|
| total_frames | 200 |
| frame_step | 4 |
| frames_per_clip | 16 |
| img_size | 224 |

native_needed = 4*(16-1)+1 = 61
Expected indices: 16 values with stride 4
Expected output shape: `[3, 16, 224, 224]`

### 1.5 Circulant Padding — Short Video (fps mode)

| Parameter | Value |
|-----------|-------|
| total_frames | 5 |
| frames_per_clip | 16 |
| clip_mode | fps |

Before padding: indices may extend beyond 5
After padding: all indices in [0, 4]
Expected output shape: `[3, 16, 224, 224]`
Verify: tensor is NOT all zeros (frames were loaded)

### 1.6 Circulant Padding — Very Short Video (1 frame)

| Parameter | Value |
|-----------|-------|
| total_frames | 1 |
| frames_per_clip | 8 |
| clip_mode | fps |

After circulant padding: all 8 indices = 0
Expected output shape: `[3, 8, 224, 224]`
All frames should be identical copies of frame 0.

### 1.7 Return Dict Structure

VideoDataset.__getitem__ must return a dict with at minimum:
- `"video"`: Tensor of shape `[C, T, H, W]`, dtype float32
- Optional: `"label"` (int), `"path"` (str)

### 1.8 All Three Clip Modes Produce Valid Frame Counts

For FPC=16, each mode must return exactly 16 frames:
```python
for mode in ["fps", "duration", "frame_step"]:
    sample = dataset[0]
    assert sample["video"].shape[1] == 16
```

---

## 2. Transform Pipeline Shapes

### 2.1 Train Transform Output Shape

| Input | Operation | Output |
|-------|-----------|--------|
| `[16, 256, 256, 3]` uint8 PIL list | get_train_transform() | `[3, 16, 224, 224]` float32 |

### 2.2 Eval Transform Output Shape

Same input as 2.1, same expected output shape `[3, 16, 224, 224]`.

### 2.3 Eval Transform Determinism

```python
transform = pipeline.get_eval_transform()
frames = generate_random_frames(T=16, H=256, W=256)
out1 = transform(frames)
out2 = transform(frames)
assert torch.allclose(out1, out2)
```

### 2.4 Normalization Range

After normalization, mean per channel should be approximately 0 across
a large batch of natural video frames. Exact check: after ClipToTensor (before
normalize), pixel values in [0, 1]. After normalize, values roughly in [-2.5, 2.5].

```python
out = transform(frames)
assert out.min() > -5.0
assert out.max() < 5.0
```

### 2.5 Spatial Size Preserved

All transforms must produce exactly `img_size x img_size` spatial dimensions.

```python
assert out.shape[-2] == img_size
assert out.shape[-1] == img_size
```

### 2.6 Channel Dimension = 3

Output tensor must have exactly 3 channels (RGB).

```python
assert out.shape[0] == 3
```

### 2.7 RandomResizedCrop Temporal Consistency

The same crop box must be applied to all frames. Verify by checking that two
identical frames at different temporal positions receive the same crop:

```python
frames = [same_image] * 16
out = rrc_transform(frames)
# All spatial slices should be identical
for t in range(1, 16):
    assert torch.allclose(out[:, 0], out[:, t])
```

### 2.8 HorizontalFlip Temporal Consistency

Same flip decision for all frames:

```python
frames = [test_image_with_asymmetry] * 16
out = flip_transform(frames)
# Either all flipped or all original
flipped_ref = out[:, 0].flip(-1)
for t in range(1, 16):
    is_all_same = torch.allclose(out[:, t], out[:, 0])
    is_all_flipped = torch.allclose(out[:, t], flipped_ref)
    assert is_all_same or is_all_flipped
```

### 2.9 RandAugment Per-Frame Independence

When auto_augment=True, frames should NOT all be identical (independent
per-frame augmentation):

```python
frames = [same_image] * 16
out = rand_augment(frames)
# At least some frames should differ
diffs = [not torch.allclose(out[0], out[t]) for t in range(1, 16)]
assert any(diffs), "RandAugment should produce per-frame variation"
```

### 2.10 RandomErasing Cube Mode

When cube_mode=True, the same region is erased from all frames:

```python
out = random_erasing_cube(tensor)  # [C, T, H, W]
# Find erased pixels in frame 0
mask = (out[:, 0] == 0)
# All other frames should have zeros in the same region
for t in range(1, T):
    assert (out[:, t][mask] == 0).all()
```

---

## 3. Augmentation Determinism

### 3.1 Fixed Seed Reproducibility

```python
torch.manual_seed(42)
random.seed(42)
out1 = train_transform(frames)

torch.manual_seed(42)
random.seed(42)
out2 = train_transform(frames)

assert torch.allclose(out1, out2)
```

### 3.2 Different Seeds Produce Different Results

```python
torch.manual_seed(0); out0 = train_transform(frames)
torch.manual_seed(1); out1 = train_transform(frames)
assert not torch.allclose(out0, out1)
```

---

## 4. Multi-Source Weight Distribution

### 4.1 Weight Proportions Respected

Two sources with weights [1.0, 2.0] across 3000 total samples:
Source A: 1000 samples, weight 1.0
Source B: 2000 samples, weight 2.0

After one epoch of weighted sampling, count how many samples come from each source.
Expected: approximately 1/3 from A, 2/3 from B.

```python
# Tolerance: 5% deviation from expected proportion
expected_a = 1/3
actual_a = count_a / total_sampled
assert abs(actual_a - expected_a) < 0.05
```

### 4.2 ConcatIndices Mapping Correctness

```python
concat = ConcatIndices([1000, 500, 250])
assert concat[0]    == (0, 0)
assert concat[999]  == (0, 999)
assert concat[1000] == (1, 0)
assert concat[1499] == (1, 499)
assert concat[1500] == (2, 0)
assert concat[1749] == (2, 249)
assert len(concat)  == 1750
```

### 4.3 ConcatIndices Boundary Conditions

```python
with pytest.raises(IndexError):
    concat[-1]
with pytest.raises(IndexError):
    concat[1750]
```

### 4.4 ConcatIndices Single Source

```python
concat = ConcatIndices([500])
assert concat[0]   == (0, 0)
assert concat[499] == (0, 499)
assert len(concat) == 500
```

---

## 5. Sampler Coverage

### 5.1 No Samples Dropped (Single Rank)

With world_size=1, rank=0, all N samples must appear in one epoch:

```python
sampler = DistributedWeightedSampler(weights, N, rank=0, world_size=1)
indices = list(sampler)
assert len(set(indices)) == N  # No duplicates
assert len(indices) >= N
```

### 5.2 No Cross-Rank Duplicates

With world_size=2:

```python
s0 = DistributedWeightedSampler(weights, N, rank=0, world_size=2)
s1 = DistributedWeightedSampler(weights, N, rank=1, world_size=2)
idx0 = set(list(s0))
idx1 = set(list(s1))
assert len(idx0 & idx1) == 0  # Disjoint sets
```

### 5.3 Full Coverage Across Ranks

```python
all_indices = set(list(s0)) | set(list(s1))
assert all_indices == set(range(N))  # All samples covered
```

### 5.4 Weight Distribution Across Ranks

Both ranks should see similar proportions from each source (weights respected
at the global level, not per-rank):

```python
# Source A: 1000 samples, weight=2.0
# Source B: 1000 samples, weight=1.0
# Expected: ~2/3 from A per rank
prop_a_rank0 = len([i for i in list(s0) if i < 1000]) / len(list(s0))
assert 0.55 < prop_a_rank0 < 0.78
```

### 5.5 set_epoch Changes Ordering

```python
sampler.set_epoch(0); order0 = list(sampler)
sampler.set_epoch(1); order1 = list(sampler)
assert order0 != order1
```

### 5.6 Reproducibility Across Runs

```python
sampler.set_epoch(5); run1 = list(sampler)
sampler.set_epoch(5); run2 = list(sampler)
assert run1 == run2
```

---

## 6. Worker Seeding Reproducibility

### 6.1 Same Worker ID Produces Same Seed

```python
# Simulate worker_init_fn calls
def get_seed(worker_id, base_seed):
    LCG_A, LCG_C, LCG_M = 1664525, 1013904223, 2**32
    return (LCG_A * (base_seed + worker_id) + LCG_C) % LCG_M

s0 = get_seed(0, 12345)
s0_repeat = get_seed(0, 12345)
assert s0 == s0_repeat
```

### 6.2 Different Workers Get Different Seeds

```python
seeds = [get_seed(w, 12345) for w in range(8)]
assert len(set(seeds)) == 8  # All unique
```

### 6.3 DataLoader Reproducibility

Two DataLoaders with same config and same epoch seed must produce same batches:

```python
loader1 = build_train_loader(dataset, sampler, ...)
loader2 = build_train_loader(dataset, sampler, ...)

sampler.set_epoch(0)
batch1 = next(iter(loader1))
sampler.set_epoch(0)
batch2 = next(iter(loader2))
assert torch.allclose(batch1["video"], batch2["video"])
```

---

## 7. DataLoader Batch Structure

### 7.1 Batch Shape

```python
batch = next(iter(loader))
assert "video" in batch
assert batch["video"].shape == (batch_size, 3, frames_per_clip, img_size, img_size)
assert batch["video"].dtype == torch.float32
```

### 7.2 Mask Collator Integration

When a MaskCollator is provided:

```python
batch, masks_enc, masks_pred = next(iter(loader))
assert batch["video"].shape[0] == batch_size
assert len(masks_enc) == num_enc_masks
assert len(masks_pred) == num_pred_masks
```

### 7.3 drop_last Behavior

With `drop_last=True`, partial batches at end of epoch are dropped:

```python
n_batches = sum(1 for _ in loader)
assert n_batches == len(dataset) // batch_size
```

---

## 8. Config Validation

### 8.1 Valid Config Passes

```python
errors = validate_config(valid_config_dict)
assert errors == []
```

### 8.2 Invalid clip_mode Detected

```python
cfg = {"data": {"clip_mode": "invalid_mode"}}
errors = validate_config(cfg)
assert any("clip_mode" in e for e in errors)
```

### 8.3 Mismatched Weights/Paths

```python
cfg = {"data": {
    "data_paths": ["/a", "/b"],
    "data_weights": [1.0, 2.0, 3.0],  # Wrong length
}}
errors = validate_config(cfg)
assert any("weight" in e.lower() for e in errors)
```

### 8.4 YAML Round-Trip

```python
config = DataConfig(batch_size=32, frames_per_clip=8, clip_mode="duration")
yaml_str = config.to_yaml_dict()
config2 = DataConfig.from_dict(yaml_str)
assert config == config2
```

### 8.5 AugConfig Defaults

```python
cfg = AugConfig()
assert cfg.crop_scale == (0.3, 1.0)
assert cfg.horizontal_flip is True
assert cfg.auto_augment is False
assert cfg.random_erasing == 0.0
assert cfg.normalize_mean == (0.485, 0.456, 0.406)
```
