# Augmentation Pipeline — Reference for Data Pipeline & Loaders Skill

This document specifies the per-modality augmentation strategies, the `AugmentationPipeline` class interface, strength presets, composition rules, and determinism guarantees. Use this as the canonical reference when implementing, auditing, or extending data augmentation for the brain_ai training pipeline.

---

## 1. AugmentationPipeline Class Interface

The `AugmentationPipeline` is the central augmentation controller. It produces modality-aware transform chains that can be injected into any phase loader.

### 1.1 Constructor

```python
class AugmentationPipeline:
    def __init__(
        self,
        modality: str,
        strength: str = "standard",
        seed: Optional[int] = None,
    ):
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `modality` | `str` | required | One of `"vision"`, `"text"`, `"audio"`, `"sequence"`, `"rl"` |
| `strength` | `str` | `"standard"` | One of `"none"`, `"light"`, `"standard"`, `"heavy"` |
| `seed` | `Optional[int]` | `None` | If set, all random transforms become deterministic |

### 1.2 Core Methods

#### `get_train_transforms() -> Callable`

Returns a composed callable that takes a single sample tensor and returns the augmented version. For training only. The callable accepts the appropriate tensor shape for the modality and returns the same shape.

#### `get_eval_transforms() -> Callable`

Returns a composed callable for validation/test data. This performs only deterministic preprocessing (normalization, center crop, etc.) with no randomness. Identical output regardless of seed or call count.

#### `add_transform(transform: Callable, prob: float = 1.0) -> None`

Appends a custom transform to the pipeline. The `prob` parameter controls application probability (0.0 to 1.0). Custom transforms are applied after built-in transforms.

#### `set_seed(seed: int) -> None`

Sets the random seed for all stochastic transforms. When set, repeated calls to the train transforms with the same input produce identical output.

---

## 2. Vision Augmentations

Vision augmentations operate on tensors of shape `[C, H, W]` (single sample, no batch dimension). All operations preserve the tensor's dtype (`float32`) and value range.

### 2.1 Transform Catalog

#### RandomCrop

Crops a random region from the image and optionally resizes back to the original dimensions. Includes reflection padding when the crop region exceeds image boundaries.

```
Parameters: size (int or tuple), padding (int) = 4, padding_mode = "reflect"
Input: [C, H, W] -> Output: [C, size, size]
```

**Strength mapping:**
- none: Disabled
- light: padding=2
- standard: padding=4
- heavy: padding=8

#### HorizontalFlip

Mirrors the image horizontally with probability `p`.

```
Parameters: p (float)
Input: [C, H, W] -> Output: [C, H, W]
```

**Strength mapping:**
- none: p=0.0
- light: p=0.3
- standard: p=0.5
- heavy: p=0.5

#### ColorJitter

Randomly adjusts brightness, contrast, saturation, and hue. Only applicable to multi-channel (RGB) images. Skipped for single-channel grayscale.

```
Parameters: brightness, contrast, saturation, hue (all floats)
Input: [C, H, W] -> Output: [C, H, W]
```

**Strength mapping:**
- none: All zero
- light: brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02
- standard: brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
- heavy: brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1

#### Cutout (Random Erasing)

Randomly selects a rectangular region and fills it with zeros or random values. Forces the network to learn redundant representations.

```
Parameters: num_holes (int), hole_size (int or fraction)
Input: [C, H, W] -> Output: [C, H, W]
```

**Strength mapping:**
- none: Disabled
- light: 1 hole, 1/8 of image
- standard: 1 hole, 1/4 of image
- heavy: 2 holes, 1/4 of image each

#### RandAugment

Applies N randomly selected transforms from a pool, each at magnitude M. The pool includes: rotation, shear, translate, brightness, contrast, sharpness, posterize, solarize, auto-contrast, equalize.

```
Parameters: N (int, number of ops), M (int, magnitude 0-30)
Input: [C, H, W] -> Output: [C, H, W]
```

**Strength mapping:**
- none: Disabled
- light: N=1, M=5
- standard: N=2, M=9
- heavy: N=3, M=15

#### Normalization (eval-only)

Applied in both train and eval transforms as the final step. Centers and scales pixel values.

```
Dev mode: Normalize to [0, 1] (divide by 255 if uint8, or identity if already float)
Production mode: ImageNet normalization mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
```

### 2.2 Vision Composition Order

Transforms are applied in this fixed order:

```
1. RandomCrop (with padding)      [train only]
2. HorizontalFlip                 [train only]
3. ColorJitter                    [train only, RGB only]
4. RandAugment                    [train only, standard+heavy only]
5. Cutout                         [train only]
6. ToFloat / Normalize            [train and eval]
```

This order is chosen because spatial transforms (crop, flip) should precede color transforms (jitter), and destructive transforms (cutout) should be applied last before normalization.

---

## 3. Text Augmentations

Text augmentations operate on token ID tensors of shape `[seq_len]` (int64). They must preserve token validity (all IDs remain in `[0, vocab_size)`).

### 3.1 Transform Catalog

#### Token Dropout

Randomly replaces tokens with a mask token or a special unknown token.

```
Parameters: p (float, dropout probability), mask_token_id (int)
Input: [seq_len] int64 -> Output: [seq_len] int64
```

**Strength mapping:**
- none: p=0.0
- light: p=0.05
- standard: p=0.10
- heavy: p=0.20

#### Synonym Replacement

Replaces random tokens with synonym tokens from a small synonym table. In dev mode, uses a trivial synonym map (token ID +/- 1). In production, uses a real synonym lookup.

```
Parameters: p (float, replacement probability), synonym_map (Dict[int, List[int]])
Input: [seq_len] int64 -> Output: [seq_len] int64
```

**Strength mapping:**
- none: p=0.0
- light: p=0.05
- standard: p=0.10
- heavy: p=0.15

#### Token Shuffle

Randomly shuffles tokens within a small window. Preserves local word order approximately while adding noise.

```
Parameters: window_size (int)
Input: [seq_len] int64 -> Output: [seq_len] int64
```

**Strength mapping:**
- none: Disabled
- light: window_size=2
- standard: window_size=3
- heavy: window_size=5

### 3.2 Text Composition Order

```
1. Token Dropout                  [train only]
2. Synonym Replacement            [train only]
3. Token Shuffle                  [train only, standard+heavy]
4. Padding verification           [train and eval]
```

**Important:** Padding tokens (ID 0) must never be modified by augmentations. All text transforms must check the attention mask and skip padded positions.

---

## 4. Audio Augmentations

Audio augmentations operate on log-mel spectrogram tensors of shape `[n_mels, T]` (float32). They simulate real-world acoustic variations.

### 4.1 Transform Catalog

#### SpecAugment

The standard speech augmentation from Park et al. (2019). Applies frequency masking and time masking independently.

**Frequency masking:** Selects a random contiguous band of `F` frequency bins and sets them to zero (or the mean value).

```
Parameters: F (int, max frequency mask width), num_freq_masks (int)
```

**Time masking:** Selects a random contiguous span of `T_mask` time frames and sets them to zero.

```
Parameters: T_mask (int, max time mask width), num_time_masks (int)
```

**Strength mapping:**
- none: Disabled
- light: F=10, num_freq=1, T_mask=10, num_time=1
- standard: F=20, num_freq=2, T_mask=30, num_time=2
- heavy: F=30, num_freq=3, T_mask=50, num_time=3

#### Time Stretch

Stretches or compresses the time axis by a random factor, then resamples to the original length. Simulates speaking rate variation.

```
Parameters: rate_range (tuple of float), e.g. (0.8, 1.2)
Input: [n_mels, T] -> Output: [n_mels, T] (resampled)
```

**Strength mapping:**
- none: Disabled
- light: rate_range=(0.95, 1.05)
- standard: rate_range=(0.9, 1.1)
- heavy: rate_range=(0.8, 1.25)

#### Noise Injection

Adds Gaussian noise to the spectrogram. Simulates background noise.

```
Parameters: noise_level (float, standard deviation)
Input: [n_mels, T] -> Output: [n_mels, T]
```

**Strength mapping:**
- none: noise_level=0.0
- light: noise_level=0.005
- standard: noise_level=0.01
- heavy: noise_level=0.02

#### Pitch Shift (Frequency Shift)

Shifts all frequency bins up or down by a random number of bins. Simulates pitch variation. Implemented as a roll along the frequency axis with zero-padding.

```
Parameters: max_shift (int, max bins to shift)
Input: [n_mels, T] -> Output: [n_mels, T]
```

**Strength mapping:**
- none: max_shift=0
- light: max_shift=2
- standard: max_shift=4
- heavy: max_shift=8

### 4.2 Audio Composition Order

```
1. Time Stretch                   [train only]
2. Pitch Shift                    [train only]
3. SpecAugment (freq mask)        [train only]
4. SpecAugment (time mask)        [train only]
5. Noise Injection                [train only]
6. Normalization                  [train and eval]
```

Time stretch is applied first because it changes the temporal structure. SpecAugment masking is applied after pitch/time modifications. Noise is added last before normalization.

---

## 5. Sequence Augmentations

Sequence augmentations operate on tensors of shape `[T, D]` (float32) with an associated mask `[T]` (bool). They must respect the mask and only modify valid timesteps.

### 5.1 Transform Catalog

#### Temporal Jittering

Adds small random offsets to the time indices, effectively resampling the sequence at slightly shifted positions. Implemented via linear interpolation.

```
Parameters: sigma (float, jitter standard deviation in timesteps)
Input: [T, D] -> Output: [T, D]
```

**Strength mapping:**
- none: sigma=0.0
- light: sigma=0.5
- standard: sigma=1.0
- heavy: sigma=2.0

#### Gaussian Noise Addition

Adds per-element Gaussian noise to all features at all timesteps.

```
Parameters: noise_std (float)
Input: [T, D] -> Output: [T, D]
```

**Strength mapping:**
- none: noise_std=0.0
- light: noise_std=0.01
- standard: noise_std=0.03
- heavy: noise_std=0.05

#### Subsequence Sampling

Randomly selects a contiguous subsequence of length `L` from the full sequence. Used when sequences are very long and the model should learn from partial views.

```
Parameters: min_ratio (float), max_ratio (float)
Input: [T, D] -> Output: [T', D] where T' = int(T * uniform(min_ratio, max_ratio))
```

**Strength mapping:**
- none: Full sequence (ratio=1.0)
- light: min_ratio=0.9, max_ratio=1.0
- standard: min_ratio=0.7, max_ratio=1.0
- heavy: min_ratio=0.5, max_ratio=1.0

#### Feature Dropout

Randomly zeroes out entire feature dimensions across all timesteps. Forces the model to not rely on any single feature.

```
Parameters: p (float, dropout probability per feature)
Input: [T, D] -> Output: [T, D]
```

**Strength mapping:**
- none: p=0.0
- light: p=0.05
- standard: p=0.1
- heavy: p=0.2

### 5.2 Sequence Composition Order

```
1. Subsequence Sampling           [train only, standard+heavy]
2. Temporal Jittering             [train only]
3. Gaussian Noise Addition        [train only]
4. Feature Dropout                [train only]
5. Normalization                  [train and eval]
```

Subsequence sampling is applied first to reduce length before other transforms. Noise and dropout are applied last.

---

## 6. RL Episode Augmentations

For active inference phase 5, RL trajectory data should NOT be augmented by default. Augmenting states, actions, or rewards can break the Markov property and invalidate the trajectory's dynamics.

The only permitted modification is observation normalization (running mean/std normalization of state vectors), which is applied identically in train and eval.

If augmentation is explicitly requested for RL data, only the following are safe:
- State noise injection (very small, <0.01 std)
- Reward scaling (multiplicative constant)

These are disabled by default regardless of strength preset.

---

## 7. Strength Presets Summary

| Preset | Purpose | Training Impact | Typical Use |
|---|---|---|---|
| `none` | No augmentation | Baseline, fastest iteration | Debugging, ablation studies |
| `light` | Minimal augmentation | Slight regularization, fast | Early training phases, small datasets |
| `standard` | Balanced augmentation | Good regularization | Default for most training |
| `heavy` | Aggressive augmentation | Strong regularization | Large datasets, overfitting prevention |

### 7.1 Preset Selection Guidelines

- **Phase 1 (SNN):** `standard` -- the SNN core benefits from standard vision augmentation.
- **Phase 2 (Encoders):** `standard` for vision, `light` for text and audio -- encoders are training from scratch and need regularization but not extreme augmentation.
- **Phase 3 (HTM):** `light` -- temporal sequences are sensitive to heavy modification.
- **Phase 4 (Workspace):** `standard` -- multimodal integration benefits from diverse views.
- **Phase 5 (Active Inference):** `none` -- RL trajectories must not be modified.
- **Phase 6 (Reasoning):** `light` -- logic tasks need minimal augmentation to avoid changing semantics.
- **Phase 7 (Meta-learning):** `standard` -- few-shot episodes benefit from augmentation to improve generalization.

---

## 8. Determinism Guarantees

### 8.1 Seeded Determinism

When `seed` is set in the `AugmentationPipeline` constructor:
1. A `torch.Generator` is created with the given seed.
2. All stochastic transforms use this generator for random number generation.
3. Calling `get_train_transforms()` and applying the result to the same input tensor produces identical output every time.
4. The generator state advances deterministically, so the N-th application always produces the same result.

### 8.2 Eval Transform Determinism

`get_eval_transforms()` is always deterministic regardless of seed setting. It applies only fixed normalization and center crop operations. Two calls with the same input always produce bitwise-identical output.

### 8.3 Cross-Worker Determinism

When using `num_workers > 0` in DataLoader, each worker gets a different seed derived from the base seed plus the worker ID. This ensures:
- Different workers produce different augmentations (desirable for diversity).
- The same worker ID with the same base seed produces identical augmentations across runs (reproducibility).

The seed derivation formula is: `worker_seed = base_seed + worker_id * 1000`.

### 8.4 Testing Determinism

To verify determinism in tests:
```python
pipeline = AugmentationPipeline("vision", strength="standard", seed=42)
transform = pipeline.get_train_transforms()
x = torch.randn(3, 32, 32)
y1 = transform(x.clone())

pipeline.set_seed(42)  # Reset seed
transform = pipeline.get_train_transforms()
y2 = transform(x.clone())

assert torch.allclose(y1, y2)  # Must pass
```

---

## 9. Custom Transform Integration

Users can add custom transforms via `add_transform()`. Custom transforms are appended after all built-in transforms but before final normalization.

```python
pipeline = AugmentationPipeline("vision", strength="standard")

def my_custom_blur(x: Tensor) -> Tensor:
    # Apply Gaussian blur
    kernel_size = 3
    padding = kernel_size // 2
    kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2)
    kernel = kernel.expand(x.shape[0], -1, -1, -1)
    return torch.nn.functional.conv2d(x.unsqueeze(0), kernel, padding=padding, groups=x.shape[0]).squeeze(0)

pipeline.add_transform(my_custom_blur, prob=0.3)
```

Requirements for custom transforms:
- Must accept and return a single tensor of the correct modality shape.
- Must not change the tensor's dtype.
- Must not change the tensor's shape (unless the pipeline explicitly supports shape changes).
- Should be picklable (for multi-worker DataLoader compatibility).
