# Corruption Benchmarks Reference

## Overview

Corruption benchmarks evaluate model robustness to common, naturally-occurring input distortions. Unlike adversarial attacks (which are worst-case), corruption benchmarks test average-case robustness to noise, blur, weather effects, and digital artifacts. The standard framework is ImageNet-C (Hendrycks & Dietterich, 2019), which defines 15 corruption types at 5 severity levels.

For brain_ai, corruption benchmarks reveal how the multi-layer architecture degrades under realistic conditions. The SNN temporal coding and HTM sparse representations provide some natural corruption tolerance, but each corruption type affects different pipeline stages differently.

---

## Corruption Taxonomy

### Category 1: Noise

Noise corruptions add random intensity variations to pixel values. They test the low-level robustness of encoders.

#### 1. Gaussian Noise

Additive white Gaussian noise with severity-dependent standard deviation.

```python
def gaussian_noise(x, severity):
    """Add Gaussian noise to image tensor.

    Args:
        x: Image tensor, shape (C, H, W) or (B, C, H, W), range [0, 1]
        severity: Integer 1-5

    Returns:
        Corrupted tensor, clamped to [0, 1]
    """
    sigma = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
    noise = torch.randn_like(x) * sigma
    return (x + noise).clamp(0, 1)
```

#### 2. Shot Noise (Poisson Noise)

Photon counting noise, intensity-dependent. Brighter pixels have more noise.

```python
def shot_noise(x, severity):
    """Add Poisson (shot) noise.

    Poisson noise is signal-dependent: variance equals the signal level.
    We scale the signal and apply Poisson sampling.
    """
    lam = [60, 25, 12, 5, 3][severity - 1]
    noisy = torch.poisson(x * lam) / lam
    return noisy.clamp(0, 1)
```

#### 3. Impulse Noise (Salt and Pepper)

Random pixels are set to 0 or 1 with a given probability.

```python
def impulse_noise(x, severity):
    """Add salt-and-pepper noise.

    Randomly sets pixels to 0 (pepper) or 1 (salt).
    """
    prob = [0.03, 0.06, 0.09, 0.17, 0.27][severity - 1]
    mask = torch.rand_like(x)
    salt = (mask > 1 - prob / 2).float()
    pepper = (mask < prob / 2).float()
    corrupted = x * (1 - salt - pepper) + salt
    return corrupted.clamp(0, 1)
```

### Category 2: Blur

Blur corruptions reduce spatial frequency content. They test the model's ability to classify from coarse features.

#### 4. Defocus Blur

Simulates out-of-focus camera with a circular blur kernel.

```python
def defocus_blur(x, severity):
    """Apply defocus (disc) blur.

    Uses a circular averaging kernel.
    """
    radius = [3, 4, 6, 8, 10][severity - 1]
    kernel_size = 2 * radius + 1

    # Create circular kernel
    y_grid, x_grid = torch.meshgrid(
        torch.arange(kernel_size) - radius,
        torch.arange(kernel_size) - radius,
        indexing='ij'
    )
    kernel = ((x_grid**2 + y_grid**2) <= radius**2).float()
    kernel = kernel / kernel.sum()

    # Apply as depthwise convolution
    # ... (see implementation in template)
```

#### 5. Motion Blur

Simulates camera or subject motion with a directional blur kernel.

```python
def motion_blur(x, severity):
    """Apply motion blur with random angle.

    Creates a line kernel at a random angle and convolves.
    """
    kernel_size = [10, 15, 15, 15, 20][severity - 1]
    angle = torch.rand(1).item() * 360  # Random direction

    # Create motion kernel (line at given angle)
    kernel = create_motion_kernel(kernel_size, angle)
    kernel = kernel / kernel.sum()

    return apply_kernel(x, kernel)
```

#### 6. Zoom Blur

Simulates zooming during exposure. Multiple shifted copies of the image at different zoom levels are averaged.

```python
def zoom_blur(x, severity):
    """Apply zoom blur by averaging multiple zoom levels.

    Progressively zooms in and averages the results.
    """
    zoom_factors = [
        [1.0, 1.05, 1.1],
        [1.0, 1.05, 1.1, 1.15],
        [1.0, 1.05, 1.1, 1.15, 1.2],
        [1.0, 1.05, 1.1, 1.15, 1.2, 1.3],
        [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
    ][severity - 1]

    result = torch.zeros_like(x)
    for factor in zoom_factors:
        zoomed = zoom_center(x, factor)
        result += zoomed
    return (result / len(zoom_factors)).clamp(0, 1)
```

### Category 3: Weather

Weather corruptions simulate environmental conditions affecting image capture.

#### 7. Brightness

Increases image brightness uniformly.

```python
def brightness(x, severity):
    """Increase brightness by adding a constant.

    Simple additive brightness change.
    """
    delta = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    return (x + delta).clamp(0, 1)
```

#### 8. Contrast

Reduces contrast by pulling pixel values toward the mean.

```python
def contrast(x, severity):
    """Reduce contrast by interpolating toward mean.

    Lower severity = more contrast reduction.
    """
    factor = [0.4, 0.3, 0.2, 0.15, 0.1][severity - 1]
    mean = x.mean()
    return ((x - mean) * factor + mean).clamp(0, 1)
```

#### 9. Fog

Simulates fog by blending the image with a white overlay, optionally with spatial variation.

```python
def fog(x, severity):
    """Simulate fog with white overlay.

    Blends image with white, with optional spatial gradient.
    """
    blend = [0.15, 0.3, 0.45, 0.6, 0.75][severity - 1]
    white = torch.ones_like(x)
    return ((1 - blend) * x + blend * white).clamp(0, 1)
```

#### 10. Snow

Simulates snow particles on the image.

```python
def snow(x, severity):
    """Simulate snow by adding bright spots.

    Adds random bright dots and slight brightness increase.
    """
    intensity = [0.1, 0.2, 0.3, 0.4, 0.55][severity - 1]
    # Random snow flake positions
    snow_mask = (torch.rand_like(x) > (1 - intensity * 0.1)).float()
    brightness_boost = intensity * 0.5
    return (x + snow_mask * 0.8 + brightness_boost).clamp(0, 1)
```

#### 11. Frost

Simulates frost patterns on the lens.

```python
def frost(x, severity):
    """Simulate frost with crystalline overlay pattern.

    Creates a semi-random frost-like pattern and blends with the image.
    """
    opacity = [0.1, 0.2, 0.35, 0.5, 0.65][severity - 1]
    # Create frost-like pattern using low-frequency noise
    frost_pattern = create_frost_pattern(x.shape[-2], x.shape[-1])
    return ((1 - opacity) * x + opacity * frost_pattern).clamp(0, 1)
```

### Category 4: Digital

Digital corruptions simulate image processing artifacts.

#### 12. Elastic Transform

Applies random elastic deformation to the image, simulating physical distortion.

```python
def elastic_transform(x, severity):
    """Apply elastic deformation.

    Creates a random displacement field, smooths it with Gaussian,
    and applies it to the image.
    """
    alpha = [50, 100, 150, 200, 250][severity - 1]
    sigma = [3, 4, 5, 6, 7][severity - 1]

    # Random displacement field
    dx = torch.randn(1, x.shape[-2], x.shape[-1]) * alpha
    dy = torch.randn(1, x.shape[-2], x.shape[-1]) * alpha

    # Smooth with Gaussian kernel
    dx = gaussian_smooth(dx, sigma)
    dy = gaussian_smooth(dy, sigma)

    # Apply displacement via grid_sample
    return apply_displacement(x, dx, dy)
```

#### 13. Pixelate

Reduces resolution and upscales back, creating blocky artifacts.

```python
def pixelate(x, severity):
    """Pixelate by downsampling and upsampling.

    Reduces resolution to create block artifacts.
    """
    factor = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
    h, w = x.shape[-2], x.shape[-1]
    small_h, small_w = max(1, int(h * factor)), max(1, int(w * factor))

    # Downsample
    small = F.interpolate(x.unsqueeze(0), size=(small_h, small_w), mode='nearest')
    # Upsample back
    return F.interpolate(small, size=(h, w), mode='nearest').squeeze(0)
```

#### 14. JPEG Compression

Simulates lossy JPEG compression artifacts.

```python
def jpeg_compression(x, severity):
    """Simulate JPEG compression by quantizing DCT coefficients.

    Applies block-wise DCT, quantizes, and reconstructs.
    For a tensor-only approach, approximate with Gaussian blur + noise.
    """
    quality = [25, 18, 15, 10, 7][severity - 1]
    # Approximate JPEG artifacts with blur + slight noise
    blur_sigma = (100 - quality) / 100.0 * 1.5
    noise_sigma = (100 - quality) / 100.0 * 0.05

    blurred = gaussian_smooth(x, blur_sigma)
    noisy = blurred + torch.randn_like(blurred) * noise_sigma
    return noisy.clamp(0, 1)
```

#### 15. Glass Blur

Simulates looking through frosted glass by randomly shuffling nearby pixels and blurring.

```python
def glass_blur(x, severity):
    """Apply glass blur (shuffle + blur).

    Randomly displaces pixels within a neighborhood, then blurs.
    """
    sigma = [0.7, 0.9, 1.0, 1.1, 1.5][severity - 1]
    displacement = [1, 1, 2, 2, 3][severity - 1]

    # Random pixel displacement
    corrupted = random_pixel_shuffle(x, displacement)
    # Gaussian blur
    return gaussian_smooth(corrupted, sigma)
```

---

## Mean Corruption Error (MCE)

### Definition

MCE measures the average error increase due to corruptions, normalized against a baseline model:

```
MCE = (1/C) * sum_c CE_model(c) / CE_baseline(c)
```

Where:
- `C` is the number of corruption types
- `CE_model(c)` is the corruption error for corruption type `c`, averaged over severities
- `CE_baseline(c)` is the corruption error of a baseline model (typically AlexNet for ImageNet-C)

### Computation

```python
def compute_mce(model_errors, baseline_errors):
    """
    Compute Mean Corruption Error.

    Args:
        model_errors: Dict mapping corruption_name -> List[float] of errors at each severity
        baseline_errors: Dict mapping corruption_name -> List[float] for baseline model

    Returns:
        MCE value (lower is better)
    """
    mce_sum = 0.0
    n_corruptions = 0

    for corruption in model_errors:
        if corruption in baseline_errors:
            # Average error across severities
            model_avg = sum(model_errors[corruption]) / len(model_errors[corruption])
            baseline_avg = sum(baseline_errors[corruption]) / len(baseline_errors[corruption])

            if baseline_avg > 0:
                mce_sum += model_avg / baseline_avg
                n_corruptions += 1

    return mce_sum / max(n_corruptions, 1)
```

### Absolute MCE (without baseline)

When no baseline model is available, report absolute corruption error:

```python
def compute_absolute_mce(model_errors):
    """
    Compute absolute Mean Corruption Error (no baseline normalization).

    Args:
        model_errors: Dict mapping corruption_name -> List[float] of errors at each severity

    Returns:
        Absolute MCE (average error across all corruptions and severities)
    """
    total_error = 0.0
    total_count = 0

    for corruption, errors in model_errors.items():
        for error in errors:
            total_error += error
            total_count += 1

    return total_error / max(total_count, 1)
```

### Relative MCE

Compares model MCE against a reference model (not necessarily a baseline):

```
Relative MCE = MCE_model / MCE_reference
```

- **< 1.0**: Model is more robust than reference
- **= 1.0**: Same robustness
- **> 1.0**: Model is less robust

---

## Severity Levels

Each corruption type has 5 severity levels, ranging from barely noticeable (1) to severe (5).

### Calibration Principles

Severity levels are calibrated to be:
1. **Monotonically increasing in difficulty**: Higher severity should always reduce accuracy
2. **Perceptible but realistic**: All severities should represent plausible real-world degradation
3. **Spanning the useful range**: Severity 1 should barely affect a good model; severity 5 should significantly degrade performance

### Monotonicity Check

A key validation: accuracy should monotonically decrease with severity.

```python
def check_monotonicity(errors_by_severity):
    """
    Check that corruption error increases monotonically with severity.

    Args:
        errors_by_severity: List of errors at severities [1, 2, 3, 4, 5]

    Returns:
        True if monotonically non-decreasing
    """
    for i in range(1, len(errors_by_severity)):
        if errors_by_severity[i] < errors_by_severity[i-1] - 0.01:  # Allow small tolerance
            return False
    return True
```

Non-monotonic results may indicate:
- The corruption function is not well-calibrated
- The model has unexpected sensitivity patterns
- Stochastic noise in evaluation (run multiple times)

---

## Benchmarking Protocol

### Standard Evaluation

1. **Evaluate clean accuracy** on the uncorrupted test set
2. **For each corruption type** (15 total):
   a. For each severity level (1-5):
      - Apply corruption to all test images
      - Evaluate model accuracy
      - Record error rate (1 - accuracy)
3. **Compute MCE** averaged across corruptions and severities
4. **Check monotonicity** for each corruption type

### Reporting Format

```
Corruption Benchmark Results
Model: brain_ai (vision, minimal config)
Dataset: MNIST (10,000 test images)

Clean accuracy: 98.5%

Corruption         | Sev 1 | Sev 2 | Sev 3 | Sev 4 | Sev 5 | Avg
--------------------|-------|-------|-------|-------|-------|------
gaussian_noise      | 96.2% | 93.1% | 88.5% | 81.2% | 72.3% | 86.3%
shot_noise          | 96.8% | 94.2% | 90.1% | 84.5% | 76.8% | 88.5%
impulse_noise       | 95.9% | 92.8% | 87.2% | 78.9% | 68.5% | 84.7%
defocus_blur        | 97.1% | 95.3% | 91.8% | 86.7% | 80.2% | 90.2%
motion_blur         | 96.5% | 94.0% | 89.5% | 83.1% | 75.5% | 87.7%
zoom_blur           | 96.0% | 93.5% | 88.8% | 82.0% | 73.8% | 86.8%
brightness          | 98.0% | 97.2% | 95.8% | 93.5% | 90.1% | 94.9%
contrast            | 97.5% | 96.0% | 93.2% | 88.5% | 82.0% | 91.4%
fog                 | 97.8% | 96.5% | 94.2% | 90.8% | 85.5% | 93.0%
snow                | 97.2% | 95.1% | 91.5% | 86.2% | 79.0% | 89.8%
elastic_transform   | 97.0% | 94.8% | 90.2% | 84.0% | 76.5% | 88.5%
pixelate            | 97.5% | 95.8% | 92.5% | 87.0% | 79.8% | 90.5%
jpeg_compression    | 97.8% | 96.2% | 93.8% | 89.5% | 83.2% | 92.1%
glass_blur          | 96.3% | 93.8% | 89.0% | 82.5% | 74.2% | 87.2%
frost               | 97.0% | 95.0% | 91.0% | 85.5% | 78.0% | 89.3%

Absolute MCE: 0.106
Monotonicity check: 15/15 passed
```

---

## Brain_ai-Specific Considerations

### SNN Noise Tolerance

The SNN core's temporal coding provides natural noise tolerance because:
- Spike timing is robust to small amplitude perturbations
- Membrane potential integration acts as a low-pass filter
- Multiple timesteps provide temporal averaging

This means noise corruptions (Gaussian, shot, impulse) may affect brain_ai less than conventional DNNs.

### HTM Sparse Representations

HTM's sparse distributed representations (SDR) are inherently robust to corruption:
- A small number of corrupted bits in a large SDR does not significantly change the pattern
- The overlap-based matching is tolerant to partial corruption
- This benefit is strongest for blur and digital corruptions

### Workspace Competition

The global workspace may amplify or attenuate corruption effects:
- If corruption degrades one modality, other modalities can compensate via workspace competition
- However, if the corrupted modality normally dominates, workspace behavior can change significantly

### Implications for Evaluation

- Report corruption results per-modality to understand which pipeline stages are most affected
- Compare brain_ai MCE against a standard CNN baseline to quantify the architectural robustness benefits
- Use corruption benchmarks alongside adversarial attacks for comprehensive robustness assessment

---

## Summary

| Category | Corruptions | Primary Effect | brain_ai Sensitivity |
|----------|------------|----------------|---------------------|
| Noise | gaussian, shot, impulse | Pixel-level noise | Low (SNN temporal averaging) |
| Blur | defocus, motion, zoom, glass | Spatial frequency loss | Medium (depends on encoder) |
| Weather | brightness, contrast, fog, snow, frost | Global appearance change | Low-Medium (workspace compensation) |
| Digital | elastic, pixelate, jpeg | Structural artifacts | Medium (encoder-dependent) |

The corruption benchmark is a necessary complement to adversarial robustness testing. While adversarial attacks measure worst-case robustness, corruption benchmarks measure average-case robustness to realistic conditions. A robust brain_ai system should perform well on both.
