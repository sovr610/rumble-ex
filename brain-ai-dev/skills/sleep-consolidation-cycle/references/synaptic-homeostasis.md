# Synaptic Homeostasis Reference

## Table of Contents

1. [Overview](#1-overview)
2. [Tononi-Cirelli SHY Theory](#2-tononi-cirelli-shy-theory)
3. [Weight Downscaling Mathematics](#3-weight-downscaling-mathematics)
4. [Capacity Restoration Metrics](#4-capacity-restoration-metrics)
5. [Global Downscaling Strategy](#5-global-downscaling-strategy)
6. [Selective Downscaling Strategy](#6-selective-downscaling-strategy)
7. [Layerwise Downscaling Strategy](#7-layerwise-downscaling-strategy)
8. [Numerical Stability](#8-numerical-stability)
9. [Appendix: Troubleshooting](#appendix-a-troubleshooting)

---

## 1. Overview

### Purpose

Synaptic homeostasis restores the network's capacity for new learning by globally reducing
weight magnitudes. This counteracts the tendency of gradient-based training to monotonically
increase weight norms, which reduces the signal-to-noise ratio for future learning and can
lead to saturation of nonlinearities.

### Biological Motivation

Giulio Tononi and Chiara Cirelli proposed the Synaptic Homeostasis Hypothesis (SHY) in 2003:
waking experience leads to a net increase in synaptic strength throughout the brain, and
sleep serves to renormalize these synapses back to a sustainable baseline. The key insight
is that the *relative* pattern of synaptic strengths (which encodes learned information) is
preserved, while the *absolute* scale is reduced.

---

## 2. Tononi-Cirelli SHY Theory

### Core Claims

1. **Waking potentiation**: during wakefulness, synaptic connections are strengthened via
   Hebbian and reward-modulated plasticity.
2. **Energetic cost**: maintaining potentiated synapses is metabolically expensive.
3. **Capacity saturation**: as synaptic weights grow, the network approaches saturation,
   reducing its ability to encode new information.
4. **Sleep downscaling**: during slow-wave sleep, synapses are globally downscaled,
   preserving the relative pattern while reducing absolute magnitudes.
5. **Capacity restoration**: after downscaling, the network can again efficiently encode
   new information.

### Experimental Support

- **Molecular**: expression of synaptic strength markers (GluA1-containing AMPA receptors)
  increases during waking and decreases during sleep.
- **Electrophysiological**: miniature excitatory postsynaptic currents (mEPSCs) are larger
  after waking and smaller after sleep.
- **Behavioral**: sleep deprivation impairs new learning; recovery sleep restores it.

### Computational Analogy

In deep networks:
- **Waking potentiation** = weight growth from gradient descent (weight norm increases).
- **Capacity saturation** = loss of effective learning rate as weights grow large.
- **Sleep downscaling** = multiplicative weight scaling (weight decay, but applied globally
  and only during a dedicated offline phase).

---

## 3. Weight Downscaling Mathematics

### Global Scaling

The simplest form: multiply all weights by a scalar factor s, where 0 < s < 1:

```
W_new = s * W_old
```

For bias terms, the scaling is typically not applied (biases represent threshold offsets,
not synaptic strengths):

```
b_new = b_old  (biases preserved)
```

### Relative Ratio Preservation

Global scaling exactly preserves relative weight ratios:

```
W_new[i,j] / W_new[k,l] = (s * W_old[i,j]) / (s * W_old[k,l]) = W_old[i,j] / W_old[k,l]
```

This is the critical property: the learned information (encoded in weight ratios) survives
downscaling.

### Computing the Optimal Scaling Factor

The target is to reduce the mean weight magnitude to a reference level (typically the
initialization scale). For a parameter tensor W:

```
s = target_norm / current_norm
```

Where:
- `current_norm = mean(|W|)` or `sqrt(mean(W^2))` (RMS)
- `target_norm` = expected norm at initialization (e.g., for Kaiming init with fan_in=n,
  target_rms = sqrt(2/n))

### Batch Normalization Interaction

For layers followed by batch normalization, weight scaling has no effect on the forward pass
(BatchNorm normalizes the output regardless of input scale). Downscaling these layers is
harmless but wasteful. The selective strategy can skip BN-preceding layers.

---

## 4. Capacity Restoration Metrics

### Mean Weight Magnitude (MWM)

```
MWM = (1/N) * sum_i |w_i|
```

Where N is the total number of parameters and w_i are individual weights. This metric
directly reflects the overall "energy" of the network.

### Weight Norm Ratio (WNR)

```
WNR = current_rms / init_rms
```

Ratio of current root-mean-square weight magnitude to initialization-time RMS. A WNR > 1.5
suggests the network may benefit from homeostasis; WNR returning to ~1.0 after downscaling
indicates successful capacity restoration.

### Effective Learning Capacity (ELC)

A more nuanced metric that considers the gradient-to-weight ratio:

```
ELC = mean(|grad_i| / (|w_i| + epsilon))
```

When weights grow large relative to gradients, ELC shrinks, indicating reduced learning
capacity. Homeostasis should restore ELC to near-initialization levels.

### Signal-to-Noise Ratio (SNR)

```
SNR = var(W) / mean(|W|)^2
```

Higher SNR means more differentiation between weights (stronger signal). Downscaling
preserves SNR because both numerator and denominator scale equally.

---

## 5. Global Downscaling Strategy

### Algorithm

```python
def global_downscale(model, factor):
    """Multiply all weight (non-bias) parameters by factor."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'bias' not in name and param.ndim >= 2:
                param.mul_(factor)
```

### Properties

- **Simplest**: one hyperparameter (factor).
- **Exact ratio preservation**: all weight ratios preserved exactly.
- **Fast**: O(N) where N = number of parameters.
- **Limitation**: treats all layers equally, which may not be optimal.

### Factor Selection Guidelines

| WNR Range | Suggested Factor | Rationale |
|---|---|---|
| 1.0 - 1.2 | 0.95 | Mild growth; gentle correction |
| 1.2 - 1.5 | 0.90 | Moderate growth; standard correction |
| 1.5 - 2.0 | 0.85 | Significant growth; default |
| 2.0+ | 0.80 | Substantial growth; aggressive correction |

### When to Use

- Networks without large layer-to-layer weight magnitude variation.
- Early training phases where all layers grow approximately uniformly.
- As a conservative default when per-layer statistics are unavailable.

---

## 6. Selective Downscaling Strategy

### Algorithm

Selective downscaling computes per-weight importance scores and protects high-importance
weights from aggressive downscaling:

```python
def selective_downscale(model, factor, importance, protect_threshold):
    """Downscale weights while protecting high-importance ones."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'bias' in name or param.ndim < 2:
                continue
            imp = importance[name]
            # Compute protection mask: top protect_threshold fraction protected
            threshold_val = torch.quantile(imp.flatten().float(), 1.0 - protect_threshold)
            protected = imp >= threshold_val

            # Protected weights get a milder scaling
            scale = torch.where(
                protected,
                torch.tensor(1.0 - (1.0 - factor) * 0.2),  # 20% of the downscaling
                torch.tensor(factor),
            )
            param.mul_(scale)
```

### Importance Metrics

Several importance metrics can be used:

#### Fisher Information Approximation

```
importance[i] = E[grad_i^2]
```

Estimated from the gradients accumulated during the last wake phase. Parameters with large
Fisher information are more important for the current task.

#### Magnitude-Based

```
importance[i] = |w_i|
```

Simple but effective: large weights are assumed to be more important. This has the
self-reinforcing property that homeostasis protects exactly those weights that have
grown the most.

#### Gradient x Weight (Sensitivity)

```
importance[i] = |w_i * grad_i|
```

Combines weight magnitude with gradient magnitude. Large values indicate parameters
that are both large and being actively used by the current training objective.

### Properties

- **Preserves critical pathways**: high-importance weights are protected.
- **More aggressive on noise**: low-importance weights (likely noise) are scaled down more.
- **Slightly breaks ratio preservation**: protected weights have different scaling than
  unprotected weights, so exact ratio preservation only holds within each class.
- **Requires importance computation**: adds overhead to compute importance scores.

---

## 7. Layerwise Downscaling Strategy

### Algorithm

Compute a per-layer scaling factor based on each layer's weight norm ratio:

```python
def layerwise_downscale(model, target_wnr=1.0):
    """Per-layer downscaling to bring each layer's WNR to target."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'bias' in name or param.ndim < 2:
                continue
            # Estimate initialization scale
            fan_in = param.shape[1] if param.ndim >= 2 else param.shape[0]
            init_rms = math.sqrt(2.0 / fan_in)  # Kaiming

            current_rms = param.data.float().pow(2).mean().sqrt().item()
            current_wnr = current_rms / init_rms

            if current_wnr > target_wnr:
                layer_factor = target_wnr / current_wnr
                param.mul_(layer_factor)
```

### Properties

- **Layer-adaptive**: layers that have grown more are scaled down more.
- **Preserves within-layer ratios**: all weights in a layer share the same factor.
- **Breaks cross-layer ratios**: different layers get different factors.
- **Initialization-aware**: uses initialization scale as the target, which is principled.

### When to Use

- Networks with heterogeneous layer sizes (e.g., early conv layers vs late dense layers).
- When some layers are known to grow faster than others (e.g., attention layers vs FFN).
- Production settings where layer-level granularity improves performance.

---

## 8. Numerical Stability

### Near-Zero Weight Handling

Weights very close to zero can become numerically zero after scaling, potentially causing
division-by-zero in subsequent operations. Apply an epsilon floor:

```python
param.data = torch.where(
    param.data.abs() < epsilon,
    torch.sign(param.data) * epsilon,
    param.data * factor,
)
```

### Half-Precision Safety

Downscaling should be performed in fp32 even when the model uses mixed precision:

```python
with torch.no_grad():
    original_dtype = param.dtype
    param_fp32 = param.float()
    param_fp32.mul_(factor)
    param.data = param_fp32.to(original_dtype)
```

### Gradient State Consistency

After modifying weights, optimizer state (momentum, adaptive learning rate accumulators)
may be stale. Options:

1. **Reset optimizer state**: safest but loses momentum information.
2. **Scale optimizer state**: scale momentum buffers by the same factor.
3. **Do nothing**: let the optimizer self-correct over subsequent steps (usually fine for
   mild downscaling).

Recommendation: for factors > 0.9, option 3 is sufficient. For more aggressive downscaling,
use option 2.

---

## Appendix A: Troubleshooting

| Issue | Cause | Resolution |
|---|---|---|
| Performance drops after homeostasis | Factor too aggressive | Increase factor (closer to 1.0); use selective strategy |
| No performance recovery after sleep | Factor too mild or network already at capacity | Decrease factor; check WNR before and after |
| NaN after downscaling | Near-zero weights or fp16 precision loss | Use fp32 for scaling; add epsilon floor |
| BatchNorm layers unaffected | Expected behavior | BN renormalizes; skip BN-adjacent layers for efficiency |
| Optimizer diverges after homeostasis | Stale momentum buffers | Scale optimizer state or reset optimizer |
| Weight norm doesn't decrease | Biases being counted in norm computation | Exclude biases from norm calculation |
