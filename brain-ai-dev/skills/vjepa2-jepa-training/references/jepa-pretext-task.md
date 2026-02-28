# JEPA Pretext Task Reference

## Architecture Overview

The Joint Embedding Predictive Architecture (JEPA) is a self-supervised learning framework that trains a model to predict masked video patch representations entirely in latent space — never reconstructing raw pixels. This design forces the model to learn semantically meaningful features rather than low-level texture statistics.

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                  JEPA Training Loop                     │
                    └─────────────────────────────────────────────────────────┘

  Input Video (B, C, T, H, W)
           │
           ├─────────────────────────────────────────────────────┐
           │                                                     │
           ▼ visible patches only                               ▼ full video (all patches)
  ┌──────────────────────┐                          ┌──────────────────────────┐
  │   Context Encoder    │                          │    Target Encoder (EMA)  │
  │  (ViT, trainable)    │                          │  (frozen, EMA of context)│
  └──────────┬───────────┘                          └──────────────┬───────────┘
             │ context_repr                                        │ target_repr
             │ [B, N_vis, D]                                       │ [B, N_all, D]
             ▼                                                     │
  ┌──────────────────────┐                                         │
  │  VisionTransformer   │◄─── mask_tokens [B, N_pred, D]         │
  │      Predictor       │                                         │
  │  (trainable)         │                                         │
  └──────────┬───────────┘                                         │
             │ pred_repr                                           │
             │ [B, N_pred, D]                                      │ target_repr[masked positions]
             ▼                                                     ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │              Smooth L1 Loss (masked patches only)                    │
  │           loss = smooth_l1(pred_repr, target_repr[masks_pred])       │
  └──────────────────────────────────────────────────────────────────────┘
```

### Detailed Data Flow

**Step 1 — Patchification**
- Input: `(B, C, T, H, W)` — batch, channels, time, height, width
- Patch grid: `(T/t_patch) x (H/patch_size) x (W/patch_size)` patches total
- Each patch becomes a `D`-dimensional token after linear projection + positional embedding

**Step 2 — Masking**
- Two complementary mask sets are sampled per batch:
  - `masks_enc`: indices of visible (unmasked) patches — fed to context encoder
  - `masks_pred`: indices of masked (target) patches — what the predictor must reconstruct
- Masking ratio typically 80–90% of all patches
- Masks are block-structured (spatial or spatiotemporal blocks) to force coherent prediction

**Step 3 — Context Encoding**
```python
# Only visible patches are passed through context encoder
visible_patches = gather(patches, masks_enc)   # [B, N_vis, D]
context_repr = encoder(visible_patches)        # [B, N_vis, D]
```

**Step 4 — Target Computation**
```python
# Target encoder processes ALL patches (no masking)
with torch.no_grad():
    target_repr = target_encoder(all_patches)  # [B, N_all, D]
    target_repr = gather(target_repr, masks_pred)  # [B, N_pred, D]
    if normalize_reps:
        target_repr = F.layer_norm(target_repr, [target_repr.size(-1)])
```

**Step 5 — Prediction**
```python
pred_repr = predictor(context_repr, masks_enc, masks_pred)  # [B, N_pred, D]
```

**Step 6 — Loss**
```python
loss = smooth_l1_loss(pred_repr, target_repr, loss_exp)
```

---

## Loss Computation

### Smooth L1 Loss (Huber Loss Variant)

The loss is computed **only on the masked (target) patch positions**. Visible patches are never included in the loss gradient.

```python
def smooth_l1_loss(pred: Tensor, target: Tensor, beta: float = 1.0,
                   loss_exp: float = 1.0) -> Tensor:
    """
    Smooth L1 loss applied to predicted masked patch representations.

    Args:
        pred:     [B, N_pred, D] — predictor output for masked positions
        target:   [B, N_pred, D] — EMA target encoder output for masked positions
        beta:     threshold for switching between L1 and L2 regime (default 1.0)
        loss_exp: exponent for raising the per-token loss (default 1.0 = standard Huber)

    Returns:
        Scalar loss value
    """
    diff = pred - target  # [B, N_pred, D]
    abs_diff = diff.abs()

    # Smooth L1 kernel: quadratic below beta, linear above
    loss = torch.where(
        abs_diff < beta,
        0.5 * diff.pow(2) / beta,
        abs_diff - 0.5 * beta
    )  # [B, N_pred, D]

    # Optional: raise per-token loss to exponent (sharpens gradient signal)
    if loss_exp != 1.0:
        loss = loss.pow(loss_exp)

    return loss.mean()
```

### Why Smooth L1 Instead of MSE?

| Criterion | MSE | Smooth L1 |
|-----------|-----|-----------|
| Outlier sensitivity | High (squares errors) | Low (linear for large errors) |
| Training stability | Can diverge on outliers | Robust |
| Gradient magnitude | Grows with error | Bounded for large errors |
| V-JEPA use | Not used | Default |

MSE squares the per-token error, making training sensitive to outlier patches (e.g., very bright or dark regions). Smooth L1 provides more stable gradients throughout training.

### Loss with Normalization (DROID Mode)

```python
def normalized_smooth_l1_loss(pred: Tensor, target: Tensor,
                               loss_exp: float = 1.0) -> Tensor:
    """Normalize representations before computing loss (DROID fine-tuning)."""
    D = pred.size(-1)
    pred_norm   = F.layer_norm(pred,   [D])
    target_norm = F.layer_norm(target, [D])
    return smooth_l1_loss(pred_norm, target_norm, loss_exp=loss_exp)
```

---

## Collapse Prevention Theory

Representation collapse is the central failure mode of self-supervised learning: the encoder learns to output identical (or near-identical) representations for all inputs, making the loss trivially zero. JEPA prevents collapse through **structural constraints** rather than explicit contrastive terms.

### The Four Pillars of JEPA Collapse Prevention

#### 1. Asymmetric Architecture

The context encoder and target encoder are **different** in their training dynamics:
- Context encoder: updated via backpropagation every step
- Target encoder: updated via exponential moving average (never gradients)

This asymmetry breaks the symmetry that would cause collapse. If both encoders were gradient-trained, the trivial solution (all-zeros output) would minimize the loss. With EMA, the target encoder is a lagged, smoothed version of the context encoder — the encoder must actively track a moving target rather than co-adapting.

```
Without asymmetry:
  encoder_A, encoder_B both trained with grad → trivial solution possible

With EMA asymmetry:
  encoder (grad) must predict ema_encoder (no grad, slowly moving) → non-trivial
```

#### 2. EMA Target (Slowly Moving Targets)

The momentum-based update gives the target encoder a longer "memory":
```
theta_target = m * theta_target + (1-m) * theta_encoder
```

With `m = 0.99925`, the effective window is `1/(1-m) ≈ 1333 steps`. This means:
- The target encodes information across ~1333 recent encoder states
- The encoder must predict future states of itself, not current states
- Trivial shortcut (copy encoder output) fails because target lags behind

#### 3. High Masking Ratio

Masking 80–90% of patches forces the predictor to synthesize predictions from sparse context. This has two effects:
- **Context is insufficient** for memorization — the predictor cannot store all patch values
- **Prediction requires generalization** — only representations that capture structure can support high-quality prediction under heavy masking

Lower masking ratios (< 50%) allow the predictor to copy patches using local interpolation, weakening the learning signal.

#### 4. No Negative Samples

Unlike contrastive methods (SimCLR, MoCo), JEPA does not require negative pairs. This is possible because:
- The prediction task in latent space is asymmetric
- The masking creates a natural information bottleneck
- EMA prevents trivial solutions

Benefits over contrastive methods:
- No large batch requirement (contrastive needs 4096+ samples for diverse negatives)
- No momentum queue needed for negatives
- No careful negative mining strategy
- Simpler implementation, better scaling

### Comparison to Contrastive Methods

| Property | SimCLR / MoCo | DINO | V-JEPA 2 |
|----------|--------------|------|----------|
| Negatives required | Yes | No (but centering) | No |
| Prediction target | Augmented view | Augmented view | Masked latent |
| Collapse prevention | Negative pairs | Centering + sharpening | Asymmetry + EMA + masking |
| Prediction space | Embedding space | Embedding space | Latent (post-encoder) |
| Video-native | No | No | Yes |
| Pixel reconstruction | No | No | No |
| Memory footprint | High (queue) | Medium | Low |

### Monitoring Collapse

During training, watch these metrics:
```python
# Feature variance across batch — should stay >> 0
feature_var = context_repr.var(dim=0).mean().item()

# Cosine similarity between pairs — should be < 0.95
cos_sim = F.cosine_similarity(
    context_repr[::2], context_repr[1::2], dim=-1
).mean().item()

# Predictor loss — if it goes to 0, check EMA
loss_value = loss.item()
```

Warning signs:
- `feature_var < 0.01`: representations collapsing
- `cos_sim > 0.98`: encoder outputting near-identical features
- `loss < 1e-6` in first 10 epochs: likely collapse, not genuine learning

---

## JEPA vs Pixel Reconstruction

JEPA predicts **latent representations**, not raw pixels. This distinction is critical:

**Pixel reconstruction (MAE, VideoMAE)**:
- Target: raw pixel values at masked positions
- Must model all image details including texture, lighting noise
- Computational burden of decoding to high-resolution space
- Encourages low-level feature learning

**Latent prediction (JEPA)**:
- Target: abstract representation from target encoder
- Target encoder filters out irrelevant details
- Prediction happens in compact latent space
- Encourages high-level semantic feature learning

The target encoder acts as a learned low-pass filter — it discards high-frequency pixel details and retains structure and semantics. This is why V-JEPA 2 requires fewer labeled samples for downstream tasks compared to pixel-reconstruction methods.

---

## Implementation Checklist

```python
# Required components
assert encoder is not None           # Context encoder (ViT)
assert target_encoder is not None    # EMA copy of encoder
assert predictor is not None         # VisionTransformerPredictor
assert masks_enc is not None         # Visible patch indices
assert masks_pred is not None        # Masked patch indices

# Required properties
assert not any(p.requires_grad for p in target_encoder.parameters())
assert masking_ratio > 0.5           # Should be 0.8+ for good features
assert ema_momentum > 0.99           # Should be 0.999+ for stable targets
assert loss_exp >= 1.0               # Standard or stronger loss

# Required training loop
for step, (batch, masks_enc, masks_pred) in enumerate(loader):
    loss_dict = trainer.train_step(batch, masks_enc, masks_pred)
    trainer.update_ema(step)
    lr_scheduler.step(step)
    wd_scheduler.step(step)
```
