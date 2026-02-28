# Progressive Training Reference

## Overview

V-JEPA 2 uses a three-stage progressive training strategy. Each stage builds on the previous, increasing spatial/temporal resolution and specializing toward the target domain.

```
Stage 1: Pretrain          Stage 2: Cooldown          Stage 3: DROID Post-Train
─────────────────────     ──────────────────────     ──────────────────────────
Resolution: 256px          Resolution: 384px           Resolution: 256px
Frames: 16                 Frames: 64                  Frames: 8
Duration: 300 epochs       Duration: 30 epochs         Duration: 100 epochs
LR: warmup + cosine        LR: linear decay            LR: warmup + cosine
EMA: yes                   EMA: yes                    EMA: no (direct loss)
Dataset: large scale       Dataset: same / larger      Dataset: DROID robotics
Encoder: trainable         Encoder: trainable          Encoder: frozen
```

---

## Stage 1: Pretrain (256px / 16 frames)

### Purpose
Learn general spatiotemporal representations from large-scale video data. The model learns to predict masked patches in latent space using the JEPA pretext task.

### Configuration
```python
pretrain_config = JEPATrainingConfig(
    # Resolution
    image_size=256,
    num_frames=16,
    patch_size=16,       # Gives 16x16 = 256 spatial patches per frame
    tubelet_size=2,      # Temporal patch size: 16/2 = 8 temporal positions

    # Predictor
    predictor_embed_dim=384,
    predictor_depth=12,
    predictor_num_heads=12,
    num_mask_tokens=10,

    # Training
    epochs=300,
    warmup_epochs=40,
    lr=1e-3,
    final_lr=1e-6,
    weight_decay=0.04,
    final_weight_decay=0.4,
    batch_size=64,
    use_bfloat16=True,

    # EMA
    ema_start=0.99925,
    ema_end=0.99925,

    # Loss
    loss_exp=1.0,
    normalize_reps=False,

    # No annealing
    is_anneal=False,
    auto_steps=0,
)
```

### Masking Strategy
- **Spatial block masking**: sample contiguous rectangular blocks
- **Temporal block masking**: extend blocks along time axis
- **Masking ratio**: 80-90% of all patches
- Multiple mask sets per sample (4-8 masks) for training efficiency

### LR Schedule (Pretrain)
```
Epoch 0 to 40:   Linear warmup from 0 to 1e-3
Epoch 40 to 300: Cosine decay from 1e-3 to 1e-6
```

### WD Schedule (Pretrain)
```
All epochs: Cosine from 0.04 to 0.4
```

### Checkpoint Structure (End of Pretrain)
```python
pretrain_checkpoint = {
    'epoch': 300,
    'encoder': encoder.state_dict(),
    'predictor': predictor.state_dict(),
    'target_encoder': ema_manager.target_encoder.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scaler': scaler.state_dict(),
    'lr': current_lr,       # Should be ~1e-6 at end
    'loss': final_loss,
    'config': pretrain_config.__dict__,
}
```

---

## Stage 2: Cooldown (384px / 64 frames)

### Purpose
Adapt the pretrained model to higher resolution and longer video clips. The LR is annealed to near-zero to prevent catastrophic forgetting while refining features.

### Key Differences from Pretrain
- **Higher resolution**: 384px instead of 256px, giving 4x more spatial patches
- **Longer clips**: 64 frames instead of 16, giving 4x more temporal positions
- **Annealing mode**: LR starts near the pretrain final LR and decays to near-zero
- **No warmup**: Start directly at a low LR (continuing from pretrain)

### Configuration
```python
cooldown_config = JEPATrainingConfig(
    image_size=384,
    num_frames=64,
    patch_size=16,    # 384/16 = 24 spatial patches per dim
    tubelet_size=2,

    epochs=30,
    warmup_epochs=0,           # No warmup -- continuing from pretrain
    lr=1e-4,                   # Start at pretrain's final LR or slightly above
    final_lr=1e-7,             # Decay to near-zero

    weight_decay=0.4,          # Already at final WD from pretrain
    final_weight_decay=0.4,    # No further increase

    batch_size=16,             # Smaller batch due to higher resolution memory
    use_bfloat16=True,

    ema_start=0.99925,
    ema_end=0.99925,
    loss_exp=1.0,
    normalize_reps=False,

    is_anneal=True,                           # Enable annealing mode
    anneal_ckpt='pretrain_epoch300.pth',      # Load from pretrain
    auto_steps=0,
)
```

### Checkpoint Loading for Cooldown

```python
def load_pretrain_for_cooldown(pretrain_ckpt_path, encoder, predictor,
                                ema_manager, optimizer, scaler,
                                cooldown_config):
    """
    Load pretrain checkpoint for cooldown phase.
    Handles architecture changes due to higher resolution.
    """
    import torch

    ckpt = torch.load(pretrain_ckpt_path, map_location='cpu')

    # Load encoder -- may need strict=False for RoPE position bias changes
    missing, unexpected = encoder.load_state_dict(
        ckpt['encoder'], strict=False
    )
    if missing:
        print(f"Missing encoder keys (expected for resolution change): {missing[:5]}")

    # Load predictor -- usually strict=True OK
    predictor.load_state_dict(ckpt['predictor'], strict=True)

    # Load target encoder (EMA copy)
    ema_manager.target_encoder.load_state_dict(
        ckpt['target_encoder'], strict=False
    )

    # Load optimizer state -- LR will be overridden by schedule
    try:
        optimizer.load_state_dict(ckpt['optimizer'])
        # Override LR to cooldown starting LR
        for pg in optimizer.param_groups:
            pg['lr'] = cooldown_config.lr
    except Exception as exc:
        print(f"Optimizer state not loaded (will restart): {exc}")

    # Load AMP scaler
    if 'scaler' in ckpt and scaler is not None:
        scaler.load_state_dict(ckpt['scaler'])

    return ckpt


def strip_module_prefix(state_dict):
    """Remove DDP 'module.' prefix from state dict keys."""
    return {
        k.replace('module.', '', 1) if k.startswith('module.') else k: v
        for k, v in state_dict.items()
    }
```

### LR Schedule (Cooldown)
```
Cooldown uses Warmup + Stable + Decay or Linear Decay:

Option A (is_anneal=True, no warmup):
  Step 0 to T_max: Linear decay from lr to final_lr

Option B (short warmup then decay):
  Step 0 to warmup_steps: Warmup from lr/10 to lr
  Step warmup to stable_step: Stable at lr
  Step stable to T_max: Linear decay from lr to final_lr
```

### Memory Management at High Resolution

384px x 64 frames significantly increases memory requirements. Strategies:

```python
# 1. Gradient checkpointing for encoder
encoder.gradient_checkpointing = True

# 2. Smaller batch size (compensated by gradient accumulation)
accumulation_steps = 4  # effective_batch = batch_size * accumulation_steps

# 3. BFloat16 (essential at high resolution)
use_bfloat16 = True

# 4. Efficient attention (SDPA or Flash Attention)
# Automatic in PyTorch 2.0+ via scaled_dot_product_attention
```

---

## Stage 3: DROID Post-Training (256px / 8 frames)

### Purpose
Specialize the video encoder for robotic manipulation tasks using the DROID dataset. The encoder is frozen; a deep frame-causal predictor is trained for autoregressive multi-step prediction.

### Key Differences from Pretraining

| Property | Pretrain | DROID |
|----------|---------|-------|
| Encoder | Trainable | Frozen |
| Predictor | Standard (bidirectional) | Frame-causal (block-causal mask) |
| Target | EMA encoder output | Direct observation |
| Loss | Smooth L1 | Normalized Smooth L1 |
| Auto-steps | 0 (single step) | 1-4 (multi-step rollout) |
| LR | High (1e-3) | Lower (5e-4) |
| EMA | Yes | No |

### Configuration
```python
droid_config = JEPATrainingConfig(
    image_size=256,
    num_frames=8,
    patch_size=16,
    tubelet_size=2,

    epochs=100,
    warmup_epochs=5,
    lr=5e-4,
    final_lr=5e-6,
    weight_decay=0.05,
    final_weight_decay=0.2,
    batch_size=32,
    use_bfloat16=True,

    # EMA disabled for DROID
    ema_start=0.99925,
    ema_end=0.99925,

    # Normalized loss for DROID
    loss_exp=1.0,
    normalize_reps=True,     # Normalize before loss

    is_anneal=False,
    anneal_ckpt='cooldown_epoch30.pth',  # Load from cooldown

    # Autoregressive rollout
    auto_steps=2,            # Predict 2 steps into future
    encoder_lr_scale=0.0,    # Frozen encoder (no gradient)
)
```

### Frozen Encoder Setup

```python
def freeze_encoder(encoder):
    """Freeze all encoder parameters for DROID fine-tuning."""
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()  # Also set to eval mode (no dropout, no BN stats update)
    print(f"Encoder frozen: {sum(p.numel() for p in encoder.parameters())} params")


def verify_frozen(encoder):
    """Verify no encoder parameters have requires_grad=True."""
    return not any(p.requires_grad for p in encoder.parameters())
```

### Autoregressive Multi-Step Prediction

```python
def autoregressive_droid_step(encoder, predictor, frames,
                               masks_enc, masks_pred,
                               auto_steps=2, normalize_reps=True):
    """
    Autoregressive rollout: predict future frames from current observations.

    The predictor outputs at step t are fed back as the context for step t+1.
    This tests the quality of the learned representation by chaining predictions.

    Args:
        encoder:       Frozen context encoder
        predictor:     Trainable frame-causal predictor
        frames:        Input video batch [B, C, T, H, W]
        masks_enc:     Visible patch masks
        masks_pred:    Target patch masks
        auto_steps:    Number of autoregressive steps
        normalize_reps: Whether to normalize representations before loss

    Returns:
        Scalar loss value averaged across steps
    """
    import torch
    import torch.nn.functional as F

    total_loss = 0.0

    # Initial encoding (frozen, no grad)
    with torch.no_grad():
        visible_patches = gather_patches(frames, masks_enc)
        context = encoder(visible_patches)  # [B, N_vis, D]

    for step_idx in range(auto_steps):
        # Predict next representation
        pred = predictor(context, masks_enc, masks_pred)  # [B, N_pred, D]

        # Get ground-truth target (from encoder on all patches)
        with torch.no_grad():
            target_full = encoder(frames)                      # [B, N_all, D]
            target = gather_patches(target_full, masks_pred)   # [B, N_pred, D]

        if normalize_reps:
            D = pred.size(-1)
            pred   = F.layer_norm(pred,   [D])
            target = F.layer_norm(target, [D])

        step_loss = smooth_l1_loss(pred, target)
        total_loss = total_loss + step_loss

        # Use prediction as next context (autoregressive)
        context = pred.detach()  # Stop gradient for autoregressive step

    return total_loss / auto_steps
```

### Frame-Causal Attention Mask

For robotics tasks, future frames should not attend to past frames (causal constraint):

```python
def make_causal_mask(seq_len, num_frames, patches_per_frame):
    """
    Block-causal attention mask for frame-level causality.

    Frame i can attend to frames 0..i but not i+1..T-1.
    Within a frame, all patches attend to each other.

    Returns:
        [seq_len, seq_len] boolean mask (True = attend, False = block)
    """
    import torch
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    for frame_i in range(num_frames):
        start_i = frame_i * patches_per_frame
        end_i   = start_i + patches_per_frame
        # Frame i attends to all patches in frames 0..i
        causal_end = end_i
        mask[start_i:end_i, :causal_end] = True
    return mask
```

---

## Checkpoint Handling Between Stages

### Checkpoint Schema

```python
CHECKPOINT_KEYS = [
    'epoch',           # int: completed epochs
    'step',            # int: total steps completed
    'encoder',         # dict: encoder state_dict
    'predictor',       # dict: predictor state_dict
    'target_encoder',  # dict: EMA target encoder state_dict
    'optimizer',       # dict: optimizer state_dict
    'scaler',          # dict: GradScaler state_dict (AMP)
    'lr',              # float: last LR (for anneal continuation)
    'loss',            # float: last loss value
    'config',          # dict: training config snapshot
    'stage',           # str: 'pretrain' | 'cooldown' | 'droid'
]
```

### Save Checkpoint

```python
def save_checkpoint(path, epoch, step, encoder, predictor, target_encoder,
                    optimizer, scaler, current_lr, current_loss, config,
                    stage):
    """Save complete training state for resumption."""
    import torch

    checkpoint = {
        'epoch': epoch,
        'step': step,
        'encoder': encoder.state_dict(),
        'predictor': predictor.state_dict(),
        'target_encoder': target_encoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict() if scaler is not None else {},
        'lr': current_lr,
        'loss': current_loss,
        'config': config.__dict__ if hasattr(config, '__dict__') else config,
        'stage': stage,
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path} (epoch={epoch}, step={step})")
```

### Load Checkpoint

```python
def load_checkpoint(path, encoder, predictor, target_encoder,
                    optimizer=None, scaler=None, strict=True):
    """
    Load checkpoint, handling DDP prefix and optional strict loading.

    Returns:
        Checkpoint dict with 'epoch', 'step', 'lr', 'loss', etc.
    """
    import torch

    ckpt = torch.load(path, map_location='cpu')

    def _load_module(module, key):
        sd = ckpt.get(key, {})
        # Strip DDP 'module.' prefix if present
        sd_clean = {k.replace('module.', '', 1): v for k, v in sd.items()}
        missing, unexpected = module.load_state_dict(sd_clean, strict=strict)
        if missing and strict:
            raise RuntimeError(f"Missing keys in {key}: {missing[:10]}")
        return missing, unexpected

    _load_module(encoder, 'encoder')
    _load_module(predictor, 'predictor')
    _load_module(target_encoder, 'target_encoder')

    if optimizer is not None and 'optimizer' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['optimizer'])
        except Exception as exc:
            print(f"Warning: could not load optimizer state: {exc}")

    if scaler is not None and 'scaler' in ckpt and ckpt['scaler']:
        scaler.load_state_dict(ckpt['scaler'])

    return ckpt
```

### Inter-Stage Compatibility

Potential issues when loading across stages:

| Issue | Cause | Fix |
|-------|-------|-----|
| size mismatch for pos_embed | Resolution changed, more patches | Use strict=False |
| unexpected key: rope_freqs | RoPE vs sinusoidal position encoding mismatch | strict=False, ignore missing |
| optimizer state size mismatch | Different param group structure | Do not load optimizer state |
| module. prefix | DDP training wrapped model | Strip module. prefix |
| missing key: predictor.blocks.N.* | Deeper predictor in DROID | strict=False |

---

## Full Pipeline Execution

```bash
# Stage 1: Pretrain
python train.py \
    --stage pretrain \
    --image-size 256 \
    --num-frames 16 \
    --epochs 300 \
    --lr 1e-3 \
    --output-dir ./checkpoints/pretrain

# Stage 2: Cooldown
python train.py \
    --stage cooldown \
    --image-size 384 \
    --num-frames 64 \
    --epochs 30 \
    --lr 1e-4 \
    --anneal-ckpt ./checkpoints/pretrain/epoch_300.pth \
    --is-anneal \
    --output-dir ./checkpoints/cooldown

# Stage 3: DROID
python train.py \
    --stage droid \
    --image-size 256 \
    --num-frames 8 \
    --epochs 100 \
    --lr 5e-4 \
    --anneal-ckpt ./checkpoints/cooldown/epoch_30.pth \
    --auto-steps 2 \
    --normalize-reps \
    --freeze-encoder \
    --output-dir ./checkpoints/droid
```

---

## Monitoring and Early Stopping

### Key Metrics to Track

```python
metrics = {
    'train/loss':         loss.item(),
    'train/lr':           current_lr,
    'train/wd':           current_wd,
    'train/ema_momentum': current_momentum,
    'train/grad_norm':    grad_norm,
    'train/feature_var':  context_repr.var(dim=0).mean().item(),  # Collapse indicator
    'train/step_time_ms': step_time * 1000,
}
```

### Stage Transition Criteria

| Stage End Condition | Check |
|--------------------|-------|
| Pretrain complete | epoch >= 300 |
| Cooldown complete | epoch >= 30 AND lr < 1e-7 |
| DROID complete | epoch >= 100 AND loss < 0.05 |

### Resumption from Interrupted Training

```python
import os

if os.path.exists(resume_ckpt):
    ckpt = load_checkpoint(resume_ckpt, encoder, predictor,
                           ema_manager.target_encoder,
                           optimizer, scaler, strict=False)
    start_epoch = ckpt['epoch'] + 1
    start_step  = ckpt.get('step', start_epoch * steps_per_epoch)
    print(f"Resuming from epoch {start_epoch}, step {start_step}")
else:
    start_epoch = 0
    start_step  = 0
```
