# Optimizer and Scheduling Reference

## AdamW Configuration

V-JEPA 2 uses AdamW with 4 parameter groups to apply weight decay selectively. This is the standard practice for transformer pretraining: weight decay on weight matrices, no weight decay on biases and 1D parameters (LayerNorm, embedding vectors).

### Why 4 Parameter Groups?

1. **Encoder weights** — 2D+ tensors (attention matrices, FFN weights) — apply weight decay
2. **Predictor weights** — 2D+ tensors — apply weight decay
3. **Encoder biases/1D** — biases, LayerNorm weights (1D) — NO weight decay
4. **Predictor biases/1D** — same as above — NO weight decay

Weight decay on 1D parameters and biases can cause training instability and hurts performance. The canonical test: `param.ndim >= 2` gets weight decay.

### Parameter Group Construction

```python
def build_optimizer(encoder: nn.Module, predictor: nn.Module,
                    lr: float, weight_decay: float,
                    betas: tuple = (0.9, 0.95)) -> torch.optim.AdamW:
    """
    Create AdamW with 4 parameter groups separating encoder/predictor
    and weight/bias-or-1D parameters.

    Args:
        encoder:      Context encoder module
        predictor:    Predictor module
        lr:           Base learning rate
        weight_decay: Initial weight decay (WD schedule will override each step)
        betas:        AdamW momentum coefficients (default: 0.9, 0.95)

    Returns:
        Configured AdamW optimizer
    """

    def split_params(module: nn.Module):
        """Split module params into (decay, no_decay) groups."""
        decay, no_decay = [], []
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            # 1D params (biases, LayerNorm weights/biases) skip weight decay
            if param.ndim == 1 or name.endswith('.bias'):
                no_decay.append(param)
            else:
                decay.append(param)
        return decay, no_decay

    enc_decay, enc_no_decay = split_params(encoder)
    pred_decay, pred_no_decay = split_params(predictor)

    param_groups = [
        # Group 0: Encoder weights (with WD)
        {
            'params': enc_decay,
            'lr': lr,
            'weight_decay': weight_decay,
            'name': 'encoder_weights',
        },
        # Group 1: Predictor weights (with WD)
        {
            'params': pred_decay,
            'lr': lr,
            'weight_decay': weight_decay,
            'name': 'predictor_weights',
        },
        # Group 2: Encoder biases / 1D (no WD)
        {
            'params': enc_no_decay,
            'lr': lr,
            'weight_decay': 0.0,
            'name': 'encoder_no_decay',
        },
        # Group 3: Predictor biases / 1D (no WD)
        {
            'params': pred_no_decay,
            'lr': lr,
            'weight_decay': 0.0,
            'name': 'predictor_no_decay',
        },
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=lr,
        betas=betas,
        eps=1e-8,
    )
    return optimizer
```

### DROID Variant: Differential Learning Rate

For DROID fine-tuning, the frozen encoder uses a lower LR scale:

```python
def build_droid_optimizer(encoder: nn.Module, predictor: nn.Module,
                           lr: float, weight_decay: float,
                           encoder_lr_scale: float = 0.1) -> torch.optim.AdamW:
    """DROID fine-tuning: encoder uses scaled-down LR."""
    enc_lr = lr * encoder_lr_scale  # e.g., 1e-4 instead of 1e-3
    pred_lr = lr

    enc_decay, enc_no_decay = split_params(encoder)
    pred_decay, pred_no_decay = split_params(predictor)

    param_groups = [
        {'params': enc_decay,    'lr': enc_lr,  'weight_decay': weight_decay},
        {'params': pred_decay,   'lr': pred_lr, 'weight_decay': weight_decay},
        {'params': enc_no_decay, 'lr': enc_lr,  'weight_decay': 0.0},
        {'params': pred_no_decay,'lr': pred_lr,  'weight_decay': 0.0},
    ]
    return torch.optim.AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)
```

---

## Learning Rate Schedules

### Schedule 1: Warmup + Cosine Decay

The standard schedule for V-JEPA 2 pretraining.

```
LR
 ^
 |              * * *
 |           *         *
 |         *              *
 |       *                    *
 |     *                          *
 |   *                                 *
 | *                                         * * * * *  <- final_lr
 +---+----------+---------------------------------> step
   0  warmup    T_max
```

**Formula:**
```
if step < warmup_steps:
    lr(step) = ref_lr * step / warmup_steps
else:
    progress = (step - warmup_steps) / (T_max - warmup_steps)
    lr(step) = final_lr + 0.5 * (ref_lr - final_lr) * (1 + cos(pi * progress))
```

**Implementation:**
```python
import math

def warmup_cosine_lr(step: int, ref_lr: float, final_lr: float,
                     warmup_steps: int, total_steps: int) -> float:
    """
    Compute LR for step t using warmup + cosine decay.

    Args:
        step:         Current step (0-indexed)
        ref_lr:       Peak learning rate after warmup
        final_lr:     Minimum learning rate at end of cosine decay
        warmup_steps: Number of linear warmup steps
        total_steps:  Total training steps

    Returns:
        Current learning rate
    """
    if step < warmup_steps:
        # Linear warmup from 0 to ref_lr
        return ref_lr * (step + 1) / warmup_steps
    else:
        # Cosine decay from ref_lr to final_lr
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return final_lr + (ref_lr - final_lr) * cosine_factor
```

### Schedule 2: Warmup + Stable + Decay (Cooldown/Annealing)

Used during the cooldown phase where the model trains at higher resolution for a shorter duration.

```
LR
 ^
 |         * * * * * * * * *
 |       *                    *
 |     *                           *
 |   *                                  *
 | *                                         *  <- final_lr
 +---+----------+----------------+-----------> step
   0  warmup   stable_end       T_max
```

**Formula:**
```
if step < warmup_steps:
    lr(step) = ref_lr * step / warmup_steps
elif step < stable_steps:
    lr(step) = ref_lr
else:
    # Linear decay from ref_lr to final_lr
    progress = (step - stable_steps) / (T_max - stable_steps)
    lr(step) = ref_lr - (ref_lr - final_lr) * progress
```

**Implementation:**
```python
def warmup_stable_decay_lr(step: int, ref_lr: float, final_lr: float,
                            warmup_steps: int, stable_steps: int,
                            total_steps: int) -> float:
    """
    Compute LR for warmup -> stable plateau -> linear decay schedule.

    Args:
        step:         Current step
        ref_lr:       Peak / plateau learning rate
        final_lr:     Terminal learning rate after decay
        warmup_steps: Steps for linear warmup
        stable_steps: Step at which decay begins (> warmup_steps)
        total_steps:  Total training steps

    Returns:
        Current learning rate
    """
    if step < warmup_steps:
        return ref_lr * (step + 1) / warmup_steps
    elif step < stable_steps:
        return ref_lr
    else:
        progress = (step - stable_steps) / max(1, total_steps - stable_steps)
        progress = min(progress, 1.0)  # clamp to [0, 1]
        return ref_lr - (ref_lr - final_lr) * progress
```

### Schedule 3: Linear Decay (Anneal Only)

Simple linear decay for the annealing sub-phase within cooldown. Used when loading from a pretrained checkpoint and immediately decaying.

```python
def linear_decay_lr(step: int, ref_lr: float, final_lr: float,
                    total_steps: int) -> float:
    """Linear LR decay from ref_lr to final_lr over total_steps."""
    progress = step / max(1, total_steps)
    progress = min(progress, 1.0)
    return ref_lr - (ref_lr - final_lr) * progress
```

---

## Weight Decay Schedule

Weight decay also follows a cosine schedule, **increasing** from `wd_start` to `wd_end`. This is the opposite direction of the LR schedule.

```
WD
 ^
 |                                         * * * *  <- wd_end
 |                                 * * * *
 |                          * * *
 |                    * * *
 |             * * *
 |       * *
 | * *                                            <- wd_start
 +-------------------------------------------------> step
   0                                     T_max
```

**Motivation:** Low WD early allows the model to explore freely. Higher WD late encourages compact, well-regularized representations.

**Formula:**
```
progress = step / T_max
wd(step) = wd_end - (wd_end - wd_start) * 0.5 * (1 + cos(pi * progress))
         = wd_start + (wd_end - wd_start) * 0.5 * (1 - cos(pi * progress))
```

**Implementation:**
```python
def cosine_wd(step: int, wd_start: float, wd_end: float,
              total_steps: int) -> float:
    """
    Cosine weight decay schedule increasing from wd_start to wd_end.

    Args:
        step:        Current step
        wd_start:    Initial weight decay (e.g., 0.04)
        wd_end:      Final weight decay (e.g., 0.4)
        total_steps: Total training steps

    Returns:
        Current weight decay value
    """
    progress = step / max(1, total_steps)
    progress = min(progress, 1.0)
    # Cosine goes from 1 at progress=0 to -1 at progress=1
    cosine_factor = 0.5 * (1.0 - math.cos(math.pi * progress))
    return wd_start + (wd_end - wd_start) * cosine_factor
```

---

## Per-Step Application

Both LR and WD must be applied **per optimizer step** by directly setting param group values:

```python
def apply_lr_and_wd(optimizer: torch.optim.AdamW,
                    lr: float, wd: float,
                    group_names: list = None) -> None:
    """
    Apply current LR and WD to all optimizer param groups.

    For DROID: only apply full LR to predictor groups;
               encoder groups use scaled LR.
    """
    for i, pg in enumerate(optimizer.param_groups):
        pg['lr'] = lr
        if pg.get('weight_decay', 0.0) > 0.0:
            pg['weight_decay'] = wd


def training_loop(loader, trainer, lr_sched, wd_sched, optimizer,
                  total_steps: int):
    """Complete training loop with per-step scheduling."""
    for step, (batch, masks_enc, masks_pred) in enumerate(loader):
        if step >= total_steps:
            break

        # Compute current LR and WD
        current_lr = lr_sched.step(step)
        current_wd = wd_sched.step(step)

        # Apply to optimizer
        apply_lr_and_wd(optimizer, current_lr, current_wd)

        # Forward / backward
        loss_dict = trainer.train_step(batch, masks_enc, masks_pred)

        # EMA update
        trainer.update_ema(step)

        if step % 100 == 0:
            print(f"step={step:6d}  loss={loss_dict['loss']:.4f}"
                  f"  lr={current_lr:.2e}  wd={current_wd:.4f}")
```

---

## Gradient Clipping

Gradient clipping is applied before the optimizer step to prevent exploding gradients:

```python
# Standard clipping for V-JEPA 2 pretraining
clip_grad = 1.0  # Max gradient norm

# With AMP scaler
scaler.unscale_(optimizer)
nn.utils.clip_grad_norm_(
    list(encoder.parameters()) + list(predictor.parameters()),
    max_norm=clip_grad
)
scaler.step(optimizer)
scaler.update()

# Without AMP
nn.utils.clip_grad_norm_(
    list(encoder.parameters()) + list(predictor.parameters()),
    max_norm=clip_grad
)
optimizer.step()
```

---

## Complete Per-Step Computation Reference

For a training run with:
- `ref_lr = 1e-3`, `final_lr = 1e-6`
- `weight_decay = 0.04`, `final_weight_decay = 0.4`
- `warmup_epochs = 40`, `total_epochs = 300`
- `steps_per_epoch = 1000` (example)

| Quantity | Step 0 | Step 40,000 (warmup end) | Step 150,000 (mid) | Step 300,000 (end) |
|---------|--------|--------------------------|---------------------|---------------------|
| LR | 1.0e-6 | 1.0e-3 | ~5.0e-4 | 1.0e-6 |
| WD | 0.04 | 0.04 | ~0.22 | 0.4 |
| EMA mom | 0.99925 | 0.99925 | 0.99925 | 0.99925 |

```python
# Derived quantities
warmup_steps = warmup_epochs * steps_per_epoch   # 40,000
total_steps  = total_epochs  * steps_per_epoch   # 300,000

# At step t:
lr = warmup_cosine_lr(t, ref_lr=1e-3, final_lr=1e-6,
                       warmup_steps=40_000, total_steps=300_000)
wd = cosine_wd(t, wd_start=0.04, wd_end=0.4, total_steps=300_000)
```

---

## Typical Hyperparameter Configurations

### Standard Pretraining (256px, 16 frames)

```python
optimizer_config = {
    'lr': 1e-3,
    'final_lr': 1e-6,
    'warmup_epochs': 40,
    'epochs': 300,
    'weight_decay': 0.04,
    'final_weight_decay': 0.4,
    'betas': (0.9, 0.95),
    'clip_grad': 1.0,
}
```

### Cooldown (384px, 64 frames)

```python
optimizer_config = {
    'lr': 1e-4,        # Start where pretrain ended
    'final_lr': 1e-7,  # Decay to near-zero
    'warmup_epochs': 0,
    'epochs': 30,
    'weight_decay': 0.4,       # Already at final WD
    'final_weight_decay': 0.4,
    'schedule': 'linear_decay',
    'is_anneal': True,
}
```

### DROID Fine-Tuning

```python
optimizer_config = {
    'lr': 5e-4,
    'final_lr': 5e-6,
    'warmup_epochs': 5,
    'epochs': 100,
    'weight_decay': 0.05,
    'final_weight_decay': 0.2,
    'encoder_lr_scale': 0.1,   # Encoder 10x lower LR
    'betas': (0.9, 0.999),      # Standard AdamW betas for fine-tuning
}
```

---

## AMP (Automatic Mixed Precision) Integration

```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler(enabled=use_bfloat16)

# Training step with AMP
optimizer.zero_grad()
with autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
    loss = compute_jepa_loss(batch, masks_enc, masks_pred)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
nn.utils.clip_grad_norm_(params, clip_grad)
scaler.step(optimizer)
scaler.update()
```

BFloat16 is preferred over Float16 for training because:
- Wider dynamic range (same exponent bits as Float32)
- Less likely to cause NaN/Inf with large gradient values
- Hardware support on A100, H100, and newer GPUs
