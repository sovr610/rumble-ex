# Testing Matrix Reference

## Overview

This matrix defines all test scenarios for the V-JEPA 2 training infrastructure. Tests are organized by component and criticality. Each scenario includes inputs, expected behavior, and assertion criteria.

---

## Test Category 1: JEPA Forward Pass

### T1.1 — Basic Forward Produces Valid Loss

**What is being tested:** Full encode -> predict -> target -> loss pipeline produces a finite, positive scalar.

```python
import math
import torch
import pytest


def test_forward_produces_finite_loss(tiny_trainer):
    context = torch.randn(2, 8, 64)
    masks_enc  = [torch.arange(8)]
    masks_pred = [torch.arange(8, 16)]

    loss_dict = tiny_trainer.train_step(
        batch=context,
        masks_enc=masks_enc,
        masks_pred=masks_pred,
    )

    assert 'loss' in loss_dict
    loss_val = loss_dict['loss']
    assert isinstance(loss_val, float)
    assert math.isfinite(loss_val), f"Loss is not finite: {loss_val}"
    assert loss_val >= 0.0, f"Loss is negative: {loss_val}"
```

**Pass criteria:**
- `loss_dict['loss']` is a Python float
- `math.isfinite(loss_val)` is True
- `loss_val >= 0.0`

---

### T1.2 — Loss Decreases Over Training Steps

**What is being tested:** The training loop can actually reduce the loss on a fixed synthetic dataset (overfitting test).

```python
def test_loss_decreases_over_steps(tiny_trainer):
    context = torch.randn(2, 8, 64)
    masks_enc  = [torch.arange(8)]
    masks_pred = [torch.arange(8, 16)]

    losses = []
    for step in range(50):
        loss_dict = tiny_trainer.train_step(
            batch=context,
            masks_enc=masks_enc,
            masks_pred=masks_pred,
        )
        tiny_trainer.update_ema(step)
        losses.append(loss_dict['loss'])

    start_loss = sum(losses[:5]) / 5
    end_loss   = sum(losses[-5:]) / 5
    assert end_loss < start_loss, (
        f"Loss did not decrease: start={start_loss:.4f}, end={end_loss:.4f}"
    )
```

**Pass criteria:**
- Mean loss over last 5 steps < mean loss over first 5 steps

---

### T1.3 — Predictor Output Shape

**What is being tested:** Predictor returns the correct shape `[B, N_pred, embed_dim]`.

```python
def test_predictor_output_shape():
    from assets.predictor_template import VisionTransformerPredictor

    B, N_vis, N_pred = 3, 10, 6
    embed_dim = 128
    pred_dim  = 64

    predictor = VisionTransformerPredictor(
        embed_dim=embed_dim,
        predictor_embed_dim=pred_dim,
        depth=2,
        num_heads=4,
        num_targets=8,
    )

    context = torch.randn(B, N_vis, embed_dim)
    masks_enc  = [torch.arange(N_vis)]
    masks_pred = [torch.arange(N_vis, N_vis + N_pred)]

    with torch.no_grad():
        output = predictor(context, masks_enc, masks_pred)

    assert output.shape == (B, N_pred, embed_dim), (
        f"Expected ({B}, {N_pred}, {embed_dim}), got {output.shape}"
    )
```

**Pass criteria:**
- `output.shape == (B, N_pred, embed_dim)` exactly

---

### T1.4 — Smooth L1 Loss Correctness

**What is being tested:** The smooth L1 function produces exact expected values for known inputs.

```python
def test_smooth_l1_zero_diff():
    """Identical inputs produce zero loss."""
    from assets.jepa_trainer_template import smooth_l1_loss

    pred   = torch.zeros(2, 4, 16)
    target = torch.zeros(2, 4, 16)
    loss = smooth_l1_loss(pred, target)
    assert abs(loss.item()) < 1e-6, f"Expected 0, got {loss.item()}"


def test_smooth_l1_small_diff():
    """Small diff uses quadratic regime: 0.5 * diff^2 / beta."""
    from assets.jepa_trainer_template import smooth_l1_loss

    pred   = torch.zeros(1, 1, 1)
    target = torch.full((1, 1, 1), 0.5)
    loss = smooth_l1_loss(pred, target, beta=1.0)
    expected = 0.5 * 0.5 ** 2 / 1.0  # = 0.125
    assert abs(loss.item() - expected) < 1e-4, (
        f"Expected {expected:.4f}, got {loss.item():.4f}"
    )


def test_smooth_l1_large_diff():
    """Large diff uses linear regime: |diff| - 0.5 * beta."""
    from assets.jepa_trainer_template import smooth_l1_loss

    pred   = torch.zeros(1, 1, 1)
    target = torch.full((1, 1, 1), 2.0)
    loss = smooth_l1_loss(pred, target, beta=1.0)
    expected = 2.0 - 0.5 * 1.0  # = 1.5
    assert abs(loss.item() - expected) < 1e-4, (
        f"Expected {expected:.4f}, got {loss.item():.4f}"
    )
```

**Pass criteria:**
- Zero loss for identical tensors
- Quadratic region exact within 1e-4 absolute tolerance
- Linear region exact within 1e-4 absolute tolerance

---

### T1.5 — Forward Pass Does Not Raise for Various Batch Sizes

```python
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_forward_various_batch_sizes(batch_size):
    from assets.predictor_template import VisionTransformerPredictor

    embed_dim = 64
    predictor = VisionTransformerPredictor(
        embed_dim=embed_dim, predictor_embed_dim=32,
        depth=2, num_heads=2, num_targets=4,
    )

    context = torch.randn(batch_size, 8, embed_dim)
    masks_enc  = [torch.arange(8)]
    masks_pred = [torch.arange(8, 12)]

    with torch.no_grad():
        output = predictor(context, masks_enc, masks_pred)

    assert output.shape[0] == batch_size
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
```

---

## Test Category 2: EMA Update Correctness

### T2.1 — Target Encoder Diverges from Context Encoder

**What is being tested:** After updates, target encoder parameters differ from context encoder.

```python
def test_ema_target_differs_from_encoder():
    from assets.ema_manager_template import EMAManager
    import torch.nn as nn

    encoder = nn.Linear(32, 32)
    ema = EMAManager(encoder, ema_schedule=(0.9, 0.99), total_steps=100)

    # Substantially modify encoder weights
    with torch.no_grad():
        for p in encoder.parameters():
            p.add_(torch.randn_like(p) * 0.5)

    ema.update(step=5)

    target = ema.get_target_encoder()
    any_differs = False
    for p_enc, p_tgt in zip(encoder.parameters(), target.parameters()):
        if not torch.allclose(p_enc, p_tgt, atol=1e-4):
            any_differs = True
            break

    assert any_differs, "Target encoder should differ from encoder after update"
```

**Pass criteria:**
- At least one parameter tensor differs between encoder and target encoder after update

---

### T2.2 — EMA Momentum at Step 0 Equals ema_start

**What is being tested:** Momentum schedule is computed correctly at the lower boundary.

```python
def test_ema_momentum_at_step_0():
    from assets.ema_manager_template import EMAManager
    import torch.nn as nn

    ema_start, ema_end = 0.99, 0.999
    encoder = nn.Linear(16, 16)
    ema = EMAManager(encoder, (ema_start, ema_end), total_steps=1000)

    momentum_0 = ema.update(step=0)

    assert abs(momentum_0 - ema_start) < 1e-5, (
        f"Momentum at step 0 should be {ema_start}, got {momentum_0}"
    )
```

**Pass criteria:**
- `|momentum_0 - ema_start| < 1e-5`

---

### T2.3 — EMA Momentum at Final Step Equals ema_end

```python
def test_ema_momentum_at_final_step():
    from assets.ema_manager_template import EMAManager
    import torch.nn as nn

    ema_start, ema_end = 0.99, 0.999
    total_steps = 1000
    encoder = nn.Linear(16, 16)
    ema = EMAManager(encoder, (ema_start, ema_end), total_steps)

    momentum_final = ema.update(step=total_steps - 1)

    assert abs(momentum_final - ema_end) < 1e-4, (
        f"Momentum at final step should be {ema_end}, got {momentum_final}"
    )
```

---

### T2.4 — EMA Update Formula Correctness

**What is being tested:** `theta_target = m * theta_target + (1-m) * theta_encoder` is applied correctly.

```python
def test_ema_update_formula():
    from assets.ema_manager_template import EMAManager
    import torch.nn as nn

    encoder = nn.Linear(16, 16)
    ema = EMAManager(encoder, (0.9, 0.9), total_steps=100)
    target = ema.get_target_encoder()

    # Record initial target params
    old_target_params = [p.clone() for p in target.parameters()]

    # Set encoder to known values
    with torch.no_grad():
        for p in encoder.parameters():
            p.fill_(1.0)

    m = ema.update(step=50)

    for old_t, p_t in zip(old_target_params, target.parameters()):
        expected = m * old_t + (1.0 - m) * 1.0
        assert torch.allclose(p_t, expected, atol=1e-5), (
            "EMA update formula is incorrect"
        )
```

---

### T2.5 — No Gradients Through Target Encoder

```python
def test_no_grad_through_target_encoder():
    from assets.ema_manager_template import EMAManager
    import torch.nn as nn

    encoder = nn.Linear(32, 32)
    ema = EMAManager(encoder, (0.99, 0.999), total_steps=100)
    target = ema.get_target_encoder()

    for p in target.parameters():
        assert not p.requires_grad, (
            f"Target encoder parameter {p.shape} has requires_grad=True"
        )
```

---

## Test Category 3: Checkpoint Round-Trip

### T3.1 — Save and Load Preserves All State

**What is being tested:** Save then load produces byte-identical parameter tensors.

```python
def test_checkpoint_round_trip(tmp_path, tiny_trainer):
    context = torch.randn(2, 8, 64)
    masks_enc  = [torch.arange(8)]
    masks_pred = [torch.arange(8, 16)]

    for step in range(3):
        tiny_trainer.train_step(context, masks_enc, masks_pred)
        tiny_trainer.update_ema(step)

    ckpt_path = str(tmp_path / 'checkpoint.pth')
    tiny_trainer.save_checkpoint(ckpt_path, epoch=1)

    # Reload into fresh trainer
    tiny_trainer2 = build_fresh_trainer()
    epoch = tiny_trainer2.load_checkpoint(ckpt_path)

    assert epoch == 1

    for p1, p2 in zip(tiny_trainer.encoder.parameters(),
                       tiny_trainer2.encoder.parameters()):
        assert torch.allclose(p1, p2, atol=1e-7), "Encoder params differ after load"

    for p1, p2 in zip(tiny_trainer.predictor.parameters(),
                       tiny_trainer2.predictor.parameters()):
        assert torch.allclose(p1, p2, atol=1e-7), "Predictor params differ after load"
```

**Pass criteria:**
- `epoch == 1`
- All encoder/predictor/target-encoder parameter tensors match within 1e-7 absolute tolerance

---

### T3.2 — Training Resumes with Matching Loss

**What is being tested:** Loss computed after loading matches loss before saving (same batch, eval mode, no_grad).

```python
def test_checkpoint_resume_identical_loss(tmp_path, tiny_trainer):
    torch.manual_seed(42)
    context = torch.randn(2, 8, 64)
    masks_enc  = [torch.arange(8)]
    masks_pred = [torch.arange(8, 16)]

    for step in range(3):
        tiny_trainer.train_step(context, masks_enc, masks_pred)
        tiny_trainer.update_ema(step)

    ref_loss = tiny_trainer.compute_loss_only(context, masks_enc, masks_pred)

    ckpt_path = str(tmp_path / 'ckpt.pth')
    tiny_trainer.save_checkpoint(ckpt_path, epoch=3)

    tiny_trainer2 = build_fresh_trainer()
    tiny_trainer2.load_checkpoint(ckpt_path)
    loaded_loss = tiny_trainer2.compute_loss_only(context, masks_enc, masks_pred)

    assert abs(ref_loss - loaded_loss) < 1e-5, (
        f"Loss mismatch: ref={ref_loss:.6f}, loaded={loaded_loss:.6f}"
    )
```

---

## Test Category 4: Predictor Token Ordering

### T4.1 — Sort / Un-sort Is Invertible

```python
def test_sort_unsort_invertible():
    B, N, D = 2, 20, 32
    x = torch.randn(B, N, D)
    positions = torch.randperm(N)
    sort_idx   = positions.argsort()
    unsort_idx = sort_idx.argsort()

    x_sorted   = x[:, sort_idx, :]
    x_restored = x_sorted[:, unsort_idx, :]

    assert torch.allclose(x, x_restored, atol=1e-7), (
        "Sort -> unsort should return original ordering"
    )
```

---

### T4.2 — Target Extraction Produces Correct Size

```python
def test_target_extraction_correctness():
    from assets.predictor_template import VisionTransformerPredictor

    B, N_vis, N_pred, D = 2, 8, 6, 64
    predictor = VisionTransformerPredictor(
        embed_dim=D, predictor_embed_dim=32,
        depth=1, num_heads=2, num_targets=8,
    )
    predictor.eval()

    context = torch.randn(B, N_vis, D)
    masks_enc  = [torch.arange(N_vis)]
    masks_pred = [torch.arange(N_vis, N_vis + N_pred)]

    with torch.no_grad():
        output = predictor(context, masks_enc, masks_pred)

    assert output.shape == (B, N_pred, D), (
        f"Output should be ({B}, {N_pred}, {D}), got {output.shape}"
    )
```

---

### T4.3 — Different Masks Produce Different Predictions

```python
def test_different_masks_produce_different_outputs():
    from assets.predictor_template import VisionTransformerPredictor

    B, D = 2, 64
    predictor = VisionTransformerPredictor(
        embed_dim=D, predictor_embed_dim=32,
        depth=2, num_heads=4, num_targets=8,
    )
    predictor.eval()

    context_a = torch.randn(B, 8, D)
    context_b = torch.randn(B, 7, D)

    masks_enc_a  = [torch.arange(8)]
    masks_pred_a = [torch.arange(8, 14)]

    masks_enc_b  = [torch.arange(7)]
    masks_pred_b = [torch.arange(7, 13)]

    with torch.no_grad():
        output_a = predictor(context_a, masks_enc_a, masks_pred_a)
        output_b = predictor(context_b, masks_enc_b, masks_pred_b)

    # Outputs may be the same shape but should differ (different context + masks)
    assert output_a.shape == output_b.shape  # Both [B, 6, D]
    assert not torch.allclose(output_a, output_b, atol=1e-3), (
        "Different masks/contexts should produce different predictions"
    )
```

---

## Test Category 5: LR Schedule Shapes

### T5.1 — Warmup Starts Near Zero and Reaches ref_lr

```python
def test_warmup_lr_reaches_ref():
    from assets.lr_scheduler_template import WarmupCosineScheduler

    warmup_steps = 100
    ref_lr = 1e-3
    scheduler = WarmupCosineScheduler(
        ref_lr=ref_lr, final_lr=1e-6,
        warmup_steps=warmup_steps, total_steps=1000,
    )

    lr_at_0   = scheduler.step(0)
    lr_at_end = scheduler.step(warmup_steps)

    assert lr_at_0 < ref_lr * 0.05, (
        f"LR at step 0 should be near 0, got {lr_at_0}"
    )
    assert abs(lr_at_end - ref_lr) / ref_lr < 0.01, (
        f"LR at warmup end should be {ref_lr}, got {lr_at_end}"
    )
```

---

### T5.2 — Cosine Decays to final_lr at Last Step

```python
def test_cosine_decay_reaches_final():
    from assets.lr_scheduler_template import WarmupCosineScheduler

    final_lr    = 1e-6
    total_steps = 1000
    scheduler = WarmupCosineScheduler(
        ref_lr=1e-3, final_lr=final_lr,
        warmup_steps=100, total_steps=total_steps,
    )

    lr_final = scheduler.step(total_steps - 1)
    assert abs(lr_final - final_lr) / final_lr < 0.02, (
        f"Final LR should be {final_lr}, got {lr_final}"
    )
```

---

### T5.3 — LR Is Monotonically Decreasing After Warmup

```python
def test_lr_monotone_after_warmup():
    from assets.lr_scheduler_template import WarmupCosineScheduler

    total_steps  = 500
    warmup_steps = 50
    scheduler = WarmupCosineScheduler(
        ref_lr=1e-3, final_lr=1e-6,
        warmup_steps=warmup_steps, total_steps=total_steps,
    )

    lrs = [scheduler.step(t) for t in range(warmup_steps, total_steps)]

    for i in range(len(lrs) - 1):
        assert lrs[i] >= lrs[i + 1] - 1e-12, (
            f"LR not monotonically decreasing at step {i + warmup_steps}: "
            f"{lrs[i]:.8e} > {lrs[i+1]:.8e}"
        )
```

---

## Test Category 6: WD Schedule Shapes

### T6.1 — WD Increases Monotonically

```python
def test_wd_increases_monotonically():
    from assets.lr_scheduler_template import CosineWDScheduler

    scheduler = CosineWDScheduler(
        wd_start=0.04, wd_end=0.4, total_steps=1000
    )
    wds = [scheduler.step(t) for t in range(1000)]

    for i in range(len(wds) - 1):
        assert wds[i] <= wds[i + 1] + 1e-10, (
            f"WD not monotonically increasing at step {i}: "
            f"{wds[i]:.6f} > {wds[i+1]:.6f}"
        )
```

---

### T6.2 — WD Starts and Ends at Correct Values

```python
def test_wd_boundary_values():
    from assets.lr_scheduler_template import CosineWDScheduler

    wd_start, wd_end = 0.04, 0.4
    total_steps = 1000
    scheduler = CosineWDScheduler(wd_start, wd_end, total_steps)

    wd_0       = scheduler.step(0)
    wd_end_val = scheduler.step(total_steps - 1)

    assert abs(wd_0 - wd_start) < 1e-5, (
        f"WD at step 0 should be {wd_start}, got {wd_0}"
    )
    assert abs(wd_end_val - wd_end) / wd_end < 0.02, (
        f"WD at final step should be {wd_end}, got {wd_end_val}"
    )
```

---

## Test Category 7: Training Configuration

### T7.1 — All Presets Are Valid

```python
def test_all_presets_valid():
    from assets.training_config_template import JEPATrainingConfig

    configs = [
        JEPATrainingConfig.pretrain_256(),
        JEPATrainingConfig.cooldown_384(),
        JEPATrainingConfig.droid_finetune(),
    ]

    for cfg in configs:
        errors = cfg.validate()
        assert len(errors) == 0, (
            f"Preset config has validation errors: {errors}"
        )
```

---

### T7.2 — Validation Catches Bad LR

```python
def test_validation_catches_bad_lr():
    from assets.training_config_template import JEPATrainingConfig

    cfg = JEPATrainingConfig(lr=-1.0)
    errors = cfg.validate()
    assert any('lr' in e.lower() for e in errors), (
        f"Validation should catch lr <= 0, got errors: {errors}"
    )
```

---

### T7.3 — Validation Catches Out-of-Range EMA

```python
def test_validation_catches_bad_ema():
    from assets.training_config_template import JEPATrainingConfig

    cfg = JEPATrainingConfig(ema_start=1.5)  # > 1.0 is invalid
    errors = cfg.validate()
    assert any('ema' in e.lower() for e in errors), (
        f"Validation should catch ema > 1.0, got errors: {errors}"
    )
```

---

## Test Coverage Summary

| Category | Tests | Priority | Covers |
|----------|-------|----------|--------|
| JEPA Forward | T1.1 - T1.5 | Critical | Loss computation, shapes, masking |
| EMA Update | T2.1 - T2.5 | Critical | Formula, schedule, no-grad |
| Checkpoint | T3.1 - T3.2 | Critical | Round-trip, resume |
| Token Ordering | T4.1 - T4.3 | High | Sort/unsort, extraction |
| LR Schedules | T5.1 - T5.3 | High | Warmup, decay, monotone |
| WD Schedules | T6.1 - T6.2 | High | Boundary values, monotone |
| Config | T7.1 - T7.3 | Medium | Presets, validation |

All critical tests (T1, T2, T3) map directly to the Done-When gates defined in SKILL.md.
