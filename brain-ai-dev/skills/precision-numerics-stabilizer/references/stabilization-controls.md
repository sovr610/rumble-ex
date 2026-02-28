# Stabilization Controls Reference

## Overview

Stabilization controls are **active interventions** that prevent or mitigate precision failures, as opposed to sentinels which are passive monitors. Each control is designed to be configurable (enabled/disabled individually) and composable with the PrecisionContext workflow.

---

## D1: Gradient Clipping with AMP

### The Ordering Problem

When using fp16 with GradScaler, the gradient tensors in `.grad` fields are **scaled** (multiplied by the scaler's loss scale factor, e.g., 65536). If you call `clip_grad_norm_` before unscaling, you are computing the norm of scaled gradients:

```
‖scaled_grad‖ = ‖scale × grad‖ = scale × ‖grad‖ = 65536 × ‖grad‖
```

The `max_norm` threshold (e.g., 1.0) is then compared against this inflated norm. Since `65536 × actual_norm >> 1.0`, almost no gradients survive the clip — they are all zeroed or nearly zeroed. The clipping is effectively applying `max_norm = 1.0 / 65536 = 1.5e-5` to the actual gradients.

**This is a silent failure** — training appears to proceed (no error thrown), but gradients are being zeroed out by the inflated scale.

### Correct Ordering

```
scaler.unscale_(optimizer)           # Step 1: divide all grads by scale factor
clip_grad_norm_(params, max_norm)    # Step 2: clip at correct magnitude
scaler.step(optimizer)               # Step 3: check for inf, step if clean
scaler.update()                      # Step 4: update scale factor
```

**Step 1 — `scaler.unscale_(optimizer)`**: Divides all gradient tensors by the current scale factor, restoring them to their true magnitudes. Also checks for inf/nan and populates `_found_inf_per_device`.

**Step 2 — `clip_grad_norm_(params, max_norm)`**: Now operates on true gradient magnitudes. The returned norm is also the true norm (not inflated).

**Step 3 — `scaler.step(optimizer)`**: Checks `_found_inf_per_device`. If any inf/nan (from unscale_ check), skips optimizer.step(). Otherwise calls optimizer.step() normally (gradients are already unscaled).

**Step 4 — `scaler.update()`**: Updates the scale factor based on overflow detection. Resets internal state for the next step.

### Implementation in PrecisionContext

```python
def unscale_and_clip(self, optimizer, parameters, max_norm=None):
    max_norm = max_norm or self.cfg.max_grad_norm
    if self.cfg.grad_scaler_enabled:
        self.scaler.unscale_(optimizer)     # Must come first
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    return grad_norm.item()                 # Returns clipped global norm
```

### For fp32 and bf16 (No Scaler)

When `grad_scaler_enabled=False`, gradients are not scaled, so clipping is straightforward:

```python
def unscale_and_clip(self, optimizer, parameters, max_norm=None):
    max_norm = max_norm or self.cfg.max_grad_norm
    # No unscale needed
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    return grad_norm.item()
```

### Clip Value Selection

- `max_grad_norm = 1.0` is standard for transformers trained with AdamW
- Reduce to `0.5` or `0.25` if gradient explosions are common
- Set to `float('inf')` to disable clipping (still compute the norm for monitoring)

---

## D2: Loss Spike Detector

### Design

Maintain a rolling window of the last N losses. Compute the **median** (not mean) as the baseline. Compare the current loss to this median.

**Why median over mean**: The mean is sensitive to outliers. One very large loss value pollutes the mean, causing subsequent normal losses to appear as "below baseline" when comparing. The median is immune to a single outlier even if that outlier is 100x the normal value.

### Rolling Median Algorithm

```python
from collections import deque
import statistics

class LossSpikeDetector:
    def __init__(self, window: int, spike_pct: float):
        self.window = window
        self.spike_pct = spike_pct
        self.buffer = deque(maxlen=window)

    def is_spike(self, loss: float) -> bool:
        if len(self.buffer) < 10:   # Need minimum samples
            self.buffer.append(loss)
            return False
        median = statistics.median(self.buffer)
        threshold = median * (1.0 + self.spike_pct / 100.0)
        self.buffer.append(loss)    # Add after computing (current not in median)
        return loss > threshold
```

**Note**: The current loss is not included in the median computation — we compare current against history.

### Spike Response

When a spike is detected:
1. **Immediate sentinel check**: Run all sentinels now (don't wait for cadence).
2. **Snapshot (optional)**: Write a snapshot tagged with reason "loss_spike".
3. **Skip update (optional)**: For extreme spikes, skip the optimizer step for this batch.
4. **Log**: Always emit a warning with the spike magnitude.

```
[WARN Step 523] Loss spike detected:
  current_loss=4.312, median(last 100)=0.891
  spike_magnitude=4.84x (384% above median, threshold=200%)
  Running immediate sentinel check...
```

### Spike Threshold Selection

- `loss_spike_pct=200` (default): Current loss > 3× rolling median triggers alert
- `loss_spike_pct=500`: Only extreme spikes (6× median) trigger alert
- Lower values → more sensitive, more false positives during early training instability
- Higher values → only catastrophic spikes trigger response

---

## D3: Deterministic Debug Toggle

### Purpose

When a NaN failure is reproduced, non-determinism in CUDA kernels makes it hard to confirm whether a fix actually resolves the issue. Deterministic mode forces all CUDA operations to use deterministic algorithms, enabling bit-exact reproducibility.

### Flags to Set

```python
import torch

# Force deterministic CUDA algorithms
torch.use_deterministic_algorithms(True)

# Disable CuDNN benchmarking (which selects fastest non-deterministic algorithm)
torch.backends.cudnn.benchmark = False

# Force CuDNN deterministic mode
torch.backends.cudnn.deterministic = True
```

### Seed Everything

```python
import random
import torch

def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # all GPUs
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
```

### Restoring Original State

```python
class DeterministicDebugToggle:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self._original_flags = None

    def enable(self):
        self._original_flags = {
            'deterministic': torch.are_deterministic_algorithms_enabled(),
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
        }
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        seed_everything(self.seed)

    def disable(self):
        if self._original_flags is None:
            return
        try:
            torch.use_deterministic_algorithms(self._original_flags['deterministic'])
        except Exception:
            torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = self._original_flags['cudnn_benchmark']
        torch.backends.cudnn.deterministic = self._original_flags['cudnn_deterministic']
```

### Performance Cost

Deterministic mode disables cuDNN autotuning and forces serialized execution for certain ops. Expected overhead: **20-50% slower** training throughput. Only use for failure reproduction runs, not production training.

### Operations That May Fail in Deterministic Mode

Some PyTorch operations do not have deterministic implementations. Calling them with `use_deterministic_algorithms(True)` raises:
```
RuntimeError: ... does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True)'
```

Common culprits:
- `torch.Tensor.index_put_()` with duplicates
- `torch.nn.functional.embedding()` with gradients (use `padding_idx`)
- `torch.histogramdd()`

Workaround: Use `torch.use_deterministic_algorithms(True, warn_only=True)` to warn instead of raising.

---

## Logit Clamping

### What It Is

Clamp logit values to `[-threshold, +threshold]` before softmax or loss computation:

```python
class LogitClamper:
    def __init__(self, threshold: float = 80.0, enabled: bool = True):
        self.threshold = threshold
        self.enabled = enabled

    def clamp(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return logits
        return torch.clamp(logits, min=-self.threshold, max=self.threshold)
```

### When to Use

This is a **nuclear option** — it changes training semantics by preventing the model from producing high-confidence predictions during training. Use only when:
- Logit overflow is causing inf/nan that cannot be resolved by other means
- fp16 training on a dataset with extreme class imbalance
- As a temporary measure while debugging the root cause

### Threshold Selection

- `threshold=80.0` (default): Conservative. exp(80) ≈ 5.7e34, well within fp32 range.
- `threshold=50.0`: More aggressive. Prevents any logit overflow in fp16 (exp(50) ≈ 5.2e21, within fp16 range if unnormalized).
- `threshold=20.0`: Very aggressive. Forces near-uniform predictions early in training.

### Not a Fix

Logit clamping treats the symptom (large logits), not the cause (learning rate too high, bad weight initialization, insufficient layer normalization). Always investigate the root cause even when clamping is enabled.

---

## Auto-Recovery Controller

### Concept

When GradScaler detects an overflow and skips the optimizer step, the underlying cause is often that the current batch produced unusually large activations or gradients. Temporarily reducing the learning rate gives the optimizer a chance to recover without human intervention.

### Implementation

```python
class AutoRecoveryController:
    def __init__(self, lr_factor: float = 0.5, recovery_steps: int = 100):
        self.lr_factor = lr_factor
        self.recovery_steps = recovery_steps
        self._original_lrs = {}
        self._recovery_countdown = 0

    def on_overflow(self, optimizer):
        """Call when overflow is detected (optimizer step was skipped)."""
        if self._recovery_countdown > 0:
            return  # Already in recovery mode
        # Save original LRs
        self._original_lrs = {
            i: pg['lr'] for i, pg in enumerate(optimizer.param_groups)
        }
        # Reduce LRs
        for pg in optimizer.param_groups:
            pg['lr'] *= self.lr_factor
        self._recovery_countdown = self.recovery_steps
        logger.warning(
            f"Overflow detected. Reducing LR by {self.lr_factor}x "
            f"for {self.recovery_steps} steps."
        )

    def step(self, optimizer):
        """Call each optimizer step (even skipped ones) to track countdown."""
        if self._recovery_countdown <= 0:
            return
        self._recovery_countdown -= 1
        if self._recovery_countdown == 0:
            # Restore original LRs
            for i, pg in enumerate(optimizer.param_groups):
                pg['lr'] = self._original_lrs[i]
            logger.info("Auto-recovery complete. LR restored.")
            self._original_lrs = {}
```

### Cascade Prevention

Without auto-recovery, a single bad batch can trigger an overflow that cascades:
1. Overflow → scale halved → next step runs at higher scale risk
2. Another overflow → scale halved again
3. If this repeats, scale can collapse from 65536 to 256 in 8 steps
4. At scale=256, gradients are barely above fp16 minimum → underflow begins

Auto-recovery breaks this cascade by reducing the gradient magnitudes (via LR reduction) until the model returns to a stable region.

---

## Composable Stabilization Usage

Full training step with all controls active:

```python
for step, batch in enumerate(dataloader):
    optimizer.zero_grad()

    # 1. Forward pass under autocast
    with precision_ctx.autocast_ctx():
        logits = model(batch['input_ids'])
        # Optional logit clamping before loss
        logits = logit_clamper.clamp(logits)
        loss = criterion(logits, batch['labels'])

    # 2. Sentinel: logit check
    if monitor.should_check(step):
        logit_report = monitor.check_logits(logits)

    # 3. Spike detection
    if spike_detector.is_spike(loss.item()):
        all_reports = monitor.aggregate_report(step)

    # 4. Backward
    precision_ctx.backward(loss)

    # 5. Sentinel: gradient norms (after backward)
    if monitor.should_check(step):
        grad_report = monitor.check_grad_norms()

    # 6. Unscale + clip (ordering guaranteed by unscale_and_clip)
    grad_norm = precision_ctx.unscale_and_clip(
        optimizer, model.parameters()
    )

    # 7. Optimizer step (returns True if not skipped)
    stepped = precision_ctx.optimizer_step(optimizer)

    # 8. Auto-recovery on overflow
    if not stepped:
        auto_recovery.on_overflow(optimizer)
    auto_recovery.step(optimizer)

    # 9. Update NaN streak
    nan_streak = snapshot.update_nan_streak(loss.item(), weight_report)
    if snapshot.should_abort(nan_streak):
        snap_path = snapshot.capture(step, model, optimizer, batch, report)
        snapshot.abort(snap_path, report)
```
