# GradScaler Internals Reference

## Overview

`torch.amp.GradScaler` implements dynamic loss scaling for fp16 mixed-precision training. Its job is to multiply the loss by a large scale factor before backward (shifting gradient magnitudes above fp16's underflow floor), then divide them back before the optimizer step. When overflow is detected, it skips the optimizer step entirely and reduces the scale factor.

---

## Scale Factor Lifecycle

### Initial State

```
init_scale = 65536.0    # 2**16, default
growth_factor = 2.0
backoff_factor = 0.5
growth_interval = 2000  # consecutive clean steps before growing
```

The scale factor starts at `init_scale`. Each `scaler.update()` call advances the lifecycle.

### Clean Step Behavior

A "clean step" is one where `unscale_()` found no inf or nan in any gradient tensor. The internal counter `_growth_tracker` increments by 1.

When `_growth_tracker >= growth_interval`:
```
scale *= growth_factor       # e.g., 65536 * 2.0 = 131072
_growth_tracker = 0          # reset counter
```

Growth keeps pushing the scale upward, maximizing gradient magnitude without hitting inf. The dynamic equilibrium settles at a scale where the model occasionally overflows but not too often.

### Overflow Step Behavior

When any inf or nan is detected during `unscale_()`:
```
scale *= backoff_factor      # e.g., 65536 * 0.5 = 32768
_growth_tracker = 0          # reset clean-step counter
# optimizer.step() is SKIPPED
```

The scale is immediately halved and the optimizer step is skipped. Parameter values are unchanged. The model trains as if that batch did not occur.

### Steady-State Dynamics

Over many steps, the scale factor oscillates around a value where:
- Overflow rate is low enough that training makes progress
- Scale is high enough to prevent gradient underflow

If overflow rate is high (> 5%), the scale keeps getting halved faster than it grows. This is the warning condition.

---

## `_found_inf_per_device` Mechanism

During `scaler.unscale_(optimizer)`, PyTorch calls:
```python
torch._amp_foreach_non_finite_check_and_unscale_(
    gradients_per_device,
    found_inf_per_device,
    inv_scale
)
```

This single CUDA kernel simultaneously:
1. Multiplies each gradient by `1/scale` (the unscaling)
2. Checks each result for inf or nan
3. Writes a per-device flag (0.0 = clean, 1.0 = overflow)

The per-device flags are stored as float tensors on their respective CUDA devices in `scaler._found_inf_per_device`.

**Why per-device?** In multi-GPU training (DDP), different GPUs may hold different parameter shards. Each device's gradients must be checked independently. The overflow determination is then `any(device has inf)`.

---

## Step Skip Logic

During `scaler.step(optimizer)`, the internal method `_maybe_opt_step()` checks:

```python
def _maybe_opt_step(optimizer, optimizer_state, *args, **kwargs):
    retval = None
    if not sum(v.item() for v in optimizer_state["found_inf_per_device"].values()):
        retval = optimizer.step(*args, **kwargs)
    return retval
```

If the sum of all per-device inf flags is nonzero (any device found inf), `optimizer.step()` is never called. The method returns `None` instead of the optimizer's return value.

This is a **hard skip** — the parameters, momentum buffers, and Adam variance estimates are not updated. The corrupted gradients never touch the parameters.

---

## Skip Detection Pattern

Since `scaler.step()` does not return a boolean indicating whether the step was taken, the practical detection method is to compare the scale factor before and after `scaler.update()`:

```python
scale_before = scaler.get_scale()
scaler.step(optimizer)
scaler.update()
scale_after = scaler.get_scale()

stepped = (scale_after >= scale_before)
if not stepped:
    # overflow occurred, optimizer was skipped, scale was reduced
    num_steps_skipped += 1
```

**Why this works**: `scaler.update()` only reduces the scale when overflow was detected. If no overflow, the scale stays the same or grows. Therefore `scale_after < scale_before` is an exact proxy for "overflow was detected."

Edge case: The scale cannot decrease on a clean step. The growth only happens after `growth_interval` consecutive clean steps, so `scale_after > scale_before` means growth, and `scale_after == scale_before` means clean step without growth.

---

## Overflow Counters

Track these metrics throughout training:

```python
num_steps_total: int       # every call to optimizer_step()
num_steps_skipped: int     # every call where scale_after < scale_before
last_overflow_step: int    # step index of most recent skip
current_loss_scale: float  # scaler.get_scale() at each step
```

Derived metrics:
```python
effective_update_rate = 1.0 - num_steps_skipped / num_steps_total
skip_rate = num_steps_skipped / num_steps_total
```

---

## 5% Skip Rate Warning

If `skip_rate > 0.05` over a rolling window, the training loop is making minimal progress. Parameters update only 95% of the time at best; if skip rate is 50%, the model is effectively training at half-speed on only the "easy" batches.

The rolling window prevents a single burst of overflows from permanently setting off alarms. Evaluate over the last N steps (e.g., 500) rather than the entire training run.

**Warning thresholds**:
- `skip_rate > 0.05` (5%): emit high-priority warning
- `skip_rate > 0.20` (20%): training is severely degraded; consider reducing learning rate or switching to bf16
- `skip_rate > 0.50` (50%): training is not converging; abort

**Emit warning example**:
```
[WARN] fp16 skip rate: 7.3% over last 500 steps (36/500 skipped).
       Effective update rate: 92.7%.
       Consider: reduce LR, check logit scale, check weight initialization.
       Current GradScaler scale: 4096.0 (was 65536.0 at step 0).
```

---

## RuntimeError on Double `unscale_()`

Calling `scaler.unscale_(optimizer)` twice between consecutive `step()` + `update()` calls raises:

```
RuntimeError: unscale_() has already been called on this optimizer since the last update().
```

The scaler tracks whether `unscale_()` has been called per-optimizer using an internal `_per_optimizer_states` dict. The state transitions are:

```
READY -> UNSCALED  (on first unscale_() call)
UNSCALED -> STEPPED  (on step())
STEPPED -> READY  (on update())
```

Calling `unscale_()` in state `UNSCALED` raises the RuntimeError.

**Common cause**: Gradient accumulation loop accidentally calls `unscale_()` inside the micro-batch loop:

```python
# WRONG
for micro_batch in micro_batches:
    with autocast():
        loss = model(micro_batch) / num_micro_batches
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)  # WRONG: called N times!

# CORRECT
for micro_batch in micro_batches:
    with autocast():
        loss = model(micro_batch) / num_micro_batches
    scaler.scale(loss).backward()

# unscale_ only once, after all micro-batches
scaler.unscale_(optimizer)
clip_grad_norm_(model.parameters(), max_norm)
scaler.step(optimizer)
scaler.update()
```

---

## Gradient Accumulation with GradScaler

In gradient accumulation (N micro-batches before one optimizer step), the full pattern is:

```python
optimizer.zero_grad()
accumulation_steps = 4

for i, micro_batch in enumerate(micro_batches):
    is_last = (i == len(micro_batches) - 1)

    with torch.amp.autocast('cuda', dtype=torch.float16):
        loss = model(micro_batch) / accumulation_steps  # normalize

    # scale(loss).backward() accumulates scaled gradients
    scaler.scale(loss).backward()

    if is_last:
        # Only on last micro-batch: unscale, clip, step, update
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        scale_after = scaler.get_scale()

        stepped = scale_after >= scale_before
```

The `scaler.scale()` accumulates scaled gradients naturally since each micro-batch's backward adds to `.grad`. The `unscale_()` at the end divides all accumulated gradients by the scale factor, giving correct per-sample gradient estimates.

---

## State Dict for Checkpointing

GradScaler state can be saved and restored for training resumption:

```python
# Save
checkpoint = {
    'scaler': scaler.state_dict(),
    'optimizer': optimizer.state_dict(),
    'model': model.state_dict(),
}
torch.save(checkpoint, path)

# Restore
scaler.load_state_dict(checkpoint['scaler'])
optimizer.load_state_dict(checkpoint['optimizer'])
model.load_state_dict(checkpoint['model'])
```

The `state_dict()` contains:
- `scale`: current scale factor value
- `growth_factor`: configured growth factor
- `backoff_factor`: configured backoff factor
- `growth_interval`: configured growth interval
- `_growth_tracker`: current consecutive-clean-steps counter

Restoring from checkpoint preserves the dynamic state, preventing a scale factor reset that would cause overflow spikes on resume.
