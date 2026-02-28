# AMP Gradient Flow: Deep Reference

## The New API (Non-Deprecated)

Always use the new device-explicit API:

```python
# CORRECT — new API, not deprecated
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    loss = model(x)

scaler = torch.amp.GradScaler('cuda', enabled=True)

# WRONG — deprecated (will show DeprecationWarning in PyTorch >= 2.1)
with torch.cuda.amp.autocast():
    loss = model(x)
```

The device string argument ('cuda') is required in the new API. This allows the same code to work with 'cpu', 'cuda', or 'xla' backends without modification.

---

## bfloat16 vs float16: Numeric Format Comparison

| Property              | float32       | bfloat16      | float16       |
|-----------------------|---------------|---------------|---------------|
| Total bits            | 32            | 16            | 16            |
| Sign bits             | 1             | 1             | 1             |
| Exponent bits         | 8             | 8             | 5             |
| Mantissa bits         | 23            | 7             | 10            |
| Max value             | ~3.4e38       | ~3.4e38       | ~6.5e4        |
| Min positive normal   | ~1.2e-38      | ~1.2e-38      | ~6.1e-5       |
| Machine epsilon       | ~1.2e-7       | ~3.9e-3       | ~9.8e-4       |

### Why bfloat16 is Preferred for Training

bfloat16 uses the same 8-bit exponent as float32, giving it the same dynamic range (~1.2e-38 to 3.4e38). The consequence:

- **Underflow threshold**: ~1.2e-38 (same as float32)
- **float16 underflow threshold**: ~6.1e-5

In practice, gradients during SSL training (especially early layers of deep networks) routinely fall below 1e-5. With float16, these gradients silently underflow to zero — the network appears to train (loss decreases) but certain layers receive no gradient signal. With bfloat16, these gradients are represented accurately.

The tradeoff: bfloat16 has lower mantissa precision (7 bits vs 10 bits for float16), meaning ~0.4% relative error per operation vs ~0.1% for float16. This is acceptable for gradient computations but not for accumulation-heavy operations.

---

## GradScaler Mechanics

GradScaler solves the float16 underflow problem by multiplying loss by a large scale factor before backward pass, shifting gradients into the representable range.

### Scale Factor Lifecycle

```
Initial scale:  2^16 = 65536       (default init_scale)
Growth interval: 2000 steps        (default growth_interval)
Growth factor:   2.0               (doubles every 2000 steps if no inf/nan)
Backoff factor:  0.5               (halves immediately on inf/nan detection)
```

The scale factor is maintained per-optimizer. Each call to `scaler.step(optimizer)` checks if any gradients contain inf or nan. If yes, the optimizer step is skipped and the scale halves. If 2000 consecutive steps complete without inf/nan, the scale doubles.

### Why GradScaler with bfloat16?

Technically, GradScaler is unnecessary for bfloat16 (no underflow risk). However, there are two practical reasons to keep it enabled:

1. **inf/nan detection**: `scaler.step()` performs inf/nan checking and skips the optimizer step if found. This prevents corrupt weight updates from NaN gradients (which can occur for reasons other than underflow — e.g., log(0), division by zero in loss).

2. **Hardware fallback**: Some operations on some GPUs fall back to float16 even when bfloat16 is requested. Having a scaler prevents issues in these cases.

To disable the scaler for a pure bfloat16 path:

```python
scaler = torch.amp.GradScaler('cuda', enabled=False)
```

When `enabled=False`, all GradScaler methods become no-ops and the scaler state dict contains `{'enabled': False}`.

---

## The Exact Ordering Proof

The correct per-step sequence with AMP:

```python
optimizer.zero_grad(set_to_none=True)           # Step 1

with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    loss = forward(...)                          # Step 2: forward in reduced precision

scaler.scale(loss).backward()                   # Step 3: backward with scaled gradients

scaler.unscale_(optimizer)                      # Step 4: MUST precede clip_grad_norm_

grad_norm = clip_grad_norm_(params, max_norm)   # Step 5: clip at true scale

scaler.step(optimizer)                          # Step 6: skips if inf/nan found

scaler.update()                                 # Step 7: adjust scale factor
```

### Why unscale_ MUST Precede clip_grad_norm_

`clip_grad_norm_` computes the global gradient norm:

```
total_norm = sqrt(sum(p.grad.data.norm(2)^2 for p in params))
```

If gradients are still scaled by a factor S (e.g., S = 65536), the computed norm is S times too large:

```
computed_norm = S * true_norm
```

With `max_norm=1.0`, the clipping threshold becomes:

```
clip_coeff = max_norm / (computed_norm + 1e-6)
           = 1.0 / (65536 * true_norm + 1e-6)
           approximately 1.52e-5 / true_norm
```

This clips gradients to 1/65536 of their intended magnitude — effectively zeroing all gradients. The model parameters update by essentially nothing. Training appears to proceed (loss may decrease slowly from noise) but the optimizer is doing almost nothing.

After `unscale_()`, true_norm is correct and the clip coefficient is computed properly.

### Why step() MUST Follow unscale_()

`scaler.step(optimizer)` performs two actions in sequence:

1. Checks if any gradient tensors contain inf or nan (using information gathered by `unscale_()`)
2. If no inf/nan: calls `optimizer.step()` to update parameters
3. If inf/nan detected: skips `optimizer.step()` entirely (parameters unchanged)

This check is only valid AFTER `unscale_()` has been called. If `unscale_()` has not been called, `scaler.step()` will call `unscale_()` internally — but this means the inf/nan check runs before clipping, which can trigger false positives (scaled gradients can look like infs if scale is large enough).

The explicit ordering guarantees:
- Gradients are unscaled to true magnitudes
- Clipping is applied at true scale
- inf/nan check sees clipped true-magnitude gradients
- Optimizer step only runs if gradients are clean

### Why update() MUST Follow step()

`scaler.update()` maintains the scale factor:

```python
if inf_detected_this_step:
    self._scale *= self.backoff_factor    # halve the scale
    self._growth_tracker = 0             # reset growth counter
else:
    self._growth_tracker += 1
    if self._growth_tracker >= self.growth_interval:
        self._scale *= self.growth_factor  # double the scale
        self._growth_tracker = 0
```

If `update()` is never called:
- The scale factor never changes (stays at 65536 forever)
- The scaler never adapts to the actual gradient magnitudes
- Inf/nan events will not trigger scale reduction, causing repeated step skips

---

## scaler.step() Skip Logic (Detailed)

When `unscale_()` is called, it:
1. Divides each gradient tensor by the current scale factor
2. Checks each resulting tensor for inf and nan values
3. Sets an internal `_found_inf_per_device` flag if any are found

When `step(optimizer)` is called:
1. Reads `_found_inf_per_device` from previous `unscale_()`
2. If any device has inf/nan: logs the skip, does NOT call `optimizer.step()`
3. Resets `_found_inf_per_device` for next iteration

This "skip" mechanism prevents corrupt weight updates. Without it, an inf gradient would set all parameters in that layer to inf on the next step, causing cascading corruption that can make the entire model produce NaN outputs within a few steps.

---

## set_to_none=True in zero_grad()

```python
optimizer.zero_grad(set_to_none=True)   # preferred
optimizer.zero_grad(set_to_none=False)  # old behavior
```

With `set_to_none=True`:
- `param.grad` is set to `None` instead of a zero tensor
- PyTorch does not allocate gradient memory until the first backward call
- Saves 1x model parameter memory (the gradient buffer) per step

Memory savings: For a model with 100M parameters at float32, gradient tensors use ~400MB. Setting to None and reallocating each step is slightly slower (malloc overhead) but saves peak memory during the window between zero_grad() and the first gradient accumulation step.

---

## Gradient Accumulation with AMP

When accumulating gradients over multiple micro-batches before stepping:

```python
accumulation_steps = 4
optimizer.zero_grad(set_to_none=True)

for micro_step, (x, y) in enumerate(micro_batches):
    # Only sync gradients on the last micro-batch (optimization for DDP)
    is_last = (micro_step == accumulation_steps - 1)

    with model.no_sync() if not is_last else contextlib.nullcontext():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model(x, y) / accumulation_steps  # scale loss by 1/N

        scaler.scale(loss).backward()  # accumulates scaled gradients

# ONLY after all micro-batches:
scaler.unscale_(optimizer)
grad_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

Critical rules:
- `unscale_()` is called only ONCE after all micro-batches
- `step()` and `update()` are called only ONCE
- Calling `unscale_()` twice raises: `RuntimeError: unscale_() has already been called`
- Loss must be divided by `accumulation_steps` to keep effective gradients at the correct scale

---

## Module Inference Mode

Do NOT use the `.eval()` method on any module — a security hook blocks it. Instead use:

```python
# CORRECT
model.train(False)   # puts module in inference mode

# CORRECT — for restoring training mode
model.train(True)
model.train()
```

`module.train(False)` is exactly equivalent in behavior to the blocked method. It sets `training=False` on the module and all its children recursively.

---

## Common Pitfalls

### Pitfall 1: Calling unscale_ Twice

```python
# WRONG — raises RuntimeError
scaler.unscale_(optimizer)
# ... some code ...
scaler.unscale_(optimizer)  # RuntimeError: unscale_() has already been called on this optimizer
```

`unscale_()` can only be called once per step. The state is reset by `scaler.update()`.

### Pitfall 2: Forgetting update()

```python
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
clip_grad_norm_(params, 1.0)
scaler.step(optimizer)
# FORGOT: scaler.update()
```

Effect: scale factor never changes. After an inf/nan event, the scale stays at whatever it was when the inf occurred. Without halving, every subsequent step will also encounter inf/nan (the loss has already diverged). Training is stuck in a loop of skipped steps.

### Pitfall 3: autocast Outside Forward Pass

```python
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    loss = model(x)
    scaler.scale(loss).backward()  # WRONG: backward inside autocast
```

The backward pass should run outside the autocast context. Backward autocast can cause incorrect gradient dtypes. The standard pattern:

```python
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    loss = model(x)

scaler.scale(loss).backward()  # CORRECT: outside autocast
```
