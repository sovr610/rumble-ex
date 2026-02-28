# Precision Modes Reference

## Overview

Three precision modes govern how compute operations, parameter storage, and gradient scaling behave during a training step. Mode selection must be deliberate — wrong mode on wrong hardware causes silent precision loss or training stalls.

---

## fp32 Mode

**No autocast. No GradScaler. All operations run in 32-bit floating point.**

- All model parameters and buffers remain `torch.float32` throughout training.
- Optimizer reads and writes float32 gradients directly.
- No loss scaling occurs. Gradient values are the raw backward-pass outputs.
- This is the **baseline and debug mode**. All results are numerically stable and reproducible across runs (given fixed seeds).
- Slowest mode. Use during architecture prototyping, debugging NaN sources, or when running on CPU.

**Invariant**: No `torch.amp.autocast` context is entered. No `torch.amp.GradScaler` is constructed or used. The PrecisionContext `autocast_ctx()` returns `contextlib.nullcontext()`.

**When to use**:
- Debugging training instability (to isolate precision from architecture bugs)
- CPU-only training
- Hardware without fp16/bf16 tensor cores
- Final validation runs requiring bit-exact reproducibility

---

## bf16 Mode

**Uses `torch.amp.autocast('cuda', dtype=torch.bfloat16)`. No GradScaler.**

### Why bf16 Skips GradScaler

bf16 has an **8-bit exponent** (same as fp32), giving it the same representable range: approximately ±3.4e38. The mantissa is shorter (7 bits vs 23 bits for fp32), reducing precision, but the exponent range means gradient underflow below the minimum representable value (~1.2e-38) is virtually never triggered in practice.

fp32 minimum normal: ~1.2e-38
bf16 minimum normal: ~1.2e-38 (same exponent field size)

Contrast with fp16's 5-bit exponent: minimum normal ~6.1e-5. Typical gradient magnitudes (1e-5 to 1e-3) sit very close to this floor, making underflow common.

**Using GradScaler with bf16 is unnecessary overhead** and should be avoided.

### Master Weights Behavior

`torch.amp.autocast` affects **compute operations only**, not parameter storage. When autocast enters its region:
- `torch.nn.Linear`, `torch.nn.Conv2d`, `torch.matmul`, `torch.bmm` operate in bf16.
- Intermediate activations produced by these ops are bf16 inside the autocast region.
- **Model parameters (.weight, .bias tensors) remain float32** in memory.
- PyTorch autocast automatically casts inputs to bf16 before eligible ops, runs them in bf16, then returns bf16 outputs.

This is the "master weights in fp32" pattern. Gradients accumulate into float32 parameter `.grad` fields (autocast does not affect gradient accumulation dtype in standard training). The optimizer then applies float32 updates.

### Hardware Requirements

bf16 tensor cores require:
- NVIDIA Ampere or newer: A100, H100, A10G, RTX 3090, RTX 4090, etc.
- No bf16 tensor cores on Volta (V100) or Turing (T4, RTX 2080). Those GPUs require fp16.

Verify at runtime:
```python
assert torch.cuda.is_bf16_supported(), "bf16 not supported on this GPU"
```

### Usage Pattern

```python
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    logits = model(input_ids)
    loss = criterion(logits, labels)

loss.backward()          # gradients accumulate in fp32
optimizer.step()         # fp32 parameter update
optimizer.zero_grad()
```

---

## fp16 Mode

**Uses `torch.amp.autocast('cuda', dtype=torch.float16)`. GradScaler REQUIRED.**

### Why fp16 Requires GradScaler

fp16 has a **5-bit exponent**, giving it range approximately ±65504 and minimum normal ~6.1e-5. Gradients smaller than this underflow to zero, silently zeroing out updates for those parameters.

GradScaler multiplies the loss by a large scale factor (default 2^16 = 65536) before backward. This shifts all gradient magnitudes upward by 65536x, keeping small gradients above the fp16 minimum. After backward, GradScaler divides the gradients back down before the optimizer step.

### GradScaler Skip Logic

If any inf or nan appears in the scaled gradients (detected during `unscale_()`), the optimizer step is **entirely skipped** for that batch. This prevents corrupted gradients from reaching the parameters.

**Detection**: Compare `scale_before = scaler.get_scale()` vs `scale_after = scaler.get_scale()` across `scaler.update()`. A reduced scale means overflow was detected and step was skipped.

### Master Weights in fp16 Mode

Same as bf16: parameters remain float32. Autocast only affects compute ops. Do NOT call `.half()` on the model — this permanently converts parameters to fp16, breaking the master-weight-in-fp32 invariant. The optimizer would then receive fp16 parameter gradients and update fp16 parameters directly, losing precision.

**Wrong (breaks master weights)**:
```python
model.half()  # Do NOT do this
```

**Correct**:
```python
with torch.amp.autocast('cuda', dtype=torch.float16):
    loss = model(x)
```

### Usage Pattern

```python
scaler = torch.amp.GradScaler('cuda', enabled=True, init_scale=65536.0)

with torch.amp.autocast('cuda', dtype=torch.float16):
    logits = model(input_ids)
    loss = criterion(logits, labels)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

---

## Autocast Operation Policies

Not all operations are cast to lower precision inside autocast. PyTorch maintains an **eligibility list** per dtype.

### Cast to lower precision (bf16/fp16)

These operations benefit from tensor core acceleration and tolerate reduced mantissa precision:
- `torch.matmul`, `torch.mm`, `torch.bmm`
- `torch.nn.Linear` (which is matmul under the hood)
- `torch.nn.Conv1d`, `Conv2d`, `Conv3d`, `ConvTranspose*`
- `torch.nn.functional.linear`
- `torch.nn.MultiheadAttention` (QKV projections and attention matmul)
- `torch.nn.GRU`, `torch.nn.LSTM` (if using cuDNN)

### Stay fp32 inside autocast

These operations require full precision for numerical stability:
- `torch.nn.Softmax`, `F.softmax` — log-sum-exp susceptible to overflow in fp16
- `torch.nn.LayerNorm`, `F.layer_norm` — variance computation needs precision
- `torch.nn.BatchNorm*` running stats accumulation
- Loss functions: `F.cross_entropy`, `F.nll_loss`, `F.mse_loss` — accumulation is fp32
- `torch.exp`, `torch.log`, `torch.pow` — nonlinear ops with narrow safe range
- `torch.sum`, `torch.mean` — reductions stay fp32 to prevent accumulation error
- `torch.nn.Embedding` — lookup output is fp32 by default

**Reference**: `torch.amp.get_autocast_dtype()`, `torch._C._jit_get_operation()` policy tables.

### Autocast Scope Rules

Autocast applies only to operations executed **within** the `with torch.amp.autocast(...)` block. The backward pass **does not inherit autocast from forward** — but PyTorch's autograd engine caches the dtype from the forward pass for backward ops. Do not manually enter autocast during the backward pass.

**Wrong** (redundant and can cause issues):
```python
with autocast():
    loss = model(x)
with autocast():          # Do NOT enter autocast again for backward
    loss.backward()
```

**Correct**:
```python
with autocast():
    loss = model(x)
loss.backward()           # autograd handles dtype consistency internally
```

---

## Mode Derivation Logic

The `PrecisionConfig` contains `autocast_dtype` and `grad_scaler_enabled` fields that can be explicitly set or derived from `mode`.

### Derivation Rules

```python
def resolve_autocast_dtype(mode: str) -> Optional[torch.dtype]:
    if mode == "bf16":
        return torch.bfloat16
    elif mode == "fp16":
        return torch.float16
    elif mode == "fp32":
        return None          # autocast is disabled
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Must be fp32, bf16, or fp16.")

def resolve_scaler_enabled(mode: str) -> bool:
    return mode == "fp16"    # True only for fp16
```

If `autocast_dtype` is explicitly provided as a string in config (e.g., `"bfloat16"`), convert with `getattr(torch, dtype_str)`.

If `grad_scaler_enabled` is explicitly `True` for bf16 mode, emit a warning (not an error) — it will work but wastes compute.

---

## Common Mistakes

### 1. Calling `.half()` on the model

```python
model.half()  # WRONG: permanently converts parameters to fp16
```

This breaks the master-weights-in-fp32 invariant. The optimizer now works with fp16 parameters directly, losing the benefit of fp32 accumulation. The optimizer state (momentum, Adam variance) also becomes fp16. Over many steps, this causes precision loss in the optimizer state itself.

**Fix**: Use `torch.amp.autocast` instead. Never cast model parameters to fp16.

### 2. Using GradScaler with bf16

```python
# WRONG: unnecessary overhead for bf16
scaler = torch.amp.GradScaler('cuda', enabled=True)
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    loss = model(x)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

bf16's exponent range matches fp32, so underflow virtually never occurs. GradScaler adds overhead and complexity. The scale factor check also means optimizer steps may be skipped spuriously if bf16 accumulation produces inf for other reasons.

**Fix**: Set `grad_scaler_enabled=False` for bf16 and fp32.

### 3. Autocast outside forward pass

```python
with torch.amp.autocast('cuda', dtype=torch.float16):
    loss = model(x).backward()  # WRONG: autocast spans backward
```

Autocast in backward can cause unexpected dtype promotion in backward ops that don't belong under autocast.

**Fix**: Keep autocast around forward only.

### 4. Not checking GPU bf16 support

```python
cfg = PrecisionConfig(mode="bf16")
```

On a V100 GPU, bf16 ops fall back to emulation in software, making training slower than fp32 with no precision benefit.

**Fix**: Always call `torch.cuda.is_bf16_supported()` before selecting bf16 mode. Fall back to fp16 or fp32 on older hardware.

### 5. Using the deprecated autocast API

```python
# DEPRECATED: do not use
with torch.cuda.amp.autocast():
    ...
```

The `torch.cuda.amp.autocast` API is deprecated as of PyTorch 2.0. Use `torch.amp.autocast('cuda', dtype=...)` explicitly, which also works for other device types.

### 6. Forgetting `scaler.update()` after `scaler.step()`

If `scaler.update()` is not called, the scale factor never grows after consecutive clean steps, staying at `init_scale` forever. This also prevents the `_found_inf_per_device` state from being cleared, causing the next `unscale_()` to fail.

### 7. Calling `unscale_()` twice before `step()`

`scaler.unscale_(optimizer)` can only be called once per optimizer between consecutive `scaler.step()` / `scaler.update()` calls. Calling it twice raises:
```
RuntimeError: unscale_() has already been called on this optimizer since the last update().
```

This typically happens in gradient accumulation loops where `unscale_` is accidentally called inside the micro-batch loop instead of outside.
