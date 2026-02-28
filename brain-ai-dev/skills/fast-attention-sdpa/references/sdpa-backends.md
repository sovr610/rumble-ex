# PyTorch SDPA Backends — Complete Reference

## Overview

`torch.nn.functional.scaled_dot_product_attention` (SDPA) is a dispatch layer introduced in PyTorch 2.0. Rather than being a single kernel, it examines the input tensors at runtime and selects among four backend implementations. Understanding this dispatch logic is essential for controlling performance and debugging fallbacks.

---

## How SDPA Dispatch Works at Runtime

When you call `F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)`, PyTorch executes a priority-ordered selection procedure:

1. **Constraint evaluation** — For each enabled backend, PyTorch evaluates whether the current inputs satisfy that backend's requirements (dtype, head_dim range, device, contiguity, etc.).
2. **Priority order** — Flash Attention > Memory-Efficient > cuDNN > Math. The highest-priority backend that passes constraints wins.
3. **Global toggle check** — Before evaluating individual constraints, PyTorch checks whether each backend has been globally enabled or disabled via `torch.backends.cuda.*` flags.
4. **sdpa_kernel override** — If a `sdpa_kernel` context manager is active, only the listed backends are considered. This narrows the candidate set before priority ordering.
5. **Fallback** — If no fused backend can run, SDPA falls through to the Math backend (unless Math is also disabled).

The dispatch happens entirely in C++ (`aten/src/ATen/native/transformers/attention.cpp`). Python has no hook into the selection — you can only influence it via the two APIs described in this document.

### What "Can Run" Means

Each backend implements a `can_use_*` predicate that checks a list of specific conditions. These predicates are exposed to Python:

```python
params = torch.backends.cuda.SDPAParams(q, k, v, attn_mask, dropout_p, is_causal)
flash_ok  = torch.backends.cuda.can_use_flash_attention(params, debug=True)
eff_ok    = torch.backends.cuda.can_use_efficient_attention(params, debug=True)
cudnn_ok  = torch.backends.cuda.can_use_cudnn_attention(params, debug=True)
```

With `debug=True`, the function prints to stderr the exact reason it returns `False`. This output is the primary diagnostic tool — never guess why a backend failed; read the debug output.

---

## The Four Backends in Detail

### 1. Flash Attention (`SDPBackend.FLASH_ATTENTION`)

**What it is**: An implementation of the Flash Attention algorithm (Dao et al., 2022) compiled directly into PyTorch. This is distinct from the external `flash-attn` PyPI package — it is built into CUDA PyTorch distributions and requires no separate installation.

**Performance characteristics**:
- O(N) memory in sequence length (tiling avoids materializing the full N×N attention matrix)
- Fastest throughput for long sequences (S ≥ 512) on Ampere or newer GPUs
- On A100 80GB with bf16, seq_len=2048, head_dim=128: typically 3–5× faster than Math

**Capability requirements**:
- Device: CUDA only (not CPU, not MPS)
- GPU: SM80+ (Ampere: A100, A10, RTX 30xx) or SM86/SM89/SM90 (Ada, Hopper)
- dtype: `torch.float16` or `torch.bfloat16` only (not float32)
- head_dim: Must be ≤ 256, and typically must be a power of 2 (64, 128, 256) for best performance. Non-power-of-2 dims up to 256 may work but can fall back depending on the build.
- attn_mask: Only boolean masks or `None` are supported. Float additive masks are not supported in the built-in Flash backend (they are supported in the external flash-attn package's varlen API).
- dropout_p: Any value in [0, 1]; dropout is applied inside the fused kernel.
- is_causal: Supported via internal triangular masking.
- Sequence length: No hard limit, but optimal for S ≥ 512.

**Memory behavior**: Flash Attention recomputes attention weights during the backward pass (gradient checkpointing within the kernel) instead of storing the N×N matrix. Peak memory is O(N·d) rather than O(N²). For a sequence length of 4096 with 16 heads and head_dim 128, this saves ~2 GB compared to Math.

**Interaction with non-contiguous tensors**: Flash Attention requires contiguous tensors. PyTorch SDPA will automatically call `.contiguous()` on non-contiguous inputs before dispatching. This is a hidden copy — profile if you suspect it.

---

### 2. Memory-Efficient Attention (`SDPBackend.EFFICIENT_ATTENTION`)

**What it is**: Based on xFormers' memory-efficient attention kernel (Rabe & Staats, 2021), integrated into PyTorch. Also uses tiling to avoid the N×N materialization, but through a different algorithmic approach.

**Performance characteristics**:
- O(N) memory like Flash, but typically 20–40% slower than Flash for large sequences
- Handles more shape configurations — this is the preferred fallback when Flash cannot run
- Particularly good for variable-length batches and head_dim values that Flash won't accept

**Capability requirements**:
- Device: CUDA only
- GPU: SM70+ (Volta: V100, T4) — lower bar than Flash, making it the right choice for pre-Ampere hardware
- dtype: `torch.float16`, `torch.bfloat16`, and `torch.float32` all supported
- head_dim: Up to 256; also supports non-power-of-2 values more reliably than Flash
- attn_mask: Supports float additive masks in addition to boolean masks
- dropout_p: Supported
- is_causal: Supported

**When to prefer explicitly**: When your model runs on V100 or T4 hardware (SM70/SM75), use `backend=efficient` to avoid Flash falling back to Math silently.

---

### 3. cuDNN Attention (`SDPBackend.CUDNN_ATTENTION`)

**What it is**: Routes the attention computation through the cuDNN multi-head attention API. cuDNN applies its own hardware-specific fusion and tuning per GPU model.

**Performance characteristics**:
- On supported hardware (H100, newer A100 builds), can match or exceed Flash
- Highly variable performance — depends on cuDNN version, GPU SKU, and batch/shape configuration
- Best used when you have cuDNN tuning infrastructure and want maximum hardware utilization on specific deployment targets

**Capability requirements**:
- Device: CUDA only
- cuDNN version: 8.9.1+ for full functionality; some features require 9.x
- dtype: `torch.float16`, `torch.bfloat16`
- head_dim: cuDNN-version and GPU-dependent; check `can_use_cudnn_attention(params, debug=True)` for your specific setup
- attn_mask: Limited mask support compared to the other backends

**When to use**: Explicitly enable on H100 clusters with cuDNN 9.x when you want to benchmark against Flash for potential gains.

---

### 4. Math Backend (`SDPBackend.MATH`)

**What it is**: A reference implementation in C++ (and optionally uses `torch.baddbmm` under the hood). Computes attention in full precision, materializing the N×N attention matrix.

**Performance characteristics**:
- No fusion, no tiling — allocates O(N²) intermediate tensors
- For N=2048, head_dim=128: 8× more memory than Flash; 3–5× slower
- Perfectly numerically stable — use as ground truth for validation comparisons

**Capability requirements**:
- Works on any device (CPU, CUDA, MPS)
- Any dtype (float32, float16, bfloat16)
- Any head_dim
- Supports additive float masks, boolean masks, `is_causal`, and `dropout_p`

**When to use**:
- CPU-only environments (Flash and Efficient are CUDA-only)
- Validation testing: compare other backends against Math to verify correctness
- Debugging shape issues: if Math runs but Flash doesn't, the debug output from `can_use_flash_attention(params, debug=True)` tells you exactly why

---

## `torch.nn.attention.sdpa_kernel` — The Preferred API

`sdpa_kernel` is a thread-local context manager that sets which backends SDPA may consider. It is strictly preferred over the global toggles (described below) because:
- It is scoped, thread-local, and stackable (safe in multi-threaded data loaders)
- It is explicit — the exact list of permitted backends is visible at the call site
- It works correctly under `torch.compile`

### Basic Usage

```python
from torch.nn.attention import sdpa_kernel, SDPBackend

# Force Flash only (fails if Flash cannot run)
with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

# Prefer Flash, fall through to Math if Flash unavailable
with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]):
    out = F.scaled_dot_product_attention(q, k, v)

# All backends (equivalent to default)
with sdpa_kernel([SDPBackend.FLASH_ATTENTION,
                  SDPBackend.EFFICIENT_ATTENTION,
                  SDPBackend.CUDNN_ATTENTION,
                  SDPBackend.MATH]):
    out = F.scaled_dot_product_attention(q, k, v)
```

### Nesting Behavior

`sdpa_kernel` contexts can be nested. The innermost active context wins:

```python
with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]):
    # Here: Flash or Math
    with sdpa_kernel([SDPBackend.MATH]):
        # Here: Math only — inner context wins
        out = F.scaled_dot_product_attention(q, k, v)
    # Back to: Flash or Math
```

This means third-party code that wraps SDPA in a `sdpa_kernel([MATH])` context will override your outer Flash context. Be aware when composing with libraries.

### `SDPBackend` Enum Values

```python
from torch.nn.attention import SDPBackend

SDPBackend.FLASH_ATTENTION     # = 0
SDPBackend.EFFICIENT_ATTENTION # = 1
SDPBackend.CUDNN_ATTENTION     # = 2
SDPBackend.MATH                # = 3
SDPBackend.OVERRIDEABLE        # = 4  (used by custom kernels)
```

Pass a list of these values to `sdpa_kernel`. Order within the list does not affect priority — PyTorch always uses its internal Flash > Efficient > cuDNN > Math priority regardless of list order.

---

## Global Toggles — When to Use vs sdpa_kernel

Before `sdpa_kernel` existed, SDPA backends were controlled via global flags:

```python
torch.backends.cuda.enable_flash_sdp(True)        # Flash
torch.backends.cuda.enable_mem_efficient_sdp(True) # Efficient
torch.backends.cuda.enable_math_sdp(True)          # Math
torch.backends.cuda.enable_cudnn_sdp(True)         # cuDNN (newer API)
```

**Do not use these** in new code. They are process-global, not thread-safe, and interact poorly with `torch.compile`. They remain in PyTorch for backward compatibility with old scripts.

**When you must use global toggles**: If you are instrumenting a codebase that calls SDPA but you cannot inject a context manager at the call site (e.g., inside a third-party module), global toggles are the only option. Set them before the forward pass and restore them after.

---

## `SDPAParams` and Capability Checks

`torch.backends.cuda.SDPAParams` packages the inputs into a structure the C++ predicates can evaluate:

```python
params = torch.backends.cuda.SDPAParams(
    query=q,           # (B, H, S, D) tensor
    key=k,
    value=v,
    attn_mask=mask,    # Optional tensor or None
    dropout_p=0.0,
    is_causal=False,
)
```

Then call any combination of:

```python
can_flash   = torch.backends.cuda.can_use_flash_attention(params, debug=True)
can_eff     = torch.backends.cuda.can_use_efficient_attention(params, debug=True)
can_cudnn   = torch.backends.cuda.can_use_cudnn_attention(params, debug=True)
```

With `debug=True`, the function writes one or more diagnostic lines to stderr explaining the failure. Common messages include:

- `"Flash attention only supports sm80 or above"` — GPU is pre-Ampere
- `"Flash attention only supports fp16 and bf16 data type"` — input is float32
- `"For ufloat8 data type, flash attention is not supported"` — exotic dtype
- `"Flash attention requires q,k,v to be contig in the last dimension"` — layout issue
- `"Efficient attention only supports..."` — similar constraint messages

Capturing this debug output requires redirecting stderr:

```python
import io
import contextlib

buf = io.StringIO()
with contextlib.redirect_stderr(buf):
    can_flash = torch.backends.cuda.can_use_flash_attention(params, debug=True)
debug_msg = buf.getvalue()
```

---

## Numeric Differences Across Backends

SDPA backends produce slightly different outputs for identical inputs. This is expected and documented by PyTorch. The reasons:

1. **Operation fusion changes accumulation order** — fused kernels compute softmax and the weighted sum in a different sequence than Math, leading to different floating-point rounding.
2. **Flash uses fp32 accumulators internally even for fp16 inputs** — Math follows the tensor dtype exactly.
3. **Tile boundaries introduce small errors** — the tiled reduction in Flash and Efficient produces slightly different partial sums than the full reduction in Math.

**Practical tolerances** (empirically derived):

| dtype | atol | rtol |
|-------|------|------|
| float32 | 1e-5 | 1e-5 |
| float16 | 1e-2 | 1e-2 |
| bfloat16 | 1e-1 | 1e-1 |

When validating a backend switch, compare against Math with these tolerances, not with `torch.allclose` at default tolerances (which will fail for fp16 and bf16).

---

## CUDA Graphs Interaction

SDPA is compatible with CUDA Graphs, but with important constraints:

- **Backend selection must be stable across graph iterations** — the backend chosen during the capture replay must match every replay. If your input shapes change between replays (e.g., variable-length sequences), the captured graph will use a fixed backend, which may not be appropriate for all replay shapes.
- **sdpa_kernel context and CUDA Graphs**: The `sdpa_kernel` context active during the `torch.cuda.graph()` capture is baked into the captured graph. Changes to the context after capture have no effect on replays.
- **Recommended pattern**: Use fixed shapes and dtypes when combining CUDA Graphs with SDPA. Enable a single fused backend explicitly to avoid any shape-dependent fallback during capture.

```python
# Correct: deterministic backend for graph capture
with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
    with torch.cuda.graph(g):
        out = model(static_input)
```

---

## torch.compile Interaction

`torch.compile` traces through `F.scaled_dot_product_attention` and can inline the backend selection. Key points:

- **Flash and Efficient are compile-friendly**: The C++ kernels are treated as opaque by Triton. `torch.compile` leaves them as-is and fuses surrounding ops.
- **Math backend may be partially compiled**: Under `torch.compile`, the Math backend may be further optimized with Triton-generated kernels, but this is not guaranteed.
- **`sdpa_kernel` inside compiled functions**: Works correctly — the context manager is respected during both tracing and execution.
- **Graph breaks**: If PyTorch cannot determine the backend statically, it may insert a graph break. Use `torch.compile(fullgraph=False)` to allow breaks, or ensure your backend choice is static (no Python conditionals based on runtime tensor properties).

---

## Common Pitfalls

### Pitfall 1: Dropout in Eval

```python
# WRONG — passes training dropout_p in eval mode
out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p)

# CORRECT — zero out dropout in eval
p = self.dropout_p if self.training else 0.0
out = F.scaled_dot_product_attention(q, k, v, dropout_p=p)
```

Passing `dropout_p > 0.0` in eval corrupts inference outputs — SDPA will apply dropout stochastically even without a training-mode guard.

### Pitfall 2: is_causal + attn_mask Conflict

```python
# WRONG — SDPA raises RuntimeError
out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=True)

# CORRECT — use one or the other
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
# OR
out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
```

### Pitfall 3: Shape Convention

SDPA expects `(batch, heads, seq_len, head_dim)`. Many Transformer implementations use `(batch, seq_len, heads, head_dim)`. Permute at the call site:

```python
# (B, S, H, D) -> (B, H, S, D)
q = q.permute(0, 2, 1, 3)
k = k.permute(0, 2, 1, 3)
v = v.permute(0, 2, 1, 3)
out = F.scaled_dot_product_attention(q, k, v)
# (B, H, S, D) -> (B, S, H, D)
out = out.permute(0, 2, 1, 3)
```

### Pitfall 4: Silently Slow on CPU

On CPU, only the Math backend runs. This is expected and correct — but easy to miss if your dev machine has no GPU and you don't explicitly test on GPU. Add a log statement at startup:

```python
import torch
if not torch.cuda.is_available():
    print("WARNING: SDPA running on CPU — fused backends unavailable")
```

### Pitfall 5: float32 Inputs Blocking Flash

Flash Attention does not support float32. If your model uses float32 activations (common for small models and debugging), SDPA will silently fall back to Math. Either use `autocast` to run in bf16, or explicitly configure `backend=efficient` which supports float32.

### Pitfall 6: head_dim=96 or Non-Power-of-2 Dims

Some models use head_dim=96 (BERT-large style: 768/8=96). Flash may not support this depending on the PyTorch version and GPU. Always run the capability check probe at startup and log the results.
