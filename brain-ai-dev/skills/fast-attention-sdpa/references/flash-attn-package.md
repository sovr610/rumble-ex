# External flash-attn Package — Complete Reference

## Overview

There are two distinct things called "Flash Attention" in the PyTorch ecosystem. Understanding the difference is essential:

1. **PyTorch's built-in Flash backend** (`SDPBackend.FLASH_ATTENTION`): Compiled into PyTorch itself. Available whenever you have a CUDA PyTorch installation on Ampere+ hardware. No additional packages required. Accessed via `F.scaled_dot_product_attention` with `sdpa_kernel`.

2. **Tri Dao's `flash-attn` PyPI package**: A separately installable CUDA extension. Provides lower-level APIs (`flash_attn_func`, `flash_attn_qkvpacked_func`) and is required by HuggingFace when `attn_implementation="flash_attention_2"` is set. Also provides FA3 APIs for Hopper GPUs.

**The recommendation**: Use PyTorch's built-in Flash backend via SDPA unless you have a specific reason to use the external package. The most common specific reason is HuggingFace model compatibility.

---

## When to Use the External flash-attn Package

Use the external package when:

1. **HuggingFace `attn_implementation="flash_attention_2"`** — HuggingFace models that support this flag call `flash_attn_func` directly, bypassing SDPA. The model will raise `ImportError` if the package is not installed.
2. **Variable-length (packed) sequences** — `flash_attn_varlen_func` handles sequences of different lengths packed into a single batch without padding. This is not exposed through SDPA's standard API.
3. **Custom attention patterns** — The external package provides finer-grained control over tile sizes, softmax scale, and alibi slopes not available through SDPA's API.
4. **FA3 on Hopper (H100)** — Flash Attention 3 (FA3) is only available in the external package for now. It provides further speedups on H100 via asynchronous WGMMA and TMA instructions.

Do not use the external package when:
- You just want fast attention on standard shapes — SDPA's built-in Flash handles this.
- You're on Windows (the package doesn't support Windows).
- You're on CPU (neither the built-in nor external Flash runs on CPU).

---

## Install Requirements

### Hardware

- GPU with **SM80 or higher** (Compute Capability 8.0+):
  - SM80: NVIDIA A100, A30
  - SM86: RTX 30xx (3090, 3080, etc.), A10, A40
  - SM89: RTX 40xx (4090, 4080), L40
  - SM90: H100, H800

- Earlier GPUs (V100/T4, SM70/SM75) are NOT supported by flash-attn. Use `SDPBackend.EFFICIENT_ATTENTION` via SDPA instead.

### Software

- **CUDA**: >= 11.6 (CUDA 11.8 or 12.x recommended for stability)
- **PyTorch**: >= 1.12 (1.13+ for bf16 support; 2.0+ recommended)
- **OS**: Linux only. macOS and Windows are not supported.
- **Python**: >= 3.7
- **GCC**: >= 7 (for compilation from source)

### CUDA Version Alignment

The `flash-attn` package must be compiled against the same CUDA version as your PyTorch installation. Mismatches cause cryptic errors. Verify alignment:

```bash
python -c "import torch; print(torch.version.cuda)"
nvcc --version
```

Both should report the same major.minor version (e.g., `12.1`).

---

## Installation

### Recommended Command

```bash
pip install flash-attn --no-build-isolation
```

**Why `--no-build-isolation`?** By default, pip creates an isolated build environment that may not have access to your system's CUDA installation or the correct PyTorch headers. `--no-build-isolation` tells pip to build in the current environment, which has the correct CUDA and PyTorch already installed.

Without this flag, the build often fails with errors like `nvcc: not found` or builds against a wrong CUDA version.

### Pre-built Wheels (Faster)

The flash-attn project provides pre-built wheels for common configurations at:
`https://github.com/Dao-AILab/flash-attention/releases`

These avoid compilation entirely (compilation can take 10–30 minutes):

```bash
# Example for PyTorch 2.3, CUDA 12.1, Python 3.11
pip install flash-attn==2.x.x+cu121torch23cxx11abiFALSE \
    --index-url https://github.com/Dao-AILab/flash-attention/releases/download/v2.x.x/
```

Check the releases page for the exact wheel filename matching your configuration.

### Compilation from Source

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
pip install . --no-build-isolation
```

Set `MAX_JOBS` to control parallel compilation (default uses all cores, which can OOM on machines with low RAM):

```bash
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

---

## FA2 vs FA3 API Differences

### Flash Attention 2 (FA2) — `flash-attn >= 2.0`

Primary API functions:

```python
from flash_attn import flash_attn_func, flash_attn_qkvpacked_func, flash_attn_kvpacked_func

# Standard API: separate q, k, v
out = flash_attn_func(
    q,              # (batch, seqlen, nheads, headdim)
    k,              # (batch, seqlen, nheads_k, headdim)
    v,              # (batch, seqlen, nheads_k, headdim)
    dropout_p=0.0,
    softmax_scale=None,  # defaults to 1/sqrt(headdim)
    causal=False,
    window_size=(-1, -1),  # For sliding window attention
    alibi_slopes=None,
    deterministic=False,
)
# returns: (batch, seqlen, nheads, headdim)

# Packed QKV API
out = flash_attn_qkvpacked_func(
    qkv,  # (batch, seqlen, 3, nheads, headdim)
    dropout_p=0.0,
    causal=False,
)

# KV-packed API (for cross-attention)
out = flash_attn_kvpacked_func(
    q,    # (batch, seqlen_q, nheads, headdim)
    kv,   # (batch, seqlen_k, 2, nheads_k, headdim)
    dropout_p=0.0,
    causal=False,
)
```

**Key FA2 shape convention**: `(batch, seqlen, nheads, headdim)` — note that seqlen comes BEFORE nheads. This is OPPOSITE to SDPA's convention of `(batch, nheads, seqlen, headdim)`. Always permute at the boundary.

Variable-length API (no padding needed):

```python
from flash_attn import flash_attn_varlen_func

out = flash_attn_varlen_func(
    q,            # (total_tokens, nheads, headdim)
    k,            # (total_tokens, nheads_k, headdim)
    v,            # (total_tokens, nheads_k, headdim)
    cu_seqlens_q, # (batch+1,) cumulative sequence lengths
    cu_seqlens_k, # (batch+1,)
    max_seqlen_q, # int
    max_seqlen_k, # int
    dropout_p=0.0,
    causal=False,
)
```

### Flash Attention 3 (FA3) — `flash-attn >= 3.0` (Hopper only)

FA3 is in the same package starting from version 3.x. It targets SM90 (H100) specifically:

```python
from flash_attn.flash_attn_interface import flash_attn_func  # same import

# FA3 is automatically used on SM90 GPUs when flash-attn >= 3.0 is installed
# The API is backward compatible with FA2
```

FA3 specific capabilities:
- FP8 support (e8m0, e4m3, e5m2 formats)
- Higher throughput via asynchronous WGMMA instructions
- Not available on pre-SM90 GPUs — falls back to FA2 automatically

---

## Supported Dtypes and Head Dimensions

### FA2 Support Matrix

| dtype | head_dim | Forward | Backward | Notes |
|-------|----------|---------|----------|-------|
| float16 | 32-256 | Yes | Yes | Best performance |
| bfloat16 | 32-256 | Yes | Yes | bf16 requires PyTorch >= 1.13 |
| float32 | any | No | No | Not supported — use SDPA Math |

Head dimensions must be in the set {32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 160, 192, 224, 256} for FA2. Non-standard dims (e.g., 48, 96) ARE supported in FA2, unlike PyTorch's built-in Flash backend which has a narrower allowed set.

### Backward Pass Constraints

- Gradient computation requires the same dtype as the forward pass
- `dropout_p > 0.0` requires the random number generator state to be saved during forward — this is handled automatically by `flash_attn_func` but requires the `return_softmax` or `rng_state` mechanism for custom gradient implementations

---

## Runtime Capability Detection

Always check capability before using the external package:

```python
def is_flash_attn_available() -> bool:
    try:
        import flash_attn
        return True
    except ImportError:
        return False

def get_flash_attn_version() -> str | None:
    try:
        import flash_attn
        return flash_attn.__version__
    except ImportError:
        return None

def check_flash_attn_device(device: torch.device) -> tuple[bool, str]:
    """Returns (can_use, reason)."""
    if device.type != "cuda":
        return False, f"flash-attn requires CUDA device, got {device.type}"

    props = torch.cuda.get_device_properties(device)
    sm = props.major * 10 + props.minor
    if sm < 80:
        return False, f"flash-attn requires SM80+ (Ampere), device is SM{sm}"

    return True, "OK"

def check_flash_attn_dtype(dtype: torch.dtype) -> tuple[bool, str]:
    """Returns (can_use, reason)."""
    supported = {torch.float16, torch.bfloat16}
    if dtype not in supported:
        return False, f"flash-attn only supports fp16/bf16, got {dtype}"
    return True, "OK"
```

---

## HuggingFace Integration

HuggingFace Transformers supports three attention implementations, controlled via `attn_implementation`:

```python
from transformers import AutoModel

# Default: eager (manual q@k.T softmax(v) chain)
model = AutoModel.from_pretrained("meta-llama/Llama-2-7b-hf")

# SDPA (built-in PyTorch): recommended baseline
model = AutoModel.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    attn_implementation="sdpa",
    torch_dtype=torch.bfloat16,
)

# External flash-attn: requires flash-attn package installed
model = AutoModel.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,  # Required: flash-attn only supports fp16/bf16
)
```

**Model compatibility**: Not all HuggingFace models support all `attn_implementation` values. The model class must implement the `_supports_flash_attn_2` property. Check with:

```python
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
print(hasattr(cfg, "_flash_attn_2_enabled"))  # Rough check
```

**Required dtype**: When using `attn_implementation="flash_attention_2"`, you MUST also specify `torch_dtype=torch.float16` or `torch.bfloat16`. The model will raise a `ValueError` if loaded in float32.

---

## Three Modes: off / prefer / require

### Mode: `off` (default)

```python
# external_flash_attn = "off"
# Ignore the flash-attn package entirely.
# All attention routes through SDPA.
# Safe on any platform.
```

### Mode: `prefer`

```python
# external_flash_attn = "prefer"
# Use flash-attn if available and constraints are met.
# Fall back to SDPA silently if not.

checker = FlashAttnChecker()
if checker.is_available():
    result = checker.check_constraints(head_dim=128, dtype=torch.float16, device=device)
    if result.available:
        # Use flash_attn_func
    else:
        # Log warning, fall back to SDPA
        logger.warning(f"flash-attn prefer mode: falling back to SDPA. Reason: {result.reason}")
else:
    logger.info("flash-attn not installed; using SDPA")
```

### Mode: `require`

```python
# external_flash_attn = "require"
# Raise an error if flash-attn is not available or constraints not met.
# Use this to catch configuration mistakes in production.

checker = FlashAttnChecker()
if not checker.is_available():
    raise RuntimeError(
        "flash-attn package required but not installed.\n"
        "Install: pip install flash-attn --no-build-isolation\n"
        "Requires: CUDA >= 11.6, Linux, Ampere+ GPU, PyTorch >= 1.12"
    )
result = checker.check_constraints(head_dim, dtype, device)
if not result.available:
    raise RuntimeError(f"flash-attn constraints not met: {result.reason}")
```

---

## Troubleshooting

### Install Failure: `nvcc: not found`

Cause: pip's build isolation environment does not have CUDA on PATH.
Fix: `pip install flash-attn --no-build-isolation`

### Install Failure: `undefined symbol: _ZN5torch...`

Cause: flash-attn compiled against different PyTorch version than installed.
Fix: Uninstall and reinstall, ensuring the PyTorch version matches.

```bash
pip uninstall flash-attn
pip install flash-attn --no-build-isolation
```

### Install Failure: `CUDA version mismatch`

Cause: `torch.version.cuda` and `nvcc --version` report different versions.
Fix: Install the CUDA toolkit matching your PyTorch's CUDA version, or use a Docker image with aligned versions.

### Runtime Error: `AssertionError: flash attention only supports sm80+`

Cause: Running on pre-Ampere GPU (V100, T4, RTX 20xx).
Fix: Use SDPA with `SDPBackend.EFFICIENT_ATTENTION` instead.

### Runtime Error: `TypeError: forward() got unexpected keyword argument 'attention_mask'`

Cause: HuggingFace model version mismatch — the model was updated to a new attention API.
Fix: Update `transformers` to the version that introduced `flash_attention_2` support for that model.

### Windows: Not Supported

The flash-attn package does not compile on Windows due to NVCC compiler limitations. On Windows:
- Use SDPA with `SDPBackend.FLASH_ATTENTION` (PyTorch's built-in — does work on Windows CUDA)
- Or use WSL2 to run in a Linux environment

### Slow Install (10-30 minutes)

Cause: Compiling CUDA kernels from source.
Fix: Use pre-built wheels from the GitHub releases page. Match PyTorch version, CUDA version, Python version, and ABI (cxx11 vs non-cxx11).

### Memory Error During Compilation

Cause: Parallel compilation exhausts RAM.
Fix:

```bash
MAX_JOBS=2 pip install flash-attn --no-build-isolation
```
