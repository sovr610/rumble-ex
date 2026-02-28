# Attention Audit — Static Scan and Runtime Probe Reference

## Overview

An attention audit answers two questions:

1. **Static**: "What attention implementation is this codebase using?" — answered by scanning Python source files for known patterns.
2. **Runtime**: "Can fused backends run on this machine with these inputs?" — answered by constructing `SDPAParams` and running `can_use_*` checks.

The audit is the first step in any attention migration. You cannot safely switch to SDPA without knowing what you're replacing.

---

## Static Scan: Regex Patterns

The `ATTENTION_PATTERNS` dictionary maps a descriptive name to a compiled regex. A match indicates which attention implementation is in use.

### SDPA Patterns

```python
import re

ATTENTION_PATTERNS = {
    # PyTorch SDPA — the target state
    "sdpa_functional": re.compile(
        r"F\.scaled_dot_product_attention\s*\("
    ),
    "sdpa_torch_nn_functional": re.compile(
        r"torch\.nn\.functional\.scaled_dot_product_attention\s*\("
    ),
    "sdpa_aten": re.compile(
        r"torch\._C\._nn\.scaled_dot_product_attention\s*\("
    ),
```

Seeing `sdpa_functional` or `sdpa_torch_nn_functional` means the file already uses SDPA. These patterns confirm the target state rather than identifying migration candidates.

### xFormers Patterns

```python
    # xFormers memory-efficient attention
    "xformers_memory_efficient": re.compile(
        r"xformers\.ops\.memory_efficient_attention\s*\("
    ),
    "xformers_mem_eff_import": re.compile(
        r"from\s+xformers\.ops\s+import\s+memory_efficient_attention"
    ),
    "xformers_ops_import": re.compile(
        r"import\s+xformers\.ops"
    ),
```

xFormers attention predates SDPA. Models using it typically have a conditional path (`if has_xformers`) that can be replaced with `sdpa_kernel([EFFICIENT_ATTENTION])`.

### External Flash Attention Patterns

```python
    # Tri Dao's flash-attn package
    "flash_attn_func": re.compile(
        r"flash_attn_func\s*\("
    ),
    "flash_attn_qkvpacked": re.compile(
        r"flash_attn_qkvpacked_func\s*\("
    ),
    "flash_attn_kvpacked": re.compile(
        r"flash_attn_kvpacked_func\s*\("
    ),
    "flash_attn_varlen": re.compile(
        r"flash_attn_varlen_func\s*\("
    ),
    "flash_attn_import": re.compile(
        r"from\s+flash_attn\s+import"
    ),
    "flash_attn_module_import": re.compile(
        r"import\s+flash_attn"
    ),
```

### Eager (Manual) Attention Patterns

These are the primary migration targets — code that computes attention without any fusion:

```python
    # Explicit matrix multiplication patterns
    "eager_matmul_transpose": re.compile(
        r"q\s*@\s*k\.transpose\s*\(\s*-2\s*,\s*-1\s*\)"
    ),
    "eager_torch_matmul": re.compile(
        r"torch\.matmul\s*\(\s*\w+\s*,\s*\w+\.transpose\s*\(\s*-2\s*,\s*-1\s*\)\s*\)"
    ),
    "eager_bmm": re.compile(
        r"torch\.bmm\s*\(\s*\w+\s*,\s*\w+\.transpose\s*\("
    ),
    # Einstein summation attention
    "eager_einsum_bhld_bhsd": re.compile(
        r'einsum\s*\(\s*["\']bhld,bhsd->bhls["\']'
    ),
    "eager_einsum_bqhd_bkhd": re.compile(
        r'einsum\s*\(\s*["\']bqhd,bkhd->bhqk["\']'
    ),
    "eager_einsum_generic_attn": re.compile(
        r'einsum\s*\(\s*["\'][^"\']*ij[^"\']*,[^"\']*kj[^"\']*->.*["\']'
    ),
    # Explicit softmax chains (the clearest sign of eager attention)
    "eager_softmax_chain": re.compile(
        r"F\.softmax\s*\(\s*.*\*\s*(?:math\.sqrt|self\._scale|scale|"
        r"1\s*/\s*math\.sqrt|\(.*\)\s*\*\*\s*-0\.5)",
        re.DOTALL,
    ),
    "eager_softmax_explicit": re.compile(
        r"torch\.softmax\s*\(\s*\w+\s*,\s*dim\s*=\s*-1\s*\)"
    ),
    # Attention weight computation
    "eager_attn_weights": re.compile(
        r"attn_weights\s*=\s*(?:torch\.)?(?:matmul|bmm|einsum)"
    ),
    "eager_scores_variable": re.compile(
        r"(?:attention_scores|attn_scores)\s*=\s*\w+\s*@\s*\w+"
    ),
}
```

### Usage in Scan

```python
def scan_file(filepath: str) -> list[dict]:
    """Scan a single file, return list of match records."""
    matches = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    for lineno, line in enumerate(lines, start=1):
        for pattern_name, pattern in ATTENTION_PATTERNS.items():
            if pattern.search(line):
                matches.append({
                    "file": filepath,
                    "line": lineno,
                    "pattern_name": pattern_name,
                    "snippet": line.rstrip(),
                })
    return matches
```

### Multi-Line Pattern Considerations

Some eager attention patterns span multiple lines (e.g., the attn score computation and softmax are on separate lines). The single-line regex catches most cases in practice, but for high-confidence scanning, also search for the co-occurrence of these tokens within a 20-line window:

```python
def scan_multiline_eager(filepath: str, window: int = 20) -> list[dict]:
    """Detect multi-line eager attention patterns."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    results = []
    for i, line in enumerate(lines):
        if re.search(r"\.transpose\(-2,\s*-1\)", line):
            # Check nearby lines for softmax
            start = max(0, i - window)
            end = min(len(lines), i + window)
            window_text = "".join(lines[start:end])
            if re.search(r"softmax", window_text):
                results.append({
                    "file": filepath,
                    "line": i + 1,
                    "pattern_name": "eager_multiline_attn",
                    "snippet": line.rstrip(),
                })
    return results
```

---

## Runtime Probe Design

The runtime probe constructs a minimal attention forward pass on representative tensors and runs PyTorch's capability checks.

### Forward Hook Design

For models you don't own (HuggingFace models, third-party), use forward hooks to intercept the attention inputs:

```python
import torch
from torch import Tensor
from typing import Optional

class AttentionProbeHook:
    """Registers hooks on attention modules to capture q, k, v tensors."""

    def __init__(self):
        self.captured_params = []
        self._handles = []

    def register(self, module: torch.nn.Module):
        """Register hooks on all attention-like submodules."""
        for name, submodule in module.named_modules():
            if self._looks_like_attention(submodule):
                h = submodule.register_forward_hook(self._hook_fn)
                self._handles.append(h)

    def _looks_like_attention(self, module: torch.nn.Module) -> bool:
        class_name = type(module).__name__.lower()
        return any(kw in class_name for kw in ["attention", "attn", "selfattn"])

    def _hook_fn(self, module, inputs, outputs):
        # Heuristic: first three inputs are often q, k, v
        if len(inputs) >= 3:
            self.captured_params.append({
                "module": type(module).__name__,
                "q_shape": inputs[0].shape if hasattr(inputs[0], "shape") else None,
                "k_shape": inputs[1].shape if hasattr(inputs[1], "shape") else None,
                "v_shape": inputs[2].shape if hasattr(inputs[2], "shape") else None,
                "dtype": inputs[0].dtype if hasattr(inputs[0], "dtype") else None,
                "device": str(inputs[0].device) if hasattr(inputs[0], "device") else None,
            })

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
```

### Representative Batch Creation

For the probe to be meaningful, tensors should match the model's expected shapes:

```python
def make_probe_tensors(
    batch_size: int = 2,
    num_heads: int = 8,
    seq_len: int = 512,
    head_dim: int = 64,
    dtype: torch.dtype = torch.float16,
    device: torch.device = torch.device("cpu"),
) -> tuple[Tensor, Tensor, Tensor]:
    """Create representative q, k, v tensors for capability probing."""
    shape = (batch_size, num_heads, seq_len, head_dim)
    q = torch.randn(*shape, dtype=dtype, device=device)
    k = torch.randn(*shape, dtype=dtype, device=device)
    v = torch.randn(*shape, dtype=dtype, device=device)
    return q, k, v
```

### SDPAParams Construction

```python
def build_sdpa_params(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> "torch.backends.cuda.SDPAParams | None":
    """Build SDPAParams if on CUDA, else return None."""
    if q.device.type != "cuda":
        return None
    try:
        return torch.backends.cuda.SDPAParams(q, k, v, attn_mask, dropout_p, is_causal)
    except Exception:
        return None
```

---

## Report Schema

### `AuditReport` Dataclass

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class AuditMatch:
    file: str          # Absolute path to the file
    line: int          # 1-indexed line number
    pattern_name: str  # Key from ATTENTION_PATTERNS
    snippet: str       # The matching line (stripped)

@dataclass
class AuditReport:
    root: str                        # Scanned directory root
    files_scanned: int               # Total .py files examined
    matches: List[AuditMatch] = field(default_factory=list)

    @property
    def has_sdpa(self) -> bool:
        return any(m.pattern_name.startswith("sdpa_") for m in self.matches)

    @property
    def has_eager(self) -> bool:
        return any(m.pattern_name.startswith("eager_") for m in self.matches)

    @property
    def has_xformers(self) -> bool:
        return any(m.pattern_name.startswith("xformers_") for m in self.matches)

    @property
    def has_external_flash(self) -> bool:
        return any(m.pattern_name.startswith("flash_attn_") for m in self.matches)

    def summary(self) -> str:
        lines = [
            f"Audit of: {self.root}",
            f"Files scanned: {self.files_scanned}",
            f"Total matches: {len(self.matches)}",
            f"  SDPA:          {self.has_sdpa}",
            f"  Eager:         {self.has_eager}",
            f"  xFormers:      {self.has_xformers}",
            f"  External FA:   {self.has_external_flash}",
        ]
        return "\n".join(lines)
```

### `CapabilityReport` Dataclass

```python
@dataclass
class CapabilityReport:
    # Device info
    device: str              # e.g., "cuda:0", "cpu"
    device_name: str         # e.g., "NVIDIA A100-SXM4-80GB"
    cuda_capability: str     # e.g., "8.0"

    # Built-in Flash backend
    flash_built_in: bool     # True if PyTorch was compiled with Flash support

    # Capability check results
    can_flash: bool          # SDPBackend.FLASH_ATTENTION
    can_efficient: bool      # SDPBackend.EFFICIENT_ATTENTION
    can_cudnn: bool          # SDPBackend.CUDNN_ATTENTION
    # Math always works, no need to check

    # Debug reasons for False results
    debug_reasons: dict      # {backend_name: [reason_lines]}

    # Probe input shapes
    q_shape: tuple
    k_shape: tuple
    v_shape: tuple
    dtype: str
    dropout_p: float
    is_causal: bool

    def recommended_backend(self) -> str:
        if self.can_flash:
            return "flash"
        if self.can_efficient:
            return "efficient"
        if self.can_cudnn:
            return "cudnn"
        return "math"
```

---

## `attn_report.json` Format Specification

The `save_report` method serializes both the audit and capability reports to a single JSON file:

```json
{
    "schema_version": "1.0",
    "generated_at": "2025-01-15T12:34:56Z",
    "audit": {
        "root": "/path/to/codebase",
        "files_scanned": 42,
        "summary": {
            "has_sdpa": true,
            "has_eager": false,
            "has_xformers": false,
            "has_external_flash": false
        },
        "matches": [
            {
                "file": "/path/to/codebase/model/attention.py",
                "line": 87,
                "pattern_name": "sdpa_functional",
                "snippet": "    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)"
            }
        ]
    },
    "capability": {
        "device": "cuda:0",
        "device_name": "NVIDIA A100-SXM4-80GB",
        "cuda_capability": "8.0",
        "flash_built_in": true,
        "can_flash": true,
        "can_efficient": true,
        "can_cudnn": false,
        "debug_reasons": {
            "flash": [],
            "efficient": [],
            "cudnn": ["cuDNN version 8.5 does not support this configuration"]
        },
        "probe_inputs": {
            "q_shape": [2, 8, 512, 64],
            "k_shape": [2, 8, 512, 64],
            "v_shape": [2, 8, 512, 64],
            "dtype": "torch.float16",
            "dropout_p": 0.0,
            "is_causal": false
        },
        "recommended_backend": "flash"
    }
}
```

---

## Interpreting `can_use_*` Debug Output

The `debug=True` output from `can_use_flash_attention` prints to stderr. Common messages and their meanings:

### Flash Attention Debug Messages

| Message | Cause | Fix |
|---------|-------|-----|
| `"Flash attention only supports sm80 or above"` | GPU is Volta/Turing | Use `efficient` backend |
| `"Flash attention only supports fp16 and bf16"` | Input is float32 | Cast to bf16 or use `efficient` |
| `"Requires q,k,v to be contig in the last dimension"` | Non-contiguous tensor | Call `.contiguous()` before SDPA |
| `"Flash attention requires head_dim <= 256"` | head_dim too large | Reduce head_dim or use `efficient` |
| `"If attn_mask is provided, it must be a boolean tensor"` | Float mask provided | Convert or use `is_causal` |
| `"Both attn_mask and is_causal cannot be set"` | API misuse | Remove one parameter |

### Efficient Attention Debug Messages

| Message | Cause | Fix |
|---------|-------|-----|
| `"Efficient attention only supports sm70 or above"` | Ancient GPU | Use math backend |
| `"Dropout is not supported for efficient attention backward"` | dropout > 0 in backward | Set dropout=0 or disable backward |

### Capturing and Parsing Debug Output

```python
import io
import contextlib
import re

def capture_debug_reasons(
    params: "torch.backends.cuda.SDPAParams",
) -> dict:
    """Run all can_use_* checks and capture their debug messages."""
    reasons = {"flash": [], "efficient": [], "cudnn": []}

    for backend, check_fn in [
        ("flash",     torch.backends.cuda.can_use_flash_attention),
        ("efficient", torch.backends.cuda.can_use_efficient_attention),
        ("cudnn",     torch.backends.cuda.can_use_cudnn_attention),
    ]:
        buf = io.StringIO()
        try:
            with contextlib.redirect_stderr(buf):
                check_fn(params, debug=True)
        except Exception as e:
            reasons[backend] = [str(e)]
        stderr_output = buf.getvalue().strip()
        if stderr_output:
            # Split into individual reason lines
            reasons[backend] = [
                line.strip() for line in stderr_output.splitlines()
                if line.strip()
            ]

    return reasons
```

---

## Hook-Based Detection for Models You Don't Own

When auditing a HuggingFace model or third-party module where you cannot inspect the source, use activation hooks to detect the attention path at runtime:

```python
def detect_attention_path(model: torch.nn.Module, sample_input: dict) -> str:
    """
    Run one forward pass and detect which attention path was used.
    Returns: "sdpa", "eager", "unknown"
    """
    sdpa_called = [False]
    eager_called = [False]

    # Patch F.scaled_dot_product_attention temporarily
    original_sdpa = torch.nn.functional.scaled_dot_product_attention

    def patched_sdpa(*args, **kwargs):
        sdpa_called[0] = True
        return original_sdpa(*args, **kwargs)

    torch.nn.functional.scaled_dot_product_attention = patched_sdpa
    try:
        model.train(False)
        with torch.no_grad():
            model(**sample_input)
    finally:
        torch.nn.functional.scaled_dot_product_attention = original_sdpa

    if sdpa_called[0]:
        return "sdpa"
    return "unknown"
```

This technique works because `F.scaled_dot_product_attention` is a Python function that can be monkeypatched. Eager implementations that use raw `@` operator or `torch.matmul` will not trigger the patch, revealing they are not using SDPA.
