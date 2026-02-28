---
name: Fast Attention Path (PyTorch SDPA + optional FlashAttention-2)
description: >
  This skill should be used when the user asks to "enable SDPA",
  "use scaled dot product attention", "add FlashAttention", "switch attention backend",
  "audit attention implementation", "detect attention path", "force flash attention",
  "use memory-efficient attention", "enable cuDNN attention", "fix attention fallback",
  "SDPA backend selection", "attention capability check", "sdpa_kernel context manager",
  "attention dropout in eval", "is_causal vs attn_mask", "attention benchmark",
  "flash-attn package integration", "attention report", "log which backend SDPA picked",
  "reduce attention memory", "fused attention kernels", "attention profile",
  or needs guidance on PyTorch SDPA integration, backend selection,
  FlashAttention-2 package setup, attention auditing, or attention performance tuning.
version: 0.1.0
---

# Fast Attention Path (PyTorch SDPA + optional FlashAttention-2)

## Overview

Standardize all Transformer attention to route through `torch.nn.functional.scaled_dot_product_attention` (SDPA), control which fused backend runs (Flash / Efficient / cuDNN / Math) via config, and optionally integrate the external `flash-attn` package for stacks that need it. The design principle: **auto-fast by default, manually controllable when needed, never silently slow.**

SDPA is a dispatch layer — it selects among backend implementations at runtime based on tensor properties. Rather than hardcoding rules about head dimensions or dtypes, use PyTorch's own `can_use_*` capability checks, which stay correct across versions.

## Public Contract

### SDPAttention

Single attention function used everywhere.

```python
def sdpa_attention(
    q: Tensor, k: Tensor, v: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    backend_cfg: Optional[BackendConfig] = None,
) -> Tensor: ...
```

### AttentionAuditor

Detect current attention paths and probe SDPA capabilities.

```python
class AttentionAuditor:
    def scan_codebase(self, root: str) -> AuditReport: ...
    def probe_runtime(self, q: Tensor, k: Tensor, v: Tensor,
                      attn_mask: Optional[Tensor], dropout_p: float,
                      is_causal: bool) -> CapabilityReport: ...
```

### FlashAttnChecker

Runtime capability checks for external flash-attn package.

```python
class FlashAttnChecker:
    def is_available(self) -> bool: ...
    def check_constraints(self, head_dim: int, dtype: torch.dtype,
                          device: torch.device) -> CheckResult: ...
```

## Key Concepts

### SDPA Backend Dispatch

PyTorch provides four SDPA backends:

| Backend | Key | Strengths | Limitations |
|---------|-----|-----------|-------------|
| FlashAttention-2 | `FLASH_ATTENTION` | Fastest for long sequences, O(N) memory | head_dim constraints, fp16/bf16 only |
| Memory-Efficient | `EFFICIENT_ATTENTION` | Good fallback, flexible shapes | Slower than Flash |
| cuDNN | `CUDNN_ATTENTION` | Hardware-tuned on supported GPUs | Limited availability |
| Math (C++) | `MATH` | Always works, any shape/dtype | No fusion, O(N^2) memory |

### Backend Selection via sdpa_kernel

Use `torch.nn.attention.sdpa_kernel` context manager (preferred over global toggles):

```python
from torch.nn.attention import sdpa_kernel, SDPBackend
with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]):
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

### Backend Policy Mapping

| Config Value | Enabled Backends | Fallback |
|-------------|-----------------|----------|
| `auto` | All (SDPA chooses) | Always works |
| `flash` | FLASH + MATH | Math if Flash unavailable |
| `efficient` | EFFICIENT + MATH | Math if Efficient unavailable |
| `cudnn` | CUDNN + MATH | Math if cuDNN unavailable |
| `flash_or_efficient` | FLASH + EFFICIENT + CUDNN + MATH | Cascade down |
| `math` | MATH only | No fusion |

When `attn.force=true`, remove MATH from the list — fail loudly if fused kernel cannot run.

### Critical SDPA Rules

1. **Dropout in eval**: Always pass `dropout_p=(p if self.training else 0.0)`. SDPA applies dropout exactly as specified — passing >0 in eval wastes compute and corrupts inference.
2. **is_causal vs attn_mask**: If `is_causal=True`, `attn_mask` must be `None` (SDPA asserts this). Pick one, not both.
3. **Numeric differences**: Outputs differ slightly across backends (operation order changes with fusion). Expect and allow small deltas.
4. **Shape convention**: SDPA expects `(..., heads, seq_len, head_dim)`. If code uses `(batch, seq, heads, head_dim)`, permute at the boundary.

### Capability Probing (Not Guessing)

Avoid hardcoding "head_dim must be 64/128" rules. Use PyTorch's runtime checks:

```python
params = torch.backends.cuda.SDPAParams(q, k, v, attn_mask, dropout_p, is_causal)
can_flash = torch.backends.cuda.can_use_flash_attention(params, debug=True)
can_efficient = torch.backends.cuda.can_use_efficient_attention(params, debug=True)
can_cudnn = torch.backends.cuda.can_use_cudnn_attention(params, debug=True)
```

The `debug=True` flag prints exactly why a backend cannot run — invaluable for triage.

### External FlashAttention-2 Package

Usually unnecessary if SDPA's Flash backend works, but some stacks (HuggingFace `attn_implementation="flash_attention_2"`) call it directly.

Three modes via `attn.external_flash_attn`:
- `off` — ignore the package entirely (default)
- `prefer` — use if available, fall back to SDPA otherwise
- `require` — raise if not installed/compatible

Install: `pip install flash-attn --no-build-isolation` (requires CUDA >= 11.6, Linux, Ampere+ GPU).

### Codebase Audit

Static scan detects current attention patterns:
- `F.scaled_dot_product_attention` → SDPA path
- `xformers.ops.memory_efficient_attention` → xFormers path
- `flash_attn` imports → external FA path
- `q @ k.transpose(-2, -1)` / explicit softmax chain → eager path

Runtime probe runs one forward pass and logs capability checks.

### Benchmark Harness Integration

Extend existing bench output (perf-regression-gate skill) with:
- `attn.backend_policy` — config string
- `attn.sdpa_can_flash` / `can_efficient` / `can_cudnn` — bools
- `attn.head_dim`, `attn.dropout_p_train`, `attn.is_causal`, `attn.mask_kind`

This makes regressions actionable: immediately see if throughput dropped because attention silently fell back to Math.

## Configuration Surface

```python
@dataclass
class AttentionConfig:
    impl: str = "sdpa"                    # "auto" | "sdpa"
    backend: str = "auto"                 # "auto" | "flash" | "efficient" | "cudnn"
                                          # | "math" | "flash_or_efficient"
    force: bool = False                   # If true, no Math fallback
    log_backend: bool = True              # Log capability report at startup
    external_flash_attn: str = "off"      # "off" | "prefer" | "require"
    dropout_policy: str = "train_only"    # Always pass 0.0 in eval
```

## Done-When Gates

1. **Audit Works** — `AttentionAuditor.scan_codebase()` identifies attention patterns (SDPA / xFormers / flash-attn / eager) with file+line references. `probe_runtime()` produces a capability report using `can_use_*` helpers.
2. **SDPA Integration Correct** — `sdpa_attention()` routes through `F.scaled_dot_product_attention` with proper dropout handling (0.0 in eval) and is_causal/mask exclusivity enforced.
3. **Backend Selection Controllable** — Setting `backend=math` forces Math-only. Setting `backend=flash` with `force=true` on incompatible inputs raises a clear error with debug reasons. Default `auto` never crashes.

## Resources

### Reference Files
- **`references/sdpa-backends.md`** — SDPA backend details, dispatch logic, capability checks, sdpa_kernel vs global toggles, numeric differences
- **`references/flash-attn-package.md`** — External flash-attn install, version matrix, runtime checks, HuggingFace integration, FA2 vs FA3
- **`references/attention-audit.md`** — Static scan patterns, runtime probe design, report schema, hook-based detection
- **`references/benchmark-integration.md`** — Bench harness fields, regression triage, A/B backend comparison
- **`references/testing-matrix.md`** — Test scenarios for all components

### Asset Files
- **`assets/sdpa_attention_template.py`** — sdpa_attention wrapper, backend policy mapping, dropout guard, shape validation
- **`assets/attention_auditor_template.py`** — AttentionAuditor with static scan and runtime probe
- **`assets/flash_attn_checker_template.py`** — FlashAttnChecker with install detection, constraint checks, fallback logic
- **`assets/backend_config_template.py`** — AttentionConfig dataclass, BackendConfig, CapabilityReport, validation
- **`assets/bench_attn_fields_template.py`** — Benchmark harness attention fields extension

### Scripts
- **`scripts/validate_attention.py`** — Validates done-when gates
- **`scripts/gen_attention_tests.py`** — Generates 100+ pytest test cases
- **`scripts/attention_benchmark.py`** — Backend-vs-backend throughput comparison
