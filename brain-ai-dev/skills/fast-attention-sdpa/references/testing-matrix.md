# Testing Matrix — Fast Attention SDPA Skill

## Overview

This matrix defines the complete set of tests for the fast-attention-sdpa skill, organized by phase. Each phase tests a distinct component. Tests are runnable via `pytest tests/test_fast_attention.py` after running `python scripts/gen_attention_tests.py`.

All phases include CPU-compatible tests that run without a CUDA device, plus CUDA-conditional tests that skip on CPU-only machines.

---

## Phase 0: Audit Scan Detection

Tests for `AttentionAuditor.scan_codebase()`.

### P0-1: Pattern Detection for Each Type

| Test ID | Input | Expected |
|---------|-------|----------|
| P0-1a | File containing `F.scaled_dot_product_attention(q, k, v)` | Match: pattern_name=`sdpa_functional`, correct file+line |
| P0-1b | File containing `torch.nn.functional.scaled_dot_product_attention(...)` | Match: pattern_name=`sdpa_torch_nn_functional` |
| P0-1c | File containing `xformers.ops.memory_efficient_attention(...)` | Match: pattern_name=`xformers_memory_efficient` |
| P0-1d | File containing `from xformers.ops import memory_efficient_attention` | Match: pattern_name=`xformers_mem_eff_import` |
| P0-1e | File containing `flash_attn_func(q, k, v)` | Match: pattern_name=`flash_attn_func` |
| P0-1f | File containing `flash_attn_qkvpacked_func(qkv)` | Match: pattern_name=`flash_attn_qkvpacked` |
| P0-1g | File containing `from flash_attn import flash_attn_func` | Match: pattern_name=`flash_attn_import` |
| P0-1h | File containing `q @ k.transpose(-2, -1)` | Match: pattern_name=`eager_matmul_transpose` |
| P0-1i | File containing `torch.matmul(q, k.transpose(-2, -1))` | Match: pattern_name=`eager_torch_matmul` |
| P0-1j | File containing `einsum("bhld,bhsd->bhls", q, k)` | Match: pattern_name=`eager_einsum_bhld_bhsd` |
| P0-1k | File with no attention patterns at all | AuditReport.matches is empty |
| P0-1l | File with multiple patterns (SDPA + eager legacy code) | Both patterns reported with correct line numbers |

### P0-2: Scan Mechanics

| Test ID | Input | Expected |
|---------|-------|----------|
| P0-2a | Scan a temp directory with 3 .py files | files_scanned == 3 |
| P0-2b | Directory with no .py files | files_scanned == 0, no crash |
| P0-2c | File with syntax error (non-UTF8 bytes) | Scan completes without exception, file may be skipped |
| P0-2d | Nested subdirectory structure | All .py files at all depths are found |
| P0-2e | extensions=['.py', '.pyi'] | Both extension types are scanned |

### P0-3: Runtime Probe (CPU)

| Test ID | Input | Expected |
|---------|-------|----------|
| P0-3a | CPU tensors, probe_runtime() called | Returns CapabilityReport, no crash |
| P0-3b | CPU probe | can_flash=False, can_efficient=False, can_cudnn=False |
| P0-3c | CPU probe | debug_reasons dict has entries for each backend |
| P0-3d | CPU probe | device field contains "cpu" |

### P0-4: Runtime Probe (CUDA, if available)

| Test ID | Input | Expected |
|---------|-------|----------|
| P0-4a | CUDA tensors, fp16 | CapabilityReport returned with device="cuda:0" |
| P0-4b | CUDA tensors, fp16, Ampere+ | can_flash=True (assuming SM80+) |
| P0-4c | CUDA tensors, float32 | can_flash=False (Flash requires fp16/bf16) |
| P0-4d | debug_reasons populated | Reasons are non-empty strings for False results |

### P0-5: Report Serialization

| Test ID | Input | Expected |
|---------|-------|----------|
| P0-5a | save_report(report, path) | JSON file created at path |
| P0-5b | Load saved JSON | Parses without error, all fields present |
| P0-5c | AuditReport with 0 matches | JSON has empty matches list |

---

## Phase 1: SDPA Wrapper Correctness

Tests for `sdpa_attention()` and `SDPAModule`.

### P1-1: Basic Forward Pass

| Test ID | Input | Expected |
|---------|-------|----------|
| P1-1a | (2,4,16,64) fp32 tensors on CPU | Output shape (2,4,16,64) |
| P1-1b | batch=1, heads=1, seq=1, head_dim=32 | Output shape (1,1,1,32) |
| P1-1c | Large batch (8,16,512,128) | No OOM, correct output shape |
| P1-1d | is_causal=True, no mask | No error, output shape correct |

### P1-2: Dropout Guard

| Test ID | Input | Expected |
|---------|-------|----------|
| P1-2a | dropout_p=0.1, training=True | Succeeds |
| P1-2b | dropout_p=0.1, training=False | AssertionError raised |
| P1-2c | dropout_p=0.0, training=False | Succeeds |
| P1-2d | dropout_p=0.0, training=True | Succeeds |
| P1-2e | SDPAModule.forward() in train mode | Module.training=True → dropout applied |
| P1-2f | SDPAModule forward, module.train(False) | Module.training=False → dropout_p set to 0.0 |

### P1-3: is_causal / attn_mask Exclusivity

| Test ID | Input | Expected |
|---------|-------|----------|
| P1-3a | is_causal=True, attn_mask=None | Succeeds |
| P1-3b | is_causal=False, attn_mask=bool_mask | Succeeds |
| P1-3c | is_causal=True, attn_mask=bool_mask | ValueError or RuntimeError raised |
| P1-3d | is_causal=True, attn_mask=float_mask | ValueError or RuntimeError raised |

### P1-4: Shape Handling and permute_for_sdpa

| Test ID | Input | Expected |
|---------|-------|----------|
| P1-4a | permute_for_sdpa(x, "BSH D") (B,S,H,D) → (B,H,S,D) | Shape transposed correctly |
| P1-4b | permute_for_sdpa(x, "BHSD") (B,H,S,D) → (B,H,S,D) | No-op, shape unchanged |
| P1-4c | Round-trip permute → sdpa → permute-back | Output matches manual attention |
| P1-4d | head_dim=64 input | Correct output |
| P1-4e | head_dim=128 input | Correct output |

### P1-5: Numerical Correctness (CPU/Math)

| Test ID | Input | Expected |
|---------|-------|----------|
| P1-5a | Compare sdpa_attention(math) vs manual (q@k.T softmax v) | allclose within fp32 tolerance |
| P1-5b | is_causal=True vs manual causal mask | allclose |
| P1-5c | Boolean mask applied | allclose to masked manual computation |

---

## Phase 2: Backend Selection

Tests for `BackendPolicy` enum and `sdpa_kernel` usage in `sdpa_attention`.

### P2-1: Each Policy

| Test ID | Config | Expected |
|---------|--------|----------|
| P2-1a | backend="auto" | No crash, output shape correct |
| P2-1b | backend="math" | Uses MATH backend (verify via profiler or log) |
| P2-1c | backend="flash" (CPU) | Math fallback (Flash not available on CPU) |
| P2-1d | backend="efficient" (CPU) | Math fallback |
| P2-1e | backend="cudnn" (CPU) | Math fallback |
| P2-1f | backend="flash_or_efficient" | No crash |

### P2-2: Force Mode

| Test ID | Config | Expected |
|---------|--------|----------|
| P2-2a | force=True, backend="math" | Succeeds (Math is always available) |
| P2-2b | force=True, backend="flash" on CPU | RuntimeError with diagnostic message |
| P2-2c | force=True, backend="flash" on CUDA Ampere+ bf16 | Succeeds |
| P2-2d | force=True, backend="flash" on CUDA with float32 | RuntimeError (Flash requires fp16/bf16) |

### P2-3: Fallback Behavior

| Test ID | Config | Expected |
|---------|--------|----------|
| P2-3a | backend="flash", force=False on CPU | Returns result using Math fallback |
| P2-3b | backend="efficient", force=False on CPU | Returns result using Math fallback |
| P2-3c | Auto mode on CPU | Returns result using Math |

### P2-4: BackendConfig Construction

| Test ID | Input | Expected |
|---------|-------|----------|
| P2-4a | `from_attention_config(AttentionConfig(backend="flash"))` | backends=[FLASH, MATH], force=False |
| P2-4b | `from_attention_config(AttentionConfig(backend="math"))` | backends=[MATH] |
| P2-4c | `from_attention_config(AttentionConfig(force=True, backend="flash"))` | backends=[FLASH], force=True |

---

## Phase 3: External Flash-Attn Checker

Tests for `FlashAttnChecker`.

### P3-1: Availability Detection

| Test ID | Setup | Expected |
|---------|-------|----------|
| P3-1a | flash_attn not installed (mocked) | is_available() returns False |
| P3-1b | flash_attn installed (mocked) | is_available() returns True |
| P3-1c | flash_attn not installed | get_version() returns None |
| P3-1d | flash_attn installed, version "2.4.0" | get_version() returns "2.4.0" |

### P3-2: Constraint Checks

| Test ID | Input | Expected |
|---------|-------|----------|
| P3-2a | head_dim=64, fp16, CUDA SM80 | CheckResult.available=True |
| P3-2b | head_dim=64, float32 | CheckResult.available=False, reason mentions "fp32" |
| P3-2c | head_dim=64, fp16, CPU | CheckResult.available=False, reason mentions "CUDA" |
| P3-2d | head_dim=512, fp16, CUDA | CheckResult.available=False, reason mentions "head_dim" |
| P3-2e | head_dim=256, fp16, CUDA SM80 | CheckResult.available=True (max supported) |
| P3-2f | CUDA SM70 (V100) | CheckResult.available=False, reason mentions "SM80" |

### P3-3: Mode Resolution

| Test ID | Mode | flash available | Expected |
|---------|------|----------------|----------|
| P3-3a | "off" | any | Returns "sdpa" |
| P3-3b | "prefer" | True + constraints met | Returns "flash_attn" |
| P3-3c | "prefer" | False | Returns "sdpa", logs warning |
| P3-3d | "prefer" | True + constraints fail | Returns "sdpa", logs warning |
| P3-3e | "require" | True + constraints met | Returns "flash_attn" |
| P3-3f | "require" | False | Raises RuntimeError with install instructions |
| P3-3g | "require" | True + constraints fail | Raises RuntimeError with reason |

---

## Phase 4: Capability Reporting

Tests for `CapabilityReport` and `AttentionConfig`.

### P4-1: AttentionConfig Validation

| Test ID | Input | Expected |
|---------|-------|----------|
| P4-1a | Default AttentionConfig() | validate() passes |
| P4-1b | impl="invalid" | validate() raises ValueError |
| P4-1c | backend="nonexistent" | validate() raises ValueError |
| P4-1d | external_flash_attn="invalid" | validate() raises ValueError |
| P4-1e | dropout_policy="invalid" | validate() raises ValueError |

### P4-2: Serialization Roundtrip

| Test ID | Input | Expected |
|---------|-------|----------|
| P4-2a | cfg.to_dict() then from_dict(d) | Roundtripped config equals original |
| P4-2b | Config with non-default values | All fields preserved after roundtrip |
| P4-2c | JSON serialization of to_dict() | Valid JSON, no type errors |

### P4-3: CapabilityReport Population

| Test ID | Input | Expected |
|---------|-------|----------|
| P4-3a | CPU probe | All boolean fields present, device_name not empty |
| P4-3b | CPU probe | debug_reasons dict has three keys: flash, efficient, cudnn |
| P4-3c | CPU probe debug_reasons | Each value is a list (possibly empty) of strings |
| P4-3d | recommended_backend() on CPU | Returns "math" |
| P4-3e | CUDA SM80+ fp16 | recommended_backend() returns "flash" |

---

## Phase 5: Benchmark Integration

Tests for `bench_attn_fields_template.py`.

### P5-1: Collection

| Test ID | Input | Expected |
|---------|-------|----------|
| P5-1a | collect_attention_metrics with SDPAModule | Returns AttentionMetrics |
| P5-1b | Model has head_dim=64 | attn_metrics.head_dim == 64 |
| P5-1c | Model has num_heads=8 | attn_metrics.num_heads == 8 |
| P5-1d | No CUDA available | Capability fields are False, no crash |

### P5-2: Injection

| Test ID | Input | Expected |
|---------|-------|----------|
| P5-2a | inject_attention_fields(metrics, attn) | metrics["attn"] key present |
| P5-2b | All required subfields | backend_policy, sdpa_can_flash, head_dim all present |
| P5-2c | inject into non-empty metrics | Existing keys preserved |

### P5-3: Formatting

| Test ID | Input | Expected |
|---------|-------|----------|
| P5-3a | format_attention_report(attn_metrics) | Returns non-empty string |
| P5-3b | Backend policy in output | String contains backend_policy value |

### P5-4: Regression Detection

| Test ID | Input | Expected |
|---------|-------|----------|
| P5-4a | Baseline has can_flash=True, current has can_flash=False | check_backend_regression returns warnings |
| P5-4b | Both have same backend | No warnings returned |

---

## Edge Cases

### EC-1: CPU-Only Environment

| Test ID | Input | Expected |
|---------|-------|----------|
| EC-1a | All tests that require CUDA | Marked `@pytest.mark.skipif(not cuda_available)` |
| EC-1b | sdpa_attention with Math only | Runs correctly, produces valid output |
| EC-1c | Audit probe on CPU | CapabilityReport produced, all can_* = False |

### EC-2: Non-Contiguous Tensors

| Test ID | Input | Expected |
|---------|-------|----------|
| EC-2a | Non-contiguous q from `.transpose(0, 1)` | sdpa_attention handles without error |
| EC-2b | Sliced tensor (non-contiguous) | sdpa_attention handles without error |

### EC-3: Unusual Head Dimensions

| Test ID | head_dim | Expected |
|---------|----------|----------|
| EC-3a | head_dim=32 | Correct output shape |
| EC-3b | head_dim=96 | Correct output shape (possibly Math fallback on Flash) |
| EC-3c | head_dim=256 | Correct output shape |
| EC-3d | head_dim=1 | Correct output shape (degenerate case) |

### EC-4: Edge Sequence Lengths

| Test ID | Input | Expected |
|---------|-------|----------|
| EC-4a | seq_len=1 | Correct output |
| EC-4b | seq_len=0 | Either correct empty output or clear error |
| EC-4c | batch_size=1 | Correct output |

### EC-5: Mixed Precision

| Test ID | Input | Expected |
|---------|-------|----------|
| EC-5a | q in fp16, k in fp16, v in fp16 | Correct output in fp16 |
| EC-5b | q in bf16 | Correct output in bf16 |
| EC-5c | fp32 on CUDA | Math fallback (Flash requires fp16/bf16), correct output |

---

## Test Infrastructure Notes

### Parametrization Strategy

Use `pytest.mark.parametrize` for the combinatorial cases:

```python
@pytest.mark.parametrize("head_dim", [32, 64, 96, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_sdpa_shapes(head_dim, dtype):
    ...
```

### CUDA Skip Decorator

```python
import pytest
import torch

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)

requires_ampere = pytest.mark.skipif(
    not (torch.cuda.is_available() and
         torch.cuda.get_device_capability()[0] >= 8),
    reason="Ampere GPU (SM80+) required"
)
```

### Tolerance Constants

```python
TOLERANCES = {
    torch.float32: {"atol": 1e-5, "rtol": 1e-5},
    torch.float16: {"atol": 1e-2, "rtol": 1e-2},
    torch.bfloat16: {"atol": 1e-1, "rtol": 1e-1},
}
```
