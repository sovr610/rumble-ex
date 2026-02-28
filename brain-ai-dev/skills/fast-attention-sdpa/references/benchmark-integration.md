# Benchmark Integration — Attention Fields Reference

## Overview

Attention backend selection has a direct and significant impact on training throughput and memory usage. Without explicit tracking of which backend ran, a throughput regression is ambiguous: did the model architecture change? Did the batch size change? Or did attention silently fall back from Flash to Math because of a shape constraint violation?

This reference describes how to extend the existing `perf-regression-gate` bench harness with attention-specific fields that make regressions immediately actionable.

---

## Fields to Add to the Bench Harness

All attention fields are nested under the `attn` key in `metrics.json`. This keeps the schema organized and allows the regression gate to check attention fields independently.

### Full Field Specification

```json
{
    "attn": {
        "backend_policy": "flash",
        "sdpa_can_flash": true,
        "sdpa_can_efficient": true,
        "sdpa_can_cudnn": false,
        "head_dim": 128,
        "num_heads": 16,
        "dropout_p_train": 0.1,
        "is_causal": true,
        "mask_kind": "none",
        "backend_actually_used": "flash",
        "external_flash_attn_mode": "off",
        "flash_attn_package_version": null,
        "probe_device": "cuda:0",
        "probe_dtype": "torch.bfloat16"
    }
}
```

### Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `attn.backend_policy` | string | The configured backend policy: `"auto"`, `"flash"`, `"efficient"`, `"cudnn"`, `"math"`, `"flash_or_efficient"` |
| `attn.sdpa_can_flash` | bool | Result of `can_use_flash_attention(probe_params)` — indicates Flash is capable on this hardware with this shape |
| `attn.sdpa_can_efficient` | bool | Result of `can_use_efficient_attention(probe_params)` |
| `attn.sdpa_can_cudnn` | bool | Result of `can_use_cudnn_attention(probe_params)` |
| `attn.head_dim` | int | Head dimension used by the model |
| `attn.num_heads` | int | Number of attention heads |
| `attn.dropout_p_train` | float | Dropout probability during training (should be 0.0 in eval) |
| `attn.is_causal` | bool | Whether causal (autoregressive) masking is used |
| `attn.mask_kind` | string | `"none"`, `"causal"`, `"float_additive"`, `"bool_mask"`, `"padding"` |
| `attn.backend_actually_used` | string | Runtime-detected backend (from profiler or hook) — may differ from policy |
| `attn.external_flash_attn_mode` | string | `"off"`, `"prefer"`, `"require"` |
| `attn.flash_attn_package_version` | string or null | Version of flash-attn if installed, else null |
| `attn.probe_device` | string | Device used for capability probe, e.g., `"cuda:0"` |
| `attn.probe_dtype` | string | dtype used for probe tensors |

---

## Integration Points with Existing BenchTrain

### `AttentionMetrics` Dataclass

Add this dataclass to the bench harness alongside existing `TrainMetrics`:

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class AttentionMetrics:
    # Policy fields (from config)
    backend_policy: str = "auto"
    external_flash_attn_mode: str = "off"
    flash_attn_package_version: Optional[str] = None

    # Capability fields (from runtime probe)
    sdpa_can_flash: bool = False
    sdpa_can_efficient: bool = False
    sdpa_can_cudnn: bool = False

    # Model fields (from model config/introspection)
    head_dim: int = 0
    num_heads: int = 0
    dropout_p_train: float = 0.0
    is_causal: bool = False
    mask_kind: str = "none"

    # Runtime detection
    backend_actually_used: Optional[str] = None
    probe_device: str = "cpu"
    probe_dtype: str = "torch.float32"
```

### Injecting into `metrics.json`

The bench harness collects a `dict` of metrics. Inject attention fields with a helper:

```python
def inject_attention_fields(metrics_dict: dict, attn: AttentionMetrics) -> dict:
    """Merge AttentionMetrics into an existing metrics dict under 'attn' key."""
    metrics_dict["attn"] = {
        "backend_policy": attn.backend_policy,
        "sdpa_can_flash": attn.sdpa_can_flash,
        "sdpa_can_efficient": attn.sdpa_can_efficient,
        "sdpa_can_cudnn": attn.sdpa_can_cudnn,
        "head_dim": attn.head_dim,
        "num_heads": attn.num_heads,
        "dropout_p_train": attn.dropout_p_train,
        "is_causal": attn.is_causal,
        "mask_kind": attn.mask_kind,
        "backend_actually_used": attn.backend_actually_used,
        "external_flash_attn_mode": attn.external_flash_attn_mode,
        "flash_attn_package_version": attn.flash_attn_package_version,
        "probe_device": attn.probe_device,
        "probe_dtype": attn.probe_dtype,
    }
    return metrics_dict
```

### Collection Point in BenchTrain

The attention metrics should be collected once per benchmark run (not per step), during the initialization phase after the model is built:

```python
class BenchTrain:
    def setup(self, model, cfg, sample_batch):
        # ... existing setup ...
        from fast_attention_sdpa.bench_attn_fields import collect_attention_metrics
        self.attn_metrics = collect_attention_metrics(model, cfg, sample_batch)

    def finalize(self, raw_metrics: dict) -> dict:
        # ... existing finalization ...
        raw_metrics = inject_attention_fields(raw_metrics, self.attn_metrics)
        return raw_metrics
```

---

## Regression Triage Workflow

When the regression gate fires (throughput dropped > threshold), the attention fields enable immediate triage:

### Step 1: Check if Backend Changed

```python
def check_backend_regression(baseline: dict, current: dict) -> list[str]:
    """Return list of attention-related regression warnings."""
    warnings = []
    b_attn = baseline.get("attn", {})
    c_attn = current.get("attn", {})

    if b_attn.get("backend_actually_used") != c_attn.get("backend_actually_used"):
        warnings.append(
            f"ATTENTION BACKEND CHANGED: "
            f"{b_attn.get('backend_actually_used')} -> "
            f"{c_attn.get('backend_actually_used')}"
        )

    if b_attn.get("sdpa_can_flash") and not c_attn.get("sdpa_can_flash"):
        warnings.append(
            "Flash backend was available in baseline but is NOT available now. "
            "Check PyTorch version, CUDA version, or head_dim changes."
        )

    return warnings
```

### Step 2: Check Capability Change

If `sdpa_can_flash` was True in the baseline but False in the regression:
- PyTorch was downgraded (older versions have stricter Flash constraints)
- head_dim changed (e.g., a refactor changed 128 to 96)
- dtype changed (float32 added for debugging and not reverted)
- GPU changed (deployment target changed)

### Step 3: A/B Backend Comparison

Run the same model with different backend configs to quantify the impact:

```bash
# Baseline: auto (Flash runs)
python bench.py --attn.backend=auto --output baseline.json

# Force Math to measure Flash speedup
python bench.py --attn.backend=math --output math_only.json

# Compare
python compare_bench.py baseline.json math_only.json
```

---

## A/B Backend Comparison Methodology

### Setup

For a valid A/B comparison:
1. Use the same model checkpoint
2. Use the same data (same random seed, same batch)
3. Vary only the backend config
4. Run on the same hardware instance (not different machine types)

### Metrics to Compare

| Metric | How to Collect | Expected Flash Speedup |
|--------|----------------|------------------------|
| `tokens_per_sec` | `(batch_size * seq_len * steps) / total_time` | 2–5× vs Math for S≥1024 |
| `step_time_ms` | `time.perf_counter()` around the forward+backward | Inversely proportional to tokens/sec |
| `peak_memory_mb` | `torch.cuda.max_memory_allocated() / 1e6` | Flash uses 4–10× less |
| `cuda_kernel_time_ms` | CUDA events around `F.scaled_dot_product_attention` | Captures pure kernel time |

### CUDA Event Timing

```python
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
end_event.record()
torch.cuda.synchronize()

kernel_time_ms = start_event.elapsed_time(end_event)
```

### Comparison Table Format

```
Backend Comparison: Llama-7B, A100, bf16, seq=2048, batch=4
======================================================================
Backend       | tokens/sec  | step_ms | peak_mem_MB | speedup_vs_math
--------------+-------------+---------+-------------+----------------
auto (flash)  |   47,832    |  34.2   |    4,021    |     4.2×
efficient     |   38,104    |  42.9   |    3,987    |     3.3×
math          |   11,380    | 140.7   |   18,432    |     1.0×
======================================================================
```

---

## Integration with Existing `metrics.json` Schema

The `perf-regression-gate` skill defines a schema with top-level keys. Extend it by adding `attn` as a new top-level key. This is backward compatible — older baselines without `attn` simply have `None` for all fields during comparison.

```json
{
    "run_id": "abc123",
    "timestamp": "2025-01-15T12:00:00Z",
    "model": "my_transformer",
    "hardware": "A100-80GB",
    "tokens_per_sec": 47832,
    "step_time_ms": 34.2,
    "peak_memory_mb": 4021,
    "loss": 2.34,
    "attn": {
        "backend_policy": "auto",
        "sdpa_can_flash": true,
        "sdpa_can_efficient": true,
        "sdpa_can_cudnn": false,
        "head_dim": 128,
        "num_heads": 32,
        "dropout_p_train": 0.0,
        "is_causal": true,
        "mask_kind": "causal",
        "backend_actually_used": "flash",
        "external_flash_attn_mode": "off",
        "flash_attn_package_version": null,
        "probe_device": "cuda:0",
        "probe_dtype": "torch.bfloat16"
    }
}
```

The regression gate checks `attn.backend_actually_used` (or `attn.sdpa_can_flash` as a proxy) and fires an additional warning if the backend changed, even if the throughput delta is within tolerance.
