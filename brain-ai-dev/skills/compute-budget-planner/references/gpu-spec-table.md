# GPU Specification Reference Table

## Overview

This reference documents peak TFLOPS, memory, and related specifications for GPUs commonly used in LLM training. The planner's `GPUSpecTable` class uses these values as its lookup database.

**Critical note on sparse vs. dense specs**: NVIDIA's official product pages frequently advertise "with sparsity" (2:4 structured sparsity) TFLOPS figures that are 2x the dense figures. The planner **defaults to DENSE figures** unless the user explicitly specifies sparse training. Always check which figure you are using.

---

## 1. GPU Specifications by Model

### NVIDIA A100

#### A100 80GB SXM (HBM2e)
| Precision | Dense TFLOPS | Sparse TFLOPS |
|-----------|-------------|---------------|
| FP64 | 9.7 | — |
| TF32 | 156 | 312 |
| BF16 | 312 | 624 |
| FP16 | 312 | 624 |
| INT8 | 624 | 1248 |

- **Memory**: 80 GB HBM2e
- **Memory Bandwidth**: 2,039 GB/s
- **NVLink**: NVLink 3.0, 600 GB/s bidirectional
- **TDP**: 400W
- **Compute Capability**: 8.0

#### A100 80GB PCIe
| Precision | Dense TFLOPS | Sparse TFLOPS |
|-----------|-------------|---------------|
| BF16 | 312 | 624 |
| FP16 | 312 | 624 |
| TF32 | 156 | 312 |

- **Memory**: 80 GB HBM2e
- **Memory Bandwidth**: 1,935 GB/s
- **Interconnect**: PCIe 4.0 x16
- **TDP**: 300W
- **Compute Capability**: 8.0

#### A100 40GB SXM
| Precision | Dense TFLOPS |
|-----------|-------------|
| BF16 | 312 |
| FP16 | 312 |
| TF32 | 156 |

- **Memory**: 40 GB HBM2
- **Memory Bandwidth**: 1,555 GB/s
- **Compute Capability**: 8.0

---

### NVIDIA H100

#### H100 SXM (HBM3)
| Precision | Dense TFLOPS | Sparse TFLOPS |
|-----------|-------------|---------------|
| FP64 | 33.5 | — |
| TF32 | 494 | 989 |
| BF16 | 989 | 1978 |
| FP16 | 989 | 1978 |
| FP8 | 1979 | 3958 |
| INT8 | 1979 | 3958 |

- **Memory**: 80 GB HBM3
- **Memory Bandwidth**: 3,350 GB/s
- **NVLink**: NVLink 4.0, 900 GB/s bidirectional
- **TDP**: 700W
- **Compute Capability**: 9.0
- **Planner default TFLOPS (bf16 dense)**: 989

#### H100 PCIe (HBM3)
| Precision | Dense TFLOPS | Sparse TFLOPS |
|-----------|-------------|---------------|
| BF16 | 756 | 1513 |
| FP16 | 756 | 1513 |
| TF32 | 378 | 756 |

- **Memory**: 80 GB HBM3
- **Memory Bandwidth**: 2,000 GB/s
- **Interconnect**: PCIe 5.0 x16
- **TDP**: 350W
- **Compute Capability**: 9.0
- **Planner default TFLOPS (bf16 dense)**: 756

---

### NVIDIA H200

#### H200 SXM (HBM3e)
| Precision | Dense TFLOPS | Sparse TFLOPS |
|-----------|-------------|---------------|
| BF16 | 989 | 1978 |
| FP16 | 989 | 1978 |
| TF32 | 494 | 989 |
| FP8 | 1979 | 3958 |

- **Memory**: 141 GB HBM3e
- **Memory Bandwidth**: 4,800 GB/s
- **NVLink**: NVLink 4.0, 900 GB/s bidirectional
- **TDP**: 700W
- **Compute Capability**: 9.0
- **Key advantage over H100**: 76% more memory (141 GB vs 80 GB); same compute as H100 SXM
- **Planner default TFLOPS (bf16 dense)**: 989

---

### NVIDIA L40S

| Precision | Dense TFLOPS |
|-----------|-------------|
| BF16 | 362 |
| FP16 | 362 |
| TF32 | 183 |
| INT8 | 724 |

- **Memory**: 48 GB GDDR6
- **Memory Bandwidth**: 864 GB/s
- **Interconnect**: PCIe 4.0 x16 (no NVLink)
- **TDP**: 350W
- **Compute Capability**: 8.9
- **Planner default TFLOPS (bf16 dense)**: 362

---

### NVIDIA RTX 4090

| Precision | Dense TFLOPS |
|-----------|-------------|
| BF16 | 330 |
| FP16 | 330 |
| TF32 | 165 |

- **Memory**: 24 GB GDDR6X
- **Memory Bandwidth**: 1,008 GB/s
- **Interconnect**: PCIe 4.0 x16 (no NVLink)
- **TDP**: 450W (reference) / varies by AIB
- **Compute Capability**: 8.9
- **Note**: Consumer GPU; NVLINK not supported; suited for inference or single-GPU training of small models
- **Planner default TFLOPS (bf16 dense)**: 330

---

### NVIDIA V100

#### V100 32GB SXM2 (HBM2)
| Precision | Dense TFLOPS |
|-----------|-------------|
| FP16 | 125 |
| FP32 | 15.7 |
| TF32 | N/A (not supported) |

- **Memory**: 32 GB HBM2
- **Memory Bandwidth**: 900 GB/s
- **NVLink**: NVLink 2.0, 300 GB/s bidirectional
- **Compute Capability**: 7.0
- **Note**: No native BF16 support; FP16 is the recommended mixed-precision dtype
- **Planner default TFLOPS (fp16 dense)**: 125

---

## 2. Sparse vs. Dense: The Critical Distinction

NVIDIA's 2:4 structured sparsity ("sparse" in product pages) compresses matrices such that 50% of weights are zero in a structured pattern. The hardware can skip multiplications for zero weights, achieving 2x throughput.

**Real-world applicability**:
- Sparse training requires specific training procedures (e.g., SparseGPT, gradual pruning)
- Most LLM training runs are dense
- The planner defaults to DENSE TFLOPS and logs this assumption

```python
# In GPUSpecTable:
spec.peak_tflops_bf16       # dense (default for planning)
spec.peak_tflops_bf16_sparse  # with 2:4 sparsity
```

---

## 3. Dtype Considerations

| dtype | Bits | Supported GPUs | Notes |
|-------|------|---------------|-------|
| BF16 | 16 | A100, H100, H200, L40S, RTX4090 | Preferred for training; same exponent range as FP32 |
| FP16 | 16 | V100+, all modern GPUs | Good for inference; limited dynamic range |
| TF32 | 19 | A100+ | Used internally by Ampere+ for FP32 operations; ~8x speedup vs FP32 |
| FP32 | 32 | All GPUs | Rarely used for training; too slow |
| FP8 | 8 | H100+ | Very fast; requires careful scaling; used in training with transformer engine |

**Planner behavior by dtype**:
- `bf16` → uses `peak_tflops_bf16` (default for A100/H100/H200/L40S/RTX4090)
- `fp16` → uses `peak_tflops_fp16` (usually same as bf16 for modern GPUs; required for V100)
- `fp32` → uses `peak_tflops_fp32` (much lower; rarely used for LLM training)
- `fp8` → uses `peak_tflops_fp8` (H100/H200 only; Transformer Engine required)

---

## 4. Multi-GPU Scaling and Interconnect

For multi-GPU and multi-node training, interconnect bandwidth affects communication efficiency:

| Topology | Bandwidth | Typical Use |
|----------|-----------|------------|
| NVLink 3.0 (A100) | 600 GB/s bidirectional | DGX A100, HGX |
| NVLink 4.0 (H100/H200) | 900 GB/s bidirectional | DGX H100, HGX H100 |
| NVLink 5.0 (B100/B200) | 1800 GB/s bidirectional | Next-gen systems |
| PCIe 4.0 | ~64 GB/s | Consumer / PCIe variants |
| InfiniBand NDR | 400 Gb/s | Inter-node; used with NVSwitch |

**Planner note**: MFU captures the combined effect of compute efficiency and communication overhead. No separate communication model is built into the planner — MFU is the single knob.

---

## 5. Memory Capacity Planning

The planner does not currently enforce memory constraints (it focuses on compute), but practitioners should verify:

```
# Approximate memory requirement for training (rule of thumb):
# - Model weights (bf16): 2 bytes/param
# - Gradients (fp32): 4 bytes/param
# - Optimizer states (AdamW, fp32): 8 bytes/param (m + v)
# - Activations: variable (depends on seq_len, batch, recompute strategy)
# Total (no recompute): ~14-18 bytes/param

mem_required_gb = n_params * 16 / 1e9   # rough estimate for bf16 training + AdamW
```

For a 7B model: ~7B * 16 = 112 GB — requires at least 2x H100 80GB with FSDP/ZeRO-3.

---

## 6. Fallback Behavior for Unknown GPUs

If `GPUSpecTable.lookup()` cannot find the requested GPU type:

1. The planner logs a warning: `"GPU '<name>' not found in spec table."`
2. It requires the user to provide `--peak_tflops` and `--mem_gb` manually
3. It proceeds with user-supplied values and marks the spec as `is_user_supplied=True`
4. The output report documents the source of the spec values

This ensures the planner never silently uses wrong hardware assumptions.

---

## 7. Fuzzy Matching Examples

The `GPUSpecTable.lookup()` method uses case-insensitive substring matching:

| User Input | Matched GPU |
|------------|------------|
| `"h100"` | H100 SXM (first match) |
| `"h100_sxm"` | H100 SXM |
| `"H100 SXM"` | H100 SXM |
| `"h100_pcie"` | H100 PCIe |
| `"H100 PCIe"` | H100 PCIe |
| `"a100"` | A100 80GB SXM (first match) |
| `"a100_40"` | A100 40GB SXM |
| `"rtx4090"` | RTX 4090 |
| `"4090"` | RTX 4090 |
| `"v100"` | V100 32GB |
| `"l40s"` | L40S |

---

## 8. Source Attribution

Specifications sourced from:
- NVIDIA A100 datasheet: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
- NVIDIA H100 datasheet: https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet
- NVIDIA H200 product page: https://www.nvidia.com/en-us/data-center/h200/
- NVIDIA L40S datasheet: https://resources.nvidia.com/en-us-l40s/l40s-datasheet-28413
- NVIDIA RTX 4090 product page: https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/
- NVIDIA V100 datasheet: https://images.nvidia.com/content/technologies/volta/pdf/volta-v100-datasheet-update-us-1165301-r5.pdf

**Last verified**: 2025. Verify with current NVIDIA datasheets for production capacity planning.
