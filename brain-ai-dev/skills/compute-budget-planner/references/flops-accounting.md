# FLOPs Accounting for LLM Training: Comprehensive Reference

## Overview

This reference covers how to accurately count floating-point operations for large Transformer training runs. Accurate FLOPs accounting underlies wallclock prediction, cost estimation, and compute-optimal analysis. The planner supports two estimator modes: "simple 6ND" and "transformer-aware" (which adds the quadratic attention term).

---

## 1. The C ≈ 6ND Derivation

The standard approximation for total training FLOPs is:

```
C = k * N * D     where k = 6 (default)
```

Where:
- `N` = number of (non-embedding) model parameters
- `D` = total training tokens
- `k = 6` accounts for forward pass (~2N FLOPs/token) + backward pass (~4N FLOPs/token)

### Forward Pass: ~2N FLOPs per Token

For a dense Transformer, the dominant computation is matrix multiplications. For each token processed:
- Each matrix multiplication `(m, k) x (k, n)` requires `2*m*k*n` FLOPs (multiply-add = 2 ops)
- Summing across all layers and projections for a model with `N` total parameters yields approximately `2N` FLOPs per token

This is exact for the parameter-compute correspondence when sequence length effects are ignored (see Section 3 for the correction).

### Backward Pass: ~4N FLOPs per Token

The backward pass requires computing:
1. **Gradient w.r.t. activations** (needed for upstream gradient propagation): ~2N FLOPs
2. **Gradient w.r.t. parameters** (needed for optimizer update): ~2N FLOPs

Total backward: ~4N FLOPs per token.

### Combined: k = 2 + 4 = 6

```
Total FLOPs per token = 2N (forward) + 4N (backward) = 6N
Total FLOPs for D tokens = 6 * N * D
```

This derivation appears in Chinchilla (Hoffmann et al. 2022) and is standard in the scaling laws literature. The planner uses k=6 as its default, but exposes k as a configurable parameter.

---

## 2. Per-Layer FLOPs Breakdown (Transformer-Aware Mode)

For a Transformer with:
- `L` layers
- `d_model` = hidden dimension
- `d_ff` = feed-forward intermediate dimension (typically 4 * d_model)
- `n_heads` = number of attention heads
- `d_head` = d_model / n_heads
- `T` = sequence length

### Attention Layer FLOPs (per token, per layer)

| Operation | FLOPs per token |
|-----------|----------------|
| Q, K, V projections (3 x d_model x d_model) | 6 * d_model^2 |
| Attention scores: Q * K^T (T tokens, T positions) | 2 * T * d_model |
| Softmax + weighted sum: attn_weights * V | 2 * T * d_model |
| Output projection (d_model x d_model) | 2 * d_model^2 |
| **Attention total per token** | 8 * d_model^2 + 4 * T * d_model |

### MLP / FFN Layer FLOPs (per token, per layer)

For standard two-layer FFN (up + down):

| Operation | FLOPs per token |
|-----------|----------------|
| Up projection (d_model -> d_ff) | 2 * d_model * d_ff |
| Down projection (d_ff -> d_model) | 2 * d_ff * d_model |
| **FFN total per token** | 4 * d_model * d_ff |

For gated FFN (SwiGLU, common in modern LLMs with d_ff ≈ 8/3 * d_model):

| Operation | FLOPs per token |
|-----------|----------------|
| Gate projection (d_model -> d_ff) | 2 * d_model * d_ff |
| Up projection (d_model -> d_ff) | 2 * d_model * d_ff |
| Down projection (d_ff -> d_model) | 2 * d_ff * d_model |
| **SwiGLU total per token** | 6 * d_model * d_ff |

### Total FLOPs per Token (Transformer-Aware)

```
flops_per_token = L * (
    8 * d_model^2 +           # QKV + output proj
    4 * T * d_model +          # attention scores (quadratic in T!)
    4 * d_model * d_ff         # FFN (standard) or 6*... (SwiGLU)
)
```

### Quadratic Attention Correction

The key difference between "simple 6ND" and "transformer-aware":

```
attention_flops = 2 * L * T * T * d_model
                = 2 * L * T^2 * d_model
```

For standard sequence lengths (T ≤ 2048), this term is small relative to the matrix-multiply terms. For long-context models (T ≥ 8192), it becomes significant:

| T | Relative attention overhead (70B, 96 layers, d_model=8192) |
|---|-----|
| 2048 | ~4% |
| 8192 | ~16% |
| 32768 | ~64% |
| 131072 | ~256% (attention dominates!) |

**Recommendation**: Use "transformer-aware" mode for T > 4096.

---

## 3. Simple 6ND vs. Transformer-Aware Estimator

| Estimator | Formula | When to Use |
|-----------|---------|-------------|
| Simple 6ND | `C = 6 * N * D` | T ≤ 4096, quick estimates |
| Transformer-aware | `C = (6N + 2*L*T^2*d_model/d_model_tokens) * D` | T > 4096, precision needed |

The planner defaults to simple 6ND and logs the assumption. Pass `--transformer_aware` to enable the correction.

---

## 4. Model FLOPs Utilization (MFU)

**MFU** measures how efficiently the hardware is used relative to its theoretical peak:

```
MFU = achieved_flops_per_s / peak_flops_per_s
```

Where:
```
achieved_flops_per_s = (6 * N * tokens_per_step) / step_time_s
peak_flops_per_s = num_gpus * peak_tflops_per_gpu * 1e12
```

### Typical MFU Ranges

| System / Setup | Typical MFU |
|---------------|-------------|
| Poorly optimized (eager, no compile) | 0.10 – 0.20 |
| Reasonably optimized (FlashAttention, bf16) | 0.30 – 0.40 |
| Well-optimized (torch.compile, FSDP2) | 0.40 – 0.55 |
| State of the art (custom kernels, NVLink) | 0.55 – 0.65 |

The planner uses **0.35 (35%) as the default MFU assumption** — a conservative estimate suitable for planning. Users should measure actual MFU on their cluster and pass it explicitly.

### Why MFU < 1.0

MFU is limited by:
- Memory bandwidth bottlenecks (loading weights, KV cache)
- Communication overhead in distributed training (all-reduce, all-gather)
- CPU-GPU synchronization and pipeline bubbles
- Non-compute operations (normalization, embedding lookups)
- Activation checkpointing overhead (adds ~30% extra compute for large models)

---

## 5. Wallclock Time Formula

```
peak_flops_per_s = num_gpus * peak_tflops * 1e12     # peak hardware throughput
achieved_flops_per_s = peak_flops_per_s * utilization  # actual throughput
wallclock_s = total_flops / achieved_flops_per_s
wallclock_h = wallclock_s / 3600
```

### Worked Example

70B model, 1.4T tokens, 8x H100 SXM at 35% MFU:

```
total_flops = 6 * 70e9 * 1.4e12 = 5.88e23
peak_per_gpu = 989e12  # H100 SXM bf16 dense
achieved = 8 * 989e12 * 0.35 = 2.7692e15 FLOPs/s
wallclock_s = 5.88e23 / 2.7692e15 = 2.1227e8 s
wallclock_h = 2.1227e8 / 3600 ≈ 58,964 hours / 8 GPUs = 7,370 H100-hours
```

At cluster level: ~7,370 hours on 1 H100, or ~920 hours (38 days) on 8x H100.

---

## 6. Cost Estimation

```
cost_usd = wallclock_h * num_gpus * cost_per_gpu_hour
```

Typical cloud spot/on-demand GPU pricing (2024-2025 estimates; verify with your provider):

| GPU | Spot ($/hr) | On-demand ($/hr) |
|-----|------------|------------------|
| A100 80GB | $2.50 – $3.50 | $3.50 – $5.00 |
| H100 SXM | $3.50 – $5.00 | $5.00 – $8.00 |
| H100 PCIe | $3.00 – $4.50 | $4.00 – $6.00 |
| RTX 4090 | $0.40 – $0.80 | $0.80 – $1.50 |

The planner's `cost_per_gpu_hour` parameter is intentionally left as `None` (optional) — pricing varies dramatically by provider, contract, and region.

---

## 7. Token Accounting with Gradient Accumulation

When using gradient accumulation:

```
micro_batch = global_batch / (num_gpus * grad_accum_steps)
tokens_per_step = global_batch * seq_len
total_tokens = steps * global_batch * seq_len
```

**Critical**: `steps` in the planner refers to **optimizer steps** (after gradient accumulation completes), not forward passes. Each optimizer step processes `global_batch * seq_len` tokens.

### Multi-Epoch Training

If the same data is repeated (multiple epochs):
```
total_tokens = dataset_size_tokens * num_epochs
```

The planner does not distinguish repeated data from unique data in its FLOPs accounting, but repeated-data training may have different loss dynamics than predicted by Chinchilla (which assumes unique tokens).

---

## 8. Non-Embedding Parameter Counting

**Standard practice**: Exclude token embeddings and (if weight-tied) the output projection from `N` when computing FLOPs.

Reasoning: Embedding lookups are memory-bound operations that do not perform matrix multiplications. Their compute cost is negligible relative to the attention and MLP layers.

```python
# For a standard Transformer:
n_params_total = vocab_size * d_model          # embedding table
                + L * (8 * d_model^2           # attention projections
                      + d_model * d_ff * 4)    # FFN
                + d_model                       # final layer norm

n_params_non_embedding = n_params_total - vocab_size * d_model
# If output projection is weight-tied with embedding:
# n_params_non_embedding stays the same (already excluded)
```

The planner uses `n_params` as provided by the user and logs that it should be **non-embedding** parameter count for accurate FLOPs estimation.

---

## 9. Activation Checkpointing FLOPs Penalty

Gradient checkpointing (recompute activations during backward) adds approximately one extra forward pass:

```
effective_k = 6 + 2 = 8    # with full activation checkpointing
effective_k = 6 + ~1 = 7   # with selective checkpointing
```

The planner uses k=6 by default (no checkpointing overhead) and notes this assumption. For accurate wallclock prediction with checkpointing enabled, users should:
1. Measure actual step time empirically
2. Compute effective MFU from measured step time and k=6
3. The checkpointing overhead is implicitly absorbed into lower measured MFU

---

## 10. Summary: FLOPs Estimation Hierarchy

```
Level 1 (Quick): C = 6 * N * D
  → Accurate within 5% for T <= 2048, no checkpointing

Level 2 (Corrected): C = (6*N + 2*L*T^2*d_model / (N/d_model)) * D
  → Adds attention term; needed for T > 4096

Level 3 (Empirical): Measure step_time, compute achieved_flops, solve for effective k
  → Ground truth; use for capacity planning and billing reconciliation
```

The planner implements Level 1 by default and logs the assumption. Level 2 is available via `--transformer_aware` flag. Level 3 requires the user to supply measured throughput data.
