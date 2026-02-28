# Benchmark Harness Reference — Phase 2

## Overview

Phase 2 implements the training micro-benchmark harness that measures wall-clock step time, tokens/second, and Model FLOPs Utilization (MFU). Every measurement decision in this phase exists to produce numbers that are reproducible across runs, comparable across machines of the same profile, and meaningful as regression signals.

---

## CUDA Synchronization Timing

### The Problem: Kernel Queueing vs. Execution

CUDA operations are asynchronous by default. When Python calls a CUDA operation, the CUDA runtime queues the kernel and immediately returns control to Python. The kernel may not begin executing—or may not finish executing—by the time Python moves to the next line.

**Without synchronization:**

```python
# WRONG — measures kernel queueing, not execution
t0 = time.perf_counter()
loss.backward()          # Returns immediately; kernels are still queued
optimizer.step()         # Also async; still queueing
t1 = time.perf_counter()
step_time = t1 - t0     # Typically 0.5–2ms regardless of true compute time
```

The measured time reflects Python overhead and kernel submission latency, not actual GPU execution. On a 7B parameter model, the true step time might be 2–5 seconds while this measurement returns 10ms.

**With correct synchronization:**

```python
# CORRECT — measures actual GPU execution
torch.cuda.synchronize()   # Flush and wait for all pending kernels
t0 = time.perf_counter()
[full training step]
torch.cuda.synchronize()   # Wait for step kernels to complete
t1 = time.perf_counter()
step_time = t1 - t0        # True wall-clock execution time
```

### Why `torch.cuda.synchronize()` Works

`torch.cuda.synchronize()` blocks the calling CPU thread until all CUDA kernels that have been submitted to the default stream (and all streams, if using multi-stream) have completed. After the call returns, you can safely measure wall time with `time.perf_counter()`.

### Bracketing Rules

Both calls are mandatory:

1. **Pre-step synchronize**: Ensures any residual work from warmup or the previous step has fully completed before the timer starts. Without this, the first measurement captures tail work from the previous step.

2. **Post-step synchronize**: Ensures all kernels in the current step (including async gradient reductions in DDP) have completed. Without this, you measure scheduling, not compute.

### Multi-GPU / DDP Considerations

In distributed training, `torch.cuda.synchronize()` only synchronizes the local GPU. NCCL allreduce operations are submitted to a separate NCCL stream. To ensure those are captured:

```python
# Option A: Use barrier (synchronizes across ranks and streams)
torch.distributed.barrier()
torch.cuda.synchronize()
t1 = time.perf_counter()
```

For benchmarking purposes, use the rank-0 measurement as the canonical step time. All ranks should be roughly synchronous at the barrier point.

### Gradient Accumulation Boundaries

When using gradient accumulation (accumulate N micro-steps before calling `optimizer.step()`), the step time **must include the full accumulation cycle**:

```python
# WRONG — measures only one micro-step
torch.cuda.synchronize()
t0 = time.perf_counter()
loss = model(batch)
loss.backward()
torch.cuda.synchronize()
step_time = time.perf_counter() - t0

# CORRECT — full accumulation cycle is the meaningful unit
torch.cuda.synchronize()
t0 = time.perf_counter()
for i in range(grad_accum_steps):
    with model.no_sync() if i < grad_accum_steps - 1 else contextlib.nullcontext():
        loss = model(batch[i]) / grad_accum_steps
        loss.backward()
optimizer.step()
optimizer.zero_grad()
torch.cuda.synchronize()
step_time = time.perf_counter() - t0
```

Tokens per step must then count tokens across all micro-steps in the accumulation:
```
tokens_per_step = global_batch_size * seq_len * grad_accum_steps
```

---

## Warmup Protocol

### Why Warmup Is Required

The first N steps of training exhibit artificially high latency due to:

1. **JIT compilation**: `torch.compile()`, CUDA fused kernels, and cuDNN autotune all run their heaviest work on the first few steps.
2. **CUDA caching allocator**: The first allocations are slow (OS memory faults). After warmup, the allocator reuses pools.
3. **cuDNN algorithm selection**: convolutions and attention ops benchmark multiple algorithms and cache the fastest.
4. **Data pipeline priming**: The dataloader prefetch queue, DALI pipeline, or filesystem cache needs to fill.

Including these in measurements produces artificially high step times. The warmup steps are discarded.

### Standard Warmup Procedure

```python
# Step 1: Reset memory statistics to a clean baseline
torch.cuda.reset_peak_memory_stats()

# Step 2: Run warmup steps (discarded from measurement)
model.train()
for _ in range(config.warmup_steps):
    batch = next(data_iter)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(batch["input_ids"], labels=batch["labels"]).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

# Step 3: Synchronize after warmup before measurement begins
torch.cuda.synchronize()

# Step 4: Optionally reset memory stats again after warmup
# (warmup may allocate persistent workspace not representative of steady state)
torch.cuda.reset_peak_memory_stats()

# Step 5: Now measure
step_times = []
for step in range(config.measure_steps):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    [training step]
    torch.cuda.synchronize()
    step_times.append(time.perf_counter() - t0)
```

### Recommended Warmup Steps

| Model Size | Recommended warmup_steps |
|-----------|--------------------------|
| < 1B params | 50–100 |
| 1B–7B params | 100–200 |
| 7B–70B params | 200–500 |
| > 70B params | 500+ |

The default `warmup_steps = 200` is conservative enough for most sizes.

---

## Tokens/Second Computation

### Definition

**Tokens per step** is the number of tokens processed in one full training step (including gradient accumulation if applicable), across all data-parallel ranks:

```
tokens_per_step = global_batch_size * seq_len
```

Where `global_batch_size = per_device_batch_size * num_gpus * grad_accum_steps`.

**Tokens per second** is the steady-state throughput:

```
tokens_per_sec = tokens_per_step / step_time_s
```

### Code Implementation

```python
def compute_tokens_per_sec(
    per_device_batch_size: int,
    seq_len: int,
    world_size: int,
    grad_accum_steps: int,
    step_time_s: float,
) -> float:
    tokens_per_step = per_device_batch_size * seq_len * world_size * grad_accum_steps
    return tokens_per_step / step_time_s
```

### What This Measures

Tokens/sec measures the **end-to-end training pipeline throughput**, including:
- Forward pass (compute)
- Loss computation
- Backward pass (compute, 2x forward FLOPs)
- Gradient synchronization across GPUs (communication)
- Optimizer step (compute)
- Memory allocation/deallocation overhead

It does not measure peak hardware capability—that is MFU's job. Tokens/sec is the practical metric: what your actual pipeline delivers.

---

## MFU (Model FLOPs Utilization) Estimation

### What MFU Measures

MFU answers: "What fraction of theoretical peak GPU compute is the training pipeline actually using?"

```
MFU = achieved_flops_per_sec / (num_gpus * peak_flops_per_gpu)
```

MFU is a hardware efficiency metric. An MFU of 0.45 means 45% of theoretical peak throughput is being utilized.

**Typical MFU ranges:**
- 0.35–0.55: Typical range for well-optimized Transformer training on A100/H100
- > 0.55: Excellent—flashattention, fused kernels, good batch sizing
- < 0.25: Investigate bottlenecks (memory bandwidth, data pipeline, communication)

### Estimator A: Simple 6ND

The 6ND rule approximates FLOPs for a Transformer by noting that:
- A matrix multiply `(M, K) @ (K, N)` costs `2 * M * K * N` FLOPs
- For each token, the model performs roughly 6 multiplications by the total non-embedding parameter count N

This is derived from Chinchilla (Hoffmann et al. 2022) and the PaLM training paper:

```
flops_per_step = 6 * N_non_embedding_params * tokens_per_step
achieved_flops_per_sec = flops_per_step / step_time_s
mfu = achieved_flops_per_sec / (num_gpus * peak_flops_per_gpu)
```

**Why 6?** Factor of 2 for the multiply-accumulate, factor of 3 for forward + backward + gradient update (backward ~2x forward).

**What "non-embedding params" means:** Exclude the token embedding matrix and position embeddings. Include all attention weights (Q, K, V, O projections), MLP weights, and layer norm parameters.

```python
def count_non_embedding_params(model: torch.nn.Module) -> int:
    total = sum(p.numel() for p in model.parameters())
    embed_params = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):
            embed_params += sum(p.numel() for p in module.parameters())
    return total - embed_params
```

**Code:**

```python
def compute_mfu_6nd(
    n_non_embedding_params: int,
    tokens_per_step: int,
    step_time_s: float,
    num_gpus: int,
    peak_tflops_per_gpu: float,
) -> float:
    flops_per_step = 6 * n_non_embedding_params * tokens_per_step
    achieved_tflops = flops_per_step / step_time_s / 1e12
    peak_tflops_total = num_gpus * peak_tflops_per_gpu
    return achieved_tflops / peak_tflops_total
```

### Estimator B: Transformer-Aware

The transformer-aware estimator adds attention FLOPs on top of 6ND. Attention FLOPs scale quadratically with sequence length:

```
# Attention FLOPs per layer per token:
# QK^T matmul: 2 * seq_len * head_dim * num_heads = 2 * seq_len * hidden_dim
# Softmax scores @ V: 2 * seq_len * head_dim * num_heads = 2 * seq_len * hidden_dim
# Total per layer: 4 * seq_len * hidden_dim
# For all layers: 4 * seq_len * hidden_dim * num_layers

attention_flops = 4 * seq_len * hidden_dim * num_layers
# Forward only; multiply by 3 for forward + backward (same 6ND logic):
attention_flops_per_token = attention_flops * 3  # But see note below

# Full formula including attention (per token):
flops_per_token = 6 * n_non_embedding_params + 4 * seq_len * hidden_dim * num_layers
# Times 2 for backward scaling relative to forward-only factor 3 convention...
```

Simplified standard form (from Megatron-LM):

```python
def compute_mfu_transformer_aware(
    n_non_embedding_params: int,
    seq_len: int,
    hidden_dim: int,
    num_layers: int,
    tokens_per_step: int,
    step_time_s: float,
    num_gpus: int,
    peak_tflops_per_gpu: float,
) -> float:
    # 6ND base
    base_flops = 6 * n_non_embedding_params * tokens_per_step
    # Attention correction: 2 * seq_len^2 * hidden_dim * num_layers per step
    # Factor of 3 (forward+backward), 2 for the two matmuls per head
    attn_flops = 2 * (seq_len ** 2) * hidden_dim * num_layers * tokens_per_step // seq_len
    # attn_flops_per_step = 12 * seq_len * hidden_dim * num_layers * batch_tokens
    attn_flops_total = 12 * seq_len * hidden_dim * num_layers * (tokens_per_step // seq_len)
    total_flops = base_flops + attn_flops_total
    achieved_tflops = total_flops / step_time_s / 1e12
    peak_tflops_total = num_gpus * peak_tflops_per_gpu
    return achieved_tflops / peak_tflops_total
```

**When to use transformer-aware:** When sequence length is long (> 2048) and the attention FLOPs are a significant fraction of total compute. For short sequences (≤ 512), 6ND and transformer-aware converge.

---

## GPU Peak TFLOPS Registry

Peak theoretical throughput at bf16 or fp16, as reported in official GPU specifications. These are non-sparse (dense) FLOPs unless noted.

| GPU | bf16 TFLOPS | fp16 TFLOPS | Notes |
|-----|-------------|-------------|-------|
| H100 SXM5 | 989 | 1979 | MXFPs excluded |
| H100 PCIe | 756 | 1513 | |
| H100 NVL | 835 | 1671 | |
| A100 SXM4 80GB | 312 | 312 | Same rate for bf16/fp16 on A100 |
| A100 PCIe 80GB | 312 | 312 | |
| A100 SXM4 40GB | 312 | 312 | |
| A100 PCIe 40GB | 312 | 312 | |
| A40 | 149.7 | 149.7 | |
| L40S | 362 | 362 | |
| L40 | 181 | 181 | |
| RTX 4090 | 330 | 165 | Note: bf16 is higher than fp16 on Ada |
| RTX 4080 | 242 | 121 | |
| RTX 3090 Ti | 40 | 40 | fp16 only; bf16 ≈ fp16 |
| V100 SXM2 | 125 | 125 | fp16 (V100 does not support bf16 natively) |
| V100 PCIe | 112 | 112 | |
| A10 | 125 | 125 | |
| A10G | 125 | 125 | |
| T4 | 65 | 65 | |

**Note on sparse TFLOPS:** NVIDIA reports 2x "sparse" TFLOPS for structured sparsity (50% weight sparsity). These numbers are rarely achieved in practice and should not be used as the denominator for MFU unless the model explicitly uses sparse formats.

**Looking up GPU from torch:**

```python
def get_peak_tflops(device_name: str) -> Optional[float]:
    """
    Maps torch.cuda.get_device_name() output to peak bf16 TFLOPS.
    Returns None if GPU not in registry.
    """
    GPU_TFLOPS = {
        "H100 SXM5": 989, "H100 SXM": 989, "H100 PCIe": 756,
        "A100 SXM4": 312, "A100 SXM": 312, "A100 PCIe": 312, "A100": 312,
        "L40S": 362, "L40": 181,
        "RTX 4090": 330, "RTX 4080": 242,
        "RTX 3090": 40, "V100": 125,
        "A10G": 125, "A10": 125, "T4": 65,
    }
    for key, val in GPU_TFLOPS.items():
        if key in device_name:
            return val
    return None
```

---

## Aggregation Statistics

### Step Time Aggregation

Gate on **medians** to reduce sensitivity to GC pauses, background processes, and thermal throttling:

| Statistic | Usage |
|-----------|-------|
| `step_time_mean` | Reporting, trend analysis |
| `step_time_p50` | **Primary gate metric** |
| `step_time_p90` | Tail latency; useful for SLA-style gates |

### Tokens/Second Aggregation

Tokens/sec is inversely related to step time, so statistics invert:

| Statistic | Usage |
|-----------|-------|
| `tokens_per_sec_mean` | Reporting |
| `tokens_per_sec_p50` | **Primary gate metric** |
| `tokens_per_sec_p10` | Worst-case throughput; conservative gate |

### Implementation

```python
import numpy as np

def aggregate_stats(step_times: list[float]) -> dict:
    arr = np.array(step_times)
    tps = [tokens_per_step / t for t in step_times]
    tps_arr = np.array(tps)
    return {
        "step_time": {
            "mean": float(np.mean(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        },
        "tokens_per_sec": {
            "mean": float(np.mean(tps_arr)),
            "p50": float(np.percentile(tps_arr, 50)),
            "p10": float(np.percentile(tps_arr, 10)),
        },
    }
```

### Memory Statistics

```python
memory_peak_bytes = torch.cuda.max_memory_allocated()
memory_reserved_bytes = torch.cuda.max_memory_reserved()
```

Report both: `allocated` is what the model is actively using, `reserved` is what PyTorch has claimed from the OS (may differ significantly due to allocator fragmentation).

---

## Complete metrics.json Schema

```json
{
  "schema_version": "1.0",
  "machine_profile": "H100x8_driver550_cuda12.4_torch2.4_sm90",
  "run_config": {
    "mode": "synthetic",
    "world_size": 8,
    "per_device_batch_size": 8,
    "seq_len": 2048,
    "global_batch_size": 64,
    "grad_accum_steps": 1,
    "warmup_steps": 200,
    "measure_steps": 100,
    "repeat": 1,
    "profile": "off",
    "mfu_estimator": "6ND",
    "dtype": "bfloat16"
  },
  "throughput": {
    "tokens_per_sec_mean": 125000.0,
    "tokens_per_sec_p50": 126500.0,
    "tokens_per_sec_p10": 118000.0,
    "tokens_per_step": 131072
  },
  "timing": {
    "step_time_mean_s": 1.048,
    "step_time_p50_s": 1.037,
    "step_time_p90_s": 1.112,
    "step_time_min_s": 1.021,
    "step_time_max_s": 1.198
  },
  "memory": {
    "peak_allocated_bytes": 68719476736,
    "peak_allocated_gb": 64.0,
    "peak_reserved_bytes": 75161927680,
    "peak_reserved_gb": 70.0
  },
  "mfu": {
    "mfu_p50": 0.412,
    "mfu_mean": 0.408,
    "n_non_embedding_params": 6700000000,
    "peak_tflops_per_gpu": 989.0,
    "estimator": "6ND"
  },
  "loss": {
    "final_loss": 2.34,
    "loss_slope": -0.0023,
    "loss_values": [2.56, 2.51, 2.46, 2.41, 2.38, 2.35, 2.34]
  },
  "env": {
    "torch_version": "2.4.0",
    "cuda_version": "12.4",
    "python_version": "3.11.4",
    "git_sha": "abc1234",
    "git_dirty": false
  },
  "timestamp": "2026-02-21T00:00:00Z"
}
```

---

## Synthetic Mode vs. E2E Mode

### Synthetic Mode

Generates random tensors matching the target batch shape in Python/CPU, then moves to GPU. Bypasses the dataloader entirely:

```python
def _create_synthetic_batch(self) -> dict:
    return {
        "input_ids": torch.randint(
            0, self.config.vocab_size,
            (self.config.per_device_batch_size, self.config.seq_len),
            device="cuda",
        ),
        "labels": torch.randint(
            0, self.config.vocab_size,
            (self.config.per_device_batch_size, self.config.seq_len),
            device="cuda",
        ),
    }
```

**Use synthetic when:**
- Data pipeline is noisy (network filesystems, slow prefetch)
- You want to isolate pure compute performance from I/O
- Gating on compute-only regressions (kernel changes, optimizer changes)

**Limitation:** Does not catch data pipeline bottlenecks. If the dataloader is slower than the GPU, synthetic mode will show perfect throughput while e2e will show a stall.

### E2E Mode

Uses the real dataloader. The step time includes:
- Data transfer from CPU to GPU (`batch.to("cuda")`)
- Any preprocessing done after DataLoader loads
- The full compute step

**Use e2e when:**
- Data pipeline is stable and fast (local NVMe, RAM cache)
- Gating holistic throughput including data movement
- Profiling data pipeline bottlenecks

### Switching Between Modes

```python
if config.mode == "synthetic":
    batch = self._create_synthetic_batch()
else:  # "e2e"
    batch = next(data_iter)
    batch = {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}
```

---

## --repeat Flag

Running `--repeat K` performs K full benchmark runs (each with their own warmup + measure phase) and reports statistics from the run with the best (lowest) median step time:

```python
best_result = None
for repeat_idx in range(config.repeat):
    result = self._single_run()
    if best_result is None or result.step_time_p50 < best_result.step_time_p50:
        best_result = result
return best_result
```

**Why take the best?** Background processes (OS updates, antivirus, other jobs) can transiently slow the GPU. Taking the best of K runs reduces this noise. Typical: `--repeat 3` in CI, `--repeat 5` for baseline establishment.

**Note:** Take the best **median** (p50), not the best single step. This avoids lucky single-step timing.

---

## Implementation Checklist

- [ ] `torch.cuda.reset_peak_memory_stats()` called before warmup
- [ ] Warmup runs without measurement
- [ ] `torch.cuda.synchronize()` before and after every timed step
- [ ] `time.perf_counter()` used (not `time.time()`)
- [ ] `tokens_per_step` counts all tokens across all ranks
- [ ] MFU denominator uses non-embedding params
- [ ] Peak TFLOPS from registry, not hardcoded
- [ ] Step times aggregated with numpy for p50/p90
- [ ] metrics.json written atomically (write to tmp, rename)
- [ ] Schema version included in output
