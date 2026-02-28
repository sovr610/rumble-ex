# Scaling Benchmark: Methodology, Metrics Schema, and Regression Detection

## Overview

Scaling benchmarks measure how efficiently a distributed training strategy uses additional GPUs. A perfectly scaling system would have `world_size=N` GPUs running exactly N times faster than a single GPU. In practice, communication overhead, synchronization barriers, and memory management reduce this efficiency. This document defines the measurement methodology, the metrics schema, diagnostic patterns for poor scaling, and integration with the perf-regression-gate skill.

---

## 1. Measurement Methodology: 1-GPU vs N-GPU

### 1.1 Invariants for Valid Comparison

For a scaling measurement to be meaningful, the following must be held constant between the 1-GPU and N-GPU runs:

1. **Same model architecture** — identical layer counts, hidden dims, and parameter counts
2. **Same per-GPU batch size** — global batch size scales with world_size, but each GPU sees the same number of tokens per step. This isolates compute efficiency from batch-size effects.
3. **Same sequence length** — varying sequence length changes memory pressure and attention compute complexity independently of scaling
4. **Same mixed precision setting** — bf16 on 1 GPU and fp32 on N GPUs would conflate dtype effects with scaling effects
5. **Same activation checkpointing setting** — the 1-GPU baseline should mirror the N-GPU config exactly
6. **Warmup steps excluded** — discard the first 5–10 steps of each run (JIT compilation, CUDA graph capture, first collective initialization)

### 1.2 What to Measure

Measure **tokens per second (throughput)** as the primary metric:

```
throughput = (batch_size * seq_len * steps) / wall_clock_time
```

Where:
- `batch_size` is the per-GPU batch size
- `seq_len` is the sequence length in tokens
- `steps` is the number of measured training steps (after warmup)
- `wall_clock_time` is elapsed wall time in seconds

Measure over at least 50 steps after warmup to get stable statistics. Compute p50 (median) and p90 for both throughput and step time. The p90 step time reveals outlier stalls (e.g., checkpoint saves, garbage collection, communication timeouts) that do not appear in the median.

### 1.3 Throughput vs Step Time

Both metrics are useful for different diagnoses:

| Metric | Good for detecting |
|--------|--------------------|
| Tokens/sec (throughput_p50) | Overall efficiency, compare strategies |
| Step time p50 | Per-step overhead from communication |
| Step time p90 | Outlier stalls, GC pauses, OOM recovery |
| Memory peak (GB) | Maximum GPU memory pressure during a step |

---

## 2. Scaling Efficiency Formula

```
scaling_efficiency = throughput_N / (N * throughput_1)
```

Where:
- `throughput_N` = tokens per second at world_size N
- `throughput_1` = tokens per second at world_size 1
- `N` = world_size

A value of 1.0 (100%) is perfect linear scaling. In practice:
- **0.85–0.95**: Good scaling (typical for FSDP on NVLink-connected GPUs)
- **0.70–0.85**: Acceptable scaling (PCIe-connected GPUs, or models with many small FSDP units)
- **0.50–0.70**: Poor scaling — investigate wrap policy, bucket sizes, or offload
- **< 0.50**: Severe issue — likely misconfiguration

### 2.1 Superlinear Scaling (> 1.0)

Scaling efficiency above 1.0 can occur when:
- The single-GPU run is memory-constrained (OOM thrash, swap to CPU) and the N-GPU run is not
- Larger global batch size enables better hardware utilization (not true scaling efficiency, but a real throughput gain)

Report superlinear scaling as a flag rather than a success — it indicates the 1-GPU baseline was not representative.

---

## 3. metrics.json Schema

All scaling benchmark results must be serialized to this schema for consumption by the perf-regression-gate skill and trend analysis.

```json
{
  "strategy": "fsdp",
  "world_size": 8,
  "scaling_efficiency": 0.91,
  "throughput_1": 12500.5,
  "throughput_n": 91500.3,
  "memory_peak_gb": 38.2,
  "step_time_p50_ms": 142.3,
  "step_time_p90_ms": 148.1,
  "throughput_p50": 91500.3,
  "throughput_p90": 89000.0,
  "model_params_b": 7.0,
  "per_gpu_batch_size": 4,
  "seq_len": 2048,
  "mixed_precision": "bf16",
  "activation_checkpointing": false,
  "warmup_steps": 10,
  "measured_steps": 50,
  "timestamp": "2026-02-21T14:30:00Z",
  "git_commit": "abc1234"
}
```

### 3.1 Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `strategy` | string | One of: `ddp`, `fsdp`, `deepspeed_zero2`, `deepspeed_zero3` |
| `world_size` | int | Number of GPUs used for the N-GPU benchmark |
| `scaling_efficiency` | float | `throughput_n / (world_size * throughput_1)` |
| `memory_peak_gb` | float | Peak GPU memory in GB (max across ranks) |
| `throughput_p50` | float | Median tokens per second over measured steps |
| `step_time_p50_ms` | float | Median step time in milliseconds |

### 3.2 Optional But Recommended Fields

| Field | Type | Description |
|-------|------|-------------|
| `throughput_1` | float | 1-GPU throughput used as baseline |
| `throughput_n` | float | N-GPU throughput (same as throughput_p50 at world_size N) |
| `step_time_p90_ms` | float | 90th percentile step time (reveals stalls) |
| `throughput_p90` | float | 10th percentile throughput (conservative bound) |
| `model_params_b` | float | Model size in billions of parameters |
| `per_gpu_batch_size` | int | Batch size per GPU |
| `seq_len` | int | Sequence length in tokens |
| `mixed_precision` | string | `bf16`, `fp16`, or `none` |
| `activation_checkpointing` | bool | Whether activation checkpointing was enabled |
| `warmup_steps` | int | Steps excluded from measurement |
| `measured_steps` | int | Steps included in measurement |
| `timestamp` | string | ISO 8601 timestamp of benchmark run |
| `git_commit` | string | Short git hash for reproducibility |

---

## 4. What Bad Scaling Reveals

### 4.1 Tiny FSDP Units (Communication Overhead)

**Symptom:** Scaling efficiency drops sharply from 1 to 2 GPUs and continues declining at 4 and 8 GPUs. Step time p50 is significantly higher than expected for the compute load.

**Cause:** The FSDP wrap policy is creating very small units (e.g., wrapping individual linear layers instead of full transformer blocks). Each tiny unit triggers its own all-gather and reduce-scatter collective. The ratio of communication operations to compute is too high.

**Diagnosis:** Print the wrapped model and count the number of `FullyShardedDataParallel(...)` units. Compare to the number of transformer blocks. If there are significantly more units than blocks, the wrap policy is misconfigured.

**Fix:** Switch to `ModuleWrapPolicy({TransformerBlock})` to wrap at block granularity.

### 4.2 Wrong Bucket Sizes

**Symptom:** Scaling efficiency is 0.60–0.75 where 0.85+ is expected. Step time p90 has many outliers 2–3x the p50.

**Cause:** `reduce_bucket_size` (DeepSpeed) or equivalent bucket granularity is too small, causing many small collective operations instead of a few large ones. NCCL has significant per-operation overhead for small tensors.

**Diagnosis:** Profile with `NCCL_DEBUG=INFO` and count the number of collective operations per step. Compare `reduce_bucket_size` against the total gradient volume (model_params * dtype_bytes).

**Fix:** Increase `reduce_bucket_size` to 20–50% of total gradient volume. For a 7B fp16 model, gradients are ~14 GB, so `reduce_bucket_size=2_000_000_000` (2 GB) is a reasonable target.

### 4.3 Accidental Offload

**Symptom:** Throughput drops catastrophically (10x+ reduction) when moving from 1 GPU to N GPUs. Memory peak is unexpectedly low.

**Cause:** CPU offload (`CPUOffload` in FSDP or `offload_param`/`offload_optimizer` in DeepSpeed) was enabled in the multi-GPU config but not the single-GPU config. Or offload was enabled inadvertently through a config merge.

**Diagnosis:** Compare the configs used for the 1-GPU and N-GPU runs. Log `cpu_offload` and `offload_param`/`offload_optimizer` settings explicitly.

**Fix:** Ensure offload settings are identical between 1-GPU baseline and N-GPU measurement.

### 4.4 OOM Thrash

**Symptom:** `step_time_p90_ms` is 5–20x the p50. Throughput variance is very high. Occasional CUDA OOM errors in logs (if PyTorch catches and recovers) or GPU memory usage is near 100% of capacity.

**Cause:** The training run is operating near the GPU memory limit. Occasional allocations push over the limit, causing PyTorch to invoke the memory allocator's garbage collection (caching allocator fragmentation cleanup), which stalls all GPU activity.

**Diagnosis:** Monitor `torch.cuda.memory_reserved()` and `torch.cuda.memory_allocated()` throughout training. If reserved approaches the device capacity and allocated is near reserved, fragmentation is occurring.

**Fix:** Reduce per-GPU batch size, enable activation checkpointing, increase FSDP sharding (move from SHARD_GRAD_OP to FULL_SHARD), or reduce sequence length.

---

## 5. Integration with perf-regression-gate Skill

### 5.1 Output Format Compatibility

The `metrics.json` file produced by `ScalingBenchmark.save_metrics()` is designed to be consumed directly by the perf-regression-gate skill. The gate reads:

- `scaling_efficiency` — must be above configurable threshold (default 0.80 for FSDP, 0.75 for DeepSpeed)
- `throughput_p50` — must not regress more than X% from baseline
- `memory_peak_gb` — must not exceed configurable limit
- `step_time_p90_ms / step_time_p50_ms` — ratio above 2.0 flags stall behavior

### 5.2 Baseline Management

The perf-regression-gate skill maintains a baseline `metrics.json` per strategy and world_size combination. When the benchmark runs, it compares the new metrics against the baseline. Regressions are flagged and block CI.

**Update baseline:** Run `scripts/scaling_report.py --update-baseline` to promote the current metrics to the baseline. This should only be done when the regression is intentional (e.g., enabling activation checkpointing that reduces throughput but improves memory).

### 5.3 CI Integration

Add the following to the CI pipeline after code changes to distributed training components:

```yaml
- name: Run scaling benchmark
  run: |
    torchrun --nproc_per_node=1 scripts/scaling_report.py \
      --strategy fsdp --world_size 1 --output_dir /tmp/bench
    python scripts/validate_distributed.py
```

For multi-GPU CI environments, run with `--world_size 4` or `--world_size 8` to detect actual scaling regressions.
