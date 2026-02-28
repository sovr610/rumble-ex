---
name: Compute/Throughput Baseline & Regression Gate
description: >
  This skill should be used when the user asks to "add performance benchmarks",
  "create a regression gate", "measure training throughput", "compute MFU",
  "benchmark step time", "profile training loop", "set up CI perf gate",
  "compare against baseline", "collect environment info", "machine profile",
  "tokens per second measurement", "CUDA sync timing", "PyTorch profiler traces",
  "TensorBoard trace handler", "eval harness", "perplexity gate",
  "update performance baseline", "scaling efficiency test",
  or needs guidance on repeatable performance measurement, baseline storage,
  regression detection, profiling integration, or CI-gated throughput checks.
version: 0.1.0
---

# Compute/Throughput Baseline & Regression Gate

## Overview

Guide implementation of a repeatable performance+quality harness that runs fast enough for every PR, produces machine-readable artifacts, and blocks merges when throughput or quality regresses beyond defined tolerances. Cover environment capture, training micro-benchmarks with proper CUDA synchronization, deterministic eval probes, baseline storage keyed by machine profile, comparison logic with configurable tolerances, PyTorch Profiler integration, and CI workflow generation.

## Public Contract

### CollectEnv

Capture deterministic environment snapshot and machine profile string.

```python
class CollectEnv:
    def collect(self) -> Dict[str, Any]: ...  # Full env dict
    def machine_profile(self) -> str: ...     # e.g. "H100x8_driver550_cuda12.4_torch2.4_sm90"
    def save(self, path: str) -> None: ...    # Write env.json
```

### BenchTrain

Training micro-run benchmark with proper timing.

```python
class BenchTrain:
    def __init__(self, config: BenchConfig): ...
    def run(self) -> BenchResult: ...
    def save(self, path: str) -> None: ...    # Write metrics.json
```

### EvalSmall

Small, fixed, deterministic quality harness.

```python
class EvalSmall:
    def __init__(self, config: EvalConfig): ...
    def run(self) -> EvalResult: ...
    def save(self, path: str) -> None: ...    # Write eval.json
```

### CompareBaseline

Baseline comparison with configurable tolerances and pass/fail.

```python
class CompareBaseline:
    def __init__(self, tolerances: ToleranceConfig): ...
    def compare(self, current: str, baseline: str) -> CompareResult: ...
    def update_baseline(self, current: str, dest: str) -> None: ...
```

### BenchRunner

Unified entrypoint orchestrating all sub-modes.

```python
class BenchRunner:
    def run(self, modes: List[str], **kwargs) -> Dict[str, Any]: ...
```

## Key Concepts

### Two Benchmark Modes

| Mode | What It Measures | When to Gate |
|------|-----------------|-------------|
| `e2e` | Real dataloader + full step | Data path is stable |
| `synthetic` | No I/O, pure training loop | Data path noisy; alert on stalls separately |

### Timing Mechanics (Critical)

Measure "end-to-end step time": forward + backward + optimizer step + distributed comm + gradient accumulation boundaries.

```
torch.cuda.reset_peak_memory_stats()        # Before warmup
[run warmup_steps]                           # Warm caches + JIT
torch.cuda.synchronize()                     # ← BEFORE timer
t0 = time.perf_counter()
[one full training step]
torch.cuda.synchronize()                     # ← AFTER step
step_time = time.perf_counter() - t0
```

Without CUDA sync, the benchmark measures kernel queueing, not execution.

### Throughput Metrics

- **tokens_per_step** = `global_batch_size * seq_len` (includes data-parallel aggregation)
- **tokens_per_sec** = `tokens_per_step / step_time_s`
- **MFU (Model FLOPs Utilization)**: `achieved_flops_per_s / (num_gpus * peak_flops_per_gpu)`

Two MFU estimators:
1. **Simple 6ND**: `flops_per_step = 6 * N_non_embedding_params * tokens_per_step`
2. **Transformer-aware** (optional): incorporates attention FLOPs

### Machine Profile Rule

Compare only within the same machine profile (GPU model/count, driver, CUDA, PyTorch build). Store baselines per profile. Otherwise: false failures.

### Aggregation for Gating

Gate on **medians (p50)** and optionally tails (p90) for jitter. Report: step_time mean/p50/p90, tokens/sec mean/p50/p10, memory peak, loss slope, MFU p50.

### Baseline Storage

```
bench/baselines/<machine_profile>.metrics.json
bench/baselines/<machine_profile>.eval.json
```

Baselines updated only on main branch or by explicit maintainer action.

### Comparison Rules (Defaults)

| Gate | Metric | Threshold | Action |
|------|--------|-----------|--------|
| Perf | tokens_per_sec_p50 drop | > 5% | FAIL |
| Perf | step_time_p50 increase | > 5% | FAIL |
| Perf | memory increase | > 10% | WARN |
| Quality | perplexity worsens | > 1.5% relative | FAIL |
| Quality | probe accuracy drops | > 2% absolute | FAIL |
| Stability | loss slope positive | > threshold | FAIL |

### PyTorch Profiler Integration

```python
with torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=W, warmup=WU, active=A, repeat=R),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(out_dir),
) as prof:
    for step in range(total):
        with torch.autograd.profiler.record_function("forward"):
            ...
        prof.step()
```

Wrap regions in `record_function("name")`: data, forward, loss, backward, optimizer_step, grad_sync.

### Scaling Efficiency

```
efficiency = tokens_sec_p50(world_size=N) / (N * tokens_sec_p50(world_size=1))
```

### Eval Harness Design

- Fixed shard bundled in-repo (no network drift)
- Fixed tokenizer + truncation rules
- Greedy decode (temperature=0), fixed seeds
- 2-3 task probes scored with exact-match or regex (no LLM-judge in CI)

## Configuration Surface

```python
@dataclass
class BenchConfig:
    warmup_steps: int = 200
    measure_steps: int = 100
    mode: str = "e2e"                    # "e2e" | "synthetic"
    world_size: int = 1                  # 1 | N | -1 for auto
    profile: str = "off"                 # "off" | "trace" | "tb"
    repeat: int = 1
    peak_tflops: Optional[float] = None  # Override for MFU
    mfu_estimator: str = "6ND"           # "6ND" | "transformer_aware"

@dataclass
class EvalConfig:
    fixed_shard_path: str = "bench/data/fixed_shard.txt"
    probes: List[str] = ("basic_reasoning_25", "format_following_30", "code_sanity_20")
    temperature: float = 0.0
    seed: int = 42

@dataclass
class ToleranceConfig:
    throughput_drop_pct: float = 5.0
    step_time_increase_pct: float = 5.0
    memory_increase_pct: float = 10.0
    ppl_increase_pct: float = 1.5
    probe_drop_abs: float = 2.0
    loss_slope_threshold: float = 0.001
```

## Done-When Gates

1. **Bench Run** — `python -m tools.bench.run --bench --eval` produces `metrics.json` and `eval.json` with stable schema; repeated runs on same hardware yield p50 within 3% variance.
2. **Profile Traces** — `--profile trace` produces `profile/` directory with traces viewable in TensorBoard via PyTorch's documented `tensorboard_trace_handler` flow.
3. **CI Gate** — `compare_baseline` correctly blocks when throughput drops > X% or eval worsens beyond tolerance on the same machine profile; passes when within tolerance.

## Resources

### Reference Files
- **`references/benchmark-harness.md`** — CUDA sync timing, tokens/sec, MFU estimators, aggregation, metrics.json schema
- **`references/eval-harness.md`** — Deterministic eval, perplexity, task probes, eval.json schema, seed management
- **`references/baseline-comparison.md`** — Storage format, comparison rules, tolerance tuning, update workflow, failure reporting
- **`references/profiling-mode.md`** — PyTorch profiler schedule, record_function ranges, TensorBoard export, Chrome traces
- **`references/testing-matrix.md`** — Test scenarios for all phases

### Asset Files
- **`assets/collect_env_template.py`** — CollectEnv with GPU detection, machine profile, env.json output
- **`assets/bench_train_template.py`** — BenchTrain with CUDA sync timing, MFU, aggregation, metrics.json
- **`assets/eval_small_template.py`** — EvalSmall with fixed-shard perplexity, task probes, eval.json
- **`assets/compare_baseline_template.py`** — CompareBaseline with tolerance rules, delta reporting, update mode
- **`assets/run_template.py`** — BenchRunner unified entrypoint for all sub-modes
- **`assets/perf_gate_config_template.py`** — All config dataclasses, GPU peak TFLOPS registry, defaults

### Scripts
- **`scripts/validate_perf_gate.py`** — Validates done-when gates
- **`scripts/gen_perf_tests.py`** — Generates 100+ pytest test cases
- **`scripts/ci_workflow_template.py`** — Generates `.github/workflows/perf_gate.yml`
