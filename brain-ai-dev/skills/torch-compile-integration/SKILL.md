---
name: Compiler & Kernel Fusion (torch.compile) Integration
description: >
  This skill should be used when the user asks to "enable torch.compile",
  "add compilation to training", "kernel fusion", "TorchDynamo integration",
  "TorchInductor optimization", "reduce-overhead mode", "max-autotune mode",
  "fix graph breaks", "compile health check", "shape stabilization",
  "dynamic shapes for compile", "bucketing for torch.compile",
  "compile allowlist", "compile blocklist", "compile smoketest",
  "CUDA graphs for training", "maybe_compile wrapper",
  "debug recompiles", "TORCH_LOGS compile", "compile + DDP",
  "compile + FSDP", "torch.compiler.disable",
  or needs guidance on torch.compile integration, shape management,
  compilation debugging, or safe fallback patterns.
version: 0.1.0
---

# Compiler & Kernel Fusion (torch.compile) Integration

## Overview

Guide implementation of an opt-in compilation layer using torch.compile (TorchDynamo + AOTAutograd + TorchInductor) for training and inference speedup. Cover the safe wrapper pattern (never breaks training), compilation modes, shape stabilization (bucketing + mark_dynamic), health checks, allowlist/blocklist for selective compilation, distributed training interactions (DDP/FSDP), debugging, and integration with the benchmark harness.

**Design principle**: Safe by default. If compilation fails or regresses, fall back to eager with explicit logs.

## Public Contract

### maybe_compile

Safe wrapper — the single entry point for all compilation.

```python
def maybe_compile(
    model: nn.Module,
    cfg: CompileConfig,
    logger: logging.Logger,
    sample_batch: Optional[Dict] = None,
) -> nn.Module: ...
```

### CompileConfig

All compilation flags in one place.

```python
@dataclass
class CompileConfig:
    enabled: bool = False
    mode: str = "default"           # default | reduce-overhead | max-autotune
    dynamic: Optional[bool] = None  # None=auto, False=static, True=force dynamic
    backend: str = "inductor"
    fullgraph: bool = False
    options: Optional[Dict] = None
    allowlist: List[str] = ()
    blocklist: List[str] = ()
    healthcheck: bool = True
    fail_policy: str = "fallback_eager"  # fallback_eager | raise
```

### ShapeStabilizer

Prevent recompile thrashing from variable sequence lengths.

```python
class ShapeStabilizer:
    def __init__(self, buckets: List[int] = (256, 512, 1024, 2048)): ...
    def bucket_batch(self, input_ids: Tensor) -> Tensor: ...
    def mark_dynamic_dims(self, batch: Dict[str, Tensor], dim: int = 1): ...
```

### CompileSmoketest

Validate compilation works before committing to a full run.

```python
class CompileSmoketest:
    def __init__(self, steps: int = 3): ...
    def run(self, model: nn.Module, sample_batch: Dict,
            optimizer: Optimizer, loss_fn: Callable) -> bool: ...
```

## Key Concepts

### Compilation Modes

| Mode | Behavior | Trade-off |
|------|----------|-----------|
| `default` | Basic TorchInductor fusion | Fastest compile, moderate speedup |
| `reduce-overhead` | Enables CUDA graphs | Lower overhead per step, more memory, stricter shape requirements |
| `max-autotune` | Autotuned kernels + CUDA graphs | Slowest compile, best steady-state throughput |

### The Safe Wrapper Pattern

```
maybe_compile(model, cfg, logger, sample_batch)
  1. Return model unchanged if cfg.enabled=False
  2. Try torch.compile(model, backend, mode, dynamic, fullgraph, options)
  3. If compile raises: log exception, return eager model
  4. If healthcheck=True and sample_batch provided:
     Run 1-3 training steps (forward → loss → backward → optimizer.step)
     If smoketest fails: log error, return eager model
  5. Return compiled model
```

Never hard-crash by default. Log: compile enabled, backend, mode, dynamic, fullgraph, success/failure, compile wall time, exception + stack on failure.

### Shape Stabilization (Priority Order)

**A) Bucketing (preferred for LLM training)**
Bucket examples by length into fixed bins (256/512/1024/2048). Pad within bucket. Result: small set of stable shapes, kernels stay efficient.

**B) Automatic dynamic shapes (`dynamic=None`, the default)**
Compile assumes static first. On shape change, installs guards and auto-generalizes.

**C) Pre-mark dynamic dimensions (advanced)**
```python
torch._dynamo.mark_dynamic(input_ids, dim=1, min=1, max=max_seq_len)
```
Must be called on input tensors **before** invoking compiled code (not inside forward).

**Avoid**: `dynamic=True` (blanket force) — PyTorch docs explicitly call this "not recommended" / testing-oriented.

### Allowlist / Blocklist

Compile only core compute modules (attention, MLP, layernorm, projections). Disable:
- Data preprocessing, tokenization
- Metrics/logging
- Text generation sampling loops (branchy, data-dependent)
- Custom ops not stable under Inductor

Use `@torch.compiler.disable` on known-problem functions. `recursive=False` to disable only the decorated function, not callees.

### Debugging Recompiles

| Env Variable | Purpose |
|-------------|---------|
| `TORCH_LOGS=dynamic` | Shows guard installation and recompile triggers |
| `TORCH_LOGS=perf_hints` | Diagnoses CUDA graph applicability |
| `TORCH_LOGS=recompiles` | Logs when and why recompiles happen |
| `torch._dynamo.config.suppress_errors = True` | Dev-only: continue on compile errors (not for production) |

### Distributed Training (DDP/FSDP)

torch.compile supports DDP and FSDP. Policy:
- Treat DDP/FSDP + compile as "supported but gated by smoketest"
- If wrapping creates issues: compile only inner transformer blocks, keep wrappers eager
- DDP: compile before wrapping, overlap/bucketing interactions may need tuning
- FSDP: configuration constraints exist; test with smoketest

### Compile Health Check

The smoketest runs 1-3 complete training steps:
1. forward → loss → backward → optimizer.step → zero_grad
2. Same device/precision/distributed wrapper as real training
3. Deterministic mini-batch (fixed tokens) for stability
4. Catches: compilation-time failures, backward failures, autograd interactions

**First step includes compilation time** — benchmark harness must warm up past this.

### Benchmark Integration

Run same benchmark twice: eager vs compiled. Report:
- Compile time (time-to-first-step)
- Steady-state tokens/sec (after warmup)
- Peak memory delta
- Recompile count

## Configuration Surface

```python
@dataclass
class CompileConfig:
    enabled: bool = False
    mode: str = "default"
    dynamic: Optional[bool] = None
    backend: str = "inductor"
    fullgraph: bool = False
    options: Optional[Dict] = None       # epilogue_fusion, shape_padding, etc.
    allowlist: List[str] = ()
    blocklist: List[str] = ()
    healthcheck: bool = True
    fail_policy: str = "fallback_eager"  # fallback_eager | raise
    smoketest_steps: int = 3

@dataclass
class BucketConfig:
    buckets: List[int] = (256, 512, 1024, 2048)
    pad_token_id: int = 0
```

## Done-When Gates

1. **Safe Fallback** — With `enabled=True` and a deliberately broken model (e.g., unsupported op), `maybe_compile` catches the error, logs it, and returns the original eager model without crashing.
2. **Smoketest Detects Failure** — `CompileSmoketest.run()` returns `False` when backward fails on a compiled model and `True` when compilation succeeds end-to-end.
3. **Shape Stability** — `ShapeStabilizer.bucket_batch()` produces padded tensors matching bucket boundaries; compiled model runs without recompiles across all bucket sizes.

## Resources

### Reference Files
- **`references/compile-modes.md`** — default/reduce-overhead/max-autotune behavior, CUDA graphs, backend options, mode selection guide
- **`references/shape-stabilization.md`** — Bucketing algorithm, mark_dynamic API, recompile avoidance, guard system, automatic dynamic shapes
- **`references/debugging-profiling.md`** — TORCH_LOGS flags, graph break diagnosis, suppress_errors, perf_hints, compile time measurement
- **`references/distributed-compile.md`** — DDP + compile, FSDP + compile, wrapping order, overlap/bucketing, selective block compilation
- **`references/testing-matrix.md`** — Test scenarios for all components

### Asset Files
- **`assets/compile_config_template.py`** — CompileConfig, BucketConfig dataclasses with validation
- **`assets/compile_wrap_template.py`** — maybe_compile safe wrapper with logging, timing, fallback
- **`assets/shape_stabilize_template.py`** — ShapeStabilizer with bucketing, mark_dynamic, pad/unpad
- **`assets/compile_smoketest_template.py`** — CompileSmoketest with forward-backward-optimizer validation
- **`assets/compile_benchmark_template.py`** — Eager vs compiled benchmark comparison, report generation

### Scripts
- **`scripts/validate_compile.py`** — Validates done-when gates
- **`scripts/gen_compile_tests.py`** — Generates 100+ pytest test cases
- **`scripts/compile_playbook.py`** — Generates docs/torch_compile.md playbook with gotchas and debug knobs
