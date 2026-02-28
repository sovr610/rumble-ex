---
name: Distributed Memory Scaling (FSDP / DeepSpeed ZeRO)
description: >
  This skill should be used when the user asks to "enable FSDP",
  "add FSDP sharding", "use DeepSpeed ZeRO", "switch distributed strategy",
  "strategy router", "wrap model with FSDP", "FSDP mixed precision",
  "FSDP activation checkpointing", "FSDP checkpoint save/load",
  "sharded state dict", "full state dict offload", "sync_module_states",
  "DeepSpeed ZeRO-2", "DeepSpeed ZeRO-3", "generate DeepSpeed config JSON",
  "zero_to_fp32.py", "offload optimizer to CPU", "offload params NVMe",
  "portable checkpoint export", "cross-strategy checkpoint conversion",
  "scaling efficiency test", "1 GPU vs N GPU benchmark",
  "distributed.strategy config", "FSDP wrap policy", "module-class wrapping",
  "HYBRID_SHARD", "SHARD_GRAD_OP", "FULL_SHARD",
  "DDP vs FSDP vs DeepSpeed comparison", "reduce bucket size tuning",
  or needs guidance on strategy-pluggable distributed training,
  FSDP/DeepSpeed integration, checkpoint portability, or scaling harness.
version: 0.1.0
---

# Distributed Memory Scaling (FSDP / DeepSpeed ZeRO)

## Overview

Make the training stack "strategy-pluggable" so the same training script runs on plain DDP, PyTorch FSDP (full parameter/grad/optimizer sharding), or DeepSpeed ZeRO-2/ZeRO-3 purely by config. Standardize checkpoint export so weights move across strategies and optimizer state resumes within each family. Extend the benchmark harness with scaling-efficiency metrics.

Design principle: **one entrypoint, config-only strategy switch, portable weights, honest benchmarks.**

## Public Contract

### StrategyRouter

Single entrypoint owning wrapping order invariants.

```python
class StrategyRouter:
    def __init__(self, cfg: DistributedConfig): ...
    def setup_distributed(self) -> None: ...
    def wrap_model(self, model: nn.Module) -> Tuple[nn.Module, StrategyContext]: ...
    def build_optimizer(self, model: nn.Module, cfg: OptimizerConfig) -> Optimizer: ...
    def load_checkpoint(self, model: nn.Module, optimizer: Optimizer,
                        path: str) -> Optional[int]: ...
    def save_checkpoint(self, model: nn.Module, optimizer: Optimizer,
                        path: str, step: int) -> None: ...
    def export_portable_weights(self, model: nn.Module, path: str) -> None: ...
```

### FSDPWrapper

FSDP-specific wrapping, mixed precision, activation checkpointing, and checkpoint I/O.

```python
class FSDPWrapper:
    def wrap(self, model: nn.Module, cfg: FSDPConfig) -> nn.Module: ...
    def save_full_state_dict(self, model: nn.Module, path: str) -> None: ...
    def save_sharded_checkpoint(self, model: nn.Module, optimizer: Optimizer,
                                 path: str) -> None: ...
    def load_full_state_dict(self, model: nn.Module, path: str) -> None: ...
```

### DeepSpeedWrapper

DeepSpeed engine creation, JSON config generation, and checkpoint export.

```python
class DeepSpeedWrapper:
    def generate_config_json(self, cfg: DeepSpeedConfig) -> Dict: ...
    def wrap(self, model: nn.Module, cfg: DeepSpeedConfig,
             optimizer: Optional[Optimizer] = None) -> Any: ...
    def export_fp32_weights(self, checkpoint_dir: str, output_path: str) -> None: ...
```

### ScalingBenchmark

1-GPU vs N-GPU scaling efficiency measurement.

```python
class ScalingBenchmark:
    def run_single(self, cfg: BenchConfig) -> BenchResult: ...
    def run_multi(self, cfg: BenchConfig) -> BenchResult: ...
    def compute_efficiency(self, single: BenchResult,
                           multi: BenchResult) -> ScalingReport: ...
```

## Key Concepts

### Strategy Families

| Strategy | Shards | Memory Savings | When to Use |
|----------|--------|---------------|-------------|
| `ddp` | Nothing (full replica) | None | Model fits on 1 GPU |
| `fsdp` (FULL_SHARD) | Params + grads + optimizer | Maximum | Model does not fit on 1 GPU |
| `fsdp` (SHARD_GRAD_OP) | Grads + optimizer only | Medium | Params fit replicated |
| `fsdp` (HYBRID_SHARD) | Shard intra-node, replicate inter-node | Multi-node efficiency | Multi-node training |
| `deepspeed_zero2` | Grads + optimizer | Medium | ZeRO ecosystem preference |
| `deepspeed_zero3` | Params + grads + optimizer | Maximum | ZeRO ecosystem, offload needed |

### Wrapping Order Invariants

The strategy router enforces this exact sequence for all strategies:

1. `setup_distributed()` — init process group, set device
2. `build_model()` — create module on correct device
3. `wrap_model_by_strategy()` — returns wrapped model + context
4. `build_optimizer()` — over wrapped model's parameters
5. `load_checkpoint_if_any()` — strategy-aware load

Violating this order (e.g., creating optimizer before FSDP wrap) causes silent bugs or crashes.

### FSDP Wrap Policy

Default to **module-class wrapping** for LLMs — wrap each Transformer block as its own FSDP unit. Size-based wrapping is a fallback when clean block classes are unavailable. Print the wrapped model and verify shard boundaries.

### FSDP Mixed Precision

Set explicitly via FSDP's `MixedPrecision` config:
- **bf16 training** (Ampere/Hopper): param_dtype=bf16, reduce_dtype=bf16, buffer_dtype=bf16
- **fp16 training**: param_dtype=fp16, reduce_dtype=fp16, buffer_dtype=fp16; consider fp32 master weights for stability

Document the choice — "bf16 mixed" vs "bf16 true" changes memory and stability.

### FSDP Checkpointing (Two Tiers)

| Tier | Purpose | Method |
|------|---------|--------|
| Portable weights | Cross-strategy export | `FULL_STATE_DICT` with `offload_to_cpu=True, rank0_only=True` |
| Efficient resume | Same-strategy resume | Sharded state dict via `torch.distributed.checkpoint` |

Load path for portable weights: load on rank0, re-wrap with `sync_module_states=True` to broadcast.

### DeepSpeed Config Generation

Generate deterministic JSON from the unified config — never hand-edit. Key ZeRO fields:
- `zero_optimization.stage` (2 or 3)
- Stage 2/3: `contiguous_gradients`, `overlap_comm`, `reduce_scatter`, `reduce_bucket_size`, `allgather_bucket_size`
- Stage 3: `stage3_prefetch_bucket_size`, `stage3_param_persistence_threshold`, `stage3_max_live_parameters`, `stage3_max_reuse_distance`

### DeepSpeed Offload Policy

Default: **no offload**. If memory-bound:
1. Start with optimizer offload to CPU (less brutal)
2. Param offload to CPU/NVMe is last resort — costs throughput significantly

Require explicit config; never auto-enable offload.

### Checkpoint Portability Matrix

| From \ To | DDP | FSDP | DeepSpeed |
|-----------|-----|------|-----------|
| **Weights** | Direct | FULL_STATE_DICT export | `zero_to_fp32.py` export |
| **Optimizer** | Within family only | FSDP sharded ckpt | ZeRO sharded ckpt |

Weights-only portability is required across all strategies. Optimizer-state portability is within-family only, with documented "weights-only resume" path across families.

### Scaling Efficiency Measurement

```
scaling_efficiency = throughput_N / (N * throughput_1)
```

Run benchmark at `world_size=1` and `world_size=N`, report per strategy. This catches bad wrapping (tiny FSDP units), wrong bucket sizes, accidental offload, and OOM thrash (visible as huge p90 step-time).

## Configuration Surface

```python
@dataclass
class DistributedConfig:
    strategy: str = "ddp"                    # ddp | fsdp | deepspeed_zero2 | deepspeed_zero3
    world_size: int = -1                     # -1 = auto from launcher
    backend: str = "nccl"
    grad_accum: int = 1

@dataclass
class FSDPConfig:
    sharding_strategy: str = "FULL_SHARD"    # FULL_SHARD | SHARD_GRAD_OP | NO_SHARD | HYBRID_SHARD
    wrap_policy: str = "transformer_block"   # transformer_block | size_based
    wrap_module_classes: List[str] = ()      # e.g. ["TransformerBlock"]
    mixed_precision: str = "bf16"            # bf16 | fp16 | none
    activation_checkpointing: str = "off"    # off | transformer_block
    state_dict_type: str = "full"            # full | sharded
    sync_module_states: bool = True
    cpu_offload: bool = False

@dataclass
class DeepSpeedConfig:
    zero_stage: int = 3                      # 2 | 3
    offload_optimizer: str = "none"          # none | cpu | nvme
    offload_param: str = "none"              # none | cpu | nvme (stage 3 only)
    reduce_bucket_size: int = 500_000_000
    allgather_bucket_size: int = 500_000_000
    stage3_prefetch_bucket_size: int = 50_000_000
    stage3_param_persistence_threshold: int = 100_000
    overlap_comm: bool = True
    contiguous_gradients: bool = True
```

## Done-When Gates

1. **Strategy Switch is Config-Only** — Same entrypoint runs with `strategy=ddp`, `strategy=fsdp`, and `strategy=deepspeed_zero3` without code changes. Router correctly wraps, builds optimizer, and checkpoints for each path.
2. **Checkpoint Portability is Real** — FSDP produces portable weights via `FULL_STATE_DICT` (offload_to_cpu + rank0_only). DeepSpeed produces portable weights via `zero_to_fp32.py`. Both load into an unwrapped model successfully.
3. **Scaling Harness Produces Stable Metrics** — `metrics.json` includes strategy, world_size, scaling_efficiency, and memory_peak. Values are stable across repeated runs (p50 within 3% variance).

## Resources

### Reference Files
- **`references/fsdp-guide.md`** — FSDP wrapping policies, mixed precision, activation checkpointing, sharding strategies, sync_module_states, gotchas
- **`references/deepspeed-guide.md`** — ZeRO-2/3 config fields, offload policies, bucket tuning, DS engine lifecycle, JSON generation
- **`references/checkpoint-portability.md`** — FULL_STATE_DICT save/load, sharded checkpoint via DCP, zero_to_fp32.py, cross-strategy conversion paths
- **`references/scaling-benchmark.md`** — 1-GPU vs N-GPU methodology, efficiency formula, metrics schema, regression detection
- **`references/testing-matrix.md`** — Test scenarios for all components

### Asset Files
- **`assets/strategy_router_template.py`** — StrategyRouter with DDP/FSDP/DeepSpeed paths, wrapping order enforcement
- **`assets/fsdp_wrapper_template.py`** — FSDPWrapper with wrap policies, mixed precision, activation checkpointing, two-tier checkpoint
- **`assets/deepspeed_wrapper_template.py`** — DeepSpeedWrapper with JSON generation, engine creation, fp32 export
- **`assets/scaling_benchmark_template.py`** — ScalingBenchmark with single/multi-GPU measurement and efficiency computation
- **`assets/distributed_config_template.py`** — All config dataclasses, validation, serialization

### Scripts
- **`scripts/validate_distributed.py`** — Validates done-when gates
- **`scripts/gen_distributed_tests.py`** — Generates 100+ pytest test cases
- **`scripts/scaling_report.py`** — Runs scaling efficiency comparison and generates report
