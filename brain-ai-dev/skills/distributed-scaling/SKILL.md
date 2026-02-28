---
name: Distributed Scaling
description: >
  This skill should be used when the user asks to "scale training to multiple GPUs",
  "set up distributed training", "configure DDP", "use FSDP", "shard the model",
  "add gradient accumulation", "run with torchrun", "multi-node training",
  "rank-aware data loading", "distributed checkpointing", "scale to 7B parameters",
  "reduce GPU memory usage", "configure mixed precision distributed", or needs
  guidance on DistributedDataParallel, FullyShardedDataParallel, multi-GPU
  orchestration, or scaling the brain_ai system beyond single-GPU.
version: 0.1.0
---

# Distributed Scaling

## Overview

Guide implementation of distributed training infrastructure for the brain_ai system. The 7B-parameter production model cannot fit on a single GPU — FSDP sharding is required. Even smaller configurations (1B-3B) benefit from DDP for throughput. Cover DDP wrapping, FSDP sharding strategies, gradient accumulation, rank-aware data loading, distributed checkpointing, and multi-node orchestration.

## Public Contract

### DDPWrapper

Standard DistributedDataParallel integration for multi-GPU training.

```python
class DDPWrapper:
    def __init__(self, model: BrainAI, config: DistributedConfig): ...
    def setup(self, rank: int, world_size: int) -> nn.Module: ...
    def cleanup(self) -> None: ...
    def all_reduce_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]: ...
```

### FSDPWrapper

FullyShardedDataParallel for models exceeding single-GPU memory.

```python
class FSDPWrapper:
    def __init__(self, model: BrainAI, config: DistributedConfig): ...
    def setup(self, rank: int, world_size: int) -> nn.Module: ...
    def get_sharding_policy(self) -> ShardingPolicy: ...
    def save_distributed_checkpoint(self, path: str) -> None: ...
    def load_distributed_checkpoint(self, path: str) -> None: ...
```

### GradientAccumulator

Simulate larger batch sizes across accumulation steps.

```python
class GradientAccumulator:
    def __init__(self, accumulation_steps: int, scaler: Optional[GradScaler] = None): ...
    def step(self, loss: Tensor, optimizer: Optimizer, step_idx: int) -> bool: ...
    def effective_batch_size(self, micro_batch: int, world_size: int) -> int: ...
```

### DistributedLauncher

Unified launcher for torchrun, SLURM, and manual multi-node setups.

```python
class DistributedLauncher:
    def __init__(self, config: DistributedConfig): ...
    def launch(self, train_fn: Callable, args: Namespace) -> None: ...
    def detect_environment(self) -> str: ...  # "torchrun" | "slurm" | "manual"
```

### RankAwareDataLoader

Data loading that prevents duplicate samples across ranks.

```python
class RankAwareDataLoader:
    def __init__(self, dataset: Dataset, config: DistributedConfig): ...
    def get_sampler(self) -> DistributedSampler: ...
    def get_loader(self) -> DataLoader: ...
    def set_epoch(self, epoch: int) -> None: ...  # Critical for shuffle correctness
```

## Key Concepts

### Sharding Strategy by Model Scale

| Scale | Strategy | GPU Requirement |
|-------|----------|----------------|
| minimal (~1M) | Single GPU, no sharding | 1× any GPU |
| 1B | DDP across 2-4 GPUs | 2-4× 24GB |
| 3B | FSDP SHARD_GRAD_OP | 4-8× 40GB |
| 7B | FSDP FULL_SHARD | 8× 80GB (A100/H100) |

### FSDP Wrapping Policy

Wrap at module boundaries aligned with BrainAI architecture:
- Each encoder as a wrapping unit
- SNNCore as a wrapping unit
- HTM, Workspace, Decision, Reasoning, Meta as individual units
- Output heads as a single wrapping unit

### Gradient Accumulation Math

`effective_batch = micro_batch × accumulation_steps × world_size`

Learning rate scaling: linear scaling rule — `lr_scaled = lr_base × (effective_batch / reference_batch)` with warmup.

### Distributed Checkpointing

- **DDP**: Standard `model.module.state_dict()` on rank 0
- **FSDP**: Use `torch.distributed.checkpoint` for sharded saves, `StateDictType.FULL_STATE_DICT` for consolidated saves
- Phase boundaries: consolidate to full state dict for portability between distributed configs

## Configuration Surface

```python
@dataclass
class DistributedConfig:
    strategy: str = "ddp"               # ddp | fsdp | single
    fsdp_sharding: str = "full_shard"   # full_shard | shard_grad_op | no_shard
    gradient_accumulation_steps: int = 1
    find_unused_parameters: bool = False
    sync_batchnorm: bool = True
    backend: str = "nccl"               # nccl | gloo
    # Multi-node
    num_nodes: int = 1
    node_rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "29500"
    # Memory
    activation_checkpointing: bool = False
    cpu_offload: bool = False           # FSDP CPU offloading
    # Mixed precision
    mixed_precision_policy: str = "fp16"  # fp16 | bf16 | fp32
```

## Done-When Gates

1. **DDP Training** — `DDPWrapper.setup()` on 2+ GPUs produces identical per-step loss to single-GPU (within fp tolerance) at higher throughput; `all_reduce_metrics()` returns correct aggregated values.
2. **FSDP Sharding** — 7B model fits in 8× 40GB GPUs with FULL_SHARD; forward+backward completes without OOM; distributed checkpoint round-trips correctly.
3. **Rank-Aware Loading** — Each rank processes unique samples per epoch; `set_epoch()` changes shuffle order; no duplicate samples across world.

## Failure Modes

| Mode | Symptom | Fix |
|------|---------|-----|
| NCCL timeout | Hangs on all_reduce | Check network, increase NCCL_TIMEOUT, verify all ranks reach same point |
| OOM with FSDP | Still exceeds memory | Enable activation_checkpointing + cpu_offload |
| Gradient mismatch | DDP loss diverges from single-GPU | Set find_unused_parameters=True if model has conditional paths |
| Shuffle duplication | Same samples on multiple ranks | Ensure set_epoch() called each epoch on DistributedSampler |
| Checkpoint incompatible | Cannot load FSDP checkpoint in single-GPU | Use FULL_STATE_DICT consolidation at phase boundaries |

## Anti-Patterns

- Forgetting `set_epoch()` on the sampler — causes identical data order every epoch
- Saving optimizer state from all ranks — only rank 0 should save in DDP
- Using `model.state_dict()` instead of `model.module.state_dict()` with DDP
- Scaling learning rate without warmup — causes training divergence
- Mixing NCCL and Gloo backends without reason — use NCCL for GPU, Gloo for CPU

## Resources

### Reference Files
- **`references/ddp-integration.md`** — DDP setup, process groups, communication patterns
- **`references/fsdp-sharding.md`** — FSDP policies, wrapping strategies, memory analysis
- **`references/gradient-accumulation.md`** — Accumulation math, LR scaling, warmup schedules
- **`references/rank-aware-loading.md`** — Distributed samplers, worker seeding, epoch shuffling
- **`references/testing-matrix.md`** — Test scenarios for distributed infrastructure

### Asset Files
- **`assets/ddp_wrapper_template.py`** — DDPWrapper with process group management, self-tests
- **`assets/fsdp_wrapper_template.py`** — FSDPWrapper with sharding policies, self-tests
- **`assets/gradient_accumulator_template.py`** — GradientAccumulator with AMP integration
- **`assets/distributed_launcher_template.py`** — DistributedLauncher with environment detection
- **`assets/rank_aware_loader_template.py`** — RankAwareDataLoader with epoch management
- **`assets/distributed_config_template.py`** — DistributedConfig + validation + presets

### Scripts
- **`scripts/validate_distributed.py`** — Validates distributed infrastructure (simulated multi-GPU)
- **`scripts/gen_distributed_tests.py`** — Generates 100+ pytest test cases
- **`scripts/distributed_benchmark.py`** — Throughput and scaling efficiency benchmarks
