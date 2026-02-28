---
name: V-JEPA 2 Distributed Training
description: >
  This skill should be used when the user asks to "distribute V-JEPA training",
  "SLURM job submission", "NCCL distributed setup", "checkpoint management",
  "preemption-safe training", "gradient scaler for AMP", "activation checkpointing",
  "torch.compile optimization", "SDPA fused attention", "memory optimization",
  "multi-GPU training", "submitit integration", "code snapshotting",
  "distributed all-gather", "custom autograd distributed ops",
  or needs guidance on distributed training infrastructure,
  checkpoint serialization, performance optimization, or SLURM integration
  for V-JEPA 2.
version: 0.1.0
---

# V-JEPA 2 Distributed Training

## Overview

Guide implementation of distributed training infrastructure for V-JEPA 2. Cover NCCL-based distributed setup (auto-detecting SLURM or local multi-GPU), custom autograd distributed ops (AllGather, AllReduceSum, AllReduce), SLURM job submission via submitit with preemption-safe checkpointing, code snapshotting for reproducibility, checkpoint serialization with retry logic, and performance optimizations (activation checkpointing, mixed precision, torch.compile, SDPA).

## Public Contract

### DistributedSetup

Initialize distributed training from environment.

```python
class DistributedSetup:
    def __init__(self): ...
    def init(self) -> Tuple[int, int, int]: ...  # (rank, local_rank, world_size)
    def is_slurm(self) -> bool: ...
    def cleanup(self) -> None: ...
```

### CustomDistributedOps

Autograd-compatible distributed communication.

```python
class AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): ...  # all_gather -> concat
    @staticmethod
    def backward(ctx, grad): ...  # all_reduce -> slice local

class AllReduceSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): ...  # all_reduce(sum)
    @staticmethod
    def backward(ctx, grad): ...  # identity

class AllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): ...  # all_reduce(sum) / world_size
    @staticmethod
    def backward(ctx, grad): ...  # identity
```

### SLURMSubmitter

Job submission and preemption management.

```python
class SLURMSubmitter:
    def __init__(self, config: SLURMConfig): ...
    def submit(self, train_fn: Callable, args: Dict) -> str: ...  # Returns job ID
    def snapshot_code(self, dest: str) -> None: ...
```

### CheckpointManager

Robust checkpoint save/load with retry logic.

```python
class CheckpointManager:
    def __init__(self, save_dir: str, max_retries: int = 5): ...
    def save(self, state: Dict, epoch: int) -> str: ...
    def load(self, path: str) -> Dict: ...  # Retries with exponential backoff
    def load_pretrained(self, path: str, strict: bool = False) -> Dict: ...
    def strip_prefix(self, state_dict: Dict, prefix: str = "module.") -> Dict: ...
```

### PerformanceOptimizer

Apply memory and compute optimizations.

```python
class PerformanceOptimizer:
    def __init__(self, model: nn.Module, config: PerfConfig): ...
    def enable_activation_checkpointing(self) -> None: ...
    def enable_mixed_precision(self) -> GradScaler: ...
    def compile_model(self) -> nn.Module: ...
    def measure_memory(self) -> Dict[str, float]: ...
```

## Key Concepts

### Distributed Initialization

Auto-detect SLURM via `SLURM_NTASKS`, `SLURM_PROCID`, `SLURM_LOCALID`.
Fallback to single-process if env vars not set.
Uses `dist.init_process_group(backend="nccl")`.
SLURM tmpdir for rendezvous when available.

### Custom Autograd Ops

Required because standard `dist.all_gather` doesn't propagate gradients:
- `AllGather`: Forward gathers tensors from all ranks, backward all-reduces then slices
- `AllReduceSum`: Forward sums across ranks, backward passes through (identity)
- `AllReduce`: Forward averages (sum/world_size), backward identity

### SLURM + submitit Pattern

- `Trainer` callable class with `__call__()` for training and `checkpoint()` for preemption recovery
- `checkpoint()` returns `submitit.helpers.DelayedSubmission` for auto-requeue
- Code snapshotted to experiment folder for reproducibility
- Job parameters: nodes, GPUs/node, memory, timeout, partition

### Checkpoint Format

```python
{
    "epoch": int,
    "encoder": state_dict,
    "predictor": state_dict,
    "target_encoder": state_dict,
    "opt": optimizer_state_dict,
    "scaler": grad_scaler_state_dict
}
```

### Checkpoint Loading Robustness

- Retry with exponential backoff: `2^n + random_jitter` seconds
- Handles transient NFS/distributed filesystem failures
- Key stripping: removes `module.` and `backbone.` prefixes
- `strict=False` for RoPE models (no `pos_embed` in checkpoint)

### Memory Optimizations

| Technique | Memory Saving | Throughput Impact |
|-----------|--------------|-------------------|
| Activation checkpointing | ~50% activation memory | ~20% slower |
| bfloat16 mixed precision | ~50% activation memory | ~10% faster |
| Token masking (75%) | ~75% attention memory | ~4x faster encoder |
| SDPA fused attention | No full attention matrix | ~30% faster attention |
| max_keep token cap | Bounded memory | Minimal |

### Compute Optimizations

- `torch.compile`: JIT compilation for 10-30% speedup
- `F.scaled_dot_product_attention`: auto-selects Flash Attention / memory-efficient
- SwiGLU 8-byte alignment: hidden dim aligned for GPU tensor cores

## Configuration Surface

```python
@dataclass
class SLURMConfig:
    nodes: int = 1
    gpus_per_node: int = 8
    mem_per_gpu: str = "64G"
    timeout_min: int = 4320
    partition: str = "learn"
    account: str = ""
    qos: str = ""

@dataclass
class PerfConfig:
    use_activation_checkpointing: bool = False
    use_bfloat16: bool = True
    compile_model: bool = False
    use_sdpa: bool = True
```

## Done-When Gates

1. **Distributed Init** — `DistributedSetup.init()` correctly detects SLURM or falls back to single-process; returns valid rank/world_size.
2. **AllGather Gradient** — `AllGather.apply(x)` in forward produces correct gathered tensor; backward correctly routes gradients to originating rank.
3. **Checkpoint Robustness** — `CheckpointManager.load()` succeeds after simulated transient failure (retry logic works); loaded state matches saved state exactly.

## Resources

### Reference Files
- **`references/distributed-setup.md`** — NCCL init, SLURM detection, local multi-GPU, process groups
- **`references/custom-autograd-ops.md`** — AllGather/AllReduceSum/AllReduce forward/backward
- **`references/slurm-submitit.md`** — submitit integration, preemption, code snapshotting
- **`references/checkpoint-format.md`** — Save/load protocol, key stripping, retry logic, pretrained loading
- **`references/testing-matrix.md`** — Test scenarios

### Asset Files
- **`assets/distributed_setup_template.py`** — DistributedSetup with SLURM detection
- **`assets/custom_dist_ops_template.py`** — AllGather, AllReduceSum, AllReduce autograd functions
- **`assets/slurm_submitter_template.py`** — SLURMSubmitter with Trainer callable, snapshotting
- **`assets/checkpoint_manager_template.py`** — CheckpointManager with retry, prefix stripping
- **`assets/perf_optimizer_template.py`** — PerformanceOptimizer with all optimization techniques

### Scripts
- **`scripts/validate_distributed.py`** — Validates done-when gates
- **`scripts/gen_distributed_tests.py`** — Generates 100+ pytest test cases
- **`scripts/perf_benchmark.py`** — Memory and throughput benchmarks with optimization combinations
