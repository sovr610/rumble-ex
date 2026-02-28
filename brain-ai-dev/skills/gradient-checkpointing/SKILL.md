---
name: Gradient Checkpointing (Activation Recomputation)
description: >
  This skill should be used when the user asks to "enable gradient checkpointing",
  "reduce training memory", "activation checkpointing", "torch.utils.checkpoint",
  "memory-compute tradeoff", "checkpoint sequential layers",
  "selective checkpointing", "recomputation strategy",
  "activation memory profiling", "per-layer memory budget",
  "checkpoint_sequential", "checkpoint_wrapper",
  "SAC selective activation checkpointing", "SNN timestep checkpoint",
  "FSDP activation checkpointing", "checkpoint per timestep",
  "memory-efficient training", "recompute activations in backward",
  or needs guidance on trading compute for memory during training,
  per-layer memory profiling, selective recomputation strategies,
  or integration with distributed training wrappers.
version: 0.1.0
---

# Gradient Checkpointing (Activation Recomputation)

## Overview

Gradient checkpointing trades compute for memory by discarding intermediate activations during the forward pass and recomputing them on-the-fly during the backward pass. For the brain_ai architecture scaling from 1M to 7B parameters, this technique is essential: a 7B model with 32 transformer-style layers can reduce peak activation memory from O(N) to O(sqrt(N)) layers worth of stored activations, at the cost of roughly one extra forward pass (~33% compute overhead). Without checkpointing, large model training exceeds GPU memory budgets even on 80 GB A100s; with it, models that would require 4 GPUs fit on 1-2.

The core mechanism is `torch.utils.checkpoint.checkpoint`: wrap a function so that its intermediate tensors are freed after forward and recomputed from saved inputs during backward. PyTorch provides this at the function level (`checkpoint`), the sequential-model level (`checkpoint_sequential`), and in PyTorch 2.x at the operator level (Selective Activation Checkpointing / SAC). This skill covers all three granularities plus SNN-specific time-unrolling checkpointing for the spiking neural network components of brain_ai.

**Design principle**: Profile first, checkpoint selectively. Blindly checkpointing every layer wastes compute on cheap layers. Measure per-layer activation memory, then checkpoint only the layers whose activations dominate the memory budget.

## Public Contract

### CheckpointWrapper

Applies `torch.utils.checkpoint.checkpoint` to a module's forward method.

```python
class CheckpointWrapper(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        use_reentrant: bool = False,
        preserve_rng_state: bool = True,
    ): ...
    def forward(self, *args, **kwargs) -> Any: ...
```

### MemoryProfiler

Measures per-layer peak activation memory to guide selective checkpointing.

```python
class MemoryProfiler:
    def __init__(self, model: nn.Module, device: torch.device): ...
    def profile(self, sample_input: Dict[str, Tensor], num_runs: int = 3) -> ProfileReport: ...
    def recommend_layers(self, memory_budget_mb: float) -> List[str]: ...
```

### SelectiveCheckpointer

Wraps only the layers that exceed a memory threshold based on profiling data.

```python
class SelectiveCheckpointer:
    def __init__(
        self,
        config: CheckpointConfig,
        profiler: Optional[MemoryProfiler] = None,
    ): ...
    def apply(self, model: nn.Module, profile_report: Optional[ProfileReport] = None) -> nn.Module: ...
    def get_checkpointed_layers(self) -> List[str]: ...
```

### CheckpointConfig

All checkpointing parameters in one place.

```python
@dataclass
class CheckpointConfig:
    enabled: bool = False
    strategy: str = "selective"          # "none" | "full" | "selective" | "sequential"
    memory_threshold_mb: float = 50.0    # layers above this get checkpointed
    use_reentrant: bool = False          # False = newer non-reentrant (recommended)
    preserve_rng_state: bool = True
    snn_timestep_checkpoint: bool = False
    snn_chunk_size: int = 1              # timesteps per checkpoint segment
    profile_before_apply: bool = True
```

## Key Concepts

### Full Checkpointing

Wrap every layer with `torch.utils.checkpoint.checkpoint`. Simple but suboptimal: cheap layers (LayerNorm, dropout, embeddings) cost almost nothing to store but still pay recomputation overhead. Use this as a baseline or when profiling is unavailable.

```python
for name, child in model.named_children():
    setattr(model, name, CheckpointWrapper(child))
```

### Selective Checkpointing

Profile each layer's activation memory, then checkpoint only layers above a configurable threshold (e.g., 50 MB). This preserves most of the memory savings while avoiding needless recomputation of cheap layers. The `SelectiveCheckpointer` automates this: run `MemoryProfiler.profile()` to get per-layer costs, then `SelectiveCheckpointer.apply()` wraps only the expensive layers.

### checkpoint_sequential

For models built as `nn.Sequential`, `torch.utils.checkpoint.checkpoint_sequential` segments the sequential into chunks and checkpoints each chunk. The `segments` parameter controls granularity: more segments means more memory savings but more recomputation. Rule of thumb: `segments = int(math.sqrt(len(layers)))` achieves the sqrt(N) memory optimum.

### Memory Profiling

`MemoryProfiler` uses `torch.cuda.memory_stats()` and forward hooks to measure peak activation memory per layer. The workflow is: (1) run a forward pass with hooks that record `torch.cuda.max_memory_allocated()` delta per layer, (2) aggregate across multiple runs to reduce noise, (3) produce a `ProfileReport` with per-layer memory in MB, sorted descending. This report feeds into `SelectiveCheckpointer.recommend_layers()`.

### SAC (Selective Activation Checkpointing) - PyTorch 2.x

PyTorch 2.x introduces operator-level selective checkpointing via `torch.utils.checkpoint.checkpoint` with `context_fn` parameter. Instead of saving or recomputing all ops in a function, SAC allows specifying which ops to recompute (e.g., recompute matrix multiplications but keep normalization activations). This is the finest granularity available and can reduce overhead to ~15% vs the ~33% of full checkpointing.

```python
from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts

def policy_fn(ctx, op, *args, **kwargs):
    # Recompute matmuls (expensive to store), keep norms (cheap)
    if op in (torch.ops.aten.mm.default, torch.ops.aten.addmm.default):
        return CheckpointPolicy.MUST_RECOMPUTE
    return CheckpointPolicy.MUST_SAVE

checkpoint(fn, *args, use_reentrant=False,
           context_fn=create_selective_checkpoint_contexts(policy_fn))
```

### Integration with FSDP/DDP

FSDP provides its own activation checkpointing API via `apply_activation_checkpointing`. This integrates with FSDP's parameter sharding to coordinate memory across ranks. For DDP, standard `torch.utils.checkpoint` works directly since DDP does not shard parameters. When using gradient accumulation with DDP, checkpointed layers correctly accumulate gradients across micro-batches without interference. Key rule: apply checkpointing before wrapping with FSDP/DDP.

### SNN Time-Unrolling Checkpoint

Brain_ai's spiking neural network (SNN) components unroll across T timesteps. Without checkpointing, all T timesteps' activations are stored simultaneously, causing memory to scale as O(T * layer_size). With per-timestep checkpointing, only one timestep's activations are stored at a time; the others are recomputed during backward. For large T (100+), this reduces SNN memory from prohibitive to manageable. The `snn_chunk_size` parameter allows grouping multiple timesteps per checkpoint segment for a compute-memory tradeoff.

## Configuration Surface

```python
@dataclass
class CheckpointConfig:
    enabled: bool = False
    strategy: str = "selective"          # "none" | "full" | "selective" | "sequential"
    memory_threshold_mb: float = 50.0    # selective: checkpoint layers above this
    use_reentrant: bool = False          # False recommended (PyTorch >= 2.0)
    preserve_rng_state: bool = True      # True for dropout reproducibility
    snn_timestep_checkpoint: bool = False # enable per-timestep SNN checkpointing
    snn_chunk_size: int = 1              # timesteps per checkpoint segment
    sequential_segments: Optional[int] = None  # None = auto sqrt(N)
    profile_before_apply: bool = True    # run profiler before selective apply
    profile_num_runs: int = 3            # profiler averaging runs
    exclude_patterns: List[str] = ()     # layer name patterns to never checkpoint
    include_patterns: List[str] = ()     # layer name patterns to always checkpoint
```

## Done-When Gates

1. **Memory Reduction Verified** -- `CheckpointWrapper` wrapping a multi-layer model reduces `torch.cuda.max_memory_allocated()` during a forward-backward pass by at least 30% compared to the unwrapped model, while producing identical gradients (within float tolerance).

2. **Selective Profiling Works** -- `MemoryProfiler.profile()` returns a `ProfileReport` with per-layer memory values that are positive, sum to a reasonable total, and are reproducible across runs (< 10% variance). `recommend_layers()` returns only layers above the threshold.

3. **Gradient Correctness** -- Gradients from a checkpointed forward-backward pass match gradients from a non-checkpointed pass within `atol=1e-5` for float32 (or `atol=1e-2` for bfloat16). Verified for all three strategies: full, selective, and sequential.

## Resources

### Reference Files
- **`references/checkpoint-theory.md`** -- Recomputation math, memory savings analysis (sqrt(N) layers), compute overhead (~33%), when to use vs not
- **`references/selective-strategies.md`** -- Per-layer memory profiling, cost-benefit selection, SAC op-level checkpointing, policy functions
- **`references/distributed-integration.md`** -- FSDP activation checkpointing API, DDP gradient accumulation interaction, wrapping order
- **`references/testing-matrix.md`** -- Test scenarios for all components: wrapper, profiler, selective, SNN, config

### Asset Files
- **`assets/checkpoint_wrapper_template.py`** -- CheckpointWrapper applying torch.utils.checkpoint with self-tests
- **`assets/memory_profiler_template.py`** -- MemoryProfiler measuring per-layer activation memory with self-tests
- **`assets/selective_checkpointer_template.py`** -- SelectiveCheckpointer with cost-benefit analysis and self-tests
- **`assets/snn_timestep_checkpoint_template.py`** -- Checkpoint per SNN timestep with self-tests
- **`assets/checkpoint_config_template.py`** -- CheckpointConfig dataclass with validation and self-tests

### Scripts
- **`scripts/validate_checkpointing.py`** -- Validates done-when gates
- **`scripts/gen_checkpoint_tests.py`** -- Generates pytest test cases for all components
- **`scripts/memory_savings_benchmark.py`** -- Benchmarks memory savings across strategies and model sizes
