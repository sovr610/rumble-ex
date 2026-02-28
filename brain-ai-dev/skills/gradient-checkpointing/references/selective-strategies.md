# Selective Checkpointing Strategies

## Overview

Full gradient checkpointing (wrapping every layer) is a blunt instrument. It wastes compute recomputing cheap layers (LayerNorm, dropout, small projections) that cost almost nothing to store. Selective checkpointing targets only the expensive layers, achieving 80-95% of the memory savings at 50-70% of the compute overhead.

## Per-Layer Memory Profiling

### Profiling Method

The `MemoryProfiler` measures per-layer activation memory using CUDA memory statistics and forward hooks:

```python
def _layer_hook(name, report):
    def hook(module, input, output):
        mem_after = torch.cuda.max_memory_allocated()
        report[name] = mem_after - report['_baseline']
        report['_baseline'] = mem_after
    return hook
```

Steps:
1. Reset CUDA peak memory stats.
2. Record baseline memory (parameters + optimizer states).
3. Register forward hooks on all target layers.
4. Run a forward pass with a representative sample input.
5. Record peak memory delta per layer.
6. Repeat `num_runs` times and average to reduce noise.
7. Sort layers by activation memory, descending.

### Profiling Considerations

- **Warm-up**: Run one throwaway forward pass before profiling to let CUDA allocator settle.
- **Batch size**: Profile with the same batch size as training. Activation memory scales linearly with batch size, but relative ordering between layers is stable.
- **Sequence length**: Profile with representative sequence lengths. Attention layers scale quadratically (or linearly with flash attention) with sequence length.
- **Device**: Profile on the same GPU type as training. Memory allocator behavior varies across devices.

### Profile Report Schema

```python
@dataclass
class LayerProfile:
    name: str
    activation_memory_mb: float
    param_count: int
    flops_estimate: Optional[float] = None
    recompute_cost_ratio: Optional[float] = None  # recompute_time / store_cost

@dataclass
class ProfileReport:
    layers: List[LayerProfile]
    total_activation_mb: float
    peak_memory_mb: float
    model_param_mb: float
    timestamp: str
```

## Cost-Benefit Selection

### Threshold-Based Selection

The simplest strategy: checkpoint all layers whose activation memory exceeds a threshold.

```python
def recommend_layers(profile: ProfileReport, threshold_mb: float) -> List[str]:
    return [lp.name for lp in profile.layers if lp.activation_mb > threshold_mb]
```

**Choosing the threshold**: Start with `total_activation_mb / num_layers` (average per-layer cost). Layers above average are candidates. Adjust based on memory budget.

### Knapsack Selection

For more sophisticated selection, treat it as a constrained optimization:
- Each layer has a "memory cost" (activation_mb) and a "compute cost" (recompute time).
- Goal: minimize total compute cost subject to total stored activations fitting in memory budget.
- This is a 0-1 knapsack problem (checkpoint or not per layer).

For N < 100 layers, dynamic programming solves this exactly. For brain_ai's 32-64 layers, this is trivial.

```python
def knapsack_select(
    profile: ProfileReport,
    memory_budget_mb: float,
) -> List[str]:
    """Select layers to checkpoint to fit within memory_budget_mb."""
    layers = sorted(profile.layers, key=lambda l: l.activation_memory_mb, reverse=True)
    total = sum(l.activation_memory_mb for l in layers)
    if total <= memory_budget_mb:
        return []  # no checkpointing needed

    to_checkpoint = []
    saved = 0.0
    needed = total - memory_budget_mb
    for layer in layers:
        if saved >= needed:
            break
        to_checkpoint.append(layer.name)
        saved += layer.activation_memory_mb
    return to_checkpoint
```

### Typical Layer Costs in Transformer-style Models

| Layer Type | Relative Activation Memory | Checkpoint? |
|-----------|---------------------------|-------------|
| Multi-Head Attention | HIGH (stores Q, K, V projections, attention weights) | Yes |
| Feed-Forward / MLP | HIGH (stores pre-activation for GELU/SiLU backward) | Yes |
| LayerNorm / RMSNorm | LOW (stores normalized input) | Usually No |
| Dropout | NEGLIGIBLE (stores mask only) | No |
| Embedding | LOW to MODERATE (depends on vocab size) | Usually No |
| Residual Addition | NEGLIGIBLE | No |
| Output Projection | MODERATE | Sometimes |

## SAC (Selective Activation Checkpointing) - Op-Level

### Overview

Introduced in PyTorch 2.x and inspired by Megatron-LM's selective recomputation. Instead of deciding per-layer whether to checkpoint, SAC decides per-operation within a layer. This allows keeping cheap-to-store activations (norm outputs, dropout masks) while recomputing expensive-to-store activations (matmul outputs, attention scores).

### Policy Function API

```python
from torch.utils.checkpoint import (
    checkpoint,
    create_selective_checkpoint_contexts,
    CheckpointPolicy,
)

def policy_fn(ctx, op, *args, **kwargs):
    """Decide whether to save or recompute each op's output."""
    # Recompute large matmul outputs (expensive to store, cheap to recompute)
    if op in (
        torch.ops.aten.mm.default,
        torch.ops.aten.bmm.default,
        torch.ops.aten.addmm.default,
    ):
        return CheckpointPolicy.MUST_RECOMPUTE

    # Recompute attention score computation
    if op == torch.ops.aten._scaled_dot_product_flash_attention.default:
        return CheckpointPolicy.MUST_RECOMPUTE

    # Keep everything else (norms, activations, small ops)
    return CheckpointPolicy.MUST_SAVE

# Usage:
output = checkpoint(
    block.forward,
    *inputs,
    use_reentrant=False,
    context_fn=create_selective_checkpoint_contexts(policy_fn),
)
```

### Common SAC Policies

**Memory-aggressive** (maximize savings):
```python
def aggressive_policy(ctx, op, *args, **kwargs):
    # Only save tiny ops, recompute everything else
    SAVE_OPS = {
        torch.ops.aten.layer_norm.default,
        torch.ops.aten.dropout.default,
        torch.ops.aten.add.Tensor,
    }
    if op in SAVE_OPS:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.MUST_RECOMPUTE
```

**Compute-conservative** (minimize overhead):
```python
def conservative_policy(ctx, op, *args, **kwargs):
    # Only recompute the largest activations
    RECOMPUTE_OPS = {
        torch.ops.aten.mm.default,
        torch.ops.aten.bmm.default,
    }
    if op in RECOMPUTE_OPS:
        return CheckpointPolicy.MUST_RECOMPUTE
    return CheckpointPolicy.MUST_SAVE
```

### SAC Overhead Analysis

| Policy | Memory Savings vs Full | Compute Overhead |
|--------|----------------------|-----------------|
| Full checkpointing | 100% (baseline) | ~33% |
| SAC aggressive | ~85-90% | ~25% |
| SAC conservative | ~50-60% | ~10-15% |
| No checkpointing | 0% | 0% |

## Integration with Profiler

The `MemoryProfiler` can be extended to collect op-level information for SAC policy tuning:

```python
class OpLevelProfiler:
    def profile_ops(self, model, sample_input):
        """Profile memory per ATen op within each layer."""
        # Uses torch.profiler with record_shapes=True
        # Groups ops by parent module
        # Returns per-op memory and compute estimates
```

This data feeds back into SAC policy design: if a specific op in a specific layer dominates memory, the policy can target it precisely.

## Combining Strategies

For brain_ai's hybrid architecture (transformers + SNN + symbolic):

1. **Transformer blocks**: Use selective (threshold-based) at the layer level. Within checkpointed layers, optionally use SAC for finer control.
2. **SNN time-unrolled blocks**: Use per-timestep checkpointing (see SNN timestep checkpoint asset).
3. **Symbolic reasoning layers**: Usually small and cheap; do not checkpoint.
4. **Cross-module attention**: Profile and checkpoint if above threshold.

The `SelectiveCheckpointer` supports `exclude_patterns` and `include_patterns` to handle these heterogeneous components.
