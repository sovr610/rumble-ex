# Distributed Training Integration

## Overview

Gradient checkpointing interacts with distributed training wrappers (FSDP, DDP) in specific ways. This reference covers the integration points, wrapping order, and edge cases for brain_ai's multi-GPU training pipeline.

## FSDP Activation Checkpointing

### FSDP's Built-in API

FSDP provides `apply_activation_checkpointing` which integrates checkpointing with FSDP's parameter sharding:

```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

# Wrap specific layer types
apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=checkpoint_wrapper,
    check_fn=lambda submodule: isinstance(submodule, TransformerBlock),
)
```

### Why Use FSDP's API Instead of Manual Wrapping

1. **Wrapping order**: FSDP's API ensures checkpointing is applied to the correct level of the module hierarchy (inside FSDP unit boundaries, not across them).
2. **Communication coordination**: FSDP needs to know which activations are checkpointed to schedule parameter all-gathers correctly during recomputation.
3. **Memory accounting**: FSDP's memory tracking accounts for checkpointed vs non-checkpointed layers.

### FSDP Wrapping Order

The correct order is:

```
1. Build model
2. Apply activation checkpointing (wraps individual blocks)
3. Apply FSDP wrapping (wraps the model or sub-modules with FSDP)
```

Applying FSDP first and then checkpointing can break FSDP's internal assumptions about module structure.

### FSDP + Selective Checkpointing

Combine FSDP's `check_fn` with profiling data:

```python
# Profile on a single GPU first (before FSDP)
profiler = MemoryProfiler(model, device)
report = profiler.profile(sample_input)
expensive_layers = set(profiler.recommend_layers(memory_threshold_mb=50.0))

def check_fn(submodule):
    # Get the layer name from the model tree
    for name, mod in model.named_modules():
        if mod is submodule and name in expensive_layers:
            return True
    return False

apply_activation_checkpointing(model, check_fn=check_fn)
```

### FSDP CheckpointImpl Options

```python
class CheckpointImpl(Enum):
    REENTRANT = "reentrant"
    NO_REENTRANT = "no_reentrant"
```

**Recommendation**: Always use `NO_REENTRANT` (non-reentrant). Reentrant checkpointing has known issues with FSDP parameter prefetching and can cause deadlocks with certain model topologies.

## DDP Integration

### Standard DDP

With DistributedDataParallel, gradient checkpointing works transparently because DDP operates on gradients (output of backward), not activations (intermediate tensors):

```python
# Build model
model = MyModel()

# Apply checkpointing
for name, child in model.named_children():
    if should_checkpoint(name):
        setattr(model, name, CheckpointWrapper(child))

# Wrap with DDP
model = DDP(model, device_ids=[local_rank])
```

DDP's gradient allreduce buckets see the same gradient tensors regardless of whether activations were stored or recomputed.

### DDP + Gradient Accumulation

When using gradient accumulation with DDP:

```python
for micro_step in range(accumulation_steps):
    with model.no_sync() if micro_step < accumulation_steps - 1 else nullcontext():
        output = model(micro_batch)  # checkpointed forward
        loss = criterion(output, target)
        (loss / accumulation_steps).backward()  # checkpointed backward

optimizer.step()
optimizer.zero_grad()
```

Checkpointing composes correctly with gradient accumulation:
- Each micro-batch forward uses checkpointed forward (reduced memory).
- Each micro-batch backward recomputes activations then computes gradients.
- Gradients accumulate across micro-batches.
- `no_sync()` defers allreduce until the last micro-batch.

### DDP Bucketing Interaction

DDP organizes parameters into communication buckets for gradient allreduce. Checkpointing does not change the bucketing because:
- Buckets are determined by parameter order, not activation flow.
- Gradient tensors are the same size with or without checkpointing.
- The only change is timing: gradients from checkpointed layers arrive slightly later (recomputation delay).

If training throughput drops significantly with DDP + checkpointing, the issue is likely recomputation serializing with allreduce. Solutions:
1. Increase DDP bucket size (`bucket_cap_mb`) to allow more gradient accumulation before communication.
2. Use `static_graph=True` in DDP to enable communication/computation overlap optimization.

## Wrapping Order Summary

| Wrapper | Correct Order | Incorrect Order |
|---------|--------------|-----------------|
| DDP | Checkpoint -> DDP | DDP -> Checkpoint (may work but not recommended) |
| FSDP | Checkpoint -> FSDP | FSDP -> Checkpoint (breaks FSDP internals) |
| torch.compile | Checkpoint -> compile | compile -> Checkpoint (compile may not see through) |
| Mixed Precision (AMP) | Either order works | N/A |

## Multi-GPU Memory Budget

### Per-GPU Memory Breakdown

For a 7B model on 4x A100 80 GB with FSDP:

| Component | Per-GPU (no ckpt) | Per-GPU (with ckpt) |
|-----------|-------------------|---------------------|
| Parameters (sharded) | ~3.5 GB (fp16) | ~3.5 GB |
| Optimizer states (sharded) | ~7 GB (Adam fp32) | ~7 GB |
| Gradients (sharded) | ~3.5 GB | ~3.5 GB |
| Activations | ~40 GB (bs=8) | ~12 GB (selective) |
| FSDP communication buffers | ~3.5 GB | ~3.5 GB |
| **Total** | **~57.5 GB** | **~29.5 GB** |

With checkpointing, batch size 8 fits comfortably. Without it, batch size 4 is the maximum.

### Profiling Distributed Memory

Profile memory on a single GPU before distributed wrapping:

```python
# Single-GPU profiling
model = MyModel().cuda()
profiler = MemoryProfiler(model, torch.device('cuda'))
report = profiler.profile(sample_input)

# Apply checkpointing based on profile
checkpointer = SelectiveCheckpointer(config)
model = checkpointer.apply(model, report)

# Then wrap with FSDP/DDP
model = FSDP(model, ...)
```

Do NOT profile after FSDP wrapping; the sharding makes per-layer memory measurements inaccurate.

## Edge Cases

### Checkpointing with Pipeline Parallelism

When using pipeline parallelism (model split across GPUs by layer groups):
- Each stage independently checkpoints its layers.
- Cross-stage boundary activations are communicated, not checkpointed.
- The pipeline schedule (1F1B, GPipe) determines which activations are live simultaneously.

### Checkpointing with Tensor Parallelism

When attention heads or MLP columns are split across GPUs:
- Each rank checkpoints its shard of the layer.
- Recomputation on each rank recomputes only its shard.
- Communication ops (allreduce for tensor parallel) happen during both forward and recompute.

### DeepSpeed ZeRO Integration

DeepSpeed's ZeRO stages interact with checkpointing similarly to FSDP:
- ZeRO-1 (optimizer sharding): checkpointing works transparently.
- ZeRO-2 (gradient sharding): checkpointing works transparently.
- ZeRO-3 (parameter sharding): use DeepSpeed's own `deepspeed.checkpointing.checkpoint` for correct parameter gathering during recomputation.

## Troubleshooting

### Common Issues

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Hang during backward | Reentrant checkpoint + FSDP | Switch to `use_reentrant=False` |
| OOM despite checkpointing | Checkpointing applied after FSDP | Apply checkpointing before FSDP |
| Gradient mismatch across ranks | RNG state not preserved | Set `preserve_rng_state=True` |
| Slower than expected | All layers checkpointed | Use selective checkpointing |
| NaN gradients | Mixed precision recomputation | Ensure autocast context in recompute |
