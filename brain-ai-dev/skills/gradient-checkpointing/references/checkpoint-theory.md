# Gradient Checkpointing Theory

## Overview

Gradient checkpointing (also called activation checkpointing or activation recomputation) is a memory optimization technique that trades compute time for memory during backpropagation. Instead of storing all intermediate activations from the forward pass for use during the backward pass, checkpointing discards them and recomputes them on-the-fly when needed.

## The Memory Problem

During standard backpropagation through an N-layer network:

1. **Forward pass**: Compute activations layer by layer. Each layer's output must be retained for gradient computation during backward. Total stored activations = N layers worth.
2. **Backward pass**: Walk layers in reverse. Each layer needs its input activation (stored from forward) to compute gradients.

For a model with N layers, each producing activations of size A bytes:
- **Peak activation memory (standard)**: O(N * A)
- This is in addition to parameter memory and optimizer state memory.

For brain_ai at 7B parameters with 32 layers, activation memory can exceed 40 GB for batch size 8 with sequence length 2048, making training on a single 80 GB GPU impossible.

## The Checkpointing Solution

### Core Idea

Divide the N layers into segments. Store activations only at segment boundaries. During backward, recompute activations within each segment from its boundary input.

### Optimal Segmentation (sqrt(N) Strategy)

The classic result from Chen et al. (2016) "Training Deep Nets with Sublinear Memory Cost":

- Divide N layers into k segments of N/k layers each.
- Store k boundary activations during forward.
- During backward, recompute each segment's internal activations from its boundary input.
- Peak memory = k boundary activations + N/k activations for one segment being recomputed.
- Minimize: k + N/k is minimized when k = sqrt(N).

**Result**: Peak activation memory reduces from O(N * A) to O(sqrt(N) * A).

### Concrete Example

For brain_ai with N=32 transformer layers:
- **Without checkpointing**: Store 32 layers of activations.
- **With sqrt(N) checkpointing**: k = sqrt(32) ~ 6 segments of ~5 layers each. Store 6 boundary activations + 5 within-segment activations = 11 total. Memory ratio: 11/32 ~ 34% of original.

For N=64 layers: k = 8, store 8 + 8 = 16. Memory ratio: 16/64 = 25%.

For N=128 layers: k = 11, store 11 + 12 = 23. Memory ratio: 23/128 ~ 18%.

The savings improve with deeper models.

## Compute Overhead

### Full Checkpointing (Every Layer)

When every layer is checkpointed individually:
- Forward pass: computed once (same as without checkpointing).
- Backward pass: each layer's activations are recomputed once before computing gradients.
- Total forward computation: 2x (one original forward + one recompute during backward).
- Total compute overhead: ~33% of total training step time (since backward is typically ~2x forward).

Derivation:
- Standard: F (forward) + 2F (backward with stored activations) = 3F
- Checkpointed: F (forward) + F (recompute) + 2F (backward) = 4F
- Overhead: (4F - 3F) / 3F = 33%

### Selective Checkpointing

When only expensive layers are checkpointed:
- Overhead is proportional to the fraction of compute in checkpointed layers.
- If 60% of compute is in checkpointed layers, overhead is ~0.6 * 33% = ~20%.
- This is why selective checkpointing (profiling + threshold) outperforms full checkpointing.

### SAC (Op-Level) Overhead

With Selective Activation Checkpointing:
- Only recompute specific ops (e.g., matmuls) while keeping cheap ops' activations (e.g., norms).
- Overhead can drop to ~15-20% while still achieving significant memory savings.
- Requires PyTorch 2.x `context_fn` parameter.

## Memory Accounting

### What Counts as Activation Memory

Per layer, activation memory includes:
- Output tensor of the layer (needed as input to next layer and for backward).
- Any intermediate tensors saved for backward (e.g., softmax output, pre-activation values).
- Buffers allocated during forward that are retained by autograd.

### What Is NOT Activation Memory

- **Parameters**: Weight and bias tensors (stored regardless of checkpointing).
- **Optimizer states**: Adam momentum/variance (stored regardless).
- **Gradients**: Accumulated during backward (stored regardless, but checkpointing can reduce peak overlap).

### Measuring Activation Memory

```python
torch.cuda.reset_peak_memory_stats()
# Run forward pass
output = model(input)
peak_after_forward = torch.cuda.max_memory_allocated()
activation_memory = peak_after_forward - param_memory - optimizer_memory
```

## When to Use Gradient Checkpointing

### Use When

- Model does not fit in GPU memory at desired batch size.
- Training is memory-bound, not compute-bound.
- 33% compute overhead is acceptable for the memory savings.
- Model has 8+ layers (below this, savings are minimal).

### Do NOT Use When

- Model fits comfortably in GPU memory with desired batch size.
- Training is already compute-bound (checkpointing makes it slower with no benefit).
- Inference only (no backward pass, no activation storage needed).
- Very shallow models (< 4 layers) where sqrt(N) savings are negligible.

### Decision Table

| Model Params | Layers | GPU Memory | Batch Size | Recommendation |
|-------------|--------|-----------|------------|----------------|
| 1M | 4 | 16 GB | 64 | No checkpointing needed |
| 50M | 12 | 16 GB | 32 | Selective if tight |
| 350M | 24 | 40 GB | 16 | Selective recommended |
| 1B | 24 | 40 GB | 8 | Full or selective required |
| 7B | 32 | 80 GB | 4 | Full required, selective preferred |
| 7B | 32 | 80 GB | 16 | Full + gradient accumulation |

## Interaction with Other Memory Optimizations

### Mixed Precision (fp16/bf16)

Checkpointing composes well with mixed precision. Activations are stored and recomputed in the same precision as the original forward. Memory savings are multiplicative: fp16 halves activation size, checkpointing reduces layer count, total savings = 2x * sqrt(N) reduction.

### Gradient Accumulation

Checkpointing reduces per-micro-batch memory. Combined with gradient accumulation over K micro-batches, effective batch size scales as K while memory stays constant per micro-batch.

### FSDP Parameter Sharding

FSDP shards parameters across ranks. Checkpointing reduces activation memory per rank. These are orthogonal and compose: FSDP handles parameter memory, checkpointing handles activation memory.

### torch.compile

Checkpointing is compatible with `torch.compile` but requires `use_reentrant=False` (non-reentrant checkpointing). The compiler can see through checkpoint boundaries and potentially fuse operations across them.

## References

- Chen, T., Xu, B., Zhang, C., & Guestrin, C. (2016). Training Deep Nets with Sublinear Memory Cost. arXiv:1604.06174.
- PyTorch documentation: `torch.utils.checkpoint` module.
- Korthikanti, V., et al. (2022). Reducing Activation Recomputation in Large Transformer Models. arXiv:2205.05198 (Selective Activation Checkpointing from Megatron-LM).
