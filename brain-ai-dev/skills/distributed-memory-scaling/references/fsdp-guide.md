# FSDP Guide: Wrapping, Mixed Precision, Checkpointing, and Sharding Strategies

## Overview

PyTorch Fully Sharded Data Parallel (FSDP) shards model parameters, gradients, and optimizer states across all ranks in a process group. Unlike DDP, which replicates the full model on every GPU, FSDP allocates only a shard of each parameter tensor per rank and reconstructs full parameters just-in-time for the forward and backward passes via all-gather collectives. This document covers everything needed to configure FSDP correctly for production training.

---

## 1. Wrapping Policies

The wrap policy determines which `nn.Module` submodules become independent FSDP units. Choosing the wrong policy is the single most common source of poor FSDP scaling efficiency.

### 1.1 ModuleWrapPolicy (Transformer Block Class)

`ModuleWrapPolicy` is the preferred policy for transformer-based architectures. It accepts a set of module classes and wraps every instance of those classes as an individual FSDP unit.

```python
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

wrap_policy = ModuleWrapPolicy({TransformerBlock, TransformerLayer})
```

Each wrapped class becomes its own FSDP unit with independent all-gather and reduce-scatter collectives. The outer model receives an additional FSDP wrapper that handles any remaining parameters (embeddings, final layer norm, language model head).

**When to use:** Any model with clearly identified repeating block classes. For GPT-style models this is `TransformerBlock` or `Block`. For encoder-decoder models this is typically `EncoderLayer` and `DecoderLayer`. Identify the correct class name by inspecting `model.named_modules()` before wrapping.

**Effect on shard boundaries:** Each block shard is independent. During the forward pass, FSDP issues an all-gather for each block just before its computation and frees the gathered parameters immediately after, reclaiming memory before the next block's all-gather. This per-unit memory management is the source of FSDP's reduced peak memory footprint.

**Verification:** After wrapping, call `print(wrapped_model)` to confirm that each transformer block appears as `FullyShardedDataParallel(...)`. If blocks are not wrapped individually, the entire model is a single FSDP unit, which defeats the purpose and will likely OOM.

### 1.2 size_based_auto_wrap_policy

`size_based_auto_wrap_policy` wraps any submodule whose total parameter count exceeds a configurable threshold.

```python
from functools import partial
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=100_000_000)
```

This policy does not require knowing the class name of the block. It inspects parameter counts at wrap time and promotes any sufficiently large submodule to an FSDP unit.

**When to use:** Models with irregular architectures where block classes are not cleanly separated, or when wrapping third-party models without inspecting their internals. Use `min_num_params` in the range of 50M–200M parameters depending on total model size. If the threshold is too low, tiny modules become individual FSDP units and communication overhead dominates. If too high, few units are created and memory savings are limited.

**Limitation:** The policy can create inconsistent wrapping across different runs if the parameter count of submodules changes. It is also harder to reason about and audit than explicit class-based wrapping. Prefer `ModuleWrapPolicy` whenever the block class is known.

### 1.3 Custom Wrap Policies

Write a custom policy function with signature `(module, recurse, nonwrapped_numel) -> bool`. Return `True` to wrap the given module, `False` to recurse into its children.

```python
def custom_policy(module, recurse, nonwrapped_numel):
    if recurse:
        return True  # always recurse
    return isinstance(module, (TransformerBlock, EmbeddingLayer))
```

Custom policies are useful when mixing module types with different wrapping requirements in a single model.

---

## 2. Mixed Precision

FSDP has first-class support for mixed precision via the `MixedPrecision` dataclass. This is separate from `torch.autocast` and controls the dtype of parameters, gradients, and buffers at the FSDP communication level.

### 2.1 MixedPrecision Dataclass

```python
from torch.distributed.fsdp import MixedPrecision
import torch

bf16_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

fp16_policy = MixedPrecision(
    param_dtype=torch.float16,
    reduce_dtype=torch.float16,
    buffer_dtype=torch.float16,
)
```

- **`param_dtype`**: dtype in which parameters are stored and used during computation. Setting this to bf16 or fp16 reduces the all-gather communication volume by 2x compared to fp32.
- **`reduce_dtype`**: dtype used for gradient reduction (reduce-scatter). Can be set to fp32 for more stable gradient accumulation while keeping param_dtype at bf16 (mixed bf16 regime).
- **`buffer_dtype`**: dtype for registered buffers (running mean/variance in BatchNorm, etc.).

### 2.2 BF16 vs FP16 Considerations

**BF16 (recommended for Ampere/Hopper GPUs):**
- Same exponent range as float32 — no risk of overflow, no need for loss scaling
- Lower mantissa precision (7 bits vs 23 bits), but training stability is generally equivalent to fp32 for large models
- Requires hardware support: A100, A10, RTX 3090, H100, H800
- Set `param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16`

**FP16 (legacy, Volta/Turing GPUs):**
- Risk of gradient underflow and overflow; requires dynamic loss scaling via `GradScaler`
- FSDP + FP16 + GradScaler interaction is fragile. The scaler must unscale gradients before the optimizer step, but FSDP's reduce-scatter happens during backward. Set `reduce_dtype=torch.float32` to keep reductions in fp32, which avoids overflow in gradients.
- Set `param_dtype=torch.float16, reduce_dtype=torch.float32, buffer_dtype=torch.float32` for maximum stability

**True BF16 vs Mixed BF16:**
- True BF16: all operations in bf16, no fp32 master weights. Saves maximum memory, sufficient for most LLM training.
- Mixed BF16: params computed in bf16, master weights in fp32. Not directly supported through `MixedPrecision` — requires keeping a separate fp32 param copy outside FSDP. Rarely necessary with bf16.

### 2.3 Applying Mixed Precision

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(
    model,
    auto_wrap_policy=wrap_policy,
    mixed_precision=bf16_policy,
    device_id=torch.cuda.current_device(),
)
```

Pass `mixed_precision` at FSDP construction. The policy applies recursively to all nested FSDP units unless overridden at each individual wrap call.

---

## 3. Activation Checkpointing

Activation checkpointing (gradient checkpointing) trades compute for memory by discarding intermediate activations during the forward pass and recomputing them during backward. Combined with FSDP, this can reduce peak memory by an additional 30–60% at the cost of ~33% more compute.

### 3.1 Applying Activation Checkpointing

Use `apply_activation_checkpointing` from `torch.distributed.fsdp.wrap` (PyTorch 2.x) on the already-wrapped model. Target the same module class used for FSDP wrapping.

```python
from torch.distributed.fsdp.wrap import apply_activation_checkpointing

check_fn = lambda m: isinstance(m, TransformerBlock)
apply_activation_checkpointing(wrapped_model, check_fn=check_fn)
```

**Critical ordering:** Call `apply_activation_checkpointing` AFTER `FSDP(model, ...)`. Applying checkpointing before wrapping interferes with FSDP's internal hooks and produces incorrect gradients.

### 3.2 Memory vs Compute Tradeoff

At sequence length 2048 with a 7B parameter model:
- Without checkpointing: peak activation memory ~8–12 GB per GPU
- With checkpointing per transformer block: peak activation memory ~1–2 GB per GPU
- Compute overhead: ~33% more FLOPs for the forward pass (one extra forward per block per training step)

Enable activation checkpointing when memory budget is tight after FSDP wrapping. Disable it during inference to recover compute throughput (use `module.train(False)` to set inference mode without touching training hooks).

### 3.3 Interaction with Mixed Precision

When both mixed precision and activation checkpointing are active, recomputed activations use the same dtype as the original forward pass. If `param_dtype=torch.bfloat16`, recomputed activations are also bf16, which is memory-efficient and correct.

---

## 4. Sharding Strategies

FSDP supports four sharding strategies, selectable via the `ShardingStrategy` enum.

```python
from torch.distributed.fsdp import ShardingStrategy
```

### 4.1 FULL_SHARD

```python
strategy = ShardingStrategy.FULL_SHARD
```

Shards parameters, gradients, and optimizer states across all ranks. Each rank holds `1/world_size` of each tensor. Provides maximum memory reduction.

**All-gather behavior:** An all-gather runs before each forward pass (per FSDP unit) and before each backward pass. Parameters are freed after each usage. This results in 2x all-gather volume per step (forward + backward).

**Use when:** Model does not fit on a single GPU even at minimum batch size. This is the default and correct choice for training 7B+ parameter models on consumer or data-center GPUs.

### 4.2 SHARD_GRAD_OP

```python
strategy = ShardingStrategy.SHARD_GRAD_OP
```

Shards gradients and optimizer states, but keeps full parameter replicas on every rank during the forward and backward passes (parameters are not freed after the forward pass). Memory savings come only from optimizer state sharding (~4x reduction for Adam's two moments).

**Use when:** Model parameters fit in GPU memory when replicated, but optimizer states do not. This is intermediate between DDP and FULL_SHARD.

### 4.3 NO_SHARD

```python
strategy = ShardingStrategy.NO_SHARD
```

Equivalent to DDP. No sharding occurs. FSDP wrapping is applied but does not reduce memory. Use only for debugging or when transitioning a codebase to FSDP incrementally.

### 4.4 HYBRID_SHARD

```python
strategy = ShardingStrategy.HYBRID_SHARD
```

Shards parameters within each node (intra-node) but replicates across nodes (inter-node). Each node forms its own FSDP group. Gradient synchronization across nodes uses the DDP-style all-reduce, not the FSDP reduce-scatter.

**Use when:** Multi-node training where inter-node bandwidth is the bottleneck. HYBRID_SHARD reduces inter-node communication by only performing all-reduce (not all-gather) across nodes. Intra-node NVLink bandwidth handles the all-gather cost efficiently.

**Configuration:**

```python
from torch.distributed.fsdp import ShardingStrategy

model = FSDP(
    model,
    auto_wrap_policy=wrap_policy,
    sharding_strategy=ShardingStrategy.HYBRID_SHARD,
    device_id=torch.cuda.current_device(),
)
```

---

## 5. sync_module_states: Rank-0 Only Load + Broadcast Pattern

`sync_module_states=True` enables loading a checkpoint on rank 0 only and broadcasting parameters and buffers to all other ranks during FSDP construction. This avoids loading the full model on every GPU simultaneously, which is critical for large models.

### 5.1 Pattern

```python
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# Step 1: Create model with random init on all ranks (no checkpoint loading yet)
model = MyModel()

# Step 2: Load checkpoint on rank 0 only
if dist.get_rank() == 0:
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)

# Step 3: Wrap with FSDP and sync_module_states=True — broadcasts rank-0 weights to all ranks
model = FSDP(
    model,
    auto_wrap_policy=wrap_policy,
    sync_module_states=True,
    device_id=torch.cuda.current_device(),
)
```

### 5.2 Why This Matters

Loading a 70B parameter model on every rank of a 64-GPU cluster simultaneously would require 64 x 140 GB = 8.96 TB of CPU RAM for weight loading. The rank-0-only load pattern requires only 140 GB on the coordinator machine. After wrapping, FSDP's internal `sync_module_states` broadcast distributes the weights before sharding.

**Critical:** `sync_module_states=True` must be set at FSDP construction time, not added afterwards.

---

## 6. CPU Offload

`CPUOffload` moves parameters to CPU RAM when not in use, further reducing GPU memory at the cost of PCIe bandwidth.

```python
from torch.distributed.fsdp import CPUOffload

model = FSDP(
    model,
    auto_wrap_policy=wrap_policy,
    cpu_offload=CPUOffload(offload_params=True),
    device_id=torch.cuda.current_device(),
)
```

### 6.1 When CPU Offload Is Appropriate

CPU offload is a last resort. It incurs PCIe 4.0 bandwidth costs (~64 GB/s bidirectional) for every all-gather, which can reduce training throughput by 3–10x compared to GPU-only FSDP.

Use CPU offload only when:
1. The model cannot fit on the GPUs even with FULL_SHARD + activation checkpointing
2. Throughput is not a primary constraint (e.g., fine-tuning a very large model once with limited hardware)

Do not combine CPU offload with HYBRID_SHARD — the interaction is unsupported and produces undefined behavior in PyTorch 2.x.

---

## 7. Common Gotchas

### 7.1 Optimizer Must Be Created After Wrapping

Creating the optimizer before FSDP wrapping causes the optimizer to hold references to unsharded parameters. After wrapping, the model's parameter names and views change, and the optimizer will not update the correct tensors.

```python
# CORRECT
model = FSDP(model, ...)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# INCORRECT — optimizer references pre-wrap parameters
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
model = FSDP(model, ...)
```

The `StrategyRouter.build_optimizer()` method enforces this by accepting the already-wrapped model as input and refusing to accept an unwrapped model.

### 7.2 Print Wrapped Model to Verify Shard Boundaries

Always log the wrapped model structure on rank 0 to confirm that shard boundaries are placed at the intended module boundaries:

```python
if dist.get_rank() == 0:
    print(model)
```

Expected output for a correctly wrapped GPT model:
```
FullyShardedDataParallel(
  (_fsdp_wrapped_module): MyGPT(
    (embed): Embedding(...)
    (blocks): ModuleList(
      (0): FullyShardedDataParallel(TransformerBlock(...))
      (1): FullyShardedDataParallel(TransformerBlock(...))
      ...
    )
    (norm): LayerNorm(...)
    (lm_head): Linear(...)
  )
)
```

If all blocks appear inside a single outer FSDP unit without individual wrapping, the wrap policy is misconfigured.

### 7.3 Parameter Naming Changes After Wrapping

FSDP inserts `_fsdp_wrapped_module` into the parameter name path. A parameter originally named `blocks.0.attn.proj.weight` becomes `_fsdp_wrapped_module.blocks.0._fsdp_wrapped_module.attn.proj.weight` after wrapping.

This means loading a non-FSDP checkpoint (e.g., a full state dict from DDP training) directly into an FSDP-wrapped model will fail. Use the `sync_module_states` pattern (load into unwrapped model, then wrap) or use `FSDP.load_state_dict()` with `StateDictType.FULL_STATE_DICT`, which handles name remapping internally.

### 7.4 Barrier Before Save

Always insert `dist.barrier()` before saving a checkpoint to ensure all ranks have completed their gradient updates:

```python
dist.barrier()
if dist.get_rank() == 0:
    save_checkpoint(...)
```

For FULL_STATE_DICT saves, all ranks must participate in the collection even though only rank 0 writes to disk.

### 7.5 Mixed Precision and Layer Norm

Layer normalization layers contain `weight` and `bias` buffers. When `buffer_dtype=torch.bfloat16`, these buffers are cast to bf16. For numerical stability, some implementations keep layer norm in fp32. If training diverges with mixed precision, try `buffer_dtype=torch.float32` while keeping `param_dtype=torch.bfloat16`.

### 7.6 Gradient Accumulation

FSDP requires `no_sync()` context manager during gradient accumulation to suppress unnecessary reduce-scatter calls on intermediate micro-batches:

```python
for i, batch in enumerate(batches):
    if i < len(batches) - 1:
        with model.no_sync():
            loss = model(batch)
            loss.backward()
    else:
        loss = model(batch)
        loss.backward()

optimizer.step()
optimizer.zero_grad()
```

Without `no_sync()`, FSDP performs a full reduce-scatter after every micro-batch backward, which is wasteful and incorrect for gradient accumulation.
