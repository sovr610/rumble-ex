# Distributed Training (DDP/FSDP) + torch.compile Reference

## DDP + torch.compile

### Support Status

DDP (DistributedDataParallel) + torch.compile is officially supported as of PyTorch 2.0. The compiler captures the full forward pass including DDP's gradient communication hooks and can optimize across them.

### Correct Wrapping Order

**Compile BEFORE wrapping with DDP.** This is critical.

```python
# CORRECT: compile first, then DDP
model = torch.compile(model, mode="default")
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

# INCORRECT: DDP first, then compile
# model = torch.nn.parallel.DistributedDataParallel(model)
# model = torch.compile(model)  # captures DDP's Python shell, not the inner model
```

**Why order matters:**
- `torch.compile` captures the forward graph of whatever module you hand it.
- If you compile the DDP wrapper, Dynamo sees DDP's Python boilerplate first.
- Compiling the inner model before wrapping lets Dynamo capture the actual compute graph. DDP then adds its allreduce hooks onto the compiled backward.
- The backward allreduce hooks from DDP are compatible with AOTAutograd's captured backward graph.

### DDP + reduce-overhead Mode

CUDA graphs under `reduce-overhead` can interact with DDP's gradient bucketing. The default DDP gradient bucket size (`bucket_cap_mb=25`) is usually fine, but:

```python
# If CUDA graph capture fails with DDP + reduce-overhead, try:
model = torch.nn.parallel.DistributedDataParallel(
    compiled_model,
    device_ids=[local_rank],
    bucket_cap_mb=0,         # Disable gradient bucketing (use per-param allreduce)
    static_graph=True,       # DDP optimization: tell DDP the graph is static
)
```

`static_graph=True` tells DDP the set of used parameters and their usage order won't change. This enables DDP optimizations that complement CUDA graph capture.

### DDP Logging and Graph Breaks

DDP inserts logging/profiling hooks that can cause graph breaks:

```python
# Suppress DDP profiling to avoid graph breaks
model = torch.nn.parallel.DistributedDataParallel(
    compiled_model,
    device_ids=[local_rank],
    find_unused_parameters=False,  # Set True only if you have unused params
)
```

`find_unused_parameters=True` causes DDP to traverse the autograd graph to find unused parameters. This traversal can cause graph breaks. Only use if you actually have unused parameters.

### Gradient Clipping with DDP + compile

```python
# This works correctly
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

Gradient clipping via `clip_grad_norm_` operates on `.grad` tensors outside the compiled graph, so it doesn't interact with Dynamo. No special handling needed.

### Multi-GPU Smoketest

Always test DDP + compile with actual distributed setup, not single-GPU emulation:

```bash
# Smoketest with 2 GPUs
torchrun --nproc_per_node=2 scripts/validate_compile.py --distributed

# Full smoketest
torchrun --nproc_per_node=8 scripts/validate_compile.py --distributed --full
```

Single-GPU compile tests do NOT cover DDP-specific graph breaks or allreduce interactions.

---

## FSDP + torch.compile

### Support Status

FSDP1 (original `torch.nn.parallel.FullyShardedDataParallel`) has limited compile support with known issues. **FSDP2** (`torch.distributed._composable.fsdp.fully_shard`) has significantly better compile support and is the recommended path for FSDP + compile.

### FSDP2 (Recommended)

```python
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

# Apply FSDP2 sharding at the block level
mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32
)

for block in model.transformer.blocks:
    fully_shard(block, mp_policy=mp_policy)

# Shard the full model
fully_shard(model, mp_policy=mp_policy)

# NOW compile the FSDP2-wrapped model
model = torch.compile(model, mode="default")
```

With FSDP2, the composable API allows Dynamo to capture the forward including weight all-gather and backward including gradient reduce-scatter.

### FSDP1 (Legacy)

```python
from torch.nn.parallel import FullyShardedDataParallel as FSDP

# With FSDP1, selective block compilation is more reliable
# Compile inner blocks BEFORE FSDP wrapping
for i, block in enumerate(model.transformer.blocks):
    model.transformer.blocks[i] = torch.compile(block, mode="default")

# Wrap with FSDP1 after block compilation
model = FSDP(model, ...)
```

**FSDP1 known issues with compile:**
- `auto_wrap_policy` configurations that use Python-level module inspection can cause graph breaks.
- State dict operations (`model.state_dict()`, `model.load_state_dict()`) during training (e.g., periodic checkpointing) can interfere. Always call these in a `torch._dynamo.disable` context.
- `sharding_strategy=ShardingStrategy.FULL_SHARD` has better support than `SHARD_GRAD_OP`.

---

## Selective Block Compilation Strategy

When full-model compilation fails with DDP or FSDP, fall back to compiling only the inner blocks:

```python
def apply_selective_compile(model, mode="default", backend="inductor"):
    """
    Compile only transformer blocks, leaving wrapper infrastructure eager.
    Useful when DDP/FSDP + full-model compile causes issues.
    """
    # Compile individual transformer blocks
    for i, block in enumerate(model.transformer.blocks):
        model.transformer.blocks[i] = torch.compile(
            block,
            mode=mode,
            backend=backend,
        )

    # Embeddings and output head can also be compiled if stable
    if hasattr(model, 'embed_tokens'):
        model.embed_tokens = torch.compile(model.embed_tokens, mode="default")

    # LM head / output projection
    if hasattr(model, 'lm_head'):
        model.lm_head = torch.compile(model.lm_head, mode="default")

    return model

# Usage
model = apply_selective_compile(model, mode="default")
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
```

**Trade-offs of selective compilation:**
- Python overhead at block boundaries (one Python call per block).
- Cross-block fusions (e.g., fusing the residual add at block N with the layernorm at block N+1) are not possible.
- Still captures most compute (attention + MLP) in compiled graphs.
- More reliable with DDP/FSDP because the distributed wrappers stay in eager.

---

## Gradient Checkpointing + compile

Gradient checkpointing (`torch.utils.checkpoint.checkpoint`) re-runs the forward pass during backward to save activation memory.

```python
from torch.utils.checkpoint import checkpoint

class TransformerBlock(nn.Module):
    def forward(self, x):
        return checkpoint(self._inner_forward, x, use_reentrant=False)

    def _inner_forward(self, x):
        # ... actual computation
        return x

# Compile works with checkpoint
model = torch.compile(model, mode="default")
```

**Important**: Use `use_reentrant=False` for better compile compatibility. The reentrant implementation uses Python-level control flow that can cause graph breaks.

---

## Common Distributed + Compile Issues

### Issue: Graph break from DDP profiling hooks

**Symptom**: `TORCH_LOGS=graph_breaks` shows breaks at DDP internal hooks.

**Fix:**
```python
# Disable DDP profiling
torch.distributed.set_debug_level(torch.distributed.DebugLevel.OFF)

# Or patch the DDP module to use compiler.disable
model._ddp_params_and_buffers_to_ignore = []  # Ensure this is set
```

### Issue: FSDP state dict during training causes graph break

**Symptom**: `checkpointing_handler()` call causes graph break.

**Fix:**
```python
@torch.compiler.disable
def save_checkpoint(model, path, step):
    """Always save checkpoints outside compiled context."""
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
        state = model.state_dict()
    torch.save(state, path)
```

### Issue: reduce-overhead fails with DDP allreduce

**Symptom**: `perf_hints` shows "Unable to capture CUDA graph due to allreduce sync".

**Fix:** DDP's gradient allreduce is a CPU-GPU sync point. Use `mode="default"` with DDP instead of `reduce-overhead`. Or use `static_graph=True` and test if CUDA graphs can capture around the allreduce.

### Issue: FSDP weight all-gather causes recompile

**Symptom**: `recompiles` log shows recompilation on every step.

**Fix:** FSDP2 handles this better. With FSDP1, try compiling only after FSDP wrapping (FSDP1 wrapping introduces dynamic control flow that Dynamo can handle with `dynamic=None`):
```python
model = FSDP(model, ...)
model = torch.compile(model, dynamic=None, mode="default")
```

---

## DDP/FSDP + compile Integration Checklist

Before deploying distributed compile:

- [ ] Tested with `validate_compile.py --distributed` on actual multi-GPU setup
- [ ] Verified wrapping order (compile before DDP/FSDP wrap)
- [ ] Ran `TORCH_LOGS=graph_breaks` to check for distributed-specific breaks
- [ ] Ran `TORCH_LOGS=recompiles` for at least 100 steps to confirm shape stability
- [ ] Compared output tensors: eager vs compiled on same random seed
- [ ] Measured peak memory on all ranks (compiled memory must fit)
- [ ] Confirmed gradient norms are equivalent between eager and compiled runs
- [ ] Verified checkpoint save/load works (`@torch.compiler.disable` on checkpoint functions)
- [ ] Ran smoketest with `CompileSmoketest` in distributed mode
- [ ] Loss curves match between eager and compiled for at least 500 steps
