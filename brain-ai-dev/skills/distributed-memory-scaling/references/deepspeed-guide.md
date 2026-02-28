# DeepSpeed Guide: ZeRO Stages, Config Generation, Offload, and Engine Lifecycle

## Overview

DeepSpeed implements the ZeRO (Zero Redundancy Optimizer) memory optimization framework. ZeRO progressively shards more state across data-parallel ranks to reduce per-GPU memory. Unlike FSDP, DeepSpeed is configured via a JSON config file that must be generated deterministically from your unified training config. This document covers ZeRO stages, config field semantics, offload policies, bucket tuning, and the DeepSpeed engine lifecycle.

---

## 1. ZeRO-2 vs ZeRO-3: What Gets Sharded

### 1.1 ZeRO-1 (Reference Only)

Shards optimizer states only. Rarely used directly — mentioned here for completeness. Each rank holds the full parameter set but only `1/world_size` of the optimizer states. Gradient reduction is still all-reduce.

### 1.2 ZeRO-2

Shards optimizer states AND gradients. Each rank holds:
- Full parameter replica (no parameter sharding)
- `1/world_size` of gradient buffers
- `1/world_size` of optimizer states (e.g., Adam first and second moments)

Memory reduction compared to DDP: approximately `(8 + K) / (8/world_size + K)` where K bytes is per-parameter for mixed precision. For a 7B model on 8 GPUs with Adam in fp16, this reduces optimizer + gradient memory by ~4x compared to DDP.

**Communication pattern:** Reduce-scatter gradients (each rank gets its shard), update optimizer, all-gather parameters before next forward pass. Same all-gather cost as FSDP `SHARD_GRAD_OP`.

**Use ZeRO-2 when:** Model parameters fit on each GPU when replicated, but optimizer states and gradients push memory over limit. ZeRO-2 has lower communication overhead than ZeRO-3 because parameters are never sharded.

### 1.3 ZeRO-3

Shards optimizer states, gradients, AND parameters. Each rank holds:
- `1/world_size` of each parameter tensor
- `1/world_size` of gradient buffers
- `1/world_size` of optimizer states

Maximum memory reduction: up to `world_size` fold reduction for a pure fp32 model. Equivalent to FSDP `FULL_SHARD` in sharding scope.

**Communication pattern:** All-gather parameters before each forward pass per partition, reduce-scatter gradients during backward, all-gather before optimizer step (or optimizer step on shards and broadcast). Total communication volume is higher than ZeRO-2.

**Use ZeRO-3 when:** Model parameters themselves exceed per-GPU memory. Also the only stage that supports `offload_param` to CPU/NVMe.

---

## 2. Config JSON Generation

DeepSpeed requires a JSON configuration dict passed to `deepspeed.initialize()`. Generate this deterministically from your `DeepSpeedConfig` dataclass — never hand-edit the JSON.

### 2.1 Minimal ZeRO-2 Config

```json
{
  "zero_optimization": {
    "stage": 2,
    "contiguous_gradients": true,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 500000000,
    "allgather_bucket_size": 500000000
  },
  "gradient_accumulation_steps": 1,
  "gradient_clipping": 1.0,
  "steps_per_print": 100,
  "wall_clock_breakdown": false
}
```

### 2.2 Minimal ZeRO-3 Config

```json
{
  "zero_optimization": {
    "stage": 3,
    "contiguous_gradients": true,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 500000000,
    "allgather_bucket_size": 500000000,
    "stage3_prefetch_bucket_size": 50000000,
    "stage3_param_persistence_threshold": 100000,
    "stage3_max_live_parameters": 1000000000,
    "stage3_max_reuse_distance": 1000000000
  },
  "gradient_accumulation_steps": 1,
  "gradient_clipping": 1.0,
  "steps_per_print": 100,
  "wall_clock_breakdown": false
}
```

### 2.3 All ZeRO Fields Documented

**Stage 2 and Stage 3 shared fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `stage` | int | 2 | ZeRO stage (2 or 3) |
| `contiguous_gradients` | bool | true | Copy gradients to contiguous buffer to avoid memory fragmentation during reduction |
| `overlap_comm` | bool | true | Overlap gradient reduction with backward computation (reduces communication idle time) |
| `reduce_scatter` | bool | true | Use reduce-scatter instead of all-reduce for gradients (enables gradient sharding) |
| `reduce_bucket_size` | int | 500M | Bytes per reduction bucket; larger reduces number of collective ops but increases memory pressure |
| `allgather_bucket_size` | int | 500M | Bytes per all-gather bucket for parameter reconstruction |

**Stage 3 only fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `stage3_prefetch_bucket_size` | int | 50M | Bytes to prefetch for the next parameter partition during backward |
| `stage3_param_persistence_threshold` | int | 100K | Parameters smaller than this threshold stay resident on all ranks (avoids all-gather for tiny params) |
| `stage3_max_live_parameters` | int | 1B | Maximum bytes of parameters kept gathered simultaneously; controls peak memory during forward |
| `stage3_max_reuse_distance` | int | 1B | If a parameter will be used again within this many bytes of other params, keep it gathered to avoid double all-gather |
| `stage3_gather_16bit_weights_on_model_save` | bool | false | Gather fp16 weights from all ranks before saving checkpoint (needed for portability) |

---

## 3. Bucket Tuning

Bucket sizes control the granularity of collective communication operations. Correct tuning is critical for throughput.

### 3.1 reduce_bucket_size

Controls how many bytes of gradients are accumulated before a reduce-scatter collective is launched. Larger buckets:
- Reduce the number of collective calls (lower per-call overhead)
- Require more memory for the gradient bucket buffer
- Increase communication latency if the backward pass is compute-bound (you wait longer before starting communication)

**Tuning rule:** Start at 500M (default). If GPU utilization is low during backward, reduce to 200M to start reductions earlier. If you see OOM during backward, reduce to 100M or lower.

### 3.2 allgather_bucket_size

Controls how many bytes of parameters are all-gathered at once during the forward pass (ZeRO-3 only). The same tradeoff applies: larger means fewer ops but more peak memory.

**Tuning rule:** Start at 500M. On machines with NVLink (A100 DGX), increase to 1B–2B. On PCIe-connected GPUs, reduce to 200M–300M to limit memory pressure from multiple simultaneous all-gathers.

### 3.3 stage3_prefetch_bucket_size

ZeRO-3 specific. Controls how much of the next parameter partition is prefetched during the backward pass of the current partition. Larger values improve hardware utilization by overlapping communication with compute.

**Tuning rule:** Set to `reduce_bucket_size / 10`. For `reduce_bucket_size=500M`, use `stage3_prefetch_bucket_size=50M`.

### 3.4 stage3_param_persistence_threshold

ZeRO-3 specific. Parameters below this byte threshold are kept resident on all ranks and never sharded. This avoids expensive all-gather collectives for small parameters like bias vectors or layer norm weights.

**Tuning rule:** Set to 100K (default) for most models. Increase to 1M if you have many small parameters (attention biases, positional embeddings) and want to keep them permanently resident.

---

## 4. Offload Policies

### 4.1 Optimizer Offload to CPU

```json
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    }
  }
}
```

Moves Adam first and second moments (and fp32 master weights if mixed precision) to CPU RAM. Available for both ZeRO-2 and ZeRO-3.

**Memory savings:** For Adam on a 7B model, optimizer states are ~56 GB (fp32 weights + 2 moments). Offloading frees this from GPU, allowing larger batch sizes.

**Throughput impact:** CPU Adam runs on CPU cores, which are ~10–50x slower than GPU for optimizer computation. Throughput reduction is typically 10–30% because the optimizer step is a small fraction of total training time (most time is forward + backward).

**When to use:** When optimizer states are the primary memory constraint and you have substantial CPU RAM (256+ GB). More acceptable than param offload because the optimizer step is sequential and does not block GPU computation.

### 4.2 Parameter Offload to CPU/NVMe (Stage 3 Only)

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    }
  }
}
```

For NVMe offload:
```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_param": {
      "device": "nvme",
      "nvme_path": "/local_nvme",
      "pin_memory": false,
      "buffer_count": 5,
      "buffer_size": 100000000
    }
  }
}
```

**Availability:** `offload_param` is ONLY valid for ZeRO-3. It is silently ignored or causes errors in ZeRO-2 config. Always validate that `offload_param != "none"` implies `zero_stage == 3`.

**Memory savings:** Can train models that do not fit in GPU memory at all — the GPU only holds the current partition being computed.

**Throughput impact:** CPU param offload costs PCIe bandwidth for every all-gather (~64 GB/s on PCIe 4.0). NVMe offload costs ~3–7 GB/s (sequential read speed). NVMe offload is 10–20x slower than CPU offload. These are last-resort options.

**Performance implications table:**

| Strategy | Memory Savings | Throughput Penalty |
|----------|---------------|-------------------|
| No offload | None | 0% |
| Optimizer to CPU | High (optimizer states) | 10–30% |
| Params to CPU (stage 3) | Maximum (GPU holds only active shard) | 50–200% |
| Params to NVMe (stage 3) | Maximum | 200–1000% |

---

## 5. DeepSpeed Engine Lifecycle

### 5.1 deepspeed.initialize()

```python
import deepspeed

engine, optimizer, _, scheduler = deepspeed.initialize(
    model=model,
    optimizer=optimizer,             # optional: pass None to let DS create optimizer
    lr_scheduler=scheduler,          # optional
    config=ds_config_dict,           # the generated JSON dict
)
```

`deepspeed.initialize()` wraps the model in a `DeepSpeedEngine`, which handles parameter sharding (ZeRO-3), gradient reduction (all stages), and optimizer stepping. After this call, `engine` replaces `model` for all training operations. The original `model` object should not be used directly.

**Passing `model_parameters`:** If using parameter groups with different learning rates, pass them as `model_parameters` instead of relying on the optimizer passed in:

```python
param_groups = [
    {"params": no_decay_params, "weight_decay": 0.0},
    {"params": decay_params, "weight_decay": 0.01},
]
engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=param_groups,
    config=ds_config_dict,
)
```

### 5.2 engine.step()

```python
engine.backward(loss)   # replaces loss.backward()
engine.step()           # replaces optimizer.step() + optimizer.zero_grad()
```

Never call `loss.backward()`, `optimizer.step()`, or `optimizer.zero_grad()` directly on a DeepSpeed-wrapped model. The engine handles all of these internally, including gradient scaling, gradient clipping, and ZeRO communication.

### 5.3 engine.save_checkpoint()

```python
engine.save_checkpoint(
    save_dir="/path/to/checkpoints",
    tag="step_1000",                  # subdirectory name
    save_latest=True,                 # write latest pointer file
    exclude_frozen_parameters=False,
)
```

This saves the sharded checkpoint in DeepSpeed's native format. The directory structure is:

```
/path/to/checkpoints/
  step_1000/
    mp_rank_00_model_states.pt       # model + optimizer state for rank 0
    mp_rank_01_model_states.pt       # ... rank 1
    ...
    zero_pp_rank_0_mp_rank_0_optim_states.pt   # ZeRO optimizer states
    ...
  latest                             # text file containing "step_1000"
```

### 5.4 engine.load_checkpoint()

```python
_, client_state = engine.load_checkpoint(
    load_dir="/path/to/checkpoints",
    tag="step_1000",                  # or None to load latest
    load_optimizer_states=True,
    load_lr_scheduler_states=True,
)
# client_state contains any metadata saved with the checkpoint
```

**Resume from latest:** Pass `tag=None` to automatically load the checkpoint pointed to by the `latest` file.

---

## 6. Integration with Custom Optimizers

### 6.1 Passing Your Own Optimizer

Pass an instantiated optimizer to `deepspeed.initialize()`. DeepSpeed will wrap it in a `DeepSpeedOptimizerCallable` that handles ZeRO state sharding:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
engine, ds_optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config=ds_config_dict,
)
```

After `deepspeed.initialize()`, the `optimizer` object passed in is replaced by the DeepSpeed-wrapped version. Use `ds_optimizer` (the second return value) for any direct optimizer access.

### 6.2 Letting DeepSpeed Create the Optimizer

Specify optimizer type in the JSON config and pass `optimizer=None`:

```json
{
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 1e-4,
      "weight_decay": 0.01,
      "betas": [0.9, 0.95],
      "eps": 1e-8
    }
  }
}
```

This is convenient but prevents using custom optimizer subclasses or per-parameter group learning rates. Prefer passing an explicit optimizer for production training.

### 6.3 Optimizer Creation Timing

Same invariant as FSDP: always create the optimizer BEFORE calling `deepspeed.initialize()`. DeepSpeed's `initialize()` will wrap the optimizer. If you need to create parameter groups, do so from the unwrapped model parameters, then pass both model and optimizer to initialize:

```python
# Create param groups from unwrapped model
param_groups = build_param_groups(model)
optimizer = torch.optim.AdamW(param_groups, lr=1e-4)

# Then wrap with DeepSpeed
engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config=config_dict,
)
# After this, use engine and optimizer (the wrapped versions)
```

---

## 7. Communication Optimization Flags

### 7.1 overlap_comm

When `True`, DeepSpeed overlaps gradient reduction with the backward computation of earlier layers. While layer N's backward pass computes gradients, layer N-1's gradients are being reduced in the background.

Set to `True` in all production configs. Disable only for debugging gradient correctness (overlap can mask timing-dependent bugs).

### 7.2 contiguous_gradients

When `True`, DeepSpeed copies gradients into a contiguous buffer before reduction. This eliminates memory fragmentation that accumulates from allocating and freeing gradient tensors of varying sizes during backward.

Set to `True` in all production configs. The memory overhead of the contiguous buffer is bounded by `reduce_bucket_size`.

### 7.3 reduce_scatter

When `True`, DeepSpeed uses reduce-scatter instead of all-reduce for gradient synchronization. Reduce-scatter naturally produces the gradient shards needed for ZeRO-2 and ZeRO-3 optimizer updates, avoiding an extra scatter step.

Must be `True` for ZeRO-2 and ZeRO-3 to function correctly. Setting to `False` reverts to ZeRO-1 behavior (all-reduce gradients, no gradient sharding).

---

## 8. Common Pitfalls

### 8.1 offload_param Requires stage 3

Setting `offload_param` in a ZeRO-2 config will either raise an error or be silently ignored depending on the DeepSpeed version. Always validate this constraint before generating the config JSON.

### 8.2 Config JSON Must Be a File or Dict

`deepspeed.initialize()` accepts either a path to a JSON file or a Python dict. Use the dict form (generated programmatically) for reproducibility. Writing to a temp file and passing the path is acceptable but introduces a potential race condition in multi-node training.

### 8.3 ZeRO-3 and nn.Module Attribute Access

With ZeRO-3, model parameters are not fully materialized on any rank during normal training. Accessing `model.layer.weight.data` directly will return only the local shard, not the full parameter. Use DeepSpeed's context manager for parameter gathering:

```python
with deepspeed.zero.GatheredParameters(model.layer.weight, modifier_rank=0):
    if dist.get_rank() == 0:
        weight_full = model.layer.weight.data.clone()
```

### 8.4 Mixed Precision in DeepSpeed

Configure mixed precision in the JSON config, not via `torch.autocast`:

```json
{
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  }
}
```

Or for bf16:
```json
{
  "bf16": {
    "enabled": true
  }
}
```

Do not enable both `fp16` and `bf16` simultaneously. The config generator must enforce mutual exclusion.
