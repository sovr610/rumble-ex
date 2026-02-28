# Checkpoint Portability: FSDP, DeepSpeed, and Cross-Strategy Conversion

## Overview

Checkpointing in distributed training has two distinct goals: efficient resume (same strategy, same world_size, continue training) and portable weights export (save weights in a format loadable by any strategy). These goals require different checkpoint formats. This document covers both paths for FSDP and DeepSpeed, cross-strategy conversion, optimizer state handling, and a compatibility matrix.

---

## 1. FSDP FULL_STATE_DICT: Portable Weight Save/Load

### 1.1 Save with FULL_STATE_DICT

```python
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig

save_policy = FullStateDictConfig(
    offload_to_cpu=True,   # gather all shards onto CPU to avoid GPU OOM
    rank0_only=True,       # only rank 0 receives the full state dict
)

with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
    state_dict = model.state_dict()

# Only rank 0 has a populated state_dict; other ranks have an empty dict
if dist.get_rank() == 0:
    torch.save(state_dict, "/path/to/portable_weights.pt")

dist.barrier()  # ensure rank 0 finishes writing before other ranks proceed
```

**What happens internally:** FSDP performs an all-gather across all ranks to reconstruct full parameters on CPU, then discards the shards. With `rank0_only=True`, only rank 0 receives the gathered parameters. This avoids duplicating the full model on every GPU.

**Parameter naming:** FSDP remaps parameter names to match the unwrapped module structure when `StateDictType.FULL_STATE_DICT` is active. The `_fsdp_wrapped_module` prefix is stripped. The resulting state dict is compatible with the original non-FSDP model's `load_state_dict()`.

### 1.2 Load FULL_STATE_DICT

**Pattern A — load into FSDP model directly (same topology):**

```python
load_policy = FullStateDictConfig(
    offload_to_cpu=True,
    rank0_only=True,
)

# Load full state dict on rank 0 only
if dist.get_rank() == 0:
    full_state = torch.load("/path/to/portable_weights.pt", map_location="cpu")
else:
    full_state = {}

with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, load_policy):
    model.load_state_dict(full_state)

dist.barrier()
```

FSDP's `load_state_dict` under `FULL_STATE_DICT` automatically scatters the full state dict from rank 0 to shards on all ranks.

**Pattern B — load into unwrapped model then re-wrap (cross-strategy):**

```python
# Create unwrapped model
model = MyModel()

# Load on rank 0 only
if dist.get_rank() == 0:
    state_dict = torch.load("/path/to/portable_weights.pt", map_location="cpu")
    model.load_state_dict(state_dict)

# Wrap with FSDP — sync_module_states broadcasts rank 0 weights to all ranks
model = FSDP(
    model,
    auto_wrap_policy=wrap_policy,
    sync_module_states=True,        # this is the key flag
    device_id=torch.cuda.current_device(),
)
```

Pattern B is more flexible: it works regardless of the checkpoint source (DDP, FSDP, or DeepSpeed fp32 export) as long as the state dict contains full parameter tensors with matching names.

---

## 2. FSDP Sharded Checkpoint: Efficient Resume via DCP

### 2.1 Save Sharded Checkpoint

PyTorch 2.x includes `torch.distributed.checkpoint` (DCP) for efficient sharded checkpointing. Each rank writes its own shard to a subdirectory.

```python
import torch.distributed.checkpoint as dist_cp
from torch.distributed.fsdp import StateDictType

with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
    state_dict = {
        "model": model.state_dict(),
        "optimizer": FSDP.optim_state_dict(model, optimizer),
        "step": current_step,
    }

dist_cp.save_state_dict(
    state_dict=state_dict,
    storage_writer=dist_cp.FileSystemWriter("/path/to/sharded_ckpt"),
)
dist.barrier()
```

The `FileSystemWriter` creates one file per rank-shard combination. The resulting directory contains metadata and per-shard binary files. All ranks participate in writing simultaneously.

### 2.2 Load Sharded Checkpoint

```python
import torch.distributed.checkpoint as dist_cp

# Create empty state dict with matching structure
with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
    state_dict = {
        "model": model.state_dict(),
        "optimizer": FSDP.optim_state_dict(model, optimizer),
        "step": 0,
    }

    dist_cp.load_state_dict(
        state_dict=state_dict,
        storage_reader=dist_cp.FileSystemReader("/path/to/sharded_ckpt"),
    )

    model.load_state_dict(state_dict["model"])
    FSDP.optim_state_dict_to_load(model, optimizer, state_dict["optimizer"])

current_step = state_dict["step"]
```

**Advantage of sharded checkpoint:** No all-gather communication required during save or load. Each rank reads only its own shard. Save/load time scales with per-rank shard size, not total model size.

**Limitation:** Sharded checkpoints are not portable across different `world_size` configurations. A checkpoint saved with `world_size=8` cannot be directly loaded into a `world_size=4` training run without resharding (use `torch.distributed.checkpoint.format_utils.dcp_to_torch_save()` for conversion).

---

## 3. DeepSpeed Checkpoint: Native Format and zero_to_fp32.py

### 3.1 Save with engine.save_checkpoint()

```python
engine.save_checkpoint(
    save_dir="/path/to/ds_checkpoints",
    tag="step_1000",
)
```

Directory structure after save:

```
/path/to/ds_checkpoints/
  step_1000/
    mp_rank_00_model_states.pt
    mp_rank_01_model_states.pt
    ...
    zero_pp_rank_0_mp_rank_00_optim_states.pt
    zero_pp_rank_1_mp_rank_00_optim_states.pt
    ...
  latest                         # contains "step_1000"
```

Each `mp_rank_XX_model_states.pt` contains the model state for that rank's pipeline-parallel partition and the ZeRO model shard. The `zero_pp_rank_*_optim_states.pt` files contain optimizer state shards.

### 3.2 Load with engine.load_checkpoint()

```python
_, client_state = engine.load_checkpoint(
    load_dir="/path/to/ds_checkpoints",
    tag="step_1000",
)
step = client_state.get("step", 0)
```

This restores model weights and optimizer state shards to the correct ranks. The world_size must match between save and load.

### 3.3 zero_to_fp32.py Consolidation

DeepSpeed ships `zero_to_fp32.py` as a utility to consolidate ZeRO-2 and ZeRO-3 sharded checkpoints into a single fp32 state dict file.

**Usage:**

```bash
python zero_to_fp32.py /path/to/ds_checkpoints/step_1000 /path/to/output_fp32.pt
```

The output file `/path/to/output_fp32.pt` contains a standard PyTorch `state_dict` with fp32 parameters, compatible with any non-DeepSpeed loader.

**Programmatic equivalent (ZeRO-3 gather):**

```python
import deepspeed

# Gather fp16 weights on rank 0 during training (for periodic export)
with deepspeed.zero.GatheredParameters(list(model.parameters()), modifier_rank=0):
    if dist.get_rank() == 0:
        state_dict = {k: v.data.float().clone() for k, v in model.named_parameters()}
        torch.save(state_dict, "/path/to/output_fp32.pt")
```

The `stage3_gather_16bit_weights_on_model_save` config flag can be set to `True` to have `engine.save_checkpoint()` automatically gather weights into a fp16 state dict alongside the sharded checkpoint, but this does not produce fp32 weights.

---

## 4. Cross-Strategy Conversion

### 4.1 ZeRO-3 → fp32 → Load into DDP/FSDP

```
Step 1: Run zero_to_fp32.py on ZeRO-3 checkpoint directory
        → produces output_fp32.pt (standard state dict, fp32)

Step 2: Load into unwrapped model on rank 0
        state = torch.load("output_fp32.pt", map_location="cpu")
        model.load_state_dict(state)

Step 3a (for DDP):
        model = model.to(device)
        model = DDP(model, device_ids=[local_rank])

Step 3b (for FSDP):
        model = FSDP(model, ..., sync_module_states=True, ...)
```

This is the canonical weights-only portability path for DeepSpeed. Optimizer states are NOT portable — the optimizer must be initialized fresh.

### 4.2 FSDP FULL_STATE_DICT → Load into DDP

```python
# Assume portable_weights.pt was saved via FSDP FULL_STATE_DICT
model = MyModel()
state_dict = torch.load("portable_weights.pt", map_location="cpu")
model.load_state_dict(state_dict)  # no wrapping needed for DDP yet

model = model.to(device)
model = DDP(model, device_ids=[local_rank])
```

DDP does not modify parameter names, so the FULL_STATE_DICT format loads directly. Optimizer must be initialized fresh.

### 4.3 FSDP FULL_STATE_DICT → Load into DeepSpeed ZeRO-3

```python
# Load portable weights into unwrapped model on all ranks (or rank 0 + sync)
model = MyModel()
if dist.get_rank() == 0:
    state_dict = torch.load("portable_weights.pt", map_location="cpu")
    model.load_state_dict(state_dict)
# Note: DS init will broadcast weights to all ranks if sync is needed;
# ensure broadcast happens before deepspeed.initialize() shards params

engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config=ds_config_dict,
)
```

**Caution:** DeepSpeed's `initialize()` with ZeRO-3 immediately shards parameters. If rank 0 has different weights than other ranks before `initialize()`, the sharding will be inconsistent. Use `dist.broadcast()` to synchronize weights before calling `deepspeed.initialize()`:

```python
for param in model.parameters():
    dist.broadcast(param.data, src=0)
```

---

## 5. Optimizer State Portability

### 5.1 Within-Family Resume

Optimizer state can be resumed when using the same strategy and the same world_size:

- **FSDP:** Use sharded checkpoint (DCP) — restores both model shards and optimizer state shards to the correct ranks.
- **DeepSpeed:** Use `engine.save_checkpoint()` / `engine.load_checkpoint()` — the ZeRO optimizer state files are per-rank and must be loaded with matching rank count.

### 5.2 Weights-Only Resume Across Families

When switching strategies (e.g., from FSDP to DeepSpeed), only weights are portable. Optimizer state must be initialized from scratch. This means:

1. Optimizer starts fresh (momentum = 0, second moment = 0 for Adam)
2. Learning rate schedule resumes from step 0 unless explicitly restored
3. First few steps may show a loss spike as the optimizer re-warms

**Mitigation:** Use a learning rate warmup after cross-strategy resume. Resume the step counter for scheduling purposes but do not restore optimizer state.

### 5.3 Changing world_size Within Family

- **FSDP (sharded DCP):** Use `torch.distributed.checkpoint.format_utils.dcp_to_torch_save()` to convert sharded checkpoint to full state dict, then reload into FSDP with new world_size using `sync_module_states=True`.
- **DeepSpeed:** Use `zero_to_fp32.py` to consolidate, then reload into DeepSpeed at the new world_size. Optimizer state is lost.

---

## 6. Compatibility Matrix

The table below indicates what is portable between training strategies. "Weights" means model parameters only. "Weights + Opt" means model parameters plus optimizer state (momentum, second moments, etc.).

| From \ To | DDP | FSDP | DeepSpeed ZeRO-2 | DeepSpeed ZeRO-3 |
|-----------|-----|------|-----------------|-----------------|
| **DDP** | Weights + Opt | Weights (rank-0 load + sync_module_states) | Weights (broadcast before init) | Weights (broadcast before init) |
| **FSDP FULL_STATE_DICT** | Weights | Weights + Opt (sharded DCP, same world_size) | Weights | Weights |
| **FSDP Sharded DCP** | Weights (convert first) | Weights + Opt (same world_size) | Weights (convert first) | Weights (convert first) |
| **DeepSpeed ZeRO-2** | Weights (zero_to_fp32) | Weights (zero_to_fp32 + sync_module_states) | Weights + Opt (same world_size) | Weights (zero_to_fp32) |
| **DeepSpeed ZeRO-3** | Weights (zero_to_fp32) | Weights (zero_to_fp32 + sync_module_states) | Weights (zero_to_fp32) | Weights + Opt (same world_size) |

**Legend:**
- `Weights` = model parameters portable, optimizer must restart
- `Weights + Opt` = full training state restorable (same world_size required)
- `convert first` = conversion tool (DCP format utils or zero_to_fp32) required before load
