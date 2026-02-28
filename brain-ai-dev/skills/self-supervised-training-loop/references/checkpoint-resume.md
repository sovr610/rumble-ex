# Checkpoint and Resume: Deep Reference

## Six-State Checkpoint Schema

A complete checkpoint for a self-supervised training loop requires exactly six states. Missing any one causes either incorrect resume behavior or silent bugs:

```python
checkpoint = {
    'model_state_dict':     ...,   # online encoder + predictor weights
    'target_state_dict':    ...,   # target encoder weights
    'optimizer_state_dict': ...,   # AdamW momentum buffers and step counts
    'scaler_state_dict':    ...,   # GradScaler scale factor and growth tracker
    'scheduler_state_dict': ...,   # cosine schedule current step and last_lr
    'step':                 ...,   # global training step (int)
}
```

### Why Each State is Required

**model_state_dict**: The online encoder weights. Without this, the model restarts from random initialization. Must be the unwrapped state (not DDP-wrapped).

**target_state_dict**: The target encoder weights. These diverge from online weights during training (by design). Without saving target state separately, resume would require re-computing EMA from step 0, which is impossible — you would need the full history of online weights.

**optimizer_state_dict**: Contains:
- `state`: per-parameter dict with `exp_avg` (first moment) and `exp_avg_sq` (second moment) for AdamW
- `param_groups`: learning rate, weight decay, betas for each parameter group

Without optimizer state, Adam's second moment estimates restart from zero. This causes extremely large effective learning rates for the first ~1000 steps after resume (denominator in Adam update is near zero), often causing loss spikes or divergence.

**scaler_state_dict**: Contains:
- `scale`: current scale factor (e.g., 131072.0 after successful training)
- `growth_factor`, `backoff_factor`, `growth_interval`
- `_growth_tracker`: steps since last scale change
- `_found_inf_per_device`: inf detection state

Without scaler state, the scale factor resets to 2^16=65536. If training was running at a higher scale (typical after long training), this causes unnecessary scale oscillation in the first few hundred steps.

**scheduler_state_dict**: Contains:
- `last_epoch`: step counter used by the scheduler
- `_last_lr`: most recent learning rate
- `base_lrs`: initial learning rates for each param group

Without scheduler state, the learning rate restarts from the initial value (or warmup start). This completely disrupts cosine annealing — the model receives too-high a learning rate if resumed mid-schedule.

**step**: The global step counter. Used to:
- Resume training from `step + 1`
- Correctly position the EMA tau scheduler
- Correctly position the cosine warmup scheduler
- Name the next checkpoint correctly

---

## DDP Module Unwrapping

The DDP wrapper adds a `.module` attribute that contains the original model. Always unwrap before saving:

```python
def save_model_state(model):
    # Handle both DDP-wrapped and unwrapped models
    if hasattr(model, 'module'):
        return model.module.state_dict()
    return model.state_dict()
```

### Why DDP State Dicts Require Care

When loading a checkpoint, you typically reconstruct the model BEFORE wrapping with DDP. If you saved the DDP-wrapped state dict and try to load it into an unwrapped model, the state dict may not match. Using `model.module.state_dict()` guarantees portability.

### Loading Order (Critical)

The correct sequence for loading a checkpoint:

```python
# 1. Create fresh model (on CPU initially to save GPU memory)
model = MyModel()

# 2. Load model weights (BEFORE DDP wrapping)
model.load_state_dict(checkpoint['model_state_dict'])

# 3. Move to GPU
model = model.cuda(rank)

# 4. SyncBatchNorm conversion (BEFORE DDP)
if sync_batchnorm:
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

# 5. Wrap with DDP
model = DDP(model, device_ids=[rank])

# 6. Create optimizer (AFTER DDP wrapping — optimizer params must reference DDP params)
optimizer = AdamW(model.parameters(), ...)

# 7. Load optimizer state
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# 8. Load remaining states
scaler.load_state_dict(checkpoint['scaler_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

# 9. Resume from next step
start_step = checkpoint['step'] + 1
```

Loading optimizer state BEFORE DDP wrapping causes a mismatch: the optimizer was created over DDP parameters, but the checkpoint was saved with non-DDP parameter references. This usually raises a silent bug where optimizer state is ignored.

---

## Auto-Resume Logic

On startup, scan the checkpoint directory for the latest checkpoint:

```python
import os
import re
import glob

def find_latest_checkpoint(checkpoint_dir):
    pattern = os.path.join(checkpoint_dir, 'checkpoint-*.pt')
    files = glob.glob(pattern)

    if not files:
        return None

    # Parse step numbers from filenames
    step_to_file = {}
    for filepath in files:
        basename = os.path.basename(filepath)
        match = re.match(r'checkpoint-(\d+)\.pt', basename)
        if match:
            step = int(match.group(1))
            step_to_file[step] = filepath

    if not step_to_file:
        return None

    # Return path with highest step number
    latest_step = max(step_to_file.keys())
    return step_to_file[latest_step], latest_step
```

The scan approach (instead of a latest-pointer file) is more crash-resilient: even if the pointer file was not updated (crash during write), the actual checkpoint files on disk determine what's available.

---

## Atomic Writes

Write to a temporary file first, then rename to prevent half-written checkpoints:

```python
def save_checkpoint_atomic(state_dict, filepath):
    # Write to temp file in the same directory (same filesystem)
    dirpath = os.path.dirname(filepath)
    tmpfile = os.path.join(dirpath, f'.tmp_{os.path.basename(filepath)}')

    try:
        torch.save(state_dict, tmpfile)
        os.rename(tmpfile, filepath)   # atomic on POSIX if same filesystem
    except Exception:
        # Clean up temp file on failure
        if os.path.exists(tmpfile):
            os.remove(tmpfile)
        raise
```

### Why os.rename is Atomic

On POSIX systems (Linux, macOS), `os.rename()` is implemented as the `rename(2)` syscall, which is guaranteed atomic by the filesystem. The destination file either fully exists or fully doesn't — there's no intermediate state.

Requirements for atomicity:
- Source and destination must be on the same filesystem
- The directory must be the same or on the same device

If saving to a network filesystem (NFS, NFS4), atomicity guarantees may not hold. In this case, use a write-barrier after rename: `os.fsync(os.open(filepath, os.O_RDONLY))`.

---

## Checkpoint Pruning

Keep only the last K checkpoints to avoid filling disk:

```python
def prune_checkpoints(checkpoint_dir, keep_last=3):
    pattern = os.path.join(checkpoint_dir, 'checkpoint-*.pt')
    files = glob.glob(pattern)

    # Parse and sort by step number
    step_to_file = {}
    for filepath in files:
        basename = os.path.basename(filepath)
        match = re.match(r'checkpoint-(\d+)\.pt', basename)
        if match:
            step = int(match.group(1))
            step_to_file[step] = filepath

    # Sort by step, keep last K
    sorted_steps = sorted(step_to_file.keys())
    steps_to_delete = sorted_steps[:-keep_last] if len(sorted_steps) > keep_last else []

    for step in steps_to_delete:
        os.remove(step_to_file[step])

    return len(steps_to_delete)  # number of files deleted
```

Default K=3 means:
- Always have at least 3 recovery points
- If a checkpoint is corrupt, fall back to the previous one
- If the last two are corrupt (very rare), fall back to the third

---

## Rank-0 Only Saving

In DDP training, only rank 0 should save checkpoints:

```python
def maybe_save_checkpoint(rank, manager, model, target, optimizer, scaler, scheduler, step):
    if rank == 0:
        manager.save(model, target, optimizer, scaler, scheduler, step)

    # Synchronize all ranks — rank 0 finishes writing before others proceed
    if dist.is_initialized():
        dist.barrier()
```

The `dist.barrier()` ensures that all ranks wait until rank 0 has finished writing. This prevents rank 1+ from loading a checkpoint that rank 0 hasn't finished writing yet (in multi-node scenarios where a second run might start while the first is still saving).

For loading: all ranks load from the same checkpoint file. The file path is deterministic (based on step number) and readable by all ranks from shared storage.

### sync_module_states Alternative

DDP supports broadcasting model state from rank 0 to all ranks on init:

```python
model = DDP(
    model,
    device_ids=[rank],
    sync_module_states=True   # broadcast rank-0 weights to all ranks
)
```

This is useful when only rank 0 loaded the checkpoint — DDP handles the broadcast automatically. However, this broadcasts the entire model state dict over the network, which is expensive for large models. Loading from shared storage on each rank is usually faster.

---

## torch.save Serialization Format

`torch.save` uses Python's serialization protocol internally (similar to Python's standard serialization). It can save:
- Complete Python objects (dicts, dataclasses, etc.)
- PyTorch tensors (stored efficiently as raw binary data)
- Optimizer state dicts, scheduler state dicts, GradScaler state dicts

For training checkpoints (full resume capability): use `torch.save`.

For model release/deployment (inference only): consider safetensors format, which only stores tensor data in a flat, safe binary format without arbitrary object execution.

---

## Checkpoint Size Estimation

A full checkpoint for a 100M parameter model (float32):

| Component            | Size Formula                    | Example (100M params) |
|----------------------|---------------------------------|-----------------------|
| model_state_dict     | N_params * 4 bytes              | 400 MB                |
| target_state_dict    | N_params * 4 bytes              | 400 MB                |
| optimizer_state_dict | N_params * 4 bytes * 2 (moments)| 800 MB                |
| scaler_state_dict    | ~1 KB                           | negligible            |
| scheduler_state_dict | ~1 KB                           | negligible            |
| step                 | 8 bytes                         | negligible            |
| **Total**            |                                 | **~1.6 GB**           |

With bfloat16 model weights (half precision):
- model + target: 200 MB each
- optimizer: still float32 (Adam moments are kept in fp32 for stability)
- Total: ~1.2 GB

The optimizer state is the largest component because AdamW stores two additional tensors per parameter (first moment `exp_avg` and second moment `exp_avg_sq`), each the same size as the parameter itself.
