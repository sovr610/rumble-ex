# Failure Snapshot Reference

## Overview

The failure snapshot system captures the exact training state at the moment of detected failure, providing everything needed to reproduce the issue in a controlled environment. The snapshot is designed to be **complete** (all state needed for repro), **atomic** (no half-written snapshot directories), and **concise** (failure summary printed before exit).

---

## 7-File Snapshot Schema

Each snapshot is written to a directory named `snapshot_<step>/` inside the configured `snapshot_dir`. All 7 files are required for a complete snapshot.

### File 1: `config.json`

Full resolved PrecisionConfig, SentinelConfig, and FailureConfig as a JSON object.

```json
{
  "precision_config": {
    "mode": "fp16",
    "autocast_dtype": "float16",
    "grad_scaler_enabled": true,
    "grad_scaler_init_scale": 65536.0,
    "grad_scaler_growth_factor": 2.0,
    "grad_scaler_backoff_factor": 0.5,
    "grad_scaler_growth_interval": 2000,
    "max_grad_norm": 1.0
  },
  "sentinel_config": { ... },
  "failure_config": { ... }
}
```

**Purpose**: Reproducer knows exactly what precision mode and thresholds were active.

### File 2: `env.json`

Hardware and software environment at time of failure.

```json
{
  "torch_version": "2.3.0",
  "cuda_version": "12.1",
  "gpu_name": "NVIDIA A100-SXM4-80GB",
  "gpu_count": 8,
  "gpu_memory_gb": 79.1,
  "git_sha": "a1b2c3d4e5f6",
  "python_version": "3.11.4",
  "platform": "Linux-5.15.0-x86_64",
  "hostname": "node-07",
  "timestamp_utc": "2024-01-15T14:23:11Z"
}
```

Collect git sha via subprocess (best effort, omit if not in a git repo):
```python
import subprocess
try:
    git_sha = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'],
        stderr=subprocess.DEVNULL
    ).decode().strip()
except Exception:
    git_sha = "unknown"
```

GPU info via `torch.cuda.get_device_name(0)` and `torch.cuda.get_device_properties(0).total_memory`.

### File 3: `rng_state.pt`

All RNG states needed for deterministic repro.

```python
import random
import torch

rng_states = {
    'torch_cpu': torch.random.get_rng_state(),
    'torch_cuda': torch.cuda.get_rng_state_all(),   # list, one per GPU
    'python': random.getstate(),
}

# numpy RNG (best effort — numpy may not be available)
try:
    import numpy as np
    rng_states['numpy'] = np.random.get_state()
except ImportError:
    pass

torch.save(rng_states, 'rng_state.pt')
```

**Purpose**: Allows exact reproduction of the failed step by restoring all RNG states before running the repro script.

### File 4: `batch.pt`

The actual batch that caused the failure.

```python
# batch is a dict from the dataloader
batch_to_save = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
torch.save(batch_to_save, 'batch.pt')
```

For dataset-aware training, also save indices if available:
```python
if hasattr(batch, 'indices') or 'dataset_indices' in batch:
    batch_to_save['_dataset_indices'] = batch.get('dataset_indices', None)
```

**Move tensors to CPU** before saving — CUDA tensors can only be loaded on machines with CUDA, whereas CPU tensors load anywhere.

### File 5: `model_state.pt`

Full model state dict for the failed checkpoint.

```python
# Unwrap DDP if model is wrapped
actual_model = model.module if hasattr(model, 'module') else model
state_dict = actual_model.state_dict()
torch.save(state_dict, 'model_state.pt')
```

**Size guard**: If the state dict would exceed 2 GB, save only the first non-finite module's parameters plus a pointer to the last good checkpoint:

```python
import sys
rough_size = sum(v.numel() * v.element_size() for v in state_dict.values())
if rough_size > 2 * 1024**3:
    # Save only the offending module
    offending_module = report.weight_report.first_nonfinite_name if report.weight_report else None
    partial_state = {
        k: v for k, v in state_dict.items()
        if offending_module and k.startswith(offending_module)
    }
    torch.save({
        'partial_state': partial_state,
        'offending_module': offending_module,
        'full_checkpoint_ptr': cfg.last_checkpoint_path  # path to last good ckpt
    }, 'model_state.pt')
```

### File 6: `optimizer_state.pt`

Optimizer state dict (best effort).

```python
try:
    opt_state = optimizer.state_dict()
    torch.save(opt_state, 'optimizer_state.pt')
except Exception as e:
    torch.save({'error': str(e), 'saved': False}, 'optimizer_state.pt')
```

Optimizer state may be large (Adam keeps 2 additional tensors per parameter). Save best-effort — if it fails, record the error in the file rather than crashing the snapshot writer.

### File 7: `numerics.json`

Latest NumericsReport serialized to JSON, plus scaler state and skip counters.

```json
{
  "step": 1050,
  "timestamp": 1705328591.234,
  "first_nonfinite_tensor": "blocks.8.attn.out_proj.weight",
  "nan_streak": 2,
  "scaler_state": {
    "scale": 4096.0,
    "growth_tracker": 0,
    "was_scale": 65536.0,
    "scale_trend": "declining"
  },
  "skip_counters": {
    "num_steps_total": 1050,
    "num_steps_skipped": 87,
    "skip_rate": 0.0829,
    "effective_update_rate": 0.9171
  },
  "grad_norm_report": { ... },
  "activation_report": { ... },
  "logit_report": { ... },
  "weight_report": { ... }
}
```

---

## Atomic Write Pattern

Snapshot writes use a temporary directory to prevent partial snapshots from appearing in the final location:

```python
import os
import tempfile
import shutil

def capture_atomic(snapshot_dir: str, write_fn: Callable[[str], None]) -> str:
    # Create temp dir in same filesystem as final location
    parent = os.path.dirname(snapshot_dir)
    os.makedirs(parent, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=parent, prefix=".tmp_snap_") as tmpdir:
        # Write all 7 files to tmpdir
        write_fn(tmpdir)

        # Atomic rename to final location
        # (tempfile context manager will clean up tmpdir if rename fails)
        if os.path.exists(snapshot_dir):
            shutil.rmtree(snapshot_dir)
        os.rename(tmpdir, snapshot_dir)

    return snapshot_dir
```

**Why os.rename() is atomic on Linux**: `rename()` system call is atomic when source and destination are on the same filesystem. The directory either appears fully formed at the destination or not at all. This is the standard technique for atomic file/directory writes.

**Note**: On Windows, `os.rename()` fails if the destination exists. Use `shutil.move()` or handle the Windows case explicitly. On Linux/macOS, this is not an issue.

---

## NaN Streak Tracking

The `nan_steps_in_a_row` counter tracks how many consecutive steps have had non-finite loss or parameters:

```python
def update_nan_streak(self, loss: float, weight_report: Optional[WeightReport]) -> int:
    """Update and return current NaN streak count."""
    is_nan = (
        not math.isfinite(loss) or
        (weight_report is not None and not weight_report.all_finite)
    )
    if is_nan:
        self.nan_steps_in_a_row += 1
    else:
        self.nan_steps_in_a_row = 0
    return self.nan_steps_in_a_row
```

**Reset on clean step**: The streak resets to 0 on any step with finite loss and finite weights. A single clean step is sufficient to reset — this prevents false aborts on training runs with occasional spiky batches.

**Persist threshold**: When `nan_steps_in_a_row >= nan_persist_steps` (default 3), take abort action.

---

## Abort Behavior

### Failure Summary Format

Before aborting, print a concise summary to stderr:

```
[FATAL] Persistent NaN detected for 3 consecutive steps. Aborting.

Failure Summary:
  First detected: step 1048
  Offending tensor: blocks.8.attn.out_proj.weight
  Scaler scale trend: 65536 → 8192 → 4096 → 2048 (declining)
  Recent grad norms (last 5): [0.34, 0.89, 2.41, nan, nan]
  Logit max_abs (last 5): [23.4, 45.1, 78.9, 88.2, nan]
  Skip rate (last 100 steps): 12.3% (12/100 steps skipped)
  Snapshot: runs/run_001/numerics/snapshot_1050/

Next steps:
  1. Load snapshot: python repro.py --snapshot runs/run_001/numerics/snapshot_1050/
  2. Check logit drift starting at step ~1040 (max_abs exceeded threshold)
  3. Consider: reduce LR, add logit clamping, or switch to bf16
```

### `on_error: "abort"` (default)

```python
import sys
print(failure_summary, file=sys.stderr)
sys.exit(1)
```

### `on_error: "raise"`

```python
raise RuntimeError(
    f"Persistent NaN for {nan_streak} steps. "
    f"Snapshot: {snapshot_path}. "
    f"First nonfinite: {first_nonfinite_name}"
)
```

Use `raise` when the training loop is called from a larger framework that needs to catch the error (e.g., hyperparameter search).

---

## Repro Script Generation

Optionally generate a `repro.py` that loads the snapshot and reproduces the failure:

```python
REPRO_TEMPLATE = '''
"""
Auto-generated repro script from snapshot at: {snapshot_path}
Failed at step: {step}
"""
import torch
import random
import sys
sys.path.insert(0, "{project_root}")

# Load snapshot
snapshot_dir = "{snapshot_path}"

# Restore RNG states for determinism
rng = torch.load(f"{snapshot_dir}/rng_state.pt")
torch.random.set_rng_state(rng['torch_cpu'])
if torch.cuda.is_available():
    torch.cuda.set_rng_state_all(rng['torch_cuda'])
random.setstate(rng['python'])

# Load batch
batch = torch.load(f"{snapshot_dir}/batch.pt")

# Load model (user must instantiate model architecture first)
# model = YourModel(...)
# model.load_state_dict(torch.load(f"{snapshot_dir}/model_state.pt"))

print("Snapshot loaded. Restore RNG + batch to reproduce failure.")
print("Implement forward pass and check for NaN at this step.")
'''
```

---

## Snapshot Validation

After writing, validate the snapshot is complete and loadable:

```python
def validate_snapshot(snapshot_dir: str) -> bool:
    required_files = [
        'config.json', 'env.json', 'rng_state.pt',
        'batch.pt', 'model_state.pt', 'optimizer_state.pt', 'numerics.json'
    ]
    for fname in required_files:
        fpath = os.path.join(snapshot_dir, fname)
        if not os.path.exists(fpath):
            return False
        if os.path.getsize(fpath) == 0:
            return False
    # Try loading torch artifacts
    try:
        torch.load(os.path.join(snapshot_dir, 'rng_state.pt'), map_location='cpu')
        torch.load(os.path.join(snapshot_dir, 'batch.pt'), map_location='cpu')
    except Exception:
        return False
    return True
```
