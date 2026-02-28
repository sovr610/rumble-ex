# Checkpoint Format Reference

## Overview

V-JEPA 2 training checkpoints capture the complete state needed to resume training exactly from any epoch. The format is designed for robustness against distributed filesystem failures and flexibility for loading pretrained models with different architectures.

---

## Checkpoint Dictionary Structure

```python
checkpoint = {
    # Training epoch (0-indexed, save BEFORE incrementing)
    "epoch": int,

    # Model state dicts
    "encoder": encoder.state_dict(),           # Online encoder (ViT)
    "predictor": predictor.state_dict(),       # Predictor network
    "target_encoder": target_encoder.state_dict(),  # EMA target encoder

    # Optimizer state
    "opt": optimizer.state_dict(),             # AdamW optimizer state

    # Mixed precision scaler (None if not using AMP)
    "scaler": scaler.state_dict() if scaler is not None else None,
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `epoch` | `int` | Last completed epoch (resume from `epoch + 1`) |
| `encoder` | `OrderedDict` | Online ViT encoder weights and biases |
| `predictor` | `OrderedDict` | Predictor MLP/transformer weights |
| `target_encoder` | `OrderedDict` | EMA (momentum-updated) target encoder |
| `opt` | `dict` | AdamW state: `state`, `param_groups` |
| `scaler` | `dict` or `None` | GradScaler state for AMP; `None` if not used |

---

## Saving Checkpoints

### Basic Save

```python
import torch
import os

def save_checkpoint(state: dict, save_dir: str, epoch: int, tag: str = "") -> str:
    os.makedirs(save_dir, exist_ok=True)
    filename = f"checkpoint_{epoch:04d}{f'_{tag}' if tag else ''}.pth"
    path = os.path.join(save_dir, filename)
    # Write to a tmp path then rename for atomicity
    tmp_path = path + ".tmp"
    torch.save(state, tmp_path)
    os.rename(tmp_path, path)
    return path
```

### DDP Save Pattern

When using DistributedDataParallel, only rank 0 should save to avoid filesystem contention:

```python
if rank == 0:
    checkpoint = {
        "epoch": epoch,
        # Unwrap DDP: use .module to access the underlying model
        "encoder": encoder.module.state_dict() if hasattr(encoder, "module") else encoder.state_dict(),
        "predictor": predictor.module.state_dict() if hasattr(predictor, "module") else predictor.state_dict(),
        "target_encoder": target_encoder.module.state_dict() if hasattr(target_encoder, "module") else target_encoder.state_dict(),
        "opt": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    save_checkpoint(checkpoint, save_dir, epoch)
```

---

## Loading Checkpoints

### Standard Load with Retry

Distributed filesystems (NFS, Lustre, GPFS) occasionally have transient failures — a file that exists may briefly appear empty or raise an I/O error. The retry pattern handles this:

```python
import time
import random
import torch

def load_checkpoint(path: str, max_retries: int = 5) -> dict:
    """Load checkpoint with exponential backoff retry for transient FS failures."""
    for attempt in range(max_retries):
        try:
            # map_location="cpu" avoids GPU memory issues during load
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            return checkpoint
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"Failed to load checkpoint after {max_retries} attempts: {e}"
                ) from e

            # Exponential backoff: 2^attempt + random jitter (0-1 seconds)
            wait_seconds = (2 ** attempt) + random.random()
            print(f"Checkpoint load attempt {attempt + 1} failed: {e}. "
                  f"Retrying in {wait_seconds:.1f}s...")
            time.sleep(wait_seconds)
```

### Backoff Schedule

| Attempt | Base Wait | Max Wait (with jitter) |
|---------|-----------|----------------------|
| 0 | 1s | 2s |
| 1 | 2s | 3s |
| 2 | 4s | 5s |
| 3 | 8s | 9s |
| 4 | 16s | 17s |

Total maximum wait before failure: approximately 32 seconds.

---

## Key Prefix Stripping

Checkpoints saved from DDP-wrapped models have keys prefixed with `"module."` (from `nn.DataParallel` or `DistributedDataParallel`). Some pretrained checkpoints additionally have a `"backbone."` prefix. These must be stripped before loading into a non-DDP model.

```python
def strip_prefix(state_dict: dict, prefix: str = "module.") -> dict:
    """Remove a prefix from all keys in a state_dict."""
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict
```

### Usage

```python
# Loading a checkpoint saved from DDP into a non-DDP model
raw_state = torch.load("checkpoint.pth", map_location="cpu")["encoder"]
cleaned_state = strip_prefix(raw_state, "module.")
model.load_state_dict(cleaned_state)

# Double-stripping for backbone.module. prefix
cleaned_state = strip_prefix(strip_prefix(raw_state, "backbone."), "module.")
```

### Key Stripping Order

When a checkpoint has `"backbone.module."` prefix:
1. Strip `"module."` first -> `"backbone.layer.weight"`
2. Strip `"backbone."` second -> `"layer.weight"`

Or strip `"backbone.module."` as a single prefix:
```python
cleaned = strip_prefix(raw_state, "backbone.module.")
```

---

## Pretrained Loading (strict=False)

### When to Use strict=False

`strict=False` allows loading a checkpoint that is missing some keys or has extra keys. This is required for:

1. **RoPE (Rotary Position Embedding) models**: RoPE does not store position embeddings as parameters (they are computed on-the-fly). A standard ViT checkpoint with `pos_embed` cannot be loaded strictly into a RoPE model because `pos_embed` is absent.

2. **Fine-tuning**: Loading a pretrained encoder into a model with an additional classification head.

3. **Architecture variations**: Minor architecture differences between checkpoint and current model.

```python
def load_pretrained(
    model: torch.nn.Module,
    path: str,
    key: str = "encoder",
    strict: bool = False,
    max_retries: int = 5,
) -> tuple:
    """
    Load pretrained weights with optional key stripping.

    Returns:
        (missing_keys, unexpected_keys) from model.load_state_dict
    """
    checkpoint = load_checkpoint(path, max_retries=max_retries)

    # Extract the relevant sub-dict
    if key in checkpoint:
        state_dict = checkpoint[key]
    else:
        state_dict = checkpoint  # Assume the checkpoint IS the state_dict

    # Strip common prefixes
    for prefix in ("module.", "backbone.", "encoder."):
        # Only strip if the majority of keys have this prefix
        n_with_prefix = sum(1 for k in state_dict if k.startswith(prefix))
        if n_with_prefix > len(state_dict) // 2:
            state_dict = strip_prefix(state_dict, prefix)

    result = model.load_state_dict(state_dict, strict=strict)

    if result.missing_keys:
        print(f"Missing keys ({len(result.missing_keys)}): {result.missing_keys[:5]}"
              f"{'...' if len(result.missing_keys) > 5 else ''}")
    if result.unexpected_keys:
        print(f"Unexpected keys ({len(result.unexpected_keys)}): {result.unexpected_keys[:5]}"
              f"{'...' if len(result.unexpected_keys) > 5 else ''}")

    return result.missing_keys, result.unexpected_keys
```

---

## Resume Logic

```python
def resume_or_start(
    checkpoint_dir: str,
    encoder,
    predictor,
    target_encoder,
    optimizer,
    scaler,
) -> int:
    """Returns the epoch to start training from."""

    # Find latest checkpoint
    ckpt_files = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pth")))
    if not ckpt_files:
        print("No checkpoint found, starting from scratch")
        return 0

    latest = ckpt_files[-1]
    print(f"Resuming from: {latest}")

    ckpt = load_checkpoint(latest)

    encoder.load_state_dict(strip_prefix(ckpt["encoder"], "module."))
    predictor.load_state_dict(strip_prefix(ckpt["predictor"], "module."))
    target_encoder.load_state_dict(strip_prefix(ckpt["target_encoder"], "module."))
    optimizer.load_state_dict(ckpt["opt"])

    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    start_epoch = ckpt["epoch"] + 1
    print(f"Resumed at epoch {start_epoch}")
    return start_epoch
```

---

## Checkpoint Rotation

To avoid filling disk, keep only the last N checkpoints:

```python
def rotate_checkpoints(save_dir: str, keep_last: int = 5) -> None:
    """Keep only the most recent `keep_last` checkpoints."""
    ckpt_files = sorted(glob.glob(os.path.join(save_dir, "checkpoint_*.pth")))
    # Always keep files tagged with "best" or "final"
    regular = [f for f in ckpt_files
               if "best" not in os.path.basename(f)
               and "final" not in os.path.basename(f)]
    for old_ckpt in regular[:-keep_last]:
        os.remove(old_ckpt)
        print(f"Removed old checkpoint: {old_ckpt}")
```

---

## File Format Notes

- **Format**: PyTorch serialization format via `torch.save` / `torch.load`
- **Serialization**: Uses Python's pickle protocol internally (PyTorch tensors and state dicts)
- **Size**: Approximately 2-8 GB per checkpoint for a 7B parameter model
- **Atomic writes**: Write to a `.tmp` file then `os.rename` for atomicity on POSIX filesystems
- **Security note**: Only load checkpoint files from trusted sources, as the PyTorch format uses Python serialization internally

---

## Common Checkpoint Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `KeyError: 'encoder'` | Different checkpoint structure | Inspect keys with `list(ckpt.keys())` before loading |
| `RuntimeError: size mismatch` | Architecture mismatch | Use `strict=False` and check missing/unexpected keys |
| `FileNotFoundError` intermittently | NFS transient failure | Use retry logic with exponential backoff |
| Missing `pos_embed` on RoPE model | RoPE has no learned pos_embed | Use `strict=False` |
| `module.` prefix mismatch | DDP vs non-DDP checkpoint | Use `strip_prefix` before loading |
