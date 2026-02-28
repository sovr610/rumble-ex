---
name: Self-Supervised Training Loop (AMP + EMA + DDP)
description: >
  This skill should be used when the user asks to "build a self-supervised training loop",
  "implement AMP mixed precision training", "add GradScaler with gradient clipping",
  "torch.amp.autocast bfloat16 training", "implement EMA target network",
  "cosine annealed EMA tau schedule", "BYOL-style momentum update",
  "DDP setup for self-supervised learning", "wrap online encoder with DDP",
  "SyncBatchNorm conversion", "checkpoint save and resume training",
  "automatic checkpoint resume", "W&B logging for training loop",
  "log gradient norm to wandb", "cosine warmup learning rate schedule",
  "AdamW with weight decay 0.04", "betas 0.9 0.95 optimizer",
  "scaler.unscale_ then clip_grad_norm_ ordering",
  "self-supervised training with EMA teacher", "DINO training loop",
  "production training loop with AMP EMA DDP checkpointing",
  or needs guidance on building a production-hardened self-supervised
  training loop with all the components wired in the correct order.
version: 0.1.0
---

# Self-Supervised Training Loop (AMP + EMA + DDP)

## Overview

Generate a complete, production-hardened self-supervised training loop combining PyTorch AMP (bfloat16 autocast + GradScaler), EMA target network with cosine-annealed tau, DDP with SyncBatchNorm, cosine warmup scheduling, structured checkpointing with auto-resume, and W&B metric logging. The loop enforces correct operation ordering to prevent the silent failures that plague self-supervised training: gradient clipping on still-scaled gradients, EMA updates before optimizer steps, DDP wrapping of the target encoder, and checkpoint state omissions.

Design principle: **correct ordering, every step logged, every state saved, resume from any crash.**

## Critical Operation Ordering

The per-step sequence is non-negotiable. Violating it causes silent training degradation:

```
1. optimizer.zero_grad(set_to_none=True)
2. with torch.amp.autocast('cuda', dtype=torch.bfloat16):
       loss = forward(online_encoder, predictor, target_encoder, views)
3. scaler.scale(loss).backward()
4. scaler.unscale_(optimizer)                          # MUST precede clip
5. grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
6. scaler.step(optimizer)                              # skips if infs found
7. scaler.update()
8. scheduler.step()
9. ema_update(online_encoder, target_encoder, tau)     # AFTER optimizer step
10. log metrics (loss, grad_norm, tau, lr, memory)
```

**Why this order matters:**
- Step 4 before 5: clipping scaled gradients uses wrong threshold
- Step 6 after 4: `step()` checks for inf/nan from `unscale_()` and skips if found
- Step 9 after 6: EMA must see the updated online weights, not pre-step weights

## Public Contract

### SelfSupervisedTrainer

```python
class SelfSupervisedTrainer:
    def __init__(self, cfg: TrainingConfig, rank: int, world_size: int): ...
    def setup(self) -> None: ...          # DDP init, model wrap, optimizer, scheduler
    def train(self) -> None: ...          # Main loop with AMP, EMA, logging
    def save_checkpoint(self, step: int) -> None: ...
    def load_checkpoint(self) -> Optional[int]: ...
    def cleanup(self) -> None: ...        # DDP destroy
```

### EMAUpdater

```python
class EMAUpdater:
    def __init__(self, tau_base: float = 0.996, tau_final: float = 0.9999,
                 total_steps: int = 100_000): ...
    def get_tau(self, step: int) -> float: ...       # cosine annealed
    def update(self, online: nn.Module, target: nn.Module, step: int) -> float: ...
```

### AMPContext

```python
class AMPContext:
    def __init__(self, dtype: torch.dtype = torch.bfloat16, enabled: bool = True): ...
    def autocast(self) -> ContextManager: ...
    def backward(self, loss: Tensor) -> None: ...
    def unscale_and_clip(self, optimizer: Optimizer, params, max_norm: float) -> float: ...
    def step_and_update(self, optimizer: Optimizer) -> None: ...
```

## Key Concepts

### AMP with bfloat16

Use `torch.amp.autocast('cuda', dtype=torch.bfloat16)` (not the deprecated `torch.cuda.amp.autocast`). bfloat16 is preferred over float16 for training stability — its 8-bit exponent matches float32's range, eliminating most underflow issues.

**GradScaler with bfloat16**: Technically optional since bfloat16 rarely underflows, but include it for two reasons: (1) graceful fallback to float16 hardware, (2) inf/nan detection via `scaler.step()` skip logic. Set `enabled=True` by default; disable via config for pure bfloat16 paths.

### Gradient Clipping Flow

```
scaler.scale(loss).backward()        # gradients are scaled
scaler.unscale_(optimizer)           # restore true magnitudes
grad_norm = clip_grad_norm_(params, max_norm=1.0)   # clip at true scale
# Log grad_norm to W&B HERE — this is the pre-clip norm at true scale
scaler.step(optimizer)               # checks for inf, skips if found
scaler.update()                      # adjust scale factor
```

`clip_grad_norm_` returns the total norm **before** clipping — log this value to detect gradient spikes early.

### EMA Target Network

The target encoder is **never trained by gradient descent**. It tracks the online encoder via exponential moving average with cosine-annealed tau:

```
tau(step) = 1 - (1 - tau_base) * (cos(pi * step / total_steps) + 1) / 2
target_param = tau * target_param + (1 - tau) * online_param
```

This ramps tau from `tau_base=0.996` (fast tracking early) to `tau_final≈0.9999` (slow, stable late). The `@torch.no_grad()` decorator is mandatory — EMA must not create a computation graph.

### DDP Setup for Self-Supervised

**Wrap only the online encoder and predictor** — never the target encoder. The target has no gradients and wrapping it wastes communication bandwidth and triggers `find_unused_parameters` errors.

Setup sequence:
1. `init_process_group(backend='nccl')`
2. Create model on `cuda:{rank}`
3. `nn.SyncBatchNorm.convert_sync_batchnorm(online_model)` — **before** DDP wrap
4. `DDP(online_model, device_ids=[rank], find_unused_parameters=False)`
5. Copy online encoder weights to target encoder (initial sync)

**W&B in DDP**: Initialize `wandb.init()` only on rank 0. Gate all logging behind `if rank == 0`.

### Optimizer and Schedule

AdamW with ViT-standard hyperparameters:
- `lr=1e-4`, `weight_decay=0.04`, `betas=(0.9, 0.95)`
- The 0.95 beta2 (vs default 0.999) reduces second-moment lag for vision transformers

Cosine warmup schedule: linear warmup over 10,000 steps, then cosine decay to `lr_min=1e-6`.

### Checkpointing

Save every 5,000 steps with **six state dicts**:
1. `model_state_dict` — online encoder + predictor (unwrap DDP first: `model.module.state_dict()`)
2. `target_state_dict` — target encoder
3. `optimizer_state_dict`
4. `scaler_state_dict`
5. `scheduler_state_dict`
6. `step` — global step counter

**Auto-resume**: On startup, scan checkpoint directory for latest `checkpoint-{step}.pt`. If found, load all six components and continue from `step + 1`. Log resumed step to W&B.

### W&B Logging

Log every step (rank 0 only):

| Metric | Key | Frequency |
|--------|-----|-----------|
| Training loss | `train/loss` | Every step |
| Gradient norm (pre-clip) | `train/grad_norm` | Every step |
| EMA tau | `train/ema_tau` | Every step |
| Learning rate | `train/lr` | Every step |
| GPU memory allocated | `train/gpu_mem_gb` | Every step |
| Sample prediction grid | `train/predictions` | Every 1,000 steps |

The prediction grid uses `torchvision.utils.make_grid()` on a fixed validation batch, logged as `wandb.Image()`.

## Configuration Surface

```python
@dataclass
class TrainingConfig:
    # AMP
    amp_enabled: bool = True
    amp_dtype: str = "bfloat16"            # bfloat16 | float16
    grad_scaler_enabled: bool = True
    max_grad_norm: float = 1.0

    # Optimizer
    lr: float = 1e-4
    weight_decay: float = 0.04
    betas: Tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 10_000
    lr_min: float = 1e-6
    total_steps: int = 100_000

    # EMA
    ema_tau_base: float = 0.996
    ema_tau_final: float = 0.9999

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 5_000
    auto_resume: bool = True

    # Logging
    wandb_project: str = "ssl-training"
    wandb_entity: Optional[str] = None
    log_every: int = 1
    sample_grid_every: int = 1_000
    grid_nrow: int = 8

    # DDP
    backend: str = "nccl"
    sync_batchnorm: bool = True
```

## Done-When Gates

1. **AMP Ordering Correct** — `unscale_()` is called before `clip_grad_norm_()`. `step()` is called after `unscale_()`. `update()` is called after `step()`. Gradient norm returned by clip is logged before clipping applies. Training produces finite loss for 100 steps.
2. **EMA Schedule Tracks** — At step 0, tau equals `tau_base`. At `total_steps`, tau approaches `tau_final`. Target parameters differ from online parameters after EMA updates. `torch.no_grad()` wraps all EMA operations.
3. **DDP Wraps Correctly** — Online encoder is wrapped with DDP. Target encoder is **not** wrapped. `find_unused_parameters=False`. SyncBatchNorm conversion happens before DDP wrap. Training runs on 2+ processes without hanging.
4. **Checkpoint Round-Trip** — Save at step N, kill process, restart — training resumes from step N+1 with identical optimizer/scaler/scheduler state. All six state dicts are present in checkpoint file.

## Resources

### Reference Files
- **`references/amp-gradient-flow.md`** — AMP autocast internals, GradScaler mechanics, bfloat16 vs float16 trade-offs, unscale/clip/step ordering proof, inf/nan skip behavior, disabling scaler for pure bfloat16
- **`references/ema-target-network.md`** — EMA update derivation, cosine tau annealing formula, BYOL/DINO tau schedules, no_grad requirement, initial weight sync, multi-param-group EMA
- **`references/ddp-self-supervised.md`** — DDP wrap strategy for online/target split, SyncBatchNorm placement, find_unused_parameters rationale, gradient_as_bucket_view, multi-node setup, process group lifecycle
- **`references/checkpoint-resume.md`** — Six-state checkpoint schema, DDP module unwrapping, auto-resume scan logic, atomic writes, checkpoint pruning, cross-strategy portability
- **`references/wandb-integration.md`** — Rank-0-only init, metric key conventions, gradient norm logging, GPU memory queries, prediction grid generation, W&B groups for DDP, run resume on crash
- **`references/testing-matrix.md`** — Test scenarios for all components

### Asset Files
- **`assets/training_loop_template.py`** — SelfSupervisedTrainer with full loop, correct ordering, self-tests
- **`assets/amp_gradient_template.py`** — AMPContext with autocast, backward, unscale_and_clip, step_and_update, self-tests
- **`assets/ema_template.py`** — EMAUpdater with cosine tau, no_grad update, initial sync, self-tests
- **`assets/ddp_setup_template.py`** — DDP initialization, SyncBatchNorm, online-only wrapping, cleanup, self-tests
- **`assets/checkpoint_template.py`** — CheckpointManager with save/load/auto-resume/prune, self-tests
- **`assets/wandb_logger_template.py`** — WandbLogger with rank-0 gating, metric logging, prediction grid, self-tests
- **`assets/training_config_template.py`** — All config dataclasses, validation, serialization

### Scripts
- **`scripts/validate_training_loop.py`** — Validates done-when gates
- **`scripts/gen_training_tests.py`** — Generates pytest test cases
- **`scripts/training_diagnostic.py`** — Diagnostic tool: runs N steps, prints ordering trace, EMA tau curve, checkpoint sizes
