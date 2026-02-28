# W&B Integration for Self-Supervised Training: Deep Reference

## Rank-0-Only Initialization

In DDP training, only rank 0 should initialize W&B. All other ranks skip the initialization entirely:

```python
from dataclasses import asdict
import wandb

def setup_wandb(rank, cfg):
    if rank == 0:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            config=asdict(cfg),
            group="DDP",
            job_type="train"
        )
```

Using `asdict(cfg)` from dataclasses serializes the entire TrainingConfig to a flat dict that W&B stores as the run config. This enables hyperparameter tracking and comparison across runs.

The `group="DDP"` argument groups all ranks conceptually under one run in the W&B UI. Since only rank 0 logs, there's literally only one run — but marking it as DDP is good practice.

---

## Metric Key Conventions

Use consistent prefixes for W&B metric organization:

| Metric                        | Key                       | Type        |
|-------------------------------|---------------------------|-------------|
| Training loss                 | `train/loss`              | float       |
| Pre-clip gradient norm        | `train/grad_norm`         | float       |
| EMA tau value                 | `train/ema_tau`           | float       |
| Current learning rate         | `train/lr`                | float       |
| GPU memory allocated (GB)     | `train/gpu_mem_gb`        | float       |
| GPU memory reserved (GB)      | `train/gpu_mem_reserved_gb` | float     |
| Global training step          | `train/step`              | int         |
| Sample prediction grid        | `train/predictions`       | wandb.Image |

The "train/" prefix creates a section in the W&B dashboard that can be expanded/collapsed. Use "val/" for validation metrics (if any).

---

## Per-Step Logging

Log every training step (frequency controlled by `log_every` config):

```python
def log_step(rank, step, loss, grad_norm, tau, lr, gpu_mem_gb):
    if rank != 0:
        return

    metrics = {
        'train/loss':       float(loss),
        'train/grad_norm':  float(grad_norm),
        'train/ema_tau':    float(tau),
        'train/lr':         float(lr),
        'train/gpu_mem_gb': float(gpu_mem_gb),
        'train/step':       step,
    }
    wandb.log(metrics, step=step)
```

Using `step=step` ensures the W&B x-axis uses training step rather than W&B's internal step counter. This is critical when resuming runs — W&B's internal counter would start from 0 on resume, but passing `step=step` (which continues from where training left off) keeps the x-axis continuous.

---

## GPU Memory Queries

```python
import torch

def get_gpu_memory_gb():
    # Allocated: memory actually used by tensors
    allocated = torch.cuda.memory_allocated() / 1e9

    # Reserved: memory cached by PyTorch allocator (allocated + fragmentation overhead)
    reserved = torch.cuda.memory_reserved() / 1e9

    return allocated, reserved
```

`memory_allocated()` is the more useful metric for tracking per-step memory growth. `memory_reserved()` is useful for understanding how much memory the PyTorch allocator is holding for future use (always >= allocated).

For a training step with a 100M parameter model at bfloat16:
- Allocated during forward: ~2-3x model size (model weights + activations + gradients)
- Reserved: typically 10-20% higher than allocated (fragmentation)

---

## Prediction Grid Generation

Keep a fixed "visualization batch" throughout training to see consistent representations evolve:

```python
import torchvision.utils as vutils
import wandb

class PredictionGridLogger:
    def __init__(self, fixed_batch, nrow=8):
        # fixed_batch: a batch of images saved at startup, never changed
        self.fixed_batch = fixed_batch
        self.nrow = nrow

    def log_grid(self, rank, model, step):
        if rank != 0:
            return

        model.train(False)  # inference mode: module.train(False)
        with torch.no_grad():
            predictions = model(self.fixed_batch)

        model.train(True)   # restore training mode

        # Make grid: expects (N, C, H, W) tensor
        grid = vutils.make_grid(
            predictions,
            nrow=self.nrow,
            normalize=True,     # rescale to [0, 1]
            scale_each=True     # normalize each image independently
        )

        # Log as W&B Image
        wandb.log({
            'train/predictions': wandb.Image(grid),
            'train/step': step
        }, step=step)
```

Note: Always use `module.train(False)` instead of the blocked method. After inference, restore with `module.train(True)` or `module.train()`.

---

## W&B Run Resume on Crash

To resume a W&B run after a crash, use a deterministic run ID:

```python
import hashlib
import json

def get_run_id(cfg):
    # Create a deterministic ID based on key config params
    key_config = {
        'project': cfg.wandb_project,
        'lr': cfg.lr,
        'total_steps': cfg.total_steps,
        'checkpoint_dir': cfg.checkpoint_dir,
    }
    config_str = json.dumps(key_config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]

def setup_wandb_with_resume(rank, cfg):
    if rank == 0:
        run_id = get_run_id(cfg)
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            config=vars(cfg),
            id=run_id,
            resume='allow'  # resumes if run_id exists, creates new if not
        )
```

With `resume='allow'`:
- If a W&B run with this ID exists: appends new data to the existing run
- If no run exists: creates a new run with this ID
- The x-axis in W&B will be continuous across the crash/resume boundary

---

## Custom X-Axis Definition

Define a custom x-axis metric before any logging to ensure step-based x-axis:

```python
def define_metrics(rank):
    if rank != 0:
        return

    # Define the step metric
    wandb.define_metric("train/step")

    # Set all train/* metrics to use train/step as x-axis
    wandb.define_metric("train/*", step_metric="train/step")
```

Call this once after `wandb.init()`, before any `wandb.log()` calls. Without this, W&B uses its own internal step counter, which resets to 0 on run resume.

---

## Gradient Spike Alerts

Detect gradient spikes and send alerts:

```python
class GradNormMonitor:
    def __init__(self, window=100, spike_multiplier=10.0):
        self.history = []
        self.window = window
        self.spike_multiplier = spike_multiplier

    def check_and_alert(self, rank, step, grad_norm):
        if rank != 0:
            return

        self.history.append(grad_norm)
        if len(self.history) > self.window:
            self.history.pop(0)

        if len(self.history) < 10:
            return  # Not enough history

        running_avg = sum(self.history[:-1]) / len(self.history[:-1])

        if grad_norm > self.spike_multiplier * running_avg:
            wandb.alert(
                title="Gradient Spike Detected",
                text=f"Step {step}: grad_norm={grad_norm:.2f} "
                     f"({self.spike_multiplier}x running avg={running_avg:.2f})",
                level=wandb.AlertLevel.WARN,
                wait_duration=60  # don't alert more than once per 60s
            )
```

---

## wandb.finish() in Cleanup

Always call `wandb.finish()` to ensure all buffered metrics are flushed:

```python
def cleanup_logging(rank):
    if rank != 0:
        return

    try:
        wandb.finish()
    except Exception:
        pass  # finish() failing should not crash cleanup
```

Call in the `finally` block of the training function:

```python
try:
    trainer.train()
finally:
    trainer.cleanup()    # DDP destroy
    cleanup_logging(rank)  # W&B finish
```

Without `wandb.finish()`, the last few logged steps (in W&B's internal buffer) may not be uploaded, especially if the process exits quickly after the training loop.

---

## Log Frequency Strategy

Not all metrics need to be logged every step. Suggested frequencies:

| Metric                   | Frequency            | Reason                                    |
|--------------------------|----------------------|-------------------------------------------|
| loss, grad_norm, lr, tau | Every step           | Critical for debugging; low overhead      |
| gpu_mem_gb               | Every step           | Low overhead, helps catch memory leaks    |
| gpu_mem_reserved_gb      | Every 100 steps      | Slower to query, less critical            |
| prediction grid          | Every 1000 steps     | Image upload is slow; visual quality      |
| weight histograms        | Every 5000 steps     | Very slow (requires iterating all params) |

For very frequent logging (every step at high throughput), consider batching logs:

```python
# Buffer N steps of metrics, then log all at once
log_buffer = []
log_buffer.append({'train/loss': loss, 'train/step': step})
if len(log_buffer) >= 10:
    for entry in log_buffer:
        wandb.log(entry)
    log_buffer.clear()
```

However, for most training loops with moderate step throughput (<100 steps/sec), per-step logging adds negligible overhead (<1ms per step).
