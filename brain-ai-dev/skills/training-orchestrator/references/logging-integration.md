# Logging Integration Reference: TensorBoard and Weights & Biases

This document is the authoritative reference for metric logging in the brain_ai seven-phase training pipeline. All training scripts, ablation runners, and phase orchestrators must follow the conventions defined here. The logging subsystem writes to TensorBoard, Weights & Biases (W&B), and an append-only JSONL file simultaneously through a unified `MetricLogger` abstraction. The system must tolerate either backend being unavailable without crashing.

---

## 1. Metric Namespace Convention

Adopt a hierarchical, slash-delimited namespace for every logged metric. Consistent naming enables cross-phase dashboards and automated comparison scripts.

### Top-Level Prefixes

| Prefix | Scope | Examples |
|---|---|---|
| `train/` | Training split metrics | `train/loss`, `train/accuracy`, `train/spike_rate` |
| `val/` | Validation split metrics | `val/loss`, `val/accuracy` |
| `test/` | Held-out test metrics | `test/accuracy`, `test/f1` |
| `system/` | Hardware and throughput | `system/gpu_utilization`, `system/memory_allocated_gb`, `system/throughput_tokens_sec` |
| `lr/` | Learning rates | `lr/base`, `lr/phase4`, `lr/inner` |
| `grad_norm/` | Gradient norms | `grad_norm/global`, `grad_norm/snn_core`, `grad_norm/workspace` |
| `weight_norm/` | Weight norms | `weight_norm/snn_core`, `weight_norm/encoder_vision` |

### Phase-Specific Metrics

Use `phase{N}/` as a secondary prefix for metrics that are meaningful only within a specific training phase. This avoids polluting the top-level namespace with transient metrics.

| Phase | Example Metrics |
|---|---|
| Phase 1 (SNN Core) | `phase1/spike_rate`, `phase1/membrane_potential_mean`, `phase1/dead_neuron_fraction` |
| Phase 2 (Encoders) | `phase2/vision_recon_loss`, `phase2/text_perplexity`, `phase2/audio_cer` |
| Phase 3 (HTM) | `phase3/anomaly_mean`, `phase3/sequence_accuracy`, `phase3/reflex_hit_rate` |
| Phase 4 (Global Workspace) | `phase4/ignition_rate`, `phase4/broadcast_entropy`, `phase4/selection_confidence` |
| Phase 5 (Active Inference) | `phase5/efe_pragmatic`, `phase5/efe_epistemic`, `phase5/empowerment` |
| Phase 6 (Reasoning) | `phase6/system1_fraction`, `phase6/ltn_sat_level`, `phase6/rule_accuracy` |
| Phase 7 (Meta-Learning) | `phase7/maml/inner_loss_step3`, `phase7/task2vec/cluster_purity`, `phase7/modulator_da_mean` |

For module-specific metrics within a phase, nest further: `phase{N}/{module}/metric_name`.

### Step Tracking

Log two step counters with every metric entry:

- **`global_step`**: Monotonically increasing across all phases. Never resets. This is the primary x-axis for TensorBoard scalars and W&B charts.
- **`phase_step`**: Resets to zero at the start of each phase. Useful for comparing learning dynamics across phases or ablations at the same relative progress.

Record both in every log call. TensorBoard uses `global_step` as the `step` argument; `phase_step` is logged as a separate scalar (`phase{N}/phase_step`) and included in JSONL entries.

---

## 2. TensorBoard Integration

Use `torch.utils.tensorboard.SummaryWriter` as the TensorBoard backend. Do not use third-party TensorBoard wrappers.

### Log Directory

Place TensorBoard event files at:

```
runs/<run_id>/logs/tensorboard/
```

Pass this path as `log_dir` to `SummaryWriter`. For ablation runs, each ablation combination gets its own `run_id`, so event files are naturally isolated.

### Scalar Logging

Log all namespace metrics as scalars. Use `global_step` as the step argument:

```python
writer.add_scalar("train/loss", loss_value, global_step=step)
writer.add_scalar("phase4/ignition_rate", ignition_rate, global_step=step)
writer.add_scalar("lr/base", current_lr, global_step=step)
writer.add_scalar("grad_norm/global", total_norm, global_step=step)
```

For batch logging of multiple scalars at the same step, prefer `add_scalars` only when the metrics belong to the same logical group and should appear on a single chart:

```python
writer.add_scalars("phase5/efe_components", {
    "pragmatic": efe_prag,
    "epistemic": efe_epist,
    "instrumental": efe_instr,
}, global_step=step)
```

### Histogram Logging

Log weight distributions per module. This is expensive; configure the interval via `histogram_interval` (default: 500 steps). Do not log histograms every step.

```python
if step % histogram_interval == 0:
    for name, param in model.named_parameters():
        if param.requires_grad:
            writer.add_histogram(f"weights/{name}", param.data, global_step=step)
            if param.grad is not None:
                writer.add_histogram(f"grads/{name}", param.grad.data, global_step=step)
```

For the SNN core, also log membrane potential distributions:

```python
writer.add_histogram("phase1/membrane_potentials", membrane_values, global_step=step)
```

### Image Logging

Log sample predictions, attention maps, and workspace competition visualizations. Configure via `image_interval` (default: 1000 steps). Resize images to a maximum of 256x256 before logging to limit event file size.

```python
import torch.nn.functional as F

if step % image_interval == 0:
    # Resize to max 256x256
    img = F.interpolate(sample_img.unsqueeze(0), size=(256, 256), mode="bilinear")
    writer.add_image("val/sample_prediction", img.squeeze(0), global_step=step)

    # Attention map (normalize to [0, 1])
    attn_map = attention_weights[0].detach().cpu()
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    writer.add_image("phase4/attention_map", attn_map.unsqueeze(0), global_step=step)
```

### Text Logging

Log configuration summaries at run start and phase transition markers at phase boundaries:

```python
import json

# At run start
config_text = json.dumps(config_dict, indent=2)
writer.add_text("config/full", f"```json\n{config_text}\n```", global_step=0)

# At phase transition
writer.add_text("events/phase_transition",
    f"Phase {from_phase} -> {to_phase} at step {step}. "
    f"Duration: {duration_str}. Best val/loss: {best_loss:.4f}",
    global_step=step)
```

### Custom Scalars Layout

Define a custom scalars layout so the TensorBoard dashboard groups metrics by phase. Register this layout once at writer creation:

```python
from torch.utils.tensorboard import SummaryWriter
from tensorboard.plugins.custom_scalar import layout_pb2

layout = layout_pb2.Layout(
    category=[
        layout_pb2.Category(
            title="Training Overview",
            chart=[
                layout_pb2.Chart(
                    title="Loss (all phases)",
                    multiline=layout_pb2.MultilineChartContent(
                        tag=[r"train/loss", r"val/loss"],
                    ),
                ),
                layout_pb2.Chart(
                    title="Learning Rate",
                    multiline=layout_pb2.MultilineChartContent(
                        tag=[r"lr/.*"],
                    ),
                ),
            ],
        ),
        layout_pb2.Category(
            title="Phase 4: Global Workspace",
            chart=[
                layout_pb2.Chart(
                    title="Ignition Dynamics",
                    multiline=layout_pb2.MultilineChartContent(
                        tag=[r"phase4/ignition_rate", r"phase4/broadcast_entropy"],
                    ),
                ),
            ],
        ),
    ],
)

writer = SummaryWriter(log_dir=tb_log_dir)
writer.file_writer.add_summary(layout)
```

### Flush Policy

Flush the writer at two trigger points:

1. **Every 100 steps** (configurable via `flush_interval`).
2. **At every phase boundary** (immediately before phase transition marker).

```python
if step % flush_interval == 0:
    writer.flush()
```

Do not call `flush()` on every step. The overhead is measurable at high throughput.

### Resume Behavior

On resume, create a new `SummaryWriter` pointing to the same `log_dir`. TensorBoard handles continuity automatically: the new event file appends to the existing log directory, and the higher `global_step` values ensure charts are continuous. Do not delete or truncate existing event files.

```python
# Same log_dir on resume -- TensorBoard appends naturally
writer = SummaryWriter(log_dir=tb_log_dir)  # tb_log_dir is unchanged from original run
```

---

## 3. Weights & Biases Integration

### Initialization

Initialize W&B once at the start of the run. Pass the full resolved `BrainAIConfig` as the config argument:

```python
import wandb
from dataclasses import asdict

wandb_run = wandb.init(
    project="brain-ai",
    name=run_id,
    id=wandb_run_id,          # Deterministic ID for resume
    config=asdict(config),     # Full BrainAIConfig
    group=ablation_id,         # Non-null only for ablation runs
    tags=[f"phase{phase}", mode],
    resume=resume_mode,        # "never", "allow", or "must"
)
```

### Resume Mechanics

Use three resume modes depending on context:

| Mode | When to Use | Behavior |
|---|---|---|
| `resume="never"` | Fresh runs (default) | Always creates a new W&B run. Fails if `id` already exists. |
| `resume="allow"` | General-purpose resume | Resumes if a run with this `id` exists; creates new otherwise. Use this for `train_full_pipeline.py` with `--start-phase`. |
| `resume="must"` | Strict resume after crash | Fails immediately if the run does not already exist in W&B. Use this when resuming from a checkpoint where the W&B run must already be present. |

Store `wandb_run_id` in `manifest.json` under the `logging` section so that resume is deterministic:

```json
{
  "logging": {
    "wandb_run_id": "abc123xyz",
    "wandb_project": "brain-ai",
    "tensorboard_log_dir": "runs/run_001/logs/tensorboard/"
  }
}
```

On resume, read `wandb_run_id` from the manifest and pass it to `wandb.init(id=...)`.

### Config Logging

Log the full `BrainAIConfig` as `wandb.config` via `asdict()`. This enables W&B's hyperparameter comparison UI. Additionally, log feature flags as individual config entries for easy filtering:

```python
wandb.config.update({
    "feature_flags": {
        "use_snn": config.use_snn,
        "use_htm": config.use_htm,
        "use_workspace": config.use_workspace,
        "use_symbolic": config.use_symbolic,
        "use_meta": config.use_meta,
        "use_engram": config.use_engram,
    }
})
```

### Metric Logging

Log metrics using `wandb.log()` with the same namespace conventions as TensorBoard. Always pass `step=global_step`:

```python
wandb.log({
    "train/loss": loss_value,
    "train/accuracy": accuracy,
    "phase4/ignition_rate": ignition_rate,
    "lr/base": current_lr,
    "grad_norm/global": total_norm,
    "global_step": step,
    "phase_step": phase_step,
}, step=step)
```

Batch multiple metrics into a single `wandb.log()` call per training step. Do not call `wandb.log()` multiple times for the same step; W&B deduplicates by step but this adds overhead.

### Artifact Logging

Log phase boundary checkpoints as W&B artifacts. This enables lineage tracking across phases:

```python
artifact = wandb.Artifact(
    name=f"phase{phase}_boundary",
    type="checkpoint",
    description=f"Phase {phase} boundary checkpoint at step {step}",
    metadata={
        "phase": phase,
        "global_step": step,
        "best_val_loss": best_val_loss,
    },
)
artifact.add_file(str(boundary_checkpoint_path))
wandb_run.log_artifact(artifact)
```

### Tables for Ablation Results

Log ablation comparison results as a `wandb.Table` for interactive visualization:

```python
columns = ["ablation_id", "use_engram", "use_ltn", "use_learnable_delays",
           "val_loss", "val_accuracy", "duration_sec", "status"]
table = wandb.Table(columns=columns)

for run in ablation_results:
    table.add_data(
        run.ablation_id, run.use_engram, run.use_ltn, run.use_learnable_delays,
        run.val_loss, run.val_accuracy, run.duration_sec, run.status,
    )

wandb.log({"ablation/results": table})
```

### Alerts

Send W&B alerts on critical failures. These trigger email or Slack notifications if configured in the W&B project:

```python
# NaN loss detection
if torch.isnan(loss):
    wandb.alert(
        title="NaN Loss Detected",
        text=f"Loss became NaN at step {step} during phase {phase}. "
             f"Run: {run_id}. Last valid loss: {last_valid_loss:.4f}.",
        level=wandb.AlertLevel.ERROR,
    )

# Training failure
wandb.alert(
    title="Training Failed",
    text=f"Phase {phase} failed with exception: {str(exc)}. Run: {run_id}.",
    level=wandb.AlertLevel.ERROR,
)
```

### Finishing a Run

Call `wandb.finish()` at the end of every run. Record final summary metrics before finishing:

```python
wandb.run.summary["final/val_loss"] = final_val_loss
wandb.run.summary["final/val_accuracy"] = final_val_accuracy
wandb.run.summary["final/total_steps"] = total_steps
wandb.run.summary["final/total_duration_sec"] = total_duration
wandb.run.summary["final/status"] = "completed"  # or "failed", "interrupted"
wandb.finish()
```

---

## 4. MetricLogger Abstraction

Implement a `MetricLogger` class in `brain_ai/training/logging_utils.py` that provides a dual-backend interface. All training scripts interact with this class exclusively; they never call TensorBoard or W&B APIs directly.

```python
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch


class MetricLogger:
    """Unified logging to TensorBoard, Weights & Biases, and JSONL.

    Thread-safe. Falls back to no-op if both backends are unavailable.
    """

    def __init__(
        self,
        run_dir: str,
        config: dict,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: str = "brain-ai",
        wandb_run_id: Optional[str] = None,
        wandb_resume: str = "never",
        wandb_group: Optional[str] = None,
        run_id: Optional[str] = None,
        flush_interval: int = 100,
        histogram_interval: int = 500,
        image_interval: int = 1000,
    ):
        self._run_dir = Path(run_dir)
        self._lock = threading.Lock()
        self._step_count = 0
        self._flush_interval = flush_interval
        self._histogram_interval = histogram_interval
        self._image_interval = image_interval

        # TensorBoard
        self._tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = self._run_dir / "logs" / "tensorboard"
                tb_dir.mkdir(parents=True, exist_ok=True)
                self._tb_writer = SummaryWriter(log_dir=str(tb_dir))
            except ImportError:
                pass

        # Weights & Biases
        self._wandb_run = None
        if use_wandb:
            try:
                import wandb
                self._wandb_run = wandb.init(
                    project=wandb_project,
                    name=run_id,
                    id=wandb_run_id,
                    config=config,
                    group=wandb_group,
                    resume=wandb_resume,
                    reinit=True,
                )
            except ImportError:
                pass

        # JSONL
        jsonl_path = self._run_dir / "metrics.jsonl"
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._jsonl_file = open(jsonl_path, "a", buffering=1)  # line-buffered

    def log_scalar(
        self,
        tag: str,
        value: float,
        step: int,
        phase_step: Optional[int] = None,
    ) -> None:
        with self._lock:
            if self._tb_writer is not None:
                self._tb_writer.add_scalar(tag, value, global_step=step)
            if self._wandb_run is not None:
                import wandb
                payload = {tag: value, "global_step": step}
                if phase_step is not None:
                    payload["phase_step"] = phase_step
                wandb.log(payload, step=step)

    def log_scalars(
        self,
        tag_value_dict: Dict[str, float],
        step: int,
        phase: Optional[int] = None,
        phase_step: Optional[int] = None,
    ) -> None:
        with self._lock:
            if self._tb_writer is not None:
                for tag, value in tag_value_dict.items():
                    self._tb_writer.add_scalar(tag, value, global_step=step)
            if self._wandb_run is not None:
                import wandb
                payload = dict(tag_value_dict)
                payload["global_step"] = step
                if phase_step is not None:
                    payload["phase_step"] = phase_step
                wandb.log(payload, step=step)
            # JSONL
            entry = {
                "step": step,
                "phase": phase,
                "phase_step": phase_step,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "metrics": tag_value_dict,
            }
            self._jsonl_file.write(json.dumps(entry) + "\n")

            self._step_count += 1
            if self._step_count % self._flush_interval == 0:
                self.flush()

    def log_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        with self._lock:
            if self._tb_writer is not None:
                self._tb_writer.add_histogram(tag, values, global_step=step)
            if self._wandb_run is not None:
                import wandb
                wandb.log({tag: wandb.Histogram(values.cpu().numpy())}, step=step)

    def log_image(
        self, tag: str, image: torch.Tensor, step: int, caption: str = ""
    ) -> None:
        with self._lock:
            if self._tb_writer is not None:
                self._tb_writer.add_image(tag, image, global_step=step)
            if self._wandb_run is not None:
                import wandb
                img = wandb.Image(image.permute(1, 2, 0).cpu().numpy(), caption=caption)
                wandb.log({tag: img}, step=step)

    def log_text(self, tag: str, text: str, step: int) -> None:
        with self._lock:
            if self._tb_writer is not None:
                self._tb_writer.add_text(tag, text, global_step=step)
            if self._wandb_run is not None:
                import wandb
                wandb.log({tag: wandb.Html(f"<pre>{text}</pre>")}, step=step)

    def log_phase_transition(
        self,
        from_phase: int,
        to_phase: int,
        step: int,
        duration_sec: float = 0.0,
        best_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        summary = (
            f"Phase {from_phase} -> {to_phase} | step={step} | "
            f"duration={duration_sec:.1f}s"
        )
        if best_metrics:
            summary += f" | best_metrics={best_metrics}"

        self.log_text("events/phase_transition", summary, step)
        self.log_scalars(
            {"phase_transition": float(to_phase)}, step, phase=to_phase, phase_step=0
        )

        # JSONL special entry
        entry = {
            "step": step,
            "event": "phase_transition",
            "from_phase": from_phase,
            "to_phase": to_phase,
            "duration_sec": duration_sec,
            "best_metrics": best_metrics,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._jsonl_file.write(json.dumps(entry) + "\n")
        self.flush()

    def flush(self) -> None:
        if self._tb_writer is not None:
            self._tb_writer.flush()
        self._jsonl_file.flush()

    def close(self) -> None:
        self.flush()
        if self._tb_writer is not None:
            self._tb_writer.close()
        if self._wandb_run is not None:
            import wandb
            wandb.finish()
        self._jsonl_file.close()
```

### Key Design Properties

- **Dual-backend**: Every `log_*` method writes to both TensorBoard and W&B (when enabled). The caller never needs to know which backends are active.
- **No-op safety**: If TensorBoard is not installed and W&B is disabled, all methods silently do nothing. Training never crashes due to a missing logging dependency.
- **Thread safety**: All public methods acquire `self._lock` before writing. This supports logging from data loader worker threads or async evaluation threads.
- **JSONL always on**: The JSONL file writes regardless of backend availability. This provides a minimal offline record even in environments without TensorBoard or W&B.

---

## 5. metrics.jsonl Format

Maintain an append-only JSONL file at `runs/<run_id>/metrics.jsonl`. Each line is a self-contained JSON object.

### Standard Metric Entry

```json
{"step": 1000, "phase": 4, "phase_step": 500, "timestamp": "2026-02-20T14:30:00Z", "metrics": {"train/loss": 0.532, "train/accuracy": 0.871, "phase4/ignition_rate": 0.43, "lr/base": 0.00028}}
```

### Phase Transition Entry

```json
{"step": 5000, "event": "phase_transition", "from_phase": 3, "to_phase": 4, "duration_sec": 3621.5, "best_metrics": {"val/loss": 0.312, "val/accuracy": 0.912}, "timestamp": "2026-02-20T15:30:21Z"}
```

### Resume Entry

```json
{"step": 5000, "event": "resume", "checkpoint": "runs/run_001/checkpoints/phase4/ckpt_step00005000.pt", "global_step": 5000, "phase": 4, "timestamp": "2026-02-20T16:00:00Z"}
```

### Offline Analysis

Write a summarization utility that reads `metrics.jsonl` and computes per-metric statistics:

```python
import json
from collections import defaultdict

def summarize_jsonl(path: str) -> dict:
    """Compute min/max/mean/final for each metric from a JSONL file."""
    metrics = defaultdict(list)
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            if "metrics" not in entry:
                continue
            for key, value in entry["metrics"].items():
                metrics[key].append((entry["step"], value))

    summary = {}
    for key, values in metrics.items():
        vals = [v for _, v in values]
        summary[key] = {
            "min": min(vals),
            "max": max(vals),
            "mean": sum(vals) / len(vals),
            "final": values[-1][1],
            "final_step": values[-1][0],
            "num_entries": len(vals),
        }
    return summary
```

---

## 6. Phase Transition Logging

At every phase boundary, record a structured marker across all three backends. This enables automated detection of phase transitions in post-hoc analysis.

### Required Actions at Phase Boundary

1. **Flush all writers** before logging the transition marker. This prevents data loss if the transition logic fails.
2. **Log a text summary** to TensorBoard with the phase number, best metrics from the completing phase, and wall-clock duration.
3. **Log a metric entry** to W&B with `phase_transition=True` as a marker field. Include the same best metrics and duration.
4. **Write a special JSONL entry** with `"event": "phase_transition"` (see format above).
5. **Log the total steps completed** in the outgoing phase as `phase{N}/total_steps`.

```python
# Example usage in pipeline orchestrator
logger.flush()
logger.log_phase_transition(
    from_phase=3,
    to_phase=4,
    step=global_step,
    duration_sec=phase_duration,
    best_metrics={"val/loss": best_val_loss, "val/accuracy": best_val_acc},
)
logger.log_scalar(f"phase{from_phase}/total_steps", phase_step, step=global_step)
```

---

## 7. Gradient and Weight Monitoring

Monitor gradient and weight statistics to detect training instabilities early. Log these metrics from within the training loop, after `loss.backward()` and before `optimizer.step()`.

### Per-Module Gradient Norms

Compute and log the L2 norm of gradients for each major module:

```python
def log_gradient_norms(model: torch.nn.Module, logger: MetricLogger, step: int):
    """Log per-module and global gradient norms."""
    total_norm_sq = 0.0
    module_norms = {}

    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        param_norm = param.grad.data.norm(2).item()
        total_norm_sq += param_norm ** 2

        # Extract top-level module name (e.g., "snn_core" from "snn_core.layers.0.weight")
        module_name = name.split(".")[0]
        if module_name not in module_norms:
            module_norms[module_name] = 0.0
        module_norms[module_name] += param_norm ** 2

    global_norm = total_norm_sq ** 0.5
    logger.log_scalar("grad_norm/global", global_norm, step)

    for module_name, norm_sq in module_norms.items():
        logger.log_scalar(f"grad_norm/{module_name}", norm_sq ** 0.5, step)

    return global_norm
```

### Weight Norms

Log weight norms at a lower frequency (every `weight_norm_interval` steps, default 500):

```python
def log_weight_norms(model: torch.nn.Module, logger: MetricLogger, step: int):
    """Log per-module weight norms."""
    module_norms = {}
    for name, param in model.named_parameters():
        module_name = name.split(".")[0]
        if module_name not in module_norms:
            module_norms[module_name] = 0.0
        module_norms[module_name] += param.data.norm(2).item() ** 2

    for module_name, norm_sq in module_norms.items():
        logger.log_scalar(f"weight_norm/{module_name}", norm_sq ** 0.5, step)
```

### Dead Neuron Detection (SNN Modules)

For spiking neural network modules, track the fraction of neurons that produce zero spikes across an evaluation window. A dead neuron fraction above 0.5 indicates potential gradient flow problems:

```python
def log_dead_neuron_fraction(
    spike_counts: torch.Tensor,  # shape: (num_neurons,), accumulated over eval window
    logger: MetricLogger,
    step: int,
    layer_name: str = "snn_core",
):
    """Log the fraction of neurons that never spike."""
    dead_fraction = (spike_counts == 0).float().mean().item()
    logger.log_scalar(f"phase1/{layer_name}/dead_neuron_fraction", dead_fraction, step)
    if dead_fraction > 0.5:
        logger.log_text(
            "warnings/dead_neurons",
            f"WARNING: {dead_fraction:.1%} of neurons in {layer_name} are dead at step {step}",
            step,
        )
```

### Gradient Explosion and Vanishing Alerts

Check gradient norms against thresholds on every step. Log warnings when norms exit the healthy range:

```python
GRAD_NORM_UPPER = 100.0
GRAD_NORM_LOWER = 1e-7

def check_gradient_health(global_norm: float, logger: MetricLogger, step: int):
    """Alert on gradient explosion or vanishing."""
    if global_norm > GRAD_NORM_UPPER:
        logger.log_text(
            "warnings/gradient_explosion",
            f"Gradient explosion detected: norm={global_norm:.2f} at step {step}",
            step,
        )
        if logger._wandb_run is not None:
            import wandb
            wandb.alert(
                title="Gradient Explosion",
                text=f"Global gradient norm {global_norm:.2f} exceeds threshold "
                     f"{GRAD_NORM_UPPER} at step {step}.",
                level=wandb.AlertLevel.WARN,
            )
    elif global_norm < GRAD_NORM_LOWER:
        logger.log_text(
            "warnings/gradient_vanishing",
            f"Gradient vanishing detected: norm={global_norm:.2e} at step {step}",
            step,
        )
```

---

## 8. Ablation-Specific Logging

### Isolation

Each ablation run gets its own `run_id`, its own TensorBoard `log_dir`, and its own W&B run. Do not share writers or run objects across ablation combinations.

### W&B Grouping

Use `wandb.init(group=ablation_id)` to group all runs that belong to the same ablation experiment. This enables W&B's grouped run view and parallel coordinates plots:

```python
wandb.init(
    project="brain-ai",
    name=f"{ablation_id}_{combo_hash}",
    id=wandb_run_id,
    group=ablation_id,  # Groups all ablation runs together
    config=combo_config,
    tags=["ablation", f"phase{phase}"],
)
```

### Ablation Tag in Metrics

Include `ablation_tag` in every JSONL entry for filtering during post-hoc analysis:

```json
{"step": 100, "phase": 4, "phase_step": 100, "ablation_tag": "engram=true_ltn=false_delays=true", "metrics": {"train/loss": 0.65}}
```

### Comparison Dashboard

After all ablation runs complete, log a comparison table to W&B. Use parallel coordinates for visualizing the relationship between toggle values and final metrics:

```python
# After all ablation runs complete
wandb.init(project="brain-ai", name=f"{ablation_id}_summary", job_type="analysis")

table = wandb.Table(
    columns=["use_engram", "use_ltn", "use_learnable_delays",
             "val_loss", "val_accuracy", "throughput_tokens_sec"],
    data=[[r.use_engram, r.use_ltn, r.use_delays,
           r.val_loss, r.val_acc, r.throughput] for r in results],
)
wandb.log({"ablation/comparison": table})
wandb.finish()
```

---

## 9. Resume Logging Semantics

Resume must be append-only across all backends. Never overwrite or truncate prior log entries.

### TensorBoard

`SummaryWriter` naturally appends when pointed to the same `log_dir`. A new event file is created on each writer instantiation, but TensorBoard merges all event files in the directory by `global_step`. Ensure `global_step` on resume is strictly greater than the last step before interruption (read from the checkpoint):

```python
# On resume
resumed_step = checkpoint["global_step"]
# Next log call uses resumed_step + 1 or higher
# TensorBoard auto-merges
writer = SummaryWriter(log_dir=same_tb_log_dir)
```

### Weights & Biases

Resume using the stored `wandb_run_id` from `manifest.json`:

```python
stored_wandb_id = manifest["logging"]["wandb_run_id"]
wandb.init(
    project="brain-ai",
    id=stored_wandb_id,
    resume="allow",  # or "must" for strict resume
)
```

W&B continues the run's history from the last logged step. New `wandb.log()` calls with higher step values extend the charts seamlessly.

### JSONL

Open the JSONL file in append mode (`"a"`). Write a resume marker entry:

```python
resume_entry = {
    "step": resumed_step,
    "event": "resume",
    "checkpoint": str(checkpoint_path),
    "global_step": resumed_step,
    "phase": current_phase,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
jsonl_file.write(json.dumps(resume_entry) + "\n")
```

### Manifest Update

On resume, append a new entry to the `resume_events` list in `manifest.json`. Do not overwrite the original manifest fields:

```json
{
  "resume_events": [
    {
      "timestamp": "2026-02-20T16:00:00Z",
      "checkpoint": "runs/run_001/checkpoints/phase4/ckpt_step00005000.pt",
      "global_step": 5000,
      "phase": 4,
      "reason": "manual_resume"
    }
  ]
}
```

---

## 10. Performance Considerations

Keep logging overhead below 1% of total training step time. The following guidelines apply at all scales.

### Batch Metrics Before Logging

Accumulate metrics during gradient accumulation micro-batches. Call `log_scalars()` once per optimizer step, not per micro-batch:

```python
accumulated_loss = 0.0
for micro_step in range(gradient_accumulation_steps):
    loss = forward_backward(micro_batch)
    accumulated_loss += loss.item()

# Log once per optimizer step
avg_loss = accumulated_loss / gradient_accumulation_steps
logger.log_scalars({"train/loss": avg_loss, "lr/base": current_lr}, step=global_step)
```

### Histogram Frequency

Histogram logging serializes full tensors and is 10-100x more expensive than scalar logging. Set `histogram_interval` to 500 at minimum. For production-scale models (1B+ parameters), use 1000:

| Scale | Recommended `histogram_interval` |
|---|---|
| Dev / minimal | 100 |
| Production 1B | 500 |
| Production 3B-7B | 1000 |

### Image Logging

Resize images to at most 256x256 before logging. For attention maps with large spatial dimensions, downsample first:

```python
import torch.nn.functional as F

if img.shape[-1] > 256 or img.shape[-2] > 256:
    img = F.interpolate(img.unsqueeze(0), size=(256, 256), mode="bilinear").squeeze(0)
```

Set `image_interval` to at least 1000 steps.

### W&B Async Uploads

W&B uploads data asynchronously by default. Do not change this behavior. If disk I/O is a bottleneck, set `WANDB_MODE=offline` and sync later with `wandb sync`.

### System Metrics Frequency

Log `system/` metrics (GPU utilization, memory, throughput) every 50-100 steps, not every step. Use `torch.cuda` for GPU metrics:

```python
if step % 50 == 0 and torch.cuda.is_available():
    logger.log_scalars({
        "system/gpu_utilization": torch.cuda.utilization(),
        "system/memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "system/memory_reserved_gb": torch.cuda.memory_reserved() / 1e9,
    }, step=global_step)
```

### Writer Lifecycle

Create the `MetricLogger` once at the start of the run (or on resume). Close it once at the end. Do not create and destroy writers per phase; use a single writer instance across all phases within a pipeline run. Phase transitions are marked with log entries, not new writer objects.

---

## Summary of Configuration Defaults

| Parameter | Default | Configurable Via |
|---|---|---|
| `flush_interval` | 100 steps | `MetricLogger.__init__` |
| `histogram_interval` | 500 steps | `MetricLogger.__init__` |
| `image_interval` | 1000 steps | `MetricLogger.__init__` |
| `weight_norm_interval` | 500 steps | Training loop |
| `system_metrics_interval` | 50 steps | Training loop |
| `grad_norm_upper_threshold` | 100.0 | `check_gradient_health` |
| `grad_norm_lower_threshold` | 1e-7 | `check_gradient_health` |
| `image_max_size` | 256x256 | Training loop |
| `wandb_resume_mode` | `"never"` | `MetricLogger.__init__` |

All intervals and thresholds should be configurable but have sensible defaults that work for both dev and production modes. Dev mode can use more aggressive logging (lower intervals) since runs are short. Production mode should use the defaults listed above to keep overhead minimal during multi-day training runs.
