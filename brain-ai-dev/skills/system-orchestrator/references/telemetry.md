# Telemetry

## Overview

The orchestrator emits structured events consumed by logging backends,
debuggers, and profilers. No ad-hoc `print()` statements in module code.

## TelemetrySink Interface

```python
class TelemetrySink:
    """Abstract interface for telemetry consumers."""

    def on_forward_start(self, batch_meta: Dict[str, Any]):
        """Called before pipeline execution begins."""
        pass

    def on_module_start(self, name: str):
        """Called before a pipeline stage runs."""
        pass

    def on_module_metrics(self, name: str, metrics: Dict[str, Any]):
        """Called with per-module metrics during/after execution."""
        pass

    def on_module_end(self, name: str):
        """Called after a pipeline stage completes."""
        pass

    def on_forward_end(self, output_meta: Dict[str, Any]):
        """Called after pipeline execution completes."""
        pass
```

## Built-in Implementations

### NullSink

Default when no telemetry is configured. All methods are no-ops.

### LoggingSink

Writes structured events to Python `logging`:

```python
class LoggingSink(TelemetrySink):
    def __init__(self, logger_name="brain_ai.telemetry"):
        self.logger = logging.getLogger(logger_name)
        self._timers = {}

    def on_module_start(self, name):
        self._timers[name] = time.perf_counter()

    def on_module_metrics(self, name, metrics):
        self.logger.info(f"[{name}] {metrics}")

    def on_module_end(self, name):
        elapsed = (time.perf_counter() - self._timers.pop(name, 0)) * 1000
        self.logger.debug(f"[{name}] {elapsed:.1f}ms")
```

### WandBSink

Logs metrics to Weights & Biases:

```python
class WandBSink(TelemetrySink):
    def __init__(self, prefix="brain_ai"):
        self.prefix = prefix
        self._step_metrics = {}

    def on_module_metrics(self, name, metrics):
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self._step_metrics[f"{self.prefix}/{name}/{k}"] = v
            elif isinstance(v, Tensor) and v.numel() == 1:
                self._step_metrics[f"{self.prefix}/{name}/{k}"] = v.item()

    def on_forward_end(self, output_meta):
        import wandb
        wandb.log(self._step_metrics)
        self._step_metrics = {}
```

### TensorBoardSink

```python
class TensorBoardSink(TelemetrySink):
    def __init__(self, writer, global_step_fn):
        self.writer = writer
        self.global_step_fn = global_step_fn

    def on_module_metrics(self, name, metrics):
        step = self.global_step_fn()
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(f"modules/{name}/{k}", v, step)
```

### ProfilerSink

Wraps `torch.profiler` for per-module profiling:

```python
class ProfilerSink(TelemetrySink):
    def on_module_start(self, name):
        torch.cuda.nvtx.range_push(name)

    def on_module_end(self, name):
        torch.cuda.nvtx.range_pop()
```

### CompositeSink

Combine multiple sinks:

```python
class CompositeSink(TelemetrySink):
    def __init__(self, *sinks):
        self.sinks = sinks

    def on_forward_start(self, batch_meta):
        for s in self.sinks:
            s.on_forward_start(batch_meta)
    # ... delegate all methods
```

## Metrics Emitted by Default

Each stage emits standard metrics via `on_module_metrics`:

| Stage | Metrics |
|-------|---------|
| encode | per-modality: `{mod}_feats_norm`, `{mod}_salience_mean` |
| workspace | `ignition_steps`, `winner_entropy`, `broadcast_norm` |
| htm | `anomaly_score`, `promoted_patterns`, `active_columns_pct` |
| reasoning | `used_sys2` (bool), `sys1_confidence`, `reasoning_steps` |
| meta | `DA`, `ACh`, `NE`, `5HT` (scalar values) |
| decision | `efe_pragmatic`, `efe_epistemic`, `action_entropy` |
| output | `confidence`, `inference_time_ms` |

## Integration in PipelinePlan

```python
# In PipelinePlan.execute():
for stage in self.stages:
    if ctx.telemetry:
        ctx.telemetry.on_module_start(stage.name)

    if stage.enabled:
        ctx = stage.run(ctx)
    else:
        ctx = stage.bypass(ctx)

    # Emit standard metrics
    if ctx.telemetry and stage.name in ctx.details:
        ctx.telemetry.on_module_metrics(stage.name, ctx.details[stage.name])

    if ctx.telemetry:
        ctx.telemetry.on_module_end(stage.name)
```

## Debugging Use Cases

### "Why did System 2 trigger?"

```python
sink = LoggingSink()
output = brain(inputs, return_details=True, telemetry=sink)
# Log output shows:
# [reasoning] {'used_sys2': True, 'sys1_confidence': 0.42, 'reasoning_steps': 5}
```

### "Why is anomaly spiking?"

```python
output = brain(inputs, return_details=True)
print(output.details.htm["anomaly_score"])  # Per-batch anomaly values
print(output.details.htm["promoted_patterns_count"])  # New patterns learned
```

### "Which modality won workspace?"

```python
output = brain(inputs, return_details=True)
print(output.details.workspace["winners"])  # Winning modality indices
print(output.details.workspace["modality_contributions"])  # Per-modality scores
```
