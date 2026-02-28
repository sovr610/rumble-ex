# Pipeline Measurement

## Timer Instrumentation

Every training step decomposes into two measurable phases:

```
t_total_step = t_data + t_fwd_bwd_opt
data_stall_ratio = t_data / t_total_step
gpu_busy_ratio = 1 - data_stall_ratio
```

### Timer Placement

```python
auditor = PipelineAuditor(device)

for batch in dataloader:
    auditor.mark_data_start()
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    torch.cuda.synchronize()          # wait for H2D to finish for accurate t_data
    auditor.mark_data_end()

    auditor.mark_compute_start()
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()          # ensure all GPU work finishes
    auditor.mark_compute_end()
```

**Key rules:**
- `mark_data_start()` fires immediately after requesting the next batch (or at loop top).
- `mark_data_end()` fires after the batch is confirmed on-GPU (after synchronize).
- `mark_compute_start()` fires right before the forward pass.
- `mark_compute_end()` fires after `synchronize()` at the end of backward+optimizer.

### CUDA Event Timers

Use `torch.cuda.Event(enable_timing=True)` for GPU-side timing:

```python
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
# ... work ...
end_event.record()
torch.cuda.synchronize()
elapsed_ms = start_event.elapsed_time(end_event)
```

CUDA events measure GPU-side elapsed time with sub-millisecond precision. They are preferred over `time.perf_counter()` for GPU work because they avoid host-device synchronization skew.

For CPU-side data loading, `time.perf_counter()` is appropriate since the work happens on the host.

## CUDA Transfer Attribution (t_h2d)

Host-to-device transfer time is the time spent in `.to(device, non_blocking=True)` plus the synchronization wait:

```python
t0 = time.perf_counter()
batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
torch.cuda.synchronize()
t_h2d = time.perf_counter() - t0
```

**non_blocking=True** allows the transfer to overlap with other CPU work, but for measurement purposes we synchronize immediately after to capture the true transfer cost.

**pin_memory=True** in the DataLoader enables page-locked host memory, which allows faster DMA transfers. When pin_memory is disabled, transfers go through a staging buffer, adding overhead.

Measure t_h2d separately from t_data_wait (time waiting for the dataloader to yield a batch) to distinguish I/O stalls from transfer stalls.

## data_metrics.json Schema

```json
{
  "version": "1.0",
  "timestamp_utc": "2025-01-15T12:00:00Z",
  "steps_measured": 500,
  "timers": {
    "t_data_p50_ms": 12.3,
    "t_data_p90_ms": 25.1,
    "t_fwd_bwd_opt_p50_ms": 85.0,
    "t_fwd_bwd_opt_p90_ms": 90.2,
    "t_total_step_p50_ms": 97.3,
    "t_total_step_p90_ms": 115.3,
    "t_h2d_p50_ms": 1.2,
    "t_h2d_p90_ms": 2.0
  },
  "stall_ratios": {
    "data_stall_ratio_p50": 0.126,
    "data_stall_ratio_p90": 0.218,
    "gpu_busy_ratio_p50": 0.874
  },
  "throughput": {
    "raw_tokens_per_sec_p50": 52000,
    "effective_tokens_per_sec_p50": 48100,
    "padding_ratio": 0.075
  },
  "dataloader_settings": {
    "num_workers": 4,
    "prefetch_factor": 2,
    "persistent_workers": true,
    "pin_memory": true,
    "batch_size": 8
  },
  "packing": {
    "mode": "sft_boundary_aware",
    "target_seq_len": 2048,
    "bucket_boundaries": [256, 512, 1024, 2048]
  }
}
```

### Required Fields

All fields are required. Metrics that are not measured should be set to `null` with a comment explaining why.

## Stall Ratio Interpretation Guide

| data_stall_ratio | Interpretation | Action |
|-----------------|----------------|--------|
| < 0.05 (5%) | Excellent. Pipeline is not a bottleneck. | No action needed. |
| 0.05 - 0.15 | Good. Minor stalls, likely acceptable. | Consider tuning if easy wins available. |
| 0.15 - 0.30 | Moderate. GPU is idle 15-30% of the time. | Tune num_workers, prefetch_factor, enable pin_memory. |
| 0.30 - 0.50 | Poor. Significant GPU underutilization. | Switch to streaming I/O, add caching, increase workers. |
| > 0.50 | Severe. GPU is idle more than half the time. | Fundamental pipeline redesign needed. Check I/O backend, storage speed, network. |

**p50 vs p90:** The p50 (median) stall ratio shows typical behavior. The p90 shows tail latency spikes. If p90 is much higher than p50, investigate intermittent stalls (GC pauses, storage hiccups, worker deaths).

**Trend over steps:** If stall ratio increases over training, suspect memory leaks in workers, cache eviction pressure, or dataset exhaustion causing reshuffle storms.

## Worker Watchdog Implementation

The watchdog detects stalled or dead DataLoader workers and logs diagnostic information.

### Design

```python
class WorkerWatchdog:
    def __init__(self, timeout_sec: float = 60.0):
        self.timeout_sec = timeout_sec
        self.last_batch_time = time.monotonic()
        self.last_sample_info = None
        self._running = True

    def on_batch_received(self, batch_info: dict):
        """Called when a batch arrives. Updates last-seen time and sample info."""
        self.last_batch_time = time.monotonic()
        self.last_sample_info = batch_info

    def check(self) -> bool:
        """Returns True if healthy, raises if stalled."""
        elapsed = time.monotonic() - self.last_batch_time
        if elapsed > self.timeout_sec:
            raise DataLoaderStallError(
                f"No batch received for {elapsed:.1f}s. "
                f"Last sample info: {self.last_sample_info}"
            )
        return True
```

### What to Log on Stall Detection

- Time since last batch
- Last shard ID / file path being read
- Last sample index within that shard
- Worker process PIDs and their status (alive/dead)
- System memory usage (workers can OOM silently)
- Disk I/O stats if available

### Integration Pattern

Run the watchdog check in the training loop or in a background thread:

```python
watchdog = WorkerWatchdog(timeout_sec=120.0)

for batch in dataloader:
    watchdog.on_batch_received({"step": step, "shard": current_shard})
    # ... training step ...
```

Or as a background monitor:

```python
import threading

def watchdog_thread(watchdog, check_interval=10.0):
    while watchdog._running:
        try:
            watchdog.check()
        except DataLoaderStallError as e:
            logging.error(f"WATCHDOG ALERT: {e}")
            # Optionally kill and restart workers
        time.sleep(check_interval)
```
