# DataLoader Tuning

Systematic guide to tuning `torch.utils.data.DataLoader` parameters for maximum throughput.

## Key Parameters

### num_workers

**What it does:** Number of subprocess workers that load data in parallel.

**Default:** 0 (main process loads data, blocking the training loop).

**Guidance:**
- Start at 4 per GPU.
- Increase by 2 until throughput stops improving.
- More is not always better: too many workers cause context switching overhead, memory pressure, and diminishing returns.
- Rule of thumb: `num_workers = min(num_cpu_cores // num_gpus, 16)`.
- Monitor: if increasing from 8 to 12 gives < 5% throughput gain, stop at 8.

**Failure modes:**
- Too many workers: CPU contention, increased memory usage, slower startup.
- Too few workers: GPU starves waiting for data (high data_stall_ratio).
- num_workers=0: All data loading on main process. Only acceptable for debugging.

### prefetch_factor

**What it does:** Number of batches prefetched per worker ahead of time.

**Default:** 2 (each worker has 2 batches ready).

**Guidance:**
- Range: 2-8.
- Higher = more memory used for prefetch buffers, but less latency when GPU finishes a step.
- Start at 2, increase to 4 if data_stall_ratio > 0.1 and num_workers is already tuned.
- Very high values (>8) rarely help and waste memory.

**Note:** Only applicable when `num_workers > 0`.

### persistent_workers

**What it does:** Keeps worker processes alive between epochs instead of respawning.

**Default:** False.

**Guidance:**
- Set `True` if training runs multiple epochs on the same dataset.
- Worker respawn cost: each worker must re-initialize its state (open files, rebuild indices, etc.).
- For single-epoch pretraining runs, this matters less.
- Always set True for fine-tuning (typically 3-5 epochs).

**Caveat:** Workers retain memory from previous epochs. If dataset changes between epochs, workers may hold stale references.

### pin_memory

**What it does:** Allocates host tensors in page-locked (pinned) memory, enabling faster DMA transfers to GPU.

**Default:** False.

**Guidance:**
- Set `True` for all GPU training. There is almost no reason to leave this False.
- Slightly increases host memory usage (pinned memory is not pageable).
- Enables `non_blocking=True` in `.to(device)` for overlapped transfers.
- Without pin_memory, transfers go through a staging buffer, adding ~10-30% transfer time.

## Tuning Grid Approach

Systematically search a small parameter grid:

```python
tuning_grid = {
    "num_workers": [2, 4, 8, 12],
    "prefetch_factor": [2, 4, 8],
}

# Fixed: persistent_workers=True, pin_memory=True
# These are almost always beneficial.

best_throughput = 0
best_config = None

for nw in tuning_grid["num_workers"]:
    for pf in tuning_grid["prefetch_factor"]:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=nw,
            prefetch_factor=pf,
            persistent_workers=True,
            pin_memory=True,
        )
        throughput = measure_throughput(loader, num_steps=50)
        if throughput > best_throughput:
            best_throughput = throughput
            best_config = {"num_workers": nw, "prefetch_factor": pf}

print(f"Best config: {best_config} at {best_throughput:.0f} tokens/sec")
```

### Measurement Protocol

1. Warm up for 10 steps (ignore these).
2. Measure for 50 steps.
3. Record median throughput (not mean, to ignore outliers).
4. Record p90 data_stall_ratio.
5. Repeat each config 2-3 times for stability.

## DistributedSampler: set_epoch

When using `DistributedSampler`, you **must** call `sampler.set_epoch(epoch)` at the start of each epoch:

```python
sampler = DistributedSampler(dataset, shuffle=True, seed=42, drop_last=True)
loader = DataLoader(dataset, sampler=sampler, ...)

for epoch in range(num_epochs):
    sampler.set_epoch(epoch)  # CRITICAL: without this, same order every epoch
    for batch in loader:
        train_step(batch)
```

**Why:** `DistributedSampler` uses `seed + epoch` to determine shuffle order. Without `set_epoch`, the epoch stays at 0, producing identical ordering every epoch. This reduces effective data diversity.

**Common bug:** Forgetting `set_epoch` causes mysteriously low validation performance because the model sees data in the same order every epoch.

## Worker Watchdog

### Purpose

DataLoader workers can die silently (segfault, OOM) or stall indefinitely (deadlock, slow I/O). The watchdog detects these conditions and fails loudly with diagnostic information.

### Implementation

```python
import time
import threading
import logging

class WorkerWatchdog:
    def __init__(self, timeout_sec: float = 120.0):
        self.timeout_sec = timeout_sec
        self.last_batch_time = time.monotonic()
        self.last_sample_info = {}
        self._lock = threading.Lock()

    def on_batch_received(self, info: dict):
        with self._lock:
            self.last_batch_time = time.monotonic()
            self.last_sample_info = info

    def check(self) -> bool:
        with self._lock:
            elapsed = time.monotonic() - self.last_batch_time
            if elapsed > self.timeout_sec:
                logging.error(
                    f"DataLoader stall detected! No batch for {elapsed:.1f}s. "
                    f"Last sample: {self.last_sample_info}"
                )
                return False
        return True
```

### Integration

```python
watchdog = WorkerWatchdog(timeout_sec=120)

for step, batch in enumerate(loader):
    watchdog.on_batch_received({
        "step": step,
        "batch_size": len(batch["input_ids"]),
    })
    # ... training ...
```

### What to Log on Stall

- Seconds since last batch
- Last sample/shard ID
- Worker PIDs and alive status
- System memory (workers may OOM)
- GPU utilization (should be 0% during stall)

## Effective Batch Availability

**Definition:** The average time the training loop waits for the next batch to become available.

```python
wait_times = []
for batch in loader:
    t_start = time.perf_counter()
    # ... process batch ...
    t_end = time.perf_counter()
    # Time between finishing processing and getting next batch
    # (measured at next iteration)
```

In practice, this is captured by `t_data` in the PipelineAuditor. If `t_data` is consistently low (< 5% of step time), the pipeline is keeping up. If it spikes, investigate:

1. **Steady high t_data:** Not enough workers or prefetch depth.
2. **Periodic spikes:** Shard boundaries, epoch transitions, or GC pauses.
3. **Increasing over time:** Memory leak in workers, cache eviction pressure.

## Complete Tuning Checklist

1. Set `pin_memory=True` (always).
2. Set `persistent_workers=True` (if multi-epoch).
3. Set `num_workers=4`, measure baseline data_stall_ratio.
4. If stall_ratio > 0.10, increase num_workers to 8.
5. If still > 0.10, increase prefetch_factor to 4.
6. If still > 0.10, check I/O backend (switch to streaming/memmap).
7. If still > 0.10, check storage speed (NVMe vs network mount).
8. Once stall_ratio < 0.05, stop tuning (diminishing returns).
9. Always call `sampler.set_epoch(epoch)`.
10. Deploy watchdog in production training.
