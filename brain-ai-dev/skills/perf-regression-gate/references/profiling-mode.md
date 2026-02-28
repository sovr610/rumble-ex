# Profiling Mode Reference — Phase 5

## Overview

Phase 5 integrates PyTorch Profiler into the benchmark harness to provide detailed per-operator, per-region timing information. Profiling is optional (gated by `--profile trace|tb|off`) and must never contaminate throughput numbers. The profiling output enables identification of bottlenecks in the training loop: which ops are slow, where memory is allocated, and how distributed communication overlaps with compute.

**Critical rule: Never include profiling steps in the gating numbers.** Profile runs are separate from benchmark runs. The overhead from profiling (10-20%) would make timing comparisons meaningless.

---

## PyTorch Profiler Integration

### Basic Pattern

```python
import torch
from torch.profiler import profile, schedule, tensorboard_trace_handler, ProfilerActivity

def run_with_profiler(
    model,
    optimizer,
    data_iter,
    out_dir: str,
    wait: int = 1,
    warmup: int = 1,
    active: int = 3,
    repeat: int = 2,
    export_chrome: bool = True,
) -> None:
    """
    Run training loop under PyTorch Profiler.

    Total profiled steps = (wait + warmup + active) * repeat
    Steps captured in traces = active * repeat
    """
    total_steps = (wait + warmup + active) * repeat
    prof_schedule = schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    with profile(
        activities=activities,
        schedule=prof_schedule,
        on_trace_ready=tensorboard_trace_handler(out_dir),
        record_shapes=True,        # Tensor shape annotations in trace
        profile_memory=True,       # Memory allocation/deallocation tracking
        with_stack=False,          # Skip Python stack (reduces overhead)
        with_flops=True,           # FLOPs estimation per operator
    ) as prof:
        for step in range(total_steps):
            # Record named regions for trace readability
            with torch.autograd.profiler.record_function("data"):
                batch = next(data_iter)
                batch = {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}

            with torch.autograd.profiler.record_function("forward"):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(**batch)
                    logits = outputs.logits

            with torch.autograd.profiler.record_function("loss"):
                loss = outputs.loss

            with torch.autograd.profiler.record_function("backward"):
                loss.backward()

            with torch.autograd.profiler.record_function("optimizer_step"):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # CRITICAL: call prof.step() at the end of every step
            # This advances the profiler schedule state machine
            prof.step()

    if export_chrome:
        chrome_path = f"{out_dir}/trace.json"
        prof.export_chrome_trace(chrome_path)
        print(f"Chrome trace exported to: {chrome_path}")
```

### Schedule Mechanics

The profiler operates as a state machine with four phases per repeat cycle:

```
[wait steps] -> [warmup steps] -> [active steps] -> repeat
     |                |                |
   Skip            Discard         Record to trace
```

| Phase | What Happens |
|-------|-------------|
| `wait` | Profiler is idle; no overhead; skip initial transient behavior |
| `warmup` | Profiler starts tracing but discards output; JIT/caches prime |
| `active` | Profiler records traces to memory; `on_trace_ready` called at end |
| `repeat` | Entire cycle repeats N times; each repeat produces one trace file |

**Call `prof.step()` once at the end of every training step**, not at the start. The profiler uses `step()` to advance its internal state counter.

### Schedule Tuning

Choose schedule parameters based on what you want to capture:

| Scenario | Recommended Schedule |
|----------|---------------------|
| Quick sanity check | `wait=1, warmup=1, active=2, repeat=1` |
| Steady-state analysis | `wait=5, warmup=2, active=5, repeat=2` |
| First-step JIT investigation | `wait=0, warmup=0, active=3, repeat=1` |
| Long-run anomaly detection | `wait=10, warmup=2, active=10, repeat=3` |

Total steps required: `(wait + warmup + active) * repeat`. Run the training loop for exactly this many steps.

---

## `record_function` Regions

### Standard Regions for Training

Wrap every major section of the training step in a named `record_function` context. These names appear in TensorBoard and Chrome traces as labeled spans:

```python
# Region naming convention: short, lowercase, underscore-separated
STANDARD_REGIONS = [
    "data",           # Data loading + H2D transfer
    "forward",        # Forward pass (model.forward)
    "loss",           # Loss computation
    "backward",       # loss.backward()
    "optimizer_step", # optimizer.step() + zero_grad
    "grad_sync",      # DDP gradient allreduce (if separate from backward)
]
```

### Using `record_function` Correctly

```python
# Context manager form (recommended)
with torch.autograd.profiler.record_function("forward"):
    output = model(input_ids)

# Decorator form (for functions)
@torch.autograd.profiler.record_function("my_custom_kernel")
def my_function():
    ...
```

**Nesting is supported:**
```python
with torch.autograd.profiler.record_function("forward"):
    with torch.autograd.profiler.record_function("attention"):
        attn_output = self_attention(hidden_states)
    with torch.autograd.profiler.record_function("mlp"):
        mlp_output = feed_forward(hidden_states)
```

### DDP Gradient Sync Region

With PyTorch DDP, gradient synchronization (allreduce) happens automatically during `backward()`. To profile it separately:

```python
# Option A: use no_sync() for all but last micro-step
for i in range(grad_accum):
    with model.no_sync() if i < grad_accum - 1 else contextlib.nullcontext():
        with torch.autograd.profiler.record_function("backward"):
            loss.backward()

# The allreduce happens implicitly on the final backward; mark it:
with torch.autograd.profiler.record_function("grad_sync"):
    torch.distributed.barrier()  # Wait for allreduce to complete
```

### NVTX Compatibility

`record_function` automatically maps to NVTX ranges when using `nvtx` mode. This means the same code works with:
- PyTorch Profiler (TensorBoard/Chrome trace)
- NVIDIA Nsight Systems (`nsys profile --trace=nvtx`)
- NVIDIA Nsight Compute (`ncu --nvtx-include "forward"`)

No extra code is needed—`record_function` regions appear as NVTX ranges in nsys/ncu output.

---

## TensorBoard Viewing

### Setup

```bash
# Install TensorBoard with PyTorch plugin
pip install torch_tb_profiler tensorboard

# Launch TensorBoard pointing at profile directory
tensorboard --logdir profile/tb/ --port 6006
```

Open `http://localhost:6006` and navigate to the "PyTorch Profiler" tab.

### Directory Structure Generated by `tensorboard_trace_handler`

```
profile/tb/
  <hostname>_<timestamp>_pt_trace.json      # Trace file (cycle 1)
  <hostname>_<timestamp>_pt_trace.json      # Trace file (cycle 2, if repeat=2)
```

Each `on_trace_ready` call (once per `active` cycle × `repeat`) writes one trace file. TensorBoard loads all of them.

### TensorBoard Profiler Views

| View | What It Shows |
|------|--------------|
| Overview | Summary statistics, step time breakdown by category |
| Operator | Per-operator CPU/CUDA time, self-time vs total-time |
| GPU Kernel | Low-level CUDA kernel names, durations, occupancy |
| Memory | Memory allocation timeline, peak memory by operator |
| Trace | Interactive timeline (Gantt chart) of all events |

### Key Things to Look For

1. **GPU idle gaps**: Long CPU->GPU or backward->optimizer idle periods indicate pipeline bubbles
2. **Memory spikes**: Sudden allocation peaks that don't release suggest leaks or unexpected temporaries
3. **Slow operators**: Any operator taking > 10% of step time is worth investigating
4. **Communication overhead**: NCCL kernels in the trace; should overlap with backward compute

---

## Chrome Trace Export

### Exporting

```python
# After the profiler context exits:
prof.export_chrome_trace("profile/trace.json")
```

Or export during training (less common):
```python
# Inside on_trace_ready callback:
def custom_trace_handler(p):
    p.export_chrome_trace(f"profile/trace_step_{p.step_num}.json")
    # Also save to TensorBoard:
    torch.profiler.tensorboard_trace_handler("profile/tb")(p)

with profile(on_trace_ready=custom_trace_handler, ...) as prof:
    ...
```

### Viewing in Chrome

1. Open Chrome browser
2. Navigate to `chrome://tracing`
3. Click "Load" and select the `trace.json` file
4. Use keyboard shortcuts: W/A/S/D to zoom/pan

### Viewing in Perfetto

Perfetto is a more capable trace viewer for large traces:
1. Navigate to `https://ui.perfetto.dev`
2. Click "Open trace file" and upload `trace.json`

Perfetto handles files > 1GB that Chrome's `chrome://tracing` cannot load.

---

## Memory Profiling

### Enabling Memory Tracking

```python
with profile(
    profile_memory=True,   # Track tensor allocation/deallocation
    with_stack=True,       # Include Python stack for allocation sites
    ...
) as prof:
    ...
```

**Note:** `with_stack=True` adds significant overhead (2-5x). Only use for memory debugging, not for timing measurements.

### Reading Memory Results

```python
# After profiling, sort by memory impact:
print(prof.key_averages().table(
    sort_by="self_cuda_memory_usage",
    row_limit=20,
))
```

### Memory Timeline Export

```python
# Export memory timeline (requires torch >= 2.1)
prof.export_memory_timeline("profile/memory_timeline.html")
```

Open the HTML file in a browser for an interactive memory allocation chart.

---

## Distributed Profiling

### Each Rank Writes Its Own Trace

With `tensorboard_trace_handler`, each rank in a distributed run writes its own trace file. The directory will contain one trace per rank per cycle:

```
profile/tb/
  rank0_<timestamp>_pt_trace.json
  rank1_<timestamp>_pt_trace.json
  ...
  rank7_<timestamp>_pt_trace.json
```

TensorBoard loads all traces and lets you compare ranks side-by-side.

### Rank-Specific Output Directories

To avoid filename conflicts, use rank-specific subdirectories:

```python
import torch.distributed as dist

rank = dist.get_rank() if dist.is_initialized() else 0
rank_out_dir = f"{base_out_dir}/rank{rank}"

with profile(
    on_trace_ready=tensorboard_trace_handler(rank_out_dir),
    ...
) as prof:
    ...
```

### Identifying Communication Bottlenecks

In the trace view, look for:
- **NCCL** kernels on the CUDA stream: allreduce, broadcast, reducescatter
- **Bubble time**: periods where GPU is idle waiting for communication to complete
- **Rank imbalance**: if one rank's forward is slower, all ranks wait at the allreduce

A well-optimized distributed setup shows NCCL communication overlapping with backward compute (the gradient for earlier layers is reduced while the backward for later layers is still running).

---

## Profiling Overhead

### Magnitude

| Configuration | Overhead vs. Unprofiiled |
|--------------|--------------------------|
| `profile_memory=False, with_stack=False` | 10-15% |
| `profile_memory=True, with_stack=False` | 15-25% |
| `profile_memory=True, with_stack=True` | 50-200% |

### Why Profiling Must Not Contaminate Gate Numbers

Profiling inserts instrumentation into every CUDA operation:
- Memory allocation tracking hooks
- Event recording before/after each kernel
- Python-side overhead for record_function spans

This overhead is reproducible but meaningless for performance gating—you're measuring the profiler's overhead, not the model's efficiency.

**Hard rule:** Run `--bench` and `--profile` as separate invocations. Never combine them.

```bash
# CORRECT: separate runs
python -m tools.bench.run --bench --out artifacts/bench
python -m tools.bench.run --profile tb --out artifacts/profile

# WRONG: combined run that poisons bench numbers
python -m tools.bench.run --bench --profile tb --out artifacts
```

---

## Output Directory Structure

```
profile/
  tb/
    <hostname>_<timestamp>_pt_trace.json   # TensorBoard/Kineto trace
  chrome/
    trace.json                             # Chrome trace export
  memory/
    memory_timeline.html                   # Memory allocation timeline
```

### File Size Considerations

Chrome/Kineto traces grow with:
- Number of active steps (3-5 steps is typical)
- Number of operators (complex models have more)
- `record_shapes=True` adds shape info (2-3x larger)
- `with_stack=True` adds Python stacks (5-10x larger)

A typical 7B model trace with 3 active steps, no stack, with shapes: 50-500 MB.
With `with_stack=True`: 500 MB to 5 GB.

For large traces, use Perfetto instead of Chrome tracing.

---

## Implementation Checklist

- [ ] `prof.step()` called at end of every training step (not beginning)
- [ ] `record_function` wraps: data, forward, loss, backward, optimizer_step, grad_sync
- [ ] `on_trace_ready=tensorboard_trace_handler(out_dir)` used (not manual save)
- [ ] Chrome trace exported via `prof.export_chrome_trace()`
- [ ] `--profile` flag is separate from `--bench` (no combined runs)
- [ ] Distributed: rank-specific output directories used
- [ ] Schedule produces correct number of trace files: `repeat` files per directory
- [ ] `profile_memory=True` used for memory investigation (not by default)
- [ ] `with_stack` is False by default (too expensive for routine use)
- [ ] Documentation points users to TensorBoard and Perfetto viewers
