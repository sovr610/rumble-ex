# torch.compile Modes Reference

## Overview

`torch.compile` is the unified compilation API introduced in PyTorch 2.0. It wraps a model or function with TorchDynamo (Python bytecode tracing), AOTAutograd (ahead-of-time autograd graph capture), and a backend (default: TorchInductor) that generates optimized C++/Triton kernels. Compilation is lazy: it occurs on the first call that matches a new input signature.

---

## Mode: `default`

```python
model = torch.compile(model, mode="default")
```

**What it does:**
- Enables TorchInductor graph lowering and basic kernel fusion.
- Fuses pointwise ops (e.g., activation + bias add) into single kernels.
- Generates Triton kernels for GPU operations where profitable.
- Eliminates Python overhead for compiled regions.

**Trade-offs:**
- Fastest compilation time of the three modes (seconds, not minutes).
- Moderate steady-state speedup — typically 1.3x–2x on transformer workloads.
- No CUDA graph capture, so per-step Python overhead is still present at the graph boundary.
- Most permissive — tolerates more graph breaks than other modes.

**Best for:**
- First attempt at compilation. Profile here before moving to stricter modes.
- Workloads with variable shapes where CUDA graphs would cause excessive recompilation.
- Quick iteration during development.

**Typical options:**
```python
model = torch.compile(
    model,
    mode="default",
    options={
        "epilogue_fusion": True,   # fuse pointwise ops into reductions
        "shape_padding": True,     # pad shapes to multiples of 8 for GPU alignment
    }
)
```

---

## Mode: `reduce-overhead`

```python
model = torch.compile(model, mode="reduce-overhead")
```

**What it does:**
- Enables CUDA graph capture on top of default kernel fusion.
- CUDA graphs record the entire forward (and/or backward) pass as a single GPU command buffer.
- Replaying a CUDA graph replays the entire recorded command sequence with zero Python dispatcher overhead.
- Dramatically reduces per-step CPU-GPU synchronization overhead.

**Trade-offs:**
- Requires static shapes within the captured region. Shape changes trigger re-capture (expensive).
- Uses more GPU memory: CUDA graph buffers store all intermediate activations statically.
- More strict about in-place operations: in-place ops on graph inputs may corrupt future replays.
- CPU-GPU sync points (e.g., `.item()`, `.numpy()`, printing tensor values) inside the compiled region will break CUDA graph capture.

**Diagnosing CUDA graph applicability:**
```bash
TORCH_LOGS=perf_hints python train.py
```
This logs whether CUDA graphs are being applied and why they may not be.

**Best for:**
- Workloads with fixed shapes (e.g., fixed batch size and sequence length).
- Scenarios where Python overhead per step is measurably dominating GPU time.
- Inference serving with fixed request sizes.

**Constraints for CUDA graph capture:**
1. No dynamic control flow that depends on tensor values (data-dependent branching).
2. No CPU-GPU sync points inside the compiled region.
3. No in-place ops on external inputs (graph inputs must be stable memory addresses).
4. No custom Python objects crossing the compiled region boundary.

---

## Mode: `max-autotune`

```python
model = torch.compile(model, mode="max-autotune")
```

**What it does:**
- Runs autotuning benchmarks on all generated Triton kernels to select the fastest tile configuration.
- Enables CUDA graphs (same as `reduce-overhead`).
- Tries all viable kernel implementations and picks the best per-op per-shape.
- May also use cuBLAS GEMM autotuning for matrix multiplications.

**Trade-offs:**
- Compilation can take many minutes (autotuning runs each kernel variant on real hardware).
- Compilation is cached to disk by default (`~/.cache/torch/inductor/`). Subsequent runs with the same model/shapes are fast.
- Best steady-state throughput of all modes — frequently 1.5x–3x vs eager.
- All CUDA graph constraints from `reduce-overhead` apply.

**Variant: `max-autotune-no-cudagraphs`**
```python
model = torch.compile(model, mode="max-autotune-no-cudagraphs")
```
Runs autotuning without CUDA graph capture. Useful when:
- Variable shapes preclude CUDA graphs.
- CUDA graph capture causes memory or correctness issues.
- You want autotuned kernels but can't meet CUDA graph constraints.

**Cache control:**
```python
import torch._inductor.config as inductor_cfg
inductor_cfg.fx_graph_cache = True  # enable persistent FX graph cache (default in recent PyTorch)
```

**Best for:**
- Production training runs where compilation time amortizes over many GPU-hours.
- Workloads with fixed shapes and high throughput requirements.
- Final production deployment after validating with `default` mode.

---

## Backend Options

### `inductor` (default)

```python
model = torch.compile(model, backend="inductor")
```

The production backend. Generates C++/Triton code via TorchInductor's graph lowering pipeline. Supports all standard PyTorch ops, fusion, and CUDA graphs. Recommended for all production use.

### `aot_eager`

```python
model = torch.compile(model, backend="aot_eager")
```

Runs AOTAutograd (graph capture + autograd) but skips code generation entirely. Executes operations via ATen/eager. Used for:
- Debugging TorchDynamo graph captures without the codegen layer.
- Isolating whether a bug is in Dynamo (tracing) vs Inductor (codegen).
- Testing environments without a GPU.

Not a speedup backend — performance is approximately equal to or slower than pure eager.

### `cudagraphs`

```python
model = torch.compile(model, backend="cudagraphs")
```

Standalone CUDA graph capture without TorchInductor kernel fusion. Captures the computation graph and replays it as a CUDA graph. Less fusion than Inductor but simpler. Useful when Inductor's codegen causes issues but CUDA graph overhead reduction is still desired.

### Custom Backends

Custom backends are possible via the `torch._dynamo.register_backend` decorator. Not recommended unless you have specific codegen requirements (e.g., XLA, TensorRT integration). The Inductor backend with its option dict covers most needs.

---

## `fullgraph` Option

```python
model = torch.compile(model, fullgraph=True)
```

**What it does:**
- Raises `torch._dynamo.exc.Unsupported` instead of silently falling back to eager at graph breaks.
- Forces the entire model to compile as one graph or fail explicitly.

**Use cases:**
1. **Debugging**: Determine whether your model can be compiled as a single graph.
2. **Maximum optimization**: Ensure no graph breaks are silently degrading performance.
3. **Production validation**: Gate on fullgraph before committing to max-autotune.

**Not recommended as default** because:
- Many valid models have intentional graph breaks (e.g., data-dependent control flow in generation).
- Raises on the first graph break encountered, requiring iterative debugging.

**Workflow:**
```python
# Step 1: Find graph breaks
import os
os.environ["TORCH_LOGS"] = "graph_breaks"
model = torch.compile(model, fullgraph=True)
try:
    model(sample_input)
except torch._dynamo.exc.Unsupported as e:
    print(f"Graph break: {e}")
```

---

## `options` Dictionary

The `options` dict is passed to TorchInductor and controls fine-grained behavior:

```python
model = torch.compile(
    model,
    options={
        "epilogue_fusion": True,
        # Fuse pointwise ops (activation, bias add, etc.) into preceding reduction kernels.
        # Usually True is better. Default: True.

        "shape_padding": True,
        # Pad tensor dimensions to multiples of 8 (or other GPU-friendly sizes).
        # Improves memory alignment for tensor cores. Small memory overhead.
        # Default: False. Recommend True for transformer workloads.

        "fallback_random": False,
        # Use PyTorch's fallback (non-fused) random op implementations.
        # Set True if fused random ops cause numerical issues or graph breaks.
        # Default: False.

        "max_autotune_gemm": True,
        # Autotune GEMM operations specifically. Only relevant with max-autotune.

        "triton.cudagraphs": True,
        # Internal flag for CUDA graph enablement within Triton codegen.
        # Prefer using mode="reduce-overhead" rather than setting this directly.
    }
)
```

---

## Mode Selection Guide

```
Start here: mode="default"
    |
    v
Profile: is Python overhead dominating? (perf_hints logs show high CPU wait)
    |
    Yes → Are shapes fixed? (no variable seq_len, fixed batch size)
    |          |
    |          Yes → mode="reduce-overhead"
    |          No  → Use bucketing + mode="default"
    |
    No  → Are you spending many GPU-hours? (long production runs)
              |
              Yes → mode="max-autotune" (compilation amortizes)
                    If CUDA graph issues → "max-autotune-no-cudagraphs"
              No  → Stay on "default"
```

**Summary table:**

| Scenario | Recommended Mode |
|----------|-----------------|
| Development / fast iteration | `default` |
| Variable sequence lengths | `default` + bucketing |
| Fixed shapes, overhead-sensitive inference | `reduce-overhead` |
| Long production training, fixed shapes | `max-autotune` |
| Long production training, variable shapes | `max-autotune-no-cudagraphs` |
| Debugging graph breaks | `default` + `fullgraph=True` |
| Debugging codegen issues | `aot_eager` backend |

---

## CUDA Graph Constraints Reference

CUDA graphs record exact GPU operations and memory addresses. For replay to be correct:

| Constraint | Why |
|-----------|-----|
| Static shapes within captured region | Shape changes require re-recording |
| No `.item()` or tensor-to-Python coercions | Creates CPU-GPU sync point, breaks capture |
| No in-place ops on graph inputs | Corrupts the recorded memory addresses |
| No dynamic Python control flow based on tensor values | Can't be recorded in the static graph |
| No CPU-side operations interleaved with GPU ops | Forces synchronization |

**Testing CUDA graph applicability:**
```bash
TORCH_LOGS=perf_hints python -c "
import torch
model = torch.compile(your_model, mode='reduce-overhead')
model(sample_input)
"
```

Look for lines like:
- `[perf_hints] Capturing CUDA graph for module ...` — CUDA graph capture in progress.
- `[perf_hints] Unable to capture CUDA graph for ... due to ...` — Explains why capture failed.
