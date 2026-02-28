# torch.compile Debugging and Profiling Reference

## Graph Breaks

### What Is a Graph Break?

A graph break occurs when TorchDynamo cannot capture a contiguous Python call graph through a function call or operation. When Dynamo encounters a graph break, it:
1. Emits the compiled graph up to the break point.
2. Falls back to executing the break-point code in Python (eager mode).
3. Starts a new compiled graph after the break point.

Multiple graph breaks fragment your model into many small compiled segments with Python execution between them. This reduces optimization opportunities (cross-op fusion can't span break points) and reintroduces Python overhead.

### Common Causes of Graph Breaks

| Cause | Example | Fix |
|-------|---------|-----|
| Data-dependent control flow | `if x.sum() > 0: ...` | Use masked ops instead |
| Calling `print()` on tensors | `print(f"loss: {loss.item()}")` | Move logging outside compiled region |
| Custom Python objects with `__getattr__` | Custom config classes in forward | Use simple dicts or dataclasses |
| Unsupported PyTorch ops | Some custom ops, older ATen ops | File issue or use `torch.compiler.disable` |
| Tensor-to-Python coercions | `for i in range(tensor.item()):` | Use tensor operations |
| NumPy interop | `x.numpy()` | Avoid inside compiled code |
| Pickling/unpickling | Lazy loading inside forward | Preload outside |
| Non-standard `__torch_function__` | Custom tensor subclasses | May need special handling |

### Finding Graph Breaks

**Method 1: Environment variable (recommended)**

```bash
TORCH_LOGS=graph_breaks python train.py
```

Output example:
```
[graph_break] Graph break in user_forward due to: print() call on tensor
  File "model.py", line 142, in forward
    print(f"attention weights shape: {attn_weights.shape}")
```

**Method 2: `fullgraph=True` (raises immediately)**

```python
model = torch.compile(model, fullgraph=True)
try:
    model(sample_input)
except torch._dynamo.exc.Unsupported as e:
    print(f"Graph break at: {e}")
```

Useful for iterative graph break debugging: fix one, re-run, find next.

**Method 3: `torch._dynamo.explain`**

```python
explanation = torch._dynamo.explain(model)(sample_input)
print(explanation.graphs)          # list of captured graph segments
print(explanation.graph_break_reasons)  # why each break occurred
print(explanation.break_reasons)   # human-readable
```

Returns a structured object with full details about all graph breaks.

---

## `torch.compiler.disable` Decorator

Use to explicitly exclude functions from compilation. The compiled graph will call the decorated function in eager mode.

```python
import torch

@torch.compiler.disable
def logging_fn(tensor):
    """This function won't be compiled — it's called from a compiled parent."""
    print(f"Debug value: {tensor.mean().item()}")
    return tensor

@torch.compiler.disable(recursive=False)
def preprocessing_fn(x):
    """Disables compilation for this function only.
    Callees (functions called by this function) may still be compiled."""
    return x * 2 + 1
```

**`recursive=False`**: Only disables compilation of the decorated function itself. Functions it calls are not affected — they may still be compiled if called from elsewhere. Use to disable just the problem function while keeping its callees in the compiled graph.

**`recursive=True` (default)**: Disables compilation of the function and all callees recursively. Use to quarantine an entire subtree.

**When to use:**
- Known problem functions that reliably cause graph breaks.
- Debugging functions (`print`, logging) that should never be in the graph.
- Custom ops or preprocessing that aren't worth optimizing.
- Text generation sampling loops (highly data-dependent, branchy).

---

## TORCH_LOGS Environment Variable

Set via environment variable or programmatically:

```bash
# Environment variable (recommended for training scripts)
TORCH_LOGS=flag1,flag2 python train.py

# Programmatic (useful for notebooks)
import logging
torch._logging.set_logs(graph_breaks=True, recompiles=True)
```

### Available Flags

| Flag | What it shows | When to use |
|------|---------------|-------------|
| `graph_breaks` | Location and reason for each graph break | Diagnosing graph break sources |
| `recompiles` | When and why recompilation triggers | Diagnosing shape instability |
| `dynamic` | Guard installation and symbolic shape analysis | Understanding dynamic shape behavior |
| `perf_hints` | CUDA graph applicability diagnosis | Diagnosing reduce-overhead issues |
| `output_code` | Generated Triton/C++ kernel source code | Understanding what code was generated |
| `schedule` | TorchInductor scheduling decisions | Debugging fusion decisions |
| `fusion` | Fusion decisions in the IR lowering | Understanding kernel fusion |
| `aot_graphs` | AOTAutograd captured graphs | Debugging autograd capture issues |
| `compiled_autograd` | Compiled autograd graph | Debugging backward compilation |
| `guards` | All guards installed by Dynamo | Verbose debugging of shape guards |
| `bytecode` | Modified Python bytecode | Low-level Dynamo tracing debug |

### Common Combinations

```bash
# Most useful for training iteration
TORCH_LOGS=graph_breaks,recompiles python train.py

# Diagnosing CUDA graph issues with reduce-overhead
TORCH_LOGS=perf_hints python train.py

# Understanding dynamic shape behavior
TORCH_LOGS=dynamic,recompiles python train.py

# Full debug (very verbose)
TORCH_LOGS=graph_breaks,recompiles,dynamic,guards,aot_graphs python train.py
```

---

## `suppress_errors`

```python
import torch._dynamo
torch._dynamo.config.suppress_errors = True
```

When True, compilation errors fall back to eager silently (with a warning log) rather than raising. Useful during development to keep training going even when compilation fails, but:

- **Never use in production.** Silently falling back means you don't know your model isn't being compiled.
- Even in development, prefer adding `@torch.compiler.disable` annotations to known problem functions rather than suppressing all errors.
- May hide bugs that would otherwise be caught by the compilation error.

Use `fail_policy="fallback_eager"` in `CompileConfig` (the `maybe_compile` wrapper) instead, which provides the same safety but with explicit logging.

---

## Compile Time Measurement

The first forward pass after `torch.compile` triggers JIT compilation. This makes the first step much slower than subsequent steps. Benchmarking must account for this.

### Measuring Compilation Time

```python
import time
import torch

model = torch.compile(model, mode="default")

# First step: includes compilation time
torch.cuda.synchronize()
t0 = time.perf_counter()
output = model(sample_input)
loss = loss_fn(output)
loss.backward()
torch.cuda.synchronize()
compile_time = time.perf_counter() - t0
print(f"First step (compile + run): {compile_time:.3f}s")

# Subsequent steps: compiled execution only
for i in range(warmup_steps):
    output = model(sample_input)
    loss = loss_fn(output)
    loss.backward()

# Benchmark after warmup
torch.cuda.synchronize()
t0 = time.perf_counter()
for i in range(benchmark_steps):
    output = model(sample_input)
    loss = loss_fn(output)
    loss.backward()
torch.cuda.synchronize()
steady_state_time = (time.perf_counter() - t0) / benchmark_steps
print(f"Steady-state step time: {steady_state_time:.3f}s")
```

### Compilation Cache

TorchInductor caches compiled kernels to disk. Location: `~/.cache/torch/inductor/` (Linux/Mac) or `%APPDATA%\torch\inductor\` (Windows).

```python
# Control cache location
import torch._inductor.config
torch._inductor.config.cache_dir = "/fast/ssd/torch_compile_cache"

# Disable cache for reproducible benchmarks
torch._inductor.config.fx_graph_cache = False
```

Second run with same model+shapes skips recompilation. For benchmarking, either use cache (to measure steady-state) or clear cache (to measure fresh compilation).

---

## Memory Profiling Under Compile

Compiled models may use more GPU memory than eager models, primarily from:
- **CUDA graph buffers**: Store all intermediate activations as static GPU memory.
- **Codegen overhead**: Some generated kernels allocate workspace buffers.
- **Inductor cache**: In-memory caches for kernel selection.

### Comparing Memory Usage

```python
import torch

def measure_peak_memory(model, input_batch, compiled=False):
    if compiled:
        model = torch.compile(model)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # Warmup
    for _ in range(3):
        out = model(**input_batch)
        loss = out.sum()
        loss.backward()

    torch.cuda.reset_peak_memory_stats()

    out = model(**input_batch)
    loss = out.sum()
    loss.backward()

    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    return peak_mb

eager_mem = measure_peak_memory(model, batch, compiled=False)
compiled_mem = measure_peak_memory(model, batch, compiled=True)
print(f"Eager peak: {eager_mem:.1f} MB, Compiled peak: {compiled_mem:.1f} MB")
print(f"Memory delta: {compiled_mem - eager_mem:+.1f} MB")
```

---

## Performance Profiling with PyTorch Profiler

Use `torch.profiler` to understand where time is being spent in compiled vs eager execution:

```python
from torch.profiler import profile, ProfilerActivity, record_function

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True
) as prof:
    with record_function("model_forward"):
        output = model(**batch)
    with record_function("loss_backward"):
        loss = output.sum()
        loss.backward()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
prof.export_chrome_trace("compile_trace.json")
```

Compare the trace between eager and compiled modes. In compiled mode, you should see:
- Fewer, larger CUDA kernels (fusion is working).
- Less CPU time between GPU launches (Python overhead reduction).
- Possible `triton_kernel` entries instead of ATen ops.

---

## Diagnosing Generated Triton Code

To see the actual generated Triton kernels:

```bash
TORCH_LOGS=output_code python train.py 2>&1 | head -200
```

Or to save to a file:
```bash
TORCH_LOGS=output_code python train.py 2>/tmp/triton_code.py
```

This is rarely needed but useful when:
- A specific kernel is slower than expected.
- You suspect a fusion isn't happening.
- Contributing to TorchInductor bug reports.

---

## Quick Debugging Checklist

```
1. Is compilation actually happening?
   → TORCH_LOGS=recompiles — look for "Compiling" log line on first step

2. How many graph breaks are there?
   → TORCH_LOGS=graph_breaks or use torch._dynamo.explain()

3. Are shapes stable?
   → TORCH_LOGS=recompiles,dynamic — watch for recompile logs during training

4. Is reduce-overhead mode using CUDA graphs?
   → TORCH_LOGS=perf_hints — look for "Capturing CUDA graph"

5. Is memory usage acceptable?
   → torch.cuda.max_memory_allocated() before and after compile

6. What's the actual speedup?
   → Benchmark: warmup 50 steps, measure 100 steps, compare with eager

7. Are there correctness issues?
   → Run same input through eager and compiled, compare outputs with torch.allclose()
```
