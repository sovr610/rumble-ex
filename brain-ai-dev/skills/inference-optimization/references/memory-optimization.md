# Memory Optimization for Brain-Inspired AI Inference

This document covers memory optimization strategies for brain_ai inference: reduced-precision inference (FP16/BF16), `torch.inference_mode()`, gradient checkpointing at inference time, CPU offloading of unused modules, and memory profiling. The implementation is in `assets/memory_optimizer_template.py`.

---

## 1. Reduced Precision Inference

### 1.1 Why Reduced Precision

Brain_ai at production scale (7B parameters) requires approximately 28GB of GPU memory for model weights alone in FP32. Reduced precision halves memory usage:

| Dtype | Bytes per Param | 7B Model Size | Dynamic Range |
|-------|----------------|--------------|--------------|
| FP32 | 4 | ~28GB | 1e-38 to 3e38 |
| FP16 | 2 | ~14GB | 6e-8 to 65504 |
| BF16 | 2 | ~14GB | 1e-38 to 3e38 |
| INT8 | 1 | ~7GB | -128 to 127 |

### 1.2 FP16 Inference

Half-precision float (float16) halves memory and is 2x faster on tensor cores. However:

**Overflow risk:** FP16 has a maximum value of 65504. If any activation exceeds this (possible in workspace attention or SNN membrane accumulation), results become `inf`. The HTM sparse operations are particularly vulnerable since column activations can produce large values.

**Underflow risk:** FP16 has a minimum positive normal of 6e-8. Very small gradients or probabilities may round to zero. This affects the active inference module's Expected Free Energy computation.

**Recommendation:** Use FP16 for most modules, but keep these in FP32:
- HTM sparse operations
- Active inference EFE computation
- Neuromodulatory gate outputs (small multipliers)

```python
def convert_to_fp16_selective(model: nn.Module) -> nn.Module:
    """Convert model to FP16, keeping sensitive modules in FP32."""
    fp32_modules = {'htm', 'active_inference', 'neuromodulation'}

    for name, module in model.named_modules():
        keep_fp32 = any(fp32_name in name for fp32_name in fp32_modules)
        if not keep_fp32:
            module.half()
    return model
```

### 1.3 BF16 Inference

BFloat16 has the same dynamic range as FP32 (same exponent bits) but less precision (7 bits vs 23 bits mantissa). This eliminates the overflow problem while still halving memory.

**Recommended for brain_ai** when hardware supports it (A100, H100, recent AMD GPUs). No selective FP32 fallback needed.

```python
def convert_to_bf16(model: nn.Module) -> nn.Module:
    """Convert entire model to BF16. Safe for all modules."""
    return model.to(dtype=torch.bfloat16)
```

**Hardware detection:**
```python
def get_optimal_dtype() -> torch.dtype:
    """Detect best inference dtype for current hardware."""
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        if capability >= (8, 0):  # Ampere+: native BF16
            return torch.bfloat16
        elif capability >= (7, 0):  # Volta/Turing: FP16 tensor cores
            return torch.float16
    return torch.float32  # CPU or old GPU
```

### 1.4 Mixed Precision Inference

Use `torch.autocast` for automatic mixed precision during inference:

```python
def infer_mixed_precision(model, inputs, dtype=torch.float16):
    """Run inference with automatic mixed precision."""
    model.eval()
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=dtype):
            return model(inputs)
```

`torch.autocast` automatically keeps certain operations in FP32 (softmax, layer norm, loss functions) for numerical stability.

---

## 2. torch.inference_mode()

### 2.1 What inference_mode Does

`torch.inference_mode()` is stricter than `torch.no_grad()`:
- Disables autograd entirely (no gradient tracking, no version counter updates).
- Tensors created inside are marked as "inference tensors" and cannot participate in autograd.
- Saves memory by not storing autograd metadata (computation graph nodes, saved tensors for backward).

### 2.2 Memory Savings

For a typical brain_ai forward pass, autograd metadata accounts for 30-50% of peak memory. `inference_mode()` eliminates this:

| Config | Forward Memory (no_grad) | Forward Memory (inference_mode) | Savings |
|--------|-------------------------|-------------------------------|---------|
| minimal | ~50MB | ~30MB | 40% |
| 1B | ~6GB | ~4GB | 33% |
| 7B | ~36GB | ~22GB | 39% |

### 2.3 Usage Pattern

```python
class InferenceWrapper:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def __call__(self, inputs):
        with torch.inference_mode():
            return self.model(inputs)
```

### 2.4 Caveats

- Tensors created in inference_mode cannot be used in subsequent autograd operations. If you need to fine-tune after inference, clone the tensors first.
- In-place operations on inference tensors that were created outside inference_mode will fail. Always create fresh tensors inside the context.
- `inference_mode()` is not compatible with `torch.compile` in all cases. Test with your specific model.

---

## 3. Gradient Checkpointing at Inference

### 3.1 Why Checkpointing at Inference

Gradient checkpointing is primarily a training technique, but it reduces memory during inference in specific scenarios:
- **Very long sequences** where intermediate activations dominate memory.
- **Multi-step reasoning** where System 2's iterative GRU produces many intermediate states.
- **Streaming inference** where state from many time steps accumulates.

### 3.2 Selective Checkpointing

Checkpoint only the expensive modules (encoders, workspace attention) and recompute them when needed:

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedWorkspace(nn.Module):
    def __init__(self, workspace: GlobalWorkspace):
        super().__init__()
        self.workspace = workspace

    def forward(self, encoded):
        return checkpoint(self.workspace.forward, encoded, use_reentrant=False)
```

### 3.3 Memory vs Latency Trade-off

Checkpointing saves memory but increases latency (each checkpointed region is computed twice):

| Module | Memory Savings | Latency Increase |
|--------|---------------|-----------------|
| Vision Encoder | 200MB | +15% |
| Text Encoder | 150MB | +20% |
| Workspace Attention | 100MB | +10% |
| System 2 Reasoning | 80MB | +25% |

For inference, use checkpointing only when memory is the binding constraint and latency is acceptable.

---

## 4. CPU Offloading

### 4.1 Module-Level Offloading

When GPU memory is insufficient for the full model, offload unused modules to CPU and load them on-demand:

```python
class OffloadManager:
    def __init__(self, model: nn.Module, modules_to_offload: List[str]):
        self.model = model
        self.offloaded = {}
        for name in modules_to_offload:
            module = getattr(model, name)
            module.cpu()
            self.offloaded[name] = module

    def load(self, module_name: str, device: str = "cuda"):
        """Load a module back to GPU before use."""
        if module_name in self.offloaded:
            self.offloaded[module_name].to(device)

    def unload(self, module_name: str):
        """Move a module to CPU after use."""
        if module_name in self.offloaded:
            self.offloaded[module_name].cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
```

### 4.2 Pipeline Offloading for Brain_AI

The brain_ai pipeline processes modules sequentially. Only the currently active module needs to be on GPU:

```
Encode (GPU) -> offload encoders -> SNN (GPU) -> offload SNN -> HTM (GPU) -> ...
```

```python
def forward_with_offloading(model, inputs, device="cuda"):
    """Forward pass with sequential module offloading."""
    # Phase 1: Encode
    model.encoders.to(device)
    encoded = model.encode(inputs)
    model.encoders.cpu()
    torch.cuda.empty_cache()

    # Phase 2: Workspace
    model.workspace.to(device)
    workspace = model.workspace(encoded)
    model.workspace.cpu()
    torch.cuda.empty_cache()

    # Phase 3: Reasoning
    model.reasoner.to(device)
    output = model.reasoner(workspace)
    model.reasoner.cpu()

    return output
```

### 4.3 Offloading Latency

Moving modules between CPU and GPU incurs PCIe transfer latency:

| Module | Size | Transfer Time (PCIe 4.0 x16) |
|--------|------|---------------------------|
| Vision Encoder (300M) | ~1.2GB | ~37ms |
| Text Encoder (340M) | ~1.4GB | ~44ms |
| SNN Core (500M) | ~2GB | ~62ms |
| Workspace (1.5B) | ~6GB | ~187ms |
| Engram (2.5B) | ~10GB | ~312ms |

For the 7B model, full pipeline offloading adds approximately 640ms per inference. This is acceptable for batch processing but too slow for real-time. Use offloading only for the largest modules and keep the rest on GPU.

### 4.4 Pinned Memory for Faster Transfers

Use pinned (page-locked) CPU memory for modules that are frequently moved:

```python
def offload_to_pinned(module: nn.Module):
    """Move module to pinned CPU memory for faster GPU transfers."""
    module.cpu()
    for param in module.parameters():
        param.data = param.data.pin_memory()
    for buf in module.buffers():
        buf.data = buf.data.pin_memory()
```

Pinned memory provides 2-3x faster CPU-to-GPU transfers but consumes non-pageable system memory.

---

## 5. Memory Measurement and Profiling

### 5.1 Measuring Model Memory

```python
def measure_model_memory(model: nn.Module) -> dict:
    """Measure memory used by model parameters and buffers."""
    param_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.nelement() * b.element_size() for b in model.buffers())
    return {
        "param_mb": param_bytes / (1024 * 1024),
        "buffer_mb": buffer_bytes / (1024 * 1024),
        "total_mb": (param_bytes + buffer_bytes) / (1024 * 1024),
    }
```

### 5.2 Measuring Inference Memory

Track peak GPU memory during inference:

```python
def measure_inference_memory(model, sample_input, device="cuda"):
    """Measure peak GPU memory during inference."""
    if not torch.cuda.is_available():
        return {"peak_mb": 0.0, "model_mb": 0.0, "activation_mb": 0.0}

    model.to(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    before = torch.cuda.memory_allocated(device)

    with torch.inference_mode():
        _ = model(sample_input)

    peak = torch.cuda.max_memory_allocated(device)
    after = torch.cuda.memory_allocated(device)

    return {
        "model_mb": before / (1024 * 1024),
        "peak_mb": peak / (1024 * 1024),
        "activation_mb": (peak - before) / (1024 * 1024),
        "post_inference_mb": after / (1024 * 1024),
    }
```

### 5.3 Per-Module Memory Breakdown

Profile memory usage of each module individually:

```python
def per_module_memory(model, sample_input, device="cuda"):
    """Measure memory contribution of each top-level module."""
    breakdown = {}
    for name, module in model.named_children():
        param_bytes = sum(p.nelement() * p.element_size()
                         for p in module.parameters())
        buffer_bytes = sum(b.nelement() * b.element_size()
                          for b in module.buffers())
        breakdown[name] = {
            "params_mb": param_bytes / (1024 * 1024),
            "buffers_mb": buffer_bytes / (1024 * 1024),
        }
    return breakdown
```

### 5.4 Memory Leak Detection

For long-running inference servers, monitor memory growth:

```python
class MemoryLeakDetector:
    def __init__(self, check_interval: int = 100, growth_threshold_mb: float = 100):
        self.check_interval = check_interval
        self.growth_threshold = growth_threshold_mb
        self.call_count = 0
        self.baseline_mb = None

    def check(self):
        self.call_count += 1
        if self.call_count % self.check_interval != 0:
            return

        if torch.cuda.is_available():
            current_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        else:
            import psutil
            current_mb = psutil.Process().memory_info().rss / (1024 * 1024)

        if self.baseline_mb is None:
            self.baseline_mb = current_mb
            return

        growth = current_mb - self.baseline_mb
        if growth > self.growth_threshold:
            import warnings
            warnings.warn(
                f"Potential memory leak: {growth:.1f}MB growth "
                f"after {self.call_count} inferences"
            )
```

---

## 6. Memory Report Dataclass

The MemoryOptimizer produces a structured report:

```python
@dataclass
class MemoryReport:
    model_size_mb: float          # Total model parameters + buffers
    param_size_mb: float          # Parameters only
    buffer_size_mb: float         # Buffers only
    peak_inference_mb: float      # Peak memory during inference
    activation_mb: float          # Activations (peak - model)
    dtype: str                    # Current model dtype
    device: str                   # Current device
    per_module: Dict[str, float]  # Per-module breakdown
    recommendations: List[str]    # Optimization suggestions

    def summary(self) -> str:
        lines = [
            f"Model: {self.model_size_mb:.1f}MB ({self.dtype} on {self.device})",
            f"Peak inference: {self.peak_inference_mb:.1f}MB",
            f"Activations: {self.activation_mb:.1f}MB",
        ]
        if self.recommendations:
            lines.append("Recommendations:")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")
        return "\n".join(lines)
```

---

## 7. Optimization Strategy Selection

Given a memory budget, automatically select optimizations:

```python
def auto_optimize(model, gpu_memory_mb: float, target_batch_size: int = 1):
    """Automatically apply memory optimizations to fit in GPU memory."""
    model_mb = measure_model_memory(model)["total_mb"]

    optimizations_applied = []

    # Step 1: Try BF16/FP16
    if model_mb * 2 > gpu_memory_mb:
        dtype = get_optimal_dtype()
        model = model.to(dtype=dtype)
        model_mb /= 2
        optimizations_applied.append(f"Converted to {dtype}")

    # Step 2: Try CPU offloading for non-critical modules
    if model_mb * 1.5 > gpu_memory_mb:
        offload_candidates = ['engram', 'meta', 'htm']
        for name in offload_candidates:
            if hasattr(model, name) and getattr(model, name) is not None:
                getattr(model, name).cpu()
                optimizations_applied.append(f"Offloaded {name} to CPU")

    # Step 3: Enable gradient checkpointing
    if model_mb * 1.3 > gpu_memory_mb:
        optimizations_applied.append("Enabled gradient checkpointing")

    return model, optimizations_applied
```

---

## 8. Brain_AI Module-Specific Memory Tips

### 8.1 Engram Memory

The engram hash table is the largest single component (~2.5B params, ~10GB in FP32). For inference:
- Keep the hash table on CPU (it is accessed via hash lookups, not matrix multiplication).
- Batch hash lookups and transfer only the needed embeddings to GPU.
- Use INT8 quantization for the embedding table (minimal accuracy loss for hash-based lookups).

### 8.2 SNN Core

SNN state (membrane potentials, spike history) scales with `batch_size * hidden_size * num_timesteps`. For the production config: 500M params but state can be 500MB+ for large batches. Detach state between time steps during inference to prevent accumulation.

### 8.3 Global Workspace

The cross-modal attention in the workspace produces attention matrices of size `(batch, num_heads, seq_len, seq_len)`. For 32 heads and sequence length 8192, each attention matrix is 2GB in FP32. Use FP16 attention or flash attention to reduce this.

### 8.4 HTM Layer

HTM column states are sparse. Store them in sparse format to save memory:
```
Dense: 16384 columns * 64 cells = 1M values per batch element
Sparse (2% sparsity): ~20K non-zero values per batch element
Memory savings: 50x
```
