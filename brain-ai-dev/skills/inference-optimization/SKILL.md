---
name: Inference Optimization
description: >
  This skill should be used when the user asks to "optimize inference",
  "speed up predictions", "add caching", "batch inference", "async inference",
  "reduce inference latency", "profile inference", "memory efficient inference",
  "inference pipeline", "warm up model", "KV cache", "inference throughput",
  "latency benchmark", "optimize forward pass", or needs guidance on
  inference performance optimization, caching strategies, batch processing,
  async execution, or latency profiling for the brain_ai system.
version: 0.1.0
---

# Inference Optimization

## Overview

Guide implementation of inference optimization infrastructure for the brain_ai system. The multi-layer cognitive pipeline has unique optimization opportunities: SNN state caching across time steps, workspace competition early-exit, System 1 fast-path bypassing System 2, engram memory hash caching, and selective module activation via feature flags. Cover batch inference, caching, async execution, memory optimization, latency profiling, and inference pipelines.

## Public Contract

### BatchInferenceEngine

Efficient batch processing with dynamic batching and padding.

```python
class BatchInferenceEngine:
    def __init__(self, model: BrainAI, config: InferenceOptConfig): ...
    def infer_batch(self, inputs: List[Dict[str, Tensor]]) -> List[InferenceResult]: ...
    def infer_stream(self, input_stream: Iterator) -> Iterator[InferenceResult]: ...
    def warmup(self, sample_input: Dict[str, Tensor], n_warmup: int = 10) -> None: ...
```

### CacheManager

Multi-level caching for intermediate representations.

```python
class CacheManager:
    def __init__(self, config: CacheConfig): ...
    def get(self, key: str, level: str = "l1") -> Optional[Tensor]: ...
    def put(self, key: str, value: Tensor, level: str = "l1") -> None: ...
    def invalidate(self, key: Optional[str] = None) -> None: ...
    def stats(self) -> CacheStats: ...
```

Cache levels:
- **L1**: In-GPU tensor cache (encoder outputs, workspace states) — fastest, limited by GPU memory
- **L2**: CPU pinned memory cache (engram embeddings) — medium speed, larger capacity
- **L3**: Disk-backed mmap cache (HTM column states) — slowest, unlimited

### AsyncInferenceEngine

Non-blocking inference with futures and callbacks.

```python
class AsyncInferenceEngine:
    def __init__(self, model: BrainAI, config: InferenceOptConfig): ...
    async def infer(self, input: Dict[str, Tensor]) -> InferenceResult: ...
    async def infer_batch(self, inputs: List[Dict]) -> List[InferenceResult]: ...
    def submit(self, input: Dict[str, Tensor]) -> Future[InferenceResult]: ...
```

### MemoryOptimizer

Reduce memory footprint during inference.

```python
class MemoryOptimizer:
    def __init__(self, model: BrainAI, config: MemoryConfig): ...
    def enable_inference_mode(self) -> nn.Module: ...  # no_grad + eval + fp16
    def enable_gradient_checkpointing(self) -> None: ...
    def offload_to_cpu(self, modules: List[str]) -> None: ...
    def measure_memory(self, sample_input: Dict[str, Tensor]) -> MemoryReport: ...
```

### LatencyProfiler

Detailed per-module latency breakdown.

```python
class LatencyProfiler:
    def __init__(self, model: BrainAI): ...
    def profile(self, input: Dict[str, Tensor], n_runs: int = 100) -> ProfileReport: ...
    def per_module_breakdown(self) -> Dict[str, float]: ...
    def bottleneck_analysis(self) -> List[Bottleneck]: ...
    def export_chrome_trace(self, path: str) -> None: ...
```

## Key Concepts

### Optimization Opportunities by Module

| Module | Optimization | Expected Speedup |
|--------|-------------|-----------------|
| SNN Core | State caching between sequential inputs | 2-3× for sequences |
| Encoders | Cache encoder outputs for repeated modalities | 5-10× cache hit |
| HTM | Sparse column state pruning | 1.5-2× |
| Workspace | Early-exit when competition converges fast | 1.2-1.5× average |
| Reasoning | System 1 fast-path skip System 2 | 3-5× for confident inputs |
| Engram | Hash table preloading + batch lookup | 2-4× |
| Meta | Skip meta-adaptation at inference | 1.3-1.5× |

### Early Exit Strategy

The dual-process router can short-circuit inference:
1. System 1 processes input (fast, ~5ms)
2. If confidence > threshold (default 0.7): return immediately
3. If confidence < threshold: engage System 2 (slow, ~50ms)

Adaptive threshold tuning based on accuracy/latency trade-off.

### Inference Pipeline Stages

```
Input → Preprocess → Encode → [Cache Check] → SNN → HTM → Workspace → Route → Output
                                    ↓ hit                                   ↓
                              Return cached                          System 1 or 2
```

### Memory Budget

| Config | Model Size | Inference Memory | With Cache |
|--------|-----------|-----------------|------------|
| minimal | ~4MB | ~50MB | ~100MB |
| 1B | ~4GB | ~6GB | ~8GB |
| 3B | ~12GB | ~16GB | ~20GB |
| 7B | ~28GB | ~36GB | ~44GB |

## Configuration Surface

```python
@dataclass
class InferenceOptConfig:
    batch_size: int = 1
    max_batch_size: int = 64
    device: str = "auto"
    dtype: str = "fp16"                  # fp32 | fp16 | bf16
    # Caching
    enable_cache: bool = True
    l1_cache_mb: int = 512
    l2_cache_mb: int = 2048
    # Async
    enable_async: bool = False
    num_inference_threads: int = 2
    # Early exit
    enable_early_exit: bool = True
    confidence_threshold: float = 0.7
    # Memory
    gradient_checkpointing: bool = False
    cpu_offload_modules: List[str] = ()
    # Profiling
    enable_profiling: bool = False
    warmup_steps: int = 10
```

## Done-When Gates

1. **Batch Throughput** — `BatchInferenceEngine.infer_batch()` achieves >2× throughput vs sequential single-sample inference on batch_size=32.
2. **Cache Hit Speedup** — Second inference on identical input is >5× faster than first with `CacheManager` enabled; `stats()` reports >0% hit rate.
3. **Latency Profiling** — `LatencyProfiler.profile()` produces per-module breakdown that sums to total latency within 5% tolerance; `bottleneck_analysis()` correctly identifies the slowest module.

## Failure Modes

| Mode | Symptom | Fix |
|------|---------|-----|
| Cache staleness | Wrong results on changed model | Invalidate cache after model update |
| Async race condition | Intermittent wrong results | Ensure model is thread-safe; use model copies |
| FP16 overflow | NaN outputs | Use bf16 or keep sensitive layers in fp32 |
| Early exit too aggressive | Accuracy drops | Lower confidence_threshold or disable |
| Profiler overhead | Profiling slower than unprofiled | Use sampling profiler; reduce n_runs |

## Anti-Patterns

- Caching without invalidation — model updates make cached values stale
- Profiling in production — disable profiler for deployment runs
- Skipping warmup — first N inferences are always slower (JIT, CUDA kernels)
- Uniform dtype — some modules need fp32 (e.g., HTM sparse ops)
- Batching variable-length without padding strategy — causes shape errors

## Resources

### Reference Files
- **`references/batch-strategies.md`** — Dynamic batching, padding, bucketing algorithms
- **`references/caching-architecture.md`** — Multi-level cache design, eviction, invalidation
- **`references/async-inference.md`** — Thread safety, futures, event loops, model copies
- **`references/memory-optimization.md`** — FP16, offloading, gradient checkpointing, memory analysis
- **`references/testing-matrix.md`** — Test scenarios for inference optimization

### Asset Files
- **`assets/batch_engine_template.py`** — BatchInferenceEngine with dynamic batching, self-tests
- **`assets/cache_manager_template.py`** — CacheManager with L1/L2/L3 levels, eviction policies
- **`assets/async_engine_template.py`** — AsyncInferenceEngine with futures and thread safety
- **`assets/memory_optimizer_template.py`** — MemoryOptimizer with offloading and measurement
- **`assets/latency_profiler_template.py`** — LatencyProfiler with Chrome trace export
- **`assets/inference_opt_config_template.py`** — All inference config dataclasses + validation

### Scripts
- **`scripts/validate_inference_opt.py`** — Validates inference optimization against done-when gates
- **`scripts/gen_inference_tests.py`** — Generates 100+ pytest test cases
- **`scripts/inference_benchmark.py`** — Latency/throughput benchmarks with various configs
