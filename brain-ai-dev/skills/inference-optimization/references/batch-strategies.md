# Dynamic Batching Strategies for Brain-Inspired AI Inference

This document covers dynamic batching strategies for the brain_ai inference pipeline. The multi-layer cognitive architecture (encoders, SNN, HTM, workspace, reasoning, meta-learning) creates unique batching challenges: variable-length inputs across modalities, stateful modules (SNN membrane potentials, HTM column states), and branching execution paths (System 1 vs System 2). All strategies described here are implemented in `assets/batch_engine_template.py`.

---

## 1. Request Queuing and Batch Assembly

### 1.1 The Queuing Problem

In a serving scenario, inference requests arrive at irregular intervals. Processing each request individually underutilizes GPU parallelism. Dynamic batching collects requests into batches, trading a small amount of latency for significantly higher throughput.

The core tension: **waiting longer fills bigger batches (higher throughput) but increases per-request latency**. The batch assembler must balance these.

### 1.2 Queue Architecture

The recommended queue architecture for brain_ai uses a bounded priority queue:

```
Incoming Requests
       |
       v
  [Priority Queue]  <-- bounded by max_queue_size
       |
  [Batch Assembler] <-- triggers on max_batch_size OR timeout
       |
       v
  [GPU Inference]
       |
       v
  [Response Router] --> routes results back to callers
```

**Priority levels:**
- **P0 (critical):** Real-time control actions (active_inference output_type). Maximum latency 10ms.
- **P1 (interactive):** Classification and generation requests from live users. Target latency 50ms.
- **P2 (batch):** Offline processing, evaluation, bulk inference. No latency constraint.

P0 requests bypass the queue entirely and execute immediately. P1 and P2 requests enter the queue and are assembled into batches.

### 1.3 Batch Assembly Triggers

A batch is dispatched to the GPU when any of these conditions is met:

1. **Size trigger:** The queue contains `max_batch_size` requests.
2. **Timeout trigger:** The oldest request in the queue has waited `batch_timeout_ms` milliseconds.
3. **Memory trigger:** Estimated GPU memory for the batch exceeds `max_batch_memory_mb`.

The timeout trigger is critical. Without it, low-traffic periods would leave requests waiting indefinitely. A typical timeout for interactive workloads is 5-10ms; for batch workloads, 50-100ms.

```python
class BatchAssembler:
    def __init__(self, max_batch_size=32, batch_timeout_ms=10.0):
        self.queue = []
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms

    def should_dispatch(self) -> bool:
        if len(self.queue) >= self.max_batch_size:
            return True
        if len(self.queue) > 0:
            oldest_age_ms = (time.time() - self.queue[0].timestamp) * 1000
            if oldest_age_ms >= self.batch_timeout_ms:
                return True
        return False
```

### 1.4 Request Metadata

Each queued request carries metadata for routing and padding decisions:

```python
@dataclass
class InferenceRequest:
    request_id: str
    inputs: Dict[str, Tensor]      # modality -> tensor
    priority: int = 1              # 0=critical, 1=interactive, 2=batch
    timestamp: float = 0.0         # time.time() at submission
    callback: Optional[Callable] = None
    modalities: Set[str] = None    # which modalities are present
    max_seq_len: int = 0           # longest sequence dimension
```

The `modalities` and `max_seq_len` fields enable bucketing (Section 3).

---

## 2. Padding Strategies for Variable-Length Inputs

### 2.1 The Variable-Length Problem

Brain_ai processes multiple modalities simultaneously. Within a batch:
- Vision inputs may have different spatial resolutions (though typically standardized).
- Text inputs have different sequence lengths.
- Audio inputs have different durations.
- Some requests may include modalities that others lack.

All tensors in a batch must have the same shape. The padding strategy determines how we handle mismatches.

### 2.2 Padding Approaches

**Right-padding (recommended for text/audio):**
Append zeros to the end of shorter sequences. This is the standard approach for transformer-based text and audio encoders. The attention mask ensures padded positions do not contribute to outputs.

```python
def pad_sequences(tensors: List[Tensor], pad_value: float = 0.0) -> Tuple[Tensor, Tensor]:
    """Pad variable-length tensors to the maximum length in the batch."""
    max_len = max(t.shape[1] for t in tensors)
    padded = torch.full((len(tensors), max_len, *tensors[0].shape[2:]), pad_value)
    mask = torch.zeros(len(tensors), max_len, dtype=torch.bool)
    for i, t in enumerate(tensors):
        padded[i, :t.shape[1]] = t
        mask[i, :t.shape[1]] = True
    return padded, mask
```

**Spatial padding (for vision):**
Pad images to the maximum height and width in the batch. Less common since vision encoders typically expect fixed-size inputs (e.g., 224x224 or 384x384 after preprocessing). If variable sizes are needed, pad with zeros and use a spatial mask.

**Missing modality handling:**
When a request lacks a modality present in other batch members, create a zero tensor of the appropriate shape and set a modality mask to False. The global workspace's attention mechanism naturally handles this: zero-content modalities receive near-zero attention weights.

```python
def collate_multimodal(requests: List[InferenceRequest],
                       all_modalities: Set[str]) -> Dict[str, Tensor]:
    """Collate requests with potentially different modalities."""
    batch = {}
    modality_masks = {}
    for mod in all_modalities:
        tensors = []
        mask = []
        for req in requests:
            if mod in req.inputs:
                tensors.append(req.inputs[mod])
                mask.append(True)
            else:
                ref_shape = _find_reference_shape(requests, mod)
                tensors.append(torch.zeros(ref_shape))
                mask.append(False)
        batch[mod] = torch.stack(tensors)
        modality_masks[mod] = torch.tensor(mask, dtype=torch.bool)
    return batch, modality_masks
```

### 2.3 Padding Efficiency

Padding wastes computation proportional to the difference between the shortest and longest sequences in the batch. For a batch where the shortest sequence is 10 tokens and the longest is 512 tokens, 98% of computation on the shortest sequence is wasted.

This motivates bucketing (Section 3).

---

## 3. Bucketing by Sequence Length

### 3.1 Bucketing Concept

Bucketing groups requests by similar sequence lengths before batching. This minimizes padding waste. The trade-off: bucketing introduces a sorting step and may increase latency for short sequences (they wait for other short sequences to arrive).

### 3.2 Bucket Definition

Define buckets by sequence length ranges:

```python
DEFAULT_BUCKETS = [
    (0, 32),      # Short: 0-32 tokens
    (33, 128),    # Medium: 33-128 tokens
    (129, 512),   # Long: 129-512 tokens
    (513, 2048),  # Very long: 513-2048 tokens
    (2049, 8192), # Ultra long: 2049-8192 tokens
]
```

Each bucket maintains its own queue. The batch assembler checks each bucket independently for dispatch triggers.

### 3.3 Bucket Assignment

For multimodal inputs, the bucket is determined by the **dominant modality's** sequence length. For text-heavy requests, this is the text sequence length. For vision-only requests, bucketing is less important since vision inputs are typically fixed-size.

```python
def assign_bucket(request: InferenceRequest, buckets: List[Tuple[int, int]]) -> int:
    seq_len = request.max_seq_len
    for i, (lo, hi) in enumerate(buckets):
        if lo <= seq_len <= hi:
            return i
    return len(buckets) - 1  # overflow bucket
```

### 3.4 Adaptive Bucketing

Static bucket boundaries may not match the actual sequence length distribution. Adaptive bucketing uses quantiles of observed sequence lengths to define boundaries:

```python
class AdaptiveBucketer:
    def __init__(self, num_buckets: int = 5, window_size: int = 1000):
        self.num_buckets = num_buckets
        self.window_size = window_size
        self.history = []

    def update(self, seq_len: int):
        self.history.append(seq_len)
        if len(self.history) > self.window_size:
            self.history.pop(0)

    def get_boundaries(self) -> List[Tuple[int, int]]:
        if len(self.history) < self.num_buckets:
            return DEFAULT_BUCKETS
        sorted_lens = sorted(self.history)
        quantiles = [sorted_lens[int(i * len(sorted_lens) / self.num_buckets)]
                     for i in range(self.num_buckets + 1)]
        return [(quantiles[i], quantiles[i+1]) for i in range(self.num_buckets)]
```

---

## 4. Throughput vs Latency Trade-offs

### 4.1 The Fundamental Trade-off

Larger batches increase throughput (requests/second) but also increase latency (time per request). The relationship is not linear:

- **Throughput** scales roughly linearly with batch size until GPU saturation, then plateaus.
- **Latency** increases roughly linearly with batch size (each request waits for the whole batch).
- **GPU utilization** increases with batch size; small batches leave compute units idle.

### 4.2 Optimal Batch Size

The optimal batch size depends on:

1. **Model size:** Larger models saturate GPU earlier. A 7B model may saturate at batch_size=8; a 1M model at batch_size=256.
2. **Sequence length:** Longer sequences use more memory, limiting batch size.
3. **GPU memory:** Batch size is constrained by available GPU memory.
4. **Latency SLA:** If p99 latency must be under 50ms, batch size is constrained.

A practical approach: start with batch_size=1 and double until throughput stops improving or latency exceeds the SLA.

```python
def find_optimal_batch_size(model, sample_input, max_batch_size=128,
                            latency_budget_ms=50.0) -> int:
    """Binary search for optimal batch size within latency budget."""
    best_throughput = 0
    best_batch_size = 1

    for bs in [1, 2, 4, 8, 16, 32, 64, 128]:
        if bs > max_batch_size:
            break
        batch = {k: v.expand(bs, *v.shape[1:]) for k, v in sample_input.items()}

        # Warmup
        for _ in range(3):
            model(batch)

        # Measure
        start = time.time()
        n_iters = 10
        for _ in range(n_iters):
            model(batch)
        elapsed = time.time() - start

        latency_ms = (elapsed / n_iters) * 1000
        throughput = (bs * n_iters) / elapsed

        if latency_ms <= latency_budget_ms and throughput > best_throughput:
            best_throughput = throughput
            best_batch_size = bs

    return best_batch_size
```

### 4.3 Brain_AI-Specific Considerations

The brain_ai pipeline has unique batching characteristics:

**SNN Core:** The SNN unrolls over `num_timesteps` time steps. Batching across requests is straightforward, but the sequential nature of time steps limits parallelism. Consider accumulating SNN outputs across time steps before passing to the next layer, rather than running the full pipeline for each time step.

**HTM Layer:** HTM maintains per-column state. When batching, each batch element needs independent HTM state. This means HTM state tensors scale linearly with batch size. For the production config with 16384 columns and 64 cells per column, each batch element requires approximately 4MB of HTM state.

**Dual-Process Routing:** Within a batch, some elements may be routed to System 1 (fast) and others to System 2 (slow). Naive implementation processes the entire batch through System 2, wasting computation on confident elements. An optimized approach splits the batch after the confidence check:

```python
# After workspace processing
confidence = compute_confidence(workspace)
system1_mask = confidence > threshold
system2_mask = ~system1_mask

# Process subsets independently
if system1_mask.any():
    output[system1_mask] = system1(workspace[system1_mask])
if system2_mask.any():
    output[system2_mask] = system2(workspace[system2_mask])
```

This dynamic splitting reduces average batch inference time when many inputs are confident.

**Global Workspace Competition:** The workspace attention competition is batched efficiently since it uses standard multi-head attention. No special handling needed.

### 4.4 Batch Size Recommendations by Configuration

| Config | Model Size | Recommended Batch Size | Expected Throughput |
|--------|-----------|----------------------|-------------------|
| minimal | ~4MB | 128-256 | >10K samples/sec |
| 1B | ~4GB | 16-32 | ~500 samples/sec |
| 3B | ~12GB | 8-16 | ~200 samples/sec |
| 7B | ~28GB | 4-8 | ~50 samples/sec |

These are estimates for a single A100 80GB GPU. Actual numbers depend on sequence length and enabled modules.

---

## 5. Streaming Inference

### 5.1 Stream Processing

For continuous input streams (e.g., real-time sensor data, video frames), the batch engine can accumulate frames and process them as micro-batches:

```python
def infer_stream(self, input_stream: Iterator, micro_batch_size: int = 8):
    """Process a stream of inputs in micro-batches."""
    buffer = []
    for item in input_stream:
        buffer.append(item)
        if len(buffer) >= micro_batch_size:
            yield from self.infer_batch(buffer)
            buffer.clear()
    if buffer:  # flush remaining
        yield from self.infer_batch(buffer)
```

### 5.2 Stateful Stream Processing

For SNN and HTM modules that maintain temporal state, stream processing must preserve state across micro-batches. The batch engine should:

1. Initialize SNN membrane potentials and HTM column states at stream start.
2. Carry state forward between micro-batches (do not reset).
3. Detach state from computation graph between micro-batches (to prevent memory accumulation from backprop through time).

```python
def infer_stream_stateful(self, input_stream, micro_batch_size=8):
    state = self.model.reset_state()
    buffer = []
    for item in input_stream:
        buffer.append(item)
        if len(buffer) >= micro_batch_size:
            batch = self._collate(buffer)
            output, state = self.model.forward_with_state(batch, state)
            state = {k: v.detach() for k, v in state.items()}
            yield from self._unbatch(output, len(buffer))
            buffer.clear()
```

---

## 6. Warmup Strategy

### 6.1 Why Warmup Matters

The first few inference calls are significantly slower due to:
- **CUDA kernel compilation:** PyTorch compiles CUDA kernels on first use.
- **Memory allocation:** First allocation triggers cudaMalloc.
- **torch.compile graph capture:** If using torch.compile, the first call traces the graph.
- **Cache population:** The CacheManager's L1 cache is empty.

### 6.2 Warmup Protocol

```python
def warmup(self, sample_input: Dict[str, Tensor], n_warmup: int = 10):
    """Run warmup inference to trigger JIT compilation and cache filling."""
    self.model.eval()
    with torch.inference_mode():
        for batch_size in [1, 2, 4, 8, self.config.max_batch_size]:
            batch = {k: v.expand(batch_size, *v.shape[1:])
                     for k, v in sample_input.items()}
            for _ in range(n_warmup):
                self.model(batch)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
```

Running warmup at multiple batch sizes ensures CUDA kernels are compiled for all expected batch sizes. This is especially important when using `torch.compile` with dynamic shapes.

---

## 7. Error Handling in Batch Inference

### 7.1 Partial Batch Failures

In a batch of N requests, some may fail (e.g., corrupted input, shape mismatch) while others succeed. The batch engine should:

1. Validate each request before adding to the batch.
2. If a batch forward pass fails, fall back to per-request inference to isolate the failing request.
3. Return error results for failed requests and valid results for successful ones.

```python
def infer_batch_safe(self, inputs):
    try:
        return self._infer_batch_inner(inputs)
    except RuntimeError:
        # Fallback: process one-by-one to find the bad request
        results = []
        for i, inp in enumerate(inputs):
            try:
                results.append(self._infer_single(inp))
            except Exception as e:
                results.append(InferenceResult(error=str(e), request_idx=i))
        return results
```

### 7.2 OOM Recovery

If a batch causes an out-of-memory error, the engine should:
1. Clear the CUDA cache (`torch.cuda.empty_cache()`).
2. Halve the batch size.
3. Retry with the smaller batch.
4. If batch_size=1 still OOMs, report an unrecoverable error.

---

## 8. Monitoring and Metrics

Track these metrics for batch inference health:

- **batch_fill_ratio:** Average `actual_batch_size / max_batch_size`. Low values indicate over-provisioning.
- **queue_wait_time_ms:** Time between request arrival and batch dispatch. Should be less than batch_timeout_ms.
- **padding_waste_ratio:** Fraction of padded (wasted) compute. High values indicate poor bucketing.
- **system2_ratio:** Fraction of batch elements routed to System 2. High values indicate many uncertain inputs.
- **throughput_rps:** Requests per second.
- **p50/p99_latency_ms:** Median and tail latency.

These metrics can be exposed via the `BatchInferenceEngine.stats()` method or integrated with Prometheus/Grafana for production monitoring.
