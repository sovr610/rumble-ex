# Async Inference for Brain-Inspired AI

This document covers asynchronous inference patterns for brain_ai: thread safety with PyTorch models, model copies vs locks, asyncio integration, futures API, callback patterns, and error handling. The implementation is in `assets/async_engine_template.py`.

---

## 1. Thread Safety with PyTorch Models

### 1.1 The Thread Safety Problem

PyTorch models are **not inherently thread-safe** for inference. While the underlying CUDA operations are thread-safe (CUDA uses its own stream-based concurrency model), the Python-level model state can be corrupted by concurrent access:

- **Module buffers** (e.g., batch normalization running mean/variance) are mutated during `forward()` even in `eval()` mode. Concurrent calls to `forward()` on the same module can produce incorrect statistics.
- **SNN state** (membrane potentials, spike history) is mutable. Concurrent inference requests would corrupt each other's temporal state.
- **HTM column state** is mutable and order-dependent.
- **Global workspace working memory** maintains recurrent state across calls.

### 1.2 Thread Safety Strategies

**Strategy 1: Global Lock (Simple, Low Throughput)**

A single lock serializes all model access. Simple to implement but eliminates all concurrency benefit.

```python
class ThreadSafeModel:
    def __init__(self, model: nn.Module):
        self.model = model
        self.lock = threading.Lock()

    def infer(self, inputs: Dict[str, Tensor]) -> Tensor:
        with self.lock:
            return self.model(inputs)
```

**Strategy 2: Model Copies (Complex, High Throughput)**

Create N copies of the model, one per inference thread. Each copy has independent state. This is the recommended approach for brain_ai.

```python
class ModelPool:
    def __init__(self, model: nn.Module, num_copies: int = 2):
        self.models = [copy.deepcopy(model) for _ in range(num_copies)]
        self.semaphore = threading.Semaphore(num_copies)
        self.available = queue.Queue()
        for m in self.models:
            m.eval()
            self.available.put(m)

    def acquire(self) -> nn.Module:
        self.semaphore.acquire()
        return self.available.get()

    def release(self, model: nn.Module):
        self.available.put(model)
        self.semaphore.release()
```

**Strategy 3: Stateless Forward (Moderate Complexity)**

Reset all stateful components before each forward pass, eliminating state corruption. Works but loses temporal context for SNN and HTM.

```python
def infer_stateless(model, inputs):
    model.reset_state()  # Clear SNN, HTM, workspace state
    with torch.inference_mode():
        return model(inputs)
```

### 1.3 Recommended Approach for Brain_AI

Use **Model Copies** (Strategy 2) with the following considerations:

- **Encoder weights are shared** across copies (they are read-only during inference). Use `model.encoders.share_memory()` to share encoder weights across processes.
- **Stateful modules** (SNN, HTM, workspace) must be independent per copy.
- **Number of copies** = number of inference threads. Default: 2. More copies use more GPU memory.

Memory overhead per copy:
| Config | Model Size | Per-Copy Overhead | 2 Copies Total |
|--------|-----------|------------------|----------------|
| minimal | ~4MB | ~4MB (full copy) | ~8MB |
| 1B | ~4GB | ~500MB (state only) | ~4.5GB |
| 7B | ~28GB | ~2GB (state only) | ~30GB |

For large models, copy only the stateful modules (SNN core, HTM, workspace working memory) and share the rest.

---

## 2. Asyncio Integration

### 2.1 Why Asyncio

Python's asyncio provides non-blocking I/O concurrency without threads. For inference serving:
- The network I/O (receiving requests, sending responses) is async.
- The model inference (GPU compute) is synchronous.
- Asyncio bridges these: network handlers are async, inference is offloaded to a thread pool.

### 2.2 Event Loop Architecture

```
  [Async Request Handler]
           |
  await infer(input)
           |
  [ThreadPoolExecutor]  <-- runs model.forward() in a thread
           |
  [GPU Inference]
           |
  return result to async handler
```

```python
class AsyncInferenceEngine:
    def __init__(self, model: nn.Module, num_workers: int = 2):
        self.model_pool = ModelPool(model, num_copies=num_workers)
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=num_workers
        )

    async def infer(self, inputs: Dict[str, Tensor]) -> Tensor:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self._infer_sync,
            inputs
        )
        return result

    def _infer_sync(self, inputs: Dict[str, Tensor]) -> Tensor:
        model = self.model_pool.acquire()
        try:
            with torch.inference_mode():
                return model(inputs)
        finally:
            self.model_pool.release(model)
```

### 2.3 Batching with Asyncio

Combine async request collection with dynamic batching:

```python
class AsyncBatchEngine:
    def __init__(self, model, batch_timeout_ms=10.0, max_batch_size=32):
        self.model = model
        self.batch_timeout = batch_timeout_ms / 1000.0
        self.max_batch_size = max_batch_size
        self.pending = asyncio.Queue()
        self._batch_task = None

    async def infer(self, inputs: Dict[str, Tensor]) -> Tensor:
        future = asyncio.get_event_loop().create_future()
        await self.pending.put((inputs, future))
        return await future

    async def _batch_loop(self):
        while True:
            batch_items = []
            item = await self.pending.get()
            batch_items.append(item)
            deadline = time.time() + self.batch_timeout
            while len(batch_items) < self.max_batch_size:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(
                        self.pending.get(), timeout=remaining
                    )
                    batch_items.append(item)
                except asyncio.TimeoutError:
                    break
            inputs_list = [item[0] for item in batch_items]
            futures = [item[1] for item in batch_items]
            results = self._process_batch(inputs_list)
            for future, result in zip(futures, results):
                future.set_result(result)
```

---

## 3. Futures API

### 3.1 Synchronous Futures

For non-async code that still wants non-blocking inference, use `concurrent.futures.Future`:

```python
class InferenceEngine:
    def submit(self, inputs: Dict[str, Tensor]) -> Future:
        """Submit inference request, get a Future back immediately."""
        return self.executor.submit(self._infer_sync, inputs)
```

Usage:
```python
engine = InferenceEngine(model)
future = engine.submit({"vision": image_tensor})
# Do other work while inference runs
result = future.result(timeout=5.0)  # Block until ready, with timeout
```

### 3.2 Future Chaining

Chain multiple inference steps (useful for the brain_ai pipeline where you might want to inspect intermediate results):

```python
def submit_with_postprocess(self, inputs, postprocess_fn):
    """Submit inference and automatically postprocess the result."""
    future = self.submit(inputs)
    result_future = concurrent.futures.Future()

    def on_done(f):
        try:
            result = f.result()
            processed = postprocess_fn(result)
            result_future.set_result(processed)
        except Exception as e:
            result_future.set_exception(e)

    future.add_done_callback(on_done)
    return result_future
```

### 3.3 Batch Futures

Submit a batch as a single future that resolves to a list of results:

```python
def submit_batch(self, inputs_list: List[Dict[str, Tensor]]) -> Future:
    """Submit a batch, returns a single Future resolving to a list."""
    return self.executor.submit(self._infer_batch_sync, inputs_list)
```

---

## 4. Callback Patterns

### 4.1 Completion Callbacks

Register a function to be called when inference completes:

```python
def infer_with_callback(self, inputs, callback):
    """Run inference and call callback(result) on completion."""
    future = self.submit(inputs)
    future.add_done_callback(lambda f: callback(f.result()))
```

### 4.2 Progress Callbacks for Stream Inference

For streaming inference (processing a sequence frame by frame), provide progress updates:

```python
def infer_stream_with_progress(self, frames, on_frame_done, on_complete):
    """Process frames, calling on_frame_done after each, on_complete at end."""
    results = []
    for i, frame in enumerate(frames):
        result = self._infer_sync(frame)
        results.append(result)
        on_frame_done(i, result)
    on_complete(results)
```

### 4.3 Error Callbacks

Separate success and error callbacks:

```python
def infer_with_callbacks(self, inputs, on_success, on_error):
    """Run inference with separate success/error callbacks."""
    def _callback(future):
        try:
            result = future.result()
            on_success(result)
        except Exception as e:
            on_error(e)

    future = self.submit(inputs)
    future.add_done_callback(_callback)
```

---

## 5. Error Handling in Async Context

### 5.1 Exception Propagation

Exceptions in thread pool workers are captured by the Future. They are re-raised when `future.result()` is called or when the async coroutine is awaited.

```python
async def safe_infer(self, inputs):
    """Inference with structured error handling."""
    try:
        result = await self.infer(inputs)
        return InferenceResult(output=result, error=None)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return InferenceResult(output=None, error="GPU OOM")
        return InferenceResult(output=None, error=str(e))
    except Exception as e:
        return InferenceResult(output=None, error=f"Unexpected: {str(e)}")
```

### 5.2 Timeout Handling

Set timeouts on futures to prevent hanging:

```python
async def infer_with_timeout(self, inputs, timeout_seconds=5.0):
    """Inference with timeout."""
    try:
        result = await asyncio.wait_for(
            self.infer(inputs), timeout=timeout_seconds
        )
        return result
    except asyncio.TimeoutError:
        return InferenceResult(output=None, error="Timeout")
```

### 5.3 Graceful Shutdown

When shutting down the async engine, drain pending requests:

```python
async def shutdown(self, timeout=10.0):
    """Gracefully shut down, completing pending requests."""
    self._accepting_requests = False
    if self._batch_task:
        try:
            await asyncio.wait_for(self._batch_task, timeout=timeout)
        except asyncio.TimeoutError:
            self._batch_task.cancel()
    self.executor.shutdown(wait=True)
```

### 5.4 Circuit Breaker Pattern

If the model is consistently failing (e.g., GPU errors), stop sending requests:

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, reset_timeout=60.0):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.last_failure_time = 0
        self.state = "closed"  # closed=normal, open=rejecting, half_open=testing

    def record_success(self):
        self.failure_count = 0
        self.state = "closed"

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"

    def allow_request(self) -> bool:
        if self.state == "closed":
            return True
        if self.state == "open":
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = "half_open"
                return True
            return False
        if self.state == "half_open":
            return True
        return False
```

---

## 6. Performance Considerations

### 6.1 Thread Pool Sizing

The optimal number of threads depends on:
- **GPU utilization:** If a single inference saturates the GPU, adding threads only adds queue delay. Use 1-2 threads.
- **CPU-bound preprocessing:** If preprocessing (tokenization, image resizing) is CPU-bound, more threads help. Use 4-8 threads.
- **Model copies:** Each thread needs a model copy (or a lock). Memory constrains the thread count.

Rule of thumb: `num_threads = min(num_model_copies, num_cpu_cores // 4)`.

### 6.2 CUDA Stream Management

Each thread can use a separate CUDA stream for true GPU-level concurrency:

```python
def _infer_with_stream(self, model, inputs):
    """Run inference on a dedicated CUDA stream."""
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        with torch.inference_mode():
            result = model(inputs)
    stream.synchronize()
    return result
```

Multiple CUDA streams allow overlapping compute and memory transfers, improving GPU utilization.

### 6.3 Overhead Analysis

Async overhead per request:
- Thread pool dispatch: ~50us
- Future creation: ~10us
- Lock acquisition (if using locks): ~1us uncontended, ~100us contended
- CUDA stream synchronization: ~10us

For models with inference time exceeding 1ms, async overhead is negligible. For very fast models (sub-millisecond, e.g., minimal config), async overhead can dominate. In that case, use synchronous batching instead.

---

## 7. Integration with Brain_AI Modules

### 7.1 Stateful Module Handling

When using model copies for async inference, each copy maintains independent state:

```python
class AsyncBrainAI:
    def __init__(self, model: BrainAI, num_workers: int = 2):
        self.models = []
        for _ in range(num_workers):
            copy_model = deepcopy(model)
            copy_model.eval()
            copy_model.reset_state()
            self.models.append(copy_model)
```

### 7.2 State Consistency

For sequential inference (e.g., processing a video stream), all frames must go to the same model copy to maintain SNN/HTM state continuity:

```python
class SessionManager:
    """Routes requests from the same session to the same model copy."""
    def __init__(self, num_models: int):
        self.session_map = {}
        self.next_model = 0
        self.num_models = num_models

    def get_model_index(self, session_id: str) -> int:
        if session_id not in self.session_map:
            self.session_map[session_id] = self.next_model
            self.next_model = (self.next_model + 1) % self.num_models
        return self.session_map[session_id]
```

### 7.3 Async Cache Integration

The CacheManager must be thread-safe when used with async inference. Use a reader-writer lock:

```python
class ThreadSafeCacheManager:
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.lock = threading.RLock()

    def get(self, key: str) -> Optional[Tensor]:
        with self.lock:
            return self.cache.get(key)

    def put(self, key: str, value: Tensor) -> None:
        with self.lock:
            self.cache.put(key, value)
```

For high-throughput scenarios, use a concurrent dictionary or shard the cache by key prefix to reduce lock contention.
