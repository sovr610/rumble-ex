# Testing Matrix: Inference Optimization

This document defines concrete test specifications for inference optimization components. Each test maps to a done-when gate or a specific optimization contract. Tests use mock models (nn.Linear or small nn.Sequential) to ensure fast execution without requiring the full brain_ai system.

---

## Category 1: Batch Throughput Tests

### 1a) Batch vs Sequential Throughput (Done-When Gate 1)

**Objective:** Verify that `BatchInferenceEngine.infer_batch()` achieves at least 2x throughput compared to sequential single-sample inference on batch_size=32.

**Procedure:**
1. Create a mock model: `nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 10))`.
2. Create an `InferenceOptConfig` with `max_batch_size=32`.
3. Create a `BatchInferenceEngine` with the mock model and config.
4. Prepare 32 identical sample inputs, each a dict `{"features": torch.randn(1, 256)}`.
5. **Sequential baseline:** Loop 32 times, calling `model(input)` for each. Record total wall time.
6. **Batch inference:** Call `engine.infer_batch(inputs_32)`. Record wall time.
7. Compute throughput: `samples_per_second = 32 / wall_time`.
8. Assert: `batch_throughput / sequential_throughput >= 2.0`.

**Edge cases to test:**
- Batch size = 1 (should be no worse than sequential)
- Batch size = max_batch_size (should be fastest)
- Batch size exceeding max_batch_size (should automatically split)

---

### 1b) Dynamic Padding Correctness

**Objective:** Verify that padding produces correct results for variable-length inputs.

**Procedure:**
1. Create a mock model that sums along the sequence dimension: output = input.sum(dim=1).
2. Create 4 inputs with lengths [10, 20, 30, 40].
3. Batch-infer all 4.
4. Sequentially infer each with its original length.
5. Assert: batch results match sequential results within tolerance 1e-5.

---

### 1c) Batch Engine Warmup

**Objective:** Verify that warmup reduces latency for subsequent inference.

**Procedure:**
1. Create a mock model and batch engine.
2. Measure latency of first inference (cold).
3. Call `engine.warmup(sample_input, n_warmup=10)`.
4. Measure latency of next inference (warm).
5. Assert: warm latency <= cold latency (warmup should not make things slower).
6. Note: On CPU, the difference may be small. The test primarily validates that warmup runs without error.

---

### 1d) Stream Inference

**Objective:** Verify that `infer_stream()` processes all items and yields correct number of results.

**Procedure:**
1. Create a mock model and batch engine.
2. Create 50 input items.
3. Call `list(engine.infer_stream(iter(inputs)))`.
4. Assert: result count equals input count (50).
5. Assert: each result has correct output shape.

---

### 1e) Batch Assembly with Timeout

**Objective:** Verify that batch assembly triggers on timeout even with partial batches.

**Procedure:**
1. Create a batch assembler with max_batch_size=16, timeout_ms=50.
2. Add 5 items (less than max_batch_size).
3. Wait 60ms.
4. Assert: `should_dispatch()` returns True.

---

## Category 2: Cache Hit Speedup Tests

### 2a) Cache Hit Speedup (Done-When Gate 2)

**Objective:** Second inference on identical input is at least 5x faster than first; `stats()` reports hit rate above 0%.

**Procedure:**
1. Create a `CacheManager` with `CacheConfig(l1_cache_mb=64)`.
2. Create a mock model with deliberate slowdown: `time.sleep(0.01)` in forward.
3. Create a sample input dict.
4. **First inference:** Compute output, store in cache. Record wall time.
5. **Second inference:** Check cache first, return if hit. Record wall time.
6. Assert: `first_time / second_time >= 5.0`.
7. Assert: `cache.stats().hit_rate > 0.0`.

---

### 2b) Cache LRU Eviction

**Objective:** Verify that LRU eviction removes the least recently used entry when the cache is full.

**Procedure:**
1. Create a CacheManager with tiny L1 budget (e.g., 1MB).
2. Insert entries until the cache is full.
3. Insert one more entry.
4. Assert: the oldest (least recently accessed) entry is evicted.
5. Assert: the newest entry is present.
6. Assert: total cache size does not exceed budget.

---

### 2c) Cache Invalidation

**Objective:** Verify that `invalidate()` clears all entries and resets stats.

**Procedure:**
1. Create a CacheManager and insert 10 entries.
2. Assert: `stats().l1_entries == 10`.
3. Call `invalidate()`.
4. Assert: `stats().l1_entries == 0`.
5. Assert: `get(any_key)` returns None.

---

### 2d) Cache Key Collision Resistance

**Objective:** Verify that different inputs produce different cache keys.

**Procedure:**
1. Create 100 random input tensors.
2. Compute cache key for each.
3. Assert: all keys are unique (no collisions).

---

### 2e) Multi-Level Cache Fallthrough

**Objective:** Verify that cache lookups fall through from L1 to L2 to L3.

**Procedure:**
1. Create a CacheManager with all three levels.
2. Insert an entry only at L2 level.
3. Call `get(key)` without specifying level.
4. Assert: entry is found (fell through to L2).
5. Assert: stats show L1 miss and L2 hit.

---

### 2f) TTL Expiration

**Objective:** Verify that entries with TTL are evicted after expiration.

**Procedure:**
1. Create a CacheManager with TTL support.
2. Insert an entry with TTL=0.1 seconds.
3. Immediately get: assert found.
4. Wait 0.15 seconds.
5. Get again: assert not found (expired).

---

## Category 3: Latency Profiling Tests

### 3a) Per-Module Breakdown Sums to Total (Done-When Gate 3)

**Objective:** `LatencyProfiler.profile()` produces per-module breakdown that sums to total latency within 5% tolerance; `bottleneck_analysis()` correctly identifies the slowest module.

**Procedure:**
1. Create a mock model with 3 named submodules:
   - `fast_layer`: `nn.Linear(64, 64)` (fast)
   - `medium_layer`: `nn.Linear(64, 256)` (medium)
   - `slow_layer`: `nn.Sequential(nn.Linear(256, 1024), nn.ReLU(), nn.Linear(1024, 10))` (slow)
2. Create a `LatencyProfiler` with the mock model.
3. Call `profiler.profile(sample_input, n_runs=50)`.
4. Get `per_module_breakdown()`.
5. Assert: `sum(per_module_times.values())` is within 5% of `total_latency`.
6. Call `bottleneck_analysis()`.
7. Assert: the bottleneck module name is `slow_layer`.

---

### 3b) Profile Report Structure

**Objective:** Verify the ProfileReport dataclass has all required fields.

**Procedure:**
1. Run profiler on any model.
2. Inspect the returned ProfileReport.
3. Assert it has: `total_latency_ms`, `per_module_ms`, `n_runs`, `std_ms`, `p50_ms`, `p99_ms`.

---

### 3c) Bottleneck Analysis Ordering

**Objective:** Verify bottleneck_analysis returns modules sorted by latency (descending).

**Procedure:**
1. Create a model with 5 submodules of varying sizes.
2. Profile and get bottleneck analysis.
3. Assert: returned list is sorted by latency in descending order.

---

### 3d) Chrome Trace Export

**Objective:** Verify that `export_chrome_trace()` produces valid JSON.

**Procedure:**
1. Run profiler on a mock model.
2. Call `export_chrome_trace("/tmp/test_trace.json")`.
3. Assert: file exists.
4. Assert: file content is valid JSON.
5. Assert: JSON contains "traceEvents" key.
6. Assert: each trace event has "name", "ph", "ts", "dur" fields.

---

### 3e) Profiler Overhead

**Objective:** Verify that the profiler does not add more than 50% overhead.

**Procedure:**
1. Run 100 inferences without profiling. Record time.
2. Run 100 inferences with profiling. Record time.
3. Compute overhead: `(profiled_time - baseline_time) / baseline_time`.
4. Assert: overhead < 0.50 (50%).

---

## Category 4: Async Correctness Tests

### 4a) Async Inference Returns Same Result as Sync

**Objective:** Verify that async inference produces the same output as synchronous inference.

**Procedure:**
1. Create a mock model and an `AsyncInferenceEngine`.
2. Run sync inference: `expected = model(input)`.
3. Run async inference: `actual = await engine.infer(input)`.
4. Assert: `torch.allclose(expected, actual)`.

---

### 4b) Future Resolution

**Objective:** Verify that `submit()` returns a Future that resolves correctly.

**Procedure:**
1. Create engine and submit an input.
2. Assert: returned object is a Future.
3. Call `future.result(timeout=5.0)`.
4. Assert: result is a valid tensor with correct shape.

---

### 4c) Multiple Concurrent Submissions

**Objective:** Verify that multiple submissions all complete correctly.

**Procedure:**
1. Submit 10 inputs simultaneously.
2. Collect all futures.
3. Wait for all to complete.
4. Assert: all 10 results are valid tensors.
5. Assert: no exceptions were raised.

---

### 4d) Error Handling in Async

**Objective:** Verify that errors in async inference are properly propagated.

**Procedure:**
1. Create a model that raises RuntimeError on certain inputs.
2. Submit a failing input.
3. Assert: `future.result()` raises the expected exception.

---

### 4e) Engine Shutdown

**Objective:** Verify graceful shutdown of the async engine.

**Procedure:**
1. Create engine and submit 5 inputs.
2. Call `engine.shutdown()`.
3. Assert: all 5 results are available.
4. Assert: submitting after shutdown raises an error.

---

## Category 5: Memory Measurement Tests

### 5a) Model Size Measurement

**Objective:** Verify that `measure_memory()` correctly reports model size.

**Procedure:**
1. Create a model with known size: `nn.Linear(1000, 1000)` = 1000*1000*4 bytes (params) + 1000*4 bytes (bias) = ~4MB.
2. Call `optimizer.measure_memory(sample_input)`.
3. Assert: `report.param_size_mb` is approximately 4.0 (within 10%).

---

### 5b) Inference Mode Reduces Memory

**Objective:** Verify that `enable_inference_mode()` applies eval mode and no_grad.

**Procedure:**
1. Create a model in training mode.
2. Call `optimizer.enable_inference_mode()`.
3. Assert: `model.training == False`.
4. Run inference inside inference_mode context.
5. Assert: no gradients are computed (param.grad is None for all params after forward).

---

### 5c) CPU Offloading

**Objective:** Verify that `offload_to_cpu()` moves specified modules to CPU.

**Procedure:**
1. Create a model on "cpu" with named submodules.
2. Call `optimizer.offload_to_cpu(["slow_module"])`.
3. Assert: `slow_module` parameters are on CPU.
4. Assert: other modules remain on their original device.

---

### 5d) Memory Report Content

**Objective:** Verify MemoryReport contains all required fields.

**Procedure:**
1. Create optimizer and measure memory.
2. Assert report has: model_size_mb, param_size_mb, buffer_size_mb, dtype, device.
3. Assert: all numeric values are non-negative.
4. Assert: model_size_mb = param_size_mb + buffer_size_mb.

---

### 5e) Dtype Conversion

**Objective:** Verify that dtype conversion actually changes parameter dtypes.

**Procedure:**
1. Create a model in FP32.
2. Apply FP16 conversion.
3. Assert: all parameters have dtype float16.
4. Run inference and assert output is finite.

---

## Category 6: Configuration Tests

### 6a) Config Validation

**Objective:** Verify that invalid configs are rejected.

**Procedure:**
1. Create config with `max_batch_size=0`. Assert: validation raises ValueError.
2. Create config with `l1_cache_mb=-1`. Assert: validation raises ValueError.
3. Create config with `confidence_threshold=1.5`. Assert: validation raises ValueError.
4. Create config with `dtype="int4"`. Assert: validation raises ValueError.

---

### 6b) Config Defaults

**Objective:** Verify default config values match SKILL.md specification.

**Procedure:**
1. Create default `InferenceOptConfig()`.
2. Assert: `batch_size == 1`.
3. Assert: `max_batch_size == 64`.
4. Assert: `dtype == "fp16"`.
5. Assert: `enable_cache == True`.
6. Assert: `confidence_threshold == 0.7`.
7. Assert: `warmup_steps == 10`.

---

### 6c) Config Serialization

**Objective:** Verify config can be serialized to and from dict/JSON.

**Procedure:**
1. Create a config with non-default values.
2. Convert to dict.
3. Recreate config from dict.
4. Assert: all fields match.

---

## Category 7: Integration Tests

### 7a) End-to-End Pipeline

**Objective:** Verify the complete optimization pipeline works together.

**Procedure:**
1. Create a mock model.
2. Apply memory optimization (inference mode, FP32 since CPU).
3. Create batch engine with cache manager.
4. Warmup the engine.
5. Run batch inference twice with the same inputs.
6. Assert: second run shows cache hits.
7. Profile the pipeline.
8. Assert: profile report is valid.

---

### 7b) Config-Driven Setup

**Objective:** Verify that all components can be configured via InferenceOptConfig.

**Procedure:**
1. Create a full InferenceOptConfig.
2. Instantiate BatchInferenceEngine, CacheManager, MemoryOptimizer, LatencyProfiler.
3. Assert: all instantiate without error.
4. Run a simple inference through each.
5. Assert: all produce valid outputs.
