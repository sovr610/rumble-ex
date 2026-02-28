# Gradient Checkpointing Testing Matrix

## Overview

This matrix covers all test scenarios for gradient checkpointing. Tests are organized by component. Each scenario includes the test objective, setup, expected behavior, and implementation notes.

---

## Category 1: CheckpointWrapper

### 1.1 Basic Wrapping Preserves Output

**Objective**: Wrapping a module in `CheckpointWrapper` produces the same forward output as the unwrapped module.

**Setup**:
```python
model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 16))
wrapped = CheckpointWrapper(model, use_reentrant=False)
x = torch.randn(4, 32)
```

**Expected**: `torch.allclose(model(x), wrapped(x), atol=1e-6)` is True.

---

### 1.2 Gradient Correctness

**Objective**: Gradients from checkpointed backward match non-checkpointed backward.

**Setup**: Same model as 1.1. Run forward + backward on both, compare `.grad` on all parameters.

**Expected**: `torch.allclose(p1.grad, p2.grad, atol=1e-5)` for all parameter pairs.

---

### 1.3 Memory Reduction

**Objective**: Checkpointed forward-backward uses less peak activation memory than non-checkpointed.

**Setup**: Multi-layer model (8+ layers). Measure `torch.cuda.max_memory_allocated()` for both paths.

**Expected**: Checkpointed peak memory is at least 20% lower than non-checkpointed. (Exact savings depend on model depth and layer sizes.)

**Implementation notes**: Requires CUDA. Skip on CPU-only machines.

---

### 1.4 Non-Reentrant Mode (use_reentrant=False)

**Objective**: Non-reentrant checkpointing works correctly with autograd.

**Expected**: Forward and backward complete without error. Gradients are correct.

---

### 1.5 Reentrant Mode (use_reentrant=True)

**Objective**: Reentrant mode still works (backward compatibility).

**Expected**: Forward and backward complete. Gradients match non-checkpointed within tolerance.

**Implementation notes**: Reentrant mode is deprecated in PyTorch 2.x but should still function.

---

### 1.6 RNG State Preservation

**Objective**: With `preserve_rng_state=True`, dropout produces identical masks in forward and recomputed forward.

**Setup**: Model with dropout layer. Run checkpointed forward-backward.

**Expected**: Gradients match non-checkpointed version exactly (dropout masks are the same).

---

### 1.7 Nested CheckpointWrapper

**Objective**: Wrapping already-wrapped modules does not break forward/backward.

**Setup**: `CheckpointWrapper(CheckpointWrapper(model))`.

**Expected**: Forward and backward complete. Gradients are correct (though this wastes compute).

---

### 1.8 Module with No Parameters

**Objective**: Wrapping a module with no learnable parameters (e.g., `nn.ReLU()`) works.

**Expected**: No error. Forward output is correct. Backward is a no-op (no gradients to compute).

---

### 1.9 Module with kwargs

**Objective**: `CheckpointWrapper` correctly passes keyword arguments to the wrapped module's forward.

**Setup**: Module whose forward takes `(x, mask=None)`.

**Expected**: `wrapped(x, mask=mask)` produces correct output.

---

## Category 2: MemoryProfiler

### 2.1 Profile Returns Positive Values

**Objective**: `MemoryProfiler.profile()` returns a `ProfileReport` with positive `activation_memory_mb` for layers that produce activations.

**Expected**: All non-trivial layers have `activation_memory_mb > 0`.

**Implementation notes**: Requires CUDA.

---

### 2.2 Profile Is Reproducible

**Objective**: Running `profile()` multiple times produces consistent results.

**Expected**: Per-layer values vary by less than 10% across runs.

---

### 2.3 Profile Sums Are Reasonable

**Objective**: Sum of per-layer activation memory is within 2x of total activation memory.

**Expected**: `0.5 * total <= sum(layers) <= 2.0 * total`. (Some overlap and allocator fragmentation is expected.)

---

### 2.4 Recommend Layers Respects Threshold

**Objective**: `recommend_layers(threshold_mb)` returns only layers above the threshold.

**Expected**: All returned layers have `activation_memory_mb > threshold_mb`. No layers above threshold are missing.

---

### 2.5 Recommend Layers with High Threshold Returns Empty

**Objective**: When threshold is higher than any layer's activation memory, return empty list.

**Expected**: `recommend_layers(1e6)` returns `[]`.

---

### 2.6 Recommend Layers with Zero Threshold Returns All

**Objective**: When threshold is 0, all non-trivial layers are returned.

**Expected**: `recommend_layers(0.0)` returns all layers with positive activation memory.

---

### 2.7 Profile Report Sorted Descending

**Objective**: `ProfileReport.layers` is sorted by `activation_memory_mb` descending.

**Expected**: `layers[i].activation_memory_mb >= layers[i+1].activation_memory_mb` for all i.

---

## Category 3: SelectiveCheckpointer

### 3.1 Apply Wraps Only Expensive Layers

**Objective**: After `apply()`, only layers above the memory threshold are wrapped in `CheckpointWrapper`.

**Setup**: 8-layer model with varying layer sizes. Set threshold to checkpoint top 3 layers.

**Expected**: Exactly 3 layers are wrapped. Others are unchanged.

---

### 3.2 Apply with strategy="full" Wraps All Layers

**Objective**: With `strategy="full"`, all eligible child modules are wrapped.

**Expected**: Every child module is a `CheckpointWrapper` instance.

---

### 3.3 Apply with strategy="none" Wraps Nothing

**Objective**: With `strategy="none"` or `enabled=False`, no layers are wrapped.

**Expected**: Model is returned unchanged.

---

### 3.4 Apply with strategy="sequential"

**Objective**: For `nn.Sequential` models, `checkpoint_sequential` is used with the configured number of segments.

**Expected**: Model runs correctly. Memory savings observed (CUDA test).

---

### 3.5 Exclude Patterns Respected

**Objective**: Layers matching `exclude_patterns` are never checkpointed, even if above threshold.

**Setup**: `exclude_patterns=["norm", "embed"]`.

**Expected**: LayerNorm and Embedding layers are not checkpointed regardless of their memory cost.

---

### 3.6 Include Patterns Respected

**Objective**: Layers matching `include_patterns` are always checkpointed, even if below threshold.

**Setup**: `include_patterns=["attention"]`.

**Expected**: All attention layers are checkpointed regardless of profiled memory.

---

### 3.7 Get Checkpointed Layers Returns Correct Names

**Objective**: After `apply()`, `get_checkpointed_layers()` returns the names of all wrapped layers.

**Expected**: Returned list matches the layers that have `CheckpointWrapper` applied.

---

### 3.8 Gradient Correctness After Selective Apply

**Objective**: Model with selective checkpointing produces correct gradients.

**Expected**: Gradients match non-checkpointed model within `atol=1e-5` (float32).

---

## Category 4: SNN Timestep Checkpointing

### 4.1 Per-Timestep Checkpoint Produces Correct Output

**Objective**: SNN unrolled with per-timestep checkpointing produces same output as without.

**Setup**: Simple SNN with 10 timesteps.

**Expected**: Output spikes match non-checkpointed version exactly.

---

### 4.2 Memory Scales with Chunk Size, Not Total Timesteps

**Objective**: Peak activation memory is proportional to `snn_chunk_size`, not total timesteps T.

**Setup**: Run SNN with T=50, chunk_size=1 vs T=50 without checkpointing.

**Expected**: Checkpointed version uses significantly less memory (requires CUDA).

---

### 4.3 Chunk Size > 1 Works

**Objective**: Grouping multiple timesteps per checkpoint segment works correctly.

**Setup**: T=20, chunk_size=5 (4 segments).

**Expected**: Output matches non-checkpointed. Memory is between chunk_size=1 and no checkpointing.

---

### 4.4 Chunk Size = T is Same as No Checkpointing

**Objective**: When chunk_size equals total timesteps, no recomputation occurs.

**Expected**: Output and memory are identical to non-checkpointed.

---

### 4.5 Spike State Preservation

**Objective**: SNN membrane potentials and spike states are correctly carried across checkpoint boundaries.

**Expected**: Final membrane potentials match non-checkpointed version.

---

## Category 5: CheckpointConfig

### 5.1 Default Config Is Valid

**Expected**: `CheckpointConfig()` constructs without error. `enabled=False`, `strategy="selective"`.

---

### 5.2 Invalid Strategy Raises

**Expected**: `CheckpointConfig(strategy="magic")` raises `ValueError` with "strategy" in message.

---

### 5.3 Negative Threshold Raises

**Expected**: `CheckpointConfig(memory_threshold_mb=-1.0)` raises `ValueError`.

---

### 5.4 Invalid SNN Chunk Size Raises

**Expected**: `CheckpointConfig(snn_chunk_size=0)` raises `ValueError`.

---

### 5.5 to_dict / from_dict Roundtrip

**Expected**: `CheckpointConfig.from_dict(cfg.to_dict())` produces an equivalent config.

---

### 5.6 All Valid Strategies Accepted

**Expected**: `CheckpointConfig(strategy=s)` constructs for `s` in `{"none", "full", "selective", "sequential"}`.

---

### 5.7 copy_with Works

**Expected**: `cfg.copy_with(enabled=True)` returns a new config with `enabled=True` and all other fields unchanged.

---

## Category 6: Integration Tests

### 6.1 Checkpoint + torch.compile

**Objective**: A checkpointed model can be compiled with `torch.compile`.

**Setup**: `CheckpointWrapper(model, use_reentrant=False)` then `torch.compile(model)`.

**Expected**: Forward and backward complete. Gradients are correct.

---

### 6.2 Checkpoint + Mixed Precision (AMP)

**Objective**: Checkpointing works correctly inside `torch.cuda.amp.autocast`.

**Expected**: Forward produces fp16/bf16 output. Backward produces correct gradients.

---

### 6.3 Checkpoint + Gradient Accumulation

**Objective**: Accumulated gradients over K micro-batches with checkpointing match non-checkpointed accumulation.

**Expected**: Gradients match within tolerance after K accumulation steps.

---

### 6.4 Checkpoint + DataParallel

**Objective**: Checkpointing works with `nn.DataParallel` (single-machine multi-GPU).

**Expected**: Forward and backward complete on all GPUs.

**Implementation notes**: Requires multi-GPU. Skip if `torch.cuda.device_count() < 2`.

---

## Category 7: Edge Cases

### 7.1 Empty Model (No Layers)

**Expected**: `SelectiveCheckpointer.apply()` returns the model unchanged. No error.

---

### 7.2 Single-Layer Model

**Expected**: Full checkpointing wraps the single layer. Memory savings are minimal but no errors.

---

### 7.3 Very Deep Model (100+ Layers)

**Objective**: Checkpointing scales to deep models without recursion or stack issues.

**Expected**: Forward-backward completes. Memory savings are substantial.

---

### 7.4 Model with Shared Parameters

**Objective**: Checkpointing handles tied weights (e.g., embedding + output projection sharing).

**Expected**: Gradients for shared parameters are accumulated correctly.

---

### 7.5 Model with In-Place Operations

**Objective**: Checkpointing handles or rejects models with in-place operations.

**Expected**: If in-place ops exist and `use_reentrant=False`, PyTorch raises a clear error. Wrapper should log this.

---

### 7.6 Batch Size 1

**Expected**: Checkpointing works correctly with single-sample batches.

---

## Test Implementation Notes

1. **Skip CUDA tests on CPU-only machines**: Use `@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")`.

2. **Use `module.train(False)` not `.eval()`**: The skill convention is to use `module.train(False)` for setting evaluation mode in tests.

3. **Reset memory stats between tests**: `torch.cuda.reset_peak_memory_stats()` and `torch.cuda.empty_cache()` in test setup.

4. **Deterministic inputs**: Fix `torch.manual_seed(42)` for reproducible results.

5. **Numerical tolerance**: Use `atol=1e-5, rtol=1e-5` for float32. Use `atol=1e-2, rtol=1e-2` for bfloat16/float16.

6. **Memory measurement noise**: Allow 10% variance in memory measurements. CUDA allocator caching introduces variability.
