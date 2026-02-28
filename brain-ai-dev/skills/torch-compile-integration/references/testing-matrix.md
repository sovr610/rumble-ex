# torch.compile Testing Matrix

## Overview

This matrix covers all test scenarios for the torch.compile integration. Tests are organized into categories from basic safety through advanced distributed scenarios. Each scenario includes the test objective, inputs, expected behavior, and implementation notes.

---

## Category 1: Safe Fallback

### 1.1 Broken Model Falls Back to Eager

**Objective**: Verify `maybe_compile` catches compilation failures and returns the original model.

**Setup**:
```python
class BrokenModel(nn.Module):
    def forward(self, x):
        raise RuntimeError("Intentional compile-time failure")
```

**Expected**: `maybe_compile` returns the original model instance (same `id()`), logs the exception with full stack trace, does not raise.

**Implementation notes**: The "broken model" for compile testing should use an unsupported op, not a runtime error. Use a model that contains a non-compilable construct (e.g., in-place mutation of an input tensor that Inductor rejects). Alternatively, mock `torch.compile` to raise.

---

### 1.2 Smoketest Failure Falls Back

**Objective**: When `healthcheck=True` and the smoketest fails (backward produces NaN or raises), `maybe_compile` falls back to eager.

**Setup**:
```python
class NaNLossModel(nn.Module):
    def forward(self, x):
        return torch.tensor(float('nan')).expand(x.shape[0], 10)
```

**Expected**: `CompileSmoketest.run()` returns `False`. `maybe_compile` with `healthcheck=True` returns eager model.

---

### 1.3 fail_policy="raise" Re-raises

**Objective**: When `fail_policy="raise"`, compile failures propagate to caller.

**Expected**: `maybe_compile` raises the original exception when `fail_policy="raise"` and compilation fails.

---

### 1.4 Disabled Config Returns Same Model

**Objective**: When `cfg.enabled=False`, `maybe_compile` is a no-op.

**Expected**: Returns exactly the same model object (`model is result` is True). No logging of compilation activity.

---

## Category 2: Successful Compilation (Smoketest)

### 2.1 Simple Model Compiles and Runs

**Objective**: Basic end-to-end compile + forward + backward succeeds.

**Setup**: Simple `nn.Linear` or 2-layer MLP.

**Expected**: `maybe_compile` returns a compiled model (not the original). Forward produces output with correct shape. Backward completes without error.

---

### 2.2 CompileSmoketest Returns True on Good Model

**Objective**: `CompileSmoketest.run()` returns `True` when all steps succeed.

**Expected**: Returns `True`. All `steps` training iterations complete. Per-step times are logged.

---

### 2.3 CompileSmoketest Returns False on NaN Loss

**Objective**: Detect NaN loss during smoketest.

**Expected**: Returns `False`. Exception or NaN detection is logged.

---

### 2.4 CompileSmoketest Returns False on Backward Failure

**Objective**: Detect backward-time errors.

**Setup**: Model with a custom autograd function that raises during backward.

**Expected**: Returns `False`. Exception logged with traceback.

---

### 2.5 Compile Time Is Logged

**Objective**: First-step compile time is measured and logged.

**Expected**: Log contains `compile_time_s` field with a positive float value.

---

## Category 3: Shape Stability

### 3.1 Bucketing Produces Correct Bucket Sizes

**Objective**: `ShapeStabilizer.bucket_batch()` pads to correct ceiling bucket.

**Test cases**:
```
Input seq_len=100  → bucket=256  → output.shape[1]==256
Input seq_len=256  → bucket=256  → output.shape[1]==256
Input seq_len=257  → bucket=512  → output.shape[1]==512
Input seq_len=512  → bucket=512  → output.shape[1]==512
Input seq_len=513  → bucket=1024 → output.shape[1]==1024
Input seq_len=1024 → bucket=1024 → output.shape[1]==1024
Input seq_len=1025 → bucket=2048 → output.shape[1]==2048
Input seq_len=2048 → bucket=2048 → output.shape[1]==2048
```

---

### 3.2 Padding Values Are Correct

**Objective**: Padded positions contain `pad_token_id`.

**Expected**: `padded[i, original_len:]` is all `pad_token_id` for each sequence `i`.

---

### 3.3 Unpad Recovers Original Sequences

**Objective**: `unpad_batch()` removes padding and returns original content.

**Expected**: `unpad_batch(padded, original_lengths)[i]` equals `original[i]` for all `i`.

---

### 3.4 All Bucket Boundaries Produce Exact Pad Target

**Objective**: Sequences at exact bucket boundaries pad to themselves (no excess padding).

**Expected**: `bucket_batch(input_ids_512)` has shape `[B, 512]` when all sequences are length 512.

---

### 3.5 No Recompile Across Buckets

**Objective**: Running compiled model through all 4 bucket sizes triggers exactly 4 compilations (one per bucket), not one per batch.

**Implementation notes**: Use `torch._dynamo.reset()` before test. Count `torch._dynamo.utils.counters['stats']['unique_graphs']` after running through all buckets twice. Should be 4 (one per bucket shape), not 8.

---

### 3.6 mark_dynamic Prevents Recompile on Varying seq_len

**Objective**: Using `mark_dynamic` on dim 1 allows varying lengths without recompile.

**Expected**: After marking dim 1 as dynamic, running with lengths 100, 200, 300 triggers only 1 compilation (the dynamic version), not 3.

---

## Category 4: Mode Switching

### 4.1 default Mode Produces Valid Output

**Objective**: Compilation with `mode="default"` produces numerically close output to eager.

**Expected**: `torch.allclose(eager_out, compiled_out, atol=1e-4)` is True.

---

### 4.2 reduce-overhead Mode Produces Valid Output

**Expected**: Same as 4.1 but with `mode="reduce-overhead"`. Note: requires fixed shapes.

---

### 4.3 max-autotune Mode Produces Valid Output

**Expected**: Same as 4.1 but with `mode="max-autotune"`. Note: slow test, may need `@pytest.mark.slow`.

---

### 4.4 max-autotune-no-cudagraphs Mode Works

**Expected**: Compiles and runs successfully with `mode="max-autotune-no-cudagraphs"`.

---

### 4.5 All Modes Produce Consistent Output

**Objective**: Output from all modes is numerically close to each other and to eager.

**Expected**: Max pairwise difference across modes is within float32 tolerance.

---

## Category 5: Allowlist and Blocklist

### 5.1 Allowlist Compiles Only Listed Modules

**Objective**: When `allowlist=["attention", "mlp"]`, only those submodules are compiled.

**Expected**: Submodules whose names match allowlist patterns are `torch.compile` wrapped. Others are not. Model forward still works end-to-end.

---

### 5.2 Blocklist Prevents Compilation of Target Modules

**Objective**: When `blocklist=["sampling"]`, matching modules have `torch.compiler.disable` applied.

**Expected**: `sampling` submodule runs in eager. Parent module's compilation continues. No graph break from the disabled submodule.

---

### 5.3 Empty Allowlist Compiles Whole Model

**Objective**: When `allowlist=[]` (empty), the whole model is compiled (no filtering).

**Expected**: Single `torch.compile` call on the full model.

---

### 5.4 Allowlist + Blocklist: Blocklist Wins for Overlap

**Objective**: If a module name matches both allowlist and blocklist, blocklist wins (module stays eager).

**Expected**: The overlapping module is NOT compiled.

---

## Category 6: Config Validation

### 6.1 Invalid Mode Raises

**Expected**: `CompileConfig(mode="super-fast")` raises `ValueError` with message containing "mode".

---

### 6.2 Invalid Backend Raises

**Expected**: `CompileConfig(backend="magic")` raises `ValueError` with message containing "backend".

---

### 6.3 Invalid fail_policy Raises

**Expected**: `CompileConfig(fail_policy="ignore_silently")` raises `ValueError`.

---

### 6.4 smoketest_steps <= 0 Raises

**Expected**: `CompileConfig(smoketest_steps=0)` raises `ValueError`.

---

### 6.5 Defaults Are Valid

**Expected**: `CompileConfig()` constructs without error. All fields have documented defaults.

---

### 6.6 from_dict Roundtrip

**Expected**: `CompileConfig.from_dict(cfg.to_dict()) == cfg` is True.

---

### 6.7 YAML Serialization Roundtrip

**Expected**: `CompileConfig.from_yaml(path)` round-trips through `to_dict()` correctly.

---

## Category 7: Benchmark Integration

### 7.1 Eager Mode Benchmark Completes

**Objective**: `CompileBenchmark.run_comparison()` with `modes=["eager"]` completes and returns expected keys.

**Expected**: Result dict contains `eager` key with sub-keys: `compile_time_s`, `step_times_s`, `peak_memory_bytes`.

---

### 7.2 Comparison Produces All Expected Keys

**Expected**: Result for each mode contains `compile_time_s`, `steady_state_tokens_per_sec_p50`, `peak_memory_bytes`, `speedup_vs_eager`.

---

### 7.3 Speedup Ratio Is Reasonable

**Objective**: Compiled mode speedup vs eager is between 0.5x and 10x.

**Rationale**: Values outside this range indicate a measurement error (not a real result).

---

### 7.4 format_report Returns Non-Empty String

**Expected**: `format_report(results)` returns a multi-line string with mode names and numeric values.

---

### 7.5 save_report Writes Valid JSON

**Expected**: File written by `save_report(results, path)` can be loaded with `json.load()` and contains all mode keys.

---

## Category 8: Edge Cases

### 8.1 Empty Batch Handling

**Setup**: `input_ids` of shape `[0, 128]`.

**Expected**: `bucket_batch` raises `ValueError` or returns gracefully (not an index error). Compiled model handles empty batch without crashing.

---

### 8.2 Single Token Sequence

**Setup**: `input_ids` of shape `[1, 1]`.

**Expected**: Bucketed to smallest bucket (e.g., 256). Compiled forward produces output of correct shape.

---

### 8.3 Maximum Sequence Length

**Setup**: `input_ids` at exactly the largest bucket boundary (e.g., 2048).

**Expected**: No recompile. Bucketed to 2048.

---

### 8.4 Sequence Longer Than Largest Bucket

**Setup**: `input_ids` of length 3000 with buckets `[256, 512, 1024, 2048]`.

**Expected**: Raises `ValueError` or extends to next valid bucket (implementation-defined, but must not silently truncate).

---

### 8.5 Single-Sample Batch

**Setup**: `input_ids` of shape `[1, 512]`.

**Expected**: Bucketing works. Compilation completes. Output shape is `[1, ...]`.

---

### 8.6 Very Small Model (1 Layer)

**Expected**: Compilation completes. Forward and backward work correctly.

---

## Category 9: Graph Break Detection

### 9.1 fullgraph=True Raises on Known Break

**Setup**: Model with `print(x.shape)` in forward.

**Expected**: Compilation with `fullgraph=True` raises `torch._dynamo.exc.Unsupported` or similar.

---

### 9.2 fullgraph=False Falls Through Graph Break

**Setup**: Same model as 9.1.

**Expected**: Compilation succeeds (with graph break). Forward produces correct output (possibly with performance penalty).

---

### 9.3 torch.compiler.disable Prevents Break Propagation

**Setup**: Model where a problematic submodule is decorated with `@torch.compiler.disable`.

**Expected**: Parent compilation succeeds. The disabled submodule runs in eager within the compiled parent.

---

## Category 10: DDP Integration (Distributed)

### 10.1 DDP + compile Forward-Backward Works

**Objective**: Model compiled before DDP wrapping runs complete forward-backward cycles.

**Expected**: Gradients are non-None on all parameters after backward. Gradient allreduce happens correctly.

---

### 10.2 Compile Before DDP vs After DDP

**Objective**: Demonstrate that compile-before-DDP works better than DDP-before-compile.

**Expected**: compile-before-DDP succeeds smoketest; DDP-before-compile may have more graph breaks (measured via `TORCH_LOGS=graph_breaks`).

---

### 10.3 Selective Block Compilation with DDP

**Objective**: Compiling only transformer blocks (not the full DDP-wrapped model) works end-to-end.

**Expected**: Block-level compiled model wrapped in DDP passes smoketest.

---

## Test Implementation Notes

1. **Skip GPU tests on CPU-only machines**: Use `@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")`.

2. **Skip distributed tests without multi-GPU**: Use `@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Multi-GPU required")`.

3. **Mark slow tests**: `max-autotune` tests can take minutes. Mark with `@pytest.mark.slow` and exclude from CI fast path.

4. **Reset Dynamo state between tests**: `torch._dynamo.reset()` in test teardown to avoid inter-test pollution.

5. **Use deterministic inputs**: Fix `torch.manual_seed(42)` for reproducible test results.

6. **Numerical tolerance**: Use `atol=1e-3, rtol=1e-3` for compiled vs eager comparison (bfloat16 precision).
