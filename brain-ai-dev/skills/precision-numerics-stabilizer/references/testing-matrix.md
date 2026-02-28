# Testing Matrix Reference

## Overview

60+ test cases organized across 7 phases. Each phase targets a distinct component. All tests use small (2-layer Linear) models and CPU-friendly configurations unless GPU testing is explicitly required. GPU tests are marked with `@pytest.mark.cuda`.

---

## Phase 1: Mode Correctness (8 tests)

Tests that precision mode configuration is applied correctly — no silent casts, no unnecessary components, correct derivation.

### P1-T1: fp32 No Autocast
**What**: Verify that `PrecisionConfig(mode="fp32")` produces a `PrecisionContext` that never enters autocast.
**How**: Wrap a forward pass in `precision_ctx.autocast_ctx()`. Inspect the context manager type — it must be `contextlib.nullcontext` or equivalent. Run a Linear on float32 input, confirm output dtype is float32.
**Pass**: No autocast context entered, output tensor dtype == torch.float32.

### P1-T2: bf16 Autocast Bfloat16
**What**: Verify bf16 mode enters `torch.amp.autocast` with dtype=bfloat16.
**How**: On CUDA, wrap forward pass in `precision_ctx.autocast_ctx()`. Run matmul inside. Confirm output dtype is torch.bfloat16.
**Pass**: Output tensor dtype == torch.bfloat16 inside autocast.
**Skip**: if not `torch.cuda.is_available()`.

### P1-T3: fp16 Autocast Float16
**What**: Verify fp16 mode enters autocast with dtype=float16.
**How**: Same as T2 but with mode="fp16". Confirm output dtype is torch.float16.
**Pass**: Output tensor dtype == torch.float16 inside autocast.

### P1-T4: bf16 No GradScaler
**What**: Verify bf16 mode never constructs or uses GradScaler.
**How**: `cfg = PrecisionConfig(mode="bf16")`. Check `cfg.grad_scaler_enabled == False`. Check `precision_ctx.scaler._enabled == False`. Run full step, confirm no scale factor applied to loss.
**Pass**: `precision_ctx.scaler` has `_enabled=False` or equivalent disabled state.

### P1-T5: fp16 GradScaler Created and Enabled
**What**: Verify fp16 mode has a functioning GradScaler.
**How**: `cfg = PrecisionConfig(mode="fp16")`. Check `cfg.grad_scaler_enabled == True`. Check `precision_ctx.scaler.get_scale() == cfg.grad_scaler_init_scale`.
**Pass**: `precision_ctx.scaler.get_scale()` returns 65536.0 (or configured init_scale).

### P1-T6: Mode Derivation — String to Enum
**What**: Verify `mode="fp32"|"bf16"|"fp16"` all produce valid configs without errors.
**How**: Construct all three configs, call `validate()` on each. No exception raised.
**Pass**: All three configs validate without error.

### P1-T7: autocast_dtype Derivation
**What**: Verify that `autocast_dtype=None` is correctly derived from mode.
**How**: Create configs with `autocast_dtype=None` for each mode. Confirm derived dtype matches expected:
  - fp32 → None (disabled)
  - bf16 → torch.bfloat16
  - fp16 → torch.float16
**Pass**: `resolve_autocast_dtype("bf16") == torch.bfloat16`, etc.

### P1-T8: grad_scaler_enabled Derivation
**What**: Verify `grad_scaler_enabled=None` is correctly derived.
**How**: `resolve_scaler_enabled("fp16") == True`, `resolve_scaler_enabled("bf16") == False`, `resolve_scaler_enabled("fp32") == False`.
**Pass**: All three derivations match expected values.

---

## Phase 2: GradScaler Mechanics (8 tests)

### P2-T1: init_scale Correct
**What**: GradScaler initializes at configured scale.
**How**: Create `PrecisionContext(PrecisionConfig(mode="fp16", grad_scaler_init_scale=32768.0))`. Check `scaler.get_scale() == 32768.0`.
**Pass**: Scale equals init_scale immediately after construction.

### P2-T2: Scale Grows After N Clean Steps
**What**: After `growth_interval` consecutive clean steps, scale doubles.
**How**: Create scaler with `growth_interval=3` (reduced for test speed). Run 3 synthetic clean `update()` calls. Check scale doubled.
**Pass**: `scale == init_scale * growth_factor` after `growth_interval` clean updates.

### P2-T3: Scale Halves on Overflow
**What**: When inf is injected, scale is halved.
**How**: Create GradScaler. Create a dummy optimizer with a parameter. Inject inf into parameter.grad. Call `unscale_()` + `step()` + `update()`. Check `scale == init_scale * backoff_factor`.
**Pass**: Scale is halved after overflow detection.

### P2-T4: Optimizer Step Skipped on Inf
**What**: When inf is present, `optimizer.step()` is not called.
**How**: Use a spy/mock on `optimizer.step`. Inject inf into gradient. Call full scaler workflow. Confirm `optimizer.step()` was not called.
**Pass**: Optimizer's step method call count remains 0 after overflow.

### P2-T5: Skip Detection via get_scale()
**What**: `optimizer_step()` returns False when overflow occurred.
**How**: Use `PrecisionContext.optimizer_step()`. Inject inf into grad before calling. Check return value is `False`.
**Pass**: `precision_ctx.optimizer_step(optimizer)` returns `False` after inf injection.

### P2-T6: Overflow Counter Increments
**What**: `num_steps_skipped` increments on each overflow.
**How**: Call `optimizer_step()` with inf three times. Check `num_steps_skipped == 3`, `num_steps_total == 3`.
**Pass**: Counters correct after 3 overflows.

### P2-T7: Double unscale_ Raises RuntimeError
**What**: Calling `scaler.unscale_(optimizer)` twice raises RuntimeError.
**How**: Call `scaler.unscale_(optimizer)` once (clean), then call again before `step()`. Wrap in `pytest.raises(RuntimeError)`.
**Pass**: RuntimeError raised on second unscale_ call.

### P2-T8: Gradient Accumulation — Only Unscale After Last Micro-batch
**What**: Accumulated gradients are correctly handled with single unscale.
**How**: Run 4 micro-batches with `scaler.scale(loss / 4).backward()` each. Call `unscale_()` once after all 4. Call `step()` + `update()`. Confirm no RuntimeError and step occurred.
**Pass**: No RuntimeError, optimizer stepped after accumulation.

---

## Phase 3: Sentinels (10 tests)

### P3-T1: Global Grad Norm Computation
**What**: `check_grad_norms()` returns correct global L2 norm.
**How**: Create a 2-layer Linear. Set `.grad` to known tensors. Compute expected norm manually. Call `monitor.check_grad_norms()`. Compare.
**Pass**: `report.global_norm` within 1e-5 of expected.

### P3-T2: Per-Module Top-k
**What**: Per-module norms are correctly grouped and sorted.
**How**: Model with named modules "layer1" and "layer2". Set known grads. Check `report.per_module_topk[0][0]` is the module with highest norm.
**Pass**: Top module name matches expected, norm value correct.

### P3-T3: Activation Hook Fires
**What**: Forward hook captures activation tensors after forward pass.
**How**: Register hooks on "fc1" pattern. Run forward pass. Check `monitor._capture.outputs` contains "fc1" key with correct shape tensor.
**Pass**: Hook output shape matches expected linear output shape.

### P3-T4: NaN Detection in Activations
**What**: NaN in activation is detected by `check_activations()`.
**How**: Register hook. Run forward pass that produces NaN output (inject NaN into weights temporarily). Call `check_activations()`. Check `report.any_nonfinite == True`.
**Pass**: `any_nonfinite=True`, `first_nonfinite_name` is set.

### P3-T5: Weight Finite Check Detects Injected NaN
**What**: `check_weights()` detects NaN in a parameter.
**How**: Set one parameter to `torch.full_like(p, float('nan'))`. Call `monitor.check_weights()`. Check `report.all_finite == False`.
**Pass**: `all_finite=False`, `first_nonfinite_name` is the injected parameter.

### P3-T6: Logit Max Abs Correct
**What**: `check_logits(logits)` reports correct max_abs.
**How**: Create logits tensor with known max abs value. Call `check_logits()`. Compare `report.max_abs` to expected.
**Pass**: `report.max_abs` within 1e-5 of expected.

### P3-T7: Logit Consecutive Alert After K Violations
**What**: `consecutive_violations` increments each step above threshold, resets below.
**How**: Call `check_logits()` with `max_abs > threshold` three times. Check `report.consecutive_violations == 3`. Then call with `max_abs < threshold`. Check `consecutive_violations == 0`.
**Pass**: Counter increments and resets correctly.

### P3-T8: Cadence — every_n_steps
**What**: `should_check(step)` returns True only at configured interval.
**How**: `cfg.every_n_steps=10`. Check `should_check(0)==True`, `should_check(5)==False`, `should_check(10)==True`, `should_check(15)==False`.
**Pass**: should_check returns True iff step % every_n_steps == 0.

### P3-T9: Block Rotation
**What**: Sentinel samples blocks [0, mid, last] from module list.
**How**: Create model with 6 identical blocks named "block.0" through "block.5". Configure `nan_check_sample_layers=("block.*",)`. Confirm hooks registered only for indices 0, 2, 5 (or equivalent).
**Pass**: Only 3 hooks registered (or all 6 if fallback to all).

### P3-T10: Full Scan on Anomaly
**What**: When anomaly detected, full scan triggers immediately.
**How**: Mock `_run_full_scan`. Inject NaN in a hooked activation. Check mock was called.
**Pass**: `_run_full_scan` called exactly once when NaN detected.

---

## Phase 4: Failure Snapshot (8 tests)

### P4-T1: All 7 Files Written
**What**: `capture()` writes all 7 required files.
**How**: Create snapshot. Call `capture(step=5, model=..., optimizer=..., batch=..., report=...)`. List files in snapshot dir. Check all 7 exist.
**Pass**: All 7 filenames present in snapshot directory.

### P4-T2: Atomic Write
**What**: Snapshot uses temp-dir-then-rename pattern.
**How**: Patch `os.rename` to fail, call `capture()`. Confirm no partial snapshot at final path.
**Pass**: Final snapshot path does not exist after failed rename.

### P4-T3: NaN Streak Increments
**What**: `nan_steps_in_a_row` increments on consecutive NaN losses.
**How**: Call `snapshot.update_nan_streak(float('nan'), None)` three times. Check count == 3.
**Pass**: `nan_steps_in_a_row == 3`.

### P4-T4: NaN Streak Resets on Clean Step
**What**: Clean step resets streak to 0.
**How**: Increment streak to 2, then call with finite loss. Check count == 0.
**Pass**: `nan_steps_in_a_row == 0` after clean step.

### P4-T5: Abort on Persist (on_error="abort")
**What**: `abort()` calls `sys.exit(1)` when on_error="abort".
**How**: `cfg = FailureConfig(on_error="abort")`. Call `snapshot.abort(path, report)` in `pytest.raises(SystemExit)`.
**Pass**: `SystemExit` raised with code 1.

### P4-T6: Raise on Persist (on_error="raise")
**What**: `abort()` raises RuntimeError when on_error="raise".
**How**: `cfg = FailureConfig(on_error="raise")`. Call `snapshot.abort(path, report)` in `pytest.raises(RuntimeError)`.
**Pass**: `RuntimeError` raised.

### P4-T7: RNG State Captured
**What**: `rng_state.pt` contains torch, cuda, and python states.
**How**: Load saved `rng_state.pt`. Check keys: `torch_cpu`, `python`.
**Pass**: All expected keys present in loaded dict.

### P4-T8: Batch Captured and Loads Without Error
**What**: `batch.pt` saves and loads without error.
**How**: Save batch with tensor values. Load `batch.pt` with `map_location='cpu'`. Check tensor shapes match original.
**Pass**: Loaded batch matches original tensor shapes.

---

## Phase 5: Stabilization (6 tests)

### P5-T1: Clip Ordering — Unscale Before Clip
**What**: For fp16, unscale_ is called before clip_grad_norm_.
**How**: Track call order using a sequence recorder. Call `precision_ctx.unscale_and_clip()`. Assert unscale_ call index < clip_grad_norm_ call index.
**Pass**: unscale_ called before clip_grad_norm_.

### P5-T2: Loss Spike Detection
**What**: `is_spike()` returns True when loss exceeds median * (1 + spike_pct/100).
**How**: Fill window with losses of 1.0. Call `is_spike(4.0)` with spike_pct=200. Expected: 4.0 > 1.0 * 3.0 → True.
**Pass**: `is_spike(4.0)` returns True.

### P5-T3: Spike Threshold Correct
**What**: Value exactly at threshold does not trigger spike.
**How**: Fill window with 1.0 values. Call `is_spike(3.0)` with spike_pct=200. Expected: 3.0 == 3.0 * 1.0, boundary → not a spike (strictly greater).
**Pass**: `is_spike(3.0)` returns False (boundary is not a spike).

### P5-T4: Deterministic Toggle Sets and Restores Flags
**What**: `DeterministicDebugToggle.enable()` sets flags; `disable()` restores them.
**How**: Record original `torch.are_deterministic_algorithms_enabled()`. Call `enable()`. Check True. Call `disable()`. Check restored.
**Pass**: Flags match original after disable().

### P5-T5: Logit Clamping
**What**: `LogitClamper.clamp()` clips logits to [-threshold, threshold].
**How**: Create logits tensor with values [-200.0, 0.0, 200.0]. Clamp at threshold=80. Check output is [-80.0, 0.0, 80.0].
**Pass**: Output matches expected clamped values exactly.

### P5-T6: Auto-Recovery Reduces Then Restores LR
**What**: LR is reduced on overflow and restored after N steps.
**How**: Create optimizer with lr=0.01. Call `on_overflow()` (reduces to 0.005). Call `step()` N times. After N steps, lr restored to 0.01.
**Pass**: LR returns to 0.01 after `recovery_steps` steps.

---

## Phase 6: Logging (6 tests)

### P6-T1: All Metrics Logged Each Check Interval
**What**: NumericsMonitor emits log entries for all 4 sentinel types.
**How**: Run monitor on step where `should_check()` returns True. Capture log output. Check for: "GradNorm", "Activation", "Logit", "Weight" or equivalent keys.
**Pass**: Log output contains all metric categories.

### P6-T2: Skip Rate Computed Correctly
**What**: `precision_ctx.skip_rate` is accurate.
**How**: Call `optimizer_step()` 10 times, 3 with injected inf. Check `skip_rate == 0.3`.
**Pass**: `skip_rate == 0.3` within 1e-10.

### P6-T3: Effective Update Rate
**What**: `effective_update_rate == 1 - skip_rate`.
**How**: Same as P6-T2. Check `effective_update_rate == 0.7`.
**Pass**: `effective_update_rate == 0.7` within 1e-10.

### P6-T4: Warning on Skip Rate > 5%
**What**: High skip rate triggers warning log.
**How**: Inject inf to create skip_rate=0.06. Check warning logged with "skip rate" in message.
**Pass**: Warning message emitted.

### P6-T5: Logit Alert Logged
**What**: When `consecutive_violations >= logit_alert_consecutive`, alert is logged.
**How**: Call `check_logits()` with high logits `logit_alert_consecutive` times. Check alert log message.
**Pass**: Alert logged after K consecutive violations.

### P6-T6: Loss Spike Alert Logged
**What**: Spike detection triggers a warning log.
**How**: Fill buffer with baseline losses. Call `is_spike()` with large loss. Check warning logged.
**Pass**: Warning message with spike magnitude logged.

---

## Phase 7: Integration (4 tests)

### P7-T1: 20-Step bf16 Training Produces Finite Loss
**What**: End-to-end 20-step training loop with bf16 runs without NaN.
**How**: Small model (Linear(64, 64)), bf16 config, random fp32 input on CUDA, run 20 steps. Check all losses are finite.
**Pass**: All 20 losses are `math.isfinite(loss)`.
**Requires**: CUDA device.

### P7-T2: fp16 Handles Injected Inf Gracefully
**What**: Injecting inf into gradients causes step skip, not crash.
**How**: Run 10-step fp16 training loop. At step 5, manually set one parameter.grad to inf. Confirm step 5 was skipped (return value False) but step 6+ proceed normally.
**Pass**: `stepped==False` at step 5, `stepped==True` at steps 6+.

### P7-T3: Snapshot Captured on NaN Injection
**What**: Injecting NaN into loss triggers snapshot after nan_persist_steps.
**How**: Configure `nan_persist_steps=2`. Run training loop that returns NaN loss for 2 consecutive steps. Verify snapshot written to configured dir.
**Pass**: Snapshot directory exists and contains all 7 files.

### P7-T4: Mode Switch Produces Equivalent Results
**What**: fp32 and bf16 modes produce losses within tolerance after 1 step.
**How**: Initialize two identical models (same weights). Run 1 forward pass in fp32 and 1 in bf16. Compare losses. Tolerance: |loss_fp32 - loss_bf16| / loss_fp32 < 0.01 (1%).
**Pass**: Losses within 1% relative difference.
