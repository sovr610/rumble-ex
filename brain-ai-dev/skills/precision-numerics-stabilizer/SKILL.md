---
name: Precision + Numerics Stabilizer (bf16/fp16 Done Right)
description: >
  This skill should be used when the user asks to "add mixed precision training",
  "implement AMP precision modes", "set up bf16 or fp16 training",
  "add GradScaler with overflow detection", "detect NaN in training",
  "monitor gradient norms per layer", "add numerics sentinels",
  "check activations for Inf", "logit overflow monitoring",
  "debug NaN failure with snapshot", "training NaN debug snapshot",
  "reproducible NaN failure", "auto-abort on persistent NaN",
  "log step-skip rate for fp16", "effective update rate monitoring",
  "precision config fp32 bf16 fp16", "loss spike detector",
  "gradient clipping with scaler unscale", "activation NaN hook",
  "numerics instrumentation", "silent degradation detection",
  "PrecisionContext autocast wrapper", "scaler overflow logging",
  or needs guidance on making mixed-precision training safe to run
  unattended with automatic NaN detection, debug snapshots, and
  stabilization controls.
version: 0.1.0
---

# Precision + Numerics Stabilizer (bf16/fp16 Done Right)

## Overview

Add a precision layer (fp32/bf16/fp16) plus numerics instrumentation that makes mixed-precision training safe to run unattended. The goal is not "never crash" — it's **"never silently degrade."** When things go wrong (NaNs, Infs, overflows, exploding logits), either (a) auto-recover with controlled step-skips or (b) abort quickly with a debug snapshot that reproduces the failure.

Design principle: **detect leading indicators, log everything, snapshot on failure, abort before corruption.**

## Failure Cascade Model

Mixed-precision failures are progressive, not sudden:

```
logits drift upward → softmax overflows → loss becomes NaN
→ gradients become NaN → weights become NaN (irrecoverable)
```

The sentinel system monitors **leading indicators** (logit max abs, grad norm top-k, scaler skip rate) to detect and respond before the cascade reaches weights.

## Public Contract

### PrecisionContext

Single wrapper for all precision operations. Eliminates ordering mistakes.

```python
class PrecisionContext:
    def __init__(self, cfg: PrecisionConfig): ...
    def autocast_ctx(self) -> ContextManager: ...
    def backward(self, loss: Tensor) -> None: ...
    def unscale_and_clip(self, optimizer, params, max_norm: float) -> float: ...
    def optimizer_step(self, optimizer) -> bool: ...    # returns True if stepped
    def state_dict(self) -> Dict: ...
    def load_state_dict(self, state: Dict) -> None: ...
```

### NumericsMonitor (Sentinels)

Sampled, low-overhead checks with configurable cadence.

```python
class NumericsMonitor:
    def __init__(self, cfg: SentinelConfig, model: nn.Module): ...
    def check_grad_norms(self) -> GradNormReport: ...
    def check_activations(self) -> ActivationReport: ...
    def check_logits(self, logits: Tensor) -> LogitReport: ...
    def check_weights(self) -> WeightReport: ...
    def should_check(self, step: int) -> bool: ...
    def aggregate_report(self, step: int) -> NumericsReport: ...
```

### FailureSnapshot

Debug snapshot writer for reproducible failure investigation.

```python
class FailureSnapshot:
    def __init__(self, cfg: FailureConfig): ...
    def capture(self, step: int, model, optimizer, batch, report) -> str: ...
    def should_abort(self, nan_streak: int) -> bool: ...
```

## Key Concepts

### Precision Modes

| Mode | autocast dtype | GradScaler | Master weights | When to use |
|------|---------------|------------|----------------|-------------|
| `fp32` | disabled | disabled | fp32 | Debug, baseline, CPU-only |
| `bf16` | `torch.bfloat16` | **disabled** | fp32 | Ampere+ GPUs (default) |
| `fp16` | `torch.float16` | **enabled** | fp32 | Pre-Ampere GPUs, TPUs |

**bf16 does NOT need GradScaler** — its 8-bit exponent matches fp32's range, virtually eliminating underflow. fp16's 5-bit exponent underflows below ~6e-5, requiring loss scaling.

Use `torch.amp.autocast('cuda', dtype=...)` (not the deprecated `torch.cuda.amp.autocast`).

### GradScaler Internals (fp16 only)

Scale factor starts at 2^16 = 65536. Dynamics:
- Every 2000 consecutive clean steps: scale *= `growth_factor` (2.0)
- On any inf/nan detection: scale *= `backoff_factor` (0.5), step is **skipped**

**Skip detection**: record `scale_before = scaler.get_scale()` before `update()`. If `scale_after < scale_before`, overflow occurred and optimizer step was skipped.

Track counters: `num_steps_total`, `num_steps_skipped`, `effective_update_rate = 1 - skipped/total`.

**Critical invariant**: if skip rate exceeds 5% over a window, emit high-priority warning — the model is "training" but rarely updating.

### Numerics Sentinels

Low-overhead checks at configurable cadence (default: every 50 steps).

**C1: Gradient Norms** — After backward (before optimizer step):
- `grad_norm_global`: L2 norm across all parameters
- `grad_norm_by_module_topk`: per-module aggregated norms, top 20
- `grad_norm_max_param`: single parameter with highest norm

**C2: Activation/Weight NaN/Inf** — Forward hooks on sampled layers:
- Check `torch.isfinite(tensor).all()` on embeddings, attention proj, MLP, output head
- Rotate through block indices: check blocks [0, mid, last] each interval
- On any NaN/Inf: trigger immediate full scan + snapshot

**C3: Logit Scale** — Monitor `logits_max_abs` and `logits_std`:
- fp16 danger zone: max|logit| > ~65 (exp overflows at ~88 in fp16)
- Configurable threshold (default 80), alert after K consecutive violations

### Stabilization Controls

**D1: Gradient Clipping** — `scaler.unscale_(optimizer)` → `clip_grad_norm_(params, max_norm)` → `scaler.step()`. The unscale MUST precede clip, otherwise the norm is inflated by the scale factor.

**D2: Loss Spike Detector** — If loss increases > X% vs rolling median over a window, trigger additional sentinel checks + optional snapshot.

**D3: Deterministic Debug Toggle** — On failure reproduction runs, enable `torch.use_deterministic_algorithms(True)`, `torch.backends.cudnn.benchmark = False`, and fixed seeds.

### Failure Snapshot

When NaNs persist for M steps (default 3), auto-abort with a debug snapshot:

**Snapshot contents** (written to `runs/<id>/numerics/snapshot_<step>/`):
1. `config.json` — full resolved config
2. `env.json` — git sha, torch/cuda versions, GPU info
3. `rng_state.pt` — `torch.random.get_rng_state()`, `torch.cuda.get_rng_state_all()`, `random.getstate()`, numpy RNG
4. `batch.pt` — actual batch tensors (input_ids, labels, masks) + dataset indices
5. `model_state.pt` — full state dict (or offending module + checkpoint pointer if too large)
6. `optimizer_state.pt` — optimizer state dict (best effort)
7. `numerics.json` — latest sentinel metrics, first non-finite tensor name, scaler state, skip counters

**Abort behavior**: print concise failure summary (first detection step, offending tensor/module, scaler trend, recent grad norms), exit with non-zero code.

## Configuration Surface

```python
@dataclass
class PrecisionConfig:
    mode: str = "bf16"                              # fp32 | bf16 | fp16
    autocast_dtype: Optional[str] = None            # derived from mode if None
    grad_scaler_enabled: Optional[bool] = None      # derived: True only for fp16
    grad_scaler_init_scale: float = 65536.0         # 2**16
    grad_scaler_growth_factor: float = 2.0
    grad_scaler_backoff_factor: float = 0.5
    grad_scaler_growth_interval: int = 2000
    max_grad_norm: float = 1.0

@dataclass
class SentinelConfig:
    every_n_steps: int = 50
    grad_norm_topk: int = 20
    nan_check_sample_layers: Tuple[str, ...] = (
        "embeddings", "blocks.*.attn", "blocks.*.mlp", "lm_head"
    )
    logit_max_abs_threshold: float = 80.0
    logit_alert_consecutive: int = 3
    loss_spike_pct: float = 200.0                   # % increase vs rolling median
    loss_spike_window: int = 100

@dataclass
class FailureConfig:
    nan_persist_steps: int = 3
    snapshot_dir: str = "runs/{run_id}/numerics"
    on_error: str = "abort"                         # abort | raise
    deterministic_debug: bool = False
```

## Done-When Gates

1. **Mode Correctness** — fp32 runs with no autocast/scaler. bf16 uses autocast(bfloat16) without GradScaler. fp16 uses autocast(float16) with GradScaler. No silent casts outside autocast regions.
2. **No Silent Failure** — Any non-finite loss/activation/weight triggers a log event with tensor name and a snapshot to disk. Persistent NaNs for M steps abort with snapshot.
3. **Actionable Artifacts** — Snapshot folder contains config, env, rng, batch, model state, optimizer state, numerics report. All files load without error.
4. **Measurable Stability** — Logs include step-skips due to overflow, effective update rate, grad norm global + top-k, logit max abs. Skip rate > 5% triggers warning.

## Resources

### Reference Files
- **`references/precision-modes.md`** — fp32/bf16/fp16 mode semantics, autocast operation policies (which ops cast to lower precision vs stay fp32), master weight behavior, GradScaler necessity per mode, mode validation logic
- **`references/gradscaler-internals.md`** — Scale factor dynamics, growth/backoff mechanism, _found_inf_per_device, skip detection via get_scale(), overflow counters, effective update rate, RuntimeError on double unscale_
- **`references/sentinel-system.md`** — Gradient norm computation (global, per-module, top-k), activation hook registration, NaN/Inf check patterns, logit monitoring, loss spike detection, sampling cadence, block rotation strategy
- **`references/failure-snapshot.md`** — Snapshot schema (7 files), RNG capture (torch/cuda/python/numpy), atomic write, abort vs raise behavior, deterministic debug toggle, repro script generation
- **`references/stabilization-controls.md`** — Gradient clipping with AMP (unscale → clip → step ordering), loss spike response, logit clamping, auto-recovery patterns, determinism flags
- **`references/testing-matrix.md`** — Test scenarios for all components across 7 phases

### Asset Files
- **`assets/precision_config_template.py`** — All config dataclasses (PrecisionConfig, SentinelConfig, FailureConfig), mode derivation, validation, serialization
- **`assets/precision_context_template.py`** — PrecisionContext with autocast_ctx, backward, unscale_and_clip, optimizer_step, skip detection, state_dict
- **`assets/numerics_monitor_template.py`** — NumericsMonitor with grad norms, activation hooks, weight checks, logit monitoring, report aggregation
- **`assets/failure_snapshot_template.py`** — FailureSnapshot with 7-file capture, atomic write, NaN streak tracking, abort logic
- **`assets/loss_spike_detector_template.py`** — LossSpikeDetector with rolling median, spike detection, alert triggering
- **`assets/stabilization_template.py`** — Gradient clipping integration, logit clamping, deterministic debug toggle

### Scripts
- **`scripts/validate_precision.py`** — Validates done-when gates (mode correctness, NaN detection, snapshot contents, stability metrics)
- **`scripts/gen_precision_tests.py`** — Generates pytest test cases covering all 7 phases
- **`scripts/numerics_diagnostic.py`** — Runs N steps, prints sentinel reports, simulates NaN injection, verifies snapshot capture
