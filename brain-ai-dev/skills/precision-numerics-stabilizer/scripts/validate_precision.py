"""
Precision + Numerics Stabilizer - Validation Script
====================================================
Validates all 4 done-when gates:
1. Mode Correctness: fp32/bf16/fp16 autocast and scaler behavior
2. No Silent Failure: NaN injection triggers detection + snapshot
3. Actionable Artifacts: snapshot contains all 7 files, each loads
4. Measurable Stability: logs include skip rate, grad norms, logit monitoring

Run with:
    python validate_precision.py
    python validate_precision.py --verbose

CRITICAL: Never call the inference-mode shorthand on PyTorch modules.
Use module.train(False) instead.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import tempfile
import time

# Add assets dir to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ASSETS_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "assets")
if _ASSETS_DIR not in sys.path:
    sys.path.insert(0, _ASSETS_DIR)

import torch
import torch.nn as nn

from precision_config_template import FailureConfig, FullConfig, PrecisionConfig, SentinelConfig
from precision_context_template import PrecisionContext
from numerics_monitor_template import NumericsMonitor, NumericsReport
from failure_snapshot_template import FailureSnapshot, _load_pt_file
from loss_spike_detector_template import LossSpikeDetector


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_tiny_model(in_dim: int = 16, out_dim: int = 8) -> nn.Module:
    """Return a simple 2-layer MLP for validation."""
    return nn.Sequential(
        nn.Linear(in_dim, in_dim),
        nn.ReLU(),
        nn.Linear(in_dim, out_dim),
    )


def _make_batch(batch_size: int = 4, in_dim: int = 16):
    """Return a minimal fake batch dict."""
    return {
        "input_ids": torch.randint(0, 100, (batch_size, in_dim)),
        "labels": torch.randint(0, 8, (batch_size,)),
        "input_tensor": torch.randn(batch_size, in_dim),
    }


def _log_gate(name: str, passed: bool, detail: str = ""):
    icon = "PASS" if passed else "FAIL"
    line = f"  [{icon}] {name}"
    if detail:
        line += f" — {detail}"
    print(line)
    return passed


# ---------------------------------------------------------------------------
# Gate 1: Mode Correctness
# ---------------------------------------------------------------------------

def validate_mode_correctness(verbose: bool = False) -> bool:
    """Gate 1: Verify fp32/bf16/fp16 autocast and scaler behavior."""
    print("\nGate 1: Mode Correctness")
    results = []

    # --- fp32: no autocast, no scaler ---
    try:
        cfg = PrecisionConfig(mode="fp32")
        ctx = PrecisionContext(cfg)

        model = _make_tiny_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        x = torch.randn(2, 16)
        optimizer.zero_grad()
        with ctx.autocast_ctx() as ac:
            out = model(x)
        loss = out.sum()
        ctx.backward(loss)
        stepped = ctx.optimizer_step(optimizer)

        assert stepped is True
        assert ctx._scaler_enabled is False
        assert out.dtype == torch.float32
        results.append(_log_gate("fp32: no autocast, no scaler, output is float32", True))
    except Exception as e:
        results.append(_log_gate("fp32: mode correctness", False, str(e)))

    # --- bf16: autocast bfloat16, no scaler ---
    try:
        cfg = PrecisionConfig(mode="bf16")
        ctx = PrecisionContext(cfg)
        assert ctx._scaler_enabled is False
        assert ctx._dtype == torch.bfloat16

        if torch.cuda.is_available():
            model = _make_tiny_model().cuda()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            x = torch.randn(2, 16, device="cuda")
            optimizer.zero_grad()
            with ctx.autocast_ctx():
                out = model(x)
            loss = out.sum()
            ctx.backward(loss)
            stepped = ctx.optimizer_step(optimizer)
            assert stepped is True
            results.append(_log_gate("bf16: autocast bfloat16, no scaler, CUDA", True))
        else:
            results.append(_log_gate("bf16: scaler=False, dtype=bfloat16 (no CUDA)", True))
    except Exception as e:
        results.append(_log_gate("bf16: mode correctness", False, str(e)))

    # --- fp16: autocast float16, scaler enabled ---
    try:
        cfg = PrecisionConfig(mode="fp16")
        ctx = PrecisionContext(cfg)
        assert ctx._scaler_enabled is True
        assert ctx._dtype == torch.float16
        assert math.isclose(ctx.current_scale, 65536.0, rel_tol=1e-3)

        if torch.cuda.is_available():
            model = _make_tiny_model().cuda()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            x = torch.randn(2, 16, device="cuda")
            optimizer.zero_grad()
            with ctx.autocast_ctx():
                out = model(x)
            loss = out.sum()
            ctx.backward(loss)
            ctx.unscale_and_clip(optimizer, model.parameters())
            stepped = ctx.optimizer_step(optimizer)
            assert isinstance(stepped, bool)
            results.append(_log_gate("fp16: autocast float16, scaler enabled, CUDA", True))
        else:
            results.append(_log_gate("fp16: scaler=True, dtype=float16 (no CUDA)", True))
    except Exception as e:
        results.append(_log_gate("fp16: mode correctness", False, str(e)))

    # --- Mode derivation logic ---
    try:
        from precision_config_template import resolve_autocast_dtype, resolve_scaler_enabled
        assert resolve_autocast_dtype("fp32") is None
        assert resolve_autocast_dtype("bf16") == torch.bfloat16
        assert resolve_autocast_dtype("fp16") == torch.float16
        assert resolve_scaler_enabled("fp32") is False
        assert resolve_scaler_enabled("bf16") is False
        assert resolve_scaler_enabled("fp16") is True
        results.append(_log_gate("Mode derivation functions correct", True))
    except Exception as e:
        results.append(_log_gate("Mode derivation functions", False, str(e)))

    passed = all(results)
    print(f"  Gate 1 result: {'PASSED' if passed else 'FAILED'} ({sum(results)}/{len(results)} checks)")
    return passed


# ---------------------------------------------------------------------------
# Gate 2: No Silent Failure
# ---------------------------------------------------------------------------

def validate_no_silent_failure(snapshot_root: str, verbose: bool = False) -> bool:
    """Gate 2: NaN injection triggers detection + snapshot write."""
    print("\nGate 2: No Silent Failure")
    results = []

    try:
        cfg_precision = PrecisionConfig(mode="fp32")
        cfg_sentinel = SentinelConfig(every_n_steps=1, logit_max_abs_threshold=50.0)
        cfg_failure = FailureConfig(
            nan_persist_steps=2,
            snapshot_dir=os.path.join(snapshot_root, "{run_id}"),
            on_error="raise",
        )
        full_cfg = FullConfig(precision=cfg_precision, sentinel=cfg_sentinel, failure=cfg_failure)

        model = _make_tiny_model()
        monitor = NumericsMonitor(cfg_sentinel, model)
        snapshot = FailureSnapshot(cfg_failure, full_cfg, run_id="gate2_test")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Step 1: Inject NaN into model weights
        with torch.no_grad():
            for p in model.parameters():
                p.data[0, 0] = float("nan")
                break

        # Check weights detect NaN
        weight_report = monitor.check_weights(step=1)
        assert not weight_report.all_finite, "Should detect NaN in weights"
        assert weight_report.first_nonfinite_name is not None
        results.append(_log_gate(
            f"NaN injection detected in {weight_report.first_nonfinite_name}",
            True,
        ))

        # Step 2: Verify snapshot is written
        batch = _make_batch()
        nan_streak = snapshot.update_nan_streak(float("nan"), weight_report)
        assert nan_streak == 1

        # Force abort threshold
        snapshot.nan_steps_in_a_row = cfg_failure.nan_persist_steps
        assert snapshot.should_abort(snapshot.nan_steps_in_a_row)

        snap_path = snapshot.capture(
            step=1,
            model=model,
            optimizer=optimizer,
            batch=batch,
            report=None,
        )
        assert os.path.isdir(snap_path), f"Snapshot dir not created: {snap_path}"
        results.append(_log_gate(f"Snapshot written to {snap_path}", True))

        # Verify all 7 files
        required = [
            "config.json", "env.json", "rng_state.pt",
            "batch.pt", "model_state.pt", "optimizer_state.pt", "numerics.json",
        ]
        for fname in required:
            fpath = os.path.join(snap_path, fname)
            exists = os.path.exists(fpath) and os.path.getsize(fpath) > 0
            if not exists:
                results.append(_log_gate(f"File {fname} present", False))

    except Exception as e:
        results.append(_log_gate("NaN injection and detection", False, str(e)))

    # Loss spike detection
    try:
        detector = LossSpikeDetector(window=20, spike_pct=200.0, min_samples=5)
        for _ in range(15):
            detector.is_spike(1.0)
        spike = detector.is_spike(5.0)  # 5x spike
        assert spike is True
        results.append(_log_gate("Loss spike detection fires on 5x jump", True))
    except Exception as e:
        results.append(_log_gate("Loss spike detection", False, str(e)))

    passed = all(results)
    print(f"  Gate 2 result: {'PASSED' if passed else 'FAILED'} ({sum(results)}/{len(results)} checks)")
    return passed


# ---------------------------------------------------------------------------
# Gate 3: Actionable Artifacts
# ---------------------------------------------------------------------------

def validate_actionable_artifacts(snapshot_root: str, verbose: bool = False) -> bool:
    """Gate 3: Snapshot folder has all 7 files, all load without error."""
    print("\nGate 3: Actionable Artifacts")
    results = []

    try:
        cfg = FailureConfig(
            snapshot_dir=os.path.join(snapshot_root, "{run_id}"),
            on_error="raise",
        )
        full_cfg = FullConfig(failure=cfg)
        snap = FailureSnapshot(cfg, full_cfg, run_id="gate3_test")

        model = _make_tiny_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        batch = _make_batch()

        path = snap.capture(step=99, model=model, optimizer=optimizer, batch=batch, report=None)

        # Check validate() passes
        is_valid = FailureSnapshot.validate(path)
        results.append(_log_gate("FailureSnapshot.validate() returns True", is_valid))

        # Load each file individually
        import json
        for jf in ["config.json", "env.json", "numerics.json"]:
            try:
                with open(os.path.join(path, jf)) as f:
                    data = json.load(f)
                results.append(_log_gate(f"{jf} loads without error", True))
            except Exception as e:
                results.append(_log_gate(f"{jf} loads", False, str(e)))

        for ptf in ["rng_state.pt", "batch.pt", "model_state.pt", "optimizer_state.pt"]:
            try:
                loaded = _load_pt_file(os.path.join(path, ptf))
                results.append(_log_gate(f"{ptf} loads without error", True))
            except Exception as e:
                results.append(_log_gate(f"{ptf} loads", False, str(e)))

        # Verify env.json has expected keys
        with open(os.path.join(path, "env.json")) as f:
            env = json.load(f)
        for key in ["torch_version", "cuda_version", "git_sha", "python_version", "platform"]:
            if key not in env:
                results.append(_log_gate(f"env.json has key '{key}'", False))

        # Verify rng_state.pt has torch_cpu and python keys
        rng = _load_pt_file(os.path.join(path, "rng_state.pt"))
        for key in ["torch_cpu", "python"]:
            has_key = key in rng
            results.append(_log_gate(f"rng_state.pt has key '{key}'", has_key))

    except Exception as e:
        results.append(_log_gate("Snapshot artifact creation and loading", False, str(e)))

    passed = all(results)
    print(f"  Gate 3 result: {'PASSED' if passed else 'FAILED'} ({sum(results)}/{len(results)} checks)")
    return passed


# ---------------------------------------------------------------------------
# Gate 4: Measurable Stability
# ---------------------------------------------------------------------------

def validate_measurable_stability(verbose: bool = False) -> bool:
    """Gate 4: Logs include skip rate, grad norms, logit monitoring."""
    print("\nGate 4: Measurable Stability")
    results = []

    # --- Verify skip rate tracking ---
    try:
        cfg = PrecisionConfig(mode="fp32")
        ctx = PrecisionContext(cfg)

        model = nn.Linear(4, 4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Run 10 steps
        for _ in range(10):
            optimizer.zero_grad()
            loss = model(torch.randn(2, 4)).sum()
            ctx.backward(loss)
            ctx.optimizer_step(optimizer)

        assert ctx.num_steps_total == 10
        assert ctx.skip_rate == 0.0
        assert ctx.effective_update_rate == 1.0
        results.append(_log_gate(
            f"Skip rate tracked correctly (10 steps, rate=0.0%)",
            True,
        ))
    except Exception as e:
        results.append(_log_gate("Skip rate tracking", False, str(e)))

    # --- Verify grad norm reporting ---
    try:
        cfg_s = SentinelConfig(every_n_steps=1)
        model = nn.Linear(8, 4)
        monitor = NumericsMonitor(cfg_s, model)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        x = torch.randn(2, 8)
        loss = model(x).sum()
        loss.backward()

        report = monitor.check_grad_norms(step=5)
        assert report.global_norm > 0
        assert report.is_finite
        assert len(report.per_module_topk) > 0
        results.append(_log_gate(
            f"Grad norm reported: global={report.global_norm:.4f}, "
            f"{len(report.per_module_topk)} modules",
            True,
        ))
    except Exception as e:
        results.append(_log_gate("Grad norm reporting", False, str(e)))

    # --- Verify logit monitoring ---
    try:
        cfg_s = SentinelConfig(logit_max_abs_threshold=50.0, logit_alert_consecutive=3)
        model = nn.Linear(4, 4)
        monitor = NumericsMonitor(cfg_s, model)

        logits = torch.tensor([[10.0, -20.0, 30.0, -5.0]])
        report = monitor.check_logits(logits, step=1)

        assert math.isclose(report.max_abs, 30.0, rel_tol=1e-5)
        assert not report.exceeds_threshold  # 30 < 50
        results.append(_log_gate(
            f"Logit monitoring: max_abs={report.max_abs:.1f}, threshold={cfg_s.logit_max_abs_threshold}",
            True,
        ))
    except Exception as e:
        results.append(_log_gate("Logit monitoring", False, str(e)))

    # --- Verify logit alert after K consecutive violations ---
    try:
        cfg_s = SentinelConfig(logit_max_abs_threshold=50.0, logit_alert_consecutive=3)
        model = nn.Linear(4, 4)
        monitor = NumericsMonitor(cfg_s, model)

        high_logits = torch.tensor([[100.0, -100.0, 80.0, 90.0]])
        for i in range(3):
            r = monitor.check_logits(high_logits, step=i)
        assert r.consecutive_violations == 3
        results.append(_log_gate(
            f"Logit alert fires after {r.consecutive_violations} consecutive violations",
            True,
        ))
    except Exception as e:
        results.append(_log_gate("Logit consecutive alert", False, str(e)))

    # --- Verify loss spike detector ---
    try:
        detector = LossSpikeDetector(window=50, spike_pct=200.0, min_samples=5)
        for _ in range(20):
            detector.is_spike(1.0)
        stats = detector.get_stats()
        assert math.isclose(stats["median"], 1.0, rel_tol=0.01)

        spike = detector.is_spike(5.0)
        assert spike is True
        results.append(_log_gate(
            f"Loss spike detector: median={stats['median']:.2f}, spike on 5.0: {spike}",
            True,
        ))
    except Exception as e:
        results.append(_log_gate("Loss spike detector", False, str(e)))

    passed = all(results)
    print(f"  Gate 4 result: {'PASSED' if passed else 'FAILED'} ({sum(results)}/{len(results)} checks)")
    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate Precision + Numerics Stabilizer done-when gates"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--gate",
        type=int,
        choices=[1, 2, 3, 4],
        help="Run only a specific gate (1-4). Default: all.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    print("=" * 60)
    print("Precision + Numerics Stabilizer: Gate Validation")
    print("=" * 60)

    # Use a temporary directory for snapshots
    with tempfile.TemporaryDirectory() as snap_root:
        gate_results = {}

        if args.gate is None or args.gate == 1:
            gate_results[1] = validate_mode_correctness(verbose=args.verbose)

        if args.gate is None or args.gate == 2:
            gate_results[2] = validate_no_silent_failure(snap_root, verbose=args.verbose)

        if args.gate is None or args.gate == 3:
            gate_results[3] = validate_actionable_artifacts(snap_root, verbose=args.verbose)

        if args.gate is None or args.gate == 4:
            gate_results[4] = validate_measurable_stability(verbose=args.verbose)

    print("\n" + "=" * 60)
    print("Gate Summary:")
    all_passed = True
    gate_names = {
        1: "Mode Correctness",
        2: "No Silent Failure",
        3: "Actionable Artifacts",
        4: "Measurable Stability",
    }
    for gate_num, passed in sorted(gate_results.items()):
        icon = "PASS" if passed else "FAIL"
        print(f"  Gate {gate_num} — {gate_names[gate_num]}: {icon}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("ALL GATES PASSED. Precision + Numerics Stabilizer is ready.")
        sys.exit(0)
    else:
        print("SOME GATES FAILED. Review output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
