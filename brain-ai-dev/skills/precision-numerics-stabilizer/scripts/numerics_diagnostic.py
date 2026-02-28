"""
Precision + Numerics Stabilizer - Numerics Diagnostic Script
=============================================================
Runs N steps with configurable precision mode, prints per-step sentinel
reports, simulates NaN injection at step N/2, and verifies snapshot capture.

Usage:
    python numerics_diagnostic.py
    python numerics_diagnostic.py --mode bf16 --steps 30
    python numerics_diagnostic.py --mode fp16 --steps 20 --no-inject
    python numerics_diagnostic.py --mode fp32 --steps 10 --verbose

CRITICAL: This script uses module.train(False) instead of the deprecated pattern.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import tempfile
import time

# Add assets directory to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ASSETS_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "assets")
if _ASSETS_DIR not in sys.path:
    sys.path.insert(0, _ASSETS_DIR)

import torch
import torch.nn as nn

from precision_config_template import FailureConfig, FullConfig, PrecisionConfig, SentinelConfig
from precision_context_template import PrecisionContext
from numerics_monitor_template import NumericsMonitor, NumericsReport
from failure_snapshot_template import FailureSnapshot
from loss_spike_detector_template import LossSpikeDetector


# ---------------------------------------------------------------------------
# Model and data helpers
# ---------------------------------------------------------------------------

class DiagnosticMLP(nn.Module):
    """Simple 3-layer MLP for diagnostic runs."""
    def __init__(self, in_dim: int = 32, hidden: int = 64, out_dim: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.fc2(torch.relu(self.fc1(x)))))


def make_batch(batch_size: int = 8, in_dim: int = 32, device: str = "cpu") -> dict:
    """Generate a random batch."""
    return {
        "input": torch.randn(batch_size, in_dim, device=device),
        "labels": torch.randint(0, 16, (batch_size,), device=device),
    }


# ---------------------------------------------------------------------------
# Per-step reporting
# ---------------------------------------------------------------------------

def print_step_report(
    step: int,
    loss: float,
    grad_norm: float,
    logit_report,
    stepped: bool,
    scaler_scale: float,
    nan_streak: int,
):
    """Print a compact per-step summary."""
    injected_marker = " [NaN INJECTED]" if not math.isfinite(loss) else ""
    skip_marker = " [SKIPPED]" if not stepped else ""
    print(
        f"  Step {step:4d}: loss={loss:8.4f}{injected_marker}  "
        f"grad_norm={grad_norm:7.4f}  "
        f"logit_max={logit_report.max_abs:7.2f}  "
        f"scale={scaler_scale:8.1f}  "
        f"stepped={stepped}{skip_marker}  "
        f"nan_streak={nan_streak}"
    )


# ---------------------------------------------------------------------------
# Main diagnostic loop
# ---------------------------------------------------------------------------

def run_diagnostic(
    mode: str = "fp32",
    steps: int = 30,
    inject_nan: bool = True,
    inject_step: int = -1,
    batch_size: int = 8,
    in_dim: int = 32,
    hidden: int = 64,
    out_dim: int = 16,
    snapshot_dir: str = "/tmp/diagnostics/{run_id}",
    verbose: bool = False,
    device: str = "cpu",
) -> dict:
    """Run the diagnostic training loop.

    Parameters
    ----------
    mode : str
        Precision mode: fp32, bf16, fp16.
    steps : int
        Number of training steps.
    inject_nan : bool
        Whether to inject NaN at inject_step.
    inject_step : int
        Step at which to inject NaN. Default: steps // 2.
    batch_size, in_dim, hidden, out_dim : int
        Model and data dimensions.
    snapshot_dir : str
        Directory template for failure snapshots.
    verbose : bool
        Print extra debug output.
    device : str
        'cpu' or 'cuda'.

    Returns
    -------
    dict
        Summary: losses, grad_norms, nan_detected, snapshot_path, etc.
    """
    if inject_step < 0:
        inject_step = steps // 2

    print(f"\n{'='*60}")
    print(f"Numerics Diagnostic Run")
    print(f"  mode={mode}  steps={steps}  inject_nan={inject_nan}  inject_step={inject_step}")
    print(f"  device={device}  model=MLP({in_dim}x{hidden}x{out_dim})")
    print(f"{'='*60}")

    # Build configs
    cfg_precision = PrecisionConfig(mode=mode)
    cfg_sentinel = SentinelConfig(
        every_n_steps=5,
        logit_max_abs_threshold=50.0,
        logit_alert_consecutive=3,
        nan_check_sample_layers=("fc1", "fc2", "head"),
    )
    cfg_failure = FailureConfig(
        nan_persist_steps=3,
        snapshot_dir=snapshot_dir,
        on_error="raise",  # Don't call sys.exit in diagnostic mode
    )
    full_cfg = FullConfig(
        precision=cfg_precision,
        sentinel=cfg_sentinel,
        failure=cfg_failure,
    )

    # Build model and move to device
    model = DiagnosticMLP(in_dim, hidden, out_dim)
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    elif device == "cuda":
        print("  [WARN] CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ctx = PrecisionContext(cfg_precision)
    monitor = NumericsMonitor(cfg_sentinel, model)
    monitor.register_hooks()

    spike_detector = LossSpikeDetector(window=20, spike_pct=200.0, min_samples=5)
    snapshot = FailureSnapshot(cfg_failure, full_cfg, run_id="diag_run")

    criterion = nn.CrossEntropyLoss()

    # Tracking
    losses = []
    grad_norms = []
    logit_max_abs_list = []
    step_results = []
    snapshot_path = None
    nan_detected_at_step = None

    print(f"\n{'Step':>6}  {'Loss':>10}  {'GradNorm':>10}  "
          f"{'LogitMax':>10}  {'Scale':>10}  {'Stepped':>8}  {'NaNStreak':>10}")
    print("-" * 80)

    for step in range(steps):
        batch = make_batch(batch_size, in_dim, device=device)
        optimizer.zero_grad()

        # Forward pass
        with ctx.autocast_ctx():
            logits = model(batch["input"])
            loss = criterion(logits, batch["labels"])

        # Sentinel checks at cadence
        do_check = monitor.should_check(step)
        logit_report = monitor.check_logits(logits.detach(), step=step)

        # NaN injection at inject_step
        if inject_nan and step == inject_step:
            print(f"\n  *** Injecting NaN into gradient at step {step} ***\n")
            ctx.backward(loss)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data[0] = float("nan")
        else:
            ctx.backward(loss)

        # Gradient norm check
        grad_norm = ctx.unscale_and_clip(optimizer, model.parameters())

        if do_check:
            grad_report = monitor.check_grad_norms(step=step)
            if verbose:
                print(f"  [Sentinel@{step}] GradNorm: {grad_report.global_norm:.4f}, "
                      f"max_param: {grad_report.max_param_name}")

        # Optimizer step
        stepped = ctx.optimizer_step(optimizer)

        # Loss spike check
        loss_val = loss.item()
        if spike_detector.is_spike(loss_val):
            print(f"  [SPIKE@{step}] Loss spike: {loss_val:.4f} vs "
                  f"median={spike_detector.get_stats()['median']:.4f}")

        # NaN streak tracking
        weight_report = monitor.check_weights(step=step) if not stepped else None
        nan_streak = snapshot.update_nan_streak(loss_val, weight_report)

        if nan_streak > 0 and nan_detected_at_step is None:
            nan_detected_at_step = step
            print(f"  [ALERT@{step}] NaN/Inf detected! Streak: {nan_streak}")

        # Print per-step report
        print_step_report(
            step=step,
            loss=loss_val,
            grad_norm=grad_norm,
            logit_report=logit_report,
            stepped=stepped,
            scaler_scale=ctx.current_scale,
            nan_streak=nan_streak,
        )

        # Tracking
        losses.append(loss_val)
        grad_norms.append(grad_norm)
        logit_max_abs_list.append(logit_report.max_abs)
        step_results.append(stepped)

        # Check for persistent NaN - capture snapshot and break
        if snapshot.should_abort(nan_streak):
            print(f"\n  [ABORT@{step}] Persistent NaN for {nan_streak} steps. Capturing snapshot...")
            try:
                snapshot_path = snapshot.capture(
                    step=step,
                    model=model,
                    optimizer=optimizer,
                    batch=batch,
                    report=monitor.aggregate_report(step),
                    scaler=ctx.scaler if ctx._scaler_enabled else None,
                    skip_counters={
                        "num_steps_total": ctx.num_steps_total,
                        "num_steps_skipped": ctx.num_steps_skipped,
                        "skip_rate": ctx.skip_rate,
                        "effective_update_rate": ctx.effective_update_rate,
                    },
                )
                print(f"  Snapshot: {snapshot_path}")
            except Exception as e:
                print(f"  [ERROR] Snapshot failed: {e}")
            break

    monitor.remove_hooks()

    # Print summary
    print(f"\n{'='*60}")
    print("Diagnostic Summary")
    print(f"{'='*60}")

    finite_losses = [l for l in losses if math.isfinite(l)]
    print(f"  Steps completed: {len(losses)}/{steps}")
    print(f"  Finite losses: {len(finite_losses)}/{len(losses)}")
    if finite_losses:
        print(f"  Loss range: [{min(finite_losses):.4f}, {max(finite_losses):.4f}]")
    print(f"  Optimizer skips: {ctx.num_steps_skipped}/{ctx.num_steps_total}")
    print(f"  Skip rate: {ctx.skip_rate*100:.2f}%")
    print(f"  Effective update rate: {ctx.effective_update_rate*100:.2f}%")
    print(f"  Current GradScaler scale: {ctx.current_scale:.1f}")

    if inject_nan and nan_detected_at_step is not None:
        print(f"  NaN first detected at step: {nan_detected_at_step}")
    elif inject_nan:
        print(f"  NaN injection did not persist (sentinel did not fire)")

    if snapshot_path:
        print(f"\nSnapshot Contents ({snapshot_path}):")
        if os.path.isdir(snapshot_path):
            for fname in sorted(os.listdir(snapshot_path)):
                fpath = os.path.join(snapshot_path, fname)
                size_kb = os.path.getsize(fpath) / 1024
                print(f"    {fname}: {size_kb:.1f} KB")

            # Verify snapshot
            from failure_snapshot_template import FailureSnapshot as FS
            is_valid = FS.validate(snapshot_path)
            print(f"  Snapshot valid: {is_valid}")
        else:
            print(f"  [ERROR] Snapshot directory not found: {snapshot_path}")
    else:
        if inject_nan:
            print(f"\n  No snapshot captured (NaN streak did not reach persist threshold)")
        else:
            print(f"\n  No snapshot captured (NaN injection disabled)")

    # Sentinel report summary
    print(f"\nSentinel Report Summary:")
    finite_grad_norms = [g for g in grad_norms if math.isfinite(g)]
    if finite_grad_norms:
        print(f"  Grad norm: min={min(finite_grad_norms):.4f}, "
              f"max={max(finite_grad_norms):.4f}, "
              f"mean={sum(finite_grad_norms)/len(finite_grad_norms):.4f}")
    finite_logits = [l for l in logit_max_abs_list if math.isfinite(l)]
    if finite_logits:
        print(f"  Logit max_abs: min={min(finite_logits):.2f}, "
              f"max={max(finite_logits):.2f}, "
              f"mean={sum(finite_logits)/len(finite_logits):.2f}")
    spike_stats = spike_detector.get_stats()
    print(f"  Loss spikes detected: {spike_detector.spike_count}")
    print(f"  Final loss median: {spike_stats['median']:.4f}")

    return {
        "losses": losses,
        "grad_norms": grad_norms,
        "logit_max_abs": logit_max_abs_list,
        "step_results": step_results,
        "nan_detected_at_step": nan_detected_at_step,
        "snapshot_path": snapshot_path,
        "skip_rate": ctx.skip_rate,
        "effective_update_rate": ctx.effective_update_rate,
        "final_scale": ctx.current_scale,
        "spike_count": spike_detector.spike_count,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Numerics Diagnostic — run N steps, print sentinel reports, "
                    "simulate NaN injection, verify snapshot"
    )
    parser.add_argument(
        "--mode", default="fp32", choices=["fp32", "bf16", "fp16"],
        help="Precision mode (default: fp32)",
    )
    parser.add_argument(
        "--steps", type=int, default=30,
        help="Number of training steps (default: 30)",
    )
    parser.add_argument(
        "--no-inject", action="store_true",
        help="Disable NaN injection at step N/2",
    )
    parser.add_argument(
        "--inject-step", type=int, default=-1,
        help="Step to inject NaN (default: N//2). -1 = auto.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size (default: 8)",
    )
    parser.add_argument(
        "--in-dim", type=int, default=32,
        help="Input dimension (default: 32)",
    )
    parser.add_argument(
        "--snapshot-dir", type=str, default="/tmp/diagnostics/{run_id}",
        help="Snapshot output directory template",
    )
    parser.add_argument(
        "--cuda", action="store_true",
        help="Use CUDA (if available)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Verbose sentinel output",
    )
    args = parser.parse_args()

    # Validate mode/device combination
    device = "cuda" if args.cuda else "cpu"
    if args.mode in ("bf16", "fp16") and not torch.cuda.is_available() and device == "cuda":
        print(f"[WARN] {args.mode} requires CUDA for autocast. "
              f"Running on CPU without autocast context.")
        device = "cpu"

    if args.mode in ("bf16", "fp16") and device == "cpu":
        print(f"[WARN] {args.mode} autocast requires CUDA. "
              f"Config will be set but autocast context won't activate on CPU.")

    result = run_diagnostic(
        mode=args.mode,
        steps=args.steps,
        inject_nan=not args.no_inject,
        inject_step=args.inject_step,
        batch_size=args.batch_size,
        in_dim=args.in_dim,
        snapshot_dir=args.snapshot_dir,
        verbose=args.verbose,
        device=device,
    )

    # Exit code: 0 if no NaN detected, 1 if NaN was detected and snapshot was captured
    if result["snapshot_path"] is not None:
        print(f"\nDiagnostic complete. NaN captured at snapshot: {result['snapshot_path']}")
        sys.exit(0)  # Still exit 0 for diagnostic - it's expected behavior
    else:
        print(f"\nDiagnostic complete. No persistent NaN detected.")
        sys.exit(0)


if __name__ == "__main__":
    main()
