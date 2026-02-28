"""
training_diagnostic.py — Diagnostic tool for the SSL training loop.

Run N steps and print:
    1. Per-step trace: loss, grad_norm, tau, lr, gpu_mem
    2. EMA tau ASCII curve (tau vs step)
    3. Checkpoint size breakdown
    4. Anomaly detection: inf/nan loss, gradient explosion, tau not increasing

Usage:
    python training_diagnostic.py                  # 20 steps (default)
    python training_diagnostic.py --steps 50       # 50 steps
    python training_diagnostic.py --no-checkpoint  # skip checkpoint size report
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn as nn

# Add assets to path
ASSETS_DIR = Path(__file__).parent.parent / 'assets'
sys.path.insert(0, str(ASSETS_DIR))

from amp_gradient_template import AMPContext
from ema_template import EMAUpdater
from checkpoint_template import CheckpointManager
from wandb_logger_template import WandbLogger
from training_config_template import TrainingConfig
from training_loop_template import CosineWarmupScheduler, SelfSupervisedTrainer


# ---------------------------------------------------------------------------
# ASCII plot utilities
# ---------------------------------------------------------------------------

def ascii_line_plot(
    values: List[float],
    title: str = '',
    width: int = 60,
    height: int = 10,
    x_label: str = 'step',
    y_label: str = '',
) -> str:
    """
    Render a simple ASCII line plot.

    Args:
        values: Y-axis values (equally spaced on x-axis).
        title: Plot title.
        width: Terminal character width of the plot area.
        height: Terminal character height of the plot area.
        x_label: X-axis label.
        y_label: Y-axis label.

    Returns:
        Multi-line string with the ASCII chart.
    """
    if not values:
        return "(no data)"

    y_min = min(values)
    y_max = max(values)
    y_range = y_max - y_min
    if y_range < 1e-12:
        y_range = 1.0

    n = len(values)
    lines = []

    # Title
    if title:
        lines.append(f"  {title}")
        lines.append("")

    # Build grid: [row][col] -> char
    grid = [[' ' for _ in range(width)] for _ in range(height)]

    # Plot each value as a '*' character
    for i, v in enumerate(values):
        x = int((i / max(n - 1, 1)) * (width - 1))
        y = int(((v - y_min) / y_range) * (height - 1))
        row = (height - 1) - y  # invert: top of grid = max value
        col = min(x, width - 1)
        grid[row][col] = '*'

    # Add vertical axis lines
    for row in range(height):
        y_val = y_max - (row / max(height - 1, 1)) * y_range
        y_str = f"{y_val:.6f}"
        lines.append(f"  {y_str:>10} |{''.join(grid[row])}")

    # X-axis
    lines.append(f"  {' ':>10} +{'-' * width}")
    lines.append(f"  {' ':>10}  {0:<8}{'':>{width - 16}}{n - 1}")
    lines.append(f"  {' ':>10}  {x_label}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Checkpoint size breakdown
# ---------------------------------------------------------------------------

def checkpoint_size_breakdown(checkpoint_path: Path) -> Dict[str, Any]:
    """
    Load a checkpoint and report the size of each component.

    Returns:
        Dict with keys: total_mb, component sizes.
    """
    if not checkpoint_path.exists():
        return {'error': f"Checkpoint not found: {checkpoint_path}"}

    state = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
    total_bytes = checkpoint_path.stat().st_size

    def tensor_bytes(obj) -> int:
        """Recursively count bytes in nested dicts/lists of tensors."""
        if isinstance(obj, torch.Tensor):
            return obj.nelement() * obj.element_size()
        elif isinstance(obj, dict):
            return sum(tensor_bytes(v) for v in obj.values())
        elif isinstance(obj, (list, tuple)):
            return sum(tensor_bytes(v) for v in obj)
        return 0

    breakdown = {}
    for key in ['model_state_dict', 'target_state_dict', 'optimizer_state_dict',
                'scaler_state_dict', 'scheduler_state_dict']:
        if key in state:
            component_bytes = tensor_bytes(state[key])
            breakdown[key] = component_bytes / (1024 ** 2)  # MB

    breakdown['total_file_mb'] = total_bytes / (1024 ** 2)
    breakdown['step'] = state.get('step', 'unknown')
    return breakdown


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def detect_anomalies(history: List[Dict]) -> List[str]:
    """
    Analyze training history for common anomalies.

    Checks:
        - NaN/inf loss
        - Gradient explosion (norm > 100)
        - Tau not increasing (EMA not working)
        - Loss not decreasing (possible collapse)

    Returns:
        List of anomaly description strings (empty if all healthy).
    """
    anomalies = []

    losses = [h['loss'] for h in history]
    grad_norms = [h['grad_norm'] for h in history]
    taus = [h['tau'] for h in history]

    # Check for NaN/inf loss
    nan_steps = [h['step'] for h in history if not math.isfinite(h['loss'])]
    if nan_steps:
        anomalies.append(f"NaN/inf loss detected at steps: {nan_steps}")

    # Check for gradient explosion
    exploded_steps = [(h['step'], h['grad_norm']) for h in history if h['grad_norm'] > 100.0]
    if exploded_steps:
        anomalies.append(
            f"Gradient explosion (norm > 100) at steps: "
            + ", ".join(f"{s}(norm={n:.1f})" for s, n in exploded_steps)
        )

    # Check tau is increasing
    if len(taus) >= 2:
        num_decreases = sum(1 for i in range(1, len(taus)) if taus[i] < taus[i-1])
        if num_decreases > 0:
            anomalies.append(
                f"EMA tau decreased {num_decreases} time(s) — cosine schedule may be broken"
            )

    # Check for very large losses (possible collapse)
    if losses:
        valid_losses = [l for l in losses if math.isfinite(l)]
        if valid_losses and max(valid_losses) > 1e6:
            anomalies.append(
                f"Extremely large loss detected: max={max(valid_losses):.2e} — possible collapse"
            )

    # Check for all-zero grad norms (optimizer not updating)
    if grad_norms and all(g < 1e-10 for g in grad_norms if math.isfinite(g)):
        anomalies.append("All gradient norms are near-zero — optimizer may not be updating")

    return anomalies


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------

def run_diagnostic(num_steps: int = 20, run_checkpoint: bool = True) -> None:
    """
    Run the full diagnostic suite.

    Args:
        num_steps: Number of training steps to run.
        run_checkpoint: If True, save and analyze checkpoint size.
    """
    print("=" * 70)
    print("Self-Supervised Training Loop — Diagnostic Tool")
    print("=" * 70)
    print(f"\nRunning {num_steps} diagnostic steps...\n")

    test_dir = tempfile.mkdtemp(prefix='ssl_diag_')

    try:
        # ----------------------------------------------------------------
        # Setup trainer
        # ----------------------------------------------------------------
        cfg = TrainingConfig(
            checkpoint_dir=test_dir,
            total_steps=max(num_steps, 100),
            warmup_steps=max(2, num_steps // 10),
            checkpoint_every=max(num_steps + 1, 5_000),  # Prevent auto-checkpoint
            auto_resume=False,
            amp_enabled=False,       # CPU-safe
            grad_scaler_enabled=False,
            log_every=1,
        )

        trainer = SelfSupervisedTrainer(cfg, rank=0, world_size=1)
        trainer.setup()

        # ----------------------------------------------------------------
        # Run training steps and collect metrics
        # ----------------------------------------------------------------
        print(f"{'Step':>6}  {'Loss':>10}  {'GradNorm':>10}  "
              f"{'Tau':>10}  {'LR':>10}  {'GPU_MB':>8}")
        print("-" * 70)

        history = []

        for step in range(num_steps):
            # Synthetic batch — send to trainer's device
            view1 = torch.randn(8, 32, device=trainer.device)
            view2 = torch.randn(8, 32, device=trainer.device)

            # Run one step (manually replicate training loop for visibility)
            trainer.optimizer.zero_grad(set_to_none=True)

            with trainer.amp_ctx.autocast():
                from training_loop_template import mock_ssl_forward
                loss = mock_ssl_forward(
                    trainer.online_encoder,
                    trainer.predictor,
                    trainer.target_encoder,
                    view1,
                    view2,
                )

            trainer.amp_ctx.backward(loss)

            online_params = (list(trainer.online_encoder.parameters()) +
                             list(trainer.predictor.parameters()))
            grad_norm = trainer.amp_ctx.unscale_and_clip(
                trainer.optimizer,
                online_params,
                max_norm=cfg.max_grad_norm,
            )

            trainer.amp_ctx.step_and_update(trainer.optimizer)
            trainer.scheduler.step()
            tau = trainer.ema_updater.update(
                trainer.online_encoder, trainer.target_encoder, step
            )

            lr = trainer.scheduler.get_last_lr()[0]
            loss_val = loss.item()
            gpu_mem_mb = (torch.cuda.memory_allocated() / (1024**2)
                          if torch.cuda.is_available() else 0.0)

            entry = {
                'step': step,
                'loss': loss_val,
                'grad_norm': grad_norm,
                'tau': tau,
                'lr': lr,
                'gpu_mem_mb': gpu_mem_mb,
            }
            history.append(entry)

            # Finite/anomaly markers
            loss_str = f"{loss_val:10.4f}" if math.isfinite(loss_val) else f"{'NaN/inf':>10}"
            gn_str = f"{grad_norm:10.4f}" if math.isfinite(grad_norm) else f"{'NaN/inf':>10}"

            print(f"{step:>6}  {loss_str}  {gn_str}  "
                  f"{tau:10.6f}  {lr:10.2e}  {gpu_mem_mb:8.1f}")

        # ----------------------------------------------------------------
        # EMA Tau ASCII curve
        # ----------------------------------------------------------------
        print("\n" + "=" * 70)
        print("EMA Tau Schedule (tau vs training step)")
        print("=" * 70)

        tau_values = [h['tau'] for h in history]
        plot = ascii_line_plot(
            tau_values,
            title="EMA tau annealing",
            width=60,
            height=8,
            x_label="step",
        )
        print(plot)

        # Also print min/max/final
        if tau_values:
            print(f"\n  tau_min={min(tau_values):.6f}, "
                  f"tau_max={max(tau_values):.6f}, "
                  f"tau_final={tau_values[-1]:.6f}")

        # ----------------------------------------------------------------
        # Loss curve
        # ----------------------------------------------------------------
        print("\n" + "=" * 70)
        print("Loss Curve (loss vs step)")
        print("=" * 70)

        finite_losses = [h['loss'] for h in history if math.isfinite(h['loss'])]
        if finite_losses:
            loss_plot = ascii_line_plot(
                finite_losses,
                title="Training loss",
                width=60,
                height=8,
                x_label="step",
            )
            print(loss_plot)
            print(f"\n  loss_start={finite_losses[0]:.6f}, "
                  f"loss_end={finite_losses[-1]:.6f}")

        # ----------------------------------------------------------------
        # Checkpoint size breakdown
        # ----------------------------------------------------------------
        if run_checkpoint:
            print("\n" + "=" * 70)
            print("Checkpoint Size Breakdown")
            print("=" * 70)

            ckpt_path = trainer.checkpoint_manager._checkpoint_path(step=num_steps - 1)
            trainer.save_checkpoint(step=num_steps - 1)

            if ckpt_path.exists():
                breakdown = checkpoint_size_breakdown(ckpt_path)

                labels = {
                    'model_state_dict':     'Online encoder weights',
                    'target_state_dict':    'Target encoder weights',
                    'optimizer_state_dict': 'Optimizer state (Adam moments)',
                    'scaler_state_dict':    'GradScaler state',
                    'scheduler_state_dict': 'Scheduler state',
                }

                print(f"\n  Checkpoint at step {breakdown.get('step', '?')}:")
                for key, label in labels.items():
                    if key in breakdown:
                        size = breakdown[key]
                        bar = '#' * max(1, int(size * 10))
                        print(f"    {label:<35} {size:8.3f} MB  {bar}")

                print(f"\n  Total file size: {breakdown.get('total_file_mb', 0):.3f} MB")
            else:
                print("  (checkpoint file not found)")

        # ----------------------------------------------------------------
        # Anomaly detection
        # ----------------------------------------------------------------
        print("\n" + "=" * 70)
        print("Anomaly Detection")
        print("=" * 70)

        anomalies = detect_anomalies(history)

        if not anomalies:
            print("\n  [OK] No anomalies detected.")
            print("  Training loop appears healthy:")
            print("    - All losses are finite")
            print(f"    - grad_norms range: [{min(h['grad_norm'] for h in history):.4f}, "
                  f"{max(h['grad_norm'] for h in history):.4f}]")
            print(f"    - tau increasing: {tau_values[0]:.6f} -> {tau_values[-1]:.6f}")
        else:
            print(f"\n  [WARNING] {len(anomalies)} anomaly(s) detected:")
            for i, anomaly in enumerate(anomalies, 1):
                print(f"    {i}. {anomaly}")

        # ----------------------------------------------------------------
        # Summary statistics
        # ----------------------------------------------------------------
        print("\n" + "=" * 70)
        print("Summary Statistics")
        print("=" * 70)

        valid_losses = [h['loss'] for h in history if math.isfinite(h['loss'])]
        valid_norms = [h['grad_norm'] for h in history if math.isfinite(h['grad_norm'])]

        if valid_losses:
            print(f"\n  Loss:")
            print(f"    First step: {valid_losses[0]:.6f}")
            print(f"    Last step:  {valid_losses[-1]:.6f}")
            print(f"    Min:        {min(valid_losses):.6f}")
            print(f"    Max:        {max(valid_losses):.6f}")
            mean_loss = sum(valid_losses) / len(valid_losses)
            print(f"    Mean:       {mean_loss:.6f}")

        if valid_norms:
            print(f"\n  Gradient Norm:")
            print(f"    Min:  {min(valid_norms):.6f}")
            print(f"    Max:  {max(valid_norms):.6f}")
            mean_norm = sum(valid_norms) / len(valid_norms)
            print(f"    Mean: {mean_norm:.6f}")

        if tau_values:
            print(f"\n  EMA Tau:")
            print(f"    Start:  {tau_values[0]:.6f}")
            print(f"    End:    {tau_values[-1]:.6f}")
            print(f"    Change: {tau_values[-1] - tau_values[0]:.8f}")

        lr_values = [h['lr'] for h in history]
        if lr_values:
            print(f"\n  Learning Rate:")
            print(f"    Start: {lr_values[0]:.2e}")
            print(f"    End:   {lr_values[-1]:.2e}")

        print("\n" + "=" * 70)
        if not anomalies:
            print("Diagnostic COMPLETE — training loop appears healthy.")
        else:
            print(f"Diagnostic COMPLETE — {len(anomalies)} anomaly(s) found. Review above.")
        print("=" * 70)

    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnostic tool for SSL training loop"
    )
    parser.add_argument(
        '--steps', type=int, default=20,
        help="Number of training steps to run (default: 20)"
    )
    parser.add_argument(
        '--no-checkpoint', action='store_true',
        help="Skip checkpoint size breakdown"
    )
    args = parser.parse_args()

    run_diagnostic(
        num_steps=args.steps,
        run_checkpoint=not args.no_checkpoint,
    )


if __name__ == "__main__":
    main()
