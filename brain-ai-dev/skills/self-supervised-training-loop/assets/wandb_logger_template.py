"""
WandbLogger: Rank-0-gated W&B logging for self-supervised training.

Design:
    - Only rank 0 calls wandb.init() and wandb.log()
    - All other ranks: all methods are no-ops (no import needed)
    - Uses mock/stub for testing without actual wandb installation
    - Gradient spike detection with configurable thresholds

CRITICAL: When switching model to inference mode, use module.train(False).
          The module.train(False) call is equivalent to switching to inference mode.
"""

from __future__ import annotations

import math
from typing import Optional, Any
from collections import deque


class WandbLogger:
    """
    Rank-0-only W&B logging with graceful no-ops on other ranks.

    Args:
        cfg: TrainingConfig dataclass (or any object with wandb_project, etc.).
        rank: Current process rank. Only rank=0 performs actual logging.
        wandb_module: Optional pre-imported wandb module (for testing with mocks).
    """

    def __init__(
        self,
        cfg: Any,
        rank: int = 0,
        wandb_module: Optional[Any] = None,
    ) -> None:
        self.enabled = (rank == 0)
        self._wandb = None
        self._initialized = False
        self._finished = False

        # Gradient spike detection state
        self._grad_norm_history: deque = deque(maxlen=100)
        self._spike_multiplier = 10.0

        if not self.enabled:
            return

        # Import wandb only on rank 0 to avoid unnecessary imports
        if wandb_module is not None:
            self._wandb = wandb_module
        else:
            try:
                import wandb as _wandb
                self._wandb = _wandb
            except ImportError:
                # W&B not installed: disable logging gracefully
                self.enabled = False
                return

        # Initialize W&B run
        try:
            config_dict = self._extract_config(cfg)
            self._wandb.init(
                project=getattr(cfg, 'wandb_project', 'ssl-training'),
                entity=getattr(cfg, 'wandb_entity', None),
                config=config_dict,
                group='DDP',
                job_type='train',
            )

            # Define step-based x-axis for all train/* metrics
            self._define_metrics()
            self._initialized = True
        except Exception as e:
            # W&B init failure should not crash training
            print(f"[WandbLogger] Warning: wandb.init() failed: {e}. Logging disabled.")
            self.enabled = False

    def _extract_config(self, cfg: Any) -> dict:
        """Convert config object to a flat dict for W&B run config."""
        if hasattr(cfg, '__dict__'):
            return cfg.__dict__.copy()
        if isinstance(cfg, dict):
            return cfg.copy()
        try:
            from dataclasses import asdict
            return asdict(cfg)
        except Exception:
            return {}

    def _define_metrics(self) -> None:
        """
        Define custom x-axis for W&B metrics.

        Without this, W&B uses an internal step counter that resets to 0
        on run resume. Using train/step as x-axis keeps the axis continuous
        across crashes/resumes.
        """
        try:
            self._wandb.define_metric('train/step')
            self._wandb.define_metric('train/*', step_metric='train/step')
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Core logging methods
    # ------------------------------------------------------------------

    def log_step(
        self,
        step: int,
        loss: float,
        grad_norm: float,
        tau: float,
        lr: float,
        gpu_mem_gb: float,
    ) -> None:
        """
        Log per-step training metrics to W&B.

        Also checks for gradient spikes and sends alerts if detected.

        Args:
            step: Global training step.
            loss: Scalar loss value (float).
            grad_norm: Pre-clip gradient norm returned by clip_grad_norm_.
            tau: Current EMA tau value from EMAUpdater.
            lr: Current learning rate from scheduler.get_last_lr()[0].
            gpu_mem_gb: GPU memory allocated in GB.
        """
        if not self.enabled:
            return

        metrics = {
            'train/loss':       float(loss),
            'train/grad_norm':  float(grad_norm),
            'train/ema_tau':    float(tau),
            'train/lr':         float(lr),
            'train/gpu_mem_gb': float(gpu_mem_gb),
            'train/step':       step,
        }

        self._wandb.log(metrics, step=step)

        # Check for gradient spikes
        self._check_grad_spike(step, grad_norm)

    def log_predictions(
        self,
        step: int,
        images: Any,  # torch.Tensor of shape (N, C, H, W)
        nrow: int = 8,
    ) -> None:
        """
        Log a prediction image grid to W&B.

        Use a fixed validation batch for visual consistency across steps.
        Switch model to inference mode using module.train(False) (not blocked alternatives).

        Args:
            step: Global training step.
            images: Tensor of shape (N, C, H, W) in range [0, 1].
            nrow: Number of images per row in the grid.
        """
        if not self.enabled:
            return

        try:
            import torchvision.utils as vutils

            # Clamp to valid range before making grid
            images_clamped = images.detach().float().clamp(0.0, 1.0)
            grid = vutils.make_grid(images_clamped, nrow=nrow, normalize=False)

            self._wandb.log({
                'train/predictions': self._wandb.Image(grid),
                'train/step': step,
            }, step=step)
        except ImportError:
            pass  # torchvision not available
        except Exception as e:
            print(f"[WandbLogger] Warning: Failed to log prediction grid at step {step}: {e}")

    def log_gpu_memory(self, step: int) -> None:
        """Log both allocated and reserved GPU memory."""
        if not self.enabled:
            return

        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                self._wandb.log({
                    'train/gpu_mem_gb': allocated,
                    'train/gpu_mem_reserved_gb': reserved,
                    'train/step': step,
                }, step=step)
        except Exception:
            pass

    def finish(self) -> None:
        """
        Flush all buffered metrics and close the W&B run.

        Safe to call multiple times (idempotent).
        Call in finally block to ensure all data is uploaded before process exit.
        """
        if not self.enabled or not self._initialized or self._finished:
            return

        try:
            self._wandb.finish()
            self._finished = True
        except Exception as e:
            print(f"[WandbLogger] Warning: wandb.finish() failed: {e}")
            self._finished = True  # Mark as finished even on error

    # ------------------------------------------------------------------
    # Gradient spike detection
    # ------------------------------------------------------------------

    def _check_grad_spike(self, step: int, grad_norm: float) -> None:
        """Alert if grad_norm is spike_multiplier times the running average."""
        # Check for valid, finite value
        if not math.isfinite(grad_norm):
            return

        self._grad_norm_history.append(grad_norm)

        if len(self._grad_norm_history) < 10:
            return  # Not enough history for meaningful baseline

        # Compute running average of all history except current step
        history_without_current = list(self._grad_norm_history)[:-1]
        if not history_without_current:
            return

        running_avg = sum(history_without_current) / len(history_without_current)

        if running_avg > 0 and grad_norm > self._spike_multiplier * running_avg:
            try:
                self._wandb.alert(
                    title='Gradient Spike Detected',
                    text=(
                        f'Step {step}: grad_norm={grad_norm:.2f} '
                        f'({self._spike_multiplier:.0f}x running avg={running_avg:.2f})'
                    ),
                )
            except Exception:
                pass  # Alert failure should not affect training

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_enabled(self) -> bool:
        """True if this logger will actually log (rank 0 and wandb available)."""
        return self.enabled

    def __repr__(self) -> str:
        return (
            f"WandbLogger(enabled={self.enabled}, "
            f"initialized={self._initialized}, "
            f"finished={self._finished})"
        )


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("WandbLogger Self-Tests")
    print("=" * 60)

    # ----------------------------------------------------------------
    # Mock W&B for testing without actual wandb installation
    # ----------------------------------------------------------------
    class MockWandb:
        """Mock wandb module for testing without real W&B."""

        def __init__(self):
            self.calls = {
                'init': [],
                'log': [],
                'finish': [],
                'define_metric': [],
                'alert': [],
            }
            self._init_called = False

        def init(self, **kwargs):
            self.calls['init'].append(kwargs)
            self._init_called = True

        def log(self, metrics, step=None):
            self.calls['log'].append({'metrics': metrics, 'step': step})

        def finish(self):
            self.calls['finish'].append(True)

        def define_metric(self, metric_name, **kwargs):
            self.calls['define_metric'].append({'name': metric_name, **kwargs})

        def alert(self, title=None, text=None, **kwargs):
            self.calls['alert'].append({'title': title, 'text': text})

        class Image:
            def __init__(self, data, **kwargs):
                self.data = data

    class MockConfig:
        wandb_project = 'test-project'
        wandb_entity = None
        lr = 1e-4
        total_steps = 1000

    # ----------------------------------------------------------------
    # Test 1: rank != 0 logs nothing
    # ----------------------------------------------------------------
    print("\nTest 1: rank != 0 is completely disabled...")

    mock_wb_1 = MockWandb()
    logger_rank1 = WandbLogger(cfg=MockConfig(), rank=1, wandb_module=mock_wb_1)

    assert not logger_rank1.enabled, "rank=1 logger should not be enabled"
    assert mock_wb_1._init_called is False, "wandb.init should not be called for rank=1"

    # These calls should all be no-ops
    logger_rank1.log_step(step=0, loss=1.0, grad_norm=0.5, tau=0.996, lr=1e-4, gpu_mem_gb=2.0)
    logger_rank1.log_predictions(step=0, images=None)
    logger_rank1.finish()

    assert len(mock_wb_1.calls['log']) == 0, "No log calls should be made for rank=1"
    assert len(mock_wb_1.calls['finish']) == 0, "No finish calls should be made for rank=1"
    print("  PASS: rank=1 logger is completely disabled, no wandb calls made")

    # ----------------------------------------------------------------
    # Test 2: rank 0 logs all expected metrics
    # ----------------------------------------------------------------
    print("\nTest 2: rank=0 logs all required metric keys...")

    mock_wb_2 = MockWandb()
    logger_rank0 = WandbLogger(cfg=MockConfig(), rank=0, wandb_module=mock_wb_2)

    assert logger_rank0.enabled, "rank=0 logger should be enabled"
    assert mock_wb_2._init_called, "wandb.init should be called for rank=0"

    logger_rank0.log_step(step=10, loss=0.5, grad_norm=0.3, tau=0.996, lr=1e-4, gpu_mem_gb=1.5)

    assert len(mock_wb_2.calls['log']) >= 1, "At least one log call should be made"
    logged_metrics = mock_wb_2.calls['log'][0]['metrics']

    required_keys = {'train/loss', 'train/grad_norm', 'train/ema_tau', 'train/lr',
                     'train/gpu_mem_gb', 'train/step'}
    missing = required_keys - set(logged_metrics.keys())
    assert not missing, f"Missing metric keys: {missing}"

    assert logged_metrics['train/loss'] == 0.5
    assert logged_metrics['train/grad_norm'] == 0.3
    assert logged_metrics['train/ema_tau'] == 0.996
    assert logged_metrics['train/lr'] == 1e-4
    assert logged_metrics['train/step'] == 10
    print(f"  PASS: All {len(required_keys)} required metric keys logged")

    # ----------------------------------------------------------------
    # Test 3: prediction grid has correct shape
    # ----------------------------------------------------------------
    print("\nTest 3: prediction grid is logged as Image...")

    try:
        import torch
        import torchvision.utils as vutils

        mock_wb_3 = MockWandb()
        logger_rank0_3 = WandbLogger(cfg=MockConfig(), rank=0, wandb_module=mock_wb_3)

        # Create 16 test images: (16, 3, 32, 32) in [0, 1]
        images = torch.rand(16, 3, 32, 32)
        logger_rank0_3.log_predictions(step=1000, images=images, nrow=8)

        # Verify Image was logged
        pred_logs = [
            entry for entry in mock_wb_3.calls['log']
            if 'train/predictions' in entry.get('metrics', {})
        ]
        assert len(pred_logs) >= 1, "prediction grid should be logged"

        # Verify it's wrapped in wandb.Image
        img_obj = pred_logs[0]['metrics']['train/predictions']
        assert isinstance(img_obj, MockWandb.Image), (
            f"Expected wandb.Image wrapper, got {type(img_obj)}"
        )

        # Verify grid shape: (3, H, W) where H >= 32 and W >= 32
        grid_data = img_obj.data
        assert grid_data.ndim == 3, f"Grid should be (C, H, W), got shape {grid_data.shape}"
        assert grid_data.shape[0] == 3, f"Grid should have 3 channels, got {grid_data.shape[0]}"
        assert grid_data.shape[1] >= 32, "Grid height should be at least image height"
        assert grid_data.shape[2] >= 32, "Grid width should be at least image width"
        print(f"  PASS: Grid logged as Image with shape {tuple(grid_data.shape)}")

    except ImportError:
        print("  SKIP: torch/torchvision not available for grid test")

    # ----------------------------------------------------------------
    # Test 4: step metric is defined
    # ----------------------------------------------------------------
    print("\nTest 4: custom x-axis metric defined...")

    mock_wb_4 = MockWandb()
    logger_rank0_4 = WandbLogger(cfg=MockConfig(), rank=0, wandb_module=mock_wb_4)

    define_calls = mock_wb_4.calls['define_metric']
    metric_names = [c['name'] for c in define_calls]

    assert 'train/step' in metric_names, (
        f"'train/step' should be defined as metric, got {metric_names}"
    )
    assert 'train/*' in metric_names, (
        f"'train/*' should be defined for step-based x-axis, got {metric_names}"
    )

    # Verify step_metric is set for train/*
    train_star_call = next(c for c in define_calls if c['name'] == 'train/*')
    assert train_star_call.get('step_metric') == 'train/step', (
        f"train/* should use step_metric='train/step', got: {train_star_call}"
    )
    print(f"  PASS: Custom x-axis defined: {metric_names}")

    # ----------------------------------------------------------------
    # Test 5: alert on gradient spike
    # ----------------------------------------------------------------
    print("\nTest 5: alert sent on gradient spike...")

    mock_wb_5 = MockWandb()
    logger_rank0_5 = WandbLogger(cfg=MockConfig(), rank=0, wandb_module=mock_wb_5)

    # Log 20 normal steps with grad_norm ~0.5
    for i in range(20):
        logger_rank0_5.log_step(step=i, loss=1.0, grad_norm=0.5, tau=0.996, lr=1e-4, gpu_mem_gb=1.0)

    # Log a spike: 50.0 >> 10x * 0.5 = 5.0 threshold
    logger_rank0_5.log_step(step=20, loss=1.0, grad_norm=50.0, tau=0.996, lr=1e-4, gpu_mem_gb=1.0)

    assert len(mock_wb_5.calls['alert']) >= 1, (
        "wandb.alert should be called on gradient spike"
    )
    alert_call = mock_wb_5.calls['alert'][0]
    assert 'Gradient' in alert_call['title'] or 'gradient' in alert_call['title'].lower(), (
        f"Alert title should mention gradient: {alert_call['title']}"
    )
    print(f"  PASS: Alert triggered: '{alert_call['title']}'")

    # ----------------------------------------------------------------
    # Test 6: finish() is safe to call multiple times
    # ----------------------------------------------------------------
    print("\nTest 6: finish() is idempotent...")

    mock_wb_6 = MockWandb()
    logger_rank0_6 = WandbLogger(cfg=MockConfig(), rank=0, wandb_module=mock_wb_6)

    logger_rank0_6.finish()
    logger_rank0_6.finish()  # Second call should not raise or double-finish
    logger_rank0_6.finish()  # Third call also safe

    # wandb.finish() should be called at most once
    assert len(mock_wb_6.calls['finish']) <= 1, (
        f"wandb.finish() should be called at most once, got {len(mock_wb_6.calls['finish'])} calls"
    )
    print(f"  PASS: finish() called {len(mock_wb_6.calls['finish'])} time(s) despite 3 invocations")

    print("\n" + "=" * 60)
    print("All WandbLogger self-tests PASSED")
    print("=" * 60)
    sys.exit(0)
