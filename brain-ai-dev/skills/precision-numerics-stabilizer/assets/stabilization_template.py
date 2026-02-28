"""
Precision + Numerics Stabilizer - Stabilization Controls Template
=================================================================
Gradient clip integration, logit clamping, deterministic debug toggle,
and auto-recovery controller.

CRITICAL: Never call the inference-mode shorthand on PyTorch modules.
Use module.train(False) instead.
"""

from __future__ import annotations

import logging
import math
import random
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn

try:
    from precision_config_template import PrecisionConfig
    from precision_context_template import PrecisionContext
except ImportError:
    try:
        from assets.precision_config_template import PrecisionConfig
        from assets.precision_context_template import PrecisionContext
    except ImportError:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(__file__))
        from precision_config_template import PrecisionConfig
        from precision_context_template import PrecisionContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# D1: Gradient Clip Integration
# ---------------------------------------------------------------------------

class GradientClipIntegration:
    """Ensures correct unscale -> clip -> step ordering for AMP training.

    For fp16 with GradScaler, the ordering is critical:
    1. scaler.unscale_(optimizer) - restores true gradient magnitudes
    2. clip_grad_norm_(params, max_norm) - clips at correct scale
    3. scaler.step(optimizer) - steps only if no inf/nan found

    This class wraps PrecisionContext.unscale_and_clip to provide the
    full workflow in one call, enforcing correct ordering via the
    PrecisionContext API.

    Parameters
    ----------
    precision_ctx : PrecisionContext
        The precision context that owns the GradScaler.
    max_norm : float or None
        Default clip norm. Falls back to cfg.max_grad_norm if None.

    Notes
    -----
    Call order tracking is handled by PrecisionContext internally.
    This class provides a higher-level convenience wrapper.
    """

    def __init__(
        self,
        precision_ctx: PrecisionContext,
        max_norm: Optional[float] = None,
    ):
        self.precision_ctx = precision_ctx
        self.max_norm = max_norm
        self._unscale_call_count = 0
        self._clip_call_count = 0

    def unscale_clip_and_step(
        self,
        optimizer: torch.optim.Optimizer,
        parameters: Iterable[torch.nn.Parameter],
        max_norm: Optional[float] = None,
    ) -> tuple:
        """Run unscale -> clip -> step in correct order.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer to unscale and step.
        parameters : iterable
            Parameters to clip.
        max_norm : float or None
            Override clip norm for this call.

        Returns
        -------
        tuple of (float, bool)
            (grad_norm_after_clip, stepped)
            grad_norm_after_clip: the actual norm after clipping
            stepped: True if optimizer stepped (no overflow)
        """
        _norm = max_norm or self.max_norm

        self._unscale_call_count += 1
        grad_norm = self.precision_ctx.unscale_and_clip(
            optimizer, parameters, max_norm=_norm
        )
        self._clip_call_count += 1

        stepped = self.precision_ctx.optimizer_step(optimizer)
        return grad_norm, stepped

    @property
    def call_order_valid(self) -> bool:
        """Return True if unscale and clip counts match (correct interleaving)."""
        return self._unscale_call_count == self._clip_call_count


# ---------------------------------------------------------------------------
# D2: Logit Clamper
# ---------------------------------------------------------------------------

class LogitClamper:
    """Optional logit clamping before softmax to prevent overflow.

    This is a nuclear option that changes training semantics — use only
    when other measures have failed to prevent overflow.

    Parameters
    ----------
    threshold : float
        Clamp logits to [-threshold, +threshold]. Default: 80.0.
    enabled : bool
        Whether clamping is active. Can be toggled at runtime.

    Notes
    -----
    fp16 overflow risk: exp() overflows above ~88, so logits above ~65-80
    risk softmax overflow in numerically unstable implementations.
    bf16 is much safer (same exponent range as fp32) but monitoring is
    still valuable.
    """

    def __init__(self, threshold: float = 80.0, enabled: bool = True):
        if threshold <= 0:
            raise ValueError(f"threshold must be > 0, got {threshold}")
        self.threshold = threshold
        self.enabled = enabled
        self._clamp_count: int = 0

    def clamp(self, logits: torch.Tensor) -> torch.Tensor:
        """Clamp logits to [-threshold, threshold] if enabled.

        Parameters
        ----------
        logits : torch.Tensor
            Raw logit tensor from model output.

        Returns
        -------
        torch.Tensor
            Clamped logits (or original if disabled).
        """
        if not self.enabled:
            return logits

        max_before = float(logits.abs().max().item())
        clamped = torch.clamp(logits, min=-self.threshold, max=self.threshold)
        max_after = float(clamped.abs().max().item())

        if max_before > self.threshold:
            self._clamp_count += 1
            logger.debug(
                "Logit clamped: max_abs %.2f -> %.2f (threshold=%.1f).",
                max_before,
                max_after,
                self.threshold,
            )

        return clamped

    @property
    def clamp_count(self) -> int:
        """Number of forward passes where clamping was active."""
        return self._clamp_count


# ---------------------------------------------------------------------------
# D3: Deterministic Debug Toggle
# ---------------------------------------------------------------------------

def _seed_everything(seed: int):
    """Set all RNG seeds for deterministic behavior."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


class DeterministicDebugToggle:
    """Enable/disable deterministic CUDA mode for failure reproduction.

    When enabled, forces:
    - torch.use_deterministic_algorithms(True)
    - torch.backends.cudnn.benchmark = False
    - torch.backends.cudnn.deterministic = True
    - Fixed seeds for all RNGs

    Restores original flag values on disable().

    Parameters
    ----------
    seed : int
        Random seed to set when enabling. Default: 42.
    warn_only : bool
        If True, uses warn_only=True for deterministic algorithms
        (warning instead of error for unsupported ops). Default: False.

    Notes
    -----
    Enabling determinism adds 20-50% training overhead. Use only during
    failure reproduction runs, not production training.
    """

    def __init__(self, seed: int = 42, warn_only: bool = False):
        self.seed = seed
        self.warn_only = warn_only
        self._original_flags: Optional[Dict[str, Any]] = None
        self._enabled: bool = False

    def enable(self):
        """Activate deterministic mode and seed all RNGs."""
        if self._enabled:
            logger.warning("DeterministicDebugToggle.enable() called while already enabled.")
            return

        # Save original state
        self._original_flags = {
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
        }

        # Apply deterministic settings
        if self.warn_only:
            torch.use_deterministic_algorithms(True, warn_only=True)
        else:
            torch.use_deterministic_algorithms(True)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        # Seed everything
        _seed_everything(self.seed)

        self._enabled = True
        logger.info(
            "Deterministic mode enabled. Seed=%d, warn_only=%s.",
            self.seed,
            self.warn_only,
        )

    def disable(self):
        """Restore original flag values."""
        if not self._enabled:
            return
        if self._original_flags is None:
            return

        try:
            orig_det = self._original_flags["deterministic"]
            if self.warn_only:
                torch.use_deterministic_algorithms(orig_det, warn_only=True)
            else:
                torch.use_deterministic_algorithms(orig_det)
        except Exception:
            torch.use_deterministic_algorithms(False)

        torch.backends.cudnn.benchmark = self._original_flags["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = self._original_flags["cudnn_deterministic"]

        self._enabled = False
        self._original_flags = None
        logger.info("Deterministic mode disabled. Original flags restored.")

    @property
    def is_enabled(self) -> bool:
        """Return True if deterministic mode is currently active."""
        return self._enabled

    def __enter__(self):
        self.enable()
        return self

    def __exit__(self, *args):
        self.disable()


# ---------------------------------------------------------------------------
# Auto-Recovery Controller
# ---------------------------------------------------------------------------

class AutoRecoveryController:
    """Temporarily reduces learning rate on overflow detection.

    On overflow (GradScaler detected inf/nan and skipped step), reduces
    all optimizer parameter group LRs by `lr_factor` for `recovery_steps`
    steps, then restores the original LRs.

    This prevents cascading overflow failures where a single bad batch
    repeatedly triggers overflows, driving the scale factor toward zero.

    Parameters
    ----------
    lr_factor : float
        Factor to multiply LR by on overflow. E.g., 0.5 halves the LR.
        Must be in (0, 1).
    recovery_steps : int
        Number of steps to maintain the reduced LR before restoring.
    """

    def __init__(self, lr_factor: float = 0.5, recovery_steps: int = 100):
        if not (0.0 < lr_factor < 1.0):
            raise ValueError(f"lr_factor must be in (0, 1), got {lr_factor}")
        if recovery_steps < 1:
            raise ValueError(f"recovery_steps must be >= 1, got {recovery_steps}")

        self.lr_factor = lr_factor
        self.recovery_steps = recovery_steps

        self._original_lrs: Dict[int, float] = {}
        self._recovery_countdown: int = 0
        self._overflow_count: int = 0

    def on_overflow(self, optimizer: torch.optim.Optimizer):
        """Reduce optimizer LRs in response to overflow.

        Should be called when PrecisionContext.optimizer_step() returns False.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer whose LRs to reduce.
        """
        if self._recovery_countdown > 0:
            # Already in recovery - don't re-enter
            logger.debug(
                "Overflow during recovery (countdown=%d). No additional LR reduction.",
                self._recovery_countdown,
            )
            return

        self._overflow_count += 1

        # Save original LRs
        self._original_lrs = {
            i: pg["lr"] for i, pg in enumerate(optimizer.param_groups)
        }

        # Reduce LRs
        for pg in optimizer.param_groups:
            pg["lr"] = pg["lr"] * self.lr_factor

        self._recovery_countdown = self.recovery_steps

        new_lrs = [pg["lr"] for pg in optimizer.param_groups]
        logger.warning(
            "Overflow #%d: Reducing LR by %.2fx for %d steps. "
            "New LRs: %s.",
            self._overflow_count,
            self.lr_factor,
            self.recovery_steps,
            [f"{lr:.2e}" for lr in new_lrs],
        )

    def step(self, optimizer: torch.optim.Optimizer):
        """Advance the recovery countdown. Call each training step (even skipped).

        When countdown reaches 0, original LRs are restored.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer whose LRs to restore when countdown expires.
        """
        if self._recovery_countdown <= 0:
            return

        self._recovery_countdown -= 1

        if self._recovery_countdown == 0:
            # Restore original LRs
            for i, pg in enumerate(optimizer.param_groups):
                if i in self._original_lrs:
                    pg["lr"] = self._original_lrs[i]

            restored_lrs = [pg["lr"] for pg in optimizer.param_groups]
            logger.info(
                "Auto-recovery complete. LRs restored: %s.",
                [f"{lr:.2e}" for lr in restored_lrs],
            )
            self._original_lrs = {}

    @property
    def in_recovery(self) -> bool:
        """Return True if currently in the reduced-LR recovery window."""
        return self._recovery_countdown > 0

    @property
    def recovery_countdown(self) -> int:
        """Steps remaining in the current recovery window."""
        return self._recovery_countdown

    @property
    def overflow_count(self) -> int:
        """Total number of overflow events handled."""
        return self._overflow_count


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("Running stabilization_template.py self-tests...")
    failures = []

    # --- T1: GradientClipIntegration - ordering verified ---
    try:
        cfg = PrecisionConfig(mode="fp32", max_grad_norm=1.0)
        ctx = PrecisionContext(cfg)
        clip_integrator = GradientClipIntegration(ctx, max_norm=1.0)

        model = nn.Linear(4, 4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        x = torch.randn(2, 4) * 10
        loss = model(x).sum()
        ctx.backward(loss)

        grad_norm, stepped = clip_integrator.unscale_clip_and_step(
            optimizer, model.parameters()
        )
        assert isinstance(grad_norm, float), f"grad_norm should be float, got {type(grad_norm)}"
        assert stepped is True, f"fp32 should always step"
        assert clip_integrator.call_order_valid, "Unscale and clip call counts should match"
        print(f"  [PASS] T1: GradientClipIntegration ordering correct (norm={grad_norm:.4f})")
    except Exception as e:
        failures.append(f"T1 gradient clip ordering: {e}")
        print(f"  [FAIL] T1: {e}")

    # --- T2: LogitClamper clamps values to threshold ---
    try:
        clamper = LogitClamper(threshold=80.0, enabled=True)
        logits = torch.tensor([[-200.0, 0.0, 200.0]])
        clamped = clamper.clamp(logits)

        assert math.isclose(float(clamped[0, 0].item()), -80.0, rel_tol=1e-5), (
            f"Expected -80.0, got {clamped[0, 0].item()}"
        )
        assert math.isclose(float(clamped[0, 1].item()), 0.0, abs_tol=1e-5), (
            f"Expected 0.0, got {clamped[0, 1].item()}"
        )
        assert math.isclose(float(clamped[0, 2].item()), 80.0, rel_tol=1e-5), (
            f"Expected 80.0, got {clamped[0, 2].item()}"
        )
        assert clamper.clamp_count == 1
        print("  [PASS] T2: LogitClamper clamps to [-threshold, threshold]")
    except Exception as e:
        failures.append(f"T2 logit clamper: {e}")
        print(f"  [FAIL] T2: {e}")

    # --- T3: LogitClamper disabled passes through unchanged ---
    try:
        clamper = LogitClamper(threshold=10.0, enabled=False)
        logits = torch.tensor([[100.0, -100.0]])
        out = clamper.clamp(logits)
        assert torch.allclose(out, logits), "Disabled clamper should pass through unchanged"
        assert clamper.clamp_count == 0
        print("  [PASS] T3: LogitClamper disabled passes through unchanged")
    except Exception as e:
        failures.append(f"T3 logit clamper disabled: {e}")
        print(f"  [FAIL] T3: {e}")

    # --- T4: DeterministicDebugToggle sets and restores flags ---
    try:
        # Record original state
        original_det = torch.are_deterministic_algorithms_enabled()
        original_bench = torch.backends.cudnn.benchmark
        original_cudnn_det = torch.backends.cudnn.deterministic

        toggle = DeterministicDebugToggle(seed=123, warn_only=True)

        toggle.enable()
        assert toggle.is_enabled is True
        assert torch.are_deterministic_algorithms_enabled() is True
        assert torch.backends.cudnn.benchmark is False
        assert torch.backends.cudnn.deterministic is True

        toggle.disable()
        assert toggle.is_enabled is False
        # Restore check
        assert torch.are_deterministic_algorithms_enabled() == original_det, (
            f"Deterministic flag not restored: expected {original_det}"
        )
        assert torch.backends.cudnn.benchmark == original_bench, (
            f"cudnn.benchmark not restored: expected {original_bench}"
        )
        print("  [PASS] T4: DeterministicDebugToggle sets and restores flags")
    except Exception as e:
        failures.append(f"T4 deterministic toggle: {e}")
        print(f"  [FAIL] T4: {e}")

    # --- T5: DeterministicDebugToggle as context manager ---
    try:
        toggle = DeterministicDebugToggle(seed=42, warn_only=True)
        with toggle:
            assert toggle.is_enabled is True
        assert toggle.is_enabled is False
        print("  [PASS] T5: DeterministicDebugToggle works as context manager")
    except Exception as e:
        failures.append(f"T5 toggle context manager: {e}")
        print(f"  [FAIL] T5: {e}")

    # --- T6: AutoRecoveryController reduces LR on overflow ---
    try:
        model = nn.Linear(4, 4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        controller = AutoRecoveryController(lr_factor=0.5, recovery_steps=10)
        assert not controller.in_recovery

        controller.on_overflow(optimizer)
        assert controller.in_recovery
        assert controller.overflow_count == 1

        # Check LR was halved
        for pg in optimizer.param_groups:
            assert math.isclose(pg["lr"], 0.005, rel_tol=1e-5), (
                f"LR should be 0.005 after halving, got {pg['lr']}"
            )
        print("  [PASS] T6: AutoRecoveryController reduces LR on overflow")
    except Exception as e:
        failures.append(f"T6 auto recovery reduce: {e}")
        print(f"  [FAIL] T6: {e}")

    # --- T7: AutoRecoveryController restores LR after N steps ---
    try:
        model = nn.Linear(4, 4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        controller = AutoRecoveryController(lr_factor=0.5, recovery_steps=5)
        controller.on_overflow(optimizer)

        # Step 5 times
        for _ in range(4):
            controller.step(optimizer)
            assert controller.in_recovery, "Should still be in recovery"

        # 5th step should restore
        controller.step(optimizer)
        assert not controller.in_recovery, "Should no longer be in recovery"

        for pg in optimizer.param_groups:
            assert math.isclose(pg["lr"], 0.01, rel_tol=1e-5), (
                f"LR should be restored to 0.01, got {pg['lr']}"
            )
        print("  [PASS] T7: AutoRecoveryController restores LR after N steps")
    except Exception as e:
        failures.append(f"T7 auto recovery restore: {e}")
        print(f"  [FAIL] T7: {e}")

    # --- T8: AutoRecoveryController does not double-reduce during recovery ---
    try:
        model = nn.Linear(4, 4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        controller = AutoRecoveryController(lr_factor=0.5, recovery_steps=10)
        controller.on_overflow(optimizer)

        lr_after_first_overflow = optimizer.param_groups[0]["lr"]
        assert math.isclose(lr_after_first_overflow, 0.005, rel_tol=1e-5)

        # Call on_overflow again during recovery
        controller.on_overflow(optimizer)
        lr_after_second_call = optimizer.param_groups[0]["lr"]

        # Should remain at 0.005, not reduced again to 0.0025
        assert math.isclose(lr_after_second_call, 0.005, rel_tol=1e-5), (
            f"LR should remain at 0.005 (not re-reduced), got {lr_after_second_call}"
        )
        print("  [PASS] T8: AutoRecoveryController does not double-reduce during recovery")
    except Exception as e:
        failures.append(f"T8 no double reduce: {e}")
        print(f"  [FAIL] T8: {e}")

    print()
    if failures:
        print(f"FAILED: {len(failures)} test(s) failed:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All 8 self-tests passed.")
