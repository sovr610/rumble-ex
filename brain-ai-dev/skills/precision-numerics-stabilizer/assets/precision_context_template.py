"""
Precision + Numerics Stabilizer - PrecisionContext Template
===========================================================
Single wrapper for all AMP operations. Eliminates ordering mistakes between
autocast, backward, gradient clipping, and optimizer step.

CRITICAL: Never use the bare .eval() method on any module.
Use module.train(False) instead.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn as nn

# Import config from companion template (allow standalone use too)
try:
    from precision_config_template import PrecisionConfig
except ImportError:
    try:
        from assets.precision_config_template import PrecisionConfig
    except ImportError:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(__file__))
        from precision_config_template import PrecisionConfig

logger = logging.getLogger(__name__)


class PrecisionContext:
    """Wraps all AMP operations for a single precision mode.

    Responsibilities
    ----------------
    - Enter autocast context during forward pass
    - Scale loss and call backward for fp16 (GradScaler.scale)
    - Unscale gradients and clip norms in correct order
    - Step optimizer and detect overflow via get_scale() comparison
    - Track skip counters and effective update rate
    - Provide state_dict / load_state_dict for checkpoint compatibility

    Usage
    -----
    ::

        ctx = PrecisionContext(PrecisionConfig(mode="bf16"))

        optimizer.zero_grad()
        with ctx.autocast_ctx():
            logits = model(batch)
            loss = criterion(logits, labels)

        ctx.backward(loss)
        grad_norm = ctx.unscale_and_clip(optimizer, model.parameters())
        stepped = ctx.optimizer_step(optimizer)

    Notes
    -----
    - For fp32/bf16 (no GradScaler): backward is loss.backward(),
      unscale_and_clip clips directly, optimizer_step always returns True.
    - For fp16 (GradScaler enabled): backward scales loss, unscale_and_clip
      calls scaler.unscale_() FIRST, then clips. optimizer_step compares
      get_scale() before/after update() to detect overflow.

    IMPORTANT: When putting a model in inference mode, always call
    module.train(False) rather than the deprecated pattern.
    """

    def __init__(self, cfg: PrecisionConfig):
        self.cfg = cfg
        self._dtype = self._resolve_dtype()
        self._scaler_enabled = cfg.resolved_scaler_enabled

        # Build GradScaler (disabled for fp32 and bf16)
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self._scaler_enabled,
            init_scale=cfg.grad_scaler_init_scale,
            growth_factor=cfg.grad_scaler_growth_factor,
            backoff_factor=cfg.grad_scaler_backoff_factor,
            growth_interval=cfg.grad_scaler_growth_interval,
        )

        # Overflow tracking counters
        self.num_steps_total: int = 0
        self.num_steps_skipped: int = 0
        self.last_overflow_step: int = -1
        self._last_scale: float = cfg.grad_scaler_init_scale

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_dtype(self) -> Optional[torch.dtype]:
        """Convert config mode to torch.dtype for autocast, or None for fp32."""
        return self.cfg.resolved_autocast_dtype

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def autocast_ctx(self):
        """Return an appropriate context manager for the forward pass.

        Returns
        -------
        context manager
            - fp32: contextlib.nullcontext() - no cast
            - bf16: torch.amp.autocast('cuda', dtype=torch.bfloat16)
            - fp16: torch.amp.autocast('cuda', dtype=torch.float16)
        """
        if self.cfg.mode == "fp32":
            return contextlib.nullcontext()
        return torch.amp.autocast("cuda", dtype=self._dtype, enabled=True)

    def backward(self, loss: torch.Tensor) -> None:
        """Call backward, scaling loss if GradScaler is enabled.

        Parameters
        ----------
        loss : torch.Tensor
            Scalar loss tensor from the forward pass.
        """
        if self._scaler_enabled:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def unscale_and_clip(
        self,
        optimizer: torch.optim.Optimizer,
        parameters: Iterable[torch.nn.Parameter],
        max_norm: Optional[float] = None,
    ) -> float:
        """Unscale gradients (fp16 only), then clip to max_norm.

        The ordering is critical for fp16:
        1. scaler.unscale_(optimizer) - divide grads by scale factor
        2. clip_grad_norm_(parameters, max_norm) - clip at correct magnitude

        For fp32/bf16, step 1 is skipped (no scaling active).

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer whose parameter groups' grads to unscale.
        parameters : iterable of Parameter
            Parameters whose grads to clip (typically model.parameters()).
        max_norm : float or None
            Clip threshold. Defaults to cfg.max_grad_norm if None.

        Returns
        -------
        float
            The clipped global gradient L2 norm (after unscaling, before update).
        """
        _max_norm = max_norm if max_norm is not None else self.cfg.max_grad_norm

        if self._scaler_enabled:
            # MUST unscale before clip - otherwise norm is inflated by scale factor
            self.scaler.unscale_(optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, _max_norm)
        return float(grad_norm)

    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> bool:
        """Step the optimizer, detecting overflow via GradScaler get_scale().

        For fp16 with GradScaler:
        - Records scale_before
        - Calls scaler.step(optimizer) - skips if inf/nan detected
        - Calls scaler.update()
        - Compares scale_after < scale_before to detect overflow

        For fp32/bf16:
        - Calls optimizer.step() directly
        - Always returns True

        Returns
        -------
        bool
            True if optimizer actually stepped (no overflow).
            False if step was skipped due to gradient overflow.
        """
        self.num_steps_total += 1

        if self._scaler_enabled:
            scale_before = self.scaler.get_scale()
            self.scaler.step(optimizer)
            self.scaler.update()
            scale_after = self.scaler.get_scale()

            # Overflow detection: scale is reduced only on overflow
            stepped = scale_after >= scale_before
            if not stepped:
                self.num_steps_skipped += 1
                self.last_overflow_step = self.num_steps_total
                logger.warning(
                    "GradScaler overflow detected at step %d. "
                    "Scale: %.1f -> %.1f. Optimizer step SKIPPED.",
                    self.num_steps_total,
                    scale_before,
                    scale_after,
                )
            self._last_scale = scale_after

            # Warn on high skip rate (computed from totals)
            if self.num_steps_total >= 50 and self.skip_rate > 0.05:
                logger.warning(
                    "fp16 skip rate %.1f%% over %d steps (%d skipped). "
                    "Effective update rate: %.1f%%. "
                    "Current scale: %.1f.",
                    self.skip_rate * 100,
                    self.num_steps_total,
                    self.num_steps_skipped,
                    self.effective_update_rate * 100,
                    scale_after,
                )
            return stepped
        else:
            optimizer.step()
            return True

    # ------------------------------------------------------------------
    # Metrics properties
    # ------------------------------------------------------------------

    @property
    def effective_update_rate(self) -> float:
        """Fraction of steps where the optimizer actually updated parameters."""
        if self.num_steps_total == 0:
            return 1.0
        return 1.0 - self.num_steps_skipped / self.num_steps_total

    @property
    def skip_rate(self) -> float:
        """Fraction of steps where the optimizer step was skipped (overflow)."""
        if self.num_steps_total == 0:
            return 0.0
        return self.num_steps_skipped / self.num_steps_total

    @property
    def current_scale(self) -> float:
        """Current GradScaler loss scale factor."""
        if self._scaler_enabled:
            return self.scaler.get_scale()
        return 1.0

    # ------------------------------------------------------------------
    # Checkpoint support
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        """Return state dict for checkpoint saving."""
        state = {
            "num_steps_total": self.num_steps_total,
            "num_steps_skipped": self.num_steps_skipped,
            "last_overflow_step": self.last_overflow_step,
            "mode": self.cfg.mode,
        }
        if self._scaler_enabled:
            state["scaler"] = self.scaler.state_dict()
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore state from a previously saved state dict."""
        self.num_steps_total = state.get("num_steps_total", 0)
        self.num_steps_skipped = state.get("num_steps_skipped", 0)
        self.last_overflow_step = state.get("last_overflow_step", -1)
        if self._scaler_enabled and "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("Running precision_context_template.py self-tests...")
    failures = []

    # --- T1: fp32 mode returns nullcontext ---
    try:
        cfg = PrecisionConfig(mode="fp32")
        ctx = PrecisionContext(cfg)
        cm = ctx.autocast_ctx()
        # nullcontext has no special __class__ guarantee, but entering it is safe
        with cm:
            pass
        assert ctx._scaler_enabled is False
        assert ctx._dtype is None
        print("  [PASS] T1: fp32 mode - nullcontext, scaler disabled")
    except Exception as e:
        failures.append(f"T1 fp32 mode: {e}")
        print(f"  [FAIL] T1: {e}")

    # --- T2: bf16 mode dtype and scaler state ---
    try:
        cfg = PrecisionConfig(mode="bf16")
        ctx = PrecisionContext(cfg)
        assert ctx._scaler_enabled is False
        assert ctx._dtype == torch.bfloat16
        if not torch.cuda.is_available():
            print("  [SKIP] T2: bf16 autocast - CUDA not available (partial pass)")
        else:
            print("  [PASS] T2: bf16 mode - bfloat16 dtype, scaler disabled")
    except Exception as e:
        failures.append(f"T2 bf16 mode: {e}")
        print(f"  [FAIL] T2: {e}")

    # --- T3: fp16 mode enables scaler ---
    try:
        cfg = PrecisionConfig(mode="fp16")
        ctx = PrecisionContext(cfg)
        assert ctx._scaler_enabled is True
        assert ctx._dtype == torch.float16
        assert ctx.scaler.get_scale() == cfg.grad_scaler_init_scale
        print("  [PASS] T3: fp16 mode - float16 dtype, scaler enabled at init_scale")
    except Exception as e:
        failures.append(f"T3 fp16 mode: {e}")
        print(f"  [FAIL] T3: {e}")

    # --- T4: effective_update_rate and skip_rate with no steps ---
    try:
        cfg = PrecisionConfig(mode="fp32")
        ctx = PrecisionContext(cfg)
        assert ctx.effective_update_rate == 1.0
        assert ctx.skip_rate == 0.0
        print("  [PASS] T4: Initial rates correct (no steps)")
    except Exception as e:
        failures.append(f"T4 initial rates: {e}")
        print(f"  [FAIL] T4: {e}")

    # --- T5: fp32 optimizer_step always returns True ---
    try:
        cfg = PrecisionConfig(mode="fp32")
        ctx = PrecisionContext(cfg)

        model = nn.Linear(4, 4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        x = torch.randn(2, 4)
        loss = model(x).sum()
        ctx.backward(loss)
        stepped = ctx.optimizer_step(optimizer)

        assert stepped is True, f"fp32 should always step, got {stepped}"
        assert ctx.num_steps_total == 1
        assert ctx.num_steps_skipped == 0
        assert ctx.effective_update_rate == 1.0
        print("  [PASS] T5: fp32 optimizer_step always returns True")
    except Exception as e:
        failures.append(f"T5 fp32 step: {e}")
        print(f"  [FAIL] T5: {e}")

    # --- T6: Skip detection on injected inf gradient (fp16 mode, if CUDA) ---
    if torch.cuda.is_available():
        try:
            cfg = PrecisionConfig(mode="fp16", grad_scaler_init_scale=256.0)
            ctx = PrecisionContext(cfg)

            # Create a simple linear model on CUDA
            linear = nn.Linear(4, 4, device="cuda")
            optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

            # Run a normal scaled backward to initialize scaler state
            optimizer.zero_grad()
            x = torch.randn(2, 4, device="cuda")
            with ctx.autocast_ctx():
                out = linear(x)
            # Use scaler.scale to initialize _scale before unscale_
            ctx.scaler.scale(out.sum()).backward()

            # Now inject inf into gradients before unscale_
            for p in linear.parameters():
                if p.grad is not None:
                    p.grad.fill_(float("inf"))

            # Unscale will find inf, step will be skipped
            ctx.scaler.unscale_(optimizer)
            stepped = ctx.optimizer_step(optimizer)

            assert stepped is False, f"Expected skip on inf grad, got stepped={stepped}"
            assert ctx.num_steps_skipped == 1
            assert ctx.skip_rate == 1.0
            assert ctx.effective_update_rate == 0.0
            print("  [PASS] T6: Skip detection on injected inf gradient")
        except Exception as e:
            failures.append(f"T6 skip detection: {e}")
            print(f"  [FAIL] T6: {e}")
    else:
        print("  [SKIP] T6: CUDA not available")

    # --- T7: State dict round-trip ---
    try:
        cfg = PrecisionConfig(mode="fp32")
        ctx = PrecisionContext(cfg)
        ctx.num_steps_total = 100
        ctx.num_steps_skipped = 5
        ctx.last_overflow_step = 42

        state = ctx.state_dict()
        assert state["num_steps_total"] == 100
        assert state["num_steps_skipped"] == 5
        assert state["last_overflow_step"] == 42
        assert state["mode"] == "fp32"

        # Restore into fresh context
        ctx2 = PrecisionContext(cfg)
        ctx2.load_state_dict(state)
        assert ctx2.num_steps_total == 100
        assert ctx2.num_steps_skipped == 5
        assert ctx2.last_overflow_step == 42
        print("  [PASS] T7: State dict round-trip")
    except Exception as e:
        failures.append(f"T7 state dict: {e}")
        print(f"  [FAIL] T7: {e}")

    # --- T8: unscale_and_clip works for fp32 (no scaler) ---
    try:
        cfg = PrecisionConfig(mode="fp32", max_grad_norm=1.0)
        ctx = PrecisionContext(cfg)

        model = nn.Linear(8, 8)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        x = torch.randn(2, 8) * 100  # Large input to get large grads
        loss = model(x).sum()
        ctx.backward(loss)

        norm = ctx.unscale_and_clip(optimizer, model.parameters())
        assert isinstance(norm, float), f"Expected float norm, got {type(norm)}"
        assert norm >= 0, f"Norm should be non-negative, got {norm}"
        print(f"  [PASS] T8: unscale_and_clip works for fp32, norm={norm:.4f}")
    except Exception as e:
        failures.append(f"T8 unscale clip fp32: {e}")
        print(f"  [FAIL] T8: {e}")

    # --- T9: current_scale returns 1.0 for non-scaler modes ---
    try:
        cfg = PrecisionConfig(mode="bf16")
        ctx = PrecisionContext(cfg)
        assert ctx.current_scale == 1.0, f"Expected 1.0 for bf16, got {ctx.current_scale}"

        cfg_fp16 = PrecisionConfig(mode="fp16", grad_scaler_init_scale=512.0)
        ctx_fp16 = PrecisionContext(cfg_fp16)
        assert ctx_fp16.current_scale == 512.0, (
            f"Expected 512.0 for fp16, got {ctx_fp16.current_scale}"
        )
        print("  [PASS] T9: current_scale correct for all modes")
    except Exception as e:
        failures.append(f"T9 current_scale: {e}")
        print(f"  [FAIL] T9: {e}")

    print()
    if failures:
        print(f"FAILED: {len(failures)} test(s) failed:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All 9 self-tests passed (CUDA tests skipped if unavailable).")
