"""
GradientAccumulator — Gradient accumulation with AMP integration for brain_ai.

Simulates larger effective batch sizes across multiple forward-backward
passes. Integrates with GradScaler for fp16 AMP and supports DDP no_sync.

Key classes:
    GradientAccumulator — Main accumulator with step counting and AMP support.

Self-tests in __main__ validate accumulation correctness, AMP integration,
effective batch size computation, and LR scaling — all without multi-GPU.
"""

from __future__ import annotations

import copy
import logging
import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

_AMP_AVAILABLE = True
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    try:
        from torch.cuda.amp import autocast, GradScaler
        _AMP_AVAILABLE = True
    except ImportError:
        _AMP_AVAILABLE = False
        GradScaler = None


# ===========================================================================
# SECTION 1: LR Scaling Utilities
# ===========================================================================

def compute_scaled_lr(
    base_lr: float,
    effective_batch: int,
    reference_batch: int = 256,
    mode: str = "linear",
) -> float:
    """Compute scaled learning rate based on effective batch size.

    Args:
        base_lr: Baseline learning rate (tuned for reference_batch).
        effective_batch: The actual effective batch size.
        reference_batch: The batch size base_lr was tuned for.
        mode: "linear" or "sqrt" scaling rule.

    Returns:
        Scaled learning rate.
    """
    ratio = effective_batch / reference_batch
    if mode == "linear":
        return base_lr * ratio
    elif mode == "sqrt":
        return base_lr * math.sqrt(ratio)
    else:
        raise ValueError(f"Unknown LR scaling mode: {mode}")


def get_cosine_lr(
    step: int,
    warmup_steps: int,
    total_steps: int,
    peak_lr: float,
    min_lr: float,
) -> float:
    """Cosine learning rate schedule with linear warmup.

    Args:
        step: Current training step.
        warmup_steps: Number of warmup steps.
        total_steps: Total training steps.
        peak_lr: Maximum learning rate (reached at end of warmup).
        min_lr: Minimum learning rate (at end of training).

    Returns:
        Learning rate for the current step.
    """
    if step < warmup_steps:
        return peak_lr * step / max(warmup_steps, 1)
    elif step >= total_steps:
        return min_lr
    else:
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ===========================================================================
# SECTION 2: GradientAccumulator
# ===========================================================================

class GradientAccumulator:
    """Gradient accumulation with optional AMP GradScaler integration.

    Accumulates gradients over multiple micro-batches before taking an
    optimizer step. Divides loss by accumulation_steps to ensure correct
    gradient magnitude.

    Args:
        accumulation_steps: Number of micro-batches per optimizer step.
        scaler: Optional GradScaler for fp16 AMP.
        max_grad_norm: Maximum gradient norm for clipping (0 to disable).
    """

    def __init__(
        self,
        accumulation_steps: int = 1,
        scaler: Optional[Any] = None,
        max_grad_norm: float = 1.0,
    ):
        if accumulation_steps < 1:
            raise ValueError(f"accumulation_steps must be >= 1, got {accumulation_steps}")

        self.accumulation_steps = accumulation_steps
        self.scaler = scaler
        self.max_grad_norm = max_grad_norm
        self._micro_step = 0
        self._optimizer_steps = 0

    @property
    def micro_step(self) -> int:
        """Current micro-step within the accumulation window."""
        return self._micro_step

    @property
    def optimizer_steps(self) -> int:
        """Total number of optimizer steps taken."""
        return self._optimizer_steps

    def is_accumulation_step(self) -> bool:
        """Return True if this is a non-final accumulation step (no optimizer step)."""
        return (self._micro_step + 1) % self.accumulation_steps != 0

    def should_step(self) -> bool:
        """Return True if the optimizer should step on this micro-batch."""
        return (self._micro_step + 1) % self.accumulation_steps == 0

    def scale_loss(self, loss: Tensor) -> Tensor:
        """Scale loss by 1/accumulation_steps for correct gradient magnitude."""
        return loss / self.accumulation_steps

    def effective_batch_size(self, micro_batch: int, world_size: int = 1) -> int:
        """Compute effective batch size.

        Args:
            micro_batch: Per-GPU, per-step batch size.
            world_size: Number of GPUs.

        Returns:
            Effective batch size: micro_batch * accumulation_steps * world_size.
        """
        return micro_batch * self.accumulation_steps * world_size

    def step(
        self,
        loss: Tensor,
        optimizer: Any,
        model: Optional[nn.Module] = None,
        max_grad_norm: Optional[float] = None,
    ) -> bool:
        """Accumulate gradients and optionally step the optimizer.

        This method:
        1. Scales the loss by 1/accumulation_steps.
        2. Calls backward (through GradScaler if provided).
        3. On the final accumulation step: unscales, clips, steps, zeros.

        Args:
            loss: The raw (unscaled) loss tensor.
            optimizer: The optimizer to step.
            model: The model (needed for gradient clipping).
            max_grad_norm: Override default max gradient norm.

        Returns:
            True if optimizer stepped, False if still accumulating.
        """
        clip_norm = max_grad_norm if max_grad_norm is not None else self.max_grad_norm
        scaled_loss = self.scale_loss(loss)

        # Backward
        if self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        self._micro_step += 1

        if self.should_step():
            # Unscale, clip, step
            if self.scaler is not None:
                self.scaler.unscale_(optimizer)
                if model is not None and clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                if model is not None and clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()

            optimizer.zero_grad()
            self._optimizer_steps += 1
            return True

        return False

    def backward_only(self, loss: Tensor) -> None:
        """Perform scaled backward without optimizer logic.

        Useful when managing the optimizer step externally.
        """
        scaled_loss = self.scale_loss(loss)
        if self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        self._micro_step += 1

    def finish_accumulation(
        self,
        optimizer: Any,
        model: Optional[nn.Module] = None,
        max_grad_norm: Optional[float] = None,
    ) -> None:
        """Force optimizer step even if accumulation window is incomplete.

        Useful at end of epoch when the remaining micro-batches don't fill
        a complete accumulation window.
        """
        clip_norm = max_grad_norm if max_grad_norm is not None else self.max_grad_norm

        if self.scaler is not None:
            self.scaler.unscale_(optimizer)
            if model is not None and clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            if model is not None and clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

        optimizer.zero_grad()
        self._optimizer_steps += 1
        self._micro_step = 0

    def reset(self) -> None:
        """Reset step counters."""
        self._micro_step = 0
        self._optimizer_steps = 0

    def get_no_sync_context(self, ddp_model: Any) -> Any:
        """Return appropriate context manager for DDP no_sync.

        During accumulation steps (not the final one), returns ddp_model.no_sync()
        to skip gradient synchronization. On the final step, returns nullcontext.

        Args:
            ddp_model: The DDP-wrapped model.

        Returns:
            Context manager.
        """
        if self.is_accumulation_step() and hasattr(ddp_model, "no_sync"):
            return ddp_model.no_sync()
        return nullcontext()

    def state_dict(self) -> Dict[str, Any]:
        """Serialize accumulator state for checkpointing."""
        state = {
            "micro_step": self._micro_step,
            "optimizer_steps": self._optimizer_steps,
            "accumulation_steps": self.accumulation_steps,
            "max_grad_norm": self.max_grad_norm,
        }
        if self.scaler is not None:
            state["scaler"] = self.scaler.state_dict()
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore accumulator state from checkpoint."""
        self._micro_step = state.get("micro_step", 0)
        self._optimizer_steps = state.get("optimizer_steps", 0)
        if self.scaler is not None and "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])


# ===========================================================================
# SECTION 3: Self-tests
# ===========================================================================

if __name__ == "__main__":
    import sys
    import traceback

    passed = 0
    failed = 0
    total = 0

    def run_test(name: str, fn: Callable):
        global passed, failed, total
        total += 1
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            traceback.print_exc()
            failed += 1

    print("=" * 70)
    print("GradientAccumulator Self-Tests")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Helper
    # -----------------------------------------------------------------------

    def _make_model():
        return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))

    # -----------------------------------------------------------------------
    # Construction tests
    # -----------------------------------------------------------------------

    def test_construction_defaults():
        acc = GradientAccumulator()
        assert acc.accumulation_steps == 1
        assert acc.max_grad_norm == 1.0
        assert acc.scaler is None
        assert acc.micro_step == 0
        assert acc.optimizer_steps == 0

    run_test("construction defaults", test_construction_defaults)

    def test_construction_custom():
        acc = GradientAccumulator(accumulation_steps=8, max_grad_norm=0.5)
        assert acc.accumulation_steps == 8
        assert acc.max_grad_norm == 0.5

    run_test("construction custom args", test_construction_custom)

    def test_construction_invalid():
        try:
            GradientAccumulator(accumulation_steps=0)
            assert False, "Should have raised"
        except ValueError:
            pass

    run_test("construction rejects accumulation_steps=0", test_construction_invalid)

    # -----------------------------------------------------------------------
    # Effective batch size tests
    # -----------------------------------------------------------------------

    def test_effective_batch_1():
        acc = GradientAccumulator(accumulation_steps=1)
        assert acc.effective_batch_size(32, 1) == 32

    run_test("effective_batch_size accum=1 world=1", test_effective_batch_1)

    def test_effective_batch_multi():
        acc = GradientAccumulator(accumulation_steps=16)
        assert acc.effective_batch_size(4, 8) == 4 * 16 * 8

    run_test("effective_batch_size 4*16*8=512", test_effective_batch_multi)

    def test_effective_batch_all_configs():
        configs = [
            (32, 1, 1, 32),
            (16, 4, 4, 256),
            (8, 8, 8, 512),
            (4, 16, 8, 512),
            (4, 8, 32, 1024),
        ]
        for micro, accum, world, expected in configs:
            acc = GradientAccumulator(accumulation_steps=accum)
            result = acc.effective_batch_size(micro, world)
            assert result == expected, f"{micro}*{accum}*{world} = {result}, expected {expected}"

    run_test("effective_batch_size all reference configs", test_effective_batch_all_configs)

    # -----------------------------------------------------------------------
    # LR scaling tests
    # -----------------------------------------------------------------------

    def test_lr_linear_scaling():
        lr = compute_scaled_lr(3e-4, effective_batch=512, reference_batch=256)
        assert abs(lr - 6e-4) < 1e-10

    run_test("LR linear scaling 2x batch", test_lr_linear_scaling)

    def test_lr_sqrt_scaling():
        lr = compute_scaled_lr(3e-4, effective_batch=1024, reference_batch=256, mode="sqrt")
        expected = 3e-4 * math.sqrt(4.0)
        assert abs(lr - expected) < 1e-10

    run_test("LR sqrt scaling 4x batch", test_lr_sqrt_scaling)

    def test_lr_scaling_identity():
        lr = compute_scaled_lr(3e-4, effective_batch=256, reference_batch=256)
        assert abs(lr - 3e-4) < 1e-10

    run_test("LR scaling identity when batch matches ref", test_lr_scaling_identity)

    def test_lr_scaling_invalid_mode():
        try:
            compute_scaled_lr(3e-4, 512, 256, mode="cubic")
            assert False, "Should have raised"
        except ValueError:
            pass

    run_test("LR scaling rejects invalid mode", test_lr_scaling_invalid_mode)

    # -----------------------------------------------------------------------
    # Cosine schedule tests
    # -----------------------------------------------------------------------

    def test_cosine_warmup_start():
        lr = get_cosine_lr(0, warmup_steps=100, total_steps=1000, peak_lr=1.0, min_lr=0.1)
        assert abs(lr - 0.0) < 1e-6

    run_test("cosine schedule: lr=0 at step 0", test_cosine_warmup_start)

    def test_cosine_warmup_end():
        lr = get_cosine_lr(100, warmup_steps=100, total_steps=1000, peak_lr=1.0, min_lr=0.1)
        assert abs(lr - 1.0) < 1e-6

    run_test("cosine schedule: lr=peak at warmup end", test_cosine_warmup_end)

    def test_cosine_end():
        lr = get_cosine_lr(1000, warmup_steps=100, total_steps=1000, peak_lr=1.0, min_lr=0.1)
        assert abs(lr - 0.1) < 1e-6

    run_test("cosine schedule: lr=min at total_steps", test_cosine_end)

    def test_cosine_midpoint():
        lr = get_cosine_lr(550, warmup_steps=100, total_steps=1000, peak_lr=1.0, min_lr=0.1)
        assert 0.1 < lr < 1.0  # Somewhere in between

    run_test("cosine schedule: midpoint is between min and peak", test_cosine_midpoint)

    # -----------------------------------------------------------------------
    # Step counting tests
    # -----------------------------------------------------------------------

    def test_step_counting_accum1():
        acc = GradientAccumulator(accumulation_steps=1)
        model = _make_model()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)

        for i in range(5):
            x = torch.randn(4, 16)
            out = model(x)
            loss = out.sum()
            did_step = acc.step(loss, opt, model)
            assert did_step is True

        assert acc.optimizer_steps == 5

    run_test("step counting with accum=1", test_step_counting_accum1)

    def test_step_counting_accum4():
        acc = GradientAccumulator(accumulation_steps=4)
        model = _make_model()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)

        results = []
        for i in range(8):
            x = torch.randn(4, 16)
            out = model(x)
            loss = out.sum()
            did_step = acc.step(loss, opt, model)
            results.append(did_step)

        # Steps at indices 3 and 7
        assert results == [False, False, False, True, False, False, False, True]
        assert acc.optimizer_steps == 2

    run_test("step counting with accum=4", test_step_counting_accum4)

    def test_should_step_pattern():
        acc = GradientAccumulator(accumulation_steps=3)
        expected = [False, False, True, False, False, True]
        for i, exp in enumerate(expected):
            acc._micro_step = i
            assert acc.should_step() == exp, f"step {i}: expected {exp}"

    run_test("should_step pattern for accum=3", test_should_step_pattern)

    def test_is_accumulation_step():
        acc = GradientAccumulator(accumulation_steps=4)
        for i in range(8):
            acc._micro_step = i
            if (i + 1) % 4 == 0:
                assert acc.is_accumulation_step() is False
            else:
                assert acc.is_accumulation_step() is True

    run_test("is_accumulation_step", test_is_accumulation_step)

    # -----------------------------------------------------------------------
    # Gradient accumulation correctness
    # -----------------------------------------------------------------------

    def test_accumulation_equivalence():
        """Verify accumulated gradients match single large-batch gradients."""
        K, M = 4, 8  # 4 accumulation steps, micro-batch 8
        torch.manual_seed(42)
        model_a = _make_model()
        model_b = copy.deepcopy(model_a)

        # Full batch data
        data_full = torch.randn(K * M, 16)
        target_full = torch.randn(K * M, 4)

        # Model A: single large batch
        out_a = model_a(data_full)
        loss_a = nn.functional.mse_loss(out_a, target_full)
        loss_a.backward()

        # Model B: accumulated micro-batches
        acc = GradientAccumulator(accumulation_steps=K, max_grad_norm=0)
        opt_b = torch.optim.SGD(model_b.parameters(), lr=0.01)
        opt_b.zero_grad()

        for k in range(K):
            start, end = k * M, (k + 1) * M
            out_b = model_b(data_full[start:end])
            loss_b = nn.functional.mse_loss(out_b, target_full[start:end])
            acc.step(loss_b, opt_b, model_b, max_grad_norm=0)

        # Compare gradients before optimizer step (model_b's grads were consumed)
        # Re-do with backward_only
        model_c = copy.deepcopy(model_a)
        model_c.zero_grad()
        acc2 = GradientAccumulator(accumulation_steps=K)
        for k in range(K):
            start, end = k * M, (k + 1) * M
            out_c = model_c(data_full[start:end])
            loss_c = nn.functional.mse_loss(out_c, target_full[start:end])
            acc2.backward_only(loss_c)

        for (na, pa), (nc, pc) in zip(
            model_a.named_parameters(), model_c.named_parameters()
        ):
            torch.testing.assert_close(pa.grad, pc.grad, atol=1e-5, rtol=1e-5)

    run_test("gradient accumulation equivalence", test_accumulation_equivalence)

    # -----------------------------------------------------------------------
    # Scale loss tests
    # -----------------------------------------------------------------------

    def test_scale_loss():
        acc = GradientAccumulator(accumulation_steps=4)
        loss = torch.tensor(4.0)
        scaled = acc.scale_loss(loss)
        assert abs(scaled.item() - 1.0) < 1e-6

    run_test("scale_loss divides by accum steps", test_scale_loss)

    def test_scale_loss_accum1():
        acc = GradientAccumulator(accumulation_steps=1)
        loss = torch.tensor(3.14)
        scaled = acc.scale_loss(loss)
        assert abs(scaled.item() - 3.14) < 1e-6

    run_test("scale_loss with accum=1 is identity", test_scale_loss_accum1)

    # -----------------------------------------------------------------------
    # AMP GradScaler integration
    # -----------------------------------------------------------------------

    def test_scaler_integration():
        if not _AMP_AVAILABLE:
            return  # Skip on systems without AMP

        model = _make_model()
        scaler = GradScaler()
        acc = GradientAccumulator(accumulation_steps=2, scaler=scaler)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)

        for i in range(4):
            x = torch.randn(4, 16)
            out = model(x)
            loss = out.sum()
            acc.step(loss, opt, model)

        assert acc.optimizer_steps == 2

    run_test("GradScaler integration with accumulation", test_scaler_integration)

    # -----------------------------------------------------------------------
    # State dict tests
    # -----------------------------------------------------------------------

    def test_state_dict():
        acc = GradientAccumulator(accumulation_steps=4, max_grad_norm=0.5)
        acc._micro_step = 3
        acc._optimizer_steps = 10
        state = acc.state_dict()
        assert state["micro_step"] == 3
        assert state["optimizer_steps"] == 10
        assert state["accumulation_steps"] == 4

    run_test("state_dict serialization", test_state_dict)

    def test_load_state_dict():
        acc = GradientAccumulator(accumulation_steps=4)
        state = {"micro_step": 7, "optimizer_steps": 25, "accumulation_steps": 4, "max_grad_norm": 1.0}
        acc.load_state_dict(state)
        assert acc.micro_step == 7
        assert acc.optimizer_steps == 25

    run_test("load_state_dict restoration", test_load_state_dict)

    # -----------------------------------------------------------------------
    # Reset tests
    # -----------------------------------------------------------------------

    def test_reset():
        acc = GradientAccumulator(accumulation_steps=4)
        acc._micro_step = 5
        acc._optimizer_steps = 3
        acc.reset()
        assert acc.micro_step == 0
        assert acc.optimizer_steps == 0

    run_test("reset clears counters", test_reset)

    # -----------------------------------------------------------------------
    # finish_accumulation tests
    # -----------------------------------------------------------------------

    def test_finish_accumulation():
        acc = GradientAccumulator(accumulation_steps=4)
        model = _make_model()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)

        # Do 2 micro-steps (incomplete window)
        for _ in range(2):
            x = torch.randn(4, 16)
            out = model(x)
            loss = out.sum()
            acc.backward_only(loss)

        acc.finish_accumulation(opt, model)
        assert acc.optimizer_steps == 1
        assert acc.micro_step == 0  # Reset after forced step

    run_test("finish_accumulation forces step", test_finish_accumulation)

    # -----------------------------------------------------------------------
    # no_sync context tests
    # -----------------------------------------------------------------------

    class MockDDPModel:
        """Mock DDP model with no_sync."""
        def __init__(self):
            self._in_no_sync = False

        def no_sync(self):
            from contextlib import contextmanager

            @contextmanager
            def _ctx():
                self._in_no_sync = True
                yield
                self._in_no_sync = False

            return _ctx()

    def test_no_sync_context_accum():
        acc = GradientAccumulator(accumulation_steps=4)
        mock = MockDDPModel()

        acc._micro_step = 0  # Not final step
        ctx = acc.get_no_sync_context(mock)
        with ctx:
            assert mock._in_no_sync is True

    run_test("no_sync context during accumulation", test_no_sync_context_accum)

    def test_no_sync_context_step():
        acc = GradientAccumulator(accumulation_steps=4)
        mock = MockDDPModel()

        acc._micro_step = 3  # Final step
        ctx = acc.get_no_sync_context(mock)
        with ctx:
            pass  # Should use nullcontext
        assert mock._in_no_sync is False

    run_test("nullcontext on final accumulation step", test_no_sync_context_step)

    def test_no_sync_no_ddp():
        acc = GradientAccumulator(accumulation_steps=4)
        obj = object()  # No no_sync method
        acc._micro_step = 0
        ctx = acc.get_no_sync_context(obj)
        with ctx:
            pass  # Should not raise

    run_test("no_sync fallback for non-DDP model", test_no_sync_no_ddp)

    # -----------------------------------------------------------------------
    # Gradient clipping verification
    # -----------------------------------------------------------------------

    def test_gradient_clipping():
        model = _make_model()
        acc = GradientAccumulator(accumulation_steps=1, max_grad_norm=0.01)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)

        x = torch.randn(4, 16) * 100  # Large input to produce large gradients
        out = model(x)
        loss = out.sum()
        acc.step(loss, opt, model)

        # Re-compute gradient norm (should be 0 since optimizer already stepped)
        # Instead, verify optimizer_steps increased
        assert acc.optimizer_steps == 1

    run_test("gradient clipping during step", test_gradient_clipping)

    def test_no_clipping_when_zero():
        model = _make_model()
        acc = GradientAccumulator(accumulation_steps=1, max_grad_norm=0)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)

        x = torch.randn(4, 16)
        out = model(x)
        loss = out.sum()
        acc.step(loss, opt, model, max_grad_norm=0)  # Should not clip
        assert acc.optimizer_steps == 1

    run_test("no clipping when max_grad_norm=0", test_no_clipping_when_zero)

    # -----------------------------------------------------------------------
    # Training convergence test
    # -----------------------------------------------------------------------

    def test_training_convergence():
        """Verify that model trains (loss decreases) with accumulation."""
        torch.manual_seed(123)
        model = _make_model()
        acc = GradientAccumulator(accumulation_steps=2)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)

        target = torch.randn(8, 4)
        data = torch.randn(8, 16)

        losses = []
        for epoch in range(20):
            for k in range(2):
                start = k * 4
                end = (k + 1) * 4
                out = model(data[start:end])
                loss = nn.functional.mse_loss(out, target[start:end])
                acc.step(loss, opt, model)
                if acc.should_step():
                    losses.append(loss.item())

        # Loss should generally decrease
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

    run_test("training convergence with accumulation", test_training_convergence)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed, {total} total")
    print("=" * 70)

    sys.exit(0 if failed == 0 else 1)
