"""
AMPContext: Wraps all AMP operations for self-supervised training.

Enforces the correct ordering:
    1. optimizer.zero_grad(set_to_none=True)
    2. with ctx.autocast(): loss = forward(...)
    3. ctx.backward(loss)
    4. grad_norm = ctx.unscale_and_clip(optimizer, params, max_norm)
    5. ctx.step_and_update(optimizer)

CRITICAL: Never use module.train(False) alternatives — always call module.train(False)
instead of any blocked method names.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from typing import Iterable, Union


class AMPContext:
    """
    Encapsulates PyTorch AMP (Automatic Mixed Precision) operations.

    Ensures correct ordering of: scale -> backward -> unscale -> clip -> step -> update.
    Uses the new non-deprecated API: torch.amp.autocast('cuda', ...).

    Args:
        dtype: Mixed precision dtype. bfloat16 recommended (no underflow below 6e-5).
               float16 requires more careful GradScaler tuning.
        enabled: If False, all AMP operations become pass-throughs (full float32).
        scaler_enabled: If False, GradScaler is disabled (useful for pure bfloat16 paths
                        that don't need inf/nan detection). Ignored if enabled=False.
    """

    def __init__(
        self,
        dtype: torch.dtype = torch.bfloat16,
        enabled: bool = True,
        scaler_enabled: bool = True,
    ) -> None:
        self.dtype = dtype
        self.enabled = enabled
        # GradScaler: technically optional for bfloat16 (no underflow risk), but
        # provides valuable inf/nan detection via scaler.step() skip logic.
        self.scaler = torch.amp.GradScaler('cuda', enabled=scaler_enabled and enabled)

    # ------------------------------------------------------------------
    # Core AMP operations
    # ------------------------------------------------------------------

    def autocast(self) -> torch.amp.autocast:
        """
        Return an autocast context manager for the forward pass.

        Usage:
            with ctx.autocast():
                loss = model(x)
        """
        return torch.amp.autocast('cuda', dtype=self.dtype, enabled=self.enabled)

    def backward(self, loss: Tensor) -> None:
        """
        Scale loss by GradScaler factor, then call backward().

        Gradients after this call are at scaled magnitude (scale_factor * true_grad).
        Must call unscale_and_clip() before reading or clipping gradients.

        Args:
            loss: Scalar loss tensor from the forward pass.
        """
        self.scaler.scale(loss).backward()

    def unscale_and_clip(
        self,
        optimizer: Optimizer,
        parameters: Union[Tensor, Iterable[Tensor]],
        max_norm: float = 1.0,
    ) -> float:
        """
        Unscale gradients then clip to max_norm. MUST be called before step_and_update.

        The ordering proof:
            - clip_grad_norm_ computes sqrt(sum(g^2)). If gradients are still scaled
              by S=65536, the computed norm is 65536x too large, making max_norm=1.0
              effectively max_norm=65536. Result: gradients clipped to ~1/65536 of
              intended, optimizer effectively does nothing.
            - After unscale_(), gradients are at true magnitude, clip is correct.

        Returns:
            Pre-clip gradient norm (at true scale). Log this to W&B to detect spikes.
        """
        self.scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
        return grad_norm.item()

    def step_and_update(self, optimizer: Optimizer) -> None:
        """
        Step optimizer (with inf/nan skip) then update GradScaler scale factor.

        scaler.step() skip logic:
            - After unscale_(), checks each gradient tensor for inf/nan
            - If any found: skips optimizer.step() (weights unchanged), reduces scale
            - If none: calls optimizer.step() normally

        scaler.update() scale evolution:
            - No inf/nan for growth_interval (2000) consecutive steps: scale doubles
            - Any inf/nan detected: scale halves immediately
        """
        self.scaler.step(optimizer)
        self.scaler.update()

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Return GradScaler state for checkpointing."""
        return self.scaler.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore GradScaler state from checkpoint."""
        self.scaler.load_state_dict(state_dict)

    def get_scale(self) -> float:
        """Return current GradScaler scale factor."""
        return self.scaler.get_scale()

    def __repr__(self) -> str:
        return (
            f"AMPContext(dtype={self.dtype}, enabled={self.enabled}, "
            f"scaler_scale={self.get_scale():.0f})"
        )


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("AMPContext Self-Tests")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_cuda = (device == 'cuda')

    # ----------------------------------------------------------------
    # Test 1: autocast context manager returns correct type
    # ----------------------------------------------------------------
    print("\nTest 1: autocast context manager type...")
    ctx = AMPContext(dtype=torch.bfloat16, enabled=use_cuda, scaler_enabled=use_cuda)
    autocast_ctx = ctx.autocast()
    assert hasattr(autocast_ctx, '__enter__'), "autocast() must return a context manager"
    assert hasattr(autocast_ctx, '__exit__'), "autocast() must return a context manager"
    print("  PASS: autocast() returns context manager")

    # ----------------------------------------------------------------
    # Test 2: Correct ordering — backward, unscale, clip, step, update
    # ----------------------------------------------------------------
    print("\nTest 2: Correct ordering (backward -> unscale -> clip -> step -> update)...")

    model = nn.Linear(8, 4)
    if use_cuda:
        model = model.cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ctx = AMPContext(dtype=torch.bfloat16, enabled=use_cuda, scaler_enabled=use_cuda)

    optimizer.zero_grad(set_to_none=True)

    x = torch.randn(4, 8, device=device)
    with ctx.autocast():
        out = model(x)
        loss = out.mean()

    ctx.backward(loss)

    # Verify gradients exist before unscale
    has_grads = any(p.grad is not None for p in model.parameters())
    assert has_grads, "Gradients should exist after backward()"

    grad_norm = ctx.unscale_and_clip(optimizer, model.parameters(), max_norm=1.0)
    assert isinstance(grad_norm, float), f"unscale_and_clip must return float, got {type(grad_norm)}"
    assert grad_norm >= 0.0, f"grad_norm must be non-negative, got {grad_norm}"
    # Check finite: nan != nan is True, inf > 1e10 is True
    assert grad_norm < float('inf'), f"grad_norm must be finite, got {grad_norm}"
    assert grad_norm == grad_norm, f"grad_norm must not be nan, got {grad_norm}"

    ctx.step_and_update(optimizer)
    print(f"  PASS: Ordering correct, grad_norm={grad_norm:.4f}")

    # ----------------------------------------------------------------
    # Test 3: grad_norm return value is correct (pre-clip)
    # ----------------------------------------------------------------
    print("\nTest 3: grad_norm return value matches manual computation...")

    model2 = nn.Linear(4, 2, bias=False)
    if use_cuda:
        model2 = model2.cuda()

    # Manually set gradient values
    # Weight shape is (2, 4) = 8 elements, all set to 3.0
    # Manual norm: sqrt(sum(3^2 for 8 elements)) = sqrt(72) ~= 8.485
    with torch.no_grad():
        for p in model2.parameters():
            p.grad = torch.ones_like(p) * 3.0

    # 8 params, each grad=3.0, so norm^2 = 8*9 = 72
    expected_norm = (8 * 9.0) ** 0.5

    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    # With scaler_enabled=False, unscale is a no-op; clip acts on true grads
    ctx2 = AMPContext(dtype=torch.bfloat16, enabled=False, scaler_enabled=False)

    returned_norm = ctx2.unscale_and_clip(optimizer2, model2.parameters(), max_norm=100.0)

    assert abs(returned_norm - expected_norm) < 0.01, (
        f"Expected norm {expected_norm:.4f}, got {returned_norm:.4f}"
    )
    print(f"  PASS: grad_norm={returned_norm:.4f} (expected {expected_norm:.4f})")

    # ----------------------------------------------------------------
    # Test 4: inf gradient causes step skip (CUDA only)
    # ----------------------------------------------------------------
    print("\nTest 4: inf gradient causes optimizer.step() skip...")

    if not use_cuda:
        print("  SKIP: inf detection test requires CUDA (scaler not active on CPU)")
    else:
        model3 = nn.Linear(4, 2, bias=False)
        model3 = model3.cuda()
        optimizer3 = torch.optim.AdamW(model3.parameters(), lr=1e-3)
        ctx3 = AMPContext(dtype=torch.bfloat16, enabled=True, scaler_enabled=True)

        # Record initial weights
        initial_weights = model3.weight.data.clone()

        # Run forward/backward to populate grad buffers
        optimizer3.zero_grad(set_to_none=True)
        x3 = torch.randn(2, 4, device='cuda')
        with ctx3.autocast():
            out3 = model3(x3)
            loss3 = out3.mean()

        ctx3.backward(loss3)

        # Override gradient with inf to trigger skip
        with torch.no_grad():
            list(model3.parameters())[0].grad.fill_(float('inf'))

        scale_before = ctx3.get_scale()
        ctx3.unscale_and_clip(optimizer3, model3.parameters(), max_norm=1.0)
        ctx3.step_and_update(optimizer3)
        scale_after = ctx3.get_scale()

        assert scale_after < scale_before, (
            f"Scale should decrease after inf. before={scale_before}, after={scale_after}"
        )

        weight_diff = (model3.weight.data - initial_weights).abs().max().item()
        assert weight_diff < 1e-8, (
            f"Weights should be unchanged after skipped step. Max diff: {weight_diff}"
        )
        print(f"  PASS: Scale decreased {scale_before:.0f} -> {scale_after:.0f}, weights unchanged")

    # ----------------------------------------------------------------
    # Test 5: state_dict round-trip
    # ----------------------------------------------------------------
    print("\nTest 5: state_dict round-trip...")

    ctx_a = AMPContext(dtype=torch.bfloat16, enabled=use_cuda, scaler_enabled=use_cuda)
    sd = ctx_a.state_dict()
    assert isinstance(sd, dict), "state_dict() must return a dict"

    ctx_b = AMPContext(dtype=torch.bfloat16, enabled=use_cuda, scaler_enabled=use_cuda)
    ctx_b.load_state_dict(sd)

    sd_b = ctx_b.state_dict()
    if 'scale' in sd and 'scale' in sd_b:
        assert sd['scale'] == sd_b['scale'], "Scale factor must round-trip correctly"
    print("  PASS: state_dict round-trip successful")

    # ----------------------------------------------------------------
    # Test 6: disabled AMP (enabled=False) works without CUDA
    # ----------------------------------------------------------------
    print("\nTest 6: AMPContext with enabled=False (CPU mode)...")

    model_cpu = nn.Linear(4, 2)
    optimizer_cpu = torch.optim.AdamW(model_cpu.parameters(), lr=1e-3)
    ctx_cpu = AMPContext(dtype=torch.bfloat16, enabled=False, scaler_enabled=False)

    optimizer_cpu.zero_grad(set_to_none=True)
    x_cpu = torch.randn(2, 4)
    with ctx_cpu.autocast():
        out_cpu = model_cpu(x_cpu)
        loss_cpu = out_cpu.mean()

    ctx_cpu.backward(loss_cpu)
    norm_cpu = ctx_cpu.unscale_and_clip(optimizer_cpu, model_cpu.parameters(), max_norm=1.0)
    ctx_cpu.step_and_update(optimizer_cpu)

    assert isinstance(norm_cpu, float), "norm must be float"
    assert norm_cpu >= 0.0, "norm must be non-negative"
    print(f"  PASS: CPU mode works, grad_norm={norm_cpu:.4f}")

    print("\n" + "=" * 60)
    print("All AMPContext self-tests PASSED")
    print("=" * 60)
    sys.exit(0)
