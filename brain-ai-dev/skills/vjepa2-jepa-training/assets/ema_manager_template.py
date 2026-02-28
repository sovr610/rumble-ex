"""
EMAManager -- Exponential Moving Average target encoder for V-JEPA 2.

Manages a deep copy of the context encoder that is updated via EMA each step.
The momentum follows a cosine schedule from ema_start to ema_end over total_steps.
No gradients flow through the target encoder.

Formula:
    theta_target = m * theta_target + (1 - m) * theta_encoder

Momentum schedule:
    m(t) = ema_end - (ema_end - ema_start) * (cos(pi * t / T) + 1) / 2

This schedule starts at ema_start (t=0) and ends at ema_end (t=T).
When ema_start == ema_end, momentum is constant throughout training.
"""

from __future__ import annotations

import copy
import math
from typing import Tuple

import torch
import torch.nn as nn


class EMAManager:
    """
    Manages an EMA copy of a context encoder for use as the JEPA target encoder.

    Args:
        encoder:       The context encoder module to track.
        ema_schedule:  (ema_start, ema_end) -- momentum at step 0 and total_steps.
        total_steps:   Total number of training steps (for schedule computation).

    Example::

        ema = EMAManager(encoder, ema_schedule=(0.99925, 0.99925), total_steps=300_000)

        for step, batch in enumerate(loader):
            # ... training step ...
            current_momentum = ema.update(step)
    """

    def __init__(
        self,
        encoder: nn.Module,
        ema_schedule: Tuple[float, float],
        total_steps: int,
    ) -> None:
        self.ema_start = ema_schedule[0]
        self.ema_end   = ema_schedule[1]
        self.total_steps = max(1, total_steps)

        # Deep copy the encoder to create the target encoder
        self.target_encoder: nn.Module = copy.deepcopy(encoder)

        # Disable gradients on all target encoder parameters
        for param in self.target_encoder.parameters():
            param.requires_grad_(False)

        # Set target encoder to inference mode (no dropout, no BN stats update)
        self.target_encoder.train(False)

        # Track the most recently computed momentum
        self._current_momentum: float = self.ema_start

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_momentum(self, step: int) -> float:
        """
        Compute the EMA momentum for the given step using a cosine schedule.

        At progress=0: m = ema_start
        At progress=1: m = ema_end

        Args:
            step: Current training step (0-indexed).

        Returns:
            Momentum value in [ema_start, ema_end].
        """
        progress = min(step / self.total_steps, 1.0)
        cosine_factor = (math.cos(math.pi * progress) + 1.0) / 2.0
        return self.ema_end - (self.ema_end - self.ema_start) * cosine_factor

    def update_from_encoder(self, encoder: nn.Module, step: int) -> float:
        """
        Update target encoder from the provided encoder module at the given step.

        Executes:
            for each param pair (target, encoder):
                target = m * target + (1 - m) * encoder

        Args:
            encoder: Current context encoder (source of ground-truth params).
            step:    Current training step (0-indexed).

        Returns:
            The momentum value used for this update.
        """
        m = self.get_momentum(step)
        self._current_momentum = m

        with torch.no_grad():
            for param_t, param_enc in zip(
                self.target_encoder.parameters(),
                encoder.parameters(),
            ):
                param_t.mul_(m).add_(param_enc.data, alpha=1.0 - m)

        return m

    def update(self, step: int) -> float:
        """
        Placeholder update -- subclasses with stored encoder reference override this.

        Args:
            step: Current training step.

        Returns:
            Momentum value (uses ema_start as fallback when no encoder reference).
        """
        m = self.get_momentum(step)
        self._current_momentum = m
        return m

    def get_target_encoder(self) -> nn.Module:
        """
        Return the target encoder (no gradients, inference mode).

        Returns:
            Target encoder module.
        """
        return self.target_encoder

    @property
    def current_momentum(self) -> float:
        """Return the momentum used in the most recent update."""
        return self._current_momentum

    def state_dict(self) -> dict:
        """
        Return serializable state for checkpoint saving.

        Includes target encoder weights and schedule metadata.
        """
        return {
            "target_encoder": self.target_encoder.state_dict(),
            "ema_start": self.ema_start,
            "ema_end": self.ema_end,
            "total_steps": self.total_steps,
            "current_momentum": self._current_momentum,
        }

    def load_state_dict(self, state: dict, strict: bool = True) -> None:
        """
        Restore state from a checkpoint dict.

        Args:
            state:  Dict previously returned by state_dict().
            strict: Whether to require exact key match in target_encoder state.
        """
        missing, unexpected = self.target_encoder.load_state_dict(
            state["target_encoder"], strict=strict
        )
        if missing and strict:
            raise RuntimeError(f"Missing target_encoder keys: {missing}")

        self.ema_start         = state.get("ema_start", self.ema_start)
        self.ema_end           = state.get("ema_end", self.ema_end)
        self.total_steps       = state.get("total_steps", self.total_steps)
        self._current_momentum = state.get("current_momentum", self.ema_start)

        # Ensure target encoder remains non-differentiable after load
        for param in self.target_encoder.parameters():
            param.requires_grad_(False)
        self.target_encoder.train(False)


# ---------------------------------------------------------------------------
# Self-contained EMAManager that stores encoder reference internally.
# ---------------------------------------------------------------------------

class EMAManagerWithRef(EMAManager):
    """
    EMAManager variant that stores a reference to the encoder at construction.

    update(step) uses the stored reference -- no need to pass encoder each call.
    """

    def __init__(
        self,
        encoder: nn.Module,
        ema_schedule: Tuple[float, float],
        total_steps: int,
    ) -> None:
        super().__init__(encoder, ema_schedule, total_steps)
        self._encoder_ref = encoder  # Store reference to live encoder

    def update(self, step: int) -> float:
        """
        Update target encoder from the stored encoder reference.

        Args:
            step: Current training step (0-indexed).

        Returns:
            Current EMA momentum value.
        """
        return self.update_from_encoder(self._encoder_ref, step)


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("EMAManager self-tests")
    print("=" * 60)

    def make_encoder(dim: int = 32) -> nn.Module:
        """Create a simple MLP encoder for testing."""
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    # --- Test 1: Momentum at step 0 equals ema_start ---
    ema_start, ema_end = 0.99, 0.999
    encoder = make_encoder()
    ema = EMAManagerWithRef(encoder, (ema_start, ema_end), total_steps=1000)

    m0 = ema.get_momentum(step=0)
    assert abs(m0 - ema_start) < 1e-6, (
        f"Momentum at step 0 should be {ema_start}, got {m0}"
    )
    print(f"[PASS] Momentum at step 0 = {m0:.6f} (expected {ema_start})")

    # --- Test 2: Momentum at final step equals ema_end ---
    m_final = ema.get_momentum(step=999)
    assert abs(m_final - ema_end) < 1e-4, (
        f"Momentum at step 999 should be ~{ema_end}, got {m_final}"
    )
    print(f"[PASS] Momentum at step 999 = {m_final:.6f} (expected ~{ema_end})")

    # --- Test 3: Target encoder parameters differ from encoder after updates ---
    encoder2 = make_encoder()
    ema2 = EMAManagerWithRef(encoder2, (0.9, 0.9), total_steps=100)
    target2 = ema2.get_target_encoder()

    # Substantially modify encoder2
    with torch.no_grad():
        for p in encoder2.parameters():
            p.add_(torch.randn_like(p) * 1.0)  # Large perturbation

    for step in range(5):
        ema2.update(step)

    any_differs = False
    for p_enc, p_tgt in zip(encoder2.parameters(), target2.parameters()):
        if not torch.allclose(p_enc, p_tgt, atol=1e-3):
            any_differs = True
            break
    assert any_differs, "Target should differ from encoder after update"
    print("[PASS] Target encoder parameters differ from context encoder after updates")

    # --- Test 4: EMA formula is applied correctly ---
    encoder3 = make_encoder(dim=4)
    ema3 = EMAManagerWithRef(encoder3, (0.9, 0.9), total_steps=100)
    target3 = ema3.get_target_encoder()

    old_target = [p.clone() for p in target3.parameters()]

    with torch.no_grad():
        for p in encoder3.parameters():
            p.fill_(1.0)

    m = ema3.update(step=50)  # constant schedule at 0.9

    for old_t, p_tgt, p_enc in zip(
        old_target, target3.parameters(), encoder3.parameters()
    ):
        expected = m * old_t + (1.0 - m) * p_enc
        assert torch.allclose(p_tgt, expected, atol=1e-5), (
            f"EMA formula incorrect: got {p_tgt.flatten()[:3]}, "
            f"expected {expected.flatten()[:3]}"
        )
    print(f"[PASS] EMA formula correct: m={m:.2f}")

    # --- Test 5: No gradients in target encoder ---
    encoder4 = make_encoder()
    ema4 = EMAManagerWithRef(encoder4, (0.99, 0.999), total_steps=500)
    target4 = ema4.get_target_encoder()

    for p in target4.parameters():
        assert not p.requires_grad, (
            f"Target encoder param {p.shape} has requires_grad=True"
        )
    print("[PASS] Target encoder has no gradient parameters")

    # --- Test 6: Momentum schedule is monotone (ema_start to ema_end) ---
    ema5 = EMAManagerWithRef(make_encoder(), (0.99, 0.999), total_steps=1000)
    momenta = [ema5.get_momentum(t) for t in range(1000)]
    for i in range(len(momenta) - 1):
        assert momenta[i] <= momenta[i + 1] + 1e-10, (
            f"Momentum not monotone at step {i}: {momenta[i]:.6f} > {momenta[i+1]:.6f}"
        )
    print("[PASS] Momentum schedule is monotonically non-decreasing")

    # --- Test 7: Checkpoint round-trip ---
    encoder5 = make_encoder()
    ema6 = EMAManagerWithRef(encoder5, (0.99, 0.999), total_steps=200)
    for step in range(10):
        with torch.no_grad():
            for p in encoder5.parameters():
                p.add_(torch.randn_like(p) * 0.01)
        ema6.update(step)

    state = ema6.state_dict()

    encoder6 = make_encoder()
    ema7 = EMAManagerWithRef(encoder6, (0.99, 0.999), total_steps=200)
    ema7.load_state_dict(state)

    for p1, p2 in zip(
        ema6.target_encoder.parameters(),
        ema7.target_encoder.parameters(),
    ):
        assert torch.allclose(p1, p2, atol=1e-7), (
            "Target encoder params do not match after checkpoint load"
        )
    print("[PASS] Checkpoint round-trip preserves target encoder parameters")

    # --- Test 8: Constant schedule (ema_start == ema_end) ---
    ema8 = EMAManagerWithRef(make_encoder(), (0.99925, 0.99925), total_steps=300_000)
    m_mid = ema8.get_momentum(step=150_000)
    assert abs(m_mid - 0.99925) < 1e-6, (
        f"Constant schedule should give 0.99925 at mid-step, got {m_mid}"
    )
    print("[PASS] Constant momentum schedule works correctly")

    # --- Test 9: Target encoder is in inference mode (not training) ---
    encoder7 = nn.Sequential(nn.Linear(16, 16), nn.Dropout(0.5))
    encoder7.train(True)
    ema9 = EMAManagerWithRef(encoder7, (0.99, 0.999), total_steps=100)
    tgt9 = ema9.get_target_encoder()
    assert not tgt9.training, "Target encoder should be in inference mode"
    print("[PASS] Target encoder is in inference mode after construction")

    # --- Test 10: update_from_encoder with separate encoder reference ---
    encoder_a = make_encoder(dim=8)
    encoder_b = copy.deepcopy(encoder_a)  # Start as copies
    ema10 = EMAManager(encoder_a, (0.8, 0.8), total_steps=50)

    # Modify encoder_b dramatically
    with torch.no_grad():
        for p in encoder_b.parameters():
            p.fill_(5.0)

    m10 = ema10.update_from_encoder(encoder_b, step=25)
    assert abs(m10 - 0.8) < 1e-6, f"Expected 0.8, got {m10}"

    # Verify target moved toward encoder_b's value of 5.0
    for p_tgt in ema10.target_encoder.parameters():
        # Initial was ~0 (random), after 1 EMA step: 0.8 * 0 + 0.2 * 5 = 1.0
        assert p_tgt.abs().mean().item() > 0.5, (
            "Target encoder should have moved toward encoder_b's large values"
        )
    print("[PASS] update_from_encoder applies EMA correctly with separate reference")

    print()
    print("All 10 self-tests passed.")
