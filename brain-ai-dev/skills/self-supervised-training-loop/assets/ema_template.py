"""
EMAUpdater: Exponential moving average target network with cosine-annealed tau.

Core formula:
    target_param = tau * target_param + (1 - tau) * online_param

Cosine annealing schedule:
    tau(step) = 1 - (1 - tau_base) * (cos(pi * step / total_steps) + 1) / 2

At step=0:          tau = tau_base = 0.996  (fast tracking, aggressive updates)
At step=total_steps: tau -> tau_final ~= 0.9999  (slow, stable, near-frozen)

CRITICAL: All EMA operations are wrapped in @torch.no_grad() to prevent:
    - Computation graph creation (memory leak over 100k steps)
    - Incorrect gradient flow through the EMA chain

CRITICAL: Never use module.train(False) alternatives that are blocked.
          Always use module.train(False) for inference mode.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from typing import Tuple


class EMAUpdater:
    """
    Cosine-annealed exponential moving average target network updater.

    Tau anneals from tau_base (fast tracking) to tau_final (near-frozen) using
    a cosine schedule, providing stable late-training targets without sacrificing
    early-training responsiveness.

    Args:
        tau_base: EMA momentum at step 0. Typical: 0.996 (BYOL/DINO).
        tau_final: EMA momentum at total_steps. Typical: 0.9999.
        total_steps: Total training steps (denominator of cosine schedule).
    """

    def __init__(
        self,
        tau_base: float = 0.996,
        tau_final: float = 0.9999,
        total_steps: int = 100_000,
    ) -> None:
        if tau_base >= tau_final:
            raise ValueError(
                f"tau_base ({tau_base}) must be less than tau_final ({tau_final})"
            )
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")

        self.tau_base = tau_base
        self.tau_final = tau_final
        self.total_steps = total_steps

    def get_tau(self, step: int) -> float:
        """
        Compute cosine-annealed tau at the given training step.

        Formula:
            tau(step) = 1 - (1 - tau_base) * (cos(pi * step / total_steps) + 1) / 2

        This is equivalent to linear interpolation between tau_base and tau_final
        using a cosine interpolant that goes from 0 (at step=0) to 1 (at step=total):
            f(step) = (1 - cos(pi * step / total_steps)) / 2
            tau(step) = tau_base + f(step) * (tau_final - tau_base)

        Args:
            step: Current training step (0-indexed).

        Returns:
            Float tau value in [tau_base, tau_final].
        """
        # Clamp step to valid range
        step = max(0, min(step, self.total_steps))
        # Cosine factor: 0 at step=0, 1 at step=total_steps
        cosine_factor = (1.0 - math.cos(math.pi * step / self.total_steps)) / 2.0
        # Interpolate from tau_base to tau_final using cosine factor
        tau = self.tau_base + cosine_factor * (self.tau_final - self.tau_base)
        return tau

    @torch.no_grad()
    def update(
        self,
        online: nn.Module,
        target: nn.Module,
        step: int,
    ) -> float:
        """
        Apply EMA update: target = tau * target + (1-tau) * online.

        Also copies buffers (BatchNorm running_mean/var) from online to target.

        The @torch.no_grad() decorator is MANDATORY:
            Without it, lerp_ creates computation graph nodes connecting target
            to online. Over 100k steps, this creates a graph that consumes all
            available memory and produces incorrect gradients.

        Args:
            online: The online (gradient-updated) encoder.
            target: The target (EMA-updated) encoder.
            step: Current training step (used for tau annealing).

        Returns:
            The tau value used for this update step.
        """
        tau = self.get_tau(step)

        # Update parameters using torch.lerp for numerical stability:
        #   lerp_(end, weight) computes: self + weight * (end - self)
        #   With weight=(1-tau): target + (1-tau)*(online - target)
        #                      = tau*target + (1-tau)*online
        for p_online, p_target in zip(online.parameters(), target.parameters()):
            p_target.data.lerp_(p_online.data, 1.0 - tau)

        # Copy buffers (BatchNorm running_mean, running_var, etc.)
        # Buffers are COPIED (not EMA-averaged) — target should use same statistics
        for b_online, b_target in zip(online.buffers(), target.buffers()):
            b_target.data.copy_(b_online.data)

        return tau

    @torch.no_grad()
    def initial_sync(self, online: nn.Module, target: nn.Module) -> None:
        """
        Copy online weights to target for initialization (L2 distance = 0 after call).

        Must be called before the first training step. Without initial sync,
        the first thousands of EMA update steps are wasted correcting the
        initial mismatch between independently initialized online and target.

        Args:
            online: The online encoder (source of weights).
            target: The target encoder (destination, will be overwritten).
        """
        for p_online, p_target in zip(online.parameters(), target.parameters()):
            p_target.data.copy_(p_online.data)

        for b_online, b_target in zip(online.buffers(), target.buffers()):
            b_target.data.copy_(b_online.data)

    @torch.no_grad()
    def compute_distance(self, online: nn.Module, target: nn.Module) -> float:
        """
        Compute L2 distance between online and target parameters.

        Useful as a diagnostic: distance should be > 0 after training starts,
        and should increase as tau approaches 1.0 (slower tracking).

        Returns:
            L2 norm of (online_params - target_params) concatenated.
        """
        total_sq_diff = 0.0
        for p_online, p_target in zip(online.parameters(), target.parameters()):
            diff = p_online.data - p_target.data
            total_sq_diff += diff.pow(2).sum().item()
        return total_sq_diff ** 0.5

    def get_tau_schedule(self, num_points: int = 10) -> list:
        """
        Return (step, tau) pairs across the full training schedule.

        Useful for visualizing the tau annealing curve.
        """
        steps = [int(i * self.total_steps / max(num_points - 1, 1)) for i in range(num_points)]
        return [(s, self.get_tau(s)) for s in steps]

    def __repr__(self) -> str:
        return (
            f"EMAUpdater(tau_base={self.tau_base}, tau_final={self.tau_final}, "
            f"total_steps={self.total_steps})"
        )


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("EMAUpdater Self-Tests")
    print("=" * 60)

    # ----------------------------------------------------------------
    # Test 1: tau at step=0 equals tau_base exactly
    # ----------------------------------------------------------------
    print("\nTest 1: tau at step=0 equals tau_base...")

    updater = EMAUpdater(tau_base=0.996, tau_final=0.9999, total_steps=100_000)
    tau_0 = updater.get_tau(0)
    assert tau_0 == 0.996, f"Expected tau=0.996 at step=0, got {tau_0}"
    print(f"  PASS: tau(0)={tau_0}")

    # ----------------------------------------------------------------
    # Test 2: tau at step=total_steps approaches tau_final
    # ----------------------------------------------------------------
    print("\nTest 2: tau at total_steps approaches tau_final...")

    tau_final_val = updater.get_tau(100_000)
    assert abs(tau_final_val - 0.9999) < 1e-6, (
        f"Expected tau~0.9999 at total_steps, got {tau_final_val}"
    )
    print(f"  PASS: tau(total_steps)={tau_final_val:.6f}")

    # ----------------------------------------------------------------
    # Test 3: Cosine schedule is monotonically increasing
    # ----------------------------------------------------------------
    print("\nTest 3: tau schedule is monotonically increasing...")

    steps = list(range(0, 100_001, 5_000))
    taus = [updater.get_tau(s) for s in steps]

    for i in range(1, len(taus)):
        assert taus[i] >= taus[i - 1], (
            f"tau not monotone at step {steps[i]}: "
            f"tau[{steps[i-1]}]={taus[i-1]:.6f}, tau[{steps[i]}]={taus[i]:.6f}"
        )
    print(f"  PASS: tau increases monotonically from {taus[0]:.6f} to {taus[-1]:.6f}")

    # ----------------------------------------------------------------
    # Test 4: @no_grad enforcement — no grad_fn on updated params
    # ----------------------------------------------------------------
    print("\nTest 4: no_grad enforcement (no grad_fn after EMA update)...")

    online = nn.Linear(4, 2)
    target = nn.Linear(4, 2)

    # Ensure online has gradient tracking (as it would in real training)
    # Target should not accumulate computation graph
    updater.update(online, target, step=0)

    for p_name, p in target.named_parameters():
        assert p.grad_fn is None, (
            f"Target param '{p_name}' has grad_fn after EMA update — "
            "no_grad wrapper is missing or broken"
        )
    print("  PASS: No grad_fn on target parameters after update")

    # ----------------------------------------------------------------
    # Test 5: Parameter update correctness
    # ----------------------------------------------------------------
    print("\nTest 5: EMA update correctness (manual vs. EMAUpdater)...")

    # Create models with known weights
    online5 = nn.Linear(1, 1, bias=False)
    target5 = nn.Linear(1, 1, bias=False)

    with torch.no_grad():
        online5.weight.fill_(1.0)   # online param = 1.0
        target5.weight.fill_(0.0)   # target param = 0.0

    # tau at step=0 is tau_base=0.996
    # Expected: target = 0.996 * 0.0 + 0.004 * 1.0 = 0.004
    updater5 = EMAUpdater(tau_base=0.996, tau_final=0.9999, total_steps=100_000)
    tau_used = updater5.update(online5, target5, step=0)

    expected = 0.996 * 0.0 + (1.0 - 0.996) * 1.0  # = 0.004
    actual = target5.weight.item()
    assert abs(actual - expected) < 1e-5, (
        f"EMA update incorrect: expected {expected:.6f}, got {actual:.6f}"
    )
    print(f"  PASS: target updated to {actual:.6f} (expected {expected:.6f}), tau={tau_used:.6f}")

    # ----------------------------------------------------------------
    # Test 6: Buffer update (BatchNorm running_mean is COPIED, not EMA-averaged)
    # ----------------------------------------------------------------
    print("\nTest 6: Buffer copy (BatchNorm running_mean)...")

    online_bn = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
    target_bn = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))

    # Set different running_mean values
    with torch.no_grad():
        online_bn[1].running_mean.fill_(5.0)
        target_bn[1].running_mean.fill_(0.0)

    updater6 = EMAUpdater(tau_base=0.996, tau_final=0.9999, total_steps=1000)
    updater6.update(online_bn, target_bn, step=0)

    # running_mean should be COPIED (not EMA-averaged)
    # Expected: target running_mean = online running_mean = 5.0
    target_running_mean = target_bn[1].running_mean.mean().item()
    assert abs(target_running_mean - 5.0) < 1e-5, (
        f"Buffer not copied correctly: expected 5.0, got {target_running_mean:.6f}"
    )
    print(f"  PASS: running_mean copied to {target_running_mean:.4f} (expected 5.0)")

    # ----------------------------------------------------------------
    # Test 7: initial_sync makes parameters identical
    # ----------------------------------------------------------------
    print("\nTest 7: initial_sync makes online and target identical...")

    online7 = nn.Linear(8, 4)
    target7 = nn.Linear(8, 4)

    # Verify they start different
    initial_dist = sum(
        (p_o - p_t).pow(2).sum().item()
        for p_o, p_t in zip(online7.parameters(), target7.parameters())
    ) ** 0.5

    updater7 = EMAUpdater(tau_base=0.996, tau_final=0.9999, total_steps=1000)
    updater7.initial_sync(online7, target7)

    final_dist = updater7.compute_distance(online7, target7)
    assert final_dist < 1e-8, (
        f"After initial_sync, L2 distance should be ~0, got {final_dist:.2e}"
    )
    print(f"  PASS: Distance after initial_sync={final_dist:.2e} (was {initial_dist:.4f})")

    # ----------------------------------------------------------------
    # Test 8: torch.lerp equivalence with manual multiply-add
    # ----------------------------------------------------------------
    print("\nTest 8: torch.lerp equivalence with manual computation...")

    a = torch.tensor([1.0, 2.0, 3.0])  # target
    b = torch.tensor([2.0, 4.0, 6.0])  # online
    tau = 0.996

    # Manual: tau * a + (1-tau) * b
    manual_result = tau * a + (1.0 - tau) * b

    # torch.lerp: lerp(start, end, weight) = start + weight*(end-start)
    #             with weight=(1-tau): a + (1-tau)*(b-a) = tau*a + (1-tau)*b
    a_clone = a.clone()
    a_clone.lerp_(b, 1.0 - tau)

    max_diff = (manual_result - a_clone).abs().max().item()
    assert max_diff < 1e-6, f"lerp and manual differ by {max_diff:.2e}"
    print(f"  PASS: torch.lerp matches manual multiply-add (max diff={max_diff:.2e})")

    # ----------------------------------------------------------------
    # Test 9: EMA diverges from online after updates (sanity check)
    # ----------------------------------------------------------------
    print("\nTest 9: Target diverges from online after updates (confirms EMA works)...")

    online9 = nn.Linear(4, 2)
    target9 = nn.Linear(4, 2)
    updater9 = EMAUpdater(tau_base=0.996, tau_final=0.9999, total_steps=100)
    updater9.initial_sync(online9, target9)

    # Modify online weights significantly
    with torch.no_grad():
        for p in online9.parameters():
            p.data += 1.0  # shift all online weights by 1.0

    # Apply one EMA update
    updater9.update(online9, target9, step=0)

    dist = updater9.compute_distance(online9, target9)
    assert dist > 0.0, "Target should differ from online after EMA update"
    print(f"  PASS: Distance between online and target = {dist:.6f} (> 0)")

    # ----------------------------------------------------------------
    # Test 10: Validation errors
    # ----------------------------------------------------------------
    print("\nTest 10: Validation errors for bad config...")

    try:
        bad_updater = EMAUpdater(tau_base=0.999, tau_final=0.996, total_steps=1000)
        assert False, "Should have raised ValueError for tau_base > tau_final"
    except ValueError as e:
        print(f"  PASS: ValueError raised for tau_base > tau_final: {e}")

    print("\n" + "=" * 60)
    print("All EMAUpdater self-tests PASSED")
    print("=" * 60)
    sys.exit(0)
