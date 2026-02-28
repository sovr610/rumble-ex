"""
LR and WD Schedulers for V-JEPA 2 self-supervised training.

Three learning rate schedulers and one weight decay scheduler:

  WarmupCosineScheduler   -- Linear warmup then cosine decay to final_lr
  LinearDecayScheduler    -- Linear decay from ref_lr to final_lr
  WarmupStableDecayScheduler -- Warmup, plateau, then linear decay (cooldown)
  CosineWDScheduler       -- Cosine weight decay schedule (wd_start -> wd_end)

Each scheduler exposes a single step(t: int) -> float method that returns
the current schedule value for step t. This design enables manual application
to optimizer param groups without coupling to a specific optimizer API.

Usage::

    lr_sched = WarmupCosineScheduler(
        ref_lr=1e-3, final_lr=1e-6,
        warmup_steps=40_000, total_steps=300_000,
    )
    wd_sched = CosineWDScheduler(wd_start=0.04, wd_end=0.4, total_steps=300_000)

    for step, batch in enumerate(loader):
        current_lr = lr_sched.step(step)
        current_wd = wd_sched.step(step)
        for pg in optimizer.param_groups:
            pg['lr'] = current_lr
            if pg.get('weight_decay', 0.0) > 0.0:
                pg['weight_decay'] = current_wd
        # ... forward / backward / optimizer.step() ...
"""

from __future__ import annotations

import math
from typing import Optional


class WarmupCosineScheduler:
    """
    Learning rate scheduler: linear warmup followed by cosine decay.

    Schedule:
        if step < warmup_steps:
            lr = ref_lr * (step + 1) / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            lr = final_lr + (ref_lr - final_lr) * 0.5 * (1 + cos(pi * progress))

    This is the standard schedule for V-JEPA 2 pretraining (Stage 1).

    Args:
        ref_lr:        Peak learning rate (achieved at end of warmup).
        final_lr:      Minimum learning rate (achieved at total_steps).
        warmup_steps:  Number of linear warmup steps.
        total_steps:   Total training steps.
    """

    def __init__(
        self,
        ref_lr: float,
        final_lr: float,
        warmup_steps: int,
        total_steps: int,
    ) -> None:
        if ref_lr <= 0:
            raise ValueError(f"ref_lr must be > 0, got {ref_lr}")
        if final_lr < 0:
            raise ValueError(f"final_lr must be >= 0, got {final_lr}")
        if final_lr > ref_lr:
            raise ValueError(
                f"final_lr ({final_lr}) must be <= ref_lr ({ref_lr})"
            )
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if total_steps <= 0:
            raise ValueError(f"total_steps must be > 0, got {total_steps}")

        self.ref_lr       = ref_lr
        self.final_lr     = final_lr
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps

    def step(self, t: int) -> float:
        """
        Compute learning rate at step t.

        Args:
            t: Current step (0-indexed).

        Returns:
            Learning rate value.
        """
        if self.warmup_steps > 0 and t < self.warmup_steps:
            # Linear warmup: first step gives lr = ref_lr / warmup_steps
            return self.ref_lr * (t + 1) / self.warmup_steps
        else:
            # Cosine decay
            steps_after_warmup = max(1, self.total_steps - self.warmup_steps)
            progress = (t - self.warmup_steps) / steps_after_warmup
            progress = min(progress, 1.0)  # Clamp past end
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.final_lr + (self.ref_lr - self.final_lr) * cosine_factor

    def __repr__(self) -> str:
        return (
            f"WarmupCosineScheduler("
            f"ref_lr={self.ref_lr}, final_lr={self.final_lr}, "
            f"warmup_steps={self.warmup_steps}, total_steps={self.total_steps})"
        )


class LinearDecayScheduler:
    """
    Learning rate scheduler: simple linear decay from ref_lr to final_lr.

    Schedule:
        progress = t / total_steps  (clamped to [0, 1])
        lr = ref_lr - (ref_lr - final_lr) * progress

    Used in annealing/cooldown phases where the model continues from a
    pretrained checkpoint and linearly anneals the LR to near-zero.

    Args:
        ref_lr:      Starting learning rate.
        final_lr:    Terminal learning rate at total_steps.
        total_steps: Duration of the decay schedule.
    """

    def __init__(
        self,
        ref_lr: float,
        final_lr: float,
        total_steps: int,
    ) -> None:
        if ref_lr <= 0:
            raise ValueError(f"ref_lr must be > 0, got {ref_lr}")
        if final_lr < 0:
            raise ValueError(f"final_lr must be >= 0, got {final_lr}")
        if total_steps <= 0:
            raise ValueError(f"total_steps must be > 0, got {total_steps}")

        self.ref_lr      = ref_lr
        self.final_lr    = final_lr
        self.total_steps = total_steps

    def step(self, t: int) -> float:
        """
        Compute learning rate at step t.

        Args:
            t: Current step (0-indexed).

        Returns:
            Learning rate value.
        """
        progress = min(t / self.total_steps, 1.0)
        return self.ref_lr - (self.ref_lr - self.final_lr) * progress

    def __repr__(self) -> str:
        return (
            f"LinearDecayScheduler("
            f"ref_lr={self.ref_lr}, final_lr={self.final_lr}, "
            f"total_steps={self.total_steps})"
        )


class WarmupStableDecayScheduler:
    """
    LR scheduler: linear warmup, stable plateau, then linear decay.

    Schedule:
        if t < warmup_steps:
            lr = ref_lr * (t + 1) / warmup_steps
        elif t < stable_steps:
            lr = ref_lr
        else:
            progress = (t - stable_steps) / (total_steps - stable_steps)
            lr = ref_lr - (ref_lr - final_lr) * progress

    Used in cooldown when a short warmup is needed before annealing.
    Also used in some DROID fine-tuning configurations.

    Args:
        ref_lr:        Peak / plateau learning rate.
        final_lr:      Terminal learning rate after decay.
        warmup_steps:  Steps for linear warmup.
        stable_steps:  Step at which linear decay begins (>= warmup_steps).
        total_steps:   Total training steps.
    """

    def __init__(
        self,
        ref_lr: float,
        final_lr: float,
        warmup_steps: int,
        stable_steps: int,
        total_steps: int,
    ) -> None:
        if ref_lr <= 0:
            raise ValueError(f"ref_lr must be > 0, got {ref_lr}")
        if final_lr < 0:
            raise ValueError(f"final_lr must be >= 0, got {final_lr}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if stable_steps < warmup_steps:
            raise ValueError(
                f"stable_steps ({stable_steps}) must be >= warmup_steps ({warmup_steps})"
            )
        if total_steps < stable_steps:
            raise ValueError(
                f"total_steps ({total_steps}) must be >= stable_steps ({stable_steps})"
            )

        self.ref_lr       = ref_lr
        self.final_lr     = final_lr
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.total_steps  = total_steps

    def step(self, t: int) -> float:
        """
        Compute learning rate at step t.

        Args:
            t: Current step (0-indexed).

        Returns:
            Learning rate value.
        """
        if self.warmup_steps > 0 and t < self.warmup_steps:
            return self.ref_lr * (t + 1) / self.warmup_steps
        elif t < self.stable_steps:
            return self.ref_lr
        else:
            decay_duration = max(1, self.total_steps - self.stable_steps)
            progress = min((t - self.stable_steps) / decay_duration, 1.0)
            return self.ref_lr - (self.ref_lr - self.final_lr) * progress

    def __repr__(self) -> str:
        return (
            f"WarmupStableDecayScheduler("
            f"ref_lr={self.ref_lr}, final_lr={self.final_lr}, "
            f"warmup={self.warmup_steps}, stable={self.stable_steps}, "
            f"total={self.total_steps})"
        )


class CosineWDScheduler:
    """
    Weight decay cosine schedule: increases from wd_start to wd_end.

    This is the opposite direction to the LR cosine decay. Low WD early
    allows the model to explore; high WD late encourages regularization.

    Schedule:
        progress = t / total_steps  (clamped to [0, 1])
        wd = wd_start + (wd_end - wd_start) * 0.5 * (1 - cos(pi * progress))

    At progress=0: wd = wd_start
    At progress=1: wd = wd_end

    Args:
        wd_start:    Initial weight decay value.
        wd_end:      Final weight decay value (>= wd_start).
        total_steps: Total training steps.
    """

    def __init__(
        self,
        wd_start: float,
        wd_end: float,
        total_steps: int,
    ) -> None:
        if wd_start < 0:
            raise ValueError(f"wd_start must be >= 0, got {wd_start}")
        if wd_end < wd_start:
            raise ValueError(
                f"wd_end ({wd_end}) must be >= wd_start ({wd_start})"
            )
        if total_steps <= 0:
            raise ValueError(f"total_steps must be > 0, got {total_steps}")

        self.wd_start    = wd_start
        self.wd_end      = wd_end
        self.total_steps = total_steps

    def step(self, t: int) -> float:
        """
        Compute weight decay at step t.

        Args:
            t: Current step (0-indexed).

        Returns:
            Weight decay value in [wd_start, wd_end].
        """
        progress = min(t / self.total_steps, 1.0)
        cosine_factor = 0.5 * (1.0 - math.cos(math.pi * progress))
        return self.wd_start + (self.wd_end - self.wd_start) * cosine_factor

    def __repr__(self) -> str:
        return (
            f"CosineWDScheduler("
            f"wd_start={self.wd_start}, wd_end={self.wd_end}, "
            f"total_steps={self.total_steps})"
        )


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def build_pretrain_schedulers(
    lr: float = 1e-3,
    final_lr: float = 1e-6,
    wd_start: float = 0.04,
    wd_end: float = 0.4,
    warmup_steps: int = 40_000,
    total_steps: int = 300_000,
) -> tuple:
    """
    Build the standard pretrain LR and WD schedulers.

    Returns:
        (WarmupCosineScheduler, CosineWDScheduler)
    """
    lr_sched = WarmupCosineScheduler(
        ref_lr=lr, final_lr=final_lr,
        warmup_steps=warmup_steps, total_steps=total_steps,
    )
    wd_sched = CosineWDScheduler(
        wd_start=wd_start, wd_end=wd_end, total_steps=total_steps
    )
    return lr_sched, wd_sched


def build_cooldown_schedulers(
    lr: float = 1e-4,
    final_lr: float = 1e-7,
    wd: float = 0.4,
    total_steps: int = 30_000,
) -> tuple:
    """
    Build cooldown LR (linear decay) and constant WD schedulers.

    Returns:
        (LinearDecayScheduler, CosineWDScheduler with zero range)
    """
    lr_sched = LinearDecayScheduler(ref_lr=lr, final_lr=final_lr,
                                    total_steps=total_steps)
    wd_sched = CosineWDScheduler(wd_start=wd, wd_end=wd,
                                  total_steps=total_steps)
    return lr_sched, wd_sched


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("LR and WD Scheduler self-tests")
    print("=" * 60)

    TOTAL = 1000
    WARMUP = 100

    # ---------------------------------------------------------------
    # WarmupCosineScheduler tests
    # ---------------------------------------------------------------
    ref_lr   = 1e-3
    final_lr = 1e-6
    sched = WarmupCosineScheduler(
        ref_lr=ref_lr, final_lr=final_lr,
        warmup_steps=WARMUP, total_steps=TOTAL,
    )

    # Test 1: Warmup reaches ref_lr at warmup_steps
    lr_at_warmup = sched.step(WARMUP)
    assert abs(lr_at_warmup - ref_lr) / ref_lr < 0.01, (
        f"LR at warmup end should be {ref_lr:.2e}, got {lr_at_warmup:.2e}"
    )
    print(f"[PASS] WarmupCosine: LR at warmup end = {lr_at_warmup:.4e}")

    # Test 2: Step 0 is near 0 (warmup fraction = 1/warmup_steps)
    lr_at_0 = sched.step(0)
    assert lr_at_0 < ref_lr * 0.05, (
        f"LR at step 0 should be near 0, got {lr_at_0:.2e}"
    )
    print(f"[PASS] WarmupCosine: LR at step 0 = {lr_at_0:.4e} (near 0)")

    # Test 3: Cosine decays to near final_lr at total_steps
    lr_final = sched.step(TOTAL - 1)
    assert abs(lr_final - final_lr) / final_lr < 0.02, (
        f"LR at final step should be ~{final_lr:.2e}, got {lr_final:.2e}"
    )
    print(f"[PASS] WarmupCosine: LR at final step = {lr_final:.4e}")

    # Test 4: Monotone after warmup
    lrs = [sched.step(t) for t in range(WARMUP, TOTAL)]
    for i in range(len(lrs) - 1):
        assert lrs[i] >= lrs[i + 1] - 1e-12, (
            f"LR not monotone at step {i + WARMUP}"
        )
    print("[PASS] WarmupCosine: Monotonically decreasing after warmup")

    # Test 5: Warmup is monotone increasing
    lrs_warmup = [sched.step(t) for t in range(WARMUP + 1)]
    for i in range(len(lrs_warmup) - 1):
        assert lrs_warmup[i] <= lrs_warmup[i + 1] + 1e-12, (
            f"Warmup not monotone at step {i}"
        )
    print("[PASS] WarmupCosine: Monotonically increasing during warmup")

    # ---------------------------------------------------------------
    # LinearDecayScheduler tests
    # ---------------------------------------------------------------
    lin_sched = LinearDecayScheduler(ref_lr=1e-4, final_lr=1e-7, total_steps=500)

    # Test 6: Step 0 equals ref_lr
    lin_0 = lin_sched.step(0)
    assert abs(lin_0 - 1e-4) / 1e-4 < 0.001, (
        f"LinearDecay step 0 should be {1e-4:.2e}, got {lin_0:.2e}"
    )
    print(f"[PASS] LinearDecay: LR at step 0 = {lin_0:.4e}")

    # Test 7: Last step equals final_lr
    lin_final = lin_sched.step(500)
    assert abs(lin_final - 1e-7) / 1e-7 < 0.001, (
        f"LinearDecay final should be {1e-7:.2e}, got {lin_final:.2e}"
    )
    print(f"[PASS] LinearDecay: LR at final step = {lin_final:.4e}")

    # Test 8: Linear is monotone
    lin_lrs = [lin_sched.step(t) for t in range(501)]
    for i in range(len(lin_lrs) - 1):
        assert lin_lrs[i] >= lin_lrs[i + 1] - 1e-12, (
            f"Linear decay not monotone at step {i}"
        )
    print("[PASS] LinearDecay: Monotonically decreasing")

    # ---------------------------------------------------------------
    # WarmupStableDecayScheduler tests
    # ---------------------------------------------------------------
    wsd_sched = WarmupStableDecayScheduler(
        ref_lr=1e-3, final_lr=1e-6,
        warmup_steps=50, stable_steps=200, total_steps=500,
    )

    # Test 9: Warmup phase
    wsd_at_0  = wsd_sched.step(0)
    wsd_at_50 = wsd_sched.step(50)
    assert wsd_at_0 < 1e-3, f"Step 0 should be < peak, got {wsd_at_0:.2e}"
    assert abs(wsd_at_50 - 1e-3) / 1e-3 < 0.01, (
        f"Step 50 (warmup end) should be {1e-3:.2e}, got {wsd_at_50:.2e}"
    )
    print(f"[PASS] WarmupStableDecay: Warmup ends at {wsd_at_50:.4e}")

    # Test 10: Stable phase
    wsd_stable = wsd_sched.step(125)  # Between 50 and 200
    assert abs(wsd_stable - 1e-3) / 1e-3 < 1e-6, (
        f"Stable phase should be {1e-3:.2e}, got {wsd_stable:.2e}"
    )
    print(f"[PASS] WarmupStableDecay: Stable plateau = {wsd_stable:.4e}")

    # Test 11: Decay phase
    wsd_final = wsd_sched.step(500)
    assert abs(wsd_final - 1e-6) / 1e-6 < 0.01, (
        f"Final step should be {1e-6:.2e}, got {wsd_final:.2e}"
    )
    print(f"[PASS] WarmupStableDecay: Final step = {wsd_final:.4e}")

    # ---------------------------------------------------------------
    # CosineWDScheduler tests
    # ---------------------------------------------------------------
    wd_sched = CosineWDScheduler(wd_start=0.04, wd_end=0.4, total_steps=1000)

    # Test 12: WD at step 0 equals wd_start
    wd_0 = wd_sched.step(0)
    assert abs(wd_0 - 0.04) < 1e-6, (
        f"WD at step 0 should be 0.04, got {wd_0:.6f}"
    )
    print(f"[PASS] CosineWD: WD at step 0 = {wd_0:.6f}")

    # Test 13: WD at final step equals wd_end
    wd_final = wd_sched.step(1000)
    assert abs(wd_final - 0.4) < 1e-4, (
        f"WD at final step should be 0.4, got {wd_final:.6f}"
    )
    print(f"[PASS] CosineWD: WD at final step = {wd_final:.6f}")

    # Test 14: WD is monotonically increasing
    wds = [wd_sched.step(t) for t in range(1001)]
    for i in range(len(wds) - 1):
        assert wds[i] <= wds[i + 1] + 1e-10, (
            f"WD not monotone at step {i}: {wds[i]:.6f} > {wds[i+1]:.6f}"
        )
    print("[PASS] CosineWD: Monotonically increasing from wd_start to wd_end")

    # Test 15: WD at midpoint is roughly midway (cosine schedule)
    wd_mid = wd_sched.step(500)
    mid_expected = 0.5 * (0.04 + 0.4)  # ~0.22 for cosine midpoint
    assert abs(wd_mid - mid_expected) < 0.05, (
        f"WD midpoint should be ~{mid_expected:.3f}, got {wd_mid:.3f}"
    )
    print(f"[PASS] CosineWD: WD at midpoint = {wd_mid:.4f} (expected ~{mid_expected:.3f})")

    # Test 16: Factory helper build_pretrain_schedulers
    lr_s, wd_s = build_pretrain_schedulers(
        lr=1e-3, final_lr=1e-6, wd_start=0.04, wd_end=0.4,
        warmup_steps=40_000, total_steps=300_000,
    )
    assert isinstance(lr_s, WarmupCosineScheduler)
    assert isinstance(wd_s, CosineWDScheduler)
    print("[PASS] build_pretrain_schedulers returns correct types")

    # Test 17: Factory helper build_cooldown_schedulers
    lr_c, wd_c = build_cooldown_schedulers(
        lr=1e-4, final_lr=1e-7, wd=0.4, total_steps=30_000,
    )
    assert isinstance(lr_c, LinearDecayScheduler)
    assert isinstance(wd_c, CosineWDScheduler)
    # Constant WD: wd_start == wd_end, so all steps should give 0.4
    assert abs(wd_c.step(0) - 0.4) < 1e-6
    assert abs(wd_c.step(15_000) - 0.4) < 1e-6
    print("[PASS] build_cooldown_schedulers returns correct types and constant WD")

    print()
    print("All 17 self-tests passed.")
