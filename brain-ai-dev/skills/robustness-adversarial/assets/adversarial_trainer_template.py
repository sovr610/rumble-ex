"""
Adversarial Trainer Template.

Provides ``AdversarialTrainer`` with PGD-AT, TRADES, and Free-AT training
methods, plus curriculum epsilon scheduling.

Dependencies: torch + standard library only.  No PIL, no numpy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _forward(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Flexible forward: accepts Tensor or wraps to dict."""
    if isinstance(x, dict):
        vals = list(x.values())
        if len(vals) == 1:
            return model(vals[0])
        return model(x)
    return model(x)


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(1) == targets).float().mean().item()


# ---------------------------------------------------------------------------
# Curriculum epsilon schedules
# ---------------------------------------------------------------------------

def linear_epsilon_schedule(epoch: int, warmup_epochs: int,
                            target_epsilon: float) -> float:
    if warmup_epochs <= 0 or epoch >= warmup_epochs:
        return target_epsilon
    return target_epsilon * (epoch + 1) / warmup_epochs


def cosine_epsilon_schedule(epoch: int, warmup_epochs: int,
                            target_epsilon: float) -> float:
    if warmup_epochs <= 0 or epoch >= warmup_epochs:
        return target_epsilon
    progress = (epoch + 1) / warmup_epochs
    return target_epsilon * 0.5 * (1.0 - math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# PGD inner loop (shared by PGD-AT and TRADES)
# ---------------------------------------------------------------------------

def _pgd_inner(
    model: nn.Module,
    x: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float,
    steps: int,
    step_size: float,
    loss_fn: Callable,
    random_start: bool = True,
) -> torch.Tensor:
    """Run PGD inner maximisation and return final delta (detached)."""
    delta = torch.zeros_like(x)
    if random_start:
        delta.uniform_(-epsilon, epsilon)
        delta = delta.clamp(-epsilon, epsilon)
        delta = (x + delta).clamp(0.0, 1.0) - x

    for _ in range(steps):
        delta.requires_grad_(True)
        x_adv = (x + delta).clamp(0.0, 1.0)
        loss = loss_fn(_forward(model, x_adv), targets)
        loss.backward()
        grad = delta.grad.detach()
        delta = delta.detach() + step_size * grad.sign()
        delta = delta.clamp(-epsilon, epsilon)
        delta = (x + delta).clamp(0.0, 1.0) - x

    return delta.detach()


# ---------------------------------------------------------------------------
# AdversarialTrainer
# ---------------------------------------------------------------------------

class AdversarialTrainer:
    """Adversarial training with PGD-AT, TRADES, or Free-AT.

    Parameters
    ----------
    model : nn.Module
        The classifier to train.
    method : str
        ``"pgd_at"`` | ``"trades"`` | ``"free_at"``.
    epsilon : float
        Perturbation budget.
    pgd_steps : int
        Inner-loop PGD steps.
    pgd_step_size : float
        Inner-loop step size.
    trades_beta : float
        TRADES regularisation weight.
    free_at_replays : int
        Number of replays for Free-AT.
    use_curriculum : bool
        Whether to ramp epsilon during warmup.
    warmup_epochs : int
        Curriculum warmup length in epochs.
    epsilon_schedule : str
        ``"linear"`` or ``"cosine"``.
    grad_clip_norm : float
        Max gradient norm (0 = disabled).
    """

    def __init__(
        self,
        model: nn.Module,
        method: str = "pgd_at",
        epsilon: float = 8.0 / 255.0,
        pgd_steps: int = 7,
        pgd_step_size: float = 2.0 / 255.0,
        trades_beta: float = 6.0,
        free_at_replays: int = 4,
        use_curriculum: bool = False,
        warmup_epochs: int = 10,
        epsilon_schedule: str = "linear",
        grad_clip_norm: float = 0.0,
    ):
        self.model = model
        self.method = method
        self.epsilon = epsilon
        self.pgd_steps = pgd_steps
        self.pgd_step_size = pgd_step_size
        self.trades_beta = trades_beta
        self.free_at_replays = free_at_replays
        self.use_curriculum = use_curriculum
        self.warmup_epochs = warmup_epochs
        self.epsilon_schedule = epsilon_schedule
        self.grad_clip_norm = grad_clip_norm

        # Free-AT running perturbation
        self._free_delta: Optional[torch.Tensor] = None
        self._current_epoch: int = 0

    # ---- curriculum helpers -----------------------------------------------

    def get_epsilon(self, epoch: Optional[int] = None) -> float:
        ep = epoch if epoch is not None else self._current_epoch
        if not self.use_curriculum:
            return self.epsilon
        if self.epsilon_schedule == "cosine":
            return cosine_epsilon_schedule(ep, self.warmup_epochs, self.epsilon)
        return linear_epsilon_schedule(ep, self.warmup_epochs, self.epsilon)

    def set_epoch(self, epoch: int) -> None:
        self._current_epoch = epoch

    # ---- unified train_step ----------------------------------------------

    def train_step(
        self,
        x: torch.Tensor,
        targets: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        epoch: Optional[int] = None,
    ) -> Dict[str, float]:
        """Single training step.  Returns dict of metric values."""
        if epoch is not None:
            self._current_epoch = epoch

        if self.method == "pgd_at":
            return self._pgd_at_step(x, targets, optimizer)
        elif self.method == "trades":
            return self._trades_step(x, targets, optimizer)
        elif self.method == "free_at":
            return self._free_at_step(x, targets, optimizer)
        else:
            raise ValueError(f"Unknown AT method: {self.method}")

    # ---- PGD-AT -----------------------------------------------------------

    def _pgd_at_step(
        self,
        x: torch.Tensor,
        targets: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        eps = self.get_epsilon()
        self.model.train()

        delta = _pgd_inner(
            self.model, x, targets, eps, self.pgd_steps, self.pgd_step_size,
            loss_fn=F.cross_entropy,
        )

        x_adv = (x + delta).clamp(0.0, 1.0)
        logits = _forward(self.model, x_adv)
        loss = F.cross_entropy(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        if self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        optimizer.step()

        return {
            "loss": loss.item(),
            "robust_acc": _accuracy(logits, targets),
            "epsilon": eps,
        }

    # ---- TRADES -----------------------------------------------------------

    def trades_loss(
        self,
        x: torch.Tensor,
        targets: torch.Tensor,
        beta: Optional[float] = None,
        epsilon: Optional[float] = None,
    ) -> torch.Tensor:
        """Compute TRADES loss: CE(clean) + beta * KL(clean || adv)."""
        eps = epsilon if epsilon is not None else self.get_epsilon()
        b = beta if beta is not None else self.trades_beta

        self.model.train()

        # Clean logits (detached for inner loop)
        with torch.no_grad():
            clean_logits = _forward(self.model, x)
            clean_probs = F.softmax(clean_logits, dim=1)

        # PGD to maximise KL divergence
        delta = torch.zeros_like(x).uniform_(-eps, eps)
        delta = (x + delta).clamp(0.0, 1.0) - x

        for _ in range(self.pgd_steps):
            delta.requires_grad_(True)
            x_adv = (x + delta).clamp(0.0, 1.0)
            adv_logits = _forward(self.model, x_adv)
            kl = F.kl_div(
                F.log_softmax(adv_logits, dim=1),
                clean_probs,
                reduction="batchmean",
            )
            kl.backward()
            grad = delta.grad.detach()
            delta = delta.detach() + self.pgd_step_size * grad.sign()
            delta = delta.clamp(-eps, eps)
            delta = (x + delta).clamp(0.0, 1.0) - x

        # Final loss
        x_adv = (x + delta.detach()).clamp(0.0, 1.0)
        clean_logits_2 = _forward(self.model, x)
        ce_loss = F.cross_entropy(clean_logits_2, targets)
        adv_logits_2 = _forward(self.model, x_adv)
        kl_loss = F.kl_div(
            F.log_softmax(adv_logits_2, dim=1),
            F.softmax(clean_logits_2.detach(), dim=1),
            reduction="batchmean",
        )

        return ce_loss + b * kl_loss

    def _trades_step(
        self,
        x: torch.Tensor,
        targets: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        loss = self.trades_loss(x, targets)
        optimizer.zero_grad()
        loss.backward()
        if self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        optimizer.step()

        self.model.eval()
        with torch.no_grad():
            clean_logits = _forward(self.model, x)
        self.model.train()

        return {
            "loss": loss.item(),
            "clean_acc": _accuracy(clean_logits, targets),
            "epsilon": self.get_epsilon(),
        }

    # ---- Free-AT ----------------------------------------------------------

    def _free_at_step(
        self,
        x: torch.Tensor,
        targets: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        eps = self.get_epsilon()
        self.model.train()

        if self._free_delta is None or self._free_delta.shape != x.shape:
            self._free_delta = torch.zeros_like(x)

        total_loss = 0.0
        for _ in range(self.free_at_replays):
            x_adv = (x + self._free_delta.detach()).clamp(0.0, 1.0)
            x_adv.requires_grad_(True)

            logits = _forward(self.model, x_adv)
            loss = F.cross_entropy(logits, targets)

            optimizer.zero_grad()
            loss.backward()

            # Update perturbation
            input_grad = x_adv.grad.detach()
            self._free_delta = self._free_delta + eps * input_grad.sign()
            self._free_delta = self._free_delta.clamp(-eps, eps)
            self._free_delta = (x + self._free_delta).clamp(0.0, 1.0) - x

            # Update model
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            optimizer.step()
            total_loss += loss.item()

        return {
            "loss": total_loss / self.free_at_replays,
            "epsilon": eps,
        }

    # ---- train_epoch convenience -----------------------------------------

    def train_epoch(
        self,
        data: Tuple[torch.Tensor, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        epoch: int,
        batch_size: int = 64,
    ) -> Dict[str, float]:
        """Train one full epoch."""
        self.set_epoch(epoch)
        x, y = data
        n = x.shape[0]
        total_loss = 0.0
        steps = 0

        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i: i + batch_size]
            metrics = self.train_step(x[idx], y[idx], optimizer, epoch=epoch)
            total_loss += metrics["loss"]
            steps += 1

        return {"avg_loss": total_loss / max(steps, 1), "epsilon": self.get_epsilon()}


# ===================================================================
# Self-tests  (25+)
# ===================================================================

def _run_self_tests() -> None:
    import sys
    passed = 0
    failed = 0

    def _assert(cond: bool, msg: str) -> None:
        nonlocal passed, failed
        if cond:
            passed += 1
            print(f"  PASS: {msg}")
        else:
            failed += 1
            print(f"  FAIL: {msg}")

    print("=" * 60)
    print("adversarial_trainer_template self-tests")
    print("=" * 60)

    torch.manual_seed(99)

    class SimpleCNN(nn.Module):
        def __init__(self, nc=10):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.fc = nn.Linear(8 * 7 * 7, nc)

        def forward(self, x):
            if isinstance(x, dict):
                x = list(x.values())[0]
            return self.fc(self.pool(F.relu(self.conv1(x))).flatten(1))

    NC = 10
    N = 64
    x_data = torch.rand(N, 1, 28, 28)
    y_data = torch.randint(0, NC, (N,))
    eps = 8.0 / 255.0

    # ==== PGD-AT tests ====================================================
    model_pgd = SimpleCNN(NC)
    opt_pgd = torch.optim.Adam(model_pgd.parameters(), lr=0.01)

    trainer_pgd = AdversarialTrainer(model_pgd, method="pgd_at", epsilon=eps,
                                     pgd_steps=3, pgd_step_size=2.0 / 255.0)

    m1 = trainer_pgd.train_step(x_data, y_data, opt_pgd)
    _assert("loss" in m1, "PGD-AT train_step returns loss")
    _assert("robust_acc" in m1, "PGD-AT train_step returns robust_acc")
    _assert("epsilon" in m1, "PGD-AT train_step returns epsilon")
    _assert(m1["loss"] > 0, "PGD-AT loss is positive")
    _assert(0.0 <= m1["robust_acc"] <= 1.0, "PGD-AT robust_acc in [0,1]")

    # Loss should decrease over steps
    losses = []
    for _ in range(20):
        m = trainer_pgd.train_step(x_data, y_data, opt_pgd)
        losses.append(m["loss"])
    _assert(losses[-1] < losses[0],
            f"PGD-AT loss decreases ({losses[0]:.3f} -> {losses[-1]:.3f})")

    # Model should still be trainable (not collapsed)
    model_pgd.eval()
    with torch.no_grad():
        clean_logits = model_pgd(x_data)
    clean_acc = _accuracy(clean_logits, y_data)
    _assert(clean_acc > 0.15,
            f"PGD-AT model not collapsed (acc={clean_acc:.2%})")

    # Gradient flow check
    model_pgd.train()
    m = trainer_pgd.train_step(x_data[:8], y_data[:8], opt_pgd)
    grads_ok = all(
        p.grad is not None and torch.isfinite(p.grad).all().item()
        for p in model_pgd.parameters() if p.requires_grad
    )
    _assert(grads_ok, "PGD-AT gradient flow: all grads finite and non-None")

    # ==== TRADES tests ====================================================
    model_tr = SimpleCNN(NC)
    opt_tr = torch.optim.Adam(model_tr.parameters(), lr=0.01)

    trainer_tr = AdversarialTrainer(model_tr, method="trades", epsilon=eps,
                                    pgd_steps=3, trades_beta=6.0)

    m_tr = trainer_tr.train_step(x_data, y_data, opt_tr)
    _assert("loss" in m_tr, "TRADES train_step returns loss")
    _assert(m_tr["loss"] > 0, "TRADES loss is positive")

    # TRADES loss directly
    tl = trainer_tr.trades_loss(x_data[:16], y_data[:16], beta=6.0)
    _assert(tl.item() > 0, "trades_loss returns positive scalar")
    _assert(torch.isfinite(tl).item(), "trades_loss is finite")

    # TRADES loss components: beta=0 should be pure CE
    tl_b0 = trainer_tr.trades_loss(x_data[:16], y_data[:16], beta=0.0)
    ce_only = F.cross_entropy(_forward(model_tr, x_data[:16]), y_data[:16])
    _assert(abs(tl_b0.item() - ce_only.item()) < 0.5,
            "TRADES beta=0 approximates pure CE")

    # TRADES beta effect: higher beta => different loss
    tl_b1 = trainer_tr.trades_loss(x_data[:16], y_data[:16], beta=1.0)
    tl_b10 = trainer_tr.trades_loss(x_data[:16], y_data[:16], beta=10.0)
    _assert(tl_b10.item() >= tl_b1.item() - 0.5,
            "TRADES higher beta generally increases total loss")

    losses_tr = []
    for _ in range(20):
        mt = trainer_tr.train_step(x_data, y_data, opt_tr)
        losses_tr.append(mt["loss"])
    _assert(losses_tr[-1] < losses_tr[0],
            f"TRADES loss decreases ({losses_tr[0]:.3f} -> {losses_tr[-1]:.3f})")

    # ==== Free-AT tests ===================================================
    model_fr = SimpleCNN(NC)
    opt_fr = torch.optim.Adam(model_fr.parameters(), lr=0.01)

    trainer_fr = AdversarialTrainer(model_fr, method="free_at", epsilon=eps,
                                    free_at_replays=4)

    m_fr = trainer_fr.train_step(x_data, y_data, opt_fr)
    _assert("loss" in m_fr, "Free-AT train_step returns loss")
    _assert(m_fr["loss"] > 0, "Free-AT loss is positive")

    losses_fr = []
    for _ in range(10):
        mf = trainer_fr.train_step(x_data, y_data, opt_fr)
        losses_fr.append(mf["loss"])
    _assert(losses_fr[-1] < losses_fr[0] + 1.0,
            "Free-AT loss does not diverge")

    # ==== Curriculum epsilon tests ========================================
    trainer_cur = AdversarialTrainer(
        model_pgd, method="pgd_at", epsilon=eps,
        use_curriculum=True, warmup_epochs=10, epsilon_schedule="linear",
    )
    e0 = trainer_cur.get_epsilon(0)
    e5 = trainer_cur.get_epsilon(5)
    e10 = trainer_cur.get_epsilon(10)
    _assert(e0 < eps, f"Curriculum eps at epoch 0 ({e0:.5f}) < target ({eps:.5f})")
    _assert(e5 > e0, "Curriculum eps at epoch 5 > epoch 0")
    _assert(abs(e10 - eps) < 1e-8, "Curriculum eps at epoch 10 = target")

    # Cosine schedule
    trainer_cos = AdversarialTrainer(
        model_pgd, method="pgd_at", epsilon=eps,
        use_curriculum=True, warmup_epochs=10, epsilon_schedule="cosine",
    )
    ec0 = trainer_cos.get_epsilon(0)
    ec10 = trainer_cos.get_epsilon(10)
    _assert(ec0 < eps, "Cosine curriculum eps < target at epoch 0")
    _assert(abs(ec10 - eps) < 1e-8, "Cosine curriculum eps = target at warmup end")

    # ==== train_epoch test ================================================
    model_ep = SimpleCNN(NC)
    opt_ep = torch.optim.Adam(model_ep.parameters(), lr=0.01)
    trainer_ep = AdversarialTrainer(model_ep, method="pgd_at", epsilon=eps,
                                    pgd_steps=2)
    ep_metrics = trainer_ep.train_epoch((x_data, y_data), opt_ep, epoch=0,
                                        batch_size=32)
    _assert("avg_loss" in ep_metrics, "train_epoch returns avg_loss")
    _assert(ep_metrics["avg_loss"] > 0, "train_epoch avg_loss > 0")

    # ==== Gradient clipping test ==========================================
    model_gc = SimpleCNN(NC)
    opt_gc = torch.optim.Adam(model_gc.parameters(), lr=0.01)
    trainer_gc = AdversarialTrainer(model_gc, method="pgd_at", epsilon=eps,
                                    pgd_steps=2, grad_clip_norm=1.0)
    trainer_gc.train_step(x_data[:16], y_data[:16], opt_gc)
    _assert(True, "Gradient clipping does not raise error")

    # ==== Invalid method test =============================================
    model_bad = SimpleCNN(NC)
    opt_bad = torch.optim.Adam(model_bad.parameters(), lr=0.01)
    trainer_bad = AdversarialTrainer(model_bad, method="invalid")
    try:
        trainer_bad.train_step(x_data[:4], y_data[:4], opt_bad)
        _assert(False, "Invalid method raises ValueError")
    except ValueError:
        _assert(True, "Invalid method raises ValueError")

    # ==== Schedule helper tests ==========================================
    _assert(linear_epsilon_schedule(0, 10, 1.0) == 0.1,
            "linear_epsilon_schedule(0, 10, 1.0) == 0.1")
    _assert(abs(linear_epsilon_schedule(10, 10, 1.0) - 1.0) < 1e-8,
            "linear_epsilon_schedule at warmup end == target")
    _assert(cosine_epsilon_schedule(0, 10, 1.0) > 0,
            "cosine_epsilon_schedule(0) > 0")

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    print("=" * 60)
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    _run_self_tests()
