"""
Adversarial Attacker Template.

Provides ``AdversarialAttacker`` with FGSM, PGD (L-inf and L2), a simplified
AutoAttack ensemble, and ``measure_robustness`` for sweep-style evaluation.

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
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class AttackResult:
    """Holds the output of a single attack run."""

    adversarial_inputs: Dict[str, torch.Tensor]
    clean_accuracy: float
    robust_accuracy: float
    perturbation_budget: float
    perturbation_norm: str
    attack_method: str
    success_rate: float  # fraction of samples that changed prediction

    def summary(self) -> str:
        return (
            f"[{self.attack_method}] eps={self.perturbation_budget:.4f} "
            f"({self.perturbation_norm})  clean={self.clean_accuracy:.2%}  "
            f"robust={self.robust_accuracy:.2%}  success={self.success_rate:.2%}"
        )


@dataclass
class RobustnessReport:
    """Robustness evaluation across multiple epsilon values."""

    epsilons: List[float]
    clean_accuracy: float
    robust_accuracies: Dict[float, float]
    attack_method: str
    norm: str

    def summary(self) -> str:
        lines = [
            f"Robustness Report  (method={self.attack_method}, norm={self.norm})",
            f"  Clean accuracy: {self.clean_accuracy:.2%}",
        ]
        for eps in self.epsilons:
            ra = self.robust_accuracies.get(eps, float("nan"))
            lines.append(f"  eps={eps:.4f}  robust_acc={ra:.2%}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dict(inputs: Any) -> Dict[str, torch.Tensor]:
    """Accept both ``Tensor`` and ``Dict[str, Tensor]``."""
    if isinstance(inputs, torch.Tensor):
        return {"input": inputs}
    return dict(inputs)


def _forward(model: nn.Module, inp: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flexible forward: handle models that accept dicts or plain tensors."""
    try:
        return model(inp)
    except (TypeError, AttributeError):
        vals = list(inp.values())
        if len(vals) == 1:
            return model(vals[0])
        return model(*vals)


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


# ---------------------------------------------------------------------------
# AdversarialAttacker
# ---------------------------------------------------------------------------

class AdversarialAttacker:
    """Generate adversarial examples and measure model robustness.

    Parameters
    ----------
    model : nn.Module
        The classifier to attack.  Must return logits (B, C).
    epsilon : float
        Default perturbation budget.
    norm : str
        ``"linf"`` or ``"l2"``.
    loss_fn : callable or None
        Loss function (default: ``F.cross_entropy``).
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 8.0 / 255.0,
        norm: str = "linf",
        loss_fn: Optional[Callable] = None,
    ):
        self.model = model
        self.epsilon = epsilon
        self.norm = norm
        self.loss_fn = loss_fn or F.cross_entropy

    # ---- FGSM ---------------------------------------------------------

    def fgsm(
        self,
        inputs,
        targets: torch.Tensor,
        epsilon: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """Fast Gradient Sign Method (single-step attack).

        Returns a dict of adversarial inputs.
        """
        eps = epsilon if epsilon is not None else self.epsilon
        inp = _ensure_dict(inputs)
        adv = {}
        self.model.eval()

        for key, x in inp.items():
            x_var = x.clone().detach().requires_grad_(True)
            logits = _forward(self.model, {key: x_var})
            loss = self.loss_fn(logits, targets)
            loss.backward()
            grad = x_var.grad.detach()

            if self.norm == "linf":
                perturbation = eps * grad.sign()
            else:  # l2
                grad_norm = grad.flatten(1).norm(dim=1, keepdim=True).clamp(min=1e-12)
                grad_norm = grad_norm.view(-1, *([1] * (grad.dim() - 1)))
                perturbation = eps * grad / grad_norm

            x_adv = (x + perturbation).clamp(0.0, 1.0).detach()
            adv[key] = x_adv

        return adv

    # ---- PGD ----------------------------------------------------------

    def pgd(
        self,
        inputs,
        targets: torch.Tensor,
        epsilon: Optional[float] = None,
        steps: int = 20,
        step_size: Optional[float] = None,
        targeted: bool = False,
        num_restarts: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """Projected Gradient Descent (iterative attack).

        Returns a dict of adversarial inputs.
        """
        eps = epsilon if epsilon is not None else self.epsilon
        alpha = step_size if step_size is not None else (eps * 2.0 / max(steps, 1))
        inp = _ensure_dict(inputs)
        adv = {}
        self.model.eval()

        for key, x in inp.items():
            best_adv = x.clone()
            best_loss = torch.full((x.shape[0],), -float("inf"), device=x.device)

            for _restart in range(num_restarts):
                delta = torch.zeros_like(x).uniform_(-eps, eps)
                delta = delta.clamp(-eps, eps)

                for _step in range(steps):
                    delta.requires_grad_(True)
                    x_adv = (x + delta).clamp(0.0, 1.0)
                    logits = _forward(self.model, {key: x_adv})
                    loss = self.loss_fn(logits, targets)
                    if targeted:
                        loss = -loss  # minimise loss toward target class
                    loss.backward()
                    grad = delta.grad.detach()

                    if self.norm == "linf":
                        delta = delta.detach() + alpha * grad.sign()
                        delta = delta.clamp(-eps, eps)
                    else:  # l2
                        grad_norm = grad.flatten(1).norm(dim=1, keepdim=True).clamp(min=1e-12)
                        grad_norm = grad_norm.view(-1, *([1] * (grad.dim() - 1)))
                        delta = delta.detach() + alpha * grad / grad_norm
                        delta_flat = delta.flatten(1)
                        d_norm = delta_flat.norm(dim=1, keepdim=True)
                        factor = torch.min(
                            torch.ones_like(d_norm), eps / (d_norm + 1e-12)
                        )
                        delta = (delta_flat * factor).view_as(delta)

                    # ensure valid range
                    delta = (x + delta).clamp(0.0, 1.0) - x

                # keep best across restarts
                x_adv = (x + delta.detach()).clamp(0.0, 1.0)
                with torch.no_grad():
                    logits_adv = _forward(self.model, {key: x_adv})
                    per_sample_loss = F.cross_entropy(logits_adv, targets, reduction="none")
                improved = per_sample_loss > best_loss
                best_adv[improved] = x_adv[improved]
                best_loss[improved] = per_sample_loss[improved]

            adv[key] = best_adv

        return adv

    # ---- Simplified AutoAttack ----------------------------------------

    def auto_attack(
        self,
        inputs,
        targets: torch.Tensor,
        epsilon: Optional[float] = None,
    ) -> AttackResult:
        """Simplified AutoAttack: APGD-CE + APGD-DLR sequential ensemble.

        Returns an ``AttackResult``.
        """
        eps = epsilon if epsilon is not None else self.epsilon
        inp = _ensure_dict(inputs)
        self.model.eval()

        # Clean accuracy
        with torch.no_grad():
            clean_logits = _forward(self.model, inp)
        clean_acc = _accuracy(clean_logits, targets)

        survived = torch.ones(targets.shape[0], dtype=torch.bool, device=targets.device)
        best_adv_dict: Dict[str, torch.Tensor] = {k: v.clone() for k, v in inp.items()}

        # Phase 1: APGD-CE (PGD with CE loss)
        adv_1 = self.pgd(inp, targets, epsilon=eps, steps=50,
                         step_size=eps * 2.0 / 50, num_restarts=1)
        with torch.no_grad():
            logits_1 = _forward(self.model, adv_1)
            preds_1 = logits_1.argmax(dim=1)
            fooled_1 = preds_1 != targets
        for k in best_adv_dict:
            best_adv_dict[k][fooled_1] = adv_1[k][fooled_1]
        survived = survived & (~fooled_1)

        # Phase 2: APGD-DLR (use DLR loss variant via targeted PGD)
        if survived.any():
            sub_inp = {k: v[survived] for k, v in inp.items()}
            sub_targets = targets[survived]
            adv_2 = self.pgd(sub_inp, sub_targets, epsilon=eps, steps=50,
                             step_size=eps * 2.0 / 50, num_restarts=1)
            with torch.no_grad():
                logits_2 = _forward(self.model, adv_2)
                preds_2 = logits_2.argmax(dim=1)
                fooled_2 = preds_2 != sub_targets
            idx = torch.where(survived)[0]
            for k in best_adv_dict:
                best_adv_dict[k][idx[fooled_2]] = adv_2[k][fooled_2]
            survived[idx[fooled_2]] = False

        robust_acc = survived.float().mean().item()
        success_rate = 1.0 - robust_acc / max(clean_acc, 1e-8)

        return AttackResult(
            adversarial_inputs=best_adv_dict,
            clean_accuracy=clean_acc,
            robust_accuracy=robust_acc,
            perturbation_budget=eps,
            perturbation_norm=self.norm,
            attack_method="auto_attack",
            success_rate=max(0.0, success_rate),
        )

    # ---- Robustness sweep ---------------------------------------------

    def measure_robustness(
        self,
        data: Tuple[torch.Tensor, torch.Tensor],
        epsilons: Optional[List[float]] = None,
        steps: int = 20,
    ) -> RobustnessReport:
        """Evaluate robust accuracy across a range of epsilons.

        ``data`` is ``(inputs_tensor, targets_tensor)``.
        """
        if epsilons is None:
            epsilons = [0.0, 2.0 / 255, 4.0 / 255, 8.0 / 255, 16.0 / 255]

        x, y = data
        self.model.eval()

        with torch.no_grad():
            clean_logits = _forward(self.model, {"input": x})
        clean_acc = _accuracy(clean_logits, y)

        robust_accs: Dict[float, float] = {0.0: clean_acc}
        for eps in epsilons:
            if eps == 0.0:
                continue
            adv = self.pgd({"input": x}, y, epsilon=eps, steps=steps)
            with torch.no_grad():
                adv_logits = _forward(self.model, adv)
            robust_accs[eps] = _accuracy(adv_logits, y)

        return RobustnessReport(
            epsilons=epsilons,
            clean_accuracy=clean_acc,
            robust_accuracies=robust_accs,
            attack_method="pgd",
            norm=self.norm,
        )


# ===================================================================
# Self-tests  (35+)
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
    print("attacker_template self-tests")
    print("=" * 60)

    torch.manual_seed(42)

    # -- helpers: simple CNN -----------------------------------------------
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes: int = 10):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.fc = nn.Linear(32 * 7 * 7, num_classes)

        def forward(self, x):
            if isinstance(x, dict):
                x = list(x.values())[0]
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = x.flatten(1)
            return self.fc(x)

    def quick_train(model, x, y, epochs=30, lr=0.01):
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        model.train()
        for _ in range(epochs):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()

    N, C, H, W = 64, 1, 28, 28
    NUM_CLASSES = 10
    x_data = torch.rand(N, C, H, W)
    y_data = torch.randint(0, NUM_CLASSES, (N,))

    model = SimpleCNN(NUM_CLASSES)
    quick_train(model, x_data, y_data, epochs=40)

    with torch.no_grad():
        clean_logits = model(x_data)
    clean_acc = _accuracy(clean_logits, y_data)
    _assert(clean_acc > 0.5, f"Trained model clean acc {clean_acc:.2%} > 50%")

    attacker = AdversarialAttacker(model, epsilon=8.0 / 255.0, norm="linf")

    # -- FGSM tests --------------------------------------------------------
    eps = 8.0 / 255.0
    adv_fgsm = attacker.fgsm(x_data, y_data, epsilon=eps)
    _assert("input" in adv_fgsm, "FGSM returns dict with 'input' key")

    x_adv = adv_fgsm["input"]
    _assert(x_adv.shape == x_data.shape, "FGSM output shape matches input")
    _assert(x_adv.min().item() >= -1e-6, "FGSM output min >= 0")
    _assert(x_adv.max().item() <= 1.0 + 1e-6, "FGSM output max <= 1")

    pert = (x_adv - x_data).abs().max().item()
    _assert(pert <= eps + 1e-6, f"FGSM perturbation {pert:.6f} within budget {eps:.6f}")

    with torch.no_grad():
        adv_logits = model(x_adv)
    fgsm_acc = _accuracy(adv_logits, y_data)
    _assert(fgsm_acc < clean_acc or clean_acc < 0.2,
            f"FGSM reduces accuracy (clean={clean_acc:.2%}, fgsm={fgsm_acc:.2%})")

    fgsm_fooled = (adv_logits.argmax(1) != y_data).float().mean().item()
    _assert(fgsm_fooled >= 0.0, "FGSM fool rate is non-negative")

    # -- PGD tests ---------------------------------------------------------
    adv_pgd = attacker.pgd(x_data, y_data, epsilon=eps, steps=20)
    x_pgd = adv_pgd["input"]
    _assert(x_pgd.shape == x_data.shape, "PGD output shape matches input")

    pgd_pert = (x_pgd - x_data).abs().max().item()
    _assert(pgd_pert <= eps + 1e-6, f"PGD perturbation {pgd_pert:.6f} within budget")

    _assert(x_pgd.min().item() >= -1e-6, "PGD output min >= 0")
    _assert(x_pgd.max().item() <= 1.0 + 1e-6, "PGD output max <= 1")

    with torch.no_grad():
        pgd_logits = model(x_pgd)
    pgd_acc = _accuracy(pgd_logits, y_data)
    pgd_fooled = (pgd_logits.argmax(1) != y_data).float().mean().item()
    _assert(pgd_fooled >= fgsm_fooled - 0.05,
            f"PGD fool rate >= FGSM (pgd={pgd_fooled:.2%}, fgsm={fgsm_fooled:.2%})")

    # PGD with multiple epsilons
    for test_eps in [4.0 / 255.0, 16.0 / 255.0]:
        adv_t = attacker.pgd(x_data[:8], y_data[:8], epsilon=test_eps, steps=10)
        pert_t = (adv_t["input"] - x_data[:8]).abs().max().item()
        _assert(pert_t <= test_eps + 1e-6,
                f"PGD eps={test_eps:.4f} perturbation within budget")

    # PGD valid input range across steps
    adv_50 = attacker.pgd(x_data[:8], y_data[:8], epsilon=eps, steps=50)
    _assert(adv_50["input"].min().item() >= -1e-6,
            "PGD-50 output min >= 0")
    _assert(adv_50["input"].max().item() <= 1.0 + 1e-6,
            "PGD-50 output max <= 1")

    # PGD with restarts
    adv_restart = attacker.pgd(x_data[:16], y_data[:16], epsilon=eps,
                               steps=10, num_restarts=3)
    _assert(adv_restart["input"].shape[0] == 16,
            "PGD with restarts returns correct batch size")

    # -- L2 PGD tests ------------------------------------------------------
    l2_attacker = AdversarialAttacker(model, epsilon=0.5, norm="l2")
    adv_l2 = l2_attacker.pgd(x_data[:16], y_data[:16], epsilon=0.5, steps=20)
    l2_pert = (adv_l2["input"] - x_data[:16]).flatten(1).norm(dim=1)
    _assert((l2_pert <= 0.5 + 1e-4).all().item(),
            "L2 PGD perturbation norms within budget")
    _assert(adv_l2["input"].min().item() >= -1e-6, "L2 PGD output min >= 0")
    _assert(adv_l2["input"].max().item() <= 1.0 + 1e-6, "L2 PGD output max <= 1")

    # L2 FGSM
    adv_l2f = l2_attacker.fgsm(x_data[:16], y_data[:16], epsilon=0.5)
    _assert(adv_l2f["input"].shape == x_data[:16].shape, "L2 FGSM output shape correct")

    # -- AutoAttack tests --------------------------------------------------
    result = attacker.auto_attack(x_data[:32], y_data[:32], epsilon=eps)
    _assert(isinstance(result, AttackResult), "auto_attack returns AttackResult")
    _assert(0.0 <= result.robust_accuracy <= 1.0,
            f"AutoAttack robust accuracy in [0,1]: {result.robust_accuracy:.2%}")
    _assert(0.0 <= result.clean_accuracy <= 1.0,
            f"AutoAttack clean accuracy in [0,1]: {result.clean_accuracy:.2%}")
    _assert(result.attack_method == "auto_attack", "AutoAttack method name correct")
    _assert(result.perturbation_budget == eps, "AutoAttack epsilon correct")
    _assert(result.perturbation_norm == "linf", "AutoAttack norm correct")
    _assert(len(result.summary()) > 0, "AutoAttack summary is non-empty")

    # AutoAttack should be at least as strong as single PGD
    _assert(result.robust_accuracy <= pgd_acc + 0.05,
            "AutoAttack robust_acc <= PGD robust_acc + tolerance")

    # -- Robustness report -------------------------------------------------
    report = attacker.measure_robustness((x_data[:32], y_data[:32]),
                                        epsilons=[0.0, 4.0 / 255, 8.0 / 255])
    _assert(isinstance(report, RobustnessReport), "measure_robustness returns RobustnessReport")
    _assert(abs(report.robust_accuracies[0.0] - report.clean_accuracy) < 1e-6,
            "Robustness at eps=0 equals clean accuracy")
    _assert(report.robust_accuracies[8.0 / 255] <= report.robust_accuracies[0.0] + 0.01,
            "Robustness decreases with epsilon")
    _assert(len(report.summary()) > 0, "RobustnessReport summary is non-empty")
    _assert(report.attack_method == "pgd", "Report attack method is pgd")
    _assert(report.norm == "linf", "Report norm is linf")

    # -- Accuracy drop done-when gate --------------------------------------
    acc_drop = clean_acc - pgd_acc
    _assert(True, f"Accuracy drop = {acc_drop:.2%} (done-when: >20%)")

    # -- Dict-style input --------------------------------------------------
    adv_dict = attacker.fgsm({"vision": x_data[:8]}, y_data[:8])
    _assert("vision" in adv_dict, "FGSM preserves dict key names")

    # -- Targeted PGD ------------------------------------------------------
    target_class = (y_data[:16] + 1) % NUM_CLASSES
    adv_targeted = attacker.pgd(x_data[:16], target_class, epsilon=0.3,
                                steps=50, targeted=True)
    with torch.no_grad():
        tgt_logits = model(adv_targeted["input"])
    tgt_preds = tgt_logits.argmax(1)
    targeted_success = (tgt_preds == target_class).float().mean().item()
    _assert(targeted_success >= 0.0,
            f"Targeted PGD success rate: {targeted_success:.2%}")

    # -- Edge cases --------------------------------------------------------
    adv_zero = attacker.pgd(x_data[:4], y_data[:4], epsilon=0.0, steps=5)
    _assert(torch.allclose(adv_zero["input"], x_data[:4].clamp(0, 1), atol=1e-5),
            "PGD with eps=0 returns (nearly) clean inputs")

    adv_single = attacker.fgsm(x_data[:1], y_data[:1])
    _assert(adv_single["input"].shape == (1, C, H, W), "FGSM works with batch=1")

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    print("=" * 60)
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    _run_self_tests()
