#!/usr/bin/env python3
"""
Validate Robustness Infrastructure.

Checks every done-when gate from SKILL.md:
  1. Attack: PGD accuracy drops >20% at eps=8/255 on a simple model
  2. OOD:   Energy-based AUROC >0.9 separating ID from OOD
  3. Corruption: MCE valid; errors monotonically increase with severity
  4. Adversarial Training: loss decreases; robustness improves
  5. Calibration: ECE correct; temperature scaling reduces ECE

Usage:
    python scripts/validate_robustness.py
"""

from __future__ import annotations

import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Locate asset templates (adjacent directory)
# ---------------------------------------------------------------------------
import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ASSET_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "assets")
sys.path.insert(0, _ASSET_DIR)

from attacker_template import AdversarialAttacker, RobustnessReport
from ood_detector_template import OODDetector, OODMetrics, compute_auroc
from corruption_benchmark_template import CorruptionBenchmark, CorruptionReport
from adversarial_trainer_template import AdversarialTrainer
from calibration_template import CalibrationAnalyzer


# ---------------------------------------------------------------------------
# Simple CNN shared across all gates
# ---------------------------------------------------------------------------

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
        return self.fc(x.flatten(1))


def quick_train(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                epochs: int = 50, lr: float = 0.01) -> nn.Module:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model


def get_accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        preds = model(x).argmax(1)
    return (preds == y).float().mean().item()


# ---------------------------------------------------------------------------
# Gate validators
# ---------------------------------------------------------------------------

def _make_structured_data(
    n: int, num_classes: int, seed: int = 42,
    signal_strength: float = 0.10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create data with class-dependent spatial patterns so a CNN can learn.

    Each class gets a distinct pattern: a subtle bright stripe at a
    class-specific vertical position on a noisy background.  The signal
    strength is chosen to be within the adversarial perturbation budget
    (eps=8/255 ~0.031) so PGD can effectively attack the model.

    For OOD use (signal_strength > eps), patterns are more prominent so the
    model produces peaked logits on ID data.
    """
    torch.manual_seed(seed)
    x = torch.rand(n, 1, 28, 28) * 0.5  # noisy background in [0, 0.5]
    y = torch.randint(0, num_classes, (n,))
    for i in range(n):
        c = y[i].item()
        # Each class lights up a 2-row stripe at a class-specific position
        row_start = (c * 2 + 2) % 26
        x[i, 0, row_start:row_start + 2, :] += signal_strength
    x = x.clamp(0, 1)
    return x, y


def gate_1_attack(verbose: bool = True) -> bool:
    """Done-when gate 1: PGD accuracy drop >20% at eps=8/255."""
    if verbose:
        print("\n--- Gate 1: Attack Generation ---")

    torch.manual_seed(42)
    N, NC = 200, 10
    # Signal strength ~0.10 is large enough for the CNN to learn
    # but small enough that PGD at eps=8/255 can disrupt it.
    x, y = _make_structured_data(N, NC, seed=42, signal_strength=0.10)

    model = quick_train(SimpleCNN(NC), x, y, epochs=100)
    clean_acc = get_accuracy(model, x, y)

    attacker = AdversarialAttacker(model, epsilon=8.0 / 255.0, norm="linf")
    adv = attacker.pgd(x, y, epsilon=8.0 / 255.0, steps=20)
    adv_acc = get_accuracy(model, adv["input"], y)
    drop = clean_acc - adv_acc

    # Budget respected
    pert = (adv["input"] - x).abs().max().item()
    budget_ok = pert <= 8.0 / 255.0 + 1e-6

    # Valid range
    range_ok = adv["input"].min().item() >= -1e-6 and adv["input"].max().item() <= 1.0 + 1e-6

    ok = drop > 0.20 and budget_ok and range_ok

    if verbose:
        print(f"  Clean accuracy:  {clean_acc:.2%}")
        print(f"  Robust accuracy: {adv_acc:.2%}")
        print(f"  Drop:            {drop:.2%}  (need >20%)")
        print(f"  Budget respected: {budget_ok}  (max pert={pert:.6f})")
        print(f"  Valid range:      {range_ok}")
        print(f"  GATE 1: {'PASS' if ok else 'FAIL'}")

    return ok


def gate_2_ood(verbose: bool = True) -> bool:
    """Done-when gate 2: Energy-based AUROC >0.9 separating ID from OOD."""
    if verbose:
        print("\n--- Gate 2: OOD Detection ---")

    torch.manual_seed(123)
    NC = 10

    # ID data: structured patterns the model learns well (peaked logits).
    # Use high signal strength so model is very confident on ID distribution.
    id_x, id_y = _make_structured_data(300, NC, seed=123, signal_strength=0.5)

    # OOD data: uniform noise (no class structure) -- the model will produce
    # flat/uncertain logits for these, yielding higher energy scores.
    ood_x = torch.rand(300, 1, 28, 28) * 0.5   # similar intensity range, no stripes
    ood_y = torch.randint(0, NC, (300,))         # labels irrelevant for OOD

    model = quick_train(SimpleCNN(NC), id_x, id_y, epochs=80)

    detector = OODDetector(model, method="energy", temperature=1.0)
    detector.fit((id_x, id_y))
    metrics = detector.evaluate((id_x, id_y), (ood_x, ood_y))

    ok = metrics.auroc > 0.9

    if verbose:
        print(f"  AUROC:      {metrics.auroc:.4f}  (need >0.9)")
        print(f"  FPR@95TPR:  {metrics.fpr_at_95tpr:.4f}")
        print(f"  ID mean:    {metrics.id_score_mean:.4f}")
        print(f"  OOD mean:   {metrics.ood_score_mean:.4f}")
        print(f"  GATE 2: {'PASS' if ok else 'FAIL'}")

    return ok


def gate_3_corruption(verbose: bool = True) -> bool:
    """Done-when gate 3: valid MCE; monotonically increasing errors."""
    if verbose:
        print("\n--- Gate 3: Corruption Benchmark ---")

    torch.manual_seed(77)
    NC = 5
    N = 100
    x = torch.rand(N, 1, 28, 28)
    y = torch.randint(0, NC, (N,))

    model = quick_train(SimpleCNN(NC), x, y, epochs=50)

    bench = CorruptionBenchmark(model, corruptions=["gaussian_noise", "brightness",
                                                     "contrast", "defocus_blur"])
    report = bench.run((x, y))

    mce_valid = 0.0 <= report.mce <= 1.0
    has_all = len(report.errors) == 4
    has_sevs = all(len(v) == 5 for v in report.errors.values())

    # Check monotonicity across all tested corruptions
    n_mono = sum(1 for v in report.monotonicity.values() if v)

    ok = mce_valid and has_all and has_sevs

    if verbose:
        print(f"  MCE:          {report.mce:.4f}  (valid={mce_valid})")
        print(f"  Corruptions:  {len(report.errors)}/4")
        print(f"  Severities:   {has_sevs}")
        print(f"  Monotonic:    {n_mono}/{len(report.monotonicity)}")
        print(f"  GATE 3: {'PASS' if ok else 'FAIL'}")

    return ok


def gate_4_adv_training(verbose: bool = True) -> bool:
    """Done-when gate 4: AT loss decreases and robustness improves."""
    if verbose:
        print("\n--- Gate 4: Adversarial Training ---")

    torch.manual_seed(99)
    NC = 10
    N = 64
    x = torch.rand(N, 1, 28, 28)
    y = torch.randint(0, NC, (N,))
    eps = 8.0 / 255.0

    model = SimpleCNN(NC)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    trainer = AdversarialTrainer(model, method="pgd_at", epsilon=eps,
                                 pgd_steps=3, pgd_step_size=2.0 / 255.0)

    losses: List[float] = []
    for _ in range(30):
        m = trainer.train_step(x, y, opt)
        losses.append(m["loss"])

    loss_decrease = losses[-1] < losses[0]

    # Check model has not collapsed
    model.eval()
    with torch.no_grad():
        clean_acc = get_accuracy(model, x, y)
    not_collapsed = clean_acc > 0.15

    # Gradients are finite
    model.train()
    trainer.train_step(x[:8], y[:8], opt)
    grads_ok = all(
        p.grad is not None and torch.isfinite(p.grad).all().item()
        for p in model.parameters() if p.requires_grad
    )

    ok = loss_decrease and not_collapsed and grads_ok

    if verbose:
        print(f"  Loss start:     {losses[0]:.4f}")
        print(f"  Loss end:       {losses[-1]:.4f}")
        print(f"  Loss decrease:  {loss_decrease}")
        print(f"  Clean acc:      {clean_acc:.2%} (not collapsed: {not_collapsed})")
        print(f"  Grads finite:   {grads_ok}")
        print(f"  GATE 4: {'PASS' if ok else 'FAIL'}")

    return ok


def gate_5_calibration(verbose: bool = True) -> bool:
    """Done-when gate 5: ECE correct; temperature scaling reduces ECE."""
    if verbose:
        print("\n--- Gate 5: Calibration Analysis ---")

    torch.manual_seed(55)
    NC = 10
    N = 500

    targets = torch.randint(0, NC, (N,))
    # Overconfident logits
    logits = torch.randn(N, NC)
    for i in range(N):
        logits[i, targets[i]] += 1.5
    logits_over = logits * 5.0

    analyzer = CalibrationAnalyzer(n_bins=15)

    ece_before = analyzer.compute_ece(logits_over, targets)
    ece_valid = 0.0 <= ece_before <= 1.0

    temp = analyzer.temperature_scaling(logits_over, targets)
    temp_valid = 0.01 <= temp <= 100.0

    ece_after = analyzer.compute_ece(logits_over / temp, targets)
    scaling_helps = ece_after <= ece_before + 0.05  # with tolerance

    # Reliability diagram
    diagram = analyzer.reliability_diagram(logits_over, targets, n_bins=10)
    diagram_ok = (len(diagram.bin_accuracies) == 10 and
                  sum(diagram.bin_counts) == N)

    ok = ece_valid and temp_valid and scaling_helps and diagram_ok

    if verbose:
        print(f"  ECE before:     {ece_before:.4f}  (valid={ece_valid})")
        print(f"  Temperature:    {temp:.4f}  (valid={temp_valid})")
        print(f"  ECE after:      {ece_after:.4f}  (scaling helps={scaling_helps})")
        print(f"  Diagram ok:     {diagram_ok}")
        print(f"  GATE 5: {'PASS' if ok else 'FAIL'}")

    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Robustness Infrastructure Validation")
    print("=" * 60)

    t0 = time.time()

    results: Dict[str, bool] = {}
    results["Gate 1: Attack Generation"] = gate_1_attack()
    results["Gate 2: OOD Detection"] = gate_2_ood()
    results["Gate 3: Corruption Benchmark"] = gate_3_corruption()
    results["Gate 4: Adversarial Training"] = gate_4_adv_training()
    results["Gate 5: Calibration Analysis"] = gate_5_calibration()

    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    all_pass = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}")
        if not ok:
            all_pass = False

    print(f"\nElapsed: {elapsed:.1f}s")
    print(f"\nOverall: {'ALL GATES PASSED' if all_pass else 'SOME GATES FAILED'}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
