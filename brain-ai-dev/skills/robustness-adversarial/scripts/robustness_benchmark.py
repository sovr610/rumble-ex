#!/usr/bin/env python3
"""
Full Robustness Evaluation Benchmark.

Runs a comprehensive robustness evaluation suite on a simple CNN:
  1. Clean accuracy baseline
  2. FGSM / PGD / AutoAttack robustness sweep
  3. OOD detection (energy, MSP, Mahalanobis)
  4. Corruption benchmark (all 15 corruptions, 5 severities)
  5. Calibration analysis (ECE, temperature scaling)
  6. Adversarial training comparison (PGD-AT, TRADES, Free-AT)

Produces a structured text report.

Usage:
    python scripts/robustness_benchmark.py
    python scripts/robustness_benchmark.py --quick     # shorter run
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ASSET_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "assets")
sys.path.insert(0, _ASSET_DIR)

from attacker_template import AdversarialAttacker, RobustnessReport
from ood_detector_template import OODDetector, OODMetrics
from corruption_benchmark_template import CorruptionBenchmark, CorruptionReport
from adversarial_trainer_template import AdversarialTrainer
from calibration_template import CalibrationAnalyzer, CalibrationResult


# ---------------------------------------------------------------------------
# Model and data
# ---------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    """Small CNN for benchmarking (28x28 single-channel)."""

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


def make_data(
    n_train: int = 200,
    n_test: int = 100,
    num_classes: int = 10,
    seed: int = 42,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    torch.manual_seed(seed)
    x_train = torch.rand(n_train, 1, 28, 28)
    y_train = torch.randint(0, num_classes, (n_train,))
    x_test = torch.rand(n_test, 1, 28, 28)
    y_test = torch.randint(0, num_classes, (n_test,))

    # OOD data (shifted distribution)
    x_ood = torch.rand(n_test, 1, 28, 28) * 0.3 + 0.7
    y_ood = torch.randint(0, num_classes, (n_test,))

    return {
        "train": (x_train, y_train),
        "test": (x_test, y_test),
        "ood": (x_ood, y_ood),
    }


def train_model(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor,
    epochs: int = 50, lr: float = 0.01,
) -> nn.Module:
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
        return (model(x).argmax(1) == y).float().mean().item()


# ---------------------------------------------------------------------------
# Benchmark sections
# ---------------------------------------------------------------------------

def section_clean(model: nn.Module, data: dict) -> List[str]:
    lines = ["", "=" * 60, "Section 1: Clean Accuracy", "=" * 60]
    x_tr, y_tr = data["train"]
    x_te, y_te = data["test"]
    train_acc = get_accuracy(model, x_tr, y_tr)
    test_acc = get_accuracy(model, x_te, y_te)
    lines.append(f"  Train accuracy: {train_acc:.2%}")
    lines.append(f"  Test accuracy:  {test_acc:.2%}")
    return lines


def section_attacks(model: nn.Module, data: dict, quick: bool) -> List[str]:
    lines = ["", "=" * 60, "Section 2: Adversarial Attack Robustness", "=" * 60]
    x, y = data["test"]
    eps_list = [2.0 / 255, 4.0 / 255, 8.0 / 255]
    if not quick:
        eps_list.append(16.0 / 255)

    attacker = AdversarialAttacker(model, norm="linf")

    # FGSM sweep
    lines.append("\n  FGSM:")
    for eps in eps_list:
        adv = attacker.fgsm(x, y, epsilon=eps)
        acc = get_accuracy(model, adv["input"], y)
        lines.append(f"    eps={eps:.4f}  robust_acc={acc:.2%}")

    # PGD sweep
    steps = 10 if quick else 20
    lines.append(f"\n  PGD-{steps}:")
    for eps in eps_list:
        adv = attacker.pgd(x, y, epsilon=eps, steps=steps)
        acc = get_accuracy(model, adv["input"], y)
        lines.append(f"    eps={eps:.4f}  robust_acc={acc:.2%}")

    # Robustness report
    report = attacker.measure_robustness((x, y), epsilons=[0.0] + eps_list,
                                          steps=steps)
    lines.append(f"\n  Robustness Report (PGD-{steps}):")
    for eps in report.epsilons:
        ra = report.robust_accuracies.get(eps, float("nan"))
        lines.append(f"    eps={eps:.4f}  robust_acc={ra:.2%}")

    # AutoAttack (small subset)
    if not quick:
        result = attacker.auto_attack(x[:32], y[:32], epsilon=8.0 / 255.0)
        lines.append(f"\n  AutoAttack (eps=8/255, n=32):")
        lines.append(f"    clean={result.clean_accuracy:.2%}  "
                      f"robust={result.robust_accuracy:.2%}  "
                      f"success={result.success_rate:.2%}")

    return lines


def section_ood(model: nn.Module, data: dict) -> List[str]:
    lines = ["", "=" * 60, "Section 3: OOD Detection", "=" * 60]
    x_id, y_id = data["test"]
    x_ood, y_ood = data["ood"]

    for method in ["energy", "msp"]:
        det = OODDetector(model, method=method, temperature=1.0)
        det.fit((x_id, y_id))
        metrics = det.evaluate((x_id, y_id), (x_ood, y_ood))
        lines.append(f"\n  Method: {method}")
        lines.append(f"    AUROC:     {metrics.auroc:.4f}")
        lines.append(f"    FPR@95:    {metrics.fpr_at_95tpr:.4f}")
        lines.append(f"    ID mean:   {metrics.id_score_mean:.4f}")
        lines.append(f"    OOD mean:  {metrics.ood_score_mean:.4f}")

    # Mahalanobis
    det_m = OODDetector(model, method="mahalanobis", regularization=1e-4)
    det_m.fit((x_id, y_id))
    metrics_m = det_m.evaluate((x_id, y_id), (x_ood, y_ood))
    lines.append(f"\n  Method: mahalanobis")
    lines.append(f"    AUROC:     {metrics_m.auroc:.4f}")
    lines.append(f"    FPR@95:    {metrics_m.fpr_at_95tpr:.4f}")

    return lines


def section_corruption(model: nn.Module, data: dict, quick: bool) -> List[str]:
    lines = ["", "=" * 60, "Section 4: Corruption Benchmark", "=" * 60]
    x, y = data["test"]

    if quick:
        corruptions = ["gaussian_noise", "brightness", "contrast",
                       "defocus_blur", "pixelate"]
    else:
        corruptions = None  # all 15

    bench = CorruptionBenchmark(model, corruptions=corruptions)
    report = bench.run((x, y))

    lines.append(f"\n  Clean accuracy: {report.clean_accuracy:.2%}")
    lines.append(f"  Absolute MCE:   {report.mce:.4f}")
    lines.append("")
    lines.append(f"  {'Corruption':<22} | {'Sev1':>5} {'Sev2':>5} "
                  f"{'Sev3':>5} {'Sev4':>5} {'Sev5':>5} | Mono")
    lines.append("  " + "-" * 65)

    for cname, accs in report.accuracies.items():
        vals = " ".join(f"{accs.get(s, 0):.0%}" for s in range(1, 6))
        mono = "ok" if report.monotonicity.get(cname, False) else "FAIL"
        lines.append(f"  {cname:<22} | {vals} | {mono}")

    n_mono = sum(1 for v in report.monotonicity.values() if v)
    lines.append(f"\n  Monotonic: {n_mono}/{len(report.monotonicity)}")

    return lines


def section_calibration(model: nn.Module, data: dict) -> List[str]:
    lines = ["", "=" * 60, "Section 5: Calibration Analysis", "=" * 60]
    x, y = data["test"]

    model.eval()
    with torch.no_grad():
        logits = model(x)

    analyzer = CalibrationAnalyzer(n_bins=15)
    result = analyzer.analyze(logits, y)

    lines.append(f"\n  ECE:           {result.ece:.4f}")
    lines.append(f"  MCE:           {result.mce:.4f}")
    lines.append(f"  Temperature:   {result.optimal_temperature:.4f}")
    lines.append(f"  ECE (scaled):  {result.ece_after_scaling:.4f}")

    # Under shift
    shifted_logits = logits + torch.randn_like(logits) * 1.5
    shift = analyzer.calibration_under_shift((logits, y), (shifted_logits, y))
    lines.append(f"\n  Calibration under shift:")
    lines.append(f"    ECE clean:   {shift['ece_clean']:.4f}")
    lines.append(f"    ECE shifted: {shift['ece_shifted']:.4f}")
    lines.append(f"    Delta:       {shift['ece_delta']:.4f}")

    return lines


def section_adv_training(data: dict, quick: bool) -> List[str]:
    lines = ["", "=" * 60, "Section 6: Adversarial Training Comparison", "=" * 60]
    x, y = data["train"]
    x_te, y_te = data["test"]
    eps = 8.0 / 255.0
    epochs = 10 if quick else 20
    pgd_steps_for_eval = 5 if quick else 10

    for method_name in ["pgd_at", "trades", "free_at"]:
        torch.manual_seed(0)
        model = SimpleCNN(10)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)

        kwargs = {"method": method_name, "epsilon": eps, "pgd_steps": 3,
                  "pgd_step_size": 2.0 / 255.0}
        if method_name == "trades":
            kwargs["trades_beta"] = 6.0
        if method_name == "free_at":
            kwargs["free_at_replays"] = 2

        trainer = AdversarialTrainer(model, **kwargs)

        losses = []
        for ep in range(epochs):
            perm = torch.randperm(x.shape[0])
            for i in range(0, x.shape[0], 64):
                idx = perm[i:i + 64]
                m = trainer.train_step(x[idx], y[idx], opt)
                losses.append(m["loss"])

        model.eval()
        clean_acc = get_accuracy(model, x_te, y_te)

        # Evaluate robustness
        attacker = AdversarialAttacker(model, epsilon=eps, norm="linf")
        adv = attacker.pgd(x_te, y_te, epsilon=eps, steps=pgd_steps_for_eval)
        robust_acc = get_accuracy(model, adv["input"], y_te)

        lines.append(f"\n  Method: {method_name}")
        lines.append(f"    Loss (first): {losses[0]:.4f}")
        lines.append(f"    Loss (last):  {losses[-1]:.4f}")
        lines.append(f"    Clean acc:    {clean_acc:.2%}")
        lines.append(f"    Robust acc:   {robust_acc:.2%} "
                      f"(PGD-{pgd_steps_for_eval}, eps=8/255)")

    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Full robustness benchmark")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode with fewer samples and steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    n_train = 100 if args.quick else 200
    n_test = 50 if args.quick else 100

    data = make_data(n_train=n_train, n_test=n_test, seed=args.seed)

    # Train baseline model
    model = SimpleCNN(10)
    train_model(model, *data["train"], epochs=30 if args.quick else 50)

    report_lines: List[str] = []
    report_lines.append("=" * 60)
    report_lines.append("ROBUSTNESS BENCHMARK REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"Seed: {args.seed}  Quick: {args.quick}")
    report_lines.append(f"Train: {n_train}  Test: {n_test}")

    t0 = time.time()

    report_lines.extend(section_clean(model, data))
    report_lines.extend(section_attacks(model, data, args.quick))
    report_lines.extend(section_ood(model, data))
    report_lines.extend(section_corruption(model, data, args.quick))
    report_lines.extend(section_calibration(model, data))
    report_lines.extend(section_adv_training(data, args.quick))

    elapsed = time.time() - t0

    report_lines.append("")
    report_lines.append("=" * 60)
    report_lines.append(f"Benchmark complete in {elapsed:.1f}s")
    report_lines.append("=" * 60)

    report_text = "\n".join(report_lines)
    print(report_text)


if __name__ == "__main__":
    main()
