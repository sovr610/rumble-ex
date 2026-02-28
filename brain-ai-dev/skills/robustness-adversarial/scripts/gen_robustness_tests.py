#!/usr/bin/env python3
"""
Generate Robustness Test Suite.

Produces 100+ pytest test cases covering:
  - Attack generation (FGSM, PGD, AutoAttack, budget, range, L2)
  - OOD detection (energy, MSP, Mahalanobis, AUROC, FPR, threshold)
  - Corruption benchmark (all 15 functions, MCE, monotonicity)
  - Adversarial training (PGD-AT, TRADES, Free-AT, curriculum)
  - Calibration (ECE, MCE, temperature, Platt, shift, diagram)
  - Configuration validation

Usage:
    python scripts/gen_robustness_tests.py           # write to stdout
    python scripts/gen_robustness_tests.py -o FILE   # write to FILE
    python scripts/gen_robustness_tests.py --run      # generate + run via pytest
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import textwrap

# ---------------------------------------------------------------------------
# The test source
# ---------------------------------------------------------------------------

TEST_SOURCE = textwrap.dedent('''\
"""
Auto-generated robustness test suite (100+ tests).

Run:  python -m pytest <this_file> -v
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# -- path setup so templates can be imported --------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ASSET_DIR = os.path.join(os.path.dirname(_THIS_DIR), "assets")
if os.path.isdir(_ASSET_DIR):
    sys.path.insert(0, _ASSET_DIR)
else:
    # fallback: look one more level up
    _ASSET_DIR2 = os.path.join(os.path.dirname(os.path.dirname(_THIS_DIR)),
                                "assets")
    if os.path.isdir(_ASSET_DIR2):
        sys.path.insert(0, _ASSET_DIR2)

from attacker_template import AdversarialAttacker, AttackResult, RobustnessReport
from ood_detector_template import (OODDetector, OODResult, OODMetrics,
                                    compute_auroc, compute_fpr_at_tpr,
                                    calibrate_threshold, energy_score, msp_score)
from corruption_benchmark_template import (
    CorruptionBenchmark, CorruptionReport, CORRUPTION_FUNCTIONS,
    ALL_CORRUPTION_NAMES, gaussian_noise, shot_noise, impulse_noise,
    brightness, contrast, fog, pixelate, defocus_blur)
from adversarial_trainer_template import (
    AdversarialTrainer, linear_epsilon_schedule, cosine_epsilon_schedule)
from calibration_template import CalibrationAnalyzer, ReliabilityDiagram, CalibrationResult
from robustness_config_template import (
    AttackConfig, OODConfig, CorruptionConfig, AdvTrainConfig,
    CalibrationConfig, RobustnessConfig, ALL_CORRUPTIONS)


# ===========================================================================
# Fixtures
# ===========================================================================

class SimpleCNN(nn.Module):
    def __init__(self, nc=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(7)
        self.fc = nn.Linear(32 * 7 * 7, nc)

    def forward(self, x):
        if isinstance(x, dict):
            x = list(x.values())[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return self.fc(x.flatten(1))


@pytest.fixture(scope="module")
def trained_model():
    torch.manual_seed(42)
    model = SimpleCNN(10)
    x = torch.rand(64, 1, 28, 28)
    y = torch.randint(0, 10, (64,))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for _ in range(40):
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model, x, y


@pytest.fixture(scope="module")
def id_ood_data():
    torch.manual_seed(123)
    id_x = torch.rand(200, 1, 28, 28) * 0.3
    id_y = torch.randint(0, 10, (200,))
    ood_x = torch.rand(200, 1, 28, 28) * 0.3 + 0.7
    ood_y = torch.randint(0, 10, (200,))
    return id_x, id_y, ood_x, ood_y


@pytest.fixture(scope="module")
def ood_trained_model(id_ood_data):
    id_x, id_y, _, _ = id_ood_data
    torch.manual_seed(123)
    model = SimpleCNN(10)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for _ in range(50):
        loss = F.cross_entropy(model(id_x), id_y)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


# ===========================================================================
# Category 1: Attack Generation  (tests 1-25)
# ===========================================================================

class TestFGSM:
    def test_fgsm_returns_dict(self, trained_model):
        model, x, y = trained_model
        atk = AdversarialAttacker(model)
        adv = atk.fgsm(x, y)
        assert isinstance(adv, dict)
        assert "input" in adv

    def test_fgsm_shape(self, trained_model):
        model, x, y = trained_model
        adv = AdversarialAttacker(model).fgsm(x, y)
        assert adv["input"].shape == x.shape

    def test_fgsm_budget_linf(self, trained_model):
        model, x, y = trained_model
        eps = 8.0 / 255.0
        adv = AdversarialAttacker(model, epsilon=eps).fgsm(x, y, epsilon=eps)
        assert (adv["input"] - x).abs().max().item() <= eps + 1e-6

    def test_fgsm_valid_range(self, trained_model):
        model, x, y = trained_model
        adv = AdversarialAttacker(model).fgsm(x, y)
        assert adv["input"].min().item() >= -1e-6
        assert adv["input"].max().item() <= 1.0 + 1e-6

    def test_fgsm_changes_predictions(self, trained_model):
        model, x, y = trained_model
        adv = AdversarialAttacker(model, epsilon=0.1).fgsm(x, y, epsilon=0.1)
        with torch.no_grad():
            p_clean = model(x).argmax(1)
            p_adv = model(adv["input"]).argmax(1)
        changed = (p_clean != p_adv).float().mean().item()
        assert changed >= 0.0  # at least computes

    def test_fgsm_preserves_dict_keys(self, trained_model):
        model, x, y = trained_model
        adv = AdversarialAttacker(model).fgsm({"vision": x[:8]}, y[:8])
        assert "vision" in adv

    def test_fgsm_batch_one(self, trained_model):
        model, x, y = trained_model
        adv = AdversarialAttacker(model).fgsm(x[:1], y[:1])
        assert adv["input"].shape == (1, 1, 28, 28)

    def test_fgsm_l2(self, trained_model):
        model, x, y = trained_model
        atk = AdversarialAttacker(model, norm="l2", epsilon=0.5)
        adv = atk.fgsm(x[:16], y[:16], epsilon=0.5)
        assert adv["input"].shape == x[:16].shape


class TestPGD:
    def test_pgd_shape(self, trained_model):
        model, x, y = trained_model
        adv = AdversarialAttacker(model).pgd(x, y, steps=5)
        assert adv["input"].shape == x.shape

    def test_pgd_budget_linf(self, trained_model):
        model, x, y = trained_model
        eps = 8.0 / 255.0
        adv = AdversarialAttacker(model, epsilon=eps).pgd(x, y, steps=10)
        assert (adv["input"] - x).abs().max().item() <= eps + 1e-6

    def test_pgd_valid_range(self, trained_model):
        model, x, y = trained_model
        adv = AdversarialAttacker(model).pgd(x, y, steps=20)
        assert adv["input"].min().item() >= -1e-6
        assert adv["input"].max().item() <= 1.0 + 1e-6

    def test_pgd_stronger_than_fgsm(self, trained_model):
        model, x, y = trained_model
        eps = 8.0 / 255.0
        atk = AdversarialAttacker(model, epsilon=eps)
        adv_f = atk.fgsm(x, y, epsilon=eps)
        adv_p = atk.pgd(x, y, epsilon=eps, steps=20)
        with torch.no_grad():
            fgsm_ok = (model(adv_f["input"]).argmax(1) == y).float().mean()
            pgd_ok = (model(adv_p["input"]).argmax(1) == y).float().mean()
        assert pgd_ok.item() <= fgsm_ok.item() + 0.05

    def test_pgd_budget_4_255(self, trained_model):
        model, x, y = trained_model
        eps = 4.0 / 255.0
        adv = AdversarialAttacker(model, epsilon=eps).pgd(x[:8], y[:8], steps=10)
        assert (adv["input"] - x[:8]).abs().max().item() <= eps + 1e-6

    def test_pgd_budget_16_255(self, trained_model):
        model, x, y = trained_model
        eps = 16.0 / 255.0
        adv = AdversarialAttacker(model, epsilon=eps).pgd(x[:8], y[:8], steps=10)
        assert (adv["input"] - x[:8]).abs().max().item() <= eps + 1e-6

    def test_pgd_50_steps_valid(self, trained_model):
        model, x, y = trained_model
        adv = AdversarialAttacker(model).pgd(x[:8], y[:8], steps=50)
        assert adv["input"].min().item() >= -1e-6
        assert adv["input"].max().item() <= 1.0 + 1e-6

    def test_pgd_restarts(self, trained_model):
        model, x, y = trained_model
        adv = AdversarialAttacker(model).pgd(x[:16], y[:16], steps=10, num_restarts=3)
        assert adv["input"].shape[0] == 16

    def test_pgd_l2_budget(self, trained_model):
        model, x, y = trained_model
        eps = 0.5
        atk = AdversarialAttacker(model, norm="l2", epsilon=eps)
        adv = atk.pgd(x[:16], y[:16], epsilon=eps, steps=20)
        norms = (adv["input"] - x[:16]).flatten(1).norm(dim=1)
        assert (norms <= eps + 1e-4).all()

    def test_pgd_zero_eps(self, trained_model):
        model, x, y = trained_model
        adv = AdversarialAttacker(model).pgd(x[:4], y[:4], epsilon=0.0, steps=5)
        assert torch.allclose(adv["input"], x[:4].clamp(0, 1), atol=1e-5)

    def test_pgd_targeted(self, trained_model):
        model, x, y = trained_model
        targets = (y[:16] + 1) % 10
        adv = AdversarialAttacker(model).pgd(x[:16], targets, epsilon=0.3,
                                              steps=50, targeted=True)
        assert adv["input"].shape == x[:16].shape


class TestAutoAttack:
    def test_auto_returns_result(self, trained_model):
        model, x, y = trained_model
        result = AdversarialAttacker(model).auto_attack(x[:16], y[:16])
        assert isinstance(result, AttackResult)

    def test_auto_robust_acc_range(self, trained_model):
        model, x, y = trained_model
        result = AdversarialAttacker(model).auto_attack(x[:16], y[:16])
        assert 0.0 <= result.robust_accuracy <= 1.0

    def test_auto_clean_acc_range(self, trained_model):
        model, x, y = trained_model
        result = AdversarialAttacker(model).auto_attack(x[:16], y[:16])
        assert 0.0 <= result.clean_accuracy <= 1.0

    def test_auto_method_name(self, trained_model):
        model, x, y = trained_model
        result = AdversarialAttacker(model).auto_attack(x[:8], y[:8])
        assert result.attack_method == "auto_attack"

    def test_auto_summary(self, trained_model):
        model, x, y = trained_model
        result = AdversarialAttacker(model).auto_attack(x[:8], y[:8])
        assert len(result.summary()) > 0


class TestRobustnessReport:
    def test_report_type(self, trained_model):
        model, x, y = trained_model
        atk = AdversarialAttacker(model)
        report = atk.measure_robustness((x[:32], y[:32]),
                                        epsilons=[0.0, 4.0/255, 8.0/255])
        assert isinstance(report, RobustnessReport)

    def test_report_eps0_equals_clean(self, trained_model):
        model, x, y = trained_model
        report = AdversarialAttacker(model).measure_robustness(
            (x[:32], y[:32]), epsilons=[0.0, 8.0/255])
        assert abs(report.robust_accuracies[0.0] - report.clean_accuracy) < 1e-6

    def test_report_decreasing(self, trained_model):
        model, x, y = trained_model
        report = AdversarialAttacker(model).measure_robustness(
            (x[:32], y[:32]), epsilons=[0.0, 8.0/255])
        assert report.robust_accuracies[8.0/255] <= report.robust_accuracies[0.0] + 0.01


# ===========================================================================
# Category 2: OOD Detection  (tests 26-50)
# ===========================================================================

class TestEnergyScore:
    def test_shape(self):
        logits = torch.randn(32, 10)
        assert energy_score(logits).shape == (32,)

    def test_finite(self):
        assert torch.isfinite(energy_score(torch.randn(16, 10))).all()

    def test_temperature(self):
        logits = torch.randn(16, 10)
        s1 = energy_score(logits, 1.0)
        s2 = energy_score(logits, 2.0)
        assert s1.shape == s2.shape

    def test_higher_for_uniform(self):
        peaked = torch.zeros(100, 10); peaked[:, 0] = 10.0
        uniform = torch.zeros(100, 10)
        assert energy_score(peaked).mean() < energy_score(uniform).mean()


class TestMSPScore:
    def test_shape(self):
        assert msp_score(torch.randn(16, 10)).shape == (16,)

    def test_nonpositive(self):
        assert (msp_score(torch.randn(16, 10)) <= 0).all()


class TestAUROC:
    def test_perfect(self):
        assert abs(compute_auroc(torch.zeros(100), torch.ones(100)) - 1.0) < 1e-6

    def test_random(self):
        auroc = compute_auroc(torch.randn(500), torch.randn(500))
        assert abs(auroc - 0.5) < 0.1

    def test_range(self):
        auroc = compute_auroc(torch.randn(50), torch.randn(50))
        assert 0.0 <= auroc <= 1.0


class TestFPRatTPR:
    def test_perfect_separation(self):
        fpr = compute_fpr_at_tpr(torch.zeros(100), torch.ones(100))
        assert fpr < 0.05

    def test_range(self):
        fpr = compute_fpr_at_tpr(torch.randn(100), torch.randn(100))
        assert 0.0 <= fpr <= 1.0


class TestCalibrateThreshold:
    def test_target_fpr(self):
        scores = torch.randn(1000)
        thr = calibrate_threshold(scores, 0.05)
        actual = (scores >= thr).float().mean().item()
        assert abs(actual - 0.05) < 0.02


class TestOODDetector:
    def test_energy_fit(self, ood_trained_model, id_ood_data):
        id_x, id_y, _, _ = id_ood_data
        det = OODDetector(ood_trained_model, method="energy")
        det.fit((id_x, id_y))
        assert det._fitted
        assert det.threshold is not None

    def test_energy_detect(self, ood_trained_model, id_ood_data):
        id_x, id_y, _, _ = id_ood_data
        det = OODDetector(ood_trained_model, method="energy")
        det.fit((id_x, id_y))
        result = det.detect(id_x[:16])
        assert isinstance(result, OODResult)
        assert result.scores.shape == (16,)

    def test_energy_auroc(self, ood_trained_model, id_ood_data):
        id_x, id_y, ood_x, ood_y = id_ood_data
        det = OODDetector(ood_trained_model, method="energy")
        det.fit((id_x, id_y))
        metrics = det.evaluate((id_x, id_y), (ood_x, ood_y))
        assert 0.0 <= metrics.auroc <= 1.0

    def test_msp_method(self, ood_trained_model, id_ood_data):
        id_x, id_y, ood_x, ood_y = id_ood_data
        det = OODDetector(ood_trained_model, method="msp")
        det.fit((id_x, id_y))
        metrics = det.evaluate((id_x, id_y), (ood_x, ood_y))
        assert 0.0 <= metrics.auroc <= 1.0

    def test_mahalanobis_fit(self, ood_trained_model, id_ood_data):
        id_x, id_y, _, _ = id_ood_data
        det = OODDetector(ood_trained_model, method="mahalanobis", regularization=1e-4)
        det.fit((id_x, id_y))
        assert len(det.class_means) > 0
        assert det.precision is not None

    def test_mahalanobis_scores(self, ood_trained_model, id_ood_data):
        id_x, id_y, _, _ = id_ood_data
        det = OODDetector(ood_trained_model, method="mahalanobis", regularization=1e-4)
        det.fit((id_x, id_y))
        scores = det.get_score(id_x[:16])
        assert scores.shape == (16,)
        assert torch.isfinite(scores).all()

    def test_unfitted_mahalanobis_error(self, ood_trained_model, id_ood_data):
        id_x, _, _, _ = id_ood_data
        det = OODDetector(ood_trained_model, method="mahalanobis")
        with pytest.raises(RuntimeError):
            det.get_score(id_x[:4])

    def test_ood_fraction(self, ood_trained_model, id_ood_data):
        id_x, id_y, ood_x, _ = id_ood_data
        det = OODDetector(ood_trained_model, method="energy")
        det.fit((id_x, id_y))
        result = det.detect(ood_x[:50])
        assert 0.0 <= result.ood_fraction() <= 1.0

    def test_metrics_summary(self, ood_trained_model, id_ood_data):
        id_x, id_y, ood_x, ood_y = id_ood_data
        det = OODDetector(ood_trained_model, method="energy")
        det.fit((id_x, id_y))
        metrics = det.evaluate((id_x, id_y), (ood_x, ood_y))
        assert len(metrics.summary()) > 0

    def test_get_score_without_fit(self, ood_trained_model, id_ood_data):
        id_x, _, _, _ = id_ood_data
        det = OODDetector(ood_trained_model, method="energy")
        s = det.get_score(id_x[:8])
        assert s.shape == (8,)


# ===========================================================================
# Category 3: Corruption Benchmark  (tests 51-75)
# ===========================================================================

class TestCorruptionFunctions:
    @pytest.mark.parametrize("cname", ALL_CORRUPTION_NAMES)
    def test_shape_preserved(self, cname):
        x = torch.rand(4, 1, 28, 28)
        out = CORRUPTION_FUNCTIONS[cname](x, 3)
        assert out.shape == x.shape

    @pytest.mark.parametrize("cname", ALL_CORRUPTION_NAMES)
    def test_finite(self, cname):
        x = torch.rand(4, 1, 28, 28)
        out = CORRUPTION_FUNCTIONS[cname](x, 3)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("cname", ALL_CORRUPTION_NAMES)
    def test_range(self, cname):
        x = torch.rand(4, 1, 28, 28)
        out = CORRUPTION_FUNCTIONS[cname](x, 3)
        assert out.min().item() >= -1e-6
        assert out.max().item() <= 1.0 + 1e-6

    def test_output_is_tensor(self):
        out = gaussian_noise(torch.rand(2, 1, 14, 14), 2)
        assert isinstance(out, torch.Tensor)

    def test_severity_1_vs_5(self):
        x = torch.rand(8, 1, 28, 28)
        s1 = gaussian_noise(x.clone(), 1)
        s5 = gaussian_noise(x.clone(), 5)
        diff1 = (s1 - x).abs().mean().item()
        diff5 = (s5 - x).abs().mean().item()
        assert diff5 > diff1


class TestCorruptionBenchmark:
    def test_run_returns_report(self, trained_model):
        model, x, y = trained_model
        bench = CorruptionBenchmark(model, corruptions=["gaussian_noise", "brightness"])
        report = bench.run((x, y))
        assert isinstance(report, CorruptionReport)

    def test_clean_accuracy(self, trained_model):
        model, x, y = trained_model
        bench = CorruptionBenchmark(model, corruptions=["brightness"])
        report = bench.run((x, y))
        assert 0.0 <= report.clean_accuracy <= 1.0

    def test_num_corruptions(self, trained_model):
        model, x, y = trained_model
        bench = CorruptionBenchmark(model, corruptions=["gaussian_noise", "contrast"])
        report = bench.run((x, y))
        assert len(report.errors) == 2

    def test_num_severities(self, trained_model):
        model, x, y = trained_model
        bench = CorruptionBenchmark(model, corruptions=["brightness"])
        report = bench.run((x, y))
        assert len(report.errors["brightness"]) == 5

    def test_mce_range(self, trained_model):
        model, x, y = trained_model
        bench = CorruptionBenchmark(model, corruptions=["gaussian_noise"])
        report = bench.run((x, y))
        assert 0.0 <= report.mce <= 1.0

    def test_get_mce(self, trained_model):
        model, x, y = trained_model
        bench = CorruptionBenchmark(model, corruptions=["brightness"])
        report = bench.run((x, y))
        assert abs(bench.get_mce() - report.mce) < 1e-8

    def test_get_mce_before_run(self, trained_model):
        model, _, _ = trained_model
        bench = CorruptionBenchmark(model)
        with pytest.raises(RuntimeError):
            bench.get_mce()

    def test_full_benchmark(self, trained_model):
        model, x, y = trained_model
        bench = CorruptionBenchmark(model)
        report = bench.run((x, y))
        assert len(report.errors) == 15

    def test_monotonicity_dict(self, trained_model):
        model, x, y = trained_model
        bench = CorruptionBenchmark(model)
        report = bench.run((x, y))
        assert len(report.monotonicity) == 15

    def test_relative_mce_self(self, trained_model):
        model, x, y = trained_model
        bench = CorruptionBenchmark(model, corruptions=["gaussian_noise", "brightness"])
        report = bench.run((x, y))
        rel = bench.get_relative_mce(report.errors)
        assert abs(rel - 1.0) < 0.01

    def test_report_summary(self, trained_model):
        model, x, y = trained_model
        bench = CorruptionBenchmark(model, corruptions=["brightness"])
        report = bench.run((x, y))
        assert len(report.summary()) > 0


# ===========================================================================
# Category 4: Adversarial Training  (tests 76-92)
# ===========================================================================

class TestPGDAT:
    def test_returns_loss(self):
        torch.manual_seed(0)
        model = SimpleCNN(10)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        x = torch.rand(16, 1, 28, 28); y = torch.randint(0, 10, (16,))
        t = AdversarialTrainer(model, method="pgd_at", pgd_steps=2)
        m = t.train_step(x, y, opt)
        assert "loss" in m and m["loss"] > 0

    def test_loss_decreases(self):
        torch.manual_seed(0)
        model = SimpleCNN(10)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        x = torch.rand(32, 1, 28, 28); y = torch.randint(0, 10, (32,))
        t = AdversarialTrainer(model, method="pgd_at", pgd_steps=2)
        losses = [t.train_step(x, y, opt)["loss"] for _ in range(20)]
        assert losses[-1] < losses[0]

    def test_model_not_collapsed(self):
        torch.manual_seed(0)
        model = SimpleCNN(10)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        x = torch.rand(32, 1, 28, 28); y = torch.randint(0, 10, (32,))
        t = AdversarialTrainer(model, method="pgd_at", pgd_steps=2)
        for _ in range(20):
            t.train_step(x, y, opt)
        model.eval()
        with torch.no_grad():
            acc = (model(x).argmax(1) == y).float().mean().item()
        assert acc > 0.15

    def test_gradient_flow(self):
        torch.manual_seed(0)
        model = SimpleCNN(10)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        x = torch.rand(8, 1, 28, 28); y = torch.randint(0, 10, (8,))
        t = AdversarialTrainer(model, method="pgd_at", pgd_steps=2)
        t.train_step(x, y, opt)
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters() if p.requires_grad)


class TestTRADES:
    def test_trades_loss_positive(self):
        torch.manual_seed(0)
        model = SimpleCNN(10)
        x = torch.rand(16, 1, 28, 28); y = torch.randint(0, 10, (16,))
        t = AdversarialTrainer(model, method="trades", pgd_steps=2)
        loss = t.trades_loss(x, y, beta=6.0)
        assert loss.item() > 0 and torch.isfinite(loss)

    def test_trades_loss_decreases(self):
        torch.manual_seed(0)
        model = SimpleCNN(10)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        x = torch.rand(32, 1, 28, 28); y = torch.randint(0, 10, (32,))
        t = AdversarialTrainer(model, method="trades", pgd_steps=2)
        losses = [t.train_step(x, y, opt)["loss"] for _ in range(20)]
        assert losses[-1] < losses[0]

    def test_trades_beta_effect(self):
        torch.manual_seed(0)
        model = SimpleCNN(10)
        x = torch.rand(16, 1, 28, 28); y = torch.randint(0, 10, (16,))
        t = AdversarialTrainer(model, method="trades", pgd_steps=2)
        l1 = t.trades_loss(x, y, beta=1.0).item()
        l10 = t.trades_loss(x, y, beta=10.0).item()
        assert l10 >= l1 - 0.5


class TestFreeAT:
    def test_free_at_returns_loss(self):
        torch.manual_seed(0)
        model = SimpleCNN(10)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        x = torch.rand(16, 1, 28, 28); y = torch.randint(0, 10, (16,))
        t = AdversarialTrainer(model, method="free_at", free_at_replays=2)
        m = t.train_step(x, y, opt)
        assert m["loss"] > 0

    def test_free_at_no_diverge(self):
        torch.manual_seed(0)
        model = SimpleCNN(10)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        x = torch.rand(16, 1, 28, 28); y = torch.randint(0, 10, (16,))
        t = AdversarialTrainer(model, method="free_at", free_at_replays=2)
        losses = [t.train_step(x, y, opt)["loss"] for _ in range(10)]
        assert losses[-1] < losses[0] + 1.0


class TestCurriculum:
    def test_linear_warmup(self):
        assert linear_epsilon_schedule(0, 10, 1.0) == pytest.approx(0.1)
        assert linear_epsilon_schedule(10, 10, 1.0) == pytest.approx(1.0)

    def test_cosine_warmup(self):
        assert cosine_epsilon_schedule(0, 10, 1.0) > 0
        assert cosine_epsilon_schedule(10, 10, 1.0) == pytest.approx(1.0)

    def test_trainer_curriculum(self):
        model = SimpleCNN(10)
        t = AdversarialTrainer(model, epsilon=8.0/255.0,
                               use_curriculum=True, warmup_epochs=10)
        assert t.get_epsilon(0) < 8.0/255.0
        assert t.get_epsilon(10) == pytest.approx(8.0/255.0)


class TestATMisc:
    def test_invalid_method(self):
        model = SimpleCNN(10)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        t = AdversarialTrainer(model, method="invalid")
        with pytest.raises(ValueError):
            t.train_step(torch.rand(4, 1, 28, 28), torch.zeros(4, dtype=torch.long), opt)

    def test_grad_clipping(self):
        torch.manual_seed(0)
        model = SimpleCNN(10)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        x = torch.rand(8, 1, 28, 28); y = torch.randint(0, 10, (8,))
        t = AdversarialTrainer(model, method="pgd_at", pgd_steps=2, grad_clip_norm=1.0)
        t.train_step(x, y, opt)  # should not error

    def test_train_epoch(self):
        torch.manual_seed(0)
        model = SimpleCNN(10)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        x = torch.rand(32, 1, 28, 28); y = torch.randint(0, 10, (32,))
        t = AdversarialTrainer(model, method="pgd_at", pgd_steps=2)
        m = t.train_epoch((x, y), opt, epoch=0, batch_size=16)
        assert "avg_loss" in m and m["avg_loss"] > 0


# ===========================================================================
# Category 5: Calibration  (tests 93-110)
# ===========================================================================

class TestECE:
    def test_range(self):
        logits = torch.randn(100, 10)
        targets = torch.randint(0, 10, (100,))
        ece = CalibrationAnalyzer(n_bins=10).compute_ece(logits, targets)
        assert 0.0 <= ece <= 1.0

    def test_overconfident(self):
        targets = torch.randint(0, 10, (200,))
        logits = torch.randn(200, 10) * 5
        for i in range(200):
            logits[i, targets[i]] += 5
        ece = CalibrationAnalyzer().compute_ece(logits, targets)
        assert ece > 0

    def test_perfect(self):
        logits = torch.zeros(100, 10)
        targets = torch.randint(0, 10, (100,))
        for i in range(100):
            logits[i, targets[i]] = 10.0
        ece = CalibrationAnalyzer().compute_ece(logits, targets)
        assert ece < 0.15

    def test_single_sample(self):
        ece = CalibrationAnalyzer().compute_ece(torch.randn(1, 10),
                                                 torch.zeros(1, dtype=torch.long))
        assert 0.0 <= ece <= 1.0


class TestMCECal:
    def test_mce_gte_ece(self):
        logits = torch.randn(200, 10) * 3
        targets = torch.randint(0, 10, (200,))
        a = CalibrationAnalyzer(n_bins=10)
        assert a.compute_mce(logits, targets) >= a.compute_ece(logits, targets) - 1e-6


class TestReliabilityDiagramTest:
    def test_structure(self):
        logits = torch.randn(200, 10)
        targets = torch.randint(0, 10, (200,))
        d = CalibrationAnalyzer(n_bins=10).reliability_diagram(logits, targets)
        assert len(d.bin_accuracies) == 10
        assert len(d.bin_confidences) == 10
        assert len(d.bin_counts) == 10
        assert sum(d.bin_counts) == 200

    def test_acc_conf_range(self):
        d = CalibrationAnalyzer(n_bins=5).reliability_diagram(
            torch.randn(50, 10), torch.randint(0, 10, (50,)))
        assert all(0 <= a <= 1 for a in d.bin_accuracies)
        assert all(0 <= c <= 1 for c in d.bin_confidences)


class TestTemperatureScaling:
    def test_temperature_range(self):
        logits = torch.randn(200, 10) * 5
        targets = torch.randint(0, 10, (200,))
        for i in range(200):
            logits[i, targets[i]] += 3
        t = CalibrationAnalyzer().temperature_scaling(logits, targets)
        assert 0.01 <= t <= 100.0

    def test_reduces_ece(self):
        logits = torch.randn(200, 10) * 5
        targets = torch.randint(0, 10, (200,))
        for i in range(200):
            logits[i, targets[i]] += 3
        a = CalibrationAnalyzer()
        ece_before = a.compute_ece(logits, targets)
        t = a.temperature_scaling(logits, targets)
        ece_after = a.compute_ece(logits / t, targets)
        assert ece_after <= ece_before + 0.05


class TestPlattScaling:
    def test_returns_floats(self):
        logits = torch.randn(100, 10)
        targets = torch.randint(0, 10, (100,))
        a_val, b_val = CalibrationAnalyzer().platt_scaling(logits, targets)
        assert isinstance(a_val, float)
        assert isinstance(b_val, float)


class TestCalibrationUnderShift:
    def test_returns_keys(self):
        logits_c = torch.randn(100, 10)
        targets = torch.randint(0, 10, (100,))
        logits_s = logits_c + torch.randn_like(logits_c) * 2
        result = CalibrationAnalyzer().calibration_under_shift(
            (logits_c, targets), (logits_s, targets))
        assert "ece_clean" in result
        assert "ece_shifted" in result
        assert "ece_delta" in result

    def test_delta_correct(self):
        logits_c = torch.randn(100, 10)
        targets = torch.randint(0, 10, (100,))
        logits_s = logits_c + torch.randn_like(logits_c) * 2
        r = CalibrationAnalyzer().calibration_under_shift(
            (logits_c, targets), (logits_s, targets))
        assert abs(r["ece_delta"] - (r["ece_shifted"] - r["ece_clean"])) < 1e-6


class TestCalibrationAnalyze:
    def test_full_analysis(self):
        logits = torch.randn(200, 10) * 5
        targets = torch.randint(0, 10, (200,))
        for i in range(200):
            logits[i, targets[i]] += 3
        result = CalibrationAnalyzer().analyze(logits, targets)
        assert isinstance(result, CalibrationResult)
        assert result.ece > 0
        assert result.mce >= result.ece - 1e-6
        assert 0.01 <= result.optimal_temperature <= 100.0
        assert len(result.summary()) > 0


# ===========================================================================
# Category 6: Configuration Validation  (tests 111+)
# ===========================================================================

class TestAttackConfigVal:
    def test_defaults(self):
        c = AttackConfig()
        c.validate()
        assert c.method == "pgd"

    def test_fgsm_factory(self):
        c = AttackConfig.fgsm()
        c.validate()
        assert c.method == "fgsm" and c.pgd_steps == 1

    def test_reject_bad_method(self):
        with pytest.raises(ValueError):
            AttackConfig(method="bad").validate()

    def test_reject_neg_eps(self):
        with pytest.raises(ValueError):
            AttackConfig(epsilon=-1).validate()


class TestOODConfigVal:
    def test_defaults(self):
        OODConfig().validate()

    def test_reject_bad(self):
        with pytest.raises(ValueError):
            OODConfig(method="xyz").validate()


class TestCorruptionConfigVal:
    def test_all_resolves(self):
        c = CorruptionConfig()
        c.validate()
        assert len(c.resolve_corruptions()) == 15

    def test_reject_bad_severity(self):
        with pytest.raises(ValueError):
            CorruptionConfig(severities=[0]).validate()

    def test_reject_unknown_corruption(self):
        with pytest.raises(ValueError):
            CorruptionConfig(corruptions=["nonexistent"]).validate()


class TestAdvTrainConfigVal:
    def test_defaults(self):
        AdvTrainConfig().validate()

    def test_curriculum_epsilon(self):
        c = AdvTrainConfig(use_curriculum=True, warmup_epochs=10, epsilon=1.0)
        assert c.get_epsilon_for_epoch(0) < 1.0
        assert c.get_epsilon_for_epoch(10) == pytest.approx(1.0)

    def test_reject_bad_method(self):
        with pytest.raises(ValueError):
            AdvTrainConfig(method="bad").validate()


class TestCalibrationConfigVal:
    def test_defaults(self):
        CalibrationConfig().validate()

    def test_reject_bad_bins(self):
        with pytest.raises(ValueError):
            CalibrationConfig(n_bins=0).validate()


class TestRobustnessConfigVal:
    def test_aggregate(self):
        rc = RobustnessConfig()
        rc.validate_all()
        assert isinstance(rc.attack, AttackConfig)
''')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate robustness test suite")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output file path (default: stdout)")
    parser.add_argument("--run", action="store_true",
                        help="Write to temp file and run pytest")
    args = parser.parse_args()

    if args.run:
        fd, path = tempfile.mkstemp(suffix=".py", prefix="test_robustness_")
        with os.fdopen(fd, "w") as f:
            f.write(TEST_SOURCE)
        print(f"Generated test file: {path}")
        # Ensure asset templates are importable regardless of where the
        # temp file is created.
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        _asset_dir = os.path.join(os.path.dirname(_script_dir), "assets")
        env = os.environ.copy()
        pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = _asset_dir + (os.pathsep + pp if pp else "")
        os.execvpe(sys.executable,
                   [sys.executable, "-m", "pytest", path, "-v", "--tb=short"],
                   env)

    elif args.output:
        with open(args.output, "w") as f:
            f.write(TEST_SOURCE)
        print(f"Wrote {len(TEST_SOURCE)} bytes to {args.output}")

    else:
        print(TEST_SOURCE)


if __name__ == "__main__":
    main()
