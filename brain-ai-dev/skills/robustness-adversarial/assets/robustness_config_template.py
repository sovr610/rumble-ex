"""
Robustness & Adversarial Testing Configuration Templates.

All configuration dataclasses for attack generation, OOD detection,
corruption benchmarking, adversarial training, and calibration analysis.

Dependencies: torch + standard library only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Attack configuration
# ---------------------------------------------------------------------------

@dataclass
class AttackConfig:
    """Configuration for adversarial attack generation.

    Attributes:
        method: Attack algorithm -- ``"fgsm"`` | ``"pgd"`` | ``"auto"``.
        epsilon: L-inf (or L2) perturbation budget.
        pgd_steps: Number of PGD inner-loop iterations.
        pgd_step_size: PGD per-step perturbation magnitude.
        norm: Threat-model norm -- ``"linf"`` | ``"l2"``.
        targeted: Whether to run a targeted attack.
        num_restarts: Number of random restarts for PGD.
        loss_fn: Loss function name -- ``"ce"`` | ``"dlr"``.
        random_start: Whether to initialise delta uniformly in the ball.
    """

    method: str = "pgd"
    epsilon: float = 8.0 / 255.0
    pgd_steps: int = 20
    pgd_step_size: float = 2.0 / 255.0
    norm: str = "linf"
    targeted: bool = False
    num_restarts: int = 1
    loss_fn: str = "ce"
    random_start: bool = True

    # -- validation ---------------------------------------------------------

    def validate(self) -> None:
        """Raise ``ValueError`` when a field is out of range."""
        if self.method not in ("fgsm", "pgd", "auto"):
            raise ValueError(f"Unknown attack method: {self.method}")
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")
        if self.pgd_steps < 1:
            raise ValueError(f"pgd_steps must be >= 1, got {self.pgd_steps}")
        if self.pgd_step_size <= 0:
            raise ValueError(f"pgd_step_size must be positive, got {self.pgd_step_size}")
        if self.norm not in ("linf", "l2"):
            raise ValueError(f"Unknown norm: {self.norm}")
        if self.num_restarts < 1:
            raise ValueError(f"num_restarts must be >= 1, got {self.num_restarts}")
        if self.loss_fn not in ("ce", "dlr"):
            raise ValueError(f"Unknown loss_fn: {self.loss_fn}")

    # -- convenience constructors -------------------------------------------

    @classmethod
    def fgsm(cls, epsilon: float = 8.0 / 255.0, norm: str = "linf") -> "AttackConfig":
        return cls(method="fgsm", epsilon=epsilon, norm=norm, pgd_steps=1,
                   pgd_step_size=epsilon, random_start=False)

    @classmethod
    def pgd_standard(cls, epsilon: float = 8.0 / 255.0) -> "AttackConfig":
        return cls(method="pgd", epsilon=epsilon, pgd_steps=20,
                   pgd_step_size=2.0 / 255.0)

    @classmethod
    def pgd_strong(cls, epsilon: float = 8.0 / 255.0) -> "AttackConfig":
        return cls(method="pgd", epsilon=epsilon, pgd_steps=100,
                   pgd_step_size=2.0 / 255.0, num_restarts=5)

    @classmethod
    def auto(cls, epsilon: float = 8.0 / 255.0) -> "AttackConfig":
        return cls(method="auto", epsilon=epsilon, pgd_steps=100,
                   pgd_step_size=2.0 / 255.0)


# ---------------------------------------------------------------------------
# OOD configuration
# ---------------------------------------------------------------------------

@dataclass
class OODConfig:
    """Configuration for out-of-distribution detection.

    Attributes:
        method: Scoring method -- ``"energy"`` | ``"mahalanobis"``
                | ``"workspace_entropy"`` | ``"msp"``.
        temperature: Softmax / energy temperature scaling.
        threshold: Hard OOD decision boundary. ``None`` triggers auto-calibration.
        target_fpr: Desired FPR when auto-calibrating.
        feature_layer: Which layer to use for Mahalanobis scoring.
        regularization: Covariance regularisation for Mahalanobis.
    """

    method: str = "energy"
    temperature: float = 1.0
    threshold: Optional[float] = None
    target_fpr: float = 0.05
    feature_layer: str = "penultimate"
    regularization: float = 1e-5

    def validate(self) -> None:
        if self.method not in ("energy", "mahalanobis", "workspace_entropy", "msp"):
            raise ValueError(f"Unknown OOD method: {self.method}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if not 0.0 < self.target_fpr < 1.0:
            raise ValueError(f"target_fpr must be in (0, 1), got {self.target_fpr}")
        if self.regularization < 0:
            raise ValueError(f"regularization must be non-negative, got {self.regularization}")

    @classmethod
    def energy(cls, temperature: float = 1.0) -> "OODConfig":
        return cls(method="energy", temperature=temperature)

    @classmethod
    def mahalanobis(cls, regularization: float = 1e-5) -> "OODConfig":
        return cls(method="mahalanobis", regularization=regularization)

    @classmethod
    def msp(cls) -> "OODConfig":
        return cls(method="msp")


# ---------------------------------------------------------------------------
# Corruption benchmark configuration
# ---------------------------------------------------------------------------

ALL_CORRUPTIONS: List[str] = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "motion_blur", "zoom_blur", "glass_blur",
    "brightness", "contrast", "fog", "snow", "frost",
    "elastic_transform", "pixelate", "jpeg_compression",
]


@dataclass
class CorruptionConfig:
    """Configuration for the corruption benchmark.

    Attributes:
        corruptions: Corruption names to evaluate, or ``["all"]`` for every one.
        severities: Severity levels (1-5) to evaluate.
        batch_size: Evaluation batch size.
        num_workers: DataLoader worker count.
        check_monotonicity: Whether to assert monotonic error increase.
        tolerance: Tolerance for monotonicity check.
    """

    corruptions: List[str] = field(default_factory=lambda: ["all"])
    severities: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    batch_size: int = 64
    num_workers: int = 0
    check_monotonicity: bool = True
    tolerance: float = 0.02

    def validate(self) -> None:
        for s in self.severities:
            if s < 1 or s > 5:
                raise ValueError(f"Severity must be in [1, 5], got {s}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.tolerance < 0:
            raise ValueError(f"tolerance must be non-negative, got {self.tolerance}")
        resolved = self.resolve_corruptions()
        for c in resolved:
            if c not in ALL_CORRUPTIONS:
                raise ValueError(f"Unknown corruption: {c}")

    def resolve_corruptions(self) -> List[str]:
        """Return the explicit list of corruption names."""
        if self.corruptions == ["all"] or self.corruptions == ("all",):
            return list(ALL_CORRUPTIONS)
        return list(self.corruptions)


# ---------------------------------------------------------------------------
# Adversarial training configuration
# ---------------------------------------------------------------------------

@dataclass
class AdvTrainConfig:
    """Configuration for adversarial training.

    Attributes:
        method: Training method -- ``"pgd_at"`` | ``"trades"`` | ``"free_at"``.
        epsilon: Perturbation budget.
        pgd_steps: Inner-loop PGD steps.
        pgd_step_size: Inner-loop step size.
        trades_beta: TRADES regularisation weight.
        free_at_replays: Number of replays for Free-AT.
        use_curriculum: Whether to ramp epsilon during warmup.
        warmup_epochs: Number of warmup epochs for curriculum.
        epsilon_schedule: Curriculum schedule -- ``"linear"`` | ``"cosine"`` | ``"step"``.
        grad_clip_norm: Max gradient norm (0 = disabled).
        mixed_training_lambda: Weighting for mixed clean + adversarial loss.
    """

    method: str = "pgd_at"
    epsilon: float = 8.0 / 255.0
    pgd_steps: int = 7
    pgd_step_size: float = 2.0 / 255.0
    trades_beta: float = 6.0
    free_at_replays: int = 4
    use_curriculum: bool = False
    warmup_epochs: int = 10
    epsilon_schedule: str = "linear"
    grad_clip_norm: float = 0.0
    mixed_training_lambda: float = 1.0

    def validate(self) -> None:
        if self.method not in ("pgd_at", "trades", "free_at"):
            raise ValueError(f"Unknown AT method: {self.method}")
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")
        if self.pgd_steps < 1:
            raise ValueError(f"pgd_steps must be >= 1, got {self.pgd_steps}")
        if self.pgd_step_size <= 0:
            raise ValueError(f"pgd_step_size must be positive, got {self.pgd_step_size}")
        if self.trades_beta < 0:
            raise ValueError(f"trades_beta must be non-negative, got {self.trades_beta}")
        if self.free_at_replays < 1:
            raise ValueError(f"free_at_replays must be >= 1, got {self.free_at_replays}")
        if self.epsilon_schedule not in ("linear", "cosine", "step"):
            raise ValueError(f"Unknown epsilon schedule: {self.epsilon_schedule}")
        if self.warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be non-negative, got {self.warmup_epochs}")
        if not 0.0 <= self.mixed_training_lambda <= 1.0:
            raise ValueError(
                f"mixed_training_lambda must be in [0, 1], got {self.mixed_training_lambda}"
            )

    def get_epsilon_for_epoch(self, epoch: int) -> float:
        """Return the effective epsilon considering curriculum warmup."""
        if not self.use_curriculum or epoch >= self.warmup_epochs:
            return self.epsilon
        if self.warmup_epochs == 0:
            return self.epsilon
        progress = (epoch + 1) / self.warmup_epochs
        if self.epsilon_schedule == "linear":
            return self.epsilon * progress
        elif self.epsilon_schedule == "cosine":
            return self.epsilon * 0.5 * (1.0 - math.cos(math.pi * progress))
        else:  # step
            if progress < 0.33:
                return self.epsilon * 0.25
            elif progress < 0.66:
                return self.epsilon * 0.5
            else:
                return self.epsilon * 0.75
        return self.epsilon

    @classmethod
    def pgd_at(cls, epsilon: float = 8.0 / 255.0, steps: int = 7) -> "AdvTrainConfig":
        return cls(method="pgd_at", epsilon=epsilon, pgd_steps=steps,
                   pgd_step_size=epsilon / 4.0)

    @classmethod
    def trades(cls, epsilon: float = 8.0 / 255.0, beta: float = 6.0) -> "AdvTrainConfig":
        return cls(method="trades", epsilon=epsilon, trades_beta=beta,
                   pgd_steps=10, pgd_step_size=epsilon / 4.0)

    @classmethod
    def free_at(cls, epsilon: float = 8.0 / 255.0, replays: int = 4) -> "AdvTrainConfig":
        return cls(method="free_at", epsilon=epsilon, free_at_replays=replays)


# ---------------------------------------------------------------------------
# Calibration configuration
# ---------------------------------------------------------------------------

@dataclass
class CalibrationConfig:
    """Configuration for calibration analysis.

    Attributes:
        n_bins: Number of bins for ECE / reliability diagram.
        temperature_lr: Learning rate for temperature optimisation.
        temperature_max_iter: Maximum optimisation iterations.
        platt_scaling: If ``True``, learn affine (a, b) instead of T only.
    """

    n_bins: int = 15
    temperature_lr: float = 0.01
    temperature_max_iter: int = 100
    platt_scaling: bool = False

    def validate(self) -> None:
        if self.n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {self.n_bins}")
        if self.temperature_lr <= 0:
            raise ValueError(f"temperature_lr must be positive, got {self.temperature_lr}")
        if self.temperature_max_iter < 1:
            raise ValueError(f"temperature_max_iter must be >= 1, got {self.temperature_max_iter}")


# ---------------------------------------------------------------------------
# Aggregate configuration
# ---------------------------------------------------------------------------

@dataclass
class RobustnessConfig:
    """Top-level container bundling every sub-config."""

    attack: AttackConfig = field(default_factory=AttackConfig)
    ood: OODConfig = field(default_factory=OODConfig)
    corruption: CorruptionConfig = field(default_factory=CorruptionConfig)
    adv_train: AdvTrainConfig = field(default_factory=AdvTrainConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)

    def validate_all(self) -> None:
        self.attack.validate()
        self.ood.validate()
        self.corruption.validate()
        self.adv_train.validate()
        self.calibration.validate()


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
    print("robustness_config_template self-tests")
    print("=" * 60)

    # -- AttackConfig -------------------------------------------------------
    c = AttackConfig()
    c.validate()
    _assert(c.method == "pgd", "AttackConfig default method is pgd")
    _assert(abs(c.epsilon - 8.0 / 255.0) < 1e-8, "AttackConfig default epsilon")
    _assert(c.pgd_steps == 20, "AttackConfig default pgd_steps")
    _assert(c.norm == "linf", "AttackConfig default norm")

    fgsm_cfg = AttackConfig.fgsm()
    fgsm_cfg.validate()
    _assert(fgsm_cfg.method == "fgsm", "AttackConfig.fgsm() method")
    _assert(fgsm_cfg.pgd_steps == 1, "AttackConfig.fgsm() single step")

    pgd_std = AttackConfig.pgd_standard()
    pgd_std.validate()
    _assert(pgd_std.pgd_steps == 20, "AttackConfig.pgd_standard() steps")

    pgd_str = AttackConfig.pgd_strong()
    pgd_str.validate()
    _assert(pgd_str.pgd_steps == 100, "AttackConfig.pgd_strong() steps")
    _assert(pgd_str.num_restarts == 5, "AttackConfig.pgd_strong() restarts")

    auto_cfg = AttackConfig.auto()
    auto_cfg.validate()
    _assert(auto_cfg.method == "auto", "AttackConfig.auto() method")

    # -- invalid AttackConfig -----------------------------------------------
    bad = AttackConfig(method="unknown")
    try:
        bad.validate()
        _assert(False, "AttackConfig rejects unknown method")
    except ValueError:
        _assert(True, "AttackConfig rejects unknown method")

    bad2 = AttackConfig(epsilon=-1)
    try:
        bad2.validate()
        _assert(False, "AttackConfig rejects negative epsilon")
    except ValueError:
        _assert(True, "AttackConfig rejects negative epsilon")

    bad3 = AttackConfig(norm="l1")
    try:
        bad3.validate()
        _assert(False, "AttackConfig rejects unsupported norm")
    except ValueError:
        _assert(True, "AttackConfig rejects unsupported norm")

    # -- OODConfig ----------------------------------------------------------
    oc = OODConfig()
    oc.validate()
    _assert(oc.method == "energy", "OODConfig default method")
    _assert(oc.temperature == 1.0, "OODConfig default temperature")

    oc_e = OODConfig.energy(temperature=2.0)
    oc_e.validate()
    _assert(oc_e.temperature == 2.0, "OODConfig.energy() custom temperature")

    oc_m = OODConfig.mahalanobis()
    oc_m.validate()
    _assert(oc_m.method == "mahalanobis", "OODConfig.mahalanobis() method")

    bad_oc = OODConfig(method="xyz")
    try:
        bad_oc.validate()
        _assert(False, "OODConfig rejects unknown method")
    except ValueError:
        _assert(True, "OODConfig rejects unknown method")

    # -- CorruptionConfig ---------------------------------------------------
    cc = CorruptionConfig()
    cc.validate()
    resolved = cc.resolve_corruptions()
    _assert(len(resolved) == 15, "CorruptionConfig resolves all 15 corruptions")
    _assert("gaussian_noise" in resolved, "CorruptionConfig includes gaussian_noise")

    cc2 = CorruptionConfig(corruptions=["gaussian_noise", "brightness"])
    cc2.validate()
    _assert(len(cc2.resolve_corruptions()) == 2, "CorruptionConfig custom corruption list")

    bad_cc = CorruptionConfig(severities=[0, 6])
    try:
        bad_cc.validate()
        _assert(False, "CorruptionConfig rejects bad severity")
    except ValueError:
        _assert(True, "CorruptionConfig rejects bad severity")

    bad_cc2 = CorruptionConfig(corruptions=["nonexistent"])
    try:
        bad_cc2.validate()
        _assert(False, "CorruptionConfig rejects unknown corruption")
    except ValueError:
        _assert(True, "CorruptionConfig rejects unknown corruption")

    # -- AdvTrainConfig -----------------------------------------------------
    ac = AdvTrainConfig()
    ac.validate()
    _assert(ac.method == "pgd_at", "AdvTrainConfig default method")
    _assert(ac.trades_beta == 6.0, "AdvTrainConfig default trades_beta")

    ac_pgd = AdvTrainConfig.pgd_at()
    ac_pgd.validate()
    _assert(ac_pgd.method == "pgd_at", "AdvTrainConfig.pgd_at() method")

    ac_tr = AdvTrainConfig.trades(beta=10.0)
    ac_tr.validate()
    _assert(ac_tr.trades_beta == 10.0, "AdvTrainConfig.trades() beta")

    ac_fr = AdvTrainConfig.free_at(replays=8)
    ac_fr.validate()
    _assert(ac_fr.free_at_replays == 8, "AdvTrainConfig.free_at() replays")

    # curriculum schedule
    ac_cur = AdvTrainConfig(use_curriculum=True, warmup_epochs=10,
                            epsilon_schedule="linear", epsilon=8.0 / 255.0)
    e0 = ac_cur.get_epsilon_for_epoch(0)
    e9 = ac_cur.get_epsilon_for_epoch(9)
    e10 = ac_cur.get_epsilon_for_epoch(10)
    _assert(e0 < ac_cur.epsilon, "Curriculum epsilon < target at epoch 0")
    _assert(abs(e10 - ac_cur.epsilon) < 1e-8, "Curriculum epsilon = target at warmup end")
    _assert(e0 < e9, "Curriculum epsilon increases over epochs")

    ac_cos = AdvTrainConfig(use_curriculum=True, warmup_epochs=10,
                            epsilon_schedule="cosine", epsilon=8.0 / 255.0)
    _assert(ac_cos.get_epsilon_for_epoch(0) < ac_cos.epsilon,
            "Cosine curriculum epsilon < target at epoch 0")
    _assert(abs(ac_cos.get_epsilon_for_epoch(10) - ac_cos.epsilon) < 1e-8,
            "Cosine curriculum epsilon = target at warmup end")

    bad_ac = AdvTrainConfig(method="unknown")
    try:
        bad_ac.validate()
        _assert(False, "AdvTrainConfig rejects unknown method")
    except ValueError:
        _assert(True, "AdvTrainConfig rejects unknown method")

    # -- CalibrationConfig --------------------------------------------------
    cal = CalibrationConfig()
    cal.validate()
    _assert(cal.n_bins == 15, "CalibrationConfig default n_bins")

    bad_cal = CalibrationConfig(n_bins=0)
    try:
        bad_cal.validate()
        _assert(False, "CalibrationConfig rejects n_bins=0")
    except ValueError:
        _assert(True, "CalibrationConfig rejects n_bins=0")

    # -- RobustnessConfig ---------------------------------------------------
    rc = RobustnessConfig()
    rc.validate_all()
    _assert(isinstance(rc.attack, AttackConfig), "RobustnessConfig has AttackConfig")
    _assert(isinstance(rc.ood, OODConfig), "RobustnessConfig has OODConfig")
    _assert(isinstance(rc.corruption, CorruptionConfig), "RobustnessConfig has CorruptionConfig")
    _assert(isinstance(rc.adv_train, AdvTrainConfig), "RobustnessConfig has AdvTrainConfig")
    _assert(isinstance(rc.calibration, CalibrationConfig), "RobustnessConfig has CalibrationConfig")

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    print("=" * 60)
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    _run_self_tests()
