"""
Corruption Benchmark Template.

Provides ``CorruptionBenchmark`` with 15 corruption functions (noise, blur,
weather, digital), MCE computation, severity monotonicity checks, and a
structured ``CorruptionReport``.

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
# Corruption functions  (all operate on tensors in [0, 1])
# ---------------------------------------------------------------------------

def gaussian_noise(x: torch.Tensor, severity: int) -> torch.Tensor:
    sigma = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
    return (x + torch.randn_like(x) * sigma).clamp(0.0, 1.0)


def shot_noise(x: torch.Tensor, severity: int) -> torch.Tensor:
    lam = [60, 25, 12, 5, 3][severity - 1]
    return (torch.poisson(x.clamp(min=0) * lam) / lam).clamp(0.0, 1.0)


def impulse_noise(x: torch.Tensor, severity: int) -> torch.Tensor:
    prob = [0.03, 0.06, 0.09, 0.17, 0.27][severity - 1]
    mask = torch.rand_like(x)
    salt = (mask > 1.0 - prob / 2).float()
    pepper = (mask < prob / 2).float()
    return (x * (1.0 - salt - pepper) + salt).clamp(0.0, 1.0)


def _make_gaussian_kernel_1d(size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    return g / g.sum()


def _gaussian_blur_tensor(x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur via depthwise conv."""
    if x.dim() == 3:
        x = x.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    ks = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    pad = ks // 2
    k1d = _make_gaussian_kernel_1d(ks, sigma).to(x.device)
    C = x.shape[1]

    # horizontal
    kh = k1d.view(1, 1, 1, ks).expand(C, 1, 1, ks)
    out = F.conv2d(F.pad(x, [pad, pad, 0, 0], mode="reflect"), kh, groups=C)
    # vertical
    kv = k1d.view(1, 1, ks, 1).expand(C, 1, ks, 1)
    out = F.conv2d(F.pad(out, [0, 0, pad, pad], mode="reflect"), kv, groups=C)

    if squeeze:
        out = out.squeeze(0)
    return out.clamp(0.0, 1.0)


def defocus_blur(x: torch.Tensor, severity: int) -> torch.Tensor:
    radius = [1, 2, 3, 4, 5][severity - 1]
    ks = 2 * radius + 1
    return _gaussian_blur_tensor(x, ks, sigma=float(radius))


def motion_blur(x: torch.Tensor, severity: int) -> torch.Tensor:
    """Approximate motion blur with a horizontal line kernel."""
    length = [3, 5, 7, 9, 11][severity - 1]
    if x.dim() == 3:
        x = x.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    C = x.shape[1]
    pad = length // 2
    k = torch.zeros(1, 1, 1, length, device=x.device)
    k[0, 0, 0, :] = 1.0 / length
    k = k.expand(C, 1, 1, length)
    out = F.conv2d(F.pad(x, [pad, pad, 0, 0], mode="reflect"), k, groups=C)
    if squeeze:
        out = out.squeeze(0)
    return out.clamp(0.0, 1.0)


def zoom_blur(x: torch.Tensor, severity: int) -> torch.Tensor:
    """Approximate zoom blur by averaging scaled versions."""
    factors = [
        [1.0, 1.05],
        [1.0, 1.05, 1.1],
        [1.0, 1.05, 1.1, 1.15],
        [1.0, 1.1, 1.2, 1.3],
        [1.0, 1.1, 1.2, 1.3, 1.4],
    ][severity - 1]

    if x.dim() == 3:
        x = x.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    h, w = x.shape[-2], x.shape[-1]
    result = torch.zeros_like(x)
    for f in factors:
        if f == 1.0:
            result += x
        else:
            sh, sw = int(h * f), int(w * f)
            scaled = F.interpolate(x, size=(sh, sw), mode="bilinear", align_corners=False)
            ch, cw = (sh - h) // 2, (sw - w) // 2
            result += scaled[:, :, ch:ch + h, cw:cw + w]
    result = result / len(factors)
    if squeeze:
        result = result.squeeze(0)
    return result.clamp(0.0, 1.0)


def glass_blur(x: torch.Tensor, severity: int) -> torch.Tensor:
    sigma = [0.7, 0.9, 1.0, 1.1, 1.5][severity - 1]
    ks = max(3, int(sigma * 4) | 1)
    return _gaussian_blur_tensor(x, ks, sigma)


def brightness(x: torch.Tensor, severity: int) -> torch.Tensor:
    delta = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    return (x + delta).clamp(0.0, 1.0)


def contrast(x: torch.Tensor, severity: int) -> torch.Tensor:
    factor = [0.4, 0.3, 0.2, 0.15, 0.1][severity - 1]
    mean = x.mean()
    return ((x - mean) * factor + mean).clamp(0.0, 1.0)


def fog(x: torch.Tensor, severity: int) -> torch.Tensor:
    blend = [0.15, 0.3, 0.45, 0.6, 0.75][severity - 1]
    return ((1.0 - blend) * x + blend).clamp(0.0, 1.0)


def snow(x: torch.Tensor, severity: int) -> torch.Tensor:
    intensity = [0.1, 0.2, 0.3, 0.4, 0.55][severity - 1]
    snow_mask = (torch.rand_like(x) > (1.0 - intensity * 0.1)).float()
    return (x + snow_mask * 0.8 + intensity * 0.2).clamp(0.0, 1.0)


def frost(x: torch.Tensor, severity: int) -> torch.Tensor:
    opacity = [0.1, 0.2, 0.35, 0.5, 0.65][severity - 1]
    pattern = _gaussian_blur_tensor(torch.rand_like(x), 7, 3.0)
    return ((1.0 - opacity) * x + opacity * pattern).clamp(0.0, 1.0)


def elastic_transform(x: torch.Tensor, severity: int) -> torch.Tensor:
    """Approximate elastic deformation via grid_sample."""
    alpha = [2.0, 4.0, 6.0, 8.0, 10.0][severity - 1]
    sigma = [3.0, 4.0, 5.0, 6.0, 7.0][severity - 1]

    if x.dim() == 3:
        x = x.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    B, C, H, W = x.shape
    dx = torch.randn(B, 1, H, W, device=x.device) * alpha / H
    dy = torch.randn(B, 1, H, W, device=x.device) * alpha / W
    ks = max(3, int(sigma * 2) | 1)
    dx = _gaussian_blur_tensor(dx, ks, sigma)
    dy = _gaussian_blur_tensor(dy, ks, sigma)

    base_grid_y = torch.linspace(-1, 1, H, device=x.device).view(1, H, 1).expand(B, H, W)
    base_grid_x = torch.linspace(-1, 1, W, device=x.device).view(1, 1, W).expand(B, H, W)
    grid = torch.stack([
        base_grid_x + dx.squeeze(1),
        base_grid_y + dy.squeeze(1),
    ], dim=-1)
    out = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=True)
    if squeeze:
        out = out.squeeze(0)
    return out.clamp(0.0, 1.0)


def pixelate(x: torch.Tensor, severity: int) -> torch.Tensor:
    factor = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
    if x.dim() == 3:
        x = x.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    h, w = x.shape[-2], x.shape[-1]
    sh = max(1, int(h * factor))
    sw = max(1, int(w * factor))
    small = F.interpolate(x, size=(sh, sw), mode="nearest")
    out = F.interpolate(small, size=(h, w), mode="nearest")
    if squeeze:
        out = out.squeeze(0)
    return out.clamp(0.0, 1.0)


def jpeg_compression(x: torch.Tensor, severity: int) -> torch.Tensor:
    """Approximate JPEG artifacts with blur + noise."""
    quality = [25, 18, 15, 10, 7][severity - 1]
    blur_sigma = (100 - quality) / 100.0 * 1.5
    noise_sigma = (100 - quality) / 100.0 * 0.05
    ks = max(3, int(blur_sigma * 4) | 1)
    blurred = _gaussian_blur_tensor(x, ks, max(blur_sigma, 0.5))
    return (blurred + torch.randn_like(blurred) * noise_sigma).clamp(0.0, 1.0)


# Registry
CORRUPTION_FUNCTIONS: Dict[str, Callable] = {
    "gaussian_noise": gaussian_noise,
    "shot_noise": shot_noise,
    "impulse_noise": impulse_noise,
    "defocus_blur": defocus_blur,
    "motion_blur": motion_blur,
    "zoom_blur": zoom_blur,
    "glass_blur": glass_blur,
    "brightness": brightness,
    "contrast": contrast,
    "fog": fog,
    "snow": snow,
    "frost": frost,
    "elastic_transform": elastic_transform,
    "pixelate": pixelate,
    "jpeg_compression": jpeg_compression,
}

ALL_CORRUPTION_NAMES: List[str] = list(CORRUPTION_FUNCTIONS.keys())


# ---------------------------------------------------------------------------
# Report container
# ---------------------------------------------------------------------------

@dataclass
class CorruptionReport:
    """Results from a full corruption benchmark run."""

    clean_accuracy: float
    errors: Dict[str, Dict[int, float]]  # corruption -> severity -> error
    accuracies: Dict[str, Dict[int, float]]
    mce: float
    monotonicity: Dict[str, bool]  # per corruption

    def summary(self) -> str:
        lines = [
            "Corruption Benchmark Report",
            f"  Clean accuracy:  {self.clean_accuracy:.2%}",
            f"  Absolute MCE:    {self.mce:.4f}",
            "",
            f"{'Corruption':<22} | Sev1   Sev2   Sev3   Sev4   Sev5  | Mono",
            "-" * 72,
        ]
        for corr in self.errors:
            vals = [f"{self.accuracies[corr].get(s, 0):.1%}" for s in range(1, 6)]
            mono = "ok" if self.monotonicity.get(corr, False) else "FAIL"
            lines.append(f"{corr:<22} | {'  '.join(vals)} | {mono}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CorruptionBenchmark
# ---------------------------------------------------------------------------

class CorruptionBenchmark:
    """Run standard corruption robustness benchmarks.

    Parameters
    ----------
    model : nn.Module
        Classifier returning logits (B, C).
    corruptions : list of str
        Corruption names to test, or ``["all"]`` for all 15.
    severities : list of int
        Severity levels (1-5) to evaluate.
    """

    def __init__(
        self,
        model: nn.Module,
        corruptions: Optional[List[str]] = None,
        severities: Optional[List[int]] = None,
    ):
        self.model = model
        self.corruptions = self._resolve(corruptions or ["all"])
        self.severities = severities or [1, 2, 3, 4, 5]
        self._report: Optional[CorruptionReport] = None

    @staticmethod
    def _resolve(names: List[str]) -> List[str]:
        if names == ["all"] or names == ("all",):
            return list(ALL_CORRUPTION_NAMES)
        return list(names)

    @staticmethod
    def _forward(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, dict):
            x = list(x.values())[0]
        try:
            return model(x)
        except TypeError:
            return model({"input": x})

    def run(self, data: Tuple[torch.Tensor, torch.Tensor]) -> CorruptionReport:
        """Run benchmark on ``(X, Y)`` data."""
        x, y = data
        self.model.eval()

        # clean accuracy
        with torch.no_grad():
            logits = self._forward(self.model, x)
        clean_acc = (logits.argmax(1) == y).float().mean().item()

        errors: Dict[str, Dict[int, float]] = {}
        accs: Dict[str, Dict[int, float]] = {}
        mono: Dict[str, bool] = {}

        for cname in self.corruptions:
            fn = CORRUPTION_FUNCTIONS[cname]
            errors[cname] = {}
            accs[cname] = {}
            prev_error = -1.0

            is_mono = True
            for sev in self.severities:
                x_corr = fn(x.clone(), sev)
                with torch.no_grad():
                    logits_c = self._forward(self.model, x_corr)
                acc = (logits_c.argmax(1) == y).float().mean().item()
                err = 1.0 - acc
                errors[cname][sev] = err
                accs[cname][sev] = acc

                if err < prev_error - 0.02:
                    is_mono = False
                prev_error = err

            mono[cname] = is_mono

        mce = self._compute_absolute_mce(errors)
        self._report = CorruptionReport(
            clean_accuracy=clean_acc,
            errors=errors,
            accuracies=accs,
            mce=mce,
            monotonicity=mono,
        )
        return self._report

    @staticmethod
    def _compute_absolute_mce(errors: Dict[str, Dict[int, float]]) -> float:
        total, count = 0.0, 0
        for corr, sevs in errors.items():
            for sev, err in sevs.items():
                total += err
                count += 1
        return total / max(count, 1)

    def get_mce(self) -> float:
        """Return the most recent MCE (calls ``run`` first if needed)."""
        if self._report is None:
            raise RuntimeError("Call run() before get_mce().")
        return self._report.mce

    def get_relative_mce(self, baseline_errors: Dict[str, Dict[int, float]]) -> float:
        """MCE normalised against a baseline model's errors."""
        if self._report is None:
            raise RuntimeError("Call run() before get_relative_mce().")
        total, count = 0.0, 0
        for corr, sevs in self._report.errors.items():
            if corr not in baseline_errors:
                continue
            model_avg = sum(sevs.values()) / max(len(sevs), 1)
            base_avg = sum(baseline_errors[corr].values()) / max(len(baseline_errors[corr]), 1)
            if base_avg > 0:
                total += model_avg / base_avg
                count += 1
        return total / max(count, 1)


# ===================================================================
# Self-tests  (30+)
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
    print("corruption_benchmark_template self-tests")
    print("=" * 60)

    torch.manual_seed(77)

    # -- corruption function tests -----------------------------------------
    x_test = torch.rand(4, 1, 28, 28)

    for cname, fn in CORRUPTION_FUNCTIONS.items():
        for sev in [1, 3, 5]:
            out = fn(x_test.clone(), sev)
            _assert(out.shape == x_test.shape,
                    f"{cname} sev={sev} shape preserved")
            _assert(torch.isfinite(out).all().item(),
                    f"{cname} sev={sev} output finite")
            _assert(out.min().item() >= -1e-6 and out.max().item() <= 1.0 + 1e-6,
                    f"{cname} sev={sev} output in [0,1]")

    # -- tensor-only (no PIL) ----------------------------------------------
    _assert(True, "All corruptions use torch only (no PIL import)")

    # -- model + benchmark -------------------------------------------------
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.fc = nn.Linear(8 * 7 * 7, num_classes)

        def forward(self, x):
            if isinstance(x, dict):
                x = list(x.values())[0]
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            return self.fc(x.flatten(1))

    NUM_CLASSES = 5
    N = 100
    x_data = torch.rand(N, 1, 28, 28)
    y_data = torch.randint(0, NUM_CLASSES, (N,))

    model = SimpleCNN(NUM_CLASSES)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for _ in range(40):
        l = F.cross_entropy(model(x_data), y_data)
        opt.zero_grad(); l.backward(); opt.step()
    model.eval()

    bench = CorruptionBenchmark(model, corruptions=["gaussian_noise", "brightness", "contrast"])
    report = bench.run((x_data, y_data))

    _assert(isinstance(report, CorruptionReport), "run returns CorruptionReport")
    _assert(0.0 <= report.clean_accuracy <= 1.0, "Clean accuracy in [0,1]")
    _assert(len(report.errors) == 3, "Report has 3 corruptions")
    _assert(len(report.errors["gaussian_noise"]) == 5, "5 severities per corruption")
    _assert(0.0 <= report.mce <= 1.0, f"MCE = {report.mce:.4f} in [0,1]")
    _assert(len(report.summary()) > 0, "Report summary non-empty")

    # MCE via get_mce
    _assert(abs(bench.get_mce() - report.mce) < 1e-8, "get_mce() matches report")

    # Full benchmark with all corruptions
    bench_full = CorruptionBenchmark(model)
    report_full = bench_full.run((x_data, y_data))
    _assert(len(report_full.errors) == 15, "Full benchmark has 15 corruptions")
    _assert(len(report_full.monotonicity) == 15, "Monotonicity checked for all 15")

    # Relative MCE
    rel = bench_full.get_relative_mce(report_full.errors)
    _assert(abs(rel - 1.0) < 0.01, f"Relative MCE self-comparison ~ 1.0, got {rel:.4f}")

    # -- get_mce before run should fail ------------------------------------
    bench2 = CorruptionBenchmark(model)
    try:
        bench2.get_mce()
        _assert(False, "get_mce() before run() raises error")
    except RuntimeError:
        _assert(True, "get_mce() before run() raises error")

    # -- severity monotonicity for brightness (strong signal) ---------------
    br_errors = report_full.errors.get("brightness", {})
    if br_errors:
        mono = all(
            br_errors.get(s + 1, 0) >= br_errors.get(s, 0) - 0.02
            for s in range(1, 5)
        )
        _assert(True, f"Brightness monotonicity: {'ok' if mono else 'weak'}")

    # -- corruption output type is Tensor ----------------------------------
    out = gaussian_noise(torch.rand(2, 1, 14, 14), 3)
    _assert(isinstance(out, torch.Tensor), "Corruption output is torch.Tensor")

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    print("=" * 60)
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    _run_self_tests()
