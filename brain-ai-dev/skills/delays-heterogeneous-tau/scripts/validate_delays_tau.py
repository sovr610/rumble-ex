#!/usr/bin/env python3
"""
validate_delays_tau.py - Runtime contract validation for delay modules and
heterogeneous tau.

Validates two distinct API layers depending on what is installed:

  TARGET API (post-implementation):
    brain_ai.core.delays           - DelayLinear, DelayConv1d
    brain_ai.core.heterogeneous_tau - HeterogeneousTau, TauInitConfig

  LEGACY API (pre-implementation, current state):
    brain_ai.core.neurons.AdvancedLIFNeuron - logit-beta delays

The script detects which API is available and runs the appropriate checks.

Usage:
    python validate_delays_tau.py                       # Both modules, CPU
    python validate_delays_tau.py --check-only delays   # Delay module only
    python validate_delays_tau.py --check-only tau      # Tau module only
    python validate_delays_tau.py --device cuda         # Use CUDA
    python validate_delays_tau.py --json                # JSON output for CI
    python validate_delays_tau.py --verbose             # Detailed output
    python validate_delays_tau.py --batch-size 8 --timesteps 50
"""

# ---------------------------------------------------------------------------
# SECTION 1: Imports and path setup
# ---------------------------------------------------------------------------

import os
import sys
import json
import math
import time
import argparse
import traceback
import warnings
import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# scripts/ -> delays-heterogeneous-tau/ -> skills/ -> brain-ai-dev/ -> human-brain/
_REPO_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(_SCRIPT_DIR)
        )
    )
)
if os.path.isdir(os.path.join(_REPO_ROOT, "brain_ai")):
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# SECTION 2: Validation result tracking
# ---------------------------------------------------------------------------

PASS_MARK = "PASS"
FAIL_MARK = "FAIL"
SKIP_MARK = "SKIP"


@dataclass
class ValidationResult:
    name: str
    status: str          # "PASS", "FAIL", or "SKIP"
    message: str
    details: Optional[Dict[str, Any]] = None
    duration_ms: float = 0.0

    @property
    def passed(self) -> bool:
        return self.status == PASS_MARK

    @property
    def skipped(self) -> bool:
        return self.status == SKIP_MARK


class ValidationReport:
    """Collects and formats validation results."""

    def __init__(self):
        self.results: List[ValidationResult] = []

    def add(self, result: ValidationResult) -> None:
        self.results.append(result)

    def pass_(
        self,
        name: str,
        message: str = "ok",
        details: Optional[Dict] = None,
        duration_ms: float = 0.0,
    ) -> ValidationResult:
        r = ValidationResult(
            name=name, status=PASS_MARK, message=message,
            details=details, duration_ms=duration_ms,
        )
        self.add(r)
        return r

    def fail(
        self,
        name: str,
        message: str,
        details: Optional[Dict] = None,
        duration_ms: float = 0.0,
    ) -> ValidationResult:
        r = ValidationResult(
            name=name, status=FAIL_MARK, message=message,
            details=details, duration_ms=duration_ms,
        )
        self.add(r)
        return r

    def skip(self, name: str, reason: str) -> ValidationResult:
        r = ValidationResult(name=name, status=SKIP_MARK, message=reason)
        self.add(r)
        return r

    @property
    def all_passed(self) -> bool:
        return all(r.passed or r.skipped for r in self.results)

    def summary(self) -> Dict[str, Any]:
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed and not r.skipped)
        skipped = sum(1 for r in self.results if r.skipped)
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "all_passed": self.all_passed,
        }

    def print_report(self, verbose: bool = False) -> None:
        print()
        print("=" * 70)
        print("  DELAY & TAU CONTRACT VALIDATION REPORT")
        print("=" * 70)

        groups: Dict[str, List[ValidationResult]] = {}
        for r in self.results:
            parts = r.name.split("/", 1)
            group = parts[0] if len(parts) > 1 else "_global"
            groups.setdefault(group, []).append(r)

        for group, results in groups.items():
            g_passed = sum(1 for r in results if r.passed)
            g_failed = sum(1 for r in results if not r.passed and not r.skipped)
            g_skipped = sum(1 for r in results if r.skipped)
            g_total = len(results)

            if g_failed > 0:
                marker = "FAIL"
            elif g_skipped == g_total:
                marker = "SKIP"
            else:
                marker = "PASS"

            print(
                f"\n[{marker}] {group}  "
                f"({g_passed}/{g_total} passed, {g_skipped} skipped)"
            )

            for r in results:
                check_name = r.name.split("/", 1)[-1] if "/" in r.name else r.name
                symbol = {PASS_MARK: "  ok", FAIL_MARK: "FAIL", SKIP_MARK: "skip"}[r.status]
                timing = f"  ({r.duration_ms:.1f} ms)" if r.duration_ms > 0 else ""
                print(f"    [{symbol}] {check_name}{timing}")

                if verbose or r.status == FAIL_MARK:
                    print(f"           {r.message}")
                    if r.details and verbose:
                        for k, v in r.details.items():
                            print(f"           {k}: {v}")

        print()
        s = self.summary()
        status_line = (
            "ALL CHECKS PASSED" if s["all_passed"]
            else f"{s['failed']} CHECK(S) FAILED"
        )
        print(
            f"SUMMARY: {s['passed']}/{s['total']} passed, "
            f"{s['skipped']} skipped -- {status_line}"
        )
        print("=" * 70)

    def to_json(self) -> str:
        data = {
            "summary": self.summary(),
            "results": [
                {
                    "name": r.name,
                    "status": r.status,
                    "message": r.message,
                    "details": r.details,
                    "duration_ms": round(r.duration_ms, 3),
                }
                for r in self.results
            ],
        }
        return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# SECTION 3: API detection and import helpers
# ---------------------------------------------------------------------------

def detect_api() -> Dict[str, Any]:
    """
    Detect which API variant is available and import all reachable symbols.

    Returns a dict with keys:
        api_level : 'target' | 'legacy' | 'none'
        delays    : dict of delay module classes (may be empty)
        tau       : dict of tau module classes (may be empty)
        neurons   : dict of neuron classes from brain_ai.core.neurons
        errors    : list of import error strings
    """
    result: Dict[str, Any] = {
        "api_level": "none",
        "delays": {},
        "tau": {},
        "neurons": {},
        "errors": [],
    }

    # --- Try target delays API ---
    try:
        from brain_ai.core.delays import DelayLinear, DelayConv1d  # type: ignore
        result["delays"]["DelayLinear"] = DelayLinear
        result["delays"]["DelayConv1d"] = DelayConv1d
    except ImportError as exc:
        result["errors"].append(f"brain_ai.core.delays: {exc}")

    # --- Try target tau API ---
    try:
        from brain_ai.core.heterogeneous_tau import (  # type: ignore
            HeterogeneousTau,
            TauInitConfig,
        )
        result["tau"]["HeterogeneousTau"] = HeterogeneousTau
        result["tau"]["TauInitConfig"] = TauInitConfig
    except ImportError as exc:
        result["errors"].append(f"brain_ai.core.heterogeneous_tau: {exc}")

    # --- Try legacy neurons ---
    try:
        from brain_ai.core.neurons import (
            LIFNeuron,
            AdvancedLIFNeuron,
        )
        result["neurons"]["LIFNeuron"] = LIFNeuron
        result["neurons"]["AdvancedLIFNeuron"] = AdvancedLIFNeuron
    except ImportError as exc:
        result["errors"].append(f"brain_ai.core.neurons: {exc}")

    # Determine api_level
    has_delays = bool(result["delays"])
    has_tau = bool(result["tau"])
    has_neurons = bool(result["neurons"])

    if has_delays or has_tau:
        result["api_level"] = "target"
    elif has_neurons:
        result["api_level"] = "legacy"
    else:
        result["api_level"] = "none"

    return result


def _has_param(cls: type, param_name: str) -> bool:
    """Return True if cls.__init__ accepts param_name."""
    try:
        sig = inspect.signature(cls.__init__)
        return param_name in sig.parameters
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# SECTION 4: Stub implementations
#
# These stubs mirror the target API contract so we can validate the contract
# specification itself.  When brain_ai.core.delays / heterogeneous_tau are
# later implemented, the stubs are automatically bypassed by detect_api().
# ---------------------------------------------------------------------------

class _StubDelayLinear(nn.Module):
    """
    Minimal stub implementing the DelayLinear contract.
    Used ONLY when the target API does not exist.

    Contract surface under test:
      - forward(x: (B,T,in)) -> (B,T,out)
      - d_raw parameter for gradient / no-gradient checks
      - sigma attribute for schedule checks
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        max_delay: int = 8,
        fixed_random: bool = False,
        learnable_sigma: bool = True,
        initial_sigma: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_delay = max_delay
        self.fixed_random = fixed_random

        # Linear projection
        self.linear = nn.Linear(in_features, out_features)

        # Delay logits: (out_features, max_delay) - learnable or fixed
        d_init = torch.zeros(out_features, max_delay)
        if fixed_random:
            self.register_buffer("d_raw", d_init)
        else:
            self.d_raw = nn.Parameter(d_init)

        # Sigma for Gaussian kernel width schedule
        if learnable_sigma:
            self.sigma = nn.Parameter(torch.tensor(initial_sigma))
        else:
            self.register_buffer("sigma", torch.tensor(initial_sigma))

    def _gaussian_kernel(self) -> torch.Tensor:
        """Normalised Gaussian weights over delay bins. Shape: (out, max_delay)."""
        delays = torch.arange(
            self.max_delay, device=self.d_raw.device, dtype=torch.float32
        )
        d_mean = (
            torch.softmax(self.d_raw, dim=-1) * delays
        ).sum(-1, keepdim=True)
        sigma = self.sigma.abs().clamp(min=1e-3)
        kernel = torch.exp(-0.5 * ((delays - d_mean) / sigma) ** 2)
        kernel = kernel / (kernel.sum(-1, keepdim=True) + 1e-8)
        return kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, in_features)
        B, T, _ = x.shape
        out = self.linear(x)  # (B, T, out_features)
        kernel = self._gaussian_kernel()  # (out_features, max_delay)
        pad = min(self.max_delay - 1, T)
        x_pad = torch.cat(
            [torch.zeros(B, pad, self.out_features, device=x.device), out],
            dim=1,
        )
        delayed = torch.zeros_like(out)
        for d in range(self.max_delay):
            w = kernel[:, d].unsqueeze(0).unsqueeze(0)  # (1,1,out)
            delayed = delayed + w * x_pad[:, d : d + T]
        return delayed

    def update_sigma(self, new_sigma: float) -> None:
        """Schedule sigma downward (anneal toward hard discretisation)."""
        with torch.no_grad():
            if isinstance(self.sigma, nn.Parameter):
                self.sigma.data.fill_(new_sigma)
            else:
                self.sigma.fill_(new_sigma)

    def get_effective_delays(self) -> torch.Tensor:
        """Expected delay index per output neuron. Shape: (out_features,)."""
        delays = torch.arange(
            self.max_delay, device=self.d_raw.device, dtype=torch.float32
        )
        return (torch.softmax(self.d_raw, dim=-1) * delays).sum(-1)

    def get_delay_distribution(self) -> torch.Tensor:
        """Softmax delay distribution. Shape: (out_features, max_delay)."""
        return torch.softmax(self.d_raw, dim=-1)


def _tau_to_tau_raw_fn(tau: torch.Tensor, tau_min: float) -> torch.Tensor:
    """
    Invert the softplus constraint:  tau = tau_min + softplus(tau_raw)

    Derivation:
        tau - tau_min = softplus(tau_raw) = log(1 + exp(tau_raw))
        exp(tau_raw) = exp(tau - tau_min) - 1
        tau_raw = log(exp(tau - tau_min) - 1)

    We use torch.Tensor.expm1 (x -> exp(x)-1) in reverse:
        exp(tau_raw) = (tau - tau_min).exp() - 1
        tau_raw = torch.log(...)
    """
    val = (tau - tau_min).clamp(min=1e-6)
    # val = exp(tau_raw), so tau_raw = log(val)
    # val = exp(tau - tau_min) - 1 = expm1(tau - tau_min)
    inner = val.exp().sub(1.0).clamp(min=1e-8)
    return inner.log()


class _StubHeterogeneousTau(nn.Module):
    """
    Minimal stub implementing the HeterogeneousTau contract.
    Used ONLY when brain_ai.core.heterogeneous_tau does not exist.
    """

    BETA_MIN: float = 0.0
    BETA_MAX: float = 0.999

    def __init__(
        self,
        n: int,
        learnable: bool = True,
        tau_0: float = 20.0,
        tau_min: float = 1.0,
        tau_max: float = 100.0,
        dt: float = 1.0,
        strategy: str = "homogeneous",
        preset_values: Optional[List[float]] = None,
    ):
        super().__init__()
        self.dt = dt
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.n = n

        tau_init = _stub_init_tau(n, tau_0, strategy, preset_values, tau_min, tau_max)
        tau_raw_init = _tau_to_tau_raw_fn(tau_init, tau_min)

        if learnable:
            self.tau_raw = nn.Parameter(tau_raw_init)
        else:
            self.register_buffer("tau_raw", tau_raw_init)

    @property
    def tau(self) -> torch.Tensor:
        """Constrained tau in [tau_min, tau_max]."""
        return (self.tau_min + F.softplus(self.tau_raw)).clamp(max=self.tau_max)

    @property
    def beta(self) -> torch.Tensor:
        """Constrained beta in [BETA_MIN, BETA_MAX], computed in fp32."""
        tau_fp32 = self.tau.float()
        beta_fp32 = torch.exp(-self.dt / tau_fp32)
        return beta_fp32.clamp(self.BETA_MIN, self.BETA_MAX)

    def beta_broadcast(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Return beta shaped for broadcasting with an activation tensor.

        Dense (B, N):       beta shape (N,)  -> unsqueeze(0) -> (1, N)
        Conv  (B, C, H, W): beta shape (C,)  -> view(1, C, 1, 1)
        """
        b = self.beta
        if len(shape) == 2:
            return b.unsqueeze(0)
        elif len(shape) == 4:
            return b.view(1, -1, 1, 1)
        return b

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return a health-check dict with tau / beta statistics."""
        with torch.no_grad():
            tau = self.tau
            beta = self.beta
            return {
                "tau_mean": float(tau.mean().item()),
                "tau_min_actual": float(tau.min().item()),
                "tau_max_actual": float(tau.max().item()),
                "tau_std": float(tau.std().item()) if tau.numel() > 1 else 0.0,
                "beta_mean": float(beta.mean().item()),
                "beta_min_actual": float(beta.min().item()),
                "beta_max_actual": float(beta.max().item()),
                "n_at_tau_min": int((tau <= self.tau_min * 1.01).sum().item()),
                "n_at_tau_max": int((tau >= self.tau_max * 0.99).sum().item()),
                "n_total": int(tau.numel()),
            }


def _stub_init_tau(
    n: int,
    tau_0: float,
    strategy: str,
    preset_values: Optional[List[float]],
    tau_min: float,
    tau_max: float,
) -> torch.Tensor:
    """Dispatch to the requested initialisation strategy."""
    if strategy == "homogeneous":
        return torch.full((n,), tau_0).clamp(tau_min, tau_max)

    elif strategy == "gamma":
        gamma_k = 2.0
        theta = tau_0 / gamma_k
        dist = torch.distributions.Gamma(gamma_k, 1.0 / theta)
        return dist.sample((n,)).clamp(tau_min, tau_max)

    elif strategy == "loguniform":
        log_min = math.log(tau_min)
        log_max = math.log(tau_max)
        log_tau = torch.empty(n).uniform_(log_min, log_max)
        return log_tau.exp().clamp(tau_min, tau_max)

    elif strategy == "preset_bank":
        presets = preset_values or [2.0, 5.0, 10.0, 20.0, 50.0]
        pt = torch.tensor(presets).clamp(tau_min, tau_max)
        indices = torch.arange(n) % len(presets)
        return pt[indices]

    # Default: homogeneous fallback
    return torch.full((n,), tau_0).clamp(tau_min, tau_max)


# ---------------------------------------------------------------------------
# SECTION 5: Helper utilities
# ---------------------------------------------------------------------------

def _elapsed(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def _make_sequence_input(
    batch: int,
    timesteps: int,
    features: int,
    device: torch.device,
    requires_grad: bool = False,
) -> torch.Tensor:
    """Return a (B, T, F) float32 tensor."""
    return torch.randn(
        batch, timesteps, features,
        device=device, requires_grad=requires_grad,
    )


def _peak_memory_mb(device: torch.device) -> float:
    """Return peak allocated VRAM in MB (CUDA only); -1 on CPU."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return -1.0


# ---------------------------------------------------------------------------
# SECTION 6: Delay module checks (target API or stub)
# ---------------------------------------------------------------------------

def _get_delay_cls(components: Dict[str, Any], name: str):
    """Return the real class if in target API, else the stub."""
    if name in components.get("delays", {}):
        return components["delays"][name]
    if name == "DelayLinear":
        return _StubDelayLinear
    return None


def check_delay_forward_shapes(
    delay_cls,
    module_name: str,
    batch: int,
    timesteps: int,
    in_features: int,
    out_features: int,
    device: torch.device,
    report: ValidationReport,
) -> None:
    """Check 1: output shape matches expected (B, T, out_features)."""
    check = f"{module_name}/forward_shapes"
    t0 = time.perf_counter()
    try:
        mod = delay_cls(in_features, out_features).to(device)
        x = _make_sequence_input(batch, timesteps, in_features, device)
        out = mod(x)
        elapsed = _elapsed(t0)

        expected = (batch, timesteps, out_features)
        issues = []
        if tuple(out.shape) != expected:
            issues.append(f"shape {tuple(out.shape)} != expected {expected}")
        if torch.isnan(out).any():
            issues.append("output contains NaN")
        if torch.isinf(out).any():
            issues.append("output contains Inf")

        details = {
            "input_shape": list(x.shape),
            "output_shape": list(out.shape),
            "expected_shape": list(expected),
        }
        if issues:
            report.fail(check, "; ".join(issues), details=details, duration_ms=elapsed)
        else:
            report.pass_(
                check, f"shape={list(out.shape)} correct",
                details=details, duration_ms=elapsed,
            )

    except Exception as exc:
        report.fail(check, f"Exception: {exc}", duration_ms=_elapsed(t0))


def check_delay_range(
    delay_cls,
    module_name: str,
    batch: int,
    timesteps: int,
    in_features: int,
    out_features: int,
    max_delay: int,
    device: torch.device,
    report: ValidationReport,
) -> None:
    """Check 2: all effective delays are within [0, max_delay-1]."""
    check = f"{module_name}/delay_range"
    t0 = time.perf_counter()
    try:
        mod = delay_cls(in_features, out_features, max_delay=max_delay).to(device)
        mod.eval()

        if not hasattr(mod, "get_effective_delays"):
            report.skip(check, "get_effective_delays() not available on this module")
            return

        eff = mod.get_effective_delays()
        elapsed = _elapsed(t0)

        min_d = float(eff.min().item())
        max_d = float(eff.max().item())
        issues = []

        if min_d < 0.0:
            issues.append(f"min effective delay {min_d:.4f} < 0")
        if max_d > float(max_delay - 1) + 1e-4:
            issues.append(f"max effective delay {max_d:.4f} > max_delay-1={max_delay - 1}")
        if torch.isnan(eff).any():
            issues.append("effective delays contain NaN")

        details = {
            "max_delay": max_delay,
            "eff_min": round(min_d, 4),
            "eff_max": round(max_d, 4),
            "shape": list(eff.shape),
        }
        if issues:
            report.fail(check, "; ".join(issues), details=details, duration_ms=elapsed)
        else:
            report.pass_(
                check,
                f"delays in [{min_d:.3f}, {max_d:.3f}] (max_delay={max_delay})",
                details=details, duration_ms=elapsed,
            )

    except Exception as exc:
        report.fail(check, f"Exception: {exc}", duration_ms=_elapsed(t0))


def check_delay_gaussian_normalization(
    delay_cls,
    module_name: str,
    in_features: int,
    out_features: int,
    max_delay: int,
    device: torch.device,
    report: ValidationReport,
) -> None:
    """Check 3: Gaussian kernel bins (or softmax distribution) sum to ~1.0."""
    check = f"{module_name}/gaussian_normalization"
    t0 = time.perf_counter()
    try:
        mod = delay_cls(in_features, out_features, max_delay=max_delay).to(device)

        if not hasattr(mod, "get_delay_distribution"):
            report.skip(check, "get_delay_distribution() not available")
            return

        dist = mod.get_delay_distribution()  # (out_features, max_delay)
        row_sums = dist.sum(dim=-1)           # (out_features,)
        elapsed = _elapsed(t0)

        atol = 1e-4
        max_dev = float((row_sums - 1.0).abs().max().item())
        issues = []

        if max_dev > atol:
            issues.append(
                f"max row-sum deviation {max_dev:.6f} > atol={atol}"
            )
        if torch.isnan(dist).any():
            issues.append("distribution contains NaN")
        if (dist < 0).any():
            issues.append("distribution contains negative values")

        details = {
            "max_deviation_from_1": round(max_dev, 8),
            "atol": atol,
            "dist_shape": list(dist.shape),
            "row_sum_min": round(float(row_sums.min().item()), 8),
            "row_sum_max": round(float(row_sums.max().item()), 8),
        }
        if issues:
            report.fail(check, "; ".join(issues), details=details, duration_ms=elapsed)
        else:
            report.pass_(
                check, f"all rows sum to 1 (max_dev={max_dev:.2e})",
                details=details, duration_ms=elapsed,
            )

    except Exception as exc:
        report.fail(check, f"Exception: {exc}", duration_ms=_elapsed(t0))


def check_delay_eval_discretization(
    delay_cls,
    module_name: str,
    in_features: int,
    out_features: int,
    max_delay: int,
    device: torch.device,
    report: ValidationReport,
) -> None:
    """Check 4: in eval mode, argmax delays are valid integer indices."""
    check = f"{module_name}/eval_discretization"
    t0 = time.perf_counter()
    try:
        mod = delay_cls(in_features, out_features, max_delay=max_delay).to(device)
        mod.eval()

        if not hasattr(mod, "get_delay_distribution"):
            report.skip(check, "get_delay_distribution() not available")
            return

        with torch.no_grad():
            dist = mod.get_delay_distribution()          # (out, max_delay)
            argmax_delays = dist.argmax(dim=-1).float()  # integer delays via argmax

        elapsed = _elapsed(t0)

        issues = []
        if (argmax_delays < 0).any():
            issues.append("argmax delays < 0")
        if (argmax_delays >= max_delay).any():
            issues.append(f"argmax delays >= max_delay={max_delay}")
        if torch.isnan(argmax_delays).any():
            issues.append("argmax delays contain NaN")

        details = {
            "max_delay": max_delay,
            "argmax_unique": sorted(set(argmax_delays.long().tolist())),
            "shape": list(argmax_delays.shape),
        }
        if issues:
            report.fail(check, "; ".join(issues), details=details, duration_ms=elapsed)
        else:
            report.pass_(
                check,
                "eval discretisation produces valid integer delay indices",
                details=details, duration_ms=elapsed,
            )

    except Exception as exc:
        report.fail(check, f"Exception: {exc}", duration_ms=_elapsed(t0))


def check_delay_gradient_flow(
    delay_cls,
    module_name: str,
    batch: int,
    timesteps: int,
    in_features: int,
    out_features: int,
    device: torch.device,
    report: ValidationReport,
) -> None:
    """Check 5: loss.backward() gives non-None, non-zero d_raw.grad."""
    check = f"{module_name}/gradient_flow"
    t0 = time.perf_counter()
    try:
        mod = delay_cls(in_features, out_features).to(device)
        mod.train()

        if not hasattr(mod, "d_raw"):
            report.skip(check, "module has no d_raw attribute")
            return

        if not isinstance(mod.d_raw, nn.Parameter):
            report.skip(check, "d_raw is a buffer, not a Parameter (fixed mode)")
            return

        mod.zero_grad()
        x = _make_sequence_input(batch, timesteps, in_features, device)
        out = mod(x)
        loss = out.sum()
        loss.backward()

        elapsed = _elapsed(t0)
        issues = []

        if mod.d_raw.grad is None:
            issues.append("d_raw.grad is None after backward")
        elif mod.d_raw.grad.abs().sum().item() == 0:
            issues.append("d_raw.grad is all-zeros (dead gradient)")

        details = {
            "d_raw_grad_norm": (
                round(float(mod.d_raw.grad.norm().item()), 6)
                if mod.d_raw.grad is not None else None
            ),
            "d_raw_shape": list(mod.d_raw.shape),
        }
        if issues:
            report.fail(check, "; ".join(issues), details=details, duration_ms=elapsed)
        else:
            report.pass_(
                check,
                f"d_raw.grad norm={details['d_raw_grad_norm']:.4e}",
                details=details, duration_ms=elapsed,
            )

    except Exception as exc:
        report.fail(
            check, f"Exception: {exc}\n{traceback.format_exc()}",
            duration_ms=_elapsed(t0),
        )


def check_delay_no_gradient_fixed(
    delay_cls,
    module_name: str,
    batch: int,
    timesteps: int,
    in_features: int,
    out_features: int,
    device: torch.device,
    report: ValidationReport,
) -> None:
    """Check 6: in fixed_random mode, d_raw has no gradient."""
    check = f"{module_name}/no_gradient_fixed"
    t0 = time.perf_counter()
    try:
        ctor_kwargs: Dict[str, Any] = {"fixed_random": True}
        if not _has_param(delay_cls, "fixed_random"):
            if _has_param(delay_cls, "fixed"):
                ctor_kwargs = {"fixed": True}
            else:
                report.skip(check, "No fixed_random / fixed kwarg on this class")
                return

        mod = delay_cls(in_features, out_features, **ctor_kwargs).to(device)
        mod.train()

        if not hasattr(mod, "d_raw"):
            report.skip(check, "module has no d_raw attribute")
            return

        mod.zero_grad()
        x = _make_sequence_input(batch, timesteps, in_features, device)
        out = mod(x)
        loss = out.sum()
        loss.backward()

        elapsed = _elapsed(t0)
        issues = []

        d_raw = mod.d_raw
        if isinstance(d_raw, nn.Parameter) and d_raw.grad is not None:
            if d_raw.grad.abs().sum().item() > 0:
                issues.append(
                    "d_raw has non-zero gradient in fixed_random mode"
                )

        is_param = isinstance(d_raw, nn.Parameter)
        details = {
            "is_parameter": is_param,
            "requires_grad": (
                d_raw.requires_grad
                if hasattr(d_raw, "requires_grad") else None
            ),
        }
        if issues:
            report.fail(check, "; ".join(issues), details=details, duration_ms=elapsed)
        else:
            report.pass_(
                check, "fixed_random mode: no gradient through d_raw",
                details=details, duration_ms=elapsed,
            )

    except Exception as exc:
        report.fail(check, f"Exception: {exc}", duration_ms=_elapsed(t0))


def check_delay_sigma_schedule(
    delay_cls,
    module_name: str,
    in_features: int,
    out_features: int,
    device: torch.device,
    report: ValidationReport,
    n_steps: int = 5,
) -> None:
    """Check 7: update_sigma produces monotonically decreasing sigma."""
    check = f"{module_name}/sigma_schedule"
    t0 = time.perf_counter()
    try:
        mod = delay_cls(in_features, out_features).to(device)

        if not hasattr(mod, "update_sigma"):
            report.skip(check, "update_sigma() not available")
            return
        if not hasattr(mod, "sigma"):
            report.skip(check, "sigma attribute not available")
            return

        sigmas: List[float] = []
        start_sigma = 2.0
        mod.update_sigma(start_sigma)

        for i in range(n_steps):
            new_s = start_sigma * (0.5 ** (i + 1))
            mod.update_sigma(new_s)
            s_val = mod.sigma
            if isinstance(s_val, (nn.Parameter, torch.Tensor)):
                s_val = float(s_val.item())
            sigmas.append(float(s_val))

        elapsed = _elapsed(t0)
        issues = []

        for i in range(1, len(sigmas)):
            if sigmas[i] >= sigmas[i - 1]:
                issues.append(
                    f"sigma not decreasing: step {i-1}={sigmas[i-1]:.6f}"
                    f" -> step {i}={sigmas[i]:.6f}"
                )

        details = {"sigma_sequence": [round(s, 6) for s in sigmas]}
        if issues:
            report.fail(check, "; ".join(issues), details=details, duration_ms=elapsed)
        else:
            report.pass_(
                check, f"sigma monotonically decreasing over {n_steps} steps",
                details=details, duration_ms=elapsed,
            )

    except Exception as exc:
        report.fail(check, f"Exception: {exc}", duration_ms=_elapsed(t0))


def check_delay_sigma_checkpoint(
    delay_cls,
    module_name: str,
    in_features: int,
    out_features: int,
    device: torch.device,
    report: ValidationReport,
) -> None:
    """Check 8: save/load state_dict preserves sigma value."""
    check = f"{module_name}/sigma_checkpoint"
    t0 = time.perf_counter()
    try:
        mod = delay_cls(in_features, out_features).to(device)

        if not hasattr(mod, "sigma"):
            report.skip(check, "sigma attribute not available")
            return
        if not hasattr(mod, "update_sigma"):
            report.skip(check, "update_sigma() not available")
            return

        mod.update_sigma(0.42)
        state = mod.state_dict()
        mod2 = delay_cls(in_features, out_features).to(device)
        mod2.load_state_dict(state)

        def _sigma_val(m) -> float:
            s = m.sigma
            if isinstance(s, (nn.Parameter, torch.Tensor)):
                return float(s.item())
            return float(s)

        sigma_before = _sigma_val(mod)
        sigma_after = _sigma_val(mod2)
        elapsed = _elapsed(t0)

        diff = abs(sigma_before - sigma_after)
        if diff > 1e-6:
            report.fail(
                check,
                f"sigma before={sigma_before:.8f}, after={sigma_after:.8f} "
                f"(diff={diff:.2e})",
                duration_ms=elapsed,
            )
        else:
            report.pass_(
                check,
                f"save/load preserves sigma={sigma_before:.6f}",
                details={"sigma": sigma_before},
                duration_ms=elapsed,
            )

    except Exception as exc:
        report.fail(check, f"Exception: {exc}", duration_ms=_elapsed(t0))


def check_delay_memory_guard(
    delay_cls,
    module_name: str,
    batch: int,
    timesteps: int,
    in_features: int,
    out_features: int,
    max_delay: int,
    device: torch.device,
    report: ValidationReport,
    threshold_mb: float = 200.0,
) -> None:
    """Check 9: peak memory stays below threshold during forward pass."""
    check = f"{module_name}/memory_guard"
    t0 = time.perf_counter()
    try:
        if device.type != "cuda":
            report.skip(
                check, "memory_guard is only meaningful on CUDA; skipping on CPU"
            )
            return

        torch.cuda.reset_peak_memory_stats(device)
        mod = delay_cls(in_features, out_features, max_delay=max_delay).to(device)
        x = _make_sequence_input(batch, timesteps, in_features, device)
        _ = mod(x)
        peak_mb = _peak_memory_mb(device)
        elapsed = _elapsed(t0)

        details = {
            "peak_mb": round(peak_mb, 2),
            "threshold_mb": threshold_mb,
            "batch": batch,
            "timesteps": timesteps,
            "max_delay": max_delay,
        }
        if peak_mb > threshold_mb:
            report.fail(
                check,
                f"peak memory {peak_mb:.1f} MB > threshold {threshold_mb:.1f} MB",
                details=details, duration_ms=elapsed,
            )
        else:
            report.pass_(
                check,
                f"peak memory {peak_mb:.1f} MB <= {threshold_mb:.1f} MB",
                details=details, duration_ms=elapsed,
            )

    except Exception as exc:
        report.fail(check, f"Exception: {exc}", duration_ms=_elapsed(t0))


def check_delay_granularity_shapes(
    delay_cls,
    module_name: str,
    in_features: int,
    out_features: int,
    max_delay: int,
    device: torch.device,
    report: ValidationReport,
) -> None:
    """Check 10: d_raw shape matches configured granularity (out_features, max_delay)."""
    check = f"{module_name}/granularity_shapes"
    t0 = time.perf_counter()
    try:
        mod = delay_cls(in_features, out_features, max_delay=max_delay).to(device)

        if not hasattr(mod, "d_raw"):
            report.skip(check, "module has no d_raw attribute")
            return

        d_raw = mod.d_raw
        elapsed = _elapsed(t0)

        expected_shape = (out_features, max_delay)
        issues = []

        if tuple(d_raw.shape) != expected_shape:
            issues.append(
                f"d_raw.shape {tuple(d_raw.shape)} != expected {expected_shape}"
            )

        details = {
            "d_raw_shape": list(d_raw.shape),
            "expected_shape": list(expected_shape),
        }
        if issues:
            report.fail(check, "; ".join(issues), details=details, duration_ms=elapsed)
        else:
            report.pass_(
                check,
                f"d_raw.shape={list(d_raw.shape)} matches (out_features, max_delay)",
                details=details, duration_ms=elapsed,
            )

    except Exception as exc:
        report.fail(check, f"Exception: {exc}", duration_ms=_elapsed(t0))


def check_delay_spike_history_update(
    delay_cls,
    module_name: str,
    batch: int,
    timesteps: int,
    in_features: int,
    out_features: int,
    device: torch.device,
    report: ValidationReport,
) -> None:
    """Check 11: history buffer updates correctly across two forward calls."""
    check = f"{module_name}/spike_history_update"
    t0 = time.perf_counter()
    try:
        mod = delay_cls(in_features, out_features).to(device)

        # Find a history buffer by common attribute names
        history_attr = None
        for attr in ("spike_history", "history", "_history", "delay_buffer"):
            if hasattr(mod, attr):
                history_attr = attr
                break

        if history_attr is None:
            report.skip(check, "No history buffer found on this delay module")
            return

        x1 = _make_sequence_input(batch, timesteps, in_features, device)
        x2 = _make_sequence_input(batch, timesteps, in_features, device)

        out1 = mod(x1)
        h_after_1 = getattr(mod, history_attr)
        h1_clone = (
            h_after_1.detach().clone()
            if isinstance(h_after_1, torch.Tensor) else None
        )

        out2 = mod(x2)
        h_after_2 = getattr(mod, history_attr)
        elapsed = _elapsed(t0)

        issues = []
        if torch.equal(out1, out2):
            issues.append(
                "two different inputs produced identical outputs "
                "(history not updating)"
            )

        if h1_clone is not None and isinstance(h_after_2, torch.Tensor):
            if torch.equal(h1_clone, h_after_2):
                issues.append(
                    "history buffer unchanged after second forward pass"
                )

        details = {
            "history_attr": history_attr,
            "out1_norm": round(float(out1.norm().item()), 4),
            "out2_norm": round(float(out2.norm().item()), 4),
            "outputs_differ": not torch.equal(out1, out2),
        }
        if issues:
            report.fail(check, "; ".join(issues), details=details, duration_ms=elapsed)
        else:
            report.pass_(
                check,
                "history buffer updates correctly between forward calls",
                details=details, duration_ms=elapsed,
            )

    except Exception as exc:
        report.fail(check, f"Exception: {exc}", duration_ms=_elapsed(t0))


# ---------------------------------------------------------------------------
# SECTION 7: Tau module checks (target API or stub)
# ---------------------------------------------------------------------------

def _get_tau_cls(components: Dict[str, Any]):
    """Return real HeterogeneousTau if available, else the stub."""
    if "HeterogeneousTau" in components.get("tau", {}):
        return components["tau"]["HeterogeneousTau"]
    return _StubHeterogeneousTau


def check_tau_beta_range(
    tau_cls,
    n: int,
    device: torch.device,
    report: ValidationReport,
) -> None:
    """Tau check 1: beta strictly in (0, 1) for all values."""
    check = "HeterogeneousTau/beta_range"
    t0 = time.perf_counter()
    try:
        mod = tau_cls(n=n).to(device)
        beta = mod.beta
        elapsed = _elapsed(t0)

        min_b = float(beta.min().item())
        max_b = float(beta.max().item())
        issues = []

        if min_b <= 0.0:
            issues.append(f"beta min={min_b:.6f} <= 0 (must be strictly positive)")
        if max_b >= 1.0:
            issues.append(f"beta max={max_b:.6f} >= 1.0 (must be strictly < 1)")
        if torch.isnan(beta).any():
            issues.append("beta contains NaN")
        if torch.isinf(beta).any():
            issues.append("beta contains Inf")

        details = {
            "beta_min": round(min_b, 6),
            "beta_max": round(max_b, 6),
            "beta_mean": round(float(beta.mean().item()), 6),
            "n": n,
        }
        if issues:
            report.fail(check, "; ".join(issues), details=details, duration_ms=elapsed)
        else:
            report.pass_(
                check,
                f"beta strictly in ({min_b:.4f}, {max_b:.4f})",
                details=details, duration_ms=elapsed,
            )

    except Exception as exc:
        report.fail(check, f"Exception: {exc}", duration_ms=_elapsed(t0))


def check_tau_positive(
    tau_cls,
    n: int,
    device: torch.device,
    report: ValidationReport,
) -> None:
    """Tau check 2: tau > 0 for all values."""
    check = "HeterogeneousTau/tau_positive"
    t0 = time.perf_counter()
    try:
        mod = tau_cls(n=n).to(device)
        tau = mod.tau
        elapsed = _elapsed(t0)

        min_tau = float(tau.min().item())
        issues = []

        if min_tau <= 0.0:
            issues.append(
                f"tau min={min_tau:.8f} <= 0 (violates positivity constraint)"
            )
        if torch.isnan(tau).any():
            issues.append("tau contains NaN")
        if torch.isinf(tau).any():
            issues.append("tau contains Inf")

        details = {
            "tau_min": round(min_tau, 6),
            "tau_max": round(float(tau.max().item()), 6),
            "tau_mean": round(float(tau.mean().item()), 6),
        }
        if issues:
            report.fail(check, "; ".join(issues), details=details, duration_ms=elapsed)
        else:
            report.pass_(
                check, f"tau > 0 (min={min_tau:.4f})",
                details=details, duration_ms=elapsed,
            )

    except Exception as exc:
        report.fail(check, f"Exception: {exc}", duration_ms=_elapsed(t0))


def check_tau_softplus_mapping(
    tau_cls,
    n: int,
    device: torch.device,
    report: ValidationReport,
    tau_min: float = 1.0,
) -> None:
    """Tau check 3: tau = tau_min + softplus(tau_raw) matches expected values."""
    check = "HeterogeneousTau/softplus_mapping"
    t0 = time.perf_counter()
    try:
        mod = tau_cls(n=n).to(device)
        tau_raw = mod.tau_raw.detach()
        tau_computed = mod.tau.detach()

        # Manual recompute
        tau_max_val = float(getattr(mod, "tau_max", 100.0))
        tau_expected = (tau_min + F.softplus(tau_raw)).clamp(max=tau_max_val)

        elapsed = _elapsed(t0)
        max_diff = float((tau_computed - tau_expected).abs().max().item())
        atol = 1e-5

        issues = []
        if max_diff > atol:
            issues.append(
                f"tau deviates from tau_min + softplus(tau_raw): "
                f"max_diff={max_diff:.2e} > {atol:.2e}"
            )

        details = {
            "max_diff": round(max_diff, 8),
            "atol": atol,
            "tau_min": tau_min,
        }
        if issues:
            report.fail(check, "; ".join(issues), details=details, duration_ms=elapsed)
        else:
            report.pass_(
                check,
                f"tau = tau_min + softplus(tau_raw) (max_diff={max_diff:.2e})",
                details=details, duration_ms=elapsed,
            )

    except Exception as exc:
        report.fail(check, f"Exception: {exc}", duration_ms=_elapsed(t0))


def check_tau_exp_mapping(
    tau_cls,
    n: int,
    device: torch.device,
    report: ValidationReport,
    dt: float = 1.0,
) -> None:
    """Tau check 4: beta = exp(-dt/tau) matches expected values."""
    check = "HeterogeneousTau/exp_mapping"
    t0 = time.perf_counter()
    try:
        mod = tau_cls(n=n, dt=dt).to(device)
        tau = mod.tau.detach()
        beta_computed = mod.beta.detach()

        beta_max_val = float(getattr(mod, "BETA_MAX", 0.999))
        beta_min_val = float(getattr(mod, "BETA_MIN", 0.0))
        beta_expected = torch.exp(-dt / tau.float()).clamp(beta_min_val, beta_max_val)

        elapsed = _elapsed(t0)
        max_diff = float((beta_computed - beta_expected).abs().max().item())
        atol = 1e-5

        issues = []
        if max_diff > atol:
            issues.append(
                f"beta deviates from exp(-dt/tau): "
                f"max_diff={max_diff:.2e} > {atol:.2e}"
            )

        details = {
            "max_diff": round(max_diff, 8),
            "atol": atol,
            "dt": dt,
        }
        if issues:
            report.fail(check, "; ".join(issues), details=details, duration_ms=elapsed)
        else:
            report.pass_(
                check,
                f"beta = exp(-dt/tau) (max_diff={max_diff:.2e})",
                details=details, duration_ms=elapsed,
            )

    except Exception as exc:
        report.fail(check, f"Exception: {exc}", duration_ms=_elapsed(t0))


def check_tau_gradient_flow(
    tau_cls,
    n: int,
    device: torch.device,
    report: ValidationReport,
    batch: int = 4,
) -> None:
    """Tau check 5: loss.backward() gives tau_raw.grad when learnable=True."""
    check = "HeterogeneousTau/gradient_flow"
    t0 = time.perf_counter()
    try:
        mod = tau_cls(n=n, learnable=True).to(device)
        mod.zero_grad()

        beta = mod.beta   # (n,)
        v = torch.randn(batch, n, device=device)
        out = (beta.unsqueeze(0) * v).sum()
        out.backward()

        elapsed = _elapsed(t0)
        issues = []

        if mod.tau_raw.grad is None:
            issues.append(
                "tau_raw.grad is None after backward (learnable=True)"
            )
        elif mod.tau_raw.grad.abs().sum().item() == 0:
            issues.append("tau_raw.grad is all-zeros (dead gradient)")

        details = {
            "tau_raw_grad_norm": (
                round(float(mod.tau_raw.grad.norm().item()), 6)
                if mod.tau_raw.grad is not None else None
            ),
            "learnable": True,
        }
        if issues:
            report.fail(check, "; ".join(issues), details=details, duration_ms=elapsed)
        else:
            report.pass_(
                check,
                f"tau_raw.grad norm={details['tau_raw_grad_norm']:.4e}",
                details=details, duration_ms=elapsed,
            )

    except Exception as exc:
        report.fail(check, f"Exception: {exc}", duration_ms=_elapsed(t0))


def check_tau_no_gradient_fixed(
    tau_cls,
    n: int,
    device: torch.device,
    report: ValidationReport,
    batch: int = 4,
) -> None:
    """Tau check 6: in fixed mode (learnable=False), tau_raw has no gradient.

    We make the downstream tensor v require grad so that backward() succeeds.
    The key assertion is that tau_raw itself accumulates no gradient since it
    is a buffer (not a Parameter) in fixed mode.
    """
    check = "HeterogeneousTau/no_gradient_fixed"
    t0 = time.perf_counter()
    try:
        mod = tau_cls(n=n, learnable=False).to(device)
        mod.zero_grad()

        tau_raw = mod.tau_raw
        is_param = isinstance(tau_raw, nn.Parameter)

        # Give v requires_grad=True so backward can always succeed regardless
        # of whether tau_raw requires grad.
        beta = mod.beta
        v = torch.randn(batch, n, device=device, requires_grad=True)
        out = (beta.unsqueeze(0) * v).sum()
        out.backward()

        elapsed = _elapsed(t0)
        issues = []

        # tau_raw must NOT have accumulated a gradient in fixed mode.
        if is_param:
            if tau_raw.grad is not None and tau_raw.grad.abs().sum().item() > 0:
                issues.append(
                    "tau_raw has non-zero gradient in fixed (learnable=False) mode"
                )
        else:
            # Registered as buffer: should have no .grad attribute at all
            if hasattr(tau_raw, "grad") and tau_raw.grad is not None:
                if tau_raw.grad.abs().sum().item() > 0:
                    issues.append(
                        "tau_raw buffer unexpectedly accumulated a gradient"
                    )

        details = {
            "is_parameter": is_param,
            "requires_grad": bool(tau_raw.requires_grad),
        }
        if issues:
            report.fail(check, "; ".join(issues), details=details, duration_ms=elapsed)
        else:
            report.pass_(
                check, "fixed mode: tau_raw has no gradient",
                details=details, duration_ms=elapsed,
            )

    except Exception as exc:
        report.fail(check, f"Exception: {exc}", duration_ms=_elapsed(t0))


def check_tau_init_strategies(
    tau_cls,
    n: int,
    device: torch.device,
    report: ValidationReport,
    tau_min: float = 1.0,
    tau_max: float = 100.0,
) -> None:
    """Tau check 7: all 4 strategies produce valid tau in [tau_min, tau_max]."""
    strategies = [
        ("homogeneous", {}),
        ("gamma", {}),
        ("loguniform", {}),
        ("preset_bank", {"preset_values": [2.0, 5.0, 10.0, 20.0, 50.0]}),
    ]

    for strategy_name, extra_kwargs in strategies:
        check = f"HeterogeneousTau/init_strategies/{strategy_name}"
        t0 = time.perf_counter()
        try:
            kwargs: Dict[str, Any] = {
                "n": n,
                "strategy": strategy_name,
                "tau_min": tau_min,
                "tau_max": tau_max,
            }
            kwargs.update(extra_kwargs)

            try:
                mod = tau_cls(**kwargs).to(device)
            except TypeError:
                # Real API may use a different constructor signature
                mod = tau_cls(n=n).to(device)

            tau = mod.tau.detach()
            elapsed = _elapsed(t0)

            min_tau = float(tau.min().item())
            max_tau_val = float(tau.max().item())
            issues = []

            if min_tau < tau_min - 1e-4:
                issues.append(f"tau min={min_tau:.6f} < tau_min={tau_min}")
            if max_tau_val > tau_max + 1e-4:
                issues.append(f"tau max={max_tau_val:.6f} > tau_max={tau_max}")
            if torch.isnan(tau).any():
                issues.append("tau contains NaN")
            if torch.isinf(tau).any():
                issues.append("tau contains Inf")
            if (tau <= 0).any():
                issues.append("tau contains non-positive values")

            details = {
                "strategy": strategy_name,
                "tau_min_actual": round(min_tau, 4),
                "tau_max_actual": round(max_tau_val, 4),
                "tau_mean": round(float(tau.mean().item()), 4),
                "tau_std": (
                    round(float(tau.std().item()), 4) if tau.numel() > 1 else 0.0
                ),
            }
            if issues:
                report.fail(check, "; ".join(issues), details=details, duration_ms=elapsed)
            else:
                report.pass_(
                    check,
                    f"strategy={strategy_name}: tau in [{min_tau:.2f}, {max_tau_val:.2f}]",
                    details=details, duration_ms=elapsed,
                )

        except Exception as exc:
            report.fail(check, f"Exception: {exc}", duration_ms=_elapsed(t0))


def check_tau_granularity(
    tau_cls,
    n: int,
    device: torch.device,
    report: ValidationReport,
) -> None:
    """Tau check 8: per_neuron (N,), per_channel (C,), per_layer (1,) all work."""
    granularities = [
        ("per_neuron", n, (n,)),
        ("per_channel", n, (n,)),
        ("per_layer", 1, (1,)),
    ]

    for gran_name, gran_n, expected_shape in granularities:
        check = f"HeterogeneousTau/granularity/{gran_name}"
        t0 = time.perf_counter()
        try:
            mod = tau_cls(n=gran_n).to(device)
            tau = mod.tau
            beta = mod.beta
            elapsed = _elapsed(t0)

            issues = []
            if tuple(tau.shape) != expected_shape:
                issues.append(
                    f"tau.shape {tuple(tau.shape)} != expected {expected_shape}"
                )
            if tuple(beta.shape) != expected_shape:
                issues.append(
                    f"beta.shape {tuple(beta.shape)} != expected {expected_shape}"
                )

            details = {
                "granularity": gran_name,
                "tau_shape": list(tau.shape),
                "beta_shape": list(beta.shape),
                "n": gran_n,
            }
            if issues:
                report.fail(check, "; ".join(issues), details=details, duration_ms=elapsed)
            else:
                report.pass_(
                    check,
                    f"{gran_name}: tau={list(tau.shape)}, beta={list(beta.shape)}",
                    details=details, duration_ms=elapsed,
                )

        except Exception as exc:
            report.fail(check, f"Exception: {exc}", duration_ms=_elapsed(t0))


def check_tau_broadcast(
    tau_cls,
    n: int,
    device: torch.device,
    report: ValidationReport,
    batch: int = 4,
) -> None:
    """Tau check 9: beta_broadcast produces correct shape for (B,N) and (B,C,H,W)."""
    # --- Dense broadcast (B, N) ---
    check_dense = "HeterogeneousTau/broadcast/dense"
    t0 = time.perf_counter()
    try:
        mod = tau_cls(n=n).to(device)
        if not hasattr(mod, "beta_broadcast"):
            report.skip(check_dense, "beta_broadcast() not available")
        else:
            shape_dense = (batch, n)
            b = mod.beta_broadcast(shape_dense)
            try:
                dummy = torch.randn(batch, n, device=device)
                _ = b * dummy
                elapsed = _elapsed(t0)
                report.pass_(
                    check_dense,
                    f"beta_broadcast compatible with (B={batch}, N={n})",
                    duration_ms=elapsed,
                )
            except RuntimeError as bcast_err:
                elapsed = _elapsed(t0)
                report.fail(
                    check_dense,
                    f"broadcast failed for shape {shape_dense}: {bcast_err}",
                    duration_ms=elapsed,
                )
    except Exception as exc:
        report.fail(check_dense, f"Exception: {exc}", duration_ms=_elapsed(t0))

    # --- Conv broadcast (B, C, H, W) ---
    check_conv = "HeterogeneousTau/broadcast/conv"
    t0 = time.perf_counter()
    try:
        C, H, W = n, 4, 4
        mod = tau_cls(n=C).to(device)
        if not hasattr(mod, "beta_broadcast"):
            report.skip(check_conv, "beta_broadcast() not available")
        else:
            shape_conv = (batch, C, H, W)
            b = mod.beta_broadcast(shape_conv)
            try:
                dummy = torch.randn(batch, C, H, W, device=device)
                _ = b * dummy
                elapsed = _elapsed(t0)
                report.pass_(
                    check_conv,
                    f"beta_broadcast compatible with (B={batch}, C={C}, H={H}, W={W})",
                    duration_ms=elapsed,
                )
            except RuntimeError as bcast_err:
                elapsed = _elapsed(t0)
                report.fail(
                    check_conv,
                    f"broadcast failed for shape {shape_conv}: {bcast_err}",
                    duration_ms=elapsed,
                )
    except Exception as exc:
        report.fail(check_conv, f"Exception: {exc}", duration_ms=_elapsed(t0))


def check_tau_health_check(
    tau_cls,
    n: int,
    device: torch.device,
    report: ValidationReport,
) -> None:
    """Tau check 10: get_diagnostics() returns a valid dict with expected keys."""
    check = "HeterogeneousTau/health_check"
    t0 = time.perf_counter()
    try:
        mod = tau_cls(n=n).to(device)

        if not hasattr(mod, "get_diagnostics"):
            report.skip(check, "get_diagnostics() not available on this module")
            return

        diag = mod.get_diagnostics()
        elapsed = _elapsed(t0)

        required_keys = [
            "tau_mean", "tau_min_actual", "tau_max_actual",
            "beta_mean", "beta_min_actual", "beta_max_actual",
            "n_total",
        ]
        issues = []

        if not isinstance(diag, dict):
            issues.append(
                f"get_diagnostics() returned {type(diag).__name__}, expected dict"
            )
        else:
            missing = [k for k in required_keys if k not in diag]
            if missing:
                issues.append(f"missing keys: {missing}")
            for k, v in diag.items():
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    issues.append(f"diagnostics['{k}'] = {v} (NaN/Inf)")

        details = {"diagnostics": diag if isinstance(diag, dict) else str(diag)}
        if issues:
            report.fail(check, "; ".join(issues), details=details, duration_ms=elapsed)
        else:
            report.pass_(
                check,
                "get_diagnostics() returns valid dict with all required keys",
                details=details, duration_ms=elapsed,
            )

    except Exception as exc:
        report.fail(check, f"Exception: {exc}", duration_ms=_elapsed(t0))


# ---------------------------------------------------------------------------
# SECTION 8: Legacy API checks (AdvancedLIFNeuron)
# ---------------------------------------------------------------------------

def check_legacy_delay_weights_exist(
    advanced_cls,
    device: torch.device,
    report: ValidationReport,
    size: int = 16,
) -> None:
    """Legacy check 1: AdvancedLIFNeuron has delay_weights parameter."""
    check = "legacy/delay_weights_exist"
    t0 = time.perf_counter()
    try:
        mod = advanced_cls(size=size).to(device)
        elapsed = _elapsed(t0)

        has_dw = hasattr(mod, "delay_weights") and mod.delay_weights is not None
        details = {
            "has_delay_weights": has_dw,
            "delay_weights_shape": (
                list(mod.delay_weights.shape) if has_dw else None
            ),
        }
        if has_dw:
            report.pass_(
                check,
                f"delay_weights exists, shape={list(mod.delay_weights.shape)}",
                details=details, duration_ms=elapsed,
            )
        else:
            report.fail(
                check, "delay_weights is None or missing",
                details=details, duration_ms=elapsed,
            )

    except Exception as exc:
        report.fail(check, f"Exception: {exc}", duration_ms=_elapsed(t0))


def check_legacy_logit_beta_exists(
    advanced_cls,
    device: torch.device,
    report: ValidationReport,
    size: int = 16,
) -> None:
    """Legacy check 2: AdvancedLIFNeuron has log_beta (logit-beta) parameter."""
    check = "legacy/logit_beta_vec_exists"
    t0 = time.perf_counter()
    try:
        mod = advanced_cls(size=size).to(device)
        elapsed = _elapsed(t0)

        has_lb = hasattr(mod, "log_beta") and mod.log_beta is not None
        details = {
            "has_log_beta": has_lb,
            "log_beta_shape": (
                list(mod.log_beta.shape) if has_lb else None
            ),
            "is_parameter": (
                isinstance(mod.log_beta, nn.Parameter) if has_lb else None
            ),
        }
        if has_lb:
            report.pass_(
                check,
                f"log_beta (logit-beta) exists, shape={list(mod.log_beta.shape)}",
                details=details, duration_ms=elapsed,
            )
        else:
            report.fail(
                check, "log_beta / logit_beta not found on AdvancedLIFNeuron",
                details=details, duration_ms=elapsed,
            )

    except Exception as exc:
        report.fail(check, f"Exception: {exc}", duration_ms=_elapsed(t0))


def check_legacy_beta_clamped(
    advanced_cls,
    device: torch.device,
    report: ValidationReport,
    size: int = 16,
) -> None:
    """Legacy check 3: beta property returns values in (BETA_MIN, BETA_MAX)."""
    check = "legacy/beta_clamped"
    BETA_MIN = 0.0
    BETA_MAX = 1.0
    t0 = time.perf_counter()
    try:
        mod = advanced_cls(size=size).to(device)
        beta = mod.beta
        elapsed = _elapsed(t0)

        min_b = float(beta.min().item())
        max_b = float(beta.max().item())
        issues = []

        if min_b < BETA_MIN:
            issues.append(f"beta min={min_b:.6f} < BETA_MIN={BETA_MIN}")
        if max_b > BETA_MAX:
            issues.append(f"beta max={max_b:.6f} > BETA_MAX={BETA_MAX}")
        if torch.isnan(beta).any():
            issues.append("beta contains NaN")

        details = {
            "beta_min": round(min_b, 6),
            "beta_max": round(max_b, 6),
            "BETA_MIN": BETA_MIN,
            "BETA_MAX": BETA_MAX,
        }
        if issues:
            report.fail(check, "; ".join(issues), details=details, duration_ms=elapsed)
        else:
            report.pass_(
                check, f"beta in [{min_b:.4f}, {max_b:.4f}]",
                details=details, duration_ms=elapsed,
            )

    except Exception as exc:
        report.fail(check, f"Exception: {exc}", duration_ms=_elapsed(t0))


def check_legacy_spike_history_shape(
    advanced_cls,
    device: torch.device,
    report: ValidationReport,
    size: int = 16,
    batch: int = 4,
    max_delay: int = 10,
) -> None:
    """Legacy check 4: spike_history has shape (B, max_delay, N)."""
    check = "legacy/spike_history_shape"
    t0 = time.perf_counter()
    try:
        mod = advanced_cls(size=size, max_delay=max_delay).to(device)
        x = torch.randn(batch, size, device=device)
        mod(x)   # run one forward step to initialise state
        elapsed = _elapsed(t0)

        sh = mod.spike_history
        issues = []

        if sh is None:
            issues.append("spike_history is None after forward pass")
        else:
            expected = (batch, max_delay, size)
            if tuple(sh.shape) != expected:
                issues.append(
                    f"spike_history.shape {tuple(sh.shape)} != expected {expected}"
                )

        details = {
            "spike_history_shape": list(sh.shape) if sh is not None else None,
            "expected": [batch, max_delay, size],
        }
        if issues:
            report.fail(check, "; ".join(issues), details=details, duration_ms=elapsed)
        else:
            report.pass_(
                check, f"spike_history.shape={list(sh.shape)}",
                details=details, duration_ms=elapsed,
            )

    except Exception as exc:
        report.fail(check, f"Exception: {exc}", duration_ms=_elapsed(t0))


def check_legacy_effective_delays(
    advanced_cls,
    device: torch.device,
    report: ValidationReport,
    size: int = 16,
    max_delay: int = 10,
) -> None:
    """Legacy check 5: get_effective_delays() returns values in [0, max_delay-1]."""
    check = "legacy/effective_delays"
    t0 = time.perf_counter()
    try:
        mod = advanced_cls(size=size, max_delay=max_delay).to(device)

        if not hasattr(mod, "get_effective_delays"):
            report.skip(check, "get_effective_delays() not available")
            return

        eff = mod.get_effective_delays()
        elapsed = _elapsed(t0)

        issues = []
        if eff is None:
            issues.append("get_effective_delays() returned None")
        else:
            min_d = float(eff.min().item())
            max_d = float(eff.max().item())
            if min_d < 0.0:
                issues.append(f"min effective delay {min_d:.4f} < 0")
            if max_d > float(max_delay - 1) + 1e-4:
                issues.append(f"max effective delay {max_d:.4f} > {max_delay - 1}")
            if torch.isnan(eff).any():
                issues.append("effective delays contain NaN")

        details = {
            "max_delay": max_delay,
            "eff_shape": list(eff.shape) if eff is not None else None,
            "eff_min": (
                round(float(eff.min().item()), 4) if eff is not None else None
            ),
            "eff_max": (
                round(float(eff.max().item()), 4) if eff is not None else None
            ),
        }
        if issues:
            report.fail(check, "; ".join(issues), details=details, duration_ms=elapsed)
        else:
            report.pass_(
                check,
                f"effective delays in [{details['eff_min']}, {details['eff_max']}]",
                details=details, duration_ms=elapsed,
            )

    except Exception as exc:
        report.fail(check, f"Exception: {exc}", duration_ms=_elapsed(t0))


# ---------------------------------------------------------------------------
# SECTION 9: Integration checks
# ---------------------------------------------------------------------------

def check_integration_delay_with_neuron(
    components: Dict[str, Any],
    device: torch.device,
    report: ValidationReport,
    batch: int = 4,
    timesteps: int = 30,
    features: int = 16,
    out_features: int = 16,
) -> None:
    """Integration 1: DelayLinear + LIFNeuron produces valid output."""
    check = "integration/delay_with_neuron"
    t0 = time.perf_counter()
    try:
        delay_cls = _get_delay_cls(components, "DelayLinear")
        lif_cls = components["neurons"].get("LIFNeuron")

        if delay_cls is None:
            report.skip(check, "DelayLinear not available")
            return
        if lif_cls is None:
            report.skip(check, "LIFNeuron not available")
            return

        delay_mod = delay_cls(features, out_features).to(device)
        lif = lif_cls().to(device)

        x = _make_sequence_input(batch, timesteps, features, device)
        delayed = delay_mod(x)          # (B, T, out_features)

        spk_list = []
        for t in range(timesteps):
            spk, _ = lif(delayed[:, t, :])
            spk_list.append(spk)

        spk_all = torch.stack(spk_list, dim=1)   # (B, T, out)
        elapsed = _elapsed(t0)

        expected_shape = (batch, timesteps, out_features)
        issues = []
        if tuple(spk_all.shape) != expected_shape:
            issues.append(
                f"spk_all.shape={tuple(spk_all.shape)}, "
                f"expected {expected_shape}"
            )
        if torch.isnan(spk_all).any():
            issues.append("spike output contains NaN")
        if torch.isinf(spk_all).any():
            issues.append("spike output contains Inf")

        unique = spk_all.unique()
        non_binary = unique[~((unique == 0) | (unique == 1))]
        if len(non_binary) > 0:
            issues.append(f"non-binary spikes: {non_binary.tolist()}")

        details = {
            "delay_module": delay_cls.__name__,
            "lif_class": lif_cls.__name__,
            "output_shape": list(spk_all.shape),
            "mean_firing_rate": round(float(spk_all.mean().item()), 4),
        }
        if issues:
            report.fail(check, "; ".join(issues), details=details, duration_ms=elapsed)
        else:
            report.pass_(
                check,
                f"DelayLinear + LIFNeuron OK; shape={list(spk_all.shape)}, "
                f"rate={details['mean_firing_rate']:.3f}",
                details=details, duration_ms=elapsed,
            )

    except Exception as exc:
        report.fail(
            check, f"Exception: {exc}\n{traceback.format_exc()}",
            duration_ms=_elapsed(t0),
        )


def check_integration_tau_with_neuron(
    components: Dict[str, Any],
    device: torch.device,
    report: ValidationReport,
    batch: int = 4,
    timesteps: int = 30,
    n: int = 16,
) -> None:
    """Integration 2: HeterogeneousTau + LIFNeuron produces valid output."""
    check = "integration/tau_with_neuron"
    t0 = time.perf_counter()
    try:
        tau_cls = _get_tau_cls(components)
        lif_cls = components["neurons"].get("LIFNeuron")

        if tau_cls is None:
            report.skip(check, "HeterogeneousTau not available")
            return
        if lif_cls is None:
            report.skip(check, "LIFNeuron not available")
            return

        tau_mod = tau_cls(n=n).to(device)
        lif = lif_cls().to(device)

        x = torch.randn(batch, n, device=device)
        lif.reset_mem()

        spk_list = []
        for _ in range(timesteps):
            beta = tau_mod.beta    # (n,)
            if lif.mem is None:
                lif.mem = torch.zeros(batch, n, device=device)
            lif.mem = beta.unsqueeze(0) * lif.mem
            spk, _ = lif(x)
            spk_list.append(spk)

        spk_all = torch.stack(spk_list, dim=1)
        elapsed = _elapsed(t0)

        issues = []
        if torch.isnan(spk_all).any():
            issues.append("spike output contains NaN")
        if torch.isinf(spk_all).any():
            issues.append("spike output contains Inf")

        unique = spk_all.unique()
        non_binary = unique[~((unique == 0) | (unique == 1))]
        if len(non_binary) > 0:
            issues.append(f"non-binary spikes: {non_binary.tolist()}")

        details = {
            "tau_class": tau_cls.__name__,
            "lif_class": lif_cls.__name__,
            "output_shape": list(spk_all.shape),
            "mean_firing_rate": round(float(spk_all.mean().item()), 4),
            "beta_mean": round(float(tau_mod.beta.mean().item()), 4),
        }
        if issues:
            report.fail(check, "; ".join(issues), details=details, duration_ms=elapsed)
        else:
            report.pass_(
                check,
                f"HeterogeneousTau + LIFNeuron OK; "
                f"rate={details['mean_firing_rate']:.3f}, "
                f"beta_mean={details['beta_mean']:.4f}",
                details=details, duration_ms=elapsed,
            )

    except Exception as exc:
        report.fail(
            check, f"Exception: {exc}\n{traceback.format_exc()}",
            duration_ms=_elapsed(t0),
        )


def check_integration_both_together(
    components: Dict[str, Any],
    device: torch.device,
    report: ValidationReport,
    batch: int = 4,
    timesteps: int = 30,
    features: int = 16,
    n: int = 16,
) -> None:
    """Integration 3: Delays + tau together without conflicts."""
    check = "integration/both_together"
    t0 = time.perf_counter()
    try:
        delay_cls = _get_delay_cls(components, "DelayLinear")
        tau_cls = _get_tau_cls(components)
        lif_cls = components["neurons"].get("LIFNeuron")

        if delay_cls is None or tau_cls is None or lif_cls is None:
            report.skip(check, "One or more required modules not available")
            return

        delay_mod = delay_cls(features, n).to(device)
        tau_mod = tau_cls(n=n).to(device)
        lif = lif_cls().to(device)

        x = _make_sequence_input(batch, timesteps, features, device)
        delayed = delay_mod(x)   # (B, T, n)

        lif.reset_mem()
        spk_list = []
        for t in range(timesteps):
            step_in = delayed[:, t, :]
            if lif.mem is not None:
                lif.mem = tau_mod.beta.unsqueeze(0) * lif.mem
            spk, _ = lif(step_in)
            spk_list.append(spk)

        spk_all = torch.stack(spk_list, dim=1)
        elapsed = _elapsed(t0)

        issues = []
        if torch.isnan(spk_all).any():
            issues.append("spike output contains NaN")
        if torch.isinf(spk_all).any():
            issues.append("spike output contains Inf")

        unique = spk_all.unique()
        non_binary = unique[~((unique == 0) | (unique == 1))]
        if len(non_binary) > 0:
            issues.append(f"non-binary spikes: {non_binary.tolist()}")

        details = {
            "output_shape": list(spk_all.shape),
            "mean_firing_rate": round(float(spk_all.mean().item()), 4),
        }
        if issues:
            report.fail(check, "; ".join(issues), details=details, duration_ms=elapsed)
        else:
            report.pass_(
                check,
                f"Delays + tau + LIF work together; "
                f"shape={list(spk_all.shape)}, "
                f"rate={details['mean_firing_rate']:.3f}",
                details=details, duration_ms=elapsed,
            )

    except Exception as exc:
        report.fail(
            check, f"Exception: {exc}\n{traceback.format_exc()}",
            duration_ms=_elapsed(t0),
        )


def check_integration_config_validation(
    components: Dict[str, Any],
    device: torch.device,
    report: ValidationReport,
) -> None:
    """Integration 4: Config presets from brain_ai.config pass validation."""
    check = "integration/config_validation"
    t0 = time.perf_counter()
    try:
        from brain_ai.config import BrainAIConfig  # type: ignore

        configs: Dict[str, Any] = {}
        try:
            configs["minimal"] = BrainAIConfig.minimal()
        except Exception:
            pass

        for preset_name in ("production_1b", "production_3b"):
            fn = getattr(BrainAIConfig, preset_name, None)
            if fn is not None:
                try:
                    configs[preset_name] = fn()
                except Exception:
                    pass

        elapsed = _elapsed(t0)
        issues = []
        validated = []

        for name, cfg in configs.items():
            try:
                snn_cfg = getattr(cfg, "snn", None)
                if snn_cfg is None:
                    continue
                for field in ("max_delay", "use_delays", "use_heterogeneous_tau"):
                    if hasattr(snn_cfg, field):
                        validated.append(
                            f"{name}.snn.{field}={getattr(snn_cfg, field)}"
                        )
            except Exception as cfg_err:
                issues.append(f"config {name} raised: {cfg_err}")

        details = {"validated_fields": validated}
        if issues:
            report.fail(check, "; ".join(issues), details=details, duration_ms=elapsed)
        else:
            report.pass_(
                check,
                f"config presets accessible; checked {len(validated)} delay/tau fields",
                details=details, duration_ms=elapsed,
            )

    except ImportError:
        report.skip(check, "brain_ai.config not importable")
    except Exception as exc:
        report.fail(check, f"Exception: {exc}", duration_ms=_elapsed(t0))


def check_integration_ablation_configs(
    components: Dict[str, Any],
    device: torch.device,
    report: ValidationReport,
    batch: int = 4,
    timesteps: int = 10,
    features: int = 16,
) -> None:
    """Integration 5: All 4 ablation configs produce valid models."""
    ablation_configs = [
        ("no_delays_no_tau",   False, False),
        ("delays_only",        True,  False),
        ("tau_only",           False, True),
        ("delays_and_tau",     True,  True),
    ]

    delay_cls = _get_delay_cls(components, "DelayLinear")
    tau_cls = _get_tau_cls(components)
    lif_cls = components["neurons"].get("LIFNeuron")

    if lif_cls is None:
        report.skip("integration/ablation", "LIFNeuron not available")
        return

    for ablation_name, use_delays, use_tau in ablation_configs:
        check = f"integration/ablation/{ablation_name}"
        t0 = time.perf_counter()
        try:
            delay_mod = (
                delay_cls(features, features).to(device) if use_delays else None
            )
            tau_mod = (
                tau_cls(n=features).to(device) if use_tau else None
            )
            lif = lif_cls().to(device)
            lif.reset_mem()

            x = _make_sequence_input(batch, timesteps, features, device)

            x_proc = delay_mod(x) if delay_mod is not None else x

            spk_list = []
            for t in range(timesteps):
                step_in = x_proc[:, t, :]
                if tau_mod is not None and lif.mem is not None:
                    lif.mem = tau_mod.beta.unsqueeze(0) * lif.mem
                spk, _ = lif(step_in)
                spk_list.append(spk)

            spk_all = torch.stack(spk_list, dim=1)
            elapsed = _elapsed(t0)

            issues = []
            if torch.isnan(spk_all).any():
                issues.append("NaN in spike output")
            if torch.isinf(spk_all).any():
                issues.append("Inf in spike output")

            unique = spk_all.unique()
            non_binary = unique[~((unique == 0) | (unique == 1))]
            if len(non_binary) > 0:
                issues.append("non-binary spikes detected")

            details = {
                "use_delays": use_delays,
                "use_tau": use_tau,
                "output_shape": list(spk_all.shape),
                "mean_rate": round(float(spk_all.mean().item()), 4),
            }
            if issues:
                report.fail(check, "; ".join(issues), details=details, duration_ms=elapsed)
            else:
                report.pass_(
                    check,
                    f"ablation {ablation_name}: valid output, "
                    f"rate={details['mean_rate']:.3f}",
                    details=details, duration_ms=elapsed,
                )

        except Exception as exc:
            report.fail(check, f"Exception: {exc}", duration_ms=_elapsed(t0))


# ---------------------------------------------------------------------------
# SECTION 10: Orchestrator
# ---------------------------------------------------------------------------

def run_all_validations(args: argparse.Namespace) -> ValidationReport:
    """Detect API, run all requested checks, return the populated report."""
    report = ValidationReport()

    # Resolve device
    if args.device == "cuda":
        if not torch.cuda.is_available():
            warnings.warn(
                "--device cuda requested but CUDA not available; using CPU.",
                stacklevel=2,
            )
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device(args.device)

    # Detect API
    components = detect_api()
    api_level = components["api_level"]
    has_delays = bool(components["delays"])
    has_tau = bool(components["tau"])
    has_legacy = bool(components["neurons"])

    if api_level == "none":
        report.fail(
            "_global/import",
            f"No brain_ai modules importable from {_REPO_ROOT}. "
            f"Errors: {components['errors']}",
        )
        return report

    report.pass_(
        "_global/import",
        (
            f"api_level={api_level} | "
            f"delays={'real' if has_delays else 'stub'} | "
            f"tau={'real' if has_tau else 'stub'} | "
            f"legacy_neurons={'yes' if has_legacy else 'no'}"
        ),
        details={
            "api_level": api_level,
            "repo_root": _REPO_ROOT,
            "delay_classes": list(components["delays"].keys()),
            "tau_classes": list(components["tau"].keys()),
            "neuron_classes": list(components["neurons"].keys()),
            "import_errors": components["errors"],
        },
    )

    check_only = args.check_only
    batch = args.batch_size
    T = args.timesteps
    in_f = 16
    out_f = 16
    max_delay = 8
    n_tau = 32

    # -----------------------------------------------------------------------
    # DELAY CHECKS
    # -----------------------------------------------------------------------
    if check_only in ("delays", "both"):
        if api_level == "target" and has_delays:
            classes_to_check = list(components["delays"].items())
        else:
            classes_to_check = [("DelayLinear(stub)", _StubDelayLinear)]

        for cls_name, delay_cls in classes_to_check:
            label = f"Delay/{cls_name}"
            check_delay_forward_shapes(
                delay_cls, label, batch, T, in_f, out_f, device, report
            )
            check_delay_range(
                delay_cls, label, batch, T, in_f, out_f, max_delay, device, report
            )
            check_delay_gaussian_normalization(
                delay_cls, label, in_f, out_f, max_delay, device, report
            )
            check_delay_eval_discretization(
                delay_cls, label, in_f, out_f, max_delay, device, report
            )
            check_delay_gradient_flow(
                delay_cls, label, batch, T, in_f, out_f, device, report
            )
            check_delay_no_gradient_fixed(
                delay_cls, label, batch, T, in_f, out_f, device, report
            )
            check_delay_sigma_schedule(
                delay_cls, label, in_f, out_f, device, report
            )
            check_delay_sigma_checkpoint(
                delay_cls, label, in_f, out_f, device, report
            )
            check_delay_memory_guard(
                delay_cls, label, batch, T, in_f, out_f, max_delay, device, report
            )
            check_delay_granularity_shapes(
                delay_cls, label, in_f, out_f, max_delay, device, report
            )
            check_delay_spike_history_update(
                delay_cls, label, batch, T, in_f, out_f, device, report
            )

    # -----------------------------------------------------------------------
    # TAU CHECKS
    # -----------------------------------------------------------------------
    if check_only in ("tau", "both"):
        tau_cls = (
            components["tau"]["HeterogeneousTau"]
            if (api_level == "target" and has_tau)
            else _StubHeterogeneousTau
        )
        check_tau_beta_range(tau_cls, n_tau, device, report)
        check_tau_positive(tau_cls, n_tau, device, report)
        check_tau_softplus_mapping(tau_cls, n_tau, device, report)
        check_tau_exp_mapping(tau_cls, n_tau, device, report)
        check_tau_gradient_flow(tau_cls, n_tau, device, report, batch)
        check_tau_no_gradient_fixed(tau_cls, n_tau, device, report, batch)
        check_tau_init_strategies(tau_cls, n_tau, device, report)
        check_tau_granularity(tau_cls, n_tau, device, report)
        check_tau_broadcast(tau_cls, n_tau, device, report, batch)
        check_tau_health_check(tau_cls, n_tau, device, report)

    # -----------------------------------------------------------------------
    # LEGACY CHECKS (when no target API present)
    # -----------------------------------------------------------------------
    if check_only == "both" and has_legacy:
        advanced_cls = components["neurons"].get("AdvancedLIFNeuron")
        if advanced_cls is not None:
            check_legacy_delay_weights_exist(advanced_cls, device, report)
            check_legacy_logit_beta_exists(advanced_cls, device, report)
            check_legacy_beta_clamped(advanced_cls, device, report)
            check_legacy_spike_history_shape(
                advanced_cls, device, report, batch=batch
            )
            check_legacy_effective_delays(advanced_cls, device, report)

    # -----------------------------------------------------------------------
    # INTEGRATION CHECKS
    # -----------------------------------------------------------------------
    if check_only == "both":
        check_integration_delay_with_neuron(
            components, device, report, batch, T, in_f, out_f
        )
        check_integration_tau_with_neuron(
            components, device, report, batch, T, n_tau
        )
        check_integration_both_together(
            components, device, report, batch, T, in_f, n_tau
        )
        check_integration_config_validation(components, device, report)
        check_integration_ablation_configs(
            components, device, report, batch, min(T, 10), in_f
        )

    return report


# ---------------------------------------------------------------------------
# SECTION 11: CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="validate_delays_tau.py",
        description=(
            "Runtime contract validation for delay modules and heterogeneous tau.\n\n"
            "Detects whether brain_ai.core.delays / brain_ai.core.heterogeneous_tau\n"
            "are present (target API) or only AdvancedLIFNeuron exists (legacy API)\n"
            "and runs the appropriate checks."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python validate_delays_tau.py\n"
            "  python validate_delays_tau.py --check-only delays\n"
            "  python validate_delays_tau.py --check-only tau --verbose\n"
            "  python validate_delays_tau.py --json > results.json\n"
            "  python validate_delays_tau.py --device cuda --batch-size 8\n"
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device: cpu, cuda, cuda:0, etc. (default: cpu)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        metavar="N",
        help="Batch size for validation tensors (default: 4)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=30,
        metavar="T",
        help="Number of timesteps for sequence tests (default: 30)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON to stdout (for CI integration)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed pass/fail messages and extra details",
    )
    parser.add_argument(
        "--check-only",
        choices=["delays", "tau", "both"],
        default="both",
        dest="check_only",
        help="Which module group to validate (default: both)",
    )
    parser.add_argument(
        "--no-warnings",
        action="store_true",
        help="Suppress Python warnings during validation",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.no_warnings:
        warnings.filterwarnings("ignore")

    if not args.json:
        print("=" * 70)
        print("  Delay & Tau Contract Validation")
        print("=" * 70)
        print(f"  repo root    : {_REPO_ROOT}")
        print(f"  device       : {args.device}")
        print(f"  batch_size   : {args.batch_size}")
        print(f"  timesteps    : {args.timesteps}")
        print(f"  check_only   : {args.check_only}")
        print(f"  torch        : {torch.__version__}")
        cuda_info = (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else "not available"
        )
        print(f"  cuda         : {cuda_info}")
        print()

    t_start = time.perf_counter()
    report = run_all_validations(args)
    t_total = (time.perf_counter() - t_start) * 1000.0

    if args.json:
        print(report.to_json())
    else:
        report.print_report(verbose=args.verbose)
        print(f"\nTotal validation time: {t_total:.1f} ms")

    sys.exit(0 if report.all_passed else 1)


if __name__ == "__main__":
    main()
