#!/usr/bin/env python3
"""
validate_spiking.py - Runtime contract validation for spiking core.

Validates:
1. Neuron state contracts (shapes, dtypes, device alignment)
2. Gradient presence and correctness for all surrogates
3. State reset/carry/detach behavior
4. Time unrolling consistency (full vs truncated BPTT)
5. Numerical stability (beta bounds, fp32 accumulation, long unrolls)
6. CPU/CUDA parity (if CUDA available)
7. Debug surface accessibility

The current implementation uses the legacy implicit-state API:
  - State is stored on the neuron instance as self.mem
  - reset_mem() clears all neuron state
  - forward() returns (spk, mem) tuple

Usage:
    python validate_spiking.py                    # Validate all
    python validate_spiking.py --neuron lif       # Single neuron type
    python validate_spiking.py --json             # JSON output
    python validate_spiking.py --verbose          # Detailed output
    python validate_spiking.py --device cuda      # Force CUDA
"""

# ---------------------------------------------------------------------------
# SECTION 1: Imports and path setup
# ---------------------------------------------------------------------------

import os
import sys
import json
import math
import argparse
import time
import traceback
import inspect
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# scripts/ -> spiking-core/ -> skills/ -> brain-ai-dev/ -> human-brain/
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

    def pass_(self, name: str, message: str = "ok",
               details: Optional[Dict] = None,
               duration_ms: float = 0.0) -> ValidationResult:
        r = ValidationResult(name=name, status=PASS_MARK, message=message,
                             details=details, duration_ms=duration_ms)
        self.add(r)
        return r

    def fail(self, name: str, message: str,
              details: Optional[Dict] = None,
              duration_ms: float = 0.0) -> ValidationResult:
        r = ValidationResult(name=name, status=FAIL_MARK, message=message,
                             details=details, duration_ms=duration_ms)
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
        print("  SPIKING CORE CONTRACT VALIDATION REPORT")
        print("=" * 70)

        # Group by first path segment (neuron name)
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

            print(f"\n[{marker}] {group}  "
                  f"({g_passed}/{g_total} passed, {g_skipped} skipped)")

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
# SECTION 3: Import helpers
# ---------------------------------------------------------------------------

def _has_param(cls: type, param_name: str) -> bool:
    """Check if a class __init__ accepts a parameter by name, using inspect."""
    try:
        sig = inspect.signature(cls.__init__)
        return param_name in sig.parameters
    except (ValueError, TypeError):
        return False


def import_spiking_core() -> Optional[Dict[str, Any]]:
    """
    Import spiking core components.

    The current codebase uses the legacy implicit-state API where each
    neuron stores self.mem internally and exposes reset_mem().

    Returns a dict with keys:
        api             : 'legacy'
        neurons         : dict mapping name -> (class, constructor_kwargs)
        surrogates      : list of surrogate name strings
        get_surrogate   : the get_surrogate function
        surrogate_classes : dict name -> class
    """
    try:
        from brain_ai.core.neurons import (
            LIFNeuron,
            AdaptiveLIFNeuron,
            RecurrentLIFNeuron,
            AdvancedLIFNeuron,
            get_surrogate,
            ATanSurrogate,
            FastSigmoidSurrogate,
            StraightThroughSurrogate,
        )

        # Each entry: (class, required_ctor_kwargs)
        neurons = {
            "lif": (LIFNeuron, {}),
            "adaptive_lif": (AdaptiveLIFNeuron, {}),
            "recurrent_lif": (RecurrentLIFNeuron, {"size": 16}),
            "advanced_lif": (AdvancedLIFNeuron, {"size": 16}),
        }

        return {
            "api": "legacy",
            "neurons": neurons,
            "surrogates": ["atan", "fast_sigmoid", "straight_through"],
            "get_surrogate": get_surrogate,
            "surrogate_classes": {
                "atan": ATanSurrogate,
                "fast_sigmoid": FastSigmoidSurrogate,
                "straight_through": StraightThroughSurrogate,
            },
        }

    except ImportError as exc:
        print(f"[ERROR] Cannot import brain_ai.core.neurons: {exc}")
        print(f"        sys.path[0] = {sys.path[0]}")
        return None


def build_neuron(
    neuron_cls: Type[nn.Module],
    ctor_kwargs: Dict[str, Any],
    device: torch.device,
) -> Optional[nn.Module]:
    """Instantiate a neuron on device, returning None on error."""
    try:
        neuron = neuron_cls(**ctor_kwargs)
        neuron = neuron.to(device)
        return neuron
    except Exception:
        return None


def make_input(
    ctor_kwargs: Dict[str, Any],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Create a (batch_size, features) input tensor.
    RecurrentLIF and AdvancedLIF require the input width to match 'size'.
    Other neurons accept any width; we default to 16.
    """
    n_features = ctor_kwargs.get("size", 16)
    return torch.randn(batch_size, n_features, device=device, dtype=dtype)


def neuron_forward(
    neuron: nn.Module,
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unified forward wrapper; always returns (spk, mem)."""
    return neuron(x)


def detach_neuron_state(neuron: nn.Module) -> None:
    """Detach all mutable state tensors stored on a neuron in-place."""
    state_attrs = ["mem", "adaptation", "prev_spk", "spike_history"]
    for attr in state_attrs:
        val = getattr(neuron, attr, None)
        if isinstance(val, torch.Tensor):
            setattr(neuron, attr, val.detach())


# ---------------------------------------------------------------------------
# SECTION 4: Per-neuron validation functions
# ---------------------------------------------------------------------------

def validate_neuron_forward(
    neuron_cls: Type[nn.Module],
    neuron_name: str,
    ctor_kwargs: Dict[str, Any],
    device: torch.device,
    report: ValidationReport,
    batch_size: int = 4,
) -> None:
    """Validate forward pass produces correct output shapes and binary spikes."""
    check = f"{neuron_name}/forward_shapes"
    t0 = time.perf_counter()
    try:
        neuron = build_neuron(neuron_cls, ctor_kwargs, device)
        if neuron is None:
            report.fail(check, "Failed to construct neuron")
            return

        x = make_input(ctor_kwargs, batch_size, device)
        spk, mem = neuron_forward(neuron, x)
        elapsed = (time.perf_counter() - t0) * 1000.0

        issues = []

        if spk.shape != x.shape:
            issues.append(f"spk.shape={spk.shape} != x.shape={x.shape}")

        if mem.shape != x.shape:
            issues.append(f"mem.shape={mem.shape} != x.shape={x.shape}")

        # Spikes must be binary float
        unique_vals = spk.unique()
        non_binary = unique_vals[~((unique_vals == 0) | (unique_vals == 1))]
        if len(non_binary) > 0:
            issues.append(
                f"spk contains non-binary values: {non_binary.tolist()[:5]}"
            )

        if spk.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            issues.append(f"spk.dtype={spk.dtype} is not a float type")

        if spk.device.type != device.type:
            issues.append(f"spk on {spk.device}, expected {device}")
        if mem.device.type != device.type:
            issues.append(f"mem on {mem.device}, expected {device}")

        if torch.isnan(spk).any() or torch.isinf(spk).any():
            issues.append("spk contains NaN or Inf")
        if torch.isnan(mem).any() or torch.isinf(mem).any():
            issues.append("mem contains NaN or Inf")

        details = {
            "spk_shape": list(spk.shape),
            "mem_shape": list(mem.shape),
            "spk_dtype": str(spk.dtype),
            "unique_spike_vals": unique_vals.tolist(),
            "device": str(device),
        }

        if issues:
            report.fail(check, "; ".join(issues), details=details,
                        duration_ms=elapsed)
        else:
            report.pass_(
                check,
                f"spk{list(spk.shape)} binary float on {device}",
                details=details,
                duration_ms=elapsed,
            )

    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000.0
        report.fail(check, f"Exception: {exc}", duration_ms=elapsed)


def validate_state_shapes(
    neuron_cls: Type[nn.Module],
    neuron_name: str,
    ctor_kwargs: Dict[str, Any],
    device: torch.device,
    report: ValidationReport,
    batch_size: int = 4,
) -> None:
    """Validate that self.mem is batch-aligned and on the correct device."""
    check = f"{neuron_name}/state_shapes"
    t0 = time.perf_counter()
    try:
        neuron = build_neuron(neuron_cls, ctor_kwargs, device)
        if neuron is None:
            report.fail(check, "Failed to construct neuron")
            return

        x = make_input(ctor_kwargs, batch_size, device)
        neuron_forward(neuron, x)
        elapsed = (time.perf_counter() - t0) * 1000.0

        issues = []

        if not hasattr(neuron, "mem") or neuron.mem is None:
            issues.append("neuron.mem is None after forward")
        else:
            if neuron.mem.shape[0] != batch_size:
                issues.append(
                    f"mem.shape[0]={neuron.mem.shape[0]} != batch_size={batch_size}"
                )
            if neuron.mem.device.type != device.type:
                issues.append(f"mem on {neuron.mem.device}, expected {device}")

        for attr in ("adaptation", "prev_spk"):
            val = getattr(neuron, attr, None)
            if isinstance(val, torch.Tensor):
                if val.shape[0] != batch_size:
                    issues.append(
                        f"{attr}.shape[0]={val.shape[0]} != {batch_size}"
                    )

        details = {
            "mem_shape": list(neuron.mem.shape)
            if (hasattr(neuron, "mem") and neuron.mem is not None)
            else None,
        }

        if issues:
            report.fail(check, "; ".join(issues), details=details,
                        duration_ms=elapsed)
        else:
            report.pass_(
                check,
                "all state tensors batch-aligned on correct device",
                details=details,
                duration_ms=elapsed,
            )

    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000.0
        report.fail(check, f"Exception: {exc}", duration_ms=elapsed)


def validate_gradient_presence(
    neuron_cls: Type[nn.Module],
    neuron_name: str,
    ctor_kwargs: Dict[str, Any],
    surrogate_name: str,
    device: torch.device,
    report: ValidationReport,
    batch_size: int = 4,
    num_timesteps: int = 5,
) -> None:
    """Validate that gradients flow through the surrogate to learnable parameters."""
    check = f"{neuron_name}/gradient_presence/{surrogate_name}"
    t0 = time.perf_counter()
    try:
        n_features = ctor_kwargs.get("size", 16)

        # Wrap neuron in a tiny two-linear network
        layer_in = nn.Linear(n_features, n_features).to(device)
        layer_out = nn.Linear(n_features, 2).to(device)

        kw = dict(ctor_kwargs)
        if _has_param(neuron_cls, "surrogate"):
            kw["surrogate"] = surrogate_name

        neuron = build_neuron(neuron_cls, kw, device)
        if neuron is None:
            report.fail(
                check,
                f"Failed to construct neuron with surrogate={surrogate_name}",
            )
            return

        layer_in.zero_grad()
        layer_out.zero_grad()
        neuron.zero_grad()

        x_static = torch.randn(batch_size, n_features, device=device)
        spike_accum = torch.zeros(batch_size, 2, device=device)

        for _ in range(num_timesteps):
            cur = layer_in(x_static)
            spk, _ = neuron_forward(neuron, cur)
            out = layer_out(spk)
            spike_accum = spike_accum + out

        loss = spike_accum.sum()
        loss.backward()

        elapsed = (time.perf_counter() - t0) * 1000.0
        issues = []

        if layer_in.weight.grad is None:
            issues.append(
                "layer_in.weight.grad is None -- surrogate may have blocked gradient"
            )
        elif layer_in.weight.grad.abs().sum().item() == 0:
            issues.append("layer_in.weight.grad is all-zeros -- dead gradient")

        if layer_out.weight.grad is None:
            issues.append("layer_out.weight.grad is None")

        for pname, param in neuron.named_parameters():
            if param.requires_grad and param.grad is None:
                issues.append(f"neuron.{pname}.grad is None")

        details = {
            "surrogate": surrogate_name,
            "in_grad_norm": (
                float(layer_in.weight.grad.norm().item())
                if layer_in.weight.grad is not None else None
            ),
            "timesteps": num_timesteps,
        }

        if issues:
            report.fail(check, "; ".join(issues), details=details,
                        duration_ms=elapsed)
        else:
            report.pass_(
                check,
                f"gradients flow for surrogate={surrogate_name}",
                details=details,
                duration_ms=elapsed,
            )

    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000.0
        report.fail(
            check,
            f"Exception: {exc}\n{traceback.format_exc()}",
            duration_ms=elapsed,
        )


def validate_surrogate_swap(
    neuron_cls: Type[nn.Module],
    neuron_name: str,
    ctor_kwargs: Dict[str, Any],
    device: torch.device,
    report: ValidationReport,
    surrogate_names: Tuple[str, ...] = ("atan", "fast_sigmoid", "straight_through"),
    batch_size: int = 4,
    num_timesteps: int = 5,
) -> None:
    """
    Validate that different surrogates produce different gradient norms.
    At least two surrogates must differ to confirm each has its own behaviour.
    """
    check = f"{neuron_name}/surrogate_swap"
    t0 = time.perf_counter()

    n_features = ctor_kwargs.get("size", 16)
    grad_norms: Dict[str, float] = {}

    torch.manual_seed(42)
    x_static = torch.randn(batch_size, n_features, device=device)

    try:
        for sur in surrogate_names:
            kw = dict(ctor_kwargs)
            if _has_param(neuron_cls, "surrogate"):
                kw["surrogate"] = sur

            layer = nn.Linear(n_features, n_features, bias=False).to(device)
            nn.init.normal_(layer.weight, std=0.5)
            layer.zero_grad()

            neuron = build_neuron(neuron_cls, kw, device)
            if neuron is None:
                continue

            accum = torch.zeros(batch_size, n_features, device=device)
            for _ in range(num_timesteps):
                cur = layer(x_static.detach().clone())
                spk, _ = neuron_forward(neuron, cur)
                accum = accum + spk

            loss = accum.sum()
            loss.backward()

            if layer.weight.grad is not None:
                grad_norms[sur] = float(layer.weight.grad.norm().item())

        elapsed = (time.perf_counter() - t0) * 1000.0

        if len(grad_norms) < 2:
            report.fail(
                check,
                "Could not compute gradient norms for multiple surrogates",
                details={"grad_norms": grad_norms},
                duration_ms=elapsed,
            )
            return

        unique_norms = set(round(v, 8) for v in grad_norms.values())
        details = {"grad_norms": {k: round(v, 6) for k, v in grad_norms.items()}}

        if len(unique_norms) < 2:
            report.fail(
                check,
                "All surrogates produced identical gradient norms",
                details=details,
                duration_ms=elapsed,
            )
        else:
            report.pass_(
                check,
                f"surrogates produce distinct gradient norms: {grad_norms}",
                details=details,
                duration_ms=elapsed,
            )

    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000.0
        report.fail(check, f"Exception: {exc}", duration_ms=elapsed)


def validate_state_reset_carry(
    neuron_cls: Type[nn.Module],
    neuron_name: str,
    ctor_kwargs: Dict[str, Any],
    device: torch.device,
    report: ValidationReport,
    batch_size: int = 4,
    num_timesteps: int = 10,
) -> None:
    """
    Validate reset determinism and carry produces different results.

    Sequence:
        y1: forward T steps with reset_mem() before
        y2: forward T steps with reset_mem() before  (must equal y1)
        y3: forward T steps WITHOUT reset  (must differ from y1)
    """
    check = f"{neuron_name}/state_reset_carry"
    t0 = time.perf_counter()
    try:
        n_features = ctor_kwargs.get("size", 16)
        torch.manual_seed(7)
        x = torch.randn(batch_size, n_features, device=device)

        def run_steps(neuron: nn.Module, inp: torch.Tensor,
                      n: int, reset: bool) -> torch.Tensor:
            if reset:
                neuron.reset_mem()
            last_mem = None
            for _ in range(n):
                _, last_mem = neuron_forward(neuron, inp)
            return last_mem.detach().clone()

        neuron = build_neuron(neuron_cls, ctor_kwargs, device)
        if neuron is None:
            report.fail(check, "Failed to construct neuron")
            return

        y1 = run_steps(neuron, x, num_timesteps, reset=True)
        y2 = run_steps(neuron, x, num_timesteps, reset=True)
        y3 = run_steps(neuron, x, num_timesteps, reset=False)

        elapsed = (time.perf_counter() - t0) * 1000.0
        issues = []

        if not torch.allclose(y1, y2, atol=1e-6):
            diff = float((y1 - y2).abs().max().item())
            issues.append(
                f"Reset not deterministic: max diff y1 vs y2 = {diff:.2e}"
            )

        if torch.allclose(y1, y3, atol=1e-6):
            issues.append(
                "Carry has no effect: y3 == y1 (state was not preserved)"
            )

        details = {
            "max_diff_y1_y2": float((y1 - y2).abs().max().item()),
            "max_diff_y1_y3": float((y1 - y3).abs().max().item()),
        }

        if issues:
            report.fail(check, "; ".join(issues), details=details,
                        duration_ms=elapsed)
        else:
            report.pass_(
                check,
                "reset is deterministic; carry produces different output",
                details=details,
                duration_ms=elapsed,
            )

    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000.0
        report.fail(check, f"Exception: {exc}", duration_ms=elapsed)


def validate_state_detach(
    neuron_cls: Type[nn.Module],
    neuron_name: str,
    ctor_kwargs: Dict[str, Any],
    device: torch.device,
    report: ValidationReport,
    batch_size: int = 4,
    chunk_size: int = 5,
) -> None:
    """
    Validate that detaching neuron state between BPTT chunks severs gradient flow.

    Procedure:
        1. Forward chunk 1 with x1 (requires_grad=True)
        2. Detach all state on the neuron
        3. Forward chunk 2 with x2 (requires_grad=True)
        4. backward from chunk 2 loss
        5. x1 must have no gradient; x2 must have gradient
    """
    check = f"{neuron_name}/state_detach"
    t0 = time.perf_counter()
    try:
        n_features = ctor_kwargs.get("size", 16)
        neuron = build_neuron(neuron_cls, ctor_kwargs, device)
        if neuron is None:
            report.fail(check, "Failed to construct neuron")
            return

        neuron.reset_mem()

        x1 = torch.randn(batch_size, n_features, device=device, requires_grad=True)
        x2 = torch.randn(batch_size, n_features, device=device, requires_grad=True)

        # Chunk 1
        for _ in range(chunk_size):
            neuron_forward(neuron, x1)

        # Detach state to sever gradient graph
        detach_neuron_state(neuron)

        # Chunk 2
        spk2_list = []
        for _ in range(chunk_size):
            spk, _ = neuron_forward(neuron, x2)
            spk2_list.append(spk)

        loss = torch.stack(spk2_list).sum()
        loss.backward()

        elapsed = (time.perf_counter() - t0) * 1000.0
        issues = []

        if x2.grad is None or x2.grad.abs().sum().item() == 0:
            issues.append(
                "x2 has no gradient -- chunk 2 failed to backprop"
            )

        if x1.grad is not None and x1.grad.abs().sum().item() > 0:
            issues.append(
                "x1 has non-zero gradient after state detach -- "
                "detach did not sever gradient flow"
            )

        details = {
            "x1_grad_norm": float(x1.grad.norm().item())
            if x1.grad is not None else 0.0,
            "x2_grad_norm": float(x2.grad.norm().item())
            if x2.grad is not None else 0.0,
        }

        if issues:
            report.fail(check, "; ".join(issues), details=details,
                        duration_ms=elapsed)
        else:
            report.pass_(
                check,
                "detach correctly severs gradient flow across BPTT chunks",
                details=details,
                duration_ms=elapsed,
            )

    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000.0
        report.fail(
            check,
            f"Exception: {exc}\n{traceback.format_exc()}",
            duration_ms=elapsed,
        )


# ---------------------------------------------------------------------------
# SECTION 5: Numerical stability validation
# ---------------------------------------------------------------------------

def validate_beta_bounds(
    neuron_cls: Type[nn.Module],
    neuron_name: str,
    ctor_kwargs: Dict[str, Any],
    device: torch.device,
    report: ValidationReport,
    batch_size: int = 4,
) -> None:
    """
    Validate that out-of-range beta values do not produce NaN/Inf.

    Tests beta values of 1.5 and -0.1 (outside the valid [0, 1) range).
    A well-implemented neuron should either clamp, raise, or handle
    gracefully -- any of these is acceptable as long as no NaN/Inf emerges.
    """
    check = f"{neuron_name}/beta_bounds"
    t0 = time.perf_counter()
    try:
        n_features = ctor_kwargs.get("size", 16)
        issues = []
        tested: List[float] = []

        # Determine which kwarg name controls beta
        beta_kwarg = None
        if _has_param(neuron_cls, "beta"):
            beta_kwarg = "beta"
        elif _has_param(neuron_cls, "beta_init"):
            beta_kwarg = "beta_init"

        if beta_kwarg is None:
            report.skip(check, "no beta kwarg accepted by this neuron class")
            return

        for bad_beta in [1.5, -0.1]:
            kw = dict(ctor_kwargs)
            kw[beta_kwarg] = bad_beta
            try:
                neuron = build_neuron(neuron_cls, kw, device)
                if neuron is None:
                    continue

                x = make_input(kw, batch_size, device)
                for _ in range(5):
                    spk, mem = neuron_forward(neuron, x)

                tested.append(bad_beta)

                if torch.isnan(spk).any() or torch.isinf(spk).any():
                    issues.append(f"spk has NaN/Inf with {beta_kwarg}={bad_beta}")
                if torch.isnan(mem).any() or torch.isinf(mem).any():
                    issues.append(f"mem has NaN/Inf with {beta_kwarg}={bad_beta}")

            except Exception:
                # Clamping via exception is acceptable
                pass

        elapsed = (time.perf_counter() - t0) * 1000.0
        details = {"tested_betas": tested, "beta_kwarg": beta_kwarg}

        if not tested:
            report.skip(check, "no bad betas could be constructed (may be clamped at init)")
            return

        if issues:
            report.fail(check, "; ".join(issues), details=details, duration_ms=elapsed)
        else:
            report.pass_(
                check,
                f"no NaN/Inf with out-of-range betas {tested}",
                details=details,
                duration_ms=elapsed,
            )

    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000.0
        report.fail(check, f"Exception: {exc}", duration_ms=elapsed)


def validate_long_unroll(
    neuron_cls: Type[nn.Module],
    neuron_name: str,
    ctor_kwargs: Dict[str, Any],
    device: torch.device,
    report: ValidationReport,
    batch_size: int = 4,
    num_timesteps: int = 200,
) -> None:
    """Validate numerical stability over T=200 timesteps."""
    check = f"{neuron_name}/long_unroll_T{num_timesteps}"
    t0 = time.perf_counter()
    try:
        neuron = build_neuron(neuron_cls, ctor_kwargs, device)
        if neuron is None:
            report.fail(check, "Failed to construct neuron")
            return

        n_features = ctor_kwargs.get("size", 16)
        torch.manual_seed(99)
        x = torch.randn(batch_size, n_features, device=device)

        neuron.reset_mem()
        total_spikes = 0.0
        mem_max = -float("inf")
        mem_min = float("inf")
        nan_found = False
        inf_found = False

        for _ in range(num_timesteps):
            spk, mem = neuron_forward(neuron, x)
            total_spikes += float(spk.sum().item())
            cur_max = float(mem.max().item())
            cur_min = float(mem.min().item())
            if math.isnan(cur_max) or math.isnan(cur_min):
                nan_found = True
                break
            if math.isinf(cur_max) or math.isinf(cur_min):
                inf_found = True
                break
            mem_max = max(mem_max, cur_max)
            mem_min = min(mem_min, cur_min)

        elapsed = (time.perf_counter() - t0) * 1000.0
        issues = []

        if nan_found:
            issues.append(f"NaN detected during {num_timesteps}-step unroll")
        if inf_found:
            issues.append(f"Inf detected during {num_timesteps}-step unroll")

        avg_firing_rate = total_spikes / (
            num_timesteps * batch_size * n_features
        )
        if avg_firing_rate > 0.99:
            issues.append(
                f"Firing rate saturated: {avg_firing_rate:.3f} "
                "(all neurons firing every timestep)"
            )

        details = {
            "timesteps": num_timesteps,
            "avg_firing_rate": round(avg_firing_rate, 4),
            "mem_min": round(mem_min, 4) if not (nan_found or inf_found) else None,
            "mem_max": round(mem_max, 4) if not (nan_found or inf_found) else None,
        }

        if issues:
            report.fail(check, "; ".join(issues), details=details, duration_ms=elapsed)
        else:
            report.pass_(
                check,
                (
                    f"stable over {num_timesteps} steps; "
                    f"avg_fire_rate={avg_firing_rate:.3f}, "
                    f"mem in [{mem_min:.3f}, {mem_max:.3f}]"
                ),
                details=details,
                duration_ms=elapsed,
            )

    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000.0
        report.fail(check, f"Exception: {exc}", duration_ms=elapsed)


def validate_fp32_state(
    neuron_cls: Type[nn.Module],
    neuron_name: str,
    ctor_kwargs: Dict[str, Any],
    device: torch.device,
    report: ValidationReport,
    batch_size: int = 4,
) -> None:
    """
    Validate that the internal membrane potential stays fp32 even when
    the input is bf16 (mixed-precision AMP scenario).

    Only meaningful on CUDA.
    """
    check = f"{neuron_name}/fp32_state_with_bf16_input"
    if device.type != "cuda":
        report.skip(check, "CUDA not available -- fp32 state check skipped")
        return

    t0 = time.perf_counter()
    try:
        neuron = build_neuron(neuron_cls, ctor_kwargs, device)
        if neuron is None:
            report.fail(check, "Failed to construct neuron")
            return

        n_features = ctor_kwargs.get("size", 16)
        x_bf16 = torch.randn(
            batch_size, n_features, device=device, dtype=torch.bfloat16
        )

        try:
            spk, mem = neuron_forward(neuron, x_bf16)
        except RuntimeError as exc:
            report.skip(check, f"bf16 not supported by this neuron: {exc}")
            return

        elapsed = (time.perf_counter() - t0) * 1000.0
        issues = []

        internal_mem = getattr(neuron, "mem", None)
        if internal_mem is not None and internal_mem.dtype != torch.float32:
            issues.append(
                f"neuron.mem dtype={internal_mem.dtype} -- expected float32"
            )

        details = {
            "input_dtype": str(x_bf16.dtype),
            "spk_dtype": str(spk.dtype),
            "mem_dtype": str(mem.dtype),
            "internal_mem_dtype": (
                str(internal_mem.dtype)
                if internal_mem is not None else "N/A"
            ),
        }

        if issues:
            report.fail(check, "; ".join(issues), details=details, duration_ms=elapsed)
        else:
            report.pass_(
                check,
                "internal mem stays fp32 with bf16 input",
                details=details,
                duration_ms=elapsed,
            )

    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000.0
        report.fail(check, f"Exception: {exc}", duration_ms=elapsed)


# ---------------------------------------------------------------------------
# SECTION 6: CPU/CUDA parity validation
# ---------------------------------------------------------------------------

def validate_cpu_cuda_parity(
    neuron_cls: Type[nn.Module],
    neuron_name: str,
    ctor_kwargs: Dict[str, Any],
    report: ValidationReport,
    batch_size: int = 4,
    num_timesteps: int = 10,
    atol: float = 1e-5,
) -> None:
    """
    Validate that CPU and CUDA produce the same spike patterns and
    membrane potentials (within floating-point tolerance).

    Spikes are binary so they should match exactly.
    Membrane potentials may differ by a tiny amount due to fp rounding.
    """
    check = f"{neuron_name}/cpu_cuda_parity"

    if not torch.cuda.is_available():
        report.skip(check, "CUDA not available")
        return

    t0 = time.perf_counter()
    try:
        cpu_dev = torch.device("cpu")
        cuda_dev = torch.device("cuda")

        n_features = ctor_kwargs.get("size", 16)
        torch.manual_seed(12345)
        x_cpu = torch.randn(batch_size, n_features)
        x_cuda = x_cpu.to(cuda_dev)

        neuron_cpu = build_neuron(neuron_cls, ctor_kwargs, cpu_dev)
        neuron_cuda = build_neuron(neuron_cls, ctor_kwargs, cuda_dev)

        if neuron_cpu is None or neuron_cuda is None:
            report.fail(check, "Failed to construct neuron on CPU or CUDA")
            return

        # Synchronise weights so only device differs
        cpu_sd = neuron_cpu.state_dict()
        cuda_sd = {k: v.to(cuda_dev) for k, v in cpu_sd.items()}
        neuron_cuda.load_state_dict(cuda_sd)

        neuron_cpu.reset_mem()
        neuron_cuda.reset_mem()

        spk_cpu_list: List[torch.Tensor] = []
        mem_cpu_list: List[torch.Tensor] = []
        spk_cuda_list: List[torch.Tensor] = []
        mem_cuda_list: List[torch.Tensor] = []

        for _ in range(num_timesteps):
            sp, me = neuron_forward(neuron_cpu, x_cpu)
            spk_cpu_list.append(sp.detach())
            mem_cpu_list.append(me.detach())

            sp, me = neuron_forward(neuron_cuda, x_cuda)
            spk_cuda_list.append(sp.detach().cpu())
            mem_cuda_list.append(me.detach().cpu())

        elapsed = (time.perf_counter() - t0) * 1000.0

        spk_cpu_t = torch.stack(spk_cpu_list)
        spk_cuda_t = torch.stack(spk_cuda_list)
        mem_cpu_t = torch.stack(mem_cpu_list)
        mem_cuda_t = torch.stack(mem_cuda_list)

        issues = []

        if not torch.equal(spk_cpu_t, spk_cuda_t):
            mismatch = int((spk_cpu_t != spk_cuda_t).sum().item())
            issues.append(
                f"Spike mismatch CPU vs CUDA: {mismatch} elements differ"
            )

        mem_max_diff = float((mem_cpu_t - mem_cuda_t).abs().max().item())
        if mem_max_diff > atol:
            issues.append(
                f"Membrane max abs diff CPU vs CUDA = "
                f"{mem_max_diff:.2e} > atol={atol:.2e}"
            )

        details = {
            "spike_match": torch.equal(spk_cpu_t, spk_cuda_t),
            "mem_max_diff": round(mem_max_diff, 8),
            "atol": atol,
            "timesteps": num_timesteps,
        }

        if issues:
            report.fail(check, "; ".join(issues), details=details, duration_ms=elapsed)
        else:
            report.pass_(
                check,
                f"CPU/CUDA parity OK; mem_max_diff={mem_max_diff:.2e}",
                details=details,
                duration_ms=elapsed,
            )

    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000.0
        report.fail(check, f"Exception: {exc}", duration_ms=elapsed)


# ---------------------------------------------------------------------------
# SECTION 7: Debug surface validation
# ---------------------------------------------------------------------------

def validate_debug_surfaces(
    neuron_cls: Type[nn.Module],
    neuron_name: str,
    ctor_kwargs: Dict[str, Any],
    device: torch.device,
    report: ValidationReport,
    batch_size: int = 4,
    num_timesteps: int = 20,
) -> None:
    """
    Validate debug/introspection surfaces:

    1. Firing rate is in [0, 1] and finite after running T timesteps.
    2. Membrane potential statistics (min, max, mean) are finite.
    3. AdvancedLIF: get_delay_distribution() and get_effective_delays()
       return valid tensors.
    """
    check = f"{neuron_name}/debug_surfaces"
    t0 = time.perf_counter()
    try:
        neuron = build_neuron(neuron_cls, ctor_kwargs, device)
        if neuron is None:
            report.fail(check, "Failed to construct neuron")
            return

        n_features = ctor_kwargs.get("size", 16)
        x = torch.randn(batch_size, n_features, device=device)
        neuron.reset_mem()

        all_spk = []
        for _ in range(num_timesteps):
            spk, _ = neuron_forward(neuron, x)
            all_spk.append(spk.detach())

        elapsed = (time.perf_counter() - t0) * 1000.0
        issues = []

        spike_tensor = torch.stack(all_spk)   # (T, B, N)
        firing_rate = float(spike_tensor.mean().item())

        if not (0.0 <= firing_rate <= 1.0):
            issues.append(f"firing_rate={firing_rate:.4f} out of [0, 1]")
        if math.isnan(firing_rate) or math.isinf(firing_rate):
            issues.append("firing_rate is NaN or Inf")

        # Membrane stats
        internal_mem = getattr(neuron, "mem", None)
        if internal_mem is not None:
            for stat_name, stat_val in [
                ("min", float(internal_mem.min().item())),
                ("max", float(internal_mem.max().item())),
                ("mean", float(internal_mem.mean().item())),
            ]:
                if math.isnan(stat_val) or math.isinf(stat_val):
                    issues.append(f"membrane {stat_name} is {stat_val}")

        # AdvancedLIF delay surfaces
        if hasattr(neuron, "get_delay_distribution"):
            try:
                dist = neuron.get_delay_distribution()
                if dist is not None:
                    if torch.isnan(dist).any():
                        issues.append("get_delay_distribution() returned NaN")
                    row_sums = dist.sum(dim=-1)
                    if not torch.allclose(
                        row_sums, torch.ones_like(row_sums), atol=1e-5
                    ):
                        issues.append(
                            f"delay distribution rows do not sum to 1: "
                            f"{row_sums[:3].tolist()}"
                        )
            except Exception as exc:
                issues.append(f"get_delay_distribution() raised: {exc}")

        if hasattr(neuron, "get_effective_delays"):
            try:
                eff = neuron.get_effective_delays()
                if eff is not None:
                    if torch.isnan(eff).any() or torch.isinf(eff).any():
                        issues.append("get_effective_delays() returned NaN/Inf")
            except Exception as exc:
                issues.append(f"get_effective_delays() raised: {exc}")

        details = {
            "firing_rate": round(firing_rate, 4),
            "timesteps": num_timesteps,
            "has_delay_surface": hasattr(neuron, "get_delay_distribution"),
        }

        if issues:
            report.fail(check, "; ".join(issues), details=details, duration_ms=elapsed)
        else:
            report.pass_(
                check,
                f"all debug surfaces valid; firing_rate={firing_rate:.3f}",
                details=details,
                duration_ms=elapsed,
            )

    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000.0
        report.fail(check, f"Exception: {exc}", duration_ms=elapsed)


# ---------------------------------------------------------------------------
# Additional: Surrogate function contract validation
# ---------------------------------------------------------------------------

def validate_surrogate_contracts(
    components: Dict[str, Any],
    device: torch.device,
    report: ValidationReport,
) -> None:
    """
    Validate each surrogate gradient function independently:
      - forward output is binary {0, 1}
      - sur(0) == 1.0  (Heaviside convention: >= 0 fires)
      - backward gradient is finite and non-negative
    """
    get_surrogate_fn = components.get("get_surrogate")
    if get_surrogate_fn is None:
        report.skip("surrogates/contract", "get_surrogate not available")
        return

    for sur_name in components.get("surrogates", []):
        check = f"surrogates/{sur_name}/contract"
        t0 = time.perf_counter()
        try:
            sur_fn = get_surrogate_fn(sur_name)

            x = torch.linspace(-2.0, 2.0, 200, device=device, requires_grad=True)
            y = sur_fn(x)

            issues = []

            unique = y.unique()
            non_binary = unique[~((unique == 0) | (unique == 1))]
            if len(non_binary) > 0:
                issues.append(
                    f"forward output contains non-binary values: "
                    f"{non_binary.tolist()}"
                )

            # Boundary check: sur(0) must equal 1.0
            x_zero = torch.zeros(1, device=device, requires_grad=True)
            y_zero = sur_fn(x_zero)
            if float(y_zero.item()) != 1.0:
                issues.append(
                    f"sur(0) = {y_zero.item()} != 1.0 (Heaviside at boundary)"
                )

            # Backward gradient
            grad_out = torch.ones_like(y)
            y.backward(grad_out, retain_graph=False)

            if x.grad is None:
                issues.append("backward produced None gradient")
            else:
                if torch.isnan(x.grad).any():
                    issues.append("backward gradient contains NaN")
                if torch.isinf(x.grad).any():
                    issues.append("backward gradient contains Inf")
                if x.grad.min().item() < -1e-6:
                    issues.append(
                        f"backward gradient has negative values "
                        f"(min={x.grad.min().item():.4f})"
                    )

            elapsed = (time.perf_counter() - t0) * 1000.0
            details = {
                "surrogate": sur_name,
                "unique_fwd_vals": unique.tolist(),
                "grad_min": float(x.grad.min().item())
                if x.grad is not None else None,
                "grad_max": float(x.grad.max().item())
                if x.grad is not None else None,
            }

            if issues:
                report.fail(check, "; ".join(issues), details=details,
                            duration_ms=elapsed)
            else:
                report.pass_(
                    check,
                    "binary fwd, finite non-negative bwd gradient",
                    details=details,
                    duration_ms=elapsed,
                )

        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000.0
            report.fail(check, f"Exception: {exc}", duration_ms=elapsed)


# ---------------------------------------------------------------------------
# Additional: Learnable-parameter gradient validation
# ---------------------------------------------------------------------------

def validate_learnable_params(
    neuron_cls: Type[nn.Module],
    neuron_name: str,
    ctor_kwargs: Dict[str, Any],
    device: torch.device,
    report: ValidationReport,
    batch_size: int = 4,
    num_timesteps: int = 5,
) -> None:
    """
    Validate that learnable parameters (beta, threshold, log_beta,
    delay_weights, etc.) receive non-zero gradients when enabled.
    """
    check = f"{neuron_name}/learnable_params"
    t0 = time.perf_counter()
    try:
        kw = dict(ctor_kwargs)

        # Enable any learnable variant params that the constructor accepts
        for flag in ("learn_beta", "learn_threshold", "learnable_beta"):
            if _has_param(neuron_cls, flag):
                kw[flag] = True

        neuron = build_neuron(neuron_cls, kw, device)
        if neuron is None:
            report.fail(check, "Failed to construct neuron with learnable params")
            return

        learnable = {
            name: param
            for name, param in neuron.named_parameters()
            if param.requires_grad
        }

        if not learnable:
            report.skip(check, "No learnable parameters in this configuration")
            return

        n_features = kw.get("size", 16)
        neuron.zero_grad()
        x = torch.randn(batch_size, n_features, device=device)

        accum = torch.tensor(0.0, device=device)
        for _ in range(num_timesteps):
            spk, mem = neuron_forward(neuron, x)
            accum = accum + spk.sum() + mem.sum()

        accum.backward()

        elapsed = (time.perf_counter() - t0) * 1000.0
        issues = []
        zero_grads = []

        for pname, param in learnable.items():
            if param.grad is None:
                issues.append(f"{pname}.grad is None")
            elif param.grad.abs().sum().item() == 0:
                zero_grads.append(pname)

        details = {
            "learnable_params": list(learnable.keys()),
            "zero_grad_params": zero_grads,
        }

        if issues:
            report.fail(check, "; ".join(issues), details=details, duration_ms=elapsed)
        elif zero_grads:
            report.fail(
                check,
                f"Learnable params with all-zero gradient: {zero_grads}",
                details=details,
                duration_ms=elapsed,
            )
        else:
            report.pass_(
                check,
                f"all learnable params received non-zero gradients: "
                f"{list(learnable.keys())}",
                details=details,
                duration_ms=elapsed,
            )

    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000.0
        report.fail(check, f"Exception: {exc}", duration_ms=elapsed)


# ---------------------------------------------------------------------------
# Additional: Full network time-unrolling consistency
# ---------------------------------------------------------------------------

def validate_time_unrolling(
    components: Dict[str, Any],
    device: torch.device,
    report: ValidationReport,
    batch_size: int = 4,
    num_timesteps: int = 25,
) -> None:
    """
    Validate SNNCore:
      1. Full built-in unroll == manual step-by-step (spike-for-spike)
      2. Truncated BPTT (detach between chunks) still produces gradients
    """
    check_full = "snn_core/full_vs_step_consistency"
    check_tbptt = "snn_core/truncated_bptt"

    t0 = time.perf_counter()
    try:
        from brain_ai.core.snn import SNNCore

        snn = SNNCore(
            input_size=32,
            hidden_sizes=[64],
            output_size=16,
            num_steps=num_timesteps,
        ).to(device)
        snn.eval()

        torch.manual_seed(5)
        x_static = torch.randn(batch_size, 32, device=device)

        # Full forward (built-in unroll)
        with torch.no_grad():
            spike_record_full, _mem_full = snn(x_static)

        # Manual step-by-step
        snn.reset_mem()
        x_repeated = x_static.unsqueeze(0).repeat(num_timesteps, 1, 1)
        spk_manual = []
        with torch.no_grad():
            for t in range(num_timesteps):
                spk, _ = snn.forward_step(x_repeated[t])
                spk_manual.append(spk)
        spike_record_manual = torch.stack(spk_manual)

        elapsed = (time.perf_counter() - t0) * 1000.0
        issues = []

        if spike_record_full.shape != spike_record_manual.shape:
            issues.append(
                f"shape mismatch: full={spike_record_full.shape}, "
                f"manual={spike_record_manual.shape}"
            )
        elif not torch.equal(spike_record_full, spike_record_manual):
            n_diff = int((spike_record_full != spike_record_manual).sum().item())
            issues.append(
                f"full vs manual spike mismatch: {n_diff} elements differ"
            )

        details = {
            "spike_shape": list(spike_record_full.shape),
            "timesteps": num_timesteps,
        }

        if issues:
            report.fail(check_full, "; ".join(issues), details=details,
                        duration_ms=elapsed)
        else:
            report.pass_(
                check_full,
                "full unroll and step-by-step match exactly",
                details=details,
                duration_ms=elapsed,
            )

    except ImportError:
        elapsed = (time.perf_counter() - t0) * 1000.0
        report.skip(check_full, "SNNCore not importable")
    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000.0
        report.fail(check_full, f"Exception: {exc}", duration_ms=elapsed)

    # Truncated BPTT
    t0 = time.perf_counter()
    try:
        from brain_ai.core.snn import SNNCore

        chunk = max(num_timesteps // 2, 1)
        snn_tbptt = SNNCore(
            input_size=32,
            hidden_sizes=[64],
            output_size=16,
            num_steps=chunk,
        ).to(device)

        x_in = torch.randn(batch_size, 32, device=device)

        # Chunk 1
        snn_tbptt.reset_mem()
        for _ in range(chunk):
            snn_tbptt.forward_step(x_in)

        # Detach all internal neuron state
        for layer in snn_tbptt.layers:
            detach_neuron_state(layer.lif)

        # Chunk 2 -- should still propagate gradients back to weights
        spk_c2 = []
        for _ in range(chunk):
            spk, _ = snn_tbptt.forward_step(x_in)
            spk_c2.append(spk)

        loss = torch.stack(spk_c2).sum()
        loss.backward()

        elapsed = (time.perf_counter() - t0) * 1000.0

        grads_found = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in snn_tbptt.parameters()
        )

        if not grads_found:
            report.fail(
                check_tbptt,
                "No non-zero gradients found after truncated BPTT backward",
                duration_ms=elapsed,
            )
        else:
            report.pass_(
                check_tbptt,
                "truncated BPTT backward succeeds with detached state",
                duration_ms=elapsed,
            )

    except ImportError:
        elapsed = (time.perf_counter() - t0) * 1000.0
        report.skip(check_tbptt, "SNNCore not importable")
    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000.0
        report.fail(
            check_tbptt,
            f"Exception: {exc}\n{traceback.format_exc()}",
            duration_ms=elapsed,
        )


# ---------------------------------------------------------------------------
# SECTION 8: Orchestrator
# ---------------------------------------------------------------------------

def run_all_validations(args: argparse.Namespace) -> ValidationReport:
    """Run all validation checks and return the populated report."""
    report = ValidationReport()

    # Resolve device
    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("[WARNING] --device cuda requested but CUDA not available. Using CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device(args.device)

    # Import spiking core
    components = import_spiking_core()
    if components is None:
        report.fail(
            "_global/import",
            f"Could not import brain_ai.core. "
            f"Searched repo root: {_REPO_ROOT}",
        )
        return report

    report.pass_(
        "_global/import",
        f"brain_ai.core imported (api={components['api']})",
        details={"repo_root": _REPO_ROOT, "api": components["api"]},
    )

    # Validate surrogate function contracts independently
    validate_surrogate_contracts(components, device, report)

    # Resolve neuron list
    available_neurons = components["neurons"]
    neuron_names = list(available_neurons.keys())

    if args.neuron:
        if args.neuron not in available_neurons:
            report.fail(
                "_global/neuron_selection",
                f"Requested neuron '{args.neuron}' not in available: "
                f"{neuron_names}",
            )
            return report
        neuron_names = [args.neuron]

    batch_size = args.batch_size
    num_timesteps = args.timesteps

    for neuron_name in neuron_names:
        neuron_cls, ctor_kwargs = available_neurons[neuron_name]

        # 1. Forward shape contract
        validate_neuron_forward(
            neuron_cls, neuron_name, ctor_kwargs, device, report, batch_size
        )

        # 2. State shape contract
        validate_state_shapes(
            neuron_cls, neuron_name, ctor_kwargs, device, report, batch_size
        )

        # 3. Gradient presence -- one check per surrogate
        for sur_name in components["surrogates"]:
            validate_gradient_presence(
                neuron_cls, neuron_name, ctor_kwargs,
                sur_name, device, report,
                batch_size=batch_size,
                num_timesteps=min(num_timesteps, 10),
            )

        # 4. Surrogate swap produces distinct gradient norms
        validate_surrogate_swap(
            neuron_cls, neuron_name, ctor_kwargs, device, report,
            batch_size=batch_size,
        )

        # 5. Reset/carry behaviour
        validate_state_reset_carry(
            neuron_cls, neuron_name, ctor_kwargs, device, report,
            batch_size=batch_size,
            num_timesteps=min(num_timesteps, 15),
        )

        # 6. State detach (truncated BPTT chunk boundary)
        validate_state_detach(
            neuron_cls, neuron_name, ctor_kwargs, device, report,
            batch_size=batch_size,
        )

        # 7. Out-of-range beta stability
        validate_beta_bounds(
            neuron_cls, neuron_name, ctor_kwargs, device, report, batch_size
        )

        # 8. Long unroll numerical stability
        validate_long_unroll(
            neuron_cls, neuron_name, ctor_kwargs, device, report,
            batch_size=batch_size,
            num_timesteps=200,
        )

        # 9. Learnable parameter gradients
        validate_learnable_params(
            neuron_cls, neuron_name, ctor_kwargs, device, report,
            batch_size=batch_size,
            num_timesteps=min(num_timesteps, 5),
        )

        # 10. CUDA-only checks
        if torch.cuda.is_available() and device.type != "cpu":
            cuda_device = torch.device("cuda")
            validate_fp32_state(
                neuron_cls, neuron_name, ctor_kwargs, cuda_device, report,
                batch_size
            )
            validate_cpu_cuda_parity(
                neuron_cls, neuron_name, ctor_kwargs, report,
                batch_size=batch_size,
                num_timesteps=min(num_timesteps, 10),
            )
        else:
            # Still record skip entries so they appear in the report
            validate_fp32_state(
                neuron_cls, neuron_name, ctor_kwargs, device, report, batch_size
            )

        # 11. Debug surfaces
        validate_debug_surfaces(
            neuron_cls, neuron_name, ctor_kwargs, device, report,
            batch_size=batch_size,
            num_timesteps=min(num_timesteps, 20),
        )

    # 12. SNNCore time-unrolling consistency + truncated BPTT
    validate_time_unrolling(
        components, device, report,
        batch_size=batch_size,
        num_timesteps=num_timesteps,
    )

    return report


# ---------------------------------------------------------------------------
# SECTION 9: CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate spiking core runtime contracts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--neuron",
        choices=["lif", "adaptive_lif", "recurrent_lif", "advanced_lif"],
        default=None,
        help="Validate a single neuron type (default: all)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device: cpu, cuda, cuda:0, etc. (default: cpu)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON to stdout",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed pass/fail messages",
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
        default=25,
        metavar="T",
        help="Number of timesteps for time-unrolling tests (default: 25)",
    )
    parser.add_argument(
        "--no-warnings",
        action="store_true",
        help="Suppress Python warnings during validation",
    )
    args = parser.parse_args()

    if args.no_warnings:
        warnings.filterwarnings("ignore")

    if not args.json:
        print("Spiking core validation starting...")
        print(f"  repo root  : {_REPO_ROOT}")
        print(f"  device     : {args.device}")
        print(f"  neuron     : {args.neuron or 'all'}")
        print(f"  batch_size : {args.batch_size}")
        print(f"  timesteps  : {args.timesteps}")
        print(f"  torch      : {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  cuda       : {torch.cuda.get_device_name(0)}")
        else:
            print(f"  cuda       : not available")

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
