"""
Precision + Numerics Stabilizer - NumericsMonitor Template
==========================================================
NumericsMonitor with all sentinel checks: gradient norms, activation hooks,
logit scale monitoring, and weight NaN/Inf detection.

CRITICAL: Never call the inference-mode shorthand method on PyTorch modules.
Use module.train(False) instead.
"""

from __future__ import annotations

import fnmatch
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

try:
    from precision_config_template import SentinelConfig
except ImportError:
    try:
        from assets.precision_config_template import SentinelConfig
    except ImportError:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(__file__))
        from precision_config_template import SentinelConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GradNormReport:
    """Gradient norm sentinel output."""
    global_norm: float
    per_module_topk: List[Tuple[str, float]]   # (module_prefix, norm), sorted desc
    max_param_name: str
    max_param_norm: float
    step: int
    is_finite: bool

    def to_dict(self) -> dict:
        return {
            "global_norm": self.global_norm,
            "per_module_topk": self.per_module_topk,
            "max_param_name": self.max_param_name,
            "max_param_norm": self.max_param_norm,
            "step": self.step,
            "is_finite": self.is_finite,
        }


@dataclass
class LayerReport:
    """Per-layer activation check result."""
    name: str
    is_finite: bool
    max_abs: float
    mean: float
    std: float


@dataclass
class ActivationReport:
    """Activation NaN/Inf sentinel output."""
    layer_reports: List[LayerReport]
    any_nonfinite: bool
    first_nonfinite_name: Optional[str]
    step: int

    def to_dict(self) -> dict:
        return {
            "layer_reports": [
                {
                    "name": lr.name,
                    "is_finite": lr.is_finite,
                    "max_abs": lr.max_abs,
                    "mean": lr.mean,
                    "std": lr.std,
                }
                for lr in self.layer_reports
            ],
            "any_nonfinite": self.any_nonfinite,
            "first_nonfinite_name": self.first_nonfinite_name,
            "step": self.step,
        }


@dataclass
class LogitReport:
    """Logit scale sentinel output."""
    max_abs: float
    std: float
    exceeds_threshold: bool
    consecutive_violations: int
    step: int

    def to_dict(self) -> dict:
        return {
            "max_abs": self.max_abs,
            "std": self.std,
            "exceeds_threshold": self.exceeds_threshold,
            "consecutive_violations": self.consecutive_violations,
            "step": self.step,
        }


@dataclass
class WeightReport:
    """Weight NaN/Inf sentinel output."""
    all_finite: bool
    first_nonfinite_name: Optional[str]
    total_params_checked: int
    step: int

    def to_dict(self) -> dict:
        return {
            "all_finite": self.all_finite,
            "first_nonfinite_name": self.first_nonfinite_name,
            "total_params_checked": self.total_params_checked,
            "step": self.step,
        }


@dataclass
class NumericsReport:
    """Aggregate report from all sentinel checks at a single step."""
    step: int
    timestamp: float
    grad_norm_report: Optional[GradNormReport] = None
    activation_report: Optional[ActivationReport] = None
    logit_report: Optional[LogitReport] = None
    weight_report: Optional[WeightReport] = None

    def any_anomaly(self) -> bool:
        """Return True if any sentinel found a problem."""
        if self.activation_report and self.activation_report.any_nonfinite:
            return True
        if self.logit_report and self.logit_report.consecutive_violations > 0:
            return True
        if self.weight_report and not self.weight_report.all_finite:
            return True
        if self.grad_norm_report and not self.grad_norm_report.is_finite:
            return True
        return False

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "timestamp": self.timestamp,
            "grad_norm_report": self.grad_norm_report.to_dict() if self.grad_norm_report else None,
            "activation_report": self.activation_report.to_dict() if self.activation_report else None,
            "logit_report": self.logit_report.to_dict() if self.logit_report else None,
            "weight_report": self.weight_report.to_dict() if self.weight_report else None,
        }


# ---------------------------------------------------------------------------
# Activation Capture Hook
# ---------------------------------------------------------------------------

class _ActivationCapture:
    """Stores forward hook outputs indexed by layer name."""

    def __init__(self):
        self.outputs: Dict[str, torch.Tensor] = {}

    def make_hook(self, name: str):
        """Return a forward hook function that stores the output tensor."""
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self.outputs[name] = output.detach()
            elif isinstance(output, (tuple, list)):
                # Modules like MultiheadAttention return (attn_output, attn_weights)
                tensors = [o for o in output if isinstance(o, torch.Tensor)]
                if tensors:
                    self.outputs[name] = tensors[0].detach()
        return hook

    def clear(self):
        self.outputs.clear()


# ---------------------------------------------------------------------------
# NumericsMonitor
# ---------------------------------------------------------------------------

class NumericsMonitor:
    """All-in-one numerics sentinel for training stability.

    Checks
    ------
    C1: Gradient norms (global, per-module top-k, max individual param)
    C2: Activation NaN/Inf via forward hooks on pattern-matched layers
    C3: Logit max abs and consecutive threshold violations
    C4: Weight NaN/Inf sweep

    Parameters
    ----------
    cfg : SentinelConfig
        Sentinel configuration (cadence, patterns, thresholds).
    model : nn.Module
        The model to monitor. Must be the unwrapped model (not DDP wrapper).

    Notes
    -----
    Call register_hooks() once before training begins, and remove_hooks()
    when monitoring is no longer needed to prevent memory leaks.

    IMPORTANT: When switching a model to inference mode, always call
    module.train(False) rather than using the deprecated shorthand.
    """

    def __init__(self, cfg: SentinelConfig, model: nn.Module):
        self.cfg = cfg
        self.model = model

        self._capture = _ActivationCapture()
        self._hook_handles: List = []
        self._consecutive_logit_violations: int = 0
        self._force_check_next: bool = False

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def register_hooks(self) -> int:
        """Register forward hooks on modules matching configured glob patterns.

        Returns
        -------
        int
            Number of hooks registered.
        """
        self.remove_hooks()  # Clean up any existing hooks first

        registered = 0
        for name, module in self.model.named_modules():
            if not name:  # skip root module
                continue
            if self._matches_any_pattern(name):
                handle = module.register_forward_hook(self._capture.make_hook(name))
                self._hook_handles.append(handle)
                registered += 1

        logger.debug("NumericsMonitor: registered %d activation hooks.", registered)
        return registered

    def remove_hooks(self):
        """Remove all registered forward hooks (prevents memory leaks)."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._capture.clear()

    def _matches_any_pattern(self, name: str) -> bool:
        """Return True if name matches any configured glob pattern."""
        return any(
            fnmatch.fnmatch(name, pattern)
            for pattern in self.cfg.nan_check_sample_layers
        )

    # ------------------------------------------------------------------
    # Cadence check
    # ------------------------------------------------------------------

    def should_check(self, step: int) -> bool:
        """Return True if sentinels should run at this step."""
        if self._force_check_next:
            self._force_check_next = False
            return True
        return step % self.cfg.every_n_steps == 0

    # ------------------------------------------------------------------
    # C1: Gradient Norm Sentinel
    # ------------------------------------------------------------------

    def check_grad_norms(self, step: int = 0) -> GradNormReport:
        """Compute gradient norms: global, per-module top-k, max param.

        Must be called after backward() and before optimizer.step().

        Parameters
        ----------
        step : int
            Current training step (for report annotation).

        Returns
        -------
        GradNormReport
        """
        module_norms_sq: Dict[str, float] = defaultdict(float)
        param_norms: List[Tuple[str, float]] = []

        total_norm_sq = 0.0

        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            grad_f = param.grad.detach().float()
            norm_sq = grad_f.norm(2).item() ** 2
            total_norm_sq += norm_sq

            # Individual param norm
            param_norms.append((name, norm_sq ** 0.5))

            # Group by module prefix (everything except the last ".xxx" segment)
            parts = name.split(".")
            prefix = ".".join(parts[:-1]) if len(parts) > 1 else name
            module_norms_sq[prefix] += norm_sq

        global_norm = total_norm_sq ** 0.5
        is_finite = math.isfinite(global_norm)

        # Per-module norms sorted descending
        per_module = [
            (k, v ** 0.5) for k, v in module_norms_sq.items()
        ]
        per_module.sort(key=lambda x: x[1], reverse=True)
        top_k = per_module[: self.cfg.grad_norm_topk]

        # Max individual param
        if param_norms:
            max_param_name, max_param_norm = max(param_norms, key=lambda x: x[1])
        else:
            max_param_name, max_param_norm = "none", 0.0

        report = GradNormReport(
            global_norm=global_norm,
            per_module_topk=top_k,
            max_param_name=max_param_name,
            max_param_norm=max_param_norm,
            step=step,
            is_finite=is_finite,
        )

        if not is_finite:
            logger.error(
                "[Step %d] Non-finite global grad norm: %s. "
                "Max param: %s (%.4f).",
                step,
                global_norm,
                max_param_name,
                max_param_norm,
            )
            self._force_check_next = True

        return report

    # ------------------------------------------------------------------
    # C2: Activation Sentinel
    # ------------------------------------------------------------------

    def check_activations(self, step: int = 0) -> ActivationReport:
        """Check all hooked activation tensors for NaN/Inf.

        Uses cached outputs from the most recent forward pass.

        Parameters
        ----------
        step : int
            Current training step.

        Returns
        -------
        ActivationReport
        """
        layer_reports: List[LayerReport] = []
        any_nonfinite = False
        first_nonfinite: Optional[str] = None

        for name, tensor in list(self._capture.outputs.items()):
            try:
                t_f = tensor.float()
                is_finite = bool(torch.isfinite(t_f).all().item())
                if is_finite:
                    max_abs = float(t_f.abs().max().item())
                    mean = float(t_f.mean().item())
                    std = float(t_f.std().item())
                else:
                    max_abs = float("inf")
                    mean = float("nan")
                    std = float("nan")
            except Exception:
                is_finite = False
                max_abs = float("inf")
                mean = float("nan")
                std = float("nan")

            layer_reports.append(LayerReport(
                name=name,
                is_finite=is_finite,
                max_abs=max_abs,
                mean=mean,
                std=std,
            ))

            if not is_finite:
                any_nonfinite = True
                if first_nonfinite is None:
                    first_nonfinite = name

        if any_nonfinite:
            logger.error(
                "[Step %d] Non-finite activation detected in layer: %s. "
                "Triggering immediate full scan.",
                step,
                first_nonfinite,
            )
            self._force_check_next = True
            self._run_full_scan(step)

        return ActivationReport(
            layer_reports=layer_reports,
            any_nonfinite=any_nonfinite,
            first_nonfinite_name=first_nonfinite,
            step=step,
        )

    def _run_full_scan(self, step: int):
        """Run a full weight check across all parameters (called on anomaly)."""
        logger.warning("[Step %d] Running full weight scan after anomaly.", step)
        report = self.check_weights(step=step)
        if not report.all_finite:
            logger.error(
                "[Step %d] FULL SCAN: Non-finite weight found: %s",
                step,
                report.first_nonfinite_name,
            )

    # ------------------------------------------------------------------
    # C3: Logit Sentinel
    # ------------------------------------------------------------------

    def check_logits(self, logits: torch.Tensor, step: int = 0) -> LogitReport:
        """Monitor logit scale for overflow risk.

        Parameters
        ----------
        logits : torch.Tensor
            Raw logit tensor from model output, before softmax/loss.
        step : int
            Current training step.

        Returns
        -------
        LogitReport
        """
        try:
            logits_f = logits.detach().float()
            max_abs = float(logits_f.abs().max().item())
            std = float(logits_f.std().item())
        except Exception:
            max_abs = float("nan")
            std = float("nan")

        exceeds = math.isfinite(max_abs) and max_abs > self.cfg.logit_max_abs_threshold

        if exceeds:
            self._consecutive_logit_violations += 1
        else:
            self._consecutive_logit_violations = 0

        if self._consecutive_logit_violations >= self.cfg.logit_alert_consecutive:
            logger.warning(
                "[Step %d] Logit overflow risk: max_abs=%.2f for %d consecutive steps. "
                "Threshold=%.1f. Immediate sentinel check triggered.",
                step,
                max_abs,
                self._consecutive_logit_violations,
                self.cfg.logit_max_abs_threshold,
            )
            self._force_check_next = True

        return LogitReport(
            max_abs=max_abs,
            std=std,
            exceeds_threshold=exceeds,
            consecutive_violations=self._consecutive_logit_violations,
            step=step,
        )

    # ------------------------------------------------------------------
    # C4: Weight Sentinel
    # ------------------------------------------------------------------

    def check_weights(self, step: int = 0) -> WeightReport:
        """Check all model parameters for NaN/Inf.

        Parameters
        ----------
        step : int
            Current training step.

        Returns
        -------
        WeightReport
        """
        first_nonfinite: Optional[str] = None
        checked = 0

        for name, param in self.model.named_parameters():
            checked += 1
            if not torch.isfinite(param.data).all():
                if first_nonfinite is None:
                    first_nonfinite = name
                    logger.error(
                        "[Step %d] Non-finite weight found: %s", step, name
                    )

        return WeightReport(
            all_finite=(first_nonfinite is None),
            first_nonfinite_name=first_nonfinite,
            total_params_checked=checked,
            step=step,
        )

    # ------------------------------------------------------------------
    # Aggregate Report
    # ------------------------------------------------------------------

    def aggregate_report(self, step: int) -> NumericsReport:
        """Run all available sentinels and return a combined report.

        Parameters
        ----------
        step : int
            Current training step.

        Returns
        -------
        NumericsReport
        """
        grad_report = self.check_grad_norms(step=step)
        act_report = self.check_activations(step=step)
        weight_report = self.check_weights(step=step)

        return NumericsReport(
            step=step,
            timestamp=time.time(),
            grad_norm_report=grad_report,
            activation_report=act_report,
            weight_report=weight_report,
        )


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("Running numerics_monitor_template.py self-tests...")
    failures = []

    # Helper: small 2-layer model
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 8)
            self.fc2 = nn.Linear(8, 4)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    cfg = SentinelConfig(every_n_steps=10, logit_max_abs_threshold=50.0, logit_alert_consecutive=3)

    # --- T1: Grad norm computation on known model ---
    try:
        model = TinyModel()
        monitor = NumericsMonitor(cfg, model)

        # Set known gradients: all 1.0
        for p in model.parameters():
            p.grad = torch.ones_like(p)

        total_ones = sum(p.numel() for p in model.parameters())
        expected_norm = float(total_ones) ** 0.5

        report = monitor.check_grad_norms(step=0)
        assert math.isclose(report.global_norm, expected_norm, rel_tol=1e-4), (
            f"Expected global_norm={expected_norm:.4f}, got {report.global_norm:.4f}"
        )
        assert report.is_finite is True
        print(f"  [PASS] T1: Grad norm computation correct (global_norm={report.global_norm:.4f})")
    except Exception as e:
        failures.append(f"T1 grad norm: {e}")
        print(f"  [FAIL] T1: {e}")

    # --- T2: Per-module top-k grouping ---
    try:
        model = TinyModel()
        monitor = NumericsMonitor(cfg, model)

        # fc1 grads = 1.0, fc2 grads = 2.0 -> fc2 should be top
        for name, p in model.named_parameters():
            if "fc2" in name:
                p.grad = torch.full_like(p, 2.0)
            else:
                p.grad = torch.ones_like(p)

        report = monitor.check_grad_norms(step=1)
        assert len(report.per_module_topk) >= 1
        top_name, top_norm = report.per_module_topk[0]
        assert "fc2" in top_name, f"Expected fc2 at top, got {top_name}"
        print(f"  [PASS] T2: Per-module top-k groups correctly (top: {top_name}, norm={top_norm:.4f})")
    except Exception as e:
        failures.append(f"T2 per-module topk: {e}")
        print(f"  [FAIL] T2: {e}")

    # --- T3: NaN detection in gradients ---
    try:
        model = TinyModel()
        monitor = NumericsMonitor(cfg, model)

        # Inject NaN into one gradient
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        # Overwrite first param's grad with NaN
        first_param = next(iter(model.parameters()))
        first_param.grad = torch.full_like(first_param, float("nan"))

        report = monitor.check_grad_norms(step=2)
        assert not report.is_finite, "Should detect non-finite global norm"
        print(f"  [PASS] T3: NaN in gradients detected (global_norm={report.global_norm})")
    except Exception as e:
        failures.append(f"T3 nan grad detection: {e}")
        print(f"  [FAIL] T3: {e}")

    # --- T4: Activation hook fires and captures output ---
    try:
        cfg_hook = SentinelConfig(
            every_n_steps=1,
            nan_check_sample_layers=("fc1",),
        )
        model = TinyModel()
        monitor = NumericsMonitor(cfg_hook, model)
        n_hooks = monitor.register_hooks()
        assert n_hooks >= 1, f"Expected at least 1 hook, got {n_hooks}"

        # Run forward pass using train(False) - NOT the deprecated pattern
        x = torch.randn(2, 8)
        model.train(False)
        with torch.no_grad():
            _ = model(x)

        # Check capture
        assert "fc1" in monitor._capture.outputs, (
            f"Expected 'fc1' in capture, got keys: {list(monitor._capture.outputs.keys())}"
        )
        out = monitor._capture.outputs["fc1"]
        assert out.shape[0] == 2, f"Expected batch size 2, got {out.shape[0]}"
        monitor.remove_hooks()
        model.train(True)
        print(f"  [PASS] T4: Activation hook fires and captures output (shape={out.shape})")
    except Exception as e:
        failures.append(f"T4 activation hook: {e}")
        print(f"  [FAIL] T4: {e}")

    # --- T5: NaN detection in activations ---
    try:
        cfg_hook = SentinelConfig(
            every_n_steps=1,
            nan_check_sample_layers=("fc1",),
        )
        model = TinyModel()
        monitor = NumericsMonitor(cfg_hook, model)
        monitor.register_hooks()

        # Inject NaN into fc1 weights to produce NaN activations
        with torch.no_grad():
            model.fc1.weight.fill_(float("nan"))

        x = torch.randn(2, 8)
        model.train(False)
        with torch.no_grad():
            _ = model(x)

        report = monitor.check_activations(step=3)
        assert report.any_nonfinite is True, "Should detect non-finite activation"
        assert report.first_nonfinite_name is not None
        monitor.remove_hooks()
        model.train(True)
        print(f"  [PASS] T5: NaN activation detected in {report.first_nonfinite_name}")
    except Exception as e:
        failures.append(f"T5 nan activation: {e}")
        print(f"  [FAIL] T5: {e}")

    # --- T6: Weight NaN detection ---
    try:
        model = TinyModel()
        monitor = NumericsMonitor(cfg, model)

        # Inject NaN into one weight
        with torch.no_grad():
            model.fc2.bias.fill_(float("nan"))

        report = monitor.check_weights(step=4)
        assert not report.all_finite, "Should detect non-finite weight"
        assert "fc2.bias" == report.first_nonfinite_name, (
            f"Expected fc2.bias, got {report.first_nonfinite_name}"
        )
        print(f"  [PASS] T6: Weight NaN detected at {report.first_nonfinite_name}")
    except Exception as e:
        failures.append(f"T6 weight nan: {e}")
        print(f"  [FAIL] T6: {e}")

    # --- T7: Logit max_abs check ---
    try:
        model = TinyModel()
        monitor = NumericsMonitor(cfg, model)

        logits = torch.tensor([[10.0, -20.0, 30.0]])
        report = monitor.check_logits(logits, step=5)
        assert math.isclose(report.max_abs, 30.0, rel_tol=1e-5), (
            f"Expected max_abs=30.0, got {report.max_abs}"
        )
        assert not report.exceeds_threshold, (
            f"30.0 should not exceed threshold {cfg.logit_max_abs_threshold}"
        )
        print(f"  [PASS] T7: Logit max_abs correct ({report.max_abs:.1f})")
    except Exception as e:
        failures.append(f"T7 logit max_abs: {e}")
        print(f"  [FAIL] T7: {e}")

    # --- T8: Logit consecutive violations counter ---
    try:
        model = TinyModel()
        cfg_logit = SentinelConfig(logit_max_abs_threshold=50.0, logit_alert_consecutive=3)
        monitor = NumericsMonitor(cfg_logit, model)

        high_logits = torch.tensor([[100.0, -100.0]])  # Exceeds threshold

        r1 = monitor.check_logits(high_logits, step=1)
        assert r1.consecutive_violations == 1
        r2 = monitor.check_logits(high_logits, step=2)
        assert r2.consecutive_violations == 2
        r3 = monitor.check_logits(high_logits, step=3)
        assert r3.consecutive_violations == 3

        # Now send normal logits - should reset
        normal_logits = torch.tensor([[1.0, -1.0]])
        r4 = monitor.check_logits(normal_logits, step=4)
        assert r4.consecutive_violations == 0, (
            f"Expected reset to 0, got {r4.consecutive_violations}"
        )
        print("  [PASS] T8: Logit consecutive violations counter increments and resets")
    except Exception as e:
        failures.append(f"T8 logit consecutive: {e}")
        print(f"  [FAIL] T8: {e}")

    # --- T9: should_check cadence ---
    try:
        model = TinyModel()
        cfg_cad = SentinelConfig(every_n_steps=10)
        monitor = NumericsMonitor(cfg_cad, model)

        assert monitor.should_check(0) is True
        assert monitor.should_check(5) is False
        assert monitor.should_check(10) is True
        assert monitor.should_check(15) is False
        assert monitor.should_check(20) is True
        print("  [PASS] T9: should_check cadence correct")
    except Exception as e:
        failures.append(f"T9 cadence: {e}")
        print(f"  [FAIL] T9: {e}")

    # --- T10: NumericsReport any_anomaly detection ---
    try:
        report = NumericsReport(
            step=10,
            timestamp=time.time(),
            weight_report=WeightReport(
                all_finite=False,
                first_nonfinite_name="fc1.weight",
                total_params_checked=4,
                step=10,
            ),
        )
        assert report.any_anomaly() is True

        clean_report = NumericsReport(
            step=11,
            timestamp=time.time(),
            weight_report=WeightReport(
                all_finite=True,
                first_nonfinite_name=None,
                total_params_checked=4,
                step=11,
            ),
        )
        assert clean_report.any_anomaly() is False
        print("  [PASS] T10: NumericsReport any_anomaly detection correct")
    except Exception as e:
        failures.append(f"T10 any_anomaly: {e}")
        print(f"  [FAIL] T10: {e}")

    print()
    if failures:
        print(f"FAILED: {len(failures)} test(s) failed:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All 10 self-tests passed.")
