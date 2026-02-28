"""
compare_baseline_template.py
=============================
CompareBaseline class for comparing current benchmark results against stored baselines
with configurable tolerances and structured pass/fail/warn reporting.

Usage:
    from compare_baseline_template import CompareBaseline
    from perf_gate_config_template import ToleranceConfig

    tol = ToleranceConfig(throughput_drop_pct=5.0, ppl_increase_pct=1.5)
    comparator = CompareBaseline(tol)
    result = comparator.compare(
        current_dir="artifacts/bench_results",
        baseline_dir="bench/baselines",
        machine_profile="H100x8_driver550_cuda12.4_torch2.4_sm90",
    )
    print(result.report)
    sys.exit(0 if result.passed else 1)

Self-test:
    python compare_baseline_template.py
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from perf_gate_config_template import (
        ToleranceConfig, CheckResult, CompareResult, Status
    )
except ImportError:
    from dataclasses import dataclass
    from enum import Enum

    class Status(str, Enum):
        PASS = "PASS"
        FAIL = "FAIL"
        WARN = "WARN"
        SKIP = "SKIP"

    @dataclass
    class ToleranceConfig:
        throughput_drop_pct: float = 5.0
        step_time_increase_pct: float = 5.0
        memory_increase_pct: float = 10.0
        ppl_increase_pct: float = 1.5
        probe_drop_abs: float = 2.0
        loss_slope_threshold: float = 0.001

        def validate(self):
            pass

    @dataclass
    class CheckResult:
        metric: str
        baseline_val: float
        current_val: float
        delta_pct: float
        threshold: float
        status: Status
        message: str

        @property
        def is_fail(self):
            return self.status == Status.FAIL

        @property
        def is_warn(self):
            return self.status == Status.WARN

        @property
        def is_pass(self):
            return self.status == Status.PASS

    @dataclass
    class CompareResult:
        overall_status: Status
        checks: List[CheckResult]
        report: str
        machine_profile: str

        @property
        def failed_checks(self):
            return [c for c in self.checks if c.status == Status.FAIL]

        @property
        def warned_checks(self):
            return [c for c in self.checks if c.status == Status.WARN]

        @property
        def passed(self):
            return self.overall_status in (Status.PASS, Status.WARN)

        @property
        def failed(self):
            return self.overall_status == Status.FAIL


# ===========================================================================
# Delta computation utility
# ===========================================================================

def compute_delta_pct(current: float, baseline: float) -> float:
    """
    Compute signed percentage change relative to baseline.

    Returns:
        Positive value means current > baseline.
        Negative value means current < baseline.
        Returns inf if baseline is 0 and current != 0.
        Returns 0 if both are 0.
    """
    if math.isnan(baseline) or math.isnan(current):
        return float("nan")
    if baseline == 0:
        return float("inf") if current != 0 else 0.0
    return ((current - baseline) / abs(baseline)) * 100.0


def _fmt_val(val: float, unit: str = "") -> str:
    """Format a float value for the report table."""
    if math.isnan(val):
        return "--"
    if math.isinf(val):
        return "inf"
    if abs(val) > 1e9:
        return f"{val / 1e9:.2f} GB" if "bytes" in unit else f"{val:.4g}"
    if abs(val) > 1e6:
        return f"{val:.0f}"
    if abs(val) < 0.01 and val != 0:
        return f"{val:.6f}"
    return f"{val:.4g}"


def _fmt_delta(delta: float) -> str:
    """Format a delta percentage for the report table."""
    if math.isnan(delta):
        return "--"
    if math.isinf(delta):
        return "+inf%"
    return f"{delta:+.2f}%"


# ===========================================================================
# CompareBaseline
# ===========================================================================

class CompareBaseline:
    """
    Compares current benchmark results against stored baseline for a given
    machine profile. Produces structured pass/fail/warn results and a
    human-readable report.

    Comparison is only valid within the same machine_profile. If the machine
    profile differs, the comparison is meaningless.

    Example:
        comparator = CompareBaseline(ToleranceConfig())
        result = comparator.compare(
            current_dir="artifacts",
            baseline_dir="bench/baselines",
            machine_profile="H100x8_...",
        )
        print(result.report)
        sys.exit(0 if result.passed else 1)
    """

    METRICS_SUFFIX = ".metrics.json"
    EVAL_SUFFIX = ".eval.json"

    def __init__(self, tolerances: ToleranceConfig) -> None:
        tolerances.validate()
        self.tol = tolerances

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def compare(
        self,
        current_dir: str,
        baseline_dir: str,
        machine_profile: str,
    ) -> CompareResult:
        """
        Load metrics and eval JSON from both current and baseline directories,
        then run all comparison checks.

        Args:
            current_dir:     Directory containing <machine_profile>.metrics.json
                             and <machine_profile>.eval.json for the current run.
            baseline_dir:    Directory containing baseline files.
            machine_profile: Profile string used as filename prefix.

        Returns:
            CompareResult with overall_status, list of CheckResults, and report.
        """
        # Load baseline — missing baseline returns SKIP
        try:
            baseline_metrics = self._load_json(baseline_dir, machine_profile, self.METRICS_SUFFIX)
            baseline_eval = self._load_json(baseline_dir, machine_profile, self.EVAL_SUFFIX)
        except FileNotFoundError as exc:
            msg = (
                f"No baseline found for profile '{machine_profile}'.\n"
                f"This appears to be the first run on this hardware.\n"
                f"Run with --update-baseline on main branch to create the baseline.\n"
                f"Details: {exc}"
            )
            report = f"=== Performance Gate Report ===\n{msg}\nOverall: SKIP"
            return CompareResult(
                overall_status=Status.SKIP,
                checks=[],
                report=report,
                machine_profile=machine_profile,
            )

        # Schema version compatibility check
        if not self._schemas_compatible(baseline_metrics, baseline_eval):
            report = (
                f"=== Performance Gate Report ===\n"
                f"WARNING: Schema version mismatch between baseline and current.\n"
                f"Baseline schema: {baseline_metrics.get('schema_version', 'unknown')}\n"
                f"Please run --update-baseline to refresh the baseline.\n"
                f"Overall: SKIP"
            )
            return CompareResult(
                overall_status=Status.SKIP,
                checks=[],
                report=report,
                machine_profile=machine_profile,
            )

        # Load current results
        try:
            current_metrics = self._load_json(current_dir, machine_profile, self.METRICS_SUFFIX)
            current_eval = self._load_json(current_dir, machine_profile, self.EVAL_SUFFIX)
        except FileNotFoundError as exc:
            report = (
                f"=== Performance Gate Report ===\n"
                f"ERROR: Current results not found: {exc}\n"
                f"Overall: FAIL"
            )
            return CompareResult(
                overall_status=Status.FAIL,
                checks=[],
                report=report,
                machine_profile=machine_profile,
            )

        # Run all checks
        checks: List[CheckResult] = []
        checks.extend(self._check_throughput(current_metrics, baseline_metrics))
        checks.extend(self._check_quality(current_eval, baseline_eval))
        checks.extend(self._check_stability(current_metrics))

        # Determine overall status
        if any(c.status == Status.FAIL for c in checks):
            overall = Status.FAIL
        elif any(c.status == Status.WARN for c in checks):
            overall = Status.WARN
        else:
            overall = Status.PASS

        report = self.format_report(checks, machine_profile, overall)

        return CompareResult(
            overall_status=overall,
            checks=checks,
            report=report,
            machine_profile=machine_profile,
        )

    def update_baseline(
        self,
        current_dir: str,
        dest_dir: str,
        machine_profile: str,
    ) -> None:
        """
        Copy current results to the baseline directory.

        This should only be called after the gate passes on main branch.

        Args:
            current_dir:     Directory containing current results.
            dest_dir:        Baseline directory to update.
            machine_profile: Profile string used as filename prefix.
        """
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)

        for suffix in (self.METRICS_SUFFIX, self.EVAL_SUFFIX):
            src = Path(current_dir) / f"{machine_profile}{suffix}"
            if not src.exists():
                raise FileNotFoundError(
                    f"Cannot update baseline: source file not found: {src}"
                )
            dst = dest / f"{machine_profile}{suffix}"
            shutil.copy2(src, dst)
            print(f"Baseline updated: {dst}")

    def format_report(
        self,
        checks: List[CheckResult],
        machine_profile: str,
        overall: Status,
    ) -> str:
        """
        Format a human-readable table report from check results.

        Args:
            checks:          List of CheckResult instances.
            machine_profile: Profile string for header.
            overall:         Overall Status enum value.

        Returns:
            Multi-line formatted report string.
        """
        lines = [
            "=== Performance Gate Report ===",
            f"Machine Profile: {machine_profile}",
            f"Timestamp:       {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
            "",
            f"{'Metric':<32} {'Baseline':>14} {'Current':>14} {'Delta':>10} {'Status':>8}",
            "-" * 84,
        ]

        for r in checks:
            if r.status == Status.SKIP:
                continue
            bas_str = _fmt_val(r.baseline_val)
            cur_str = _fmt_val(r.current_val)
            delta_str = _fmt_delta(r.delta_pct)
            status_marker = {
                Status.PASS: "PASS",
                Status.FAIL: "FAIL",
                Status.WARN: "WARN",
                Status.SKIP: "SKIP",
            }.get(r.status, "????")
            lines.append(
                f"{r.metric:<32} {bas_str:>14} {cur_str:>14} {delta_str:>10} {status_marker:>8}"
            )

        fails = [r for r in checks if r.status == Status.FAIL]
        warns = [r for r in checks if r.status == Status.WARN]

        lines.append("")
        lines.append(f"Overall: {overall.value} ({len(fails)} failures, {len(warns)} warnings)")

        if fails:
            lines.append("")
            lines.append("FAIL details:")
            for r in fails:
                lines.append(f"  - {r.message}")
                lines.append(
                    f"    Baseline: {_fmt_val(r.baseline_val)}, "
                    f"Current: {_fmt_val(r.current_val)}, "
                    f"Delta: {_fmt_delta(r.delta_pct)} (threshold: {r.threshold:.1f})"
                )

        if warns:
            lines.append("")
            lines.append("WARN details:")
            for r in warns:
                lines.append(f"  - {r.message}")

        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Internal: individual checks
    # -----------------------------------------------------------------------

    def _check_throughput(
        self,
        current: dict,
        baseline: dict,
    ) -> List[CheckResult]:
        """
        Check throughput metrics: tokens/sec p50 and step time p50.
        Memory is checked here as a WARN-only gate.
        """
        results: List[CheckResult] = []

        # --- tokens_per_sec_p50: higher is better; negative delta = regression ---
        try:
            cur_tps = float(current["throughput"]["tokens_per_sec_p50"])
            bas_tps = float(baseline["throughput"]["tokens_per_sec_p50"])
            delta = compute_delta_pct(cur_tps, bas_tps)
            # Drop means negative delta
            is_fail = delta < -self.tol.throughput_drop_pct
            results.append(CheckResult(
                metric="tokens_per_sec_p50",
                baseline_val=bas_tps,
                current_val=cur_tps,
                delta_pct=delta,
                threshold=-self.tol.throughput_drop_pct,
                status=Status.FAIL if is_fail else Status.PASS,
                message=(
                    f"tokens/sec p50 {'dropped' if is_fail else 'OK'}: "
                    f"{delta:+.2f}% (threshold: -{self.tol.throughput_drop_pct:.1f}%)"
                ),
            ))
        except (KeyError, TypeError, ValueError) as exc:
            results.append(self._skip_check("tokens_per_sec_p50", str(exc)))

        # --- step_time_p50: lower is better; positive delta = regression ---
        try:
            cur_st = float(current["timing"]["step_time_p50_s"])
            bas_st = float(baseline["timing"]["step_time_p50_s"])
            delta_st = compute_delta_pct(cur_st, bas_st)
            is_fail_st = delta_st > self.tol.step_time_increase_pct
            results.append(CheckResult(
                metric="step_time_p50_s",
                baseline_val=bas_st,
                current_val=cur_st,
                delta_pct=delta_st,
                threshold=self.tol.step_time_increase_pct,
                status=Status.FAIL if is_fail_st else Status.PASS,
                message=(
                    f"step time p50 {'increased' if is_fail_st else 'OK'}: "
                    f"{delta_st:+.2f}% (threshold: +{self.tol.step_time_increase_pct:.1f}%)"
                ),
            ))
        except (KeyError, TypeError, ValueError) as exc:
            results.append(self._skip_check("step_time_p50_s", str(exc)))

        # --- peak_allocated_bytes: WARN only ---
        try:
            cur_mem = float(current["memory"]["peak_allocated_bytes"])
            bas_mem = float(baseline["memory"]["peak_allocated_bytes"])
            delta_mem = compute_delta_pct(cur_mem, bas_mem)
            is_warn = delta_mem > self.tol.memory_increase_pct
            results.append(CheckResult(
                metric="peak_allocated_bytes",
                baseline_val=bas_mem,
                current_val=cur_mem,
                delta_pct=delta_mem,
                threshold=self.tol.memory_increase_pct,
                status=Status.WARN if is_warn else Status.PASS,
                message=(
                    f"peak memory {'WARN: increased' if is_warn else 'OK'}: "
                    f"{delta_mem:+.2f}% (threshold: +{self.tol.memory_increase_pct:.1f}%)"
                ),
            ))
        except (KeyError, TypeError, ValueError) as exc:
            results.append(self._skip_check("peak_allocated_bytes", str(exc)))

        return results

    def _check_quality(
        self,
        current_eval: dict,
        baseline_eval: dict,
    ) -> List[CheckResult]:
        """
        Check quality metrics: perplexity and probe accuracies.
        """
        results: List[CheckResult] = []

        # --- ppl_fixed_shard: lower is better; positive delta = regression ---
        try:
            cur_ppl = float(current_eval["ppl_fixed_shard"])
            bas_ppl = float(baseline_eval["ppl_fixed_shard"])
            delta_ppl = compute_delta_pct(cur_ppl, bas_ppl)
            is_fail_ppl = delta_ppl > self.tol.ppl_increase_pct
            results.append(CheckResult(
                metric="ppl_fixed_shard",
                baseline_val=bas_ppl,
                current_val=cur_ppl,
                delta_pct=delta_ppl,
                threshold=self.tol.ppl_increase_pct,
                status=Status.FAIL if is_fail_ppl else Status.PASS,
                message=(
                    f"perplexity {'worsened' if is_fail_ppl else 'OK'}: "
                    f"{delta_ppl:+.2f}% (threshold: +{self.tol.ppl_increase_pct:.1f}%)"
                ),
            ))
        except (KeyError, TypeError, ValueError) as exc:
            results.append(self._skip_check("ppl_fixed_shard", str(exc)))

        # --- Probe accuracies: higher is better; absolute drop in pp ---
        cur_probes = current_eval.get("task_probe_accuracy", {})
        bas_probes = baseline_eval.get("task_probe_accuracy", {})

        for probe_name, bas_acc in bas_probes.items():
            if probe_name not in cur_probes:
                results.append(CheckResult(
                    metric=f"probe/{probe_name}",
                    baseline_val=float(bas_acc),
                    current_val=float("nan"),
                    delta_pct=float("nan"),
                    threshold=self.tol.probe_drop_abs,
                    status=Status.WARN,
                    message=f"probe {probe_name} missing from current run",
                ))
                continue

            cur_acc = float(cur_probes[probe_name])
            bas_acc_f = float(bas_acc)
            # Absolute drop in percentage points (not relative %)
            abs_drop_pp = (bas_acc_f - cur_acc) * 100.0
            delta_rel = compute_delta_pct(cur_acc, bas_acc_f)
            is_fail_probe = abs_drop_pp > self.tol.probe_drop_abs
            results.append(CheckResult(
                metric=f"probe/{probe_name}",
                baseline_val=bas_acc_f,
                current_val=cur_acc,
                delta_pct=delta_rel,
                threshold=self.tol.probe_drop_abs,
                status=Status.FAIL if is_fail_probe else Status.PASS,
                message=(
                    f"probe {probe_name} {'FAIL: dropped' if is_fail_probe else 'OK'}: "
                    f"{-abs_drop_pp:+.1f}pp (threshold: -{self.tol.probe_drop_abs:.1f}pp)"
                ),
            ))

        return results

    def _check_stability(self, current: dict) -> List[CheckResult]:
        """
        Check training stability via loss slope.
        A positive slope above the threshold indicates the model is diverging.
        """
        results: List[CheckResult] = []

        try:
            loss_slope = current.get("loss", {}).get("loss_slope")
            if loss_slope is None:
                return results  # No loss data available

            loss_slope = float(loss_slope)
            is_fail = loss_slope > self.tol.loss_slope_threshold
            results.append(CheckResult(
                metric="loss_slope",
                baseline_val=0.0,
                current_val=loss_slope,
                delta_pct=0.0,
                threshold=self.tol.loss_slope_threshold,
                status=Status.FAIL if is_fail else Status.PASS,
                message=(
                    f"loss slope {'FAIL: positive divergence trend' if is_fail else 'OK'}: "
                    f"{loss_slope:.6f} (threshold: +{self.tol.loss_slope_threshold:.6f})"
                ),
            ))
        except (KeyError, TypeError, ValueError) as exc:
            results.append(self._skip_check("loss_slope", str(exc)))

        return results

    # -----------------------------------------------------------------------
    # Internal: utilities
    # -----------------------------------------------------------------------

    def _load_json(self, directory: str, profile: str, suffix: str) -> dict:
        """
        Load a JSON file from directory/profile+suffix.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(directory) / f"{profile}{suffix}"
        if not path.exists():
            raise FileNotFoundError(
                f"Expected file not found: {path}"
            )
        with open(path) as f:
            return json.load(f)

    def _schemas_compatible(self, baseline_metrics: dict, baseline_eval: dict) -> bool:
        """Return True if schema versions are compatible."""
        supported = {"1.0"}
        m_ver = baseline_metrics.get("schema_version", "unknown")
        e_ver = baseline_eval.get("schema_version", "unknown")
        return m_ver in supported and e_ver in supported

    def _skip_check(self, metric: str, reason: str) -> CheckResult:
        """Create a SKIP check result when a metric cannot be read."""
        return CheckResult(
            metric=metric,
            baseline_val=float("nan"),
            current_val=float("nan"),
            delta_pct=float("nan"),
            threshold=0.0,
            status=Status.SKIP,
            message=f"SKIP {metric}: {reason}",
        )


# ===========================================================================
# Helpers for tests
# ===========================================================================

def make_metrics(
    tokens_per_sec_p50: float = 100_000.0,
    step_time_p50_s: float = 1.0,
    peak_allocated_bytes: int = 50 * 1024 ** 3,
    mfu_p50: float = 0.40,
    loss_slope: float = -0.001,
    machine_profile: str = "test_profile",
) -> dict:
    """Create a minimal valid metrics dict for testing."""
    return {
        "schema_version": "1.0",
        "machine_profile": machine_profile,
        "throughput": {
            "tokens_per_sec_mean": tokens_per_sec_p50,
            "tokens_per_sec_p50": tokens_per_sec_p50,
            "tokens_per_sec_p10": tokens_per_sec_p50 * 0.9,
        },
        "timing": {
            "step_time_mean_s": step_time_p50_s,
            "step_time_p50_s": step_time_p50_s,
            "step_time_p90_s": step_time_p50_s * 1.1,
        },
        "memory": {
            "peak_allocated_bytes": peak_allocated_bytes,
            "peak_allocated_gb": peak_allocated_bytes / (1024 ** 3),
        },
        "mfu": {"mfu_p50": mfu_p50},
        "loss": {"loss_slope": loss_slope, "loss_values": [], "final_loss": None},
    }


def make_eval_result(
    ppl: float = 12.0,
    basic_reasoning: float = 0.92,
    format_following: float = 0.87,
    code_sanity: float = 0.90,
    machine_profile: str = "test_profile",
) -> dict:
    """Create a minimal valid eval dict for testing."""
    return {
        "schema_version": "1.0",
        "machine_profile": machine_profile,
        "ppl_fixed_shard": ppl,
        "task_probe_accuracy": {
            "basic_reasoning_25": basic_reasoning,
            "format_following_30": format_following,
            "code_sanity_20": code_sanity,
        },
    }


# ===========================================================================
# Self-Tests
# ===========================================================================

def _run_self_tests() -> None:
    """Run all self-tests."""
    print("Running compare_baseline_template self-tests...")
    failures: List[str] = []

    def check(name: str, condition: bool, msg: str = "") -> None:
        if not condition:
            failures.append(f"FAIL [{name}]: {msg}")
        else:
            print(f"  PASS  {name}")

    tol = ToleranceConfig()
    comparator = CompareBaseline(tol)

    # --- delta computation ---
    check("delta.positive", abs(compute_delta_pct(105.0, 100.0) - 5.0) < 1e-9)
    check("delta.negative", abs(compute_delta_pct(95.0, 100.0) - (-5.0)) < 1e-9)
    check("delta.zero", compute_delta_pct(100.0, 100.0) == 0.0)
    check("delta.zero_baseline", math.isinf(compute_delta_pct(1.0, 0.0)))

    # --- PASS when within tolerance ---
    with tempfile.TemporaryDirectory() as tmp_dir:
        cur_dir = os.path.join(tmp_dir, "current")
        bas_dir = os.path.join(tmp_dir, "baseline")
        os.makedirs(cur_dir)
        os.makedirs(bas_dir)

        profile = "test_profile"
        base_m = make_metrics(tokens_per_sec_p50=100_000.0, step_time_p50_s=1.0)
        cur_m = make_metrics(tokens_per_sec_p50=98_000.0, step_time_p50_s=1.02)  # -2%, +2%
        base_e = make_eval_result(ppl=12.0, basic_reasoning=0.92)
        cur_e = make_eval_result(ppl=12.1, basic_reasoning=0.91)  # +0.83% ppl, -1pp probe

        for d, m, e in [(cur_dir, cur_m, cur_e), (bas_dir, base_m, base_e)]:
            with open(os.path.join(d, f"{profile}.metrics.json"), "w") as f:
                json.dump(m, f)
            with open(os.path.join(d, f"{profile}.eval.json"), "w") as f:
                json.dump(e, f)

        result = comparator.compare(cur_dir, bas_dir, profile)
        check("compare.pass_within_tolerance", result.overall_status == Status.PASS, result.report)

    # --- FAIL on throughput drop > 5% ---
    with tempfile.TemporaryDirectory() as tmp_dir:
        cur_dir = os.path.join(tmp_dir, "current")
        bas_dir = os.path.join(tmp_dir, "baseline")
        os.makedirs(cur_dir)
        os.makedirs(bas_dir)
        profile = "test_profile"
        base_m = make_metrics(tokens_per_sec_p50=100_000.0)
        cur_m = make_metrics(tokens_per_sec_p50=93_000.0)  # -7% -> FAIL
        base_e = make_eval_result()
        cur_e = make_eval_result()
        for d, m, e in [(cur_dir, cur_m, cur_e), (bas_dir, base_m, base_e)]:
            with open(os.path.join(d, f"{profile}.metrics.json"), "w") as f:
                json.dump(m, f)
            with open(os.path.join(d, f"{profile}.eval.json"), "w") as f:
                json.dump(e, f)
        result = comparator.compare(cur_dir, bas_dir, profile)
        check("compare.fail_on_throughput_drop", result.overall_status == Status.FAIL, result.report)
        fail_metrics = [c.metric for c in result.failed_checks]
        check("compare.fail_metric_is_tps", "tokens_per_sec_p50" in fail_metrics)

    # --- Pass exactly at threshold boundary (5.0% drop with 5.0% threshold = PASS) ---
    with tempfile.TemporaryDirectory() as tmp_dir:
        cur_dir = os.path.join(tmp_dir, "current")
        bas_dir = os.path.join(tmp_dir, "baseline")
        os.makedirs(cur_dir)
        os.makedirs(bas_dir)
        profile = "test_profile"
        base_m = make_metrics(tokens_per_sec_p50=100_000.0)
        cur_m = make_metrics(tokens_per_sec_p50=95_000.0)  # Exactly -5%
        base_e = make_eval_result()
        cur_e = make_eval_result()
        for d, m, e in [(cur_dir, cur_m, cur_e), (bas_dir, base_m, base_e)]:
            with open(os.path.join(d, f"{profile}.metrics.json"), "w") as f:
                json.dump(m, f)
            with open(os.path.join(d, f"{profile}.eval.json"), "w") as f:
                json.dump(e, f)
        result = comparator.compare(cur_dir, bas_dir, profile)
        tps_check = next((c for c in result.checks if c.metric == "tokens_per_sec_p50"), None)
        check(
            "compare.boundary_exactly_at_threshold_is_pass",
            tps_check is not None and tps_check.status == Status.PASS,
            f"Expected PASS at boundary, got {tps_check.status if tps_check else 'None'}",
        )

    # --- WARN on memory increase > 10% ---
    with tempfile.TemporaryDirectory() as tmp_dir:
        cur_dir = os.path.join(tmp_dir, "current")
        bas_dir = os.path.join(tmp_dir, "baseline")
        os.makedirs(cur_dir)
        os.makedirs(bas_dir)
        profile = "test_profile"
        base_m = make_metrics(peak_allocated_bytes=50 * 1024 ** 3)
        cur_m = make_metrics(peak_allocated_bytes=56 * 1024 ** 3)  # +12% -> WARN
        base_e = make_eval_result()
        cur_e = make_eval_result()
        for d, m, e in [(cur_dir, cur_m, cur_e), (bas_dir, base_m, base_e)]:
            with open(os.path.join(d, f"{profile}.metrics.json"), "w") as f:
                json.dump(m, f)
            with open(os.path.join(d, f"{profile}.eval.json"), "w") as f:
                json.dump(e, f)
        result = comparator.compare(cur_dir, bas_dir, profile)
        mem_check = next((c for c in result.checks if c.metric == "peak_allocated_bytes"), None)
        check(
            "compare.warn_on_memory_increase",
            mem_check is not None and mem_check.status == Status.WARN,
            f"Memory check status: {mem_check.status if mem_check else 'None'}",
        )
        # Overall should not be FAIL just from memory warning
        check("compare.memory_warn_not_fail", result.overall_status != Status.FAIL, result.report)

    # --- FAIL on perplexity worsening > 1.5% ---
    with tempfile.TemporaryDirectory() as tmp_dir:
        cur_dir = os.path.join(tmp_dir, "current")
        bas_dir = os.path.join(tmp_dir, "baseline")
        os.makedirs(cur_dir)
        os.makedirs(bas_dir)
        profile = "test_profile"
        base_m = make_metrics()
        cur_m = make_metrics()
        base_e = make_eval_result(ppl=10.0)
        cur_e = make_eval_result(ppl=10.22)  # +2.2% -> FAIL
        for d, m, e in [(cur_dir, cur_m, cur_e), (bas_dir, base_m, base_e)]:
            with open(os.path.join(d, f"{profile}.metrics.json"), "w") as f:
                json.dump(m, f)
            with open(os.path.join(d, f"{profile}.eval.json"), "w") as f:
                json.dump(e, f)
        result = comparator.compare(cur_dir, bas_dir, profile)
        check("compare.fail_on_ppl_worsening", result.overall_status == Status.FAIL, result.report)

    # --- FAIL on probe accuracy drop > 2pp ---
    with tempfile.TemporaryDirectory() as tmp_dir:
        cur_dir = os.path.join(tmp_dir, "current")
        bas_dir = os.path.join(tmp_dir, "baseline")
        os.makedirs(cur_dir)
        os.makedirs(bas_dir)
        profile = "test_profile"
        base_m = make_metrics()
        cur_m = make_metrics()
        base_e = make_eval_result(basic_reasoning=0.90)
        cur_e = make_eval_result(basic_reasoning=0.87)  # -3pp -> FAIL
        for d, m, e in [(cur_dir, cur_m, cur_e), (bas_dir, base_m, base_e)]:
            with open(os.path.join(d, f"{profile}.metrics.json"), "w") as f:
                json.dump(m, f)
            with open(os.path.join(d, f"{profile}.eval.json"), "w") as f:
                json.dump(e, f)
        result = comparator.compare(cur_dir, bas_dir, profile)
        check("compare.fail_on_probe_drop", result.overall_status == Status.FAIL, result.report)

    # --- SKIP on missing baseline ---
    with tempfile.TemporaryDirectory() as tmp_dir:
        cur_dir = os.path.join(tmp_dir, "current")
        bas_dir = os.path.join(tmp_dir, "baseline_empty")
        os.makedirs(cur_dir)
        os.makedirs(bas_dir)
        result_skip = comparator.compare(cur_dir, bas_dir, "nonexistent_profile")
        check(
            "compare.skip_on_missing_baseline",
            result_skip.overall_status == Status.SKIP,
            f"Expected SKIP, got {result_skip.overall_status}",
        )

    # --- WARN on missing probe in current ---
    with tempfile.TemporaryDirectory() as tmp_dir:
        cur_dir = os.path.join(tmp_dir, "current")
        bas_dir = os.path.join(tmp_dir, "baseline")
        os.makedirs(cur_dir)
        os.makedirs(bas_dir)
        profile = "test_profile"
        base_m = make_metrics()
        cur_m = make_metrics()
        base_e = make_eval_result(basic_reasoning=0.92)
        cur_e = {
            "schema_version": "1.0",
            "machine_profile": profile,
            "ppl_fixed_shard": 12.0,
            "task_probe_accuracy": {
                # basic_reasoning_25 intentionally missing
                "format_following_30": 0.87,
                "code_sanity_20": 0.90,
            },
        }
        for d, m, e in [(cur_dir, cur_m, cur_e), (bas_dir, base_m, base_e)]:
            with open(os.path.join(d, f"{profile}.metrics.json"), "w") as f:
                json.dump(m, f)
            with open(os.path.join(d, f"{profile}.eval.json"), "w") as f:
                json.dump(e, f)
        result = comparator.compare(cur_dir, bas_dir, profile)
        missing_check = next(
            (c for c in result.checks if "basic_reasoning_25" in c.metric), None
        )
        check(
            "compare.warn_on_missing_probe",
            missing_check is not None and missing_check.status == Status.WARN,
            f"Expected WARN for missing probe, got {missing_check.status if missing_check else 'None'}",
        )

    # --- FAIL on positive loss slope ---
    with tempfile.TemporaryDirectory() as tmp_dir:
        cur_dir = os.path.join(tmp_dir, "current")
        bas_dir = os.path.join(tmp_dir, "baseline")
        os.makedirs(cur_dir)
        os.makedirs(bas_dir)
        profile = "test_profile"
        base_m = make_metrics(loss_slope=-0.001)
        cur_m = make_metrics(loss_slope=0.002)  # Positive slope -> FAIL
        base_e = make_eval_result()
        cur_e = make_eval_result()
        for d, m, e in [(cur_dir, cur_m, cur_e), (bas_dir, base_m, base_e)]:
            with open(os.path.join(d, f"{profile}.metrics.json"), "w") as f:
                json.dump(m, f)
            with open(os.path.join(d, f"{profile}.eval.json"), "w") as f:
                json.dump(e, f)
        result = comparator.compare(cur_dir, bas_dir, profile)
        check("compare.fail_on_positive_loss_slope", result.overall_status == Status.FAIL, result.report)

    # --- update_baseline copies files ---
    with tempfile.TemporaryDirectory() as tmp_dir:
        src_dir = os.path.join(tmp_dir, "current")
        dst_dir = os.path.join(tmp_dir, "baselines")
        os.makedirs(src_dir)
        profile = "test_profile"
        m = make_metrics()
        e = make_eval_result()
        with open(os.path.join(src_dir, f"{profile}.metrics.json"), "w") as f:
            json.dump(m, f)
        with open(os.path.join(src_dir, f"{profile}.eval.json"), "w") as f:
            json.dump(e, f)
        comparator.update_baseline(src_dir, dst_dir, profile)
        check(
            "update_baseline.copies_metrics",
            os.path.isfile(os.path.join(dst_dir, f"{profile}.metrics.json")),
        )
        check(
            "update_baseline.copies_eval",
            os.path.isfile(os.path.join(dst_dir, f"{profile}.eval.json")),
        )

    # --- report contains metric names ---
    checks_list = [
        CheckResult(
            metric="tokens_per_sec_p50",
            baseline_val=100_000.0,
            current_val=90_000.0,
            delta_pct=-10.0,
            threshold=-5.0,
            status=Status.FAIL,
            message="FAIL",
        )
    ]
    report = comparator.format_report(checks_list, "test_profile", Status.FAIL)
    check("format_report.contains_metric_name", "tokens_per_sec_p50" in report)
    check("format_report.contains_overall_fail", "FAIL" in report)

    # Summary
    print()
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    else:
        print("All compare_baseline_template self-tests passed.")


if __name__ == "__main__":
    _run_self_tests()
