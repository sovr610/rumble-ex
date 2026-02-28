"""
brain_ai/inference/profiler.py — Latency Profiler with Per-Module Breakdown

This module provides the LatencyProfiler for detailed per-module latency
analysis. Supports profile(), per_module_breakdown(), bottleneck_analysis(),
and export_chrome_trace() for visualization in Chrome's chrome://tracing.

Key classes:
    Bottleneck       — A single identified bottleneck module
    ProfileReport    — Complete profiling report with percentiles
    LatencyProfiler  — Main class: profile(), bottleneck_analysis(), export_chrome_trace()

Design principles:
    1. Hook-based profiling — registers forward hooks on all submodules.
    2. Non-intrusive — hooks are removed after profiling completes.
    3. Statistical — runs multiple iterations and computes percentiles.
    4. Chrome trace export — produces JSON compatible with chrome://tracing.
    5. Bottleneck identification — ranks modules by latency contribution.

References:
    SKILL.md section on LatencyProfiler contract
    references/testing-matrix.md section on Latency Profiling Tests
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


# ===========================================================================
# SECTION 1: Dataclasses
# ===========================================================================

@dataclass
class Bottleneck:
    """A single bottleneck identified by profiling.

    Attributes:
        module_name: Name of the bottleneck module.
        latency_ms: Average latency in milliseconds.
        percentage: Percentage of total latency.
        recommendation: Suggested optimization.
    """

    module_name: str = ""
    latency_ms: float = 0.0
    percentage: float = 0.0
    recommendation: str = ""


@dataclass
class ProfileReport:
    """Complete profiling report.

    Attributes:
        total_latency_ms: Mean total forward pass latency.
        per_module_ms: Per-module latency breakdown.
        n_runs: Number of profiling runs.
        std_ms: Standard deviation of total latency.
        p50_ms: 50th percentile latency.
        p90_ms: 90th percentile latency.
        p95_ms: 95th percentile latency.
        p99_ms: 99th percentile latency.
        bottlenecks: List of identified bottlenecks.
        latencies_ms: Raw per-run total latency measurements.
        per_module_raw: Raw per-module per-run measurements.
    """

    total_latency_ms: float = 0.0
    per_module_ms: Dict[str, float] = field(default_factory=dict)
    n_runs: int = 0
    std_ms: float = 0.0
    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    bottlenecks: List[Bottleneck] = field(default_factory=list)
    latencies_ms: List[float] = field(default_factory=list)
    per_module_raw: Dict[str, List[float]] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"ProfileReport (n_runs={self.n_runs}):",
            f"  Total: {self.total_latency_ms:.3f}ms "
            f"(std={self.std_ms:.3f}ms)",
            f"  p50={self.p50_ms:.3f}ms  p90={self.p90_ms:.3f}ms  "
            f"p99={self.p99_ms:.3f}ms",
        ]
        if self.per_module_ms:
            lines.append("  Per-module breakdown:")
            for name, ms in sorted(
                self.per_module_ms.items(), key=lambda x: x[1], reverse=True
            ):
                pct = (ms / self.total_latency_ms * 100) if self.total_latency_ms > 0 else 0
                lines.append(f"    {name}: {ms:.3f}ms ({pct:.1f}%)")
        return "\n".join(lines)


# ===========================================================================
# SECTION 2: Percentile computation
# ===========================================================================

def compute_percentile(data: List[float], p: float) -> float:
    """Compute the p-th percentile of a list of values.

    Args:
        data: List of numeric values.
        p: Percentile in [0, 100].

    Returns:
        The p-th percentile value.
    """
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (p / 100.0) * (len(sorted_data) - 1)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[-1]
    d = k - f
    return sorted_data[f] + d * (sorted_data[c] - sorted_data[f])


def compute_mean(data: List[float]) -> float:
    """Compute the mean of a list of values."""
    if not data:
        return 0.0
    return sum(data) / len(data)


def compute_std_dev(data: List[float]) -> float:
    """Compute the standard deviation of a list of values."""
    if len(data) < 2:
        return 0.0
    m = compute_mean(data)
    variance = sum((x - m) ** 2 for x in data) / (len(data) - 1)
    return math.sqrt(variance)


# ===========================================================================
# SECTION 3: Module timing hook
# ===========================================================================

class ModuleTimer:
    """Context-like timing hook for a single module.

    Registers a forward pre-hook (records start time) and a forward hook
    (records end time) on the target module.
    """

    def __init__(self, module_name: str):
        self.module_name = module_name
        self.start_time: float = 0.0
        self.elapsed_ms: float = 0.0
        self.measurements: List[float] = []

    def pre_hook(self, module, inputs):
        """Forward pre-hook: record start time."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()

    def post_hook(self, module, inputs, output):
        """Forward hook: record elapsed time."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        self.measurements.append(self.elapsed_ms)

    @property
    def avg_ms(self) -> float:
        return compute_mean(self.measurements)

    def reset(self):
        self.measurements.clear()


# ===========================================================================
# SECTION 4: LatencyProfiler
# ===========================================================================

class LatencyProfiler:
    """Detailed per-module latency profiler for neural network models.

    Hooks into all top-level submodules to measure per-module latency.
    Produces ProfileReport with percentiles and bottleneck analysis.

    Args:
        model: The model to profile.
        device: Device to profile on.
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
        self._timers: Dict[str, ModuleTimer] = {}
        self._hooks: List[Any] = []
        self._last_report: Optional[ProfileReport] = None
        self._total_latencies: List[float] = []

    def _register_hooks(self) -> None:
        """Register timing hooks on all top-level submodules."""
        self._remove_hooks()
        self._timers.clear()

        for name, module in self.model.named_children():
            timer = ModuleTimer(name)
            self._timers[name] = timer
            h1 = module.register_forward_pre_hook(timer.pre_hook)
            h2 = module.register_forward_hook(timer.post_hook)
            self._hooks.append(h1)
            self._hooks.append(h2)

    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def profile(
        self,
        inputs: Dict[str, Tensor],
        n_runs: int = 100,
        n_warmup: int = 10,
    ) -> ProfileReport:
        """Profile the model with the given inputs.

        Runs n_warmup warmup iterations, then n_runs profiling iterations.
        Measures per-module latency and total forward pass latency.

        Args:
            inputs: Input dict for the model.
            n_runs: Number of profiling runs.
            n_warmup: Number of warmup runs before profiling.

        Returns:
            ProfileReport with complete profiling data.
        """
        self.model.eval()

        # Move inputs to device
        device_inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Register hooks
        self._register_hooks()

        try:
            # Warmup phase (do not record)
            with torch.inference_mode():
                for _ in range(n_warmup):
                    self.model(device_inputs)

            # Reset timer measurements after warmup
            for timer in self._timers.values():
                timer.reset()

            # Profiling phase
            total_latencies: List[float] = []

            with torch.inference_mode():
                for _ in range(n_runs):
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()

                    self.model(device_inputs)

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    total_latencies.append(elapsed_ms)

            self._total_latencies = total_latencies

            # Compute per-module averages
            per_module_ms = {
                name: timer.avg_ms for name, timer in self._timers.items()
            }

            # Compute per-module raw measurements
            per_module_raw = {
                name: list(timer.measurements) for name, timer in self._timers.items()
            }

            # Compute statistics
            total_mean_val = compute_mean(total_latencies)
            total_std_val = compute_std_dev(total_latencies)
            p50 = compute_percentile(total_latencies, 50)
            p90 = compute_percentile(total_latencies, 90)
            p95 = compute_percentile(total_latencies, 95)
            p99 = compute_percentile(total_latencies, 99)

            # Build bottleneck analysis
            bottlenecks = self._compute_bottlenecks(per_module_ms, total_mean_val)

            report = ProfileReport(
                total_latency_ms=total_mean_val,
                per_module_ms=per_module_ms,
                n_runs=n_runs,
                std_ms=total_std_val,
                p50_ms=p50,
                p90_ms=p90,
                p95_ms=p95,
                p99_ms=p99,
                bottlenecks=bottlenecks,
                latencies_ms=total_latencies,
                per_module_raw=per_module_raw,
            )

            self._last_report = report
            return report

        finally:
            self._remove_hooks()

    def per_module_breakdown(self) -> Dict[str, float]:
        """Return per-module latency breakdown from the last profile run.

        Returns:
            Dict mapping module name to average latency in ms.
        """
        if self._last_report is None:
            return {}
        return dict(self._last_report.per_module_ms)

    def bottleneck_analysis(self) -> List[Bottleneck]:
        """Return bottleneck analysis from the last profile run.

        Returns:
            List of Bottleneck objects sorted by latency (descending).
        """
        if self._last_report is None:
            return []
        return list(self._last_report.bottlenecks)

    def _compute_bottlenecks(
        self, per_module_ms: Dict[str, float], total_ms: float
    ) -> List[Bottleneck]:
        """Compute bottleneck analysis from per-module latencies."""
        bottlenecks = []
        for name, ms in per_module_ms.items():
            pct = (ms / total_ms * 100) if total_ms > 0 else 0

            # Generate recommendation based on percentage
            if pct > 50:
                rec = (
                    f"Module '{name}' accounts for {pct:.0f}% of latency. "
                    f"Consider optimization or caching."
                )
            elif pct > 25:
                rec = (
                    f"Module '{name}' is significant ({pct:.0f}%). "
                    f"Consider FP16 or pruning."
                )
            else:
                rec = ""

            bottlenecks.append(Bottleneck(
                module_name=name,
                latency_ms=ms,
                percentage=pct,
                recommendation=rec,
            ))

        # Sort by latency descending
        bottlenecks.sort(key=lambda b: b.latency_ms, reverse=True)
        return bottlenecks

    def export_chrome_trace(self, path: str) -> None:
        """Export profiling results as Chrome trace JSON.

        The output file can be loaded in chrome://tracing or
        Perfetto (ui.perfetto.dev) for visualization.

        Args:
            path: File path for the output JSON.
        """
        if self._last_report is None:
            raise RuntimeError(
                "No profile data available. Call profile() first."
            )

        report = self._last_report
        trace_events = []

        # Create trace events for each run and each module
        for run_idx in range(report.n_runs):
            run_offset_us = 0
            for name in report.per_module_ms:
                if name in report.per_module_raw and run_idx < len(report.per_module_raw[name]):
                    dur_ms = report.per_module_raw[name][run_idx]
                else:
                    dur_ms = report.per_module_ms.get(name, 0)

                dur_us = dur_ms * 1000  # convert ms to us
                event = {
                    "name": name,
                    "cat": "inference",
                    "ph": "X",  # complete event
                    "ts": run_idx * report.total_latency_ms * 1000 + run_offset_us,
                    "dur": dur_us,
                    "pid": 0,
                    "tid": 0,
                    "args": {
                        "run": run_idx,
                        "latency_ms": dur_ms,
                    },
                }
                trace_events.append(event)
                run_offset_us += dur_us

        trace_data = {"traceEvents": trace_events}

        with open(path, "w") as f:
            json.dump(trace_data, f, indent=2)

        logger.info(f"Chrome trace exported to {path}")


# ===========================================================================
# SECTION 5: DictModel for testing
# ===========================================================================

class DictModel(nn.Module):
    """Wrapper that makes a model accept dict inputs.
    Passes the first tensor value to the inner sequential model."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, inputs):
        if isinstance(inputs, dict):
            x = next(iter(inputs.values()))
        else:
            x = inputs
        return self.inner(x)


class MultiModuleModel(nn.Module):
    """Model with named submodules for profiling tests."""

    def __init__(self):
        super().__init__()
        self.fast_layer = nn.Linear(64, 64)
        self.medium_layer = nn.Linear(64, 256)
        self.slow_layer = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10),
        )

    def forward(self, inputs):
        if isinstance(inputs, dict):
            x = next(iter(inputs.values()))
        else:
            x = inputs
        x = self.fast_layer(x)
        x = self.medium_layer(x)
        x = self.slow_layer(x)
        return x


# ===========================================================================
# SECTION 6: Self-tests
# ===========================================================================

def _run_self_tests():
    """Run self-tests for latency profiler."""
    import os
    import tempfile
    import traceback

    passed = 0
    failed = 0
    test_results = []

    def _test(name, fn):
        nonlocal passed, failed
        try:
            fn()
            passed += 1
            test_results.append(f"  PASS: {name}")
        except Exception as e:
            failed += 1
            test_results.append(f"  FAIL: {name} -- {e}")
            traceback.print_exc()

    # --- Utility function tests ---
    def test_percentile_basic():
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert abs(compute_percentile(data, 50) - 3.0) < 1e-6
        assert abs(compute_percentile(data, 0) - 1.0) < 1e-6
        assert abs(compute_percentile(data, 100) - 5.0) < 1e-6

    def test_percentile_empty():
        assert compute_percentile([], 50) == 0.0

    def test_percentile_single():
        assert abs(compute_percentile([42.0], 50) - 42.0) < 1e-6

    def test_mean_basic():
        assert abs(compute_mean([1, 2, 3, 4, 5]) - 3.0) < 1e-6

    def test_mean_empty():
        assert compute_mean([]) == 0.0

    def test_std_dev_basic():
        data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        s = compute_std_dev(data)
        assert s > 0

    def test_std_dev_single():
        assert compute_std_dev([5.0]) == 0.0

    # --- ModuleTimer tests ---
    def test_module_timer_measures():
        model = nn.Linear(10, 10)
        timer = ModuleTimer("test_layer")
        h1 = model.register_forward_pre_hook(timer.pre_hook)
        h2 = model.register_forward_hook(timer.post_hook)
        with torch.inference_mode():
            model(torch.randn(1, 10))
        h1.remove()
        h2.remove()
        assert len(timer.measurements) == 1
        assert timer.measurements[0] >= 0

    def test_module_timer_reset():
        timer = ModuleTimer("test")
        timer.measurements = [1.0, 2.0, 3.0]
        timer.reset()
        assert len(timer.measurements) == 0

    def test_module_timer_avg():
        timer = ModuleTimer("test")
        timer.measurements = [1.0, 2.0, 3.0]
        assert abs(timer.avg_ms - 2.0) < 1e-6

    # --- LatencyProfiler tests ---
    def test_profile_basic():
        model = MultiModuleModel()
        profiler = LatencyProfiler(model)
        inputs = {"features": torch.randn(1, 64)}
        report = profiler.profile(inputs, n_runs=10, n_warmup=2)
        assert report.n_runs == 10
        assert report.total_latency_ms > 0

    def test_profile_per_module():
        model = MultiModuleModel()
        profiler = LatencyProfiler(model)
        inputs = {"features": torch.randn(1, 64)}
        report = profiler.profile(inputs, n_runs=20, n_warmup=5)
        assert "fast_layer" in report.per_module_ms
        assert "medium_layer" in report.per_module_ms
        assert "slow_layer" in report.per_module_ms

    def test_profile_per_module_sum_tolerance():
        """Per-module sum should be close to total (within tolerance)."""
        model = MultiModuleModel()
        profiler = LatencyProfiler(model)
        inputs = {"features": torch.randn(1, 64)}
        report = profiler.profile(inputs, n_runs=50, n_warmup=10)
        module_sum = sum(report.per_module_ms.values())
        assert module_sum > 0
        ratio = module_sum / report.total_latency_ms if report.total_latency_ms > 0 else 0
        assert ratio > 0.3, f"Module sum ratio too low: {ratio:.2f}"

    def test_profile_percentiles():
        model = MultiModuleModel()
        profiler = LatencyProfiler(model)
        inputs = {"features": torch.randn(1, 64)}
        report = profiler.profile(inputs, n_runs=50, n_warmup=5)
        assert report.p50_ms > 0
        assert report.p90_ms >= report.p50_ms
        assert report.p99_ms >= report.p90_ms

    def test_profile_std():
        model = MultiModuleModel()
        profiler = LatencyProfiler(model)
        inputs = {"features": torch.randn(1, 64)}
        report = profiler.profile(inputs, n_runs=50, n_warmup=5)
        assert report.std_ms >= 0

    def test_profile_latencies_list():
        model = MultiModuleModel()
        profiler = LatencyProfiler(model)
        inputs = {"features": torch.randn(1, 64)}
        report = profiler.profile(inputs, n_runs=20, n_warmup=2)
        assert len(report.latencies_ms) == 20
        assert all(t >= 0 for t in report.latencies_ms)

    def test_bottleneck_analysis_ordering():
        model = MultiModuleModel()
        profiler = LatencyProfiler(model)
        inputs = {"features": torch.randn(1, 64)}
        profiler.profile(inputs, n_runs=30, n_warmup=5)
        bottlenecks = profiler.bottleneck_analysis()
        assert len(bottlenecks) == 3
        for i in range(len(bottlenecks) - 1):
            assert bottlenecks[i].latency_ms >= bottlenecks[i + 1].latency_ms

    def test_bottleneck_identifies_slow():
        model = MultiModuleModel()
        profiler = LatencyProfiler(model)
        inputs = {"features": torch.randn(1, 64)}
        profiler.profile(inputs, n_runs=50, n_warmup=10)
        bottlenecks = profiler.bottleneck_analysis()
        assert bottlenecks[0].module_name == "slow_layer"

    def test_bottleneck_percentage():
        model = MultiModuleModel()
        profiler = LatencyProfiler(model)
        inputs = {"features": torch.randn(1, 64)}
        profiler.profile(inputs, n_runs=30, n_warmup=5)
        bottlenecks = profiler.bottleneck_analysis()
        total_pct = sum(b.percentage for b in bottlenecks)
        assert total_pct > 50

    def test_per_module_breakdown_method():
        model = MultiModuleModel()
        profiler = LatencyProfiler(model)
        inputs = {"features": torch.randn(1, 64)}
        profiler.profile(inputs, n_runs=10, n_warmup=2)
        breakdown = profiler.per_module_breakdown()
        assert isinstance(breakdown, dict)
        assert len(breakdown) == 3

    def test_per_module_breakdown_before_profile():
        model = MultiModuleModel()
        profiler = LatencyProfiler(model)
        breakdown = profiler.per_module_breakdown()
        assert breakdown == {}

    def test_bottleneck_before_profile():
        model = MultiModuleModel()
        profiler = LatencyProfiler(model)
        bottlenecks = profiler.bottleneck_analysis()
        assert bottlenecks == []

    def test_chrome_trace_export():
        model = MultiModuleModel()
        profiler = LatencyProfiler(model)
        inputs = {"features": torch.randn(1, 64)}
        profiler.profile(inputs, n_runs=10, n_warmup=2)

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            trace_path = f.name

        try:
            profiler.export_chrome_trace(trace_path)
            assert os.path.exists(trace_path)

            with open(trace_path, "r") as f:
                trace_data = json.load(f)

            assert "traceEvents" in trace_data
            events = trace_data["traceEvents"]
            assert len(events) > 0

            for event in events:
                assert "name" in event
                assert "ph" in event
                assert "ts" in event
                assert "dur" in event
        finally:
            if os.path.exists(trace_path):
                os.remove(trace_path)

    def test_chrome_trace_before_profile():
        model = MultiModuleModel()
        profiler = LatencyProfiler(model)
        try:
            profiler.export_chrome_trace("/tmp/test.json")
            assert False, "Should have raised RuntimeError"
        except RuntimeError:
            pass

    def test_profile_report_summary():
        report = ProfileReport(
            total_latency_ms=10.0, n_runs=100, std_ms=1.0,
            p50_ms=9.5, p90_ms=11.0, p99_ms=15.0,
            per_module_ms={"layer1": 5.0, "layer2": 5.0},
        )
        s = report.summary()
        assert "10.000ms" in s
        assert "layer1" in s
        assert "n_runs=100" in s

    def test_profile_hooks_removed():
        """Hooks should be removed after profiling."""
        model = MultiModuleModel()
        profiler = LatencyProfiler(model)
        inputs = {"features": torch.randn(1, 64)}
        profiler.profile(inputs, n_runs=5, n_warmup=1)
        assert len(profiler._hooks) == 0

    def test_profile_with_dict_model():
        inner = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        model = DictModel(inner)
        profiler = LatencyProfiler(model)
        inputs = {"x": torch.randn(1, 32)}
        report = profiler.profile(inputs, n_runs=10, n_warmup=2)
        assert report.total_latency_ms > 0
        assert "inner" in report.per_module_ms

    def test_profiler_overhead():
        """Profiling should not add excessive overhead."""
        model = MultiModuleModel()
        inputs = {"features": torch.randn(4, 64)}
        n = 50

        model.eval()
        with torch.inference_mode():
            for _ in range(5):
                model(inputs)
            t0 = time.perf_counter()
            for _ in range(n):
                model(inputs)
            baseline = (time.perf_counter() - t0) * 1000

        profiler = LatencyProfiler(model)
        report = profiler.profile(inputs, n_runs=n, n_warmup=5)
        profiled = sum(report.latencies_ms)

        if baseline > 0:
            overhead = (profiled - baseline) / baseline
            assert overhead < 1.0, f"Profiling overhead too high: {overhead:.1%}"

    def test_bottleneck_dataclass():
        b = Bottleneck(
            module_name="slow", latency_ms=50.0, percentage=80.0,
            recommendation="Optimize",
        )
        assert b.module_name == "slow"
        assert b.latency_ms == 50.0

    # Run all tests
    tests = [
        ("percentile basic", test_percentile_basic),
        ("percentile empty", test_percentile_empty),
        ("percentile single", test_percentile_single),
        ("mean basic", test_mean_basic),
        ("mean empty", test_mean_empty),
        ("std_dev basic", test_std_dev_basic),
        ("std_dev single", test_std_dev_single),
        ("ModuleTimer measures", test_module_timer_measures),
        ("ModuleTimer reset", test_module_timer_reset),
        ("ModuleTimer avg", test_module_timer_avg),
        ("profile basic", test_profile_basic),
        ("profile per_module", test_profile_per_module),
        ("profile per_module sum tolerance", test_profile_per_module_sum_tolerance),
        ("profile percentiles", test_profile_percentiles),
        ("profile std", test_profile_std),
        ("profile latencies list", test_profile_latencies_list),
        ("bottleneck ordering", test_bottleneck_analysis_ordering),
        ("bottleneck identifies slow", test_bottleneck_identifies_slow),
        ("bottleneck percentage", test_bottleneck_percentage),
        ("per_module_breakdown method", test_per_module_breakdown_method),
        ("per_module_breakdown before profile", test_per_module_breakdown_before_profile),
        ("bottleneck before profile", test_bottleneck_before_profile),
        ("chrome trace export", test_chrome_trace_export),
        ("chrome trace before profile", test_chrome_trace_before_profile),
        ("profile report summary", test_profile_report_summary),
        ("hooks removed after profile", test_profile_hooks_removed),
        ("profile with DictModel", test_profile_with_dict_model),
        ("profiler overhead", test_profiler_overhead),
        ("Bottleneck dataclass", test_bottleneck_dataclass),
    ]

    print(f"Running {len(tests)} self-tests for latency_profiler_template...")
    for name, fn in tests:
        _test(name, fn)

    print("\n".join(test_results))
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    if failed == 0:
        print("ALL TESTS PASSED")
    return failed == 0


if __name__ == "__main__":
    _run_self_tests()
