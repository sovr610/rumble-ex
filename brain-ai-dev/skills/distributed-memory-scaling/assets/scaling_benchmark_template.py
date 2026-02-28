#!/usr/bin/env python3
"""
scaling_benchmark_template.py
------------------------------
ScalingBenchmark: measures 1-GPU and N-GPU training throughput, computes
scaling efficiency, and outputs metrics compatible with perf-regression-gate.

Scaling efficiency formula:
    scaling_efficiency = throughput_N / (N * throughput_1)

1.0 is perfect linear scaling. Values below 0.80 typically indicate
misconfiguration (wrong wrap policy, bad bucket sizes, accidental offload).

Usage
-----
    from scaling_benchmark_template import ScalingBenchmark, BenchConfig

    bench = ScalingBenchmark()
    single = bench.run_single(BenchConfig(strategy="fsdp", seq_len=2048))
    multi = bench.run_multi(BenchConfig(strategy="fsdp", world_size=4, seq_len=2048))
    report = bench.compute_efficiency(single, multi)
    bench.save_metrics(report, "/tmp/metrics.json")
    print(bench.format_report(report))
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BenchConfig
# ---------------------------------------------------------------------------


@dataclass
class BenchConfig:
    """Configuration for a single benchmark run.

    Attributes
    ----------
    strategy:
        Training strategy: 'ddp', 'fsdp', 'deepspeed_zero2', 'deepspeed_zero3'.
    world_size:
        Number of GPUs to use.
    per_gpu_batch_size:
        Batch size per GPU.
    seq_len:
        Sequence length in tokens.
    warmup_steps:
        Steps to discard from statistics (allow JIT warm-up).
    measured_steps:
        Steps to include in statistics.
    mixed_precision:
        'bf16', 'fp16', or 'none'.
    activation_checkpointing:
        Whether activation checkpointing is enabled.
    model_params_b:
        Model size in billions of parameters (metadata only).
    """

    strategy: str = "ddp"
    world_size: int = 1
    per_gpu_batch_size: int = 4
    seq_len: int = 2048
    warmup_steps: int = 10
    measured_steps: int = 50
    mixed_precision: str = "bf16"
    activation_checkpointing: bool = False
    model_params_b: float = 0.0

    def validate(self) -> None:
        valid_strategies = frozenset(
            ["ddp", "fsdp", "deepspeed_zero2", "deepspeed_zero3"]
        )
        if self.strategy not in valid_strategies:
            raise ValueError(
                f"Invalid strategy '{self.strategy}'. "
                f"Must be one of: {sorted(valid_strategies)}"
            )
        if self.world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {self.world_size}")
        if self.per_gpu_batch_size < 1:
            raise ValueError(
                f"per_gpu_batch_size must be >= 1, got {self.per_gpu_batch_size}"
            )
        if self.seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {self.seq_len}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        if self.measured_steps < 1:
            raise ValueError(f"measured_steps must be >= 1, got {self.measured_steps}")


# ---------------------------------------------------------------------------
# BenchResult
# ---------------------------------------------------------------------------


@dataclass
class BenchResult:
    """Raw timing data from a single benchmark run."""

    strategy: str
    world_size: int
    throughput_p50: float
    throughput_p90: float  # 10th percentile (conservative)
    step_time_p50_ms: float
    step_time_p90_ms: float  # 90th percentile (reveal stalls)
    memory_peak_gb: float
    measured_steps: int
    warmup_steps: int
    per_gpu_batch_size: int
    seq_len: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# ScalingReport
# ---------------------------------------------------------------------------


@dataclass
class ScalingReport:
    """Result of a scaling efficiency computation.

    This is the canonical output artifact of ScalingBenchmark. The schema
    matches the metrics.json format consumed by perf-regression-gate.
    """

    strategy: str
    world_size: int
    throughput_1: float
    throughput_n: float
    scaling_efficiency: float
    memory_peak_gb: float
    step_time_p50_ms: float
    step_time_p90_ms: float
    throughput_p50: float
    throughput_p90: float
    model_params_b: float
    per_gpu_batch_size: int
    seq_len: int
    mixed_precision: str
    activation_checkpointing: bool
    warmup_steps: int
    measured_steps: int
    timestamp: str = ""
    git_commit: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ScalingReport":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# ScalingBenchmark
# ---------------------------------------------------------------------------


class ScalingBenchmark:
    """Measures training throughput and computes scaling efficiency."""

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # run_single
    # ------------------------------------------------------------------

    def run_single(
        self,
        cfg: BenchConfig,
        step_fn: Optional[Any] = None,
    ) -> BenchResult:
        """Run benchmark at world_size=1.

        Parameters
        ----------
        cfg:
            BenchConfig. world_size is overridden to 1.
        step_fn:
            Optional callable(step_index) -> None that performs one training
            step. If None, a synthetic timing loop is used for testing.

        Returns
        -------
        BenchResult
            Raw timing statistics from world_size=1 run.
        """
        cfg.validate()
        single_cfg = BenchConfig(
            strategy=cfg.strategy,
            world_size=1,
            per_gpu_batch_size=cfg.per_gpu_batch_size,
            seq_len=cfg.seq_len,
            warmup_steps=cfg.warmup_steps,
            measured_steps=cfg.measured_steps,
            mixed_precision=cfg.mixed_precision,
            activation_checkpointing=cfg.activation_checkpointing,
            model_params_b=cfg.model_params_b,
        )
        return self._run_benchmark(single_cfg, step_fn)

    # ------------------------------------------------------------------
    # run_multi
    # ------------------------------------------------------------------

    def run_multi(
        self,
        cfg: BenchConfig,
        step_fn: Optional[Any] = None,
    ) -> BenchResult:
        """Run benchmark at world_size=N (cfg.world_size).

        Parameters
        ----------
        cfg:
            BenchConfig. Uses cfg.world_size as-is.
        step_fn:
            Optional callable(step_index) -> None. If None, a synthetic
            timing loop is used.

        Returns
        -------
        BenchResult
            Raw timing statistics from world_size=N run.
        """
        cfg.validate()
        return self._run_benchmark(cfg, step_fn)

    # ------------------------------------------------------------------
    # _run_benchmark (internal)
    # ------------------------------------------------------------------

    def _run_benchmark(
        self,
        cfg: BenchConfig,
        step_fn: Optional[Any] = None,
    ) -> BenchResult:
        """Execute the timed training loop and collect statistics."""
        total_steps = cfg.warmup_steps + cfg.measured_steps
        if total_steps < 1:
            raise ValueError(
                f"warmup_steps ({cfg.warmup_steps}) + measured_steps "
                f"({cfg.measured_steps}) must be >= 1."
            )

        step_times_ms: List[float] = []
        tokens_per_step = cfg.per_gpu_batch_size * cfg.seq_len * cfg.world_size

        # Reset GPU memory stats if available
        memory_peak_gb = 0.0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for step in range(total_steps):
            t_start = time.perf_counter()

            if step_fn is not None:
                step_fn(step)
            else:
                # Synthetic step: simulate GPU work
                self._synthetic_step(cfg)

            t_end = time.perf_counter()
            elapsed_ms = (t_end - t_start) * 1000.0

            # Only record after warmup
            if step >= cfg.warmup_steps:
                step_times_ms.append(elapsed_ms)

        # Collect peak memory
        if torch.cuda.is_available():
            memory_peak_gb = (
                torch.cuda.max_memory_allocated() / (1024 ** 3)
            )

        if not step_times_ms:
            raise ValueError("No steps were measured (all steps were warmup).")

        # Compute statistics
        step_time_p50_ms = statistics.median(step_times_ms)
        step_times_sorted = sorted(step_times_ms)
        p90_idx = min(
            int(0.9 * len(step_times_sorted)),
            len(step_times_sorted) - 1,
        )
        step_time_p90_ms = step_times_sorted[p90_idx]

        # Throughput in tokens/sec
        step_times_sec = [t / 1000.0 for t in step_times_ms]
        throughputs = [tokens_per_step / t for t in step_times_sec if t > 0]

        if not throughputs:
            throughput_p50 = 0.0
            throughput_p90 = 0.0
        else:
            throughput_p50 = statistics.median(throughputs)
            throughputs_sorted = sorted(throughputs)
            p10_idx = max(0, int(0.1 * len(throughputs_sorted)) - 1)
            throughput_p90 = throughputs_sorted[p10_idx]

        return BenchResult(
            strategy=cfg.strategy,
            world_size=cfg.world_size,
            throughput_p50=throughput_p50,
            throughput_p90=throughput_p90,
            step_time_p50_ms=step_time_p50_ms,
            step_time_p90_ms=step_time_p90_ms,
            memory_peak_gb=memory_peak_gb,
            measured_steps=len(step_times_ms),
            warmup_steps=cfg.warmup_steps,
            per_gpu_batch_size=cfg.per_gpu_batch_size,
            seq_len=cfg.seq_len,
        )

    @staticmethod
    def _synthetic_step(cfg: BenchConfig) -> None:
        """Perform a synthetic computation to simulate a training step."""
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        # Small matmul to simulate work — scaled by batch and seq_len
        size = max(64, min(512, cfg.per_gpu_batch_size * 16))
        x = torch.randn(size, size, device=device)
        y = torch.matmul(x, x.T)
        z = y.sum()
        # Simulate backward
        _ = z.item()

        if device == "cuda":
            torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # compute_efficiency
    # ------------------------------------------------------------------

    def compute_efficiency(
        self,
        single: BenchResult,
        multi: BenchResult,
        cfg: Optional[BenchConfig] = None,
    ) -> ScalingReport:
        """Compute scaling efficiency from single and multi GPU results.

        Parameters
        ----------
        single:
            BenchResult from world_size=1 run.
        multi:
            BenchResult from world_size=N run.
        cfg:
            Optional BenchConfig for metadata fields.

        Returns
        -------
        ScalingReport

        Raises
        ------
        ValueError
            If single.throughput_p50 is zero (invalid baseline).
        ZeroDivisionError
            (covered by ValueError above)
        """
        if single.throughput_p50 <= 0:
            raise ValueError(
                f"Single-GPU throughput is {single.throughput_p50}, "
                "cannot compute scaling efficiency with zero or negative baseline."
            )

        n = multi.world_size
        perfect_throughput = n * single.throughput_p50
        scaling_efficiency = multi.throughput_p50 / perfect_throughput

        timestamp = datetime.now(tz=timezone.utc).isoformat()

        git_commit = _get_git_commit()

        return ScalingReport(
            strategy=multi.strategy,
            world_size=n,
            throughput_1=single.throughput_p50,
            throughput_n=multi.throughput_p50,
            scaling_efficiency=scaling_efficiency,
            memory_peak_gb=multi.memory_peak_gb,
            step_time_p50_ms=multi.step_time_p50_ms,
            step_time_p90_ms=multi.step_time_p90_ms,
            throughput_p50=multi.throughput_p50,
            throughput_p90=multi.throughput_p90,
            model_params_b=cfg.model_params_b if cfg else 0.0,
            per_gpu_batch_size=multi.per_gpu_batch_size,
            seq_len=multi.seq_len,
            mixed_precision=cfg.mixed_precision if cfg else "none",
            activation_checkpointing=cfg.activation_checkpointing if cfg else False,
            warmup_steps=multi.warmup_steps,
            measured_steps=multi.measured_steps,
            timestamp=timestamp,
            git_commit=git_commit,
        )

    # ------------------------------------------------------------------
    # format_report
    # ------------------------------------------------------------------

    def format_report(self, report: ScalingReport) -> str:
        """Return a human-readable scaling report string.

        Parameters
        ----------
        report:
            ScalingReport instance.

        Returns
        -------
        str
            Formatted multi-line report.
        """
        efficiency_pct = report.scaling_efficiency * 100.0
        efficiency_label = _efficiency_label(report.scaling_efficiency)

        lines = [
            "=" * 60,
            "  Scaling Efficiency Report",
            "=" * 60,
            f"  Strategy:             {report.strategy}",
            f"  World Size:           {report.world_size} GPU(s)",
            f"  Mixed Precision:      {report.mixed_precision}",
            f"  Activation Ckpt:      {report.activation_checkpointing}",
            f"  Model Size:           {report.model_params_b:.2f}B params",
            f"  Per-GPU Batch:        {report.per_gpu_batch_size}",
            f"  Sequence Length:      {report.seq_len}",
            "-" * 60,
            f"  Throughput (1 GPU):   {report.throughput_1:>12,.1f} tok/sec",
            f"  Throughput (N GPUs):  {report.throughput_n:>12,.1f} tok/sec",
            f"  Scaling Efficiency:   {efficiency_pct:>10.1f}%  [{efficiency_label}]",
            f"  Memory Peak:          {report.memory_peak_gb:>10.2f} GB",
            f"  Step Time p50:        {report.step_time_p50_ms:>10.1f} ms",
            f"  Step Time p90:        {report.step_time_p90_ms:>10.1f} ms",
            f"  Measured Steps:       {report.measured_steps}",
            f"  Timestamp:            {report.timestamp}",
            "=" * 60,
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # save_metrics
    # ------------------------------------------------------------------

    def save_metrics(
        self,
        report: ScalingReport,
        output_path: str,
    ) -> None:
        """Save metrics to a JSON file compatible with perf-regression-gate.

        Parameters
        ----------
        report:
            ScalingReport to serialize.
        output_path:
            File path to write (created if not exists).
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        data = report.to_dict()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Scaling metrics saved to %s", output_path)

    def load_metrics(self, path: str) -> ScalingReport:
        """Load a metrics.json file into a ScalingReport.

        Parameters
        ----------
        path:
            Path to the metrics.json file.

        Returns
        -------
        ScalingReport
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ScalingReport.from_dict(data)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _efficiency_label(efficiency: float) -> str:
    """Return a diagnostic label for the scaling efficiency value."""
    if efficiency >= 0.95:
        return "EXCELLENT"
    elif efficiency >= 0.85:
        return "GOOD"
    elif efficiency >= 0.70:
        return "ACCEPTABLE"
    elif efficiency >= 0.50:
        return "POOR — investigate wrap policy / bucket sizes"
    else:
        return "CRITICAL — likely misconfiguration"


def _get_git_commit() -> str:
    """Return the short git commit hash, or empty string if unavailable."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# Required fields validator (used by scripts)
# ---------------------------------------------------------------------------

REQUIRED_METRICS_FIELDS = frozenset(
    [
        "strategy",
        "world_size",
        "scaling_efficiency",
        "memory_peak_gb",
        "throughput_p50",
        "step_time_p50_ms",
    ]
)


def validate_metrics_schema(data: Dict[str, Any]) -> List[str]:
    """Return list of missing required fields, empty if schema is valid."""
    missing = [f for f in REQUIRED_METRICS_FIELDS if f not in data]
    return missing


# ---------------------------------------------------------------------------
# Self-Tests
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    failures: List[str] = []

    def check(name: str, condition: bool) -> None:
        if condition:
            print(f"  PASS  {name}")
        else:
            print(f"  FAIL  {name}")
            failures.append(name)

    def expect_raises(name: str, exc_type: type, fn) -> None:
        try:
            fn()
            print(f"  FAIL  {name} (no exception raised)")
            failures.append(name)
        except exc_type:
            print(f"  PASS  {name}")
        except Exception as e:
            print(f"  FAIL  {name} (wrong exception {type(e).__name__}: {e})")
            failures.append(name)

    bench = ScalingBenchmark()

    # ------------------------------------------------------------------
    # Efficiency computation tests
    # ------------------------------------------------------------------
    print("=== Efficiency Computation tests ===")

    def make_result(strategy, world_size, throughput, step_ms=100.0):
        return BenchResult(
            strategy=strategy,
            world_size=world_size,
            throughput_p50=throughput,
            throughput_p90=throughput * 0.95,
            step_time_p50_ms=step_ms,
            step_time_p90_ms=step_ms * 1.1,
            memory_peak_gb=10.0,
            measured_steps=50,
            warmup_steps=10,
            per_gpu_batch_size=4,
            seq_len=2048,
        )

    # Perfect scaling: world_size=4, throughput=4x
    single = make_result("fsdp", 1, 10000.0)
    multi_perfect = make_result("fsdp", 4, 40000.0)
    report = bench.compute_efficiency(single, multi_perfect)
    check("perfect_scaling_efficiency_1.0", abs(report.scaling_efficiency - 1.0) < 1e-9)
    check("perfect_scaling_world_size", report.world_size == 4)
    check("perfect_scaling_throughput_1", report.throughput_1 == 10000.0)
    check("perfect_scaling_throughput_n", report.throughput_n == 40000.0)

    # 90% scaling
    multi_90 = make_result("fsdp", 4, 36000.0)
    report_90 = bench.compute_efficiency(single, multi_90)
    check("90pct_scaling", abs(report_90.scaling_efficiency - 0.9) < 1e-9)

    # Superlinear scaling
    multi_super = make_result("fsdp", 4, 42000.0)
    report_super = bench.compute_efficiency(single, multi_super)
    check("superlinear_scaling_above_1", report_super.scaling_efficiency > 1.0)

    # world_size=1 efficiency must be 1.0
    single2 = make_result("ddp", 1, 5000.0)
    multi1 = make_result("ddp", 1, 5000.0)
    report1 = bench.compute_efficiency(single2, multi1)
    check("world_size_1_efficiency_is_1", abs(report1.scaling_efficiency - 1.0) < 1e-9)

    # Zero throughput raises
    expect_raises(
        "zero_throughput_raises",
        ValueError,
        lambda: bench.compute_efficiency(make_result("ddp", 1, 0.0), multi_perfect),
    )

    # ------------------------------------------------------------------
    # Report formatting tests
    # ------------------------------------------------------------------
    print("\n=== Report Formatting tests ===")

    formatted = bench.format_report(report)
    check("format_contains_strategy", "fsdp" in formatted)
    check("format_contains_efficiency", "%" in formatted)
    check("format_contains_throughput", "tok/sec" in formatted)
    check("format_contains_world_size", "4" in formatted)

    # ------------------------------------------------------------------
    # Metrics schema tests
    # ------------------------------------------------------------------
    print("\n=== Metrics Schema tests ===")

    report_dict = report.to_dict()
    missing = validate_metrics_schema(report_dict)
    check("all_required_fields_present", len(missing) == 0)

    # Check types
    check("strategy_is_str", isinstance(report_dict["strategy"], str))
    check("world_size_is_int", isinstance(report_dict["world_size"], int))
    check(
        "scaling_efficiency_is_float",
        isinstance(report_dict["scaling_efficiency"], float),
    )
    check(
        "memory_peak_gb_is_float",
        isinstance(report_dict["memory_peak_gb"], float),
    )
    check(
        "throughput_p50_is_float",
        isinstance(report_dict["throughput_p50"], float),
    )
    check(
        "step_time_p50_ms_is_float",
        isinstance(report_dict["step_time_p50_ms"], float),
    )

    # Missing required field detection
    incomplete = {"strategy": "fsdp", "world_size": 4}
    missing2 = validate_metrics_schema(incomplete)
    check("detects_missing_fields", len(missing2) > 0)

    # ------------------------------------------------------------------
    # Save/load round-trip
    # ------------------------------------------------------------------
    print("\n=== Save/Load Round-Trip tests ===")

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        metrics_path = os.path.join(tmpdir, "metrics.json")
        bench.save_metrics(report, metrics_path)
        check("metrics_file_exists", os.path.isfile(metrics_path))

        loaded = bench.load_metrics(metrics_path)
        check(
            "loaded_strategy_matches",
            loaded.strategy == report.strategy,
        )
        check(
            "loaded_world_size_matches",
            loaded.world_size == report.world_size,
        )
        check(
            "loaded_efficiency_matches",
            abs(loaded.scaling_efficiency - report.scaling_efficiency) < 1e-9,
        )

    # ------------------------------------------------------------------
    # Synthetic benchmark run test (fast, no real model)
    # ------------------------------------------------------------------
    print("\n=== Synthetic Benchmark Run tests ===")

    cfg = BenchConfig(
        strategy="ddp",
        world_size=1,
        per_gpu_batch_size=2,
        seq_len=64,
        warmup_steps=2,
        measured_steps=5,
    )
    result = bench.run_single(cfg)
    check("run_single_positive_throughput", result.throughput_p50 > 0)
    check("run_single_measured_steps", result.measured_steps == 5)
    check("run_single_step_time_positive", result.step_time_p50_ms > 0)

    # ------------------------------------------------------------------
    # Efficiency label
    # ------------------------------------------------------------------
    print("\n=== Efficiency Label tests ===")

    check("label_excellent", "EXCELLENT" in _efficiency_label(0.97))
    check("label_good", "GOOD" in _efficiency_label(0.88))
    check("label_acceptable", "ACCEPTABLE" in _efficiency_label(0.75))
    check("label_poor", "POOR" in _efficiency_label(0.60))
    check("label_critical", "CRITICAL" in _efficiency_label(0.30))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*50}")
    if failures:
        print(f"FAIL: {len(failures)} test(s) failed: {failures}")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")
        sys.exit(0)
