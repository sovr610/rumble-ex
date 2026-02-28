#!/usr/bin/env python3
"""
scaling_report.py
-----------------
CLI tool for running scaling benchmarks and generating comparison reports.

Usage
-----
    # Run a benchmark for FSDP at world_size=4
    python scripts/scaling_report.py --strategy fsdp --world_size 4

    # Compare pre-computed metrics files
    python scripts/scaling_report.py --compare \
        --metrics_dir /path/to/metrics/ \
        --output_dir /path/to/reports/

    # Run benchmark and output to specific directory
    python scripts/scaling_report.py \
        --strategy deepspeed_zero3 \
        --world_size 8 \
        --output_dir /tmp/reports/

    # Update the perf-regression baseline
    python scripts/scaling_report.py \
        --strategy fsdp --world_size 4 --update-baseline

Outputs
-------
    scaling_report.json  — Machine-readable report in metrics.json format
    scaling_report.txt   — Human-readable table of results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(SKILL_DIR, "assets")
if ASSETS_DIR not in sys.path:
    sys.path.insert(0, ASSETS_DIR)


# ---------------------------------------------------------------------------
# Imports from assets
# ---------------------------------------------------------------------------


def _import_assets():
    """Import asset modules, returning (ScalingBenchmark, BenchConfig, BenchResult, ScalingReport)."""
    try:
        from scaling_benchmark_template import (
            BenchConfig,
            BenchResult,
            ScalingBenchmark,
            ScalingReport,
            validate_metrics_schema,
        )
        return ScalingBenchmark, BenchConfig, BenchResult, ScalingReport, validate_metrics_schema
    except ImportError as e:
        print(f"ERROR: Cannot import scaling_benchmark_template: {e}", file=sys.stderr)
        print(f"  Add {ASSETS_DIR} to PYTHONPATH or run from the skill directory.", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def build_comparison_table(reports: List[Dict[str, Any]]) -> str:
    """Build a human-readable comparison table from a list of report dicts."""
    if not reports:
        return "No reports to compare.\n"

    headers = [
        "Strategy",
        "GPUs",
        "Efficiency",
        "Throughput (p50)",
        "Mem Peak (GB)",
        "Step p50 (ms)",
        "Step p90 (ms)",
    ]

    col_widths = [max(len(h), 20) for h in headers]
    col_widths[0] = max(col_widths[0], max(len(r.get("strategy", "")) for r in reports))

    def fmt_row(cols):
        return "  ".join(str(c).ljust(w) for c, w in zip(cols, col_widths))

    sep = "  ".join("-" * w for w in col_widths)

    lines = [
        "",
        "Scaling Efficiency Comparison",
        "=" * len(sep),
        fmt_row(headers),
        sep,
    ]

    for r in sorted(reports, key=lambda x: (x.get("strategy", ""), x.get("world_size", 0))):
        eff = r.get("scaling_efficiency", 0.0)
        eff_str = f"{eff * 100:.1f}%"
        tp = r.get("throughput_p50", 0.0)
        tp_str = f"{tp:,.0f} tok/s"
        mem = r.get("memory_peak_gb", 0.0)
        st_p50 = r.get("step_time_p50_ms", 0.0)
        st_p90 = r.get("step_time_p90_ms", 0.0)

        lines.append(
            fmt_row([
                r.get("strategy", "unknown"),
                r.get("world_size", "?"),
                eff_str,
                tp_str,
                f"{mem:.1f}",
                f"{st_p50:.1f}",
                f"{st_p90:.1f}",
            ])
        )

    lines.append(sep)
    lines.append("")
    lines.append(f"Generated: {datetime.now(tz=timezone.utc).isoformat()}")
    lines.append("")
    return "\n".join(lines)


def load_metrics_dir(metrics_dir: str) -> List[Dict[str, Any]]:
    """Load all metrics.json files from a directory."""
    reports = []
    if not os.path.isdir(metrics_dir):
        print(f"WARNING: metrics_dir '{metrics_dir}' does not exist.", file=sys.stderr)
        return reports

    for fname in sorted(os.listdir(metrics_dir)):
        if fname.endswith(".json"):
            fpath = os.path.join(metrics_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                reports.append(data)
            except (json.JSONDecodeError, OSError) as e:
                print(f"WARNING: Skipping {fpath}: {e}", file=sys.stderr)

    return reports


def save_report_files(
    reports: List[Dict[str, Any]],
    output_dir: str,
    primary_report: Optional[Dict[str, Any]] = None,
) -> None:
    """Save scaling_report.json and scaling_report.txt to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    # scaling_report.json: list of all reports (or just the primary one)
    report_data = primary_report if primary_report is not None else reports
    json_path = os.path.join(output_dir, "scaling_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"  Saved: {json_path}")

    # scaling_report.txt: comparison table
    all_reports = reports if reports else ([primary_report] if primary_report else [])
    table = build_comparison_table(all_reports)
    txt_path = os.path.join(output_dir, "scaling_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(table)
    print(f"  Saved: {txt_path}")
    print(table)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(
    strategy: str,
    world_size: int,
    per_gpu_batch_size: int,
    seq_len: int,
    warmup_steps: int,
    measured_steps: int,
    mixed_precision: str,
    activation_checkpointing: bool,
    model_params_b: float,
) -> Dict[str, Any]:
    """Run 1-GPU and N-GPU benchmarks and return the ScalingReport as a dict."""
    (
        ScalingBenchmark,
        BenchConfig,
        BenchResult,
        ScalingReport,
        validate_metrics_schema,
    ) = _import_assets()

    bench = ScalingBenchmark()

    # Single GPU baseline
    cfg_single = BenchConfig(
        strategy=strategy,
        world_size=1,
        per_gpu_batch_size=per_gpu_batch_size,
        seq_len=seq_len,
        warmup_steps=warmup_steps,
        measured_steps=measured_steps,
        mixed_precision=mixed_precision,
        activation_checkpointing=activation_checkpointing,
        model_params_b=model_params_b,
    )

    print(f"\nRunning 1-GPU baseline benchmark...")
    print(f"  strategy={strategy}, seq_len={seq_len}, batch={per_gpu_batch_size}")
    single_result = bench.run_single(cfg_single)
    print(
        f"  Throughput (p50): {single_result.throughput_p50:,.1f} tok/sec"
        f"  Step time (p50): {single_result.step_time_p50_ms:.1f} ms"
    )

    if world_size == 1:
        # No multi-GPU run needed
        report = bench.compute_efficiency(single_result, single_result, cfg=cfg_single)
    else:
        # Multi-GPU run
        cfg_multi = BenchConfig(
            strategy=strategy,
            world_size=world_size,
            per_gpu_batch_size=per_gpu_batch_size,
            seq_len=seq_len,
            warmup_steps=warmup_steps,
            measured_steps=measured_steps,
            mixed_precision=mixed_precision,
            activation_checkpointing=activation_checkpointing,
            model_params_b=model_params_b,
        )
        print(f"\nRunning {world_size}-GPU benchmark...")
        multi_result = bench.run_multi(cfg_multi)
        print(
            f"  Throughput (p50): {multi_result.throughput_p50:,.1f} tok/sec"
            f"  Step time (p50): {multi_result.step_time_p50_ms:.1f} ms"
        )
        report = bench.compute_efficiency(single_result, multi_result, cfg=cfg_multi)

    eff_pct = report.scaling_efficiency * 100.0
    print(f"\n  Scaling Efficiency: {eff_pct:.1f}%")
    print(bench.format_report(report))

    return report.to_dict()


# ---------------------------------------------------------------------------
# Baseline management
# ---------------------------------------------------------------------------


def update_baseline(report: Dict[str, Any], output_dir: str) -> None:
    """Write report as the new perf-regression-gate baseline."""
    strategy = report.get("strategy", "unknown")
    world_size = report.get("world_size", 1)
    baseline_path = os.path.join(
        output_dir, f"baseline_{strategy}_w{world_size}.json"
    )
    os.makedirs(output_dir, exist_ok=True)
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  Baseline updated: {baseline_path}")


def compare_to_baseline(
    report: Dict[str, Any],
    output_dir: str,
    regression_threshold: float = 0.05,
) -> bool:
    """Compare report against baseline. Returns True if within threshold."""
    strategy = report.get("strategy", "unknown")
    world_size = report.get("world_size", 1)
    baseline_path = os.path.join(
        output_dir, f"baseline_{strategy}_w{world_size}.json"
    )

    if not os.path.exists(baseline_path):
        print(f"  No baseline found at {baseline_path}. Skipping regression check.")
        return True

    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    baseline_tp = baseline.get("throughput_p50", 0.0)
    current_tp = report.get("throughput_p50", 0.0)

    if baseline_tp <= 0:
        print("  Baseline throughput is zero. Skipping regression check.")
        return True

    regression_ratio = (baseline_tp - current_tp) / baseline_tp
    passed = regression_ratio <= regression_threshold

    print(f"\n  Regression Check:")
    print(f"    Baseline throughput:  {baseline_tp:,.1f} tok/sec")
    print(f"    Current throughput:   {current_tp:,.1f} tok/sec")
    print(f"    Regression:           {regression_ratio * 100:.1f}%")
    print(f"    Threshold:            {regression_threshold * 100:.1f}%")
    print(f"    Result:               {'PASS' if passed else 'FAIL'}")

    return passed


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run distributed training scaling benchmarks and generate reports.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default="fsdp",
        choices=["ddp", "fsdp", "deepspeed_zero2", "deepspeed_zero3"],
        help="Distributed training strategy to benchmark.",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="Number of GPUs for the N-GPU run. Use 1 for baseline only.",
    )
    parser.add_argument(
        "--per_gpu_batch_size",
        type=int,
        default=4,
        help="Batch size per GPU.",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=2048,
        help="Sequence length in tokens.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=10,
        help="Steps to discard before measuring (warm-up).",
    )
    parser.add_argument(
        "--measured_steps",
        type=int,
        default=50,
        help="Steps to include in statistics.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "none"],
        help="Mixed precision setting.",
    )
    parser.add_argument(
        "--activation_checkpointing",
        action="store_true",
        default=False,
        help="Enable activation checkpointing during benchmark.",
    )
    parser.add_argument(
        "--model_params_b",
        type=float,
        default=0.0,
        help="Model size in billions of parameters (metadata only).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./scaling_reports",
        help="Directory to write scaling_report.json and scaling_report.txt.",
    )
    parser.add_argument(
        "--metrics_dir",
        type=str,
        default=None,
        help="Directory containing pre-computed metrics.json files to compare.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        default=False,
        help="Load and compare all metrics.json files from --metrics_dir.",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        default=False,
        help="After running, write the result as the new perf-regression baseline.",
    )
    parser.add_argument(
        "--regression-threshold",
        type=float,
        default=0.05,
        help="Maximum allowed throughput regression fraction (0.05 = 5%%).",
    )

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    print("=" * 60)
    print("  Distributed Scaling Report Tool")
    print("=" * 60)

    (
        ScalingBenchmark,
        BenchConfig,
        BenchResult,
        ScalingReport,
        validate_metrics_schema,
    ) = _import_assets()

    # --- Compare mode: load existing metrics files ---
    if args.compare:
        metrics_dir = args.metrics_dir or args.output_dir
        print(f"\nLoading metrics from: {metrics_dir}")
        reports = load_metrics_dir(metrics_dir)
        if not reports:
            print(f"No metrics.json files found in '{metrics_dir}'.", file=sys.stderr)
            return 1
        print(f"Loaded {len(reports)} report(s).")
        save_report_files(reports, args.output_dir)
        return 0

    # --- Benchmark mode ---
    report_dict = run_benchmark(
        strategy=args.strategy,
        world_size=args.world_size,
        per_gpu_batch_size=args.per_gpu_batch_size,
        seq_len=args.seq_len,
        warmup_steps=args.warmup_steps,
        measured_steps=args.measured_steps,
        mixed_precision=args.mixed_precision,
        activation_checkpointing=args.activation_checkpointing,
        model_params_b=args.model_params_b,
    )

    # Validate schema
    missing = validate_metrics_schema(report_dict)
    if missing:
        print(f"WARNING: metrics schema missing fields: {missing}", file=sys.stderr)

    # Save output
    print(f"\nSaving reports to: {args.output_dir}")
    existing_reports = []
    if args.metrics_dir:
        existing_reports = load_metrics_dir(args.metrics_dir)
    all_reports = existing_reports + [report_dict]
    save_report_files(all_reports, args.output_dir, primary_report=report_dict)

    # Update baseline if requested
    if args.update_baseline:
        print("\nUpdating performance baseline...")
        update_baseline(report_dict, args.output_dir)

    # Regression check
    regression_ok = compare_to_baseline(
        report_dict,
        args.output_dir,
        regression_threshold=args.regression_threshold,
    )

    print("\n" + "=" * 60)
    if regression_ok:
        print(f"  DONE: Reports written to {args.output_dir}")
        return 0
    else:
        print(f"  REGRESSION DETECTED: Throughput below baseline threshold.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
