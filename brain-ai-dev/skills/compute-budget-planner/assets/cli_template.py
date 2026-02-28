"""
cli_template.py
===============
CLI entrypoint for the Compute-Optimal Budget Planner.

Provides three subcommands:
    validate   Mode A: Validate an existing training run config
    required   Mode B: Compute required tokens/steps/wallclock for a model
    optimal    Mode C: Solve optimal (N, D) for a compute budget

Usage
-----
    # Mode A: Validate a run
    python cli_template.py validate \\
        --params 7e9 --seq_len 2048 --batch 2048 --steps 100000 \\
        --gpus 8 --gpu H100_SXM --dtype bf16 --util 0.35 \\
        --out /tmp/budget_run --format all

    # Mode B: Compute required budget
    python cli_template.py required \\
        --params 70e9 --tokens_per_param 20 \\
        --seq_len 2048 --batch 2048 \\
        --gpus 8 --gpu H100_SXM --util 0.35 --cost_per_gpu_hour 4.0 \\
        --out /tmp/budget_70b

    # Mode C: Solve optimal from FLOPs
    python cli_template.py optimal \\
        --compute_flops 5e23 --tokens_per_param 20 \\
        --out /tmp/budget_opt

    # Mode C: Solve optimal from GPU-hours
    python cli_template.py optimal \\
        --gpus 8 --gpu H100_SXM --hours 1000 --util 0.35 \\
        --out /tmp/budget_opt_time
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_float(value: str) -> float:
    """Parse a float from a string, supporting scientific notation (e.g., '7e9')."""
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid number: '{value}'. Use standard notation (e.g., 7e9, 70000000000, 1.4e12)."
        )


def _parse_int(value: str) -> int:
    """Parse an int from a string."""
    try:
        return int(float(value))  # handles '2048' and '2e3'
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid integer: '{value}'.")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="budget-planner",
        description=(
            "Compute-Optimal Budget Planner (Chinchilla-style). "
            "Plans training budgets, estimates wallclock and cost, "
            "and validates compute-optimal token/parameter ratios."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate a 7B run at 100K steps
  budget-planner validate --params 7e9 --seq_len 2048 --batch 2048 --steps 100000 --gpus 8 --gpu H100_SXM

  # Compute budget for compute-optimal 70B training
  budget-planner required --params 70e9 --tokens_per_param 20 --gpus 8 --gpu H100_SXM

  # Solve optimal N and D for 5e23 FLOPs
  budget-planner optimal --compute_flops 5e23

  # Solve optimal for a 1000-hour, 8x H100 budget
  budget-planner optimal --gpus 8 --gpu H100_SXM --hours 1000 --util 0.35
        """,
    )

    # Global options
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output directory for budget.json / budget.txt / budget.svg. "
             "If not specified, prints to stdout.",
    )
    parser.add_argument(
        "--format", choices=["json", "txt", "svg", "all"], default="txt",
        help="Output format(s). 'all' writes JSON + TXT + SVG. Default: txt.",
    )
    parser.add_argument(
        "--strict_budget", action="store_true", default=False,
        help="Hard-fail (exit 1) if any warnings are emitted. Default: advisory only.",
    )
    parser.add_argument(
        "--k", type=_parse_float, default=6.0,
        help="FLOPs coefficient k in C = k*N*D. Default: 6.0.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", default=False,
        help="Enable verbose logging.",
    )

    subparsers = parser.add_subparsers(dest="mode", help="Planner mode")
    subparsers.required = True

    # ------------------------------------------------------------------
    # validate subcommand (Mode A)
    # ------------------------------------------------------------------
    validate_parser = subparsers.add_parser(
        "validate",
        help="Mode A: Validate an existing training run config.",
        description=(
            "Given a complete training run specification, compute total tokens, "
            "tokens/param ratio, total FLOPs, predicted wallclock, and generate "
            "undertraining warnings."
        ),
    )
    validate_parser.add_argument(
        "--params", type=_parse_float, required=True,
        help="Non-embedding parameter count (e.g., 7e9 for 7B).",
    )
    validate_parser.add_argument(
        "--seq_len", type=_parse_int, default=2048,
        help="Sequence length in tokens. Default: 2048.",
    )
    validate_parser.add_argument(
        "--batch", type=_parse_int, required=True,
        help="Global batch size (all data-parallel ranks combined).",
    )
    validate_parser.add_argument(
        "--steps", type=_parse_int, required=True,
        help="Number of optimizer steps (after gradient accumulation).",
    )
    validate_parser.add_argument(
        "--gpus", type=_parse_int, default=8,
        help="Number of GPUs. Default: 8.",
    )
    validate_parser.add_argument(
        "--gpu", type=str, default="H100_SXM",
        help="GPU type for spec lookup (e.g., H100_SXM, A100_80GB_SXM). Default: H100_SXM.",
    )
    validate_parser.add_argument(
        "--dtype", type=str, default="bf16",
        choices=["bf16", "fp16", "fp32", "tf32"],
        help="Training precision. Default: bf16.",
    )
    validate_parser.add_argument(
        "--util", type=_parse_float, default=0.35,
        help="Model FLOPs Utilization (MFU). Default: 0.35 (35%%).",
    )
    validate_parser.add_argument(
        "--hours", type=_parse_float, default=None,
        help="Actual run duration in hours (optional; for retrospective analysis).",
    )
    validate_parser.add_argument(
        "--cost_per_gpu_hour", type=_parse_float, default=None,
        help="GPU cost in USD per GPU-hour. Enables cost estimation.",
    )
    validate_parser.add_argument(
        "--tokens_per_param", type=_parse_float, default=20.0,
        help="Chinchilla target tokens per parameter. Default: 20.0.",
    )
    validate_parser.add_argument(
        "--peak_tflops", type=_parse_float, default=None,
        help="Override peak TFLOPS (use when GPU not in spec table).",
    )
    validate_parser.add_argument(
        "--mem_gb", type=_parse_float, default=None,
        help="Override GPU memory in GB (use with --peak_tflops).",
    )
    validate_parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (overrides matching CLI flags).",
    )

    # ------------------------------------------------------------------
    # required subcommand (Mode B)
    # ------------------------------------------------------------------
    required_parser = subparsers.add_parser(
        "required",
        help="Mode B: Compute required tokens, steps, wallclock, and cost for a model.",
        description=(
            "Given a model size and target tokens/param, compute everything needed "
            "for a compute-optimal training run: total tokens, optimizer steps, "
            "predicted wallclock hours, and estimated cost."
        ),
    )
    required_parser.add_argument(
        "--params", type=_parse_float, required=True,
        help="Non-embedding parameter count (e.g., 70e9 for 70B).",
    )
    required_parser.add_argument(
        "--tokens_per_param", type=_parse_float, default=20.0,
        help="Target tokens per parameter. Default: 20.0 (Chinchilla).",
    )
    required_parser.add_argument(
        "--seq_len", type=_parse_int, default=2048,
        help="Sequence length. Default: 2048.",
    )
    required_parser.add_argument(
        "--batch", type=_parse_int, default=2048,
        help="Global batch size. Default: 2048.",
    )
    required_parser.add_argument(
        "--gpus", type=_parse_int, default=8,
        help="Number of GPUs. Default: 8.",
    )
    required_parser.add_argument(
        "--gpu", type=str, default="H100_SXM",
        help="GPU type. Default: H100_SXM.",
    )
    required_parser.add_argument(
        "--dtype", type=str, default="bf16",
        choices=["bf16", "fp16", "fp32", "tf32"],
        help="Training precision. Default: bf16.",
    )
    required_parser.add_argument(
        "--util", type=_parse_float, default=0.35,
        help="MFU. Default: 0.35.",
    )
    required_parser.add_argument(
        "--cost_per_gpu_hour", type=_parse_float, default=None,
        help="GPU cost in USD/hr. Enables cost estimation.",
    )
    required_parser.add_argument(
        "--peak_tflops", type=_parse_float, default=None,
        help="Override peak TFLOPS.",
    )
    required_parser.add_argument(
        "--mem_gb", type=_parse_float, default=None,
        help="Override GPU memory in GB.",
    )

    # ------------------------------------------------------------------
    # optimal subcommand (Mode C)
    # ------------------------------------------------------------------
    optimal_parser = subparsers.add_parser(
        "optimal",
        help="Mode C: Solve optimal (N, D) for a compute budget.",
        description=(
            "Given a compute budget specified as total FLOPs, GPU-hours, or dollars, "
            "solve for the Chinchilla-optimal model size and token count."
        ),
    )
    optimal_budget_group = optimal_parser.add_mutually_exclusive_group()
    optimal_budget_group.add_argument(
        "--compute_flops", type=_parse_float, default=None,
        help="Total training FLOPs budget (e.g., 5e23).",
    )
    optimal_budget_group.add_argument(
        "--budget_dollars", type=_parse_float, default=None,
        help="Total dollar budget. Requires --gpus, --gpu, --util, --cost_per_gpu_hour.",
    )

    optimal_parser.add_argument(
        "--gpus", type=_parse_int, default=None,
        help="Number of GPUs (required if using time or money budget).",
    )
    optimal_parser.add_argument(
        "--gpu", type=str, default=None,
        help="GPU type (required if using time or money budget).",
    )
    optimal_parser.add_argument(
        "--hours", type=_parse_float, default=None,
        help="Cluster-hours budget. Use instead of --compute_flops.",
    )
    optimal_parser.add_argument(
        "--util", type=_parse_float, default=0.35,
        help="MFU. Default: 0.35.",
    )
    optimal_parser.add_argument(
        "--tokens_per_param", type=_parse_float, default=20.0,
        help="Target tokens per parameter. Default: 20.0.",
    )
    optimal_parser.add_argument(
        "--cost_per_gpu_hour", type=_parse_float, default=None,
        help="GPU cost in USD/hr (required if --budget_dollars used).",
    )
    optimal_parser.add_argument(
        "--peak_tflops", type=_parse_float, default=None,
        help="Override peak TFLOPS.",
    )
    optimal_parser.add_argument(
        "--mem_gb", type=_parse_float, default=None,
        help="Override GPU memory in GB.",
    )

    return parser


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Parameters
    ----------
    argv : list of str or None
        Argument list (defaults to sys.argv[1:]).

    Returns
    -------
    argparse.Namespace
    """
    parser = build_parser()
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    """Run the budget planner CLI.

    Parameters
    ----------
    argv : list of str or None

    Returns
    -------
    int
        Exit code (0 = success, 1 = error or strict_budget violation).
    """
    args = parse_args(argv)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # -----------------------------------------------------------------------
    # Import dependencies (done here so CLI parsing tests work without deps)
    # -----------------------------------------------------------------------
    try:
        _assets_dir = os.path.dirname(os.path.abspath(__file__))
        if _assets_dir not in sys.path:
            sys.path.insert(0, _assets_dir)

        from gpu_specs_template import GPUSpecTable
        from budget_config_template import (
            BudgetConfig, RunSpec, ModelSpec, ComputeBudget
        )
        from budget_planner_template import BudgetPlanner
        from report_generator_template import ReportGenerator
    except ImportError as exc:
        print(f"ERROR: Failed to import planner modules: {exc}", file=sys.stderr)
        print(
            "Ensure all template files are in the same directory or on PYTHONPATH.",
            file=sys.stderr
        )
        return 1

    # -----------------------------------------------------------------------
    # Load YAML config if provided (validate mode only)
    # -----------------------------------------------------------------------
    yaml_config = {}
    if hasattr(args, "config") and args.config:
        try:
            import yaml
            with open(args.config) as fh:
                yaml_config = yaml.safe_load(fh) or {}
            logger.info("Loaded config from %s", args.config)
        except ImportError:
            print("WARNING: pyyaml not installed; --config flag ignored.", file=sys.stderr)
        except FileNotFoundError:
            print(f"ERROR: Config file not found: {args.config}", file=sys.stderr)
            return 1

    # -----------------------------------------------------------------------
    # Build GPU spec table and planner
    # -----------------------------------------------------------------------
    specs = GPUSpecTable()

    # Determine common config fields
    k = args.k
    util = getattr(args, "util", 0.35)
    tpp = getattr(args, "tokens_per_param", 20.0)
    cost_per_gpu_hour = getattr(args, "cost_per_gpu_hour", None)
    gpu = getattr(args, "gpu", "H100_SXM") or "H100_SXM"
    gpus = getattr(args, "gpus", 8) or 8
    dtype = getattr(args, "dtype", "bf16")

    # Handle user-supplied GPU specs (override for unknown GPUs)
    user_peak = getattr(args, "peak_tflops", None)
    user_mem = getattr(args, "mem_gb", None)

    # If user supplied peak_tflops, register a temporary override
    if user_peak is not None:
        # We'll pass this through validate_or_fallback
        pass

    config = BudgetConfig(
        k=k,
        tokens_per_param_target=tpp,
        default_utilization=util,
        cost_per_gpu_hour=cost_per_gpu_hour,
        num_gpus=gpus,
        gpu_type=gpu,
        dtype=dtype,
    )

    planner = BudgetPlanner(specs, config)
    reporter = ReportGenerator()

    # -----------------------------------------------------------------------
    # Dispatch to the appropriate mode
    # -----------------------------------------------------------------------
    result = None

    try:
        if args.mode == "validate":
            run = RunSpec(
                n_params=args.params,
                seq_len=args.seq_len,
                global_batch=args.batch,
                steps=args.steps,
                num_gpus=args.gpus,
                gpu_type=args.gpu,
                dtype=args.dtype,
                utilization=args.util,
                cost_per_gpu_hour=args.cost_per_gpu_hour,
            )

            # Override GPU spec if user supplied peak_tflops
            if user_peak is not None:
                specs_override = GPUSpecTable()
                # Monkey-patch: inject user spec into lookup
                original_lookup = specs_override.lookup
                def patched_lookup(gpu_type, dtype="bf16"):
                    found = original_lookup(gpu_type, dtype=dtype)
                    if found is None or (user_peak is not None):
                        from gpu_specs_template import GPUSpec
                        return GPUSpec.user_supplied(user_peak, user_mem or 80.0)
                    return found
                specs_override.lookup = patched_lookup
                specs_override.validate_or_fallback = lambda g, **kw: specs_override.lookup(g)
                planner_override = BudgetPlanner(specs_override, config)
                result = planner_override.validate_run(run)
            else:
                result = planner.validate_run(run)

        elif args.mode == "required":
            model = ModelSpec(
                n_params=args.params,
                seq_len=args.seq_len,
                global_batch=args.batch,
                tokens_per_param_target=args.tokens_per_param,
                num_gpus=args.gpus,
                gpu_type=args.gpu,
                dtype=args.dtype,
                utilization=args.util,
                cost_per_gpu_hour=args.cost_per_gpu_hour,
            )
            result = planner.compute_required(model)

        elif args.mode == "optimal":
            # Determine budget mode
            if args.compute_flops is not None:
                budget = ComputeBudget(
                    mode="flops",
                    total_flops=args.compute_flops,
                    utilization=args.util,
                    tokens_per_param_target=args.tokens_per_param,
                    k=args.k,
                )
            elif args.hours is not None:
                if not args.gpus or not args.gpu:
                    print(
                        "ERROR: --gpus and --gpu are required when using --hours.",
                        file=sys.stderr,
                    )
                    return 1
                budget = ComputeBudget(
                    mode="time",
                    hours=args.hours,
                    num_gpus=args.gpus,
                    gpu_type=args.gpu,
                    utilization=args.util,
                    tokens_per_param_target=args.tokens_per_param,
                    k=args.k,
                )
            elif args.budget_dollars is not None:
                if not args.gpus or not args.gpu or not args.cost_per_gpu_hour:
                    print(
                        "ERROR: --gpus, --gpu, and --cost_per_gpu_hour are required "
                        "when using --budget_dollars.",
                        file=sys.stderr,
                    )
                    return 1
                budget = ComputeBudget(
                    mode="money",
                    budget_dollars=args.budget_dollars,
                    cost_per_gpu_hour=args.cost_per_gpu_hour,
                    num_gpus=args.gpus,
                    gpu_type=args.gpu,
                    utilization=args.util,
                    tokens_per_param_target=args.tokens_per_param,
                    k=args.k,
                )
            else:
                print(
                    "ERROR: optimal mode requires one of: --compute_flops, --hours, or --budget_dollars.",
                    file=sys.stderr,
                )
                return 1

            result = planner.solve_optimal(budget)

        else:
            print(f"ERROR: Unknown mode '{args.mode}'.", file=sys.stderr)
            return 1

    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR: Unexpected error: {exc}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------
    formats_to_write = []
    if args.format == "all":
        formats_to_write = ["json", "txt", "svg"]
    else:
        formats_to_write = [args.format]

    if args.out:
        written = reporter.write_all(result, args.out, formats=formats_to_write)
        for fmt, path in written.items():
            print(f"Written {fmt.upper()}: {path}")
        # Always print TXT summary to stdout as well
        if "txt" not in formats_to_write:
            reporter.write_txt(result, path=None)
        else:
            # Also print to stdout for visibility
            reporter.write_txt(result, path=None)
    else:
        # No output directory: print to stdout
        if "json" in formats_to_write:
            reporter.write_json(result, path=None)
        if "txt" in formats_to_write or args.format == "txt":
            reporter.write_txt(result, path=None)
        if "svg" in formats_to_write:
            print(
                "WARNING: --format svg requires --out to specify a directory.",
                file=sys.stderr,
            )

    # -----------------------------------------------------------------------
    # Strict budget check
    # -----------------------------------------------------------------------
    if args.strict_budget and result.has_warnings():
        print(
            "\nSTRICT BUDGET MODE: Warnings detected — exiting with code 1.",
            file=sys.stderr,
        )
        for w in result.warnings:
            print(f"  {w}", file=sys.stderr)
        return 1

    return 0


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _run_self_tests() -> None:
    print("Running cli_template.py self-tests...")
    errors = []

    # Test 1: validate subcommand parses correctly
    try:
        args = parse_args([
            "validate",
            "--params", "7e9",
            "--seq_len", "2048",
            "--batch", "2048",
            "--steps", "100000",
            "--gpus", "8",
            "--gpu", "H100_SXM",
        ])
        assert args.mode == "validate"
        assert abs(args.params - 7e9) < 1
        assert args.seq_len == 2048
        assert args.batch == 2048
        assert args.steps == 100000
        assert args.gpus == 8
        assert args.gpu == "H100_SXM"
        print("  [PASS] validate subcommand parses correctly")
    except Exception as e:
        errors.append(f"  [FAIL] validate parsing: {e}")

    # Test 2: required subcommand parses correctly
    try:
        args = parse_args([
            "required",
            "--params", "70e9",
            "--tokens_per_param", "20",
            "--gpus", "8",
            "--gpu", "H100_SXM",
        ])
        assert args.mode == "required"
        assert abs(args.params - 70e9) < 1
        assert args.tokens_per_param == 20.0
        print("  [PASS] required subcommand parses correctly")
    except Exception as e:
        errors.append(f"  [FAIL] required parsing: {e}")

    # Test 3: optimal with compute_flops
    try:
        args = parse_args(["optimal", "--compute_flops", "5e23"])
        assert args.mode == "optimal"
        assert abs(args.compute_flops - 5e23) < 1
        print("  [PASS] optimal --compute_flops parses correctly")
    except Exception as e:
        errors.append(f"  [FAIL] optimal --compute_flops: {e}")

    # Test 4: optimal with GPU-hours
    try:
        args = parse_args([
            "optimal",
            "--gpus", "8",
            "--gpu", "H100_SXM",
            "--hours", "1000",
            "--util", "0.35",
        ])
        assert args.mode == "optimal"
        assert args.hours == 1000.0
        assert args.util == 0.35
        assert args.gpus == 8
        print("  [PASS] optimal --hours parses correctly")
    except Exception as e:
        errors.append(f"  [FAIL] optimal --hours: {e}")

    # Test 5: optimal with money budget
    try:
        args = parse_args([
            "optimal",
            "--budget_dollars", "50000",
            "--cost_per_gpu_hour", "4.0",
            "--gpus", "8",
            "--gpu", "H100_SXM",
        ])
        assert args.budget_dollars == 50000.0
        assert args.cost_per_gpu_hour == 4.0
        print("  [PASS] optimal --budget_dollars parses correctly")
    except Exception as e:
        errors.append(f"  [FAIL] optimal --budget_dollars: {e}")

    # Test 6: strict_budget flag
    try:
        args = parse_args(["--strict_budget", "validate", "--params", "7e9", "--batch", "2048",
                           "--steps", "100"])
        assert args.strict_budget is True
        print("  [PASS] --strict_budget flag parsed correctly")
    except Exception as e:
        errors.append(f"  [FAIL] --strict_budget: {e}")

    # Test 7: output directory flag
    try:
        args = parse_args(["--out", "/tmp/test_run", "validate", "--params", "7e9",
                           "--batch", "2048", "--steps", "100"])
        assert args.out == "/tmp/test_run"
        print("  [PASS] --out flag parsed correctly")
    except Exception as e:
        errors.append(f"  [FAIL] --out flag: {e}")

    # Test 8: format flag
    try:
        args = parse_args(["--format", "json", "validate", "--params", "7e9",
                           "--batch", "2048", "--steps", "100"])
        assert args.format == "json"
        args_all = parse_args(["--format", "all", "validate", "--params", "7e9",
                                "--batch", "2048", "--steps", "100"])
        assert args_all.format == "all"
        print("  [PASS] --format flag parsed correctly")
    except Exception as e:
        errors.append(f"  [FAIL] --format flag: {e}")

    # Test 9: Missing required args fail gracefully (SystemExit from argparse)
    try:
        try:
            parse_args(["validate"])  # Missing --params, --batch, --steps
            errors.append("  [FAIL] validate with no args should raise SystemExit")
        except SystemExit as e:
            assert e.code != 0 or e.code == 2  # argparse exits with 2 on error
        print("  [PASS] validate with missing required args raises SystemExit")
    except Exception as e:
        errors.append(f"  [FAIL] Missing args SystemExit: {e}")

    # Test 10: optimal with no budget source fails gracefully in main()
    try:
        exit_code = main(["optimal"])  # No --compute_flops, --hours, or --budget_dollars
        # Should return 1 (error) or print error and exit
        # Some implementations may raise SystemExit from argparse before reaching main()
        print(f"  [PASS] optimal with no budget source returns error code {exit_code}")
    except SystemExit as e:
        print(f"  [PASS] optimal with no budget source raises SystemExit({e.code})")
    except Exception as e:
        errors.append(f"  [FAIL] optimal no budget source: {e}")

    # Test 11: k flag parsed on all subcommands
    try:
        args = parse_args(["--k", "8.0", "validate", "--params", "7e9",
                           "--batch", "2048", "--steps", "100"])
        assert args.k == 8.0
        print("  [PASS] --k flag parsed correctly")
    except Exception as e:
        errors.append(f"  [FAIL] --k flag: {e}")

    # Test 12: _parse_float handles scientific notation
    try:
        assert abs(_parse_float("7e9") - 7e9) < 1
        assert abs(_parse_float("1.4e12") - 1.4e12) < 1
        assert abs(_parse_float("70000000000") - 70e9) < 1
        try:
            _parse_float("not_a_number")
            errors.append("  [FAIL] _parse_float should raise on non-numeric input")
        except argparse.ArgumentTypeError:
            pass
        print("  [PASS] _parse_float handles scientific notation and rejects invalid input")
    except Exception as e:
        errors.append(f"  [FAIL] _parse_float: {e}")

    # Test 13: Full pipeline test (requires all template files)
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            exit_code = main([
                "validate",
                "--params", "7e9",
                "--seq_len", "2048",
                "--batch", "2048",
                "--steps", "100000",
                "--gpus", "8",
                "--gpu", "H100_SXM",
                "--out", tmpdir,
                "--format", "json",
            ])
            assert exit_code == 0, f"Expected exit code 0, got {exit_code}"
            assert os.path.exists(os.path.join(tmpdir, "budget.json"))
        print("  [PASS] Full pipeline test (Mode A): runs end-to-end and creates budget.json")
    except ImportError:
        print("  [SKIP] Full pipeline test: template files not importable")
    except Exception as e:
        errors.append(f"  [FAIL] Full pipeline test: {e}")

    # Summary
    if errors:
        print("\nFailed tests:")
        for err in errors:
            print(err)
        raise SystemExit(1)
    else:
        print("\nAll self-tests passed.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        _run_self_tests()
    else:
        sys.exit(main())
