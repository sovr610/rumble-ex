"""
chinchilla_table.py
===================
Generates a reference table of compute-optimal N and D values for common
compute budgets, along with predicted wallclock on 8x H100 SXM at 35% MFU.

Output formats:
    - Text table (always printed to stdout)
    - CSV file (optional, with --csv flag)

Usage
-----
    python scripts/chinchilla_table.py
    python scripts/chinchilla_table.py --k 6 --tokens_per_param 20 --csv chinchilla_table.csv
    python scripts/chinchilla_table.py --num_gpus 64 --gpu H100_SXM --util 0.45
    python scripts/chinchilla_table.py --budgets 1e20 1e21 1e22 1e23 1e24

Self-test:
    python scripts/chinchilla_table.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Add assets directory to path
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ASSETS_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "assets")
sys.path.insert(0, _ASSETS_DIR)

try:
    from gpu_specs_template import GPUSpecTable
    from chinchilla_solver_template import ChinchillaSolver
    _HAVE_MODULES = True
except ImportError:
    _HAVE_MODULES = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default compute budgets (log-spaced from 1e18 to 1e26)
DEFAULT_BUDGETS = [
    1e18, 3e18,
    1e19, 3e19,
    1e20, 3e20,
    1e21, 3e21,
    1e22, 3e22,
    1e23, 3e23,
    1e24, 3e24,
    1e25, 3e25,
    1e26,
]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_flops(c: float) -> str:
    """Format FLOPs in scientific notation."""
    return f"{c:.2e}"


def _fmt_params(n: float) -> str:
    """Format parameter count with suffix."""
    if n >= 1e12:
        return f"{n/1e12:.2f}T"
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    return f"{n:.2e}"


def _fmt_tokens(d: float) -> str:
    """Format token count with suffix."""
    if d >= 1e12:
        return f"{d/1e12:.2f}T"
    if d >= 1e9:
        return f"{d/1e9:.2f}B"
    if d >= 1e6:
        return f"{d/1e6:.2f}M"
    return f"{d:.2e}"


def _fmt_hours(h: float) -> str:
    """Format hours as hours and days."""
    if h >= 24 * 365:
        return f"{h:.0f}h ({h/24/365:.1f}yr)"
    if h >= 24 * 30:
        return f"{h:.0f}h ({h/24/30:.1f}mo)"
    if h >= 24:
        return f"{h:.0f}h ({h/24:.1f}d)"
    return f"{h:.1f}h"


def _fmt_tpp(d_opt: float, n_opt: float) -> str:
    """Format tokens/param ratio."""
    return f"{d_opt / n_opt:.1f}"


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

def generate_table(
    budgets: List[float],
    k: float = 6.0,
    tokens_per_param: float = 20.0,
    num_gpus: int = 8,
    peak_tflops: float = 989.0,
    utilization: float = 0.35,
) -> List[Dict]:
    """Generate the reference table as a list of dicts.

    Parameters
    ----------
    budgets : list of float
        Compute budgets in FLOPs to evaluate.
    k : float
        FLOPs coefficient. Default 6.0.
    tokens_per_param : float
        Chinchilla target. Default 20.0.
    num_gpus : int
        Number of GPUs for wallclock estimation. Default 8.
    peak_tflops : float
        Peak TFLOPS per GPU. Default 989.0 (H100 SXM bf16 dense).
    utilization : float
        MFU. Default 0.35.

    Returns
    -------
    list of dict
        Each dict has keys:
            compute_flops, n_opt, d_opt, tokens_per_param,
            predicted_wallclock_hours, gpu_hours_total
    """
    solver = ChinchillaSolver(k=k, tokens_per_param=tokens_per_param)

    achieved_flops_per_s = num_gpus * peak_tflops * 1e12 * utilization

    rows = []
    for C in budgets:
        if C <= 0 or not math.isfinite(C):
            continue
        try:
            n = solver.n_opt(C)
            d = solver.d_opt(C)
            wallclock_s = C / achieved_flops_per_s
            wallclock_h = wallclock_s / 3600.0
            gpu_hours = wallclock_h * num_gpus

            rows.append({
                "compute_flops": C,
                "n_opt": n,
                "d_opt": d,
                "tokens_per_param_actual": d / n,
                "predicted_wallclock_hours": wallclock_h,
                "gpu_hours_total": gpu_hours,
            })
        except (ValueError, ZeroDivisionError):
            continue

    return rows


# ---------------------------------------------------------------------------
# Text table printing
# ---------------------------------------------------------------------------

def print_text_table(
    rows: List[Dict],
    k: float,
    tokens_per_param: float,
    num_gpus: int,
    peak_tflops: float,
    utilization: float,
    gpu_name: str,
) -> str:
    """Print and return the formatted text table.

    Returns
    -------
    str
        The formatted table string.
    """
    lines = []

    def w(s: str = "") -> None:
        lines.append(s)

    w("=" * 85)
    w("  CHINCHILLA COMPUTE-OPTIMAL REFERENCE TABLE")
    w(f"  k={k:.1f} | tokens_per_param_target={tokens_per_param:.1f}")
    w(f"  Hardware: {num_gpus}x {gpu_name} | Peak TFLOPS: {peak_tflops:.0f} | MFU: {utilization*100:.0f}%")
    w("=" * 85)
    w()
    header = (
        f"{'Compute Budget':>16}  "
        f"{'N_opt':>10}  "
        f"{'D_opt':>10}  "
        f"{'tok/param':>9}  "
        f"{'Wallclock':>14}  "
        f"{'GPU-hours':>12}"
    )
    w(header)
    w("-" * 85)

    for row in rows:
        c = row["compute_flops"]
        n = row["n_opt"]
        d = row["d_opt"]
        tpp = row["tokens_per_param_actual"]
        wc = row["predicted_wallclock_hours"]
        gh = row["gpu_hours_total"]

        w(
            f"{_fmt_flops(c):>16}  "
            f"{_fmt_params(n):>10}  "
            f"{_fmt_tokens(d):>10}  "
            f"{tpp:>9.1f}  "
            f"{_fmt_hours(wc):>14}  "
            f"{gh:>12,.0f}"
        )

    w()
    w("-" * 85)
    w("  Notes:")
    w(f"  * N_opt = sqrt(C / (k * a))  where a = tokens_per_param_target = {tokens_per_param:.1f}")
    w(f"  * D_opt = a * N_opt")
    w(f"  * Wallclock = C / (num_gpus * peak_tflops * 1e12 * utilization)")
    w(f"  * GPU-hours = wallclock_hours * num_gpus")
    w(f"  * k={k:.1f} (forward ~2ND + backward ~4ND, standard Chinchilla coefficient)")
    w(f"  * MFU={utilization*100:.0f}% is conservative planning estimate (achievable: 35-55%)")
    w("=" * 85)

    text = "\n".join(lines)
    print(text)
    return text


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_csv(rows: List[Dict], path: str) -> None:
    """Write the table rows to a CSV file."""
    if not rows:
        return

    fieldnames = [
        "compute_flops_sci",
        "compute_flops",
        "n_opt_params",
        "n_opt_billions",
        "d_opt_tokens",
        "d_opt_trillions",
        "tokens_per_param",
        "predicted_wallclock_hours",
        "predicted_wallclock_days",
        "gpu_hours_total",
    ]

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "compute_flops_sci": f"{row['compute_flops']:.2e}",
                "compute_flops": row["compute_flops"],
                "n_opt_params": row["n_opt"],
                "n_opt_billions": row["n_opt"] / 1e9,
                "d_opt_tokens": row["d_opt"],
                "d_opt_trillions": row["d_opt"] / 1e12,
                "tokens_per_param": row["tokens_per_param_actual"],
                "predicted_wallclock_hours": row["predicted_wallclock_hours"],
                "predicted_wallclock_days": row["predicted_wallclock_hours"] / 24,
                "gpu_hours_total": row["gpu_hours_total"],
            })
    print(f"CSV written to: {path}")


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _run_self_tests() -> None:
    print("Running chinchilla_table.py self-tests...")
    errors = []

    # Test 1: generate_table returns rows for all budgets
    try:
        budgets = [1e20, 1e21, 1e22, 1e23, 1e24]
        rows = generate_table(budgets, k=6.0, tokens_per_param=20.0,
                              num_gpus=8, peak_tflops=989.0, utilization=0.35)
        assert len(rows) == len(budgets), f"Expected {len(budgets)} rows, got {len(rows)}"
        print(f"  [PASS] generate_table returns {len(rows)} rows for {len(budgets)} budgets")
    except Exception as e:
        errors.append(f"  [FAIL] generate_table basic: {e}")

    # Test 2: N_opt and D_opt are monotonically increasing
    try:
        budgets = [10 ** exp for exp in range(18, 27)]
        rows = generate_table(budgets, k=6.0, tokens_per_param=20.0,
                              num_gpus=8, peak_tflops=989.0, utilization=0.35)
        ns = [r["n_opt"] for r in rows]
        ds = [r["d_opt"] for r in rows]
        wcs = [r["predicted_wallclock_hours"] for r in rows]

        for i in range(1, len(ns)):
            assert ns[i] > ns[i-1], f"N_opt not monotonic at index {i}: {ns[i-1]:.3e} >= {ns[i]:.3e}"
            assert ds[i] > ds[i-1], f"D_opt not monotonic at index {i}: {ds[i-1]:.3e} >= {ds[i]:.3e}"
            assert wcs[i] > wcs[i-1], f"Wallclock not monotonic at index {i}: {wcs[i-1]:.3f} >= {wcs[i]:.3f}"

        print(f"  [PASS] N_opt, D_opt, wallclock all monotonically increasing over {len(rows)} budgets")
    except Exception as e:
        errors.append(f"  [FAIL] Monotonicity: {e}")

    # Test 3: tokens_per_param matches target for all rows
    try:
        target = 20.0
        budgets = [1e20, 5e23, 1e25]
        rows = generate_table(budgets, k=6.0, tokens_per_param=target,
                              num_gpus=8, peak_tflops=989.0, utilization=0.35)
        for row in rows:
            tpp = row["tokens_per_param_actual"]
            assert abs(tpp - target) < 1e-6, f"tokens_per_param={tpp:.8f} != target={target}"
        print(f"  [PASS] tokens_per_param_actual == {target:.1f} for all rows")
    except Exception as e:
        errors.append(f"  [FAIL] tokens_per_param consistency: {e}")

    # Test 4: k * N_opt * D_opt == C for all rows
    try:
        k = 6.0
        budgets = [1e20, 1e22, 5e23, 1e25]
        rows = generate_table(budgets, k=k, tokens_per_param=20.0,
                              num_gpus=8, peak_tflops=989.0, utilization=0.35)
        for row in rows:
            C = row["compute_flops"]
            n = row["n_opt"]
            d = row["d_opt"]
            reconstructed = k * n * d
            rel_err = abs(reconstructed - C) / C
            assert rel_err < 1e-9, (
                f"k*N*D={reconstructed:.4e} != C={C:.4e}, rel_err={rel_err:.2e}"
            )
        print(f"  [PASS] k * N_opt * D_opt == C within 1e-9 for all test budgets")
    except Exception as e:
        errors.append(f"  [FAIL] k*N*D == C: {e}")

    # Test 5: Chinchilla 70B/1.4T sanity check in table
    try:
        C_chinchilla = 6 * 70e9 * 1.4e12  # 5.88e23
        rows = generate_table([C_chinchilla], k=6.0, tokens_per_param=20.0,
                              num_gpus=8, peak_tflops=989.0, utilization=0.35)
        assert len(rows) == 1
        n_opt = rows[0]["n_opt"]
        d_opt = rows[0]["d_opt"]
        assert abs(n_opt - 70e9) / 70e9 < 1e-6, f"N_opt={n_opt:.4e}, expected 70e9"
        assert abs(d_opt - 1.4e12) / 1.4e12 < 1e-6, f"D_opt={d_opt:.4e}, expected 1.4e12"
        print(f"  [PASS] Chinchilla 70B/1.4T sanity: N_opt={n_opt/1e9:.2f}B, D_opt={d_opt/1e12:.2f}T")
    except Exception as e:
        errors.append(f"  [FAIL] Chinchilla 70B/1.4T sanity: {e}")

    # Test 6: Wallclock scales correctly with GPUs
    try:
        budgets = [1e23]
        r8 = generate_table(budgets, k=6.0, tokens_per_param=20.0,
                             num_gpus=8, peak_tflops=989.0, utilization=0.35)[0]
        r64 = generate_table(budgets, k=6.0, tokens_per_param=20.0,
                              num_gpus=64, peak_tflops=989.0, utilization=0.35)[0]
        ratio = r8["predicted_wallclock_hours"] / r64["predicted_wallclock_hours"]
        assert abs(ratio - 8.0) < 1e-9, f"Expected 8x speedup, got {ratio:.4f}"
        print(f"  [PASS] Wallclock scales correctly with GPU count: 8x GPUs -> 8x speedup")
    except Exception as e:
        errors.append(f"  [FAIL] Wallclock GPU scaling: {e}")

    # Test 7: Wallclock scales correctly with utilization
    try:
        budgets = [1e23]
        r35 = generate_table(budgets, k=6.0, tokens_per_param=20.0,
                              num_gpus=8, peak_tflops=989.0, utilization=0.35)[0]
        r70 = generate_table(budgets, k=6.0, tokens_per_param=20.0,
                              num_gpus=8, peak_tflops=989.0, utilization=0.70)[0]
        ratio = r35["predicted_wallclock_hours"] / r70["predicted_wallclock_hours"]
        assert abs(ratio - 2.0) < 1e-9, f"Expected 2x difference, got {ratio:.4f}"
        print(f"  [PASS] Wallclock scales correctly with utilization: 2x MFU -> 0.5x wallclock")
    except Exception as e:
        errors.append(f"  [FAIL] Wallclock utilization scaling: {e}")

    # Test 8: Text table contains expected content
    try:
        budgets = [1e20, 5e23, 1e25]
        rows = generate_table(budgets, k=6.0, tokens_per_param=20.0,
                              num_gpus=8, peak_tflops=989.0, utilization=0.35)

        # Capture output
        import io, sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        text = print_text_table(rows, k=6.0, tokens_per_param=20.0, num_gpus=8,
                                 peak_tflops=989.0, utilization=0.35, gpu_name="H100 SXM")
        sys.stdout = old_stdout

        assert "CHINCHILLA" in text or "chinchilla" in text.lower()
        assert "N_opt" in text or "n_opt" in text.lower()
        assert "D_opt" in text or "d_opt" in text.lower()
        assert len(text) > 200
        print(f"  [PASS] Text table contains expected content ({len(text)} chars)")
    except Exception as e:
        errors.append(f"  [FAIL] Text table content: {e}")
        # Restore stdout if test failed
        sys.stdout = sys.__stdout__

    # Test 9: CSV writing
    try:
        import tempfile
        budgets = [1e20, 1e22, 1e24]
        rows = generate_table(budgets, k=6.0, tokens_per_param=20.0,
                              num_gpus=8, peak_tflops=989.0, utilization=0.35)
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "test.csv")
            write_csv(rows, csv_path)
            assert os.path.exists(csv_path)
            assert os.path.getsize(csv_path) > 0
            with open(csv_path) as fh:
                content = fh.read()
            # Check headers
            assert "compute_flops" in content
            assert "n_opt_billions" in content
            assert "d_opt_trillions" in content
            # Check data
            lines = content.strip().split("\n")
            assert len(lines) == len(rows) + 1  # header + data rows
        print(f"  [PASS] CSV written with {len(rows)} data rows and correct headers")
    except Exception as e:
        errors.append(f"  [FAIL] CSV writing: {e}")

    # Test 10: Different tokens_per_param values
    try:
        C = 1e23
        for tpp in [5.0, 20.0, 100.0]:
            rows = generate_table([C], k=6.0, tokens_per_param=tpp,
                                  num_gpus=8, peak_tflops=989.0, utilization=0.35)
            assert len(rows) == 1
            assert abs(rows[0]["tokens_per_param_actual"] - tpp) < 1e-6
        print("  [PASS] Different tokens_per_param values produce correct ratios")
    except Exception as e:
        errors.append(f"  [FAIL] Different tokens_per_param: {e}")

    # Test 11: Default budgets cover full range 1e18 to 1e26
    try:
        assert min(DEFAULT_BUDGETS) <= 1e18
        assert max(DEFAULT_BUDGETS) >= 1e26
        # All budgets positive and finite
        for b in DEFAULT_BUDGETS:
            assert b > 0 and math.isfinite(b)
        print(f"  [PASS] DEFAULT_BUDGETS covers 1e18 to 1e26 ({len(DEFAULT_BUDGETS)} entries)")
    except Exception as e:
        errors.append(f"  [FAIL] DEFAULT_BUDGETS: {e}")

    # Test 12: Large budget range monotonicity (full DEFAULT_BUDGETS)
    try:
        rows = generate_table(DEFAULT_BUDGETS, k=6.0, tokens_per_param=20.0,
                              num_gpus=8, peak_tflops=989.0, utilization=0.35)
        ns = [r["n_opt"] for r in rows]
        for i in range(1, len(ns)):
            assert ns[i] > ns[i-1], f"Not monotonic at idx {i}: {ns[i-1]:.3e} >= {ns[i]:.3e}"
        print(f"  [PASS] Full DEFAULT_BUDGETS ({len(rows)} rows) all monotonically increasing")
    except Exception as e:
        errors.append(f"  [FAIL] Full range monotonicity: {e}")

    # Summary
    if errors:
        print("\nFailed tests:")
        for err in errors:
            print(err)
        raise SystemExit(1)
    else:
        print(f"\nAll self-tests passed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Chinchilla compute-optimal reference table showing N_opt, D_opt, "
            "and predicted wallclock for a range of compute budgets."
        )
    )
    parser.add_argument(
        "--k", type=float, default=6.0,
        help="FLOPs coefficient. Default: 6.0."
    )
    parser.add_argument(
        "--tokens_per_param", type=float, default=20.0,
        help="Chinchilla target tokens per parameter. Default: 20.0."
    )
    parser.add_argument(
        "--num_gpus", type=int, default=8,
        help="Number of GPUs for wallclock estimate. Default: 8."
    )
    parser.add_argument(
        "--gpu", type=str, default="H100_SXM",
        help="GPU type for spec lookup. Default: H100_SXM."
    )
    parser.add_argument(
        "--util", type=float, default=0.35,
        help="MFU (Model FLOPs Utilization). Default: 0.35."
    )
    parser.add_argument(
        "--peak_tflops", type=float, default=None,
        help="Override peak TFLOPS (use when GPU not in spec table)."
    )
    parser.add_argument(
        "--budgets", type=float, nargs="+", default=None,
        help="Custom compute budgets in FLOPs (space-separated). Default: log-spaced 1e18-1e26."
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to write CSV output. If not specified, only text table is printed."
    )
    parser.add_argument(
        "--self-test", action="store_true",
        help="Run self-tests and exit."
    )

    args = parser.parse_args()

    if args.self_test:
        _run_self_tests()
        return 0

    # Resolve GPU specs
    peak_tflops = args.peak_tflops
    gpu_name = args.gpu

    if peak_tflops is None:
        if not _HAVE_MODULES:
            print(
                "ERROR: Cannot import gpu_specs_template. "
                "Provide --peak_tflops manually or ensure assets/ is on PYTHONPATH.",
                file=sys.stderr,
            )
            return 1
        try:
            table = GPUSpecTable()
            spec = table.validate_or_fallback(args.gpu)
            peak_tflops = spec.peak_tflops
            gpu_name = spec.name
        except Exception as exc:
            print(f"ERROR: Failed to resolve GPU spec for '{args.gpu}': {exc}", file=sys.stderr)
            print("Use --peak_tflops to provide the value manually.", file=sys.stderr)
            return 1

    if not _HAVE_MODULES:
        print(
            "ERROR: Cannot import planner modules. "
            "Ensure assets/ directory is accessible.",
            file=sys.stderr,
        )
        return 1

    budgets = args.budgets if args.budgets else DEFAULT_BUDGETS

    # Generate table
    rows = generate_table(
        budgets=budgets,
        k=args.k,
        tokens_per_param=args.tokens_per_param,
        num_gpus=args.num_gpus,
        peak_tflops=peak_tflops,
        utilization=args.util,
    )

    if not rows:
        print("ERROR: No valid budgets to display.", file=sys.stderr)
        return 1

    # Print text table
    print_text_table(
        rows=rows,
        k=args.k,
        tokens_per_param=args.tokens_per_param,
        num_gpus=args.num_gpus,
        peak_tflops=peak_tflops,
        utilization=args.util,
        gpu_name=gpu_name,
    )

    # Write CSV if requested
    if args.csv:
        write_csv(rows, args.csv)

    return 0


if __name__ == "__main__":
    sys.exit(main())
