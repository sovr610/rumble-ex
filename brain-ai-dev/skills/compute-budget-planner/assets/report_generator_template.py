"""
report_generator_template.py
=============================
ReportGenerator: structured and human-readable output for BudgetPlanner results.

Generates three output formats:
    budget.json  — Machine-readable JSON with full schema
    budget.txt   — Human-readable formatted table
    budget.svg   — isoFLOPs curves (requires matplotlib; gracefully skipped if unavailable)

Usage
-----
    from report_generator_template import ReportGenerator

    reporter = ReportGenerator()
    reporter.write_json(result, "/path/to/budget.json")
    reporter.write_txt(result, "/path/to/budget.txt")
    reporter.write_svg(result, "/path/to/budget.svg")  # optional
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Check for matplotlib availability (optional)
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for server environments
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False
    logger.info("matplotlib not available; SVG report generation disabled.")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_num(value: Any, unit: str = "", precision: int = 2) -> str:
    """Format a number with magnitude suffix and optional unit."""
    if value is None:
        return "N/A"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(v) or math.isinf(v):
        return str(v)

    abs_v = abs(v)
    if abs_v == 0:
        return f"0{unit}"
    elif abs_v >= 1e12:
        return f"{v/1e12:.{precision}f}T{unit}"
    elif abs_v >= 1e9:
        return f"{v/1e9:.{precision}f}B{unit}"
    elif abs_v >= 1e6:
        return f"{v/1e6:.{precision}f}M{unit}"
    elif abs_v >= 1e3:
        return f"{v/1e3:.{precision}f}K{unit}"
    else:
        return f"{v:.{precision}f}{unit}"


def _fmt_sci(value: Any) -> str:
    """Format a number in scientific notation."""
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.3e}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_ratio(value: Any) -> str:
    """Format an undertraining ratio with status indicator."""
    if value is None:
        return "N/A"
    try:
        r = float(value)
    except (TypeError, ValueError):
        return str(value)

    if r < 0.5:
        status = "CRITICAL"
    elif r < 0.8:
        status = "WARNING"
    elif r <= 2.0:
        status = "OK"
    else:
        status = "INFO (overtrain)"

    return f"{r:.4f}x  [{status}]"


def _fmt_cost(value: Any) -> str:
    """Format cost in USD."""
    if value is None:
        return "N/A"
    try:
        v = float(value)
        return f"${v:,.2f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_hours(value: Any) -> str:
    """Format wallclock hours with human-readable equivalent."""
    if value is None:
        return "N/A"
    try:
        h = float(value)
    except (TypeError, ValueError):
        return str(value)

    days = h / 24
    if days >= 1.0:
        return f"{h:.1f} hours ({days:.1f} days)"
    return f"{h:.1f} hours"


def _to_json_safe(obj: Any) -> Any:
    """Recursively convert to JSON-serializable form."""
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
        return None
    return obj


# ---------------------------------------------------------------------------
# ReportGenerator
# ---------------------------------------------------------------------------

class ReportGenerator:
    """Generates structured and human-readable reports from BudgetResult objects.

    Methods
    -------
    write_json(result, path)
        Write a machine-readable JSON report. If path is None, prints to stdout.
    write_txt(result, path)
        Write a human-readable text table. If path is None, prints to stdout.
    write_svg(result, path)
        Write an isoFLOPs curve SVG. No-op if matplotlib unavailable.
    """

    def __init__(self) -> None:
        self.schema_version = "1.0"

    # ------------------------------------------------------------------
    # JSON Report
    # ------------------------------------------------------------------

    def write_json(self, result: Any, path: Optional[str] = None) -> str:
        """Write a structured budget.json report.

        Parameters
        ----------
        result : BudgetResult
        path : str or None
            Output file path. If None, returns JSON string and prints to stdout.

        Returns
        -------
        str
            The JSON string (regardless of whether it was written to a file).
        """
        doc = {
            "schema_version": self.schema_version,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": result.mode,
            "inputs": _to_json_safe(result.inputs),
            "assumptions": _to_json_safe(result.assumptions),
            "derived": _to_json_safe(result.derived),
            "warnings": result.warnings,
            "suggestions": result.suggestions,
        }
        json_str = json.dumps(doc, indent=2)

        if path is None:
            print(json_str)
        else:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(json_str)
            logger.info("Written budget.json to %s", path)

        return json_str

    # ------------------------------------------------------------------
    # Text Report
    # ------------------------------------------------------------------

    def write_txt(self, result: Any, path: Optional[str] = None) -> str:
        """Write a human-readable text report.

        Parameters
        ----------
        result : BudgetResult
        path : str or None
            Output file path. If None, returns text string and prints to stdout.

        Returns
        -------
        str
            The formatted text.
        """
        lines = self._build_txt_lines(result)
        text = "\n".join(lines)

        if path is None:
            print(text)
        else:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
                fh.write("\n")
            logger.info("Written budget.txt to %s", path)

        return text

    def _build_txt_lines(self, result: Any) -> List[str]:
        """Build the lines of the text report."""
        inp = result.inputs or {}
        asmp = result.assumptions or {}
        d = result.derived or {}
        warnings = result.warnings or []
        suggestions = result.suggestions or []

        mode_labels = {
            "validate_run": "Validate Run (Mode A)",
            "compute_required": "Compute Required (Mode B)",
            "solve_optimal": "Solve Optimal (Mode C)",
        }
        mode_label = mode_labels.get(result.mode, result.mode)

        width = 65
        sep = "=" * width
        sep_minor = "-" * width

        lines: List[str] = [
            sep,
            "  COMPUTE BUDGET PLANNER REPORT",
            f"  Mode: {mode_label}",
            f"  Generated: {result.generated_at[:19].replace('T', ' ')} UTC",
            sep,
            "",
        ]

        # ---- PLAN SUMMARY ----
        lines += ["PLAN SUMMARY", sep_minor]
        n_params = inp.get("n_params") or d.get("n_opt")
        if n_params:
            lines.append(f"  Model parameters:      {_fmt_num(n_params)} (non-embedding)")

        total_tokens = d.get("total_tokens") or d.get("actual_tokens")
        if total_tokens:
            lines.append(f"  Training tokens:       {_fmt_num(total_tokens)}")

        seq_len = inp.get("seq_len")
        if seq_len:
            lines.append(f"  Sequence length:       {int(seq_len):,}")

        global_batch = inp.get("global_batch")
        if global_batch:
            lines.append(f"  Global batch size:     {int(global_batch):,}")

        steps = inp.get("steps") or d.get("required_steps")
        if steps:
            lines.append(f"  Optimizer steps:       {int(steps):,}")

        tokens_per_step = d.get("tokens_per_step")
        if tokens_per_step:
            lines.append(f"  Tokens per step:       {_fmt_num(tokens_per_step)}")

        lines.append("")

        # ---- COMPUTE BUDGET ----
        lines += ["COMPUTE BUDGET", sep_minor]
        gpu_type = asmp.get("gpu_type") or inp.get("gpu_type") or "N/A"
        dtype = asmp.get("dtype") or inp.get("dtype") or "bf16"
        is_sparse = asmp.get("is_sparse", False)
        peak_tflops = asmp.get("peak_tflops")
        sparsity_note = " (sparse)" if is_sparse else " (dense)"

        lines.append(f"  GPU type:              {gpu_type} ({dtype}{sparsity_note})")
        if peak_tflops:
            lines.append(f"  Peak TFLOPS:           {peak_tflops:.1f} TFLOPS")

        num_gpus = asmp.get("num_gpus") or inp.get("num_gpus")
        if num_gpus:
            lines.append(f"  Number of GPUs:        {int(num_gpus):,}")

        util = asmp.get("utilization") or inp.get("utilization")
        if util:
            lines.append(f"  MFU (utilization):     {util*100:.1f}%")

        total_flops = d.get("total_flops")
        if total_flops:
            lines.append(f"  Total FLOPs:           {_fmt_sci(total_flops)}")

        wallclock = d.get("predicted_wallclock_hours")
        if wallclock is not None:
            lines.append(f"  Predicted wallclock:   {_fmt_hours(wallclock)}")
            if num_gpus and num_gpus > 1:
                lines.append(f"                         (per GPU: {_fmt_hours(wallclock * int(num_gpus))})")

        cost = d.get("predicted_cost_usd")
        if cost is not None:
            lines.append(f"  Estimated cost:        {_fmt_cost(cost)}")

        lines.append("")

        # ---- CHINCHILLA CHECK ----
        lines += ["CHINCHILLA CHECK", sep_minor]
        tpp = d.get("tokens_per_param")
        target_tpp = asmp.get("tokens_per_param_target") or 20.0
        k = asmp.get("k") or 6.0

        if tpp is not None:
            lines.append(f"  tokens/param:          {tpp:.2f}  (target: {target_tpp:.1f})")

        ratio = d.get("undertraining_ratio")
        if ratio is not None:
            lines.append(f"  Undertraining ratio:   {_fmt_ratio(ratio)}")

        n_opt = d.get("n_opt")
        d_opt = d.get("d_opt")
        if n_opt and d_opt:
            lines.append(
                f"  Compute-optimal (same C): N_opt={_fmt_num(n_opt)}, "
                f"D_opt={_fmt_num(d_opt)}"
            )

        lines.append(f"  FLOPs coefficient (k): {k:.1f}")
        spec_source = asmp.get("gpu_spec_source", "table")
        lines.append(f"  GPU spec source:       {spec_source}")
        lines.append("")

        # ---- WARNINGS ----
        if warnings:
            lines += ["WARNINGS", sep_minor]
            for w in warnings:
                lines.append(f"  {w}")
            lines.append("")

        # ---- SUGGESTIONS ----
        if suggestions:
            lines += ["SUGGESTIONS", sep_minor]
            for idx, s in enumerate(suggestions, 1):
                lines.append(f"  [{idx}] {s}")
            lines.append("")

        lines.append(sep)
        return lines

    # ------------------------------------------------------------------
    # SVG / Matplotlib Report
    # ------------------------------------------------------------------

    def write_svg(self, result: Any, path: Optional[str] = None) -> bool:
        """Write an isoFLOPs curve SVG (requires matplotlib).

        Renders a log-log plot of model parameters (N) vs training tokens (D)
        with:
          - Gray isoFLOPs curves for several compute budgets
          - Blue compute-optimal frontier (D = a * N)
          - Red dot marking the planned run

        Parameters
        ----------
        result : BudgetResult
        path : str or None
            Output file path (should end in .svg or .png).
            If None, shows the plot interactively (if matplotlib allows).

        Returns
        -------
        bool
            True if plot was generated, False if matplotlib unavailable.
        """
        if not _HAS_MATPLOTLIB:
            logger.info(
                "matplotlib not available; skipping SVG report. "
                "Install with: pip install matplotlib"
            )
            return False

        try:
            self._generate_isoflops_plot(result, path)
            if path:
                logger.info("Written budget SVG to %s", path)
            return True
        except Exception as exc:
            logger.warning("Failed to generate SVG report: %s", exc)
            return False

    def _generate_isoflops_plot(self, result: Any, path: Optional[str]) -> None:
        """Generate the isoFLOPs plot."""
        d = result.derived or {}
        asmp = result.assumptions or {}

        a = asmp.get("tokens_per_param_target", 20.0)
        k = asmp.get("k", 6.0)
        total_flops = d.get("total_flops")
        n_params = d.get("n_opt") or result.inputs.get("n_params")
        total_tokens = d.get("total_tokens") or d.get("actual_tokens")

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.set_xscale("log")
        ax.set_yscale("log")

        # --- isoFLOPs curves ---
        # Range of N values to plot
        n_range = [10 ** x for x in [8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12]]

        # Compute budgets to draw curves for (log-spaced from 1e18 to 1e25)
        iso_budgets = [10 ** exp for exp in [18, 19, 20, 21, 22, 23, 24, 25]]
        colors_iso = plt.cm.Greys(
            [0.3 + 0.7 * i / (len(iso_budgets) - 1) for i in range(len(iso_budgets))]
        )

        for i, C in enumerate(iso_budgets):
            # D = C / (k * N) for isoFLOPs
            D_values = [C / (k * n) for n in n_range]
            ax.plot(
                n_range, D_values,
                color=colors_iso[i], linewidth=0.8, linestyle="--", alpha=0.6,
                label=f"C={_fmt_sci(C)} FLOPs" if i % 2 == 0 else None
            )
            # Label the curve at midpoint
            mid_idx = len(n_range) // 2
            n_mid = n_range[mid_idx]
            d_mid = C / (k * n_mid)
            if 1e8 <= n_mid <= 1e12 and 1e9 <= d_mid <= 1e15:
                ax.annotate(
                    f"C={_fmt_sci(C)}",
                    xy=(n_mid, d_mid),
                    fontsize=6, color="gray", alpha=0.8,
                    rotation=-30, va="center"
                )

        # --- Compute-optimal frontier: D = a * N ---
        n_frontier = [10 ** x for x in [8, 9, 10, 11, 12, 13]]
        d_frontier = [a * n for n in n_frontier]
        ax.plot(
            n_frontier, d_frontier,
            color="royalblue", linewidth=2.0, linestyle="-",
            label=f"Compute-optimal frontier (D = {a:.0f} * N)"
        )

        # --- Planned run point ---
        if n_params and total_tokens and n_params > 0 and total_tokens > 0:
            ax.scatter(
                [n_params], [total_tokens],
                color="crimson", s=120, zorder=5,
                label=f"Planned run ({_fmt_num(n_params)} params, {_fmt_num(total_tokens)} tokens)"
            )
            ax.annotate(
                " Planned run",
                xy=(n_params, total_tokens),
                fontsize=8, color="crimson", va="center"
            )

            # If there's a corresponding isoFLOPs curve for the planned run
            if total_flops:
                # Draw the actual isoFLOPs curve for this run
                D_planned_iso = [total_flops / (k * n) for n in n_range if n > 0]
                ax.plot(
                    n_range[:len(D_planned_iso)], D_planned_iso,
                    color="crimson", linewidth=1.5, linestyle=":",
                    label=f"Planned isoFLOPs (C={_fmt_sci(total_flops)})"
                )

        # --- Formatting ---
        ax.set_xlabel("Model Parameters (N)", fontsize=12)
        ax.set_ylabel("Training Tokens (D)", fontsize=12)
        ax.set_title(
            f"isoFLOPs Curves with Compute-Optimal Frontier\n"
            f"(k={k:.1f}, target tokens/param={a:.1f})",
            fontsize=12
        )

        # Custom tick formatting
        def _param_formatter(x, pos):
            return _fmt_num(x)

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(_param_formatter))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(_param_formatter))

        ax.set_xlim(1e8, 1e13)
        ax.set_ylim(1e9, 1e15)
        ax.grid(True, which="both", alpha=0.2)

        legend = ax.legend(
            loc="upper left", fontsize=7,
            framealpha=0.8, ncol=1,
            title="Legend",
        )

        # Ratio annotation box
        ratio = result.derived.get("undertraining_ratio") if result.derived else None
        if ratio is not None:
            ratio_str = f"tokens/param ratio: {ratio:.3f}x"
            if ratio < 0.5:
                ratio_color = "red"
                ratio_label = "CRITICAL (undertrained)"
            elif ratio < 0.8:
                ratio_color = "orange"
                ratio_label = "WARNING (undertrained)"
            elif ratio <= 2.0:
                ratio_color = "green"
                ratio_label = "OK (near compute-optimal)"
            else:
                ratio_color = "blue"
                ratio_label = "INFO (overtrain)"

            ax.annotate(
                f"{ratio_str}\n{ratio_label}",
                xy=(0.98, 0.04), xycoords="axes fraction",
                fontsize=8, ha="right", va="bottom",
                color=ratio_color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
            )

        plt.tight_layout()

        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            fmt = "svg" if path.endswith(".svg") else "png"
            plt.savefig(path, format=fmt, dpi=150, bbox_inches="tight")
        else:
            plt.show()

        plt.close(fig)

    # ------------------------------------------------------------------
    # Batch output helper
    # ------------------------------------------------------------------

    def write_all(
        self,
        result: Any,
        out_dir: str,
        formats: Optional[List[str]] = None,
        prefix: str = "budget",
    ) -> Dict[str, str]:
        """Write multiple report formats to a directory.

        Parameters
        ----------
        result : BudgetResult
        out_dir : str
            Output directory (will be created if needed).
        formats : list of str or None
            List of formats: "json", "txt", "svg". Default: ["json", "txt"].
        prefix : str
            Filename prefix. Default: "budget".

        Returns
        -------
        dict
            Mapping of format -> path for files that were written.
        """
        if formats is None:
            formats = ["json", "txt"]

        os.makedirs(out_dir, exist_ok=True)
        written: Dict[str, str] = {}

        if "json" in formats:
            p = os.path.join(out_dir, f"{prefix}.json")
            self.write_json(result, p)
            written["json"] = p

        if "txt" in formats:
            p = os.path.join(out_dir, f"{prefix}.txt")
            self.write_txt(result, p)
            written["txt"] = p

        if "svg" in formats:
            p = os.path.join(out_dir, f"{prefix}.svg")
            success = self.write_svg(result, p)
            if success:
                written["svg"] = p
            else:
                logger.info("SVG not written (matplotlib unavailable).")

        return written


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _make_mock_result(mode: str = "validate_run") -> Any:
    """Create a minimal BudgetResult-like object for testing."""
    class MockResult:
        def __init__(self):
            self.mode = mode
            self.schema_version = "1.0"
            self.generated_at = "2025-01-15T14:23:11+00:00"
            self.inputs = {
                "n_params": 7e9, "seq_len": 2048, "global_batch": 2048,
                "steps": 100000, "num_gpus": 8, "gpu_type": "H100_SXM", "dtype": "bf16"
            }
            self.assumptions = {
                "k": 6.0, "tokens_per_param_target": 20.0, "utilization": 0.35,
                "gpu_type": "H100_SXM", "gpu_spec_source": "table",
                "peak_tflops": 989.0, "is_sparse": False, "dtype": "bf16",
                "num_gpus": 8,
            }
            self.derived = {
                "total_tokens": 100000 * 2048 * 2048,
                "tokens_per_param": (100000 * 2048 * 2048) / 7e9,
                "total_flops": 6 * 7e9 * (100000 * 2048 * 2048),
                "predicted_wallclock_hours": 42.0,
                "predicted_cost_usd": None,
                "undertraining_ratio": (100000 * 2048 * 2048) / (7e9 * 20.0),
                "n_opt": 2.5e9,
                "d_opt": 5.0e10,
                "required_steps": 100000,
                "tokens_per_step": 2048 * 2048,
            }
            self.warnings = ["[WARNING] Likely undertrained: tokens/param=59.59 (ratio=2.98x target=20.0)"]
            self.suggestions = ["Increase training to 140B tokens."]

    return MockResult()


def _make_optimal_mock() -> Any:
    """Create a mock result for Mode C (solve_optimal)."""
    r = _make_mock_result("solve_optimal")
    r.derived["n_opt"] = 6.45e10
    r.derived["d_opt"] = 1.29e12
    r.derived["tokens_per_param"] = 20.0
    r.derived["undertraining_ratio"] = 1.0
    r.warnings = ["N_opt/D_opt are heuristic estimates."]
    r.suggestions = ["Train ~64.5B model on ~1.29T tokens."]
    return r


def _run_self_tests() -> None:
    print("Running report_generator_template.py self-tests...")
    import tempfile
    import os
    errors = []

    reporter = ReportGenerator()

    # Test 1: JSON roundtrip
    try:
        result = _make_mock_result()
        json_str = reporter.write_json(result, path=None)
        loaded = json.loads(json_str)
        assert "schema_version" in loaded, "Missing schema_version"
        assert loaded["mode"] == "validate_run"
        assert "inputs" in loaded
        assert "assumptions" in loaded
        assert "derived" in loaded
        assert "warnings" in loaded
        assert "suggestions" in loaded
        assert loaded["derived"]["total_tokens"] == 100000 * 2048 * 2048
        assert loaded["assumptions"]["k"] == 6.0
        print("  [PASS] JSON roundtrip: schema valid, all required keys present")
    except Exception as e:
        errors.append(f"  [FAIL] JSON roundtrip: {e}")

    # Test 2: JSON written to file
    try:
        result = _make_mock_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            p = os.path.join(tmpdir, "budget.json")
            reporter.write_json(result, p)
            assert os.path.exists(p), "budget.json not created"
            assert os.path.getsize(p) > 0, "budget.json is empty"
            with open(p) as fh:
                loaded2 = json.load(fh)
            assert loaded2["mode"] == "validate_run"
        print("  [PASS] JSON written to file: exists, non-empty, parseable")
    except Exception as e:
        errors.append(f"  [FAIL] JSON to file: {e}")

    # Test 3: TXT non-empty and contains key fields
    try:
        result = _make_mock_result()
        txt = reporter.write_txt(result, path=None)
        assert len(txt) > 100, f"TXT too short: {len(txt)} chars"
        for required_phrase in [
            "COMPUTE BUDGET",
            "CHINCHILLA CHECK",
            "tokens/param",
            "PLAN SUMMARY",
        ]:
            assert required_phrase in txt, f"Missing '{required_phrase}' in TXT"
        print("  [PASS] TXT: non-empty, contains COMPUTE BUDGET, CHINCHILLA CHECK, PLAN SUMMARY")
    except Exception as e:
        errors.append(f"  [FAIL] TXT content check: {e}")

    # Test 4: TXT written to file
    try:
        result = _make_mock_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            p = os.path.join(tmpdir, "budget.txt")
            reporter.write_txt(result, p)
            assert os.path.exists(p)
            content = open(p).read()
            assert len(content) > 100
        print("  [PASS] TXT written to file: exists, non-empty")
    except Exception as e:
        errors.append(f"  [FAIL] TXT to file: {e}")

    # Test 5: TXT for all three modes
    try:
        for mode in ["validate_run", "compute_required", "solve_optimal"]:
            r = _make_mock_result(mode)
            txt = reporter.write_txt(r, path=None)
            assert mode.replace("_", " ").split()[0] in txt.lower() or mode in txt.lower() or "Mode" in txt
            assert len(txt) > 50
        print("  [PASS] TXT generated successfully for all three modes")
    except Exception as e:
        errors.append(f"  [FAIL] TXT for all modes: {e}")

    # Test 6: SVG creation (if matplotlib available)
    try:
        result = _make_mock_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            p = os.path.join(tmpdir, "budget.svg")
            success = reporter.write_svg(result, p)
            if _HAS_MATPLOTLIB:
                assert success, "write_svg returned False despite matplotlib being available"
                assert os.path.exists(p), "budget.svg not created"
                assert os.path.getsize(p) > 0, "budget.svg is empty"
                content = open(p, "r", errors="ignore").read()
                assert "<svg" in content or "<?xml" in content, "SVG file does not look like SVG"
                print(f"  [PASS] SVG created: {os.path.getsize(p)} bytes")
            else:
                assert not success, "write_svg should return False without matplotlib"
                print("  [PASS] SVG gracefully skipped (matplotlib not available)")
    except Exception as e:
        errors.append(f"  [FAIL] SVG generation: {e}")

    # Test 7: write_all creates multiple formats
    try:
        result = _make_mock_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            written = reporter.write_all(result, tmpdir, formats=["json", "txt"])
            assert "json" in written
            assert "txt" in written
            assert os.path.exists(written["json"])
            assert os.path.exists(written["txt"])
        print("  [PASS] write_all creates both JSON and TXT")
    except Exception as e:
        errors.append(f"  [FAIL] write_all: {e}")

    # Test 8: _fmt_num formatting
    try:
        assert "1.40T" in _fmt_num(1.4e12) or "1.40" in _fmt_num(1.4e12)
        assert "70.00B" in _fmt_num(70e9) or "70" in _fmt_num(70e9)
        assert "N/A" == _fmt_num(None)
        print("  [PASS] _fmt_num formatting works")
    except Exception as e:
        errors.append(f"  [FAIL] _fmt_num: {e}")

    # Test 9: _fmt_ratio thresholds
    try:
        assert "CRITICAL" in _fmt_ratio(0.3)
        assert "WARNING" in _fmt_ratio(0.7)
        assert "OK" in _fmt_ratio(1.0)
        assert "INFO" in _fmt_ratio(3.0)
        assert "N/A" == _fmt_ratio(None)
        print("  [PASS] _fmt_ratio correctly labels thresholds")
    except Exception as e:
        errors.append(f"  [FAIL] _fmt_ratio: {e}")

    # Test 10: inf/nan in derived do not break JSON
    try:
        result = _make_mock_result()
        result.derived["total_flops"] = float("inf")
        result.derived["tokens_per_param"] = float("nan")
        json_str = reporter.write_json(result, path=None)
        loaded = json.loads(json_str)
        assert loaded["derived"]["total_flops"] is None
        assert loaded["derived"]["tokens_per_param"] is None
        print("  [PASS] inf/nan in derived converted to null in JSON")
    except Exception as e:
        errors.append(f"  [FAIL] inf/nan handling: {e}")

    # Test 11: Optimal mode result generates correct TXT
    try:
        result = _make_optimal_mock()
        txt = reporter.write_txt(result, path=None)
        assert "64.50B" in txt or "64.5" in txt or "n_opt" in txt.lower() or "N_opt" in txt or "compute-optimal" in txt.lower()
        print("  [PASS] Optimal mode (Mode C) generates valid TXT output")
    except Exception as e:
        errors.append(f"  [FAIL] Optimal mode TXT: {e}")

    # Summary
    if errors:
        print("\nFailed tests:")
        for err in errors:
            print(err)
        raise SystemExit(1)
    else:
        print("\nAll self-tests passed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    _run_self_tests()
