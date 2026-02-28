#!/usr/bin/env python3
"""
Validation script for Visualization & Interpretability skill.

Validates the three done-when gates from SKILL.md:

1. **Spike Raster** -- SpikeRasterPlotter.plot_raster() produces correct raster
   from synthetic spike tensor; neurons on y-axis, time on x-axis; exported
   PNG is non-empty.

2. **Attention Heatmap** -- AttentionHeatmapper.plot_attention() renders correct
   heatmap from synthetic weight matrix; colorbar present; labels correct.

3. **Dashboard Report** -- TrainingDashboard.generate_report() produces HTML file
   with loss curves, neuromodulator levels, and phase boundaries from synthetic
   metrics data.

Usage:
    python scripts/validate_viz.py

Exit code 0 if all gates pass, 1 otherwise.

Requires: matplotlib, numpy, torch
Backend: Agg (headless, no display required)
"""

import matplotlib
matplotlib.use('Agg')

import sys
import os
import tempfile
import traceback

import matplotlib.pyplot as plt
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Resolve asset imports
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SKILL_DIR = os.path.dirname(SCRIPT_DIR)
ASSETS_DIR = os.path.join(SKILL_DIR, 'assets')

if ASSETS_DIR not in sys.path:
    sys.path.insert(0, ASSETS_DIR)

from spike_raster_template import SpikeRasterPlotter, VizConfig
from attention_heatmap_template import AttentionHeatmapper
from attention_heatmap_template import VizConfig as AttVizConfig
from viz_config_template import TrainingDashboard
from viz_config_template import VizConfig as DashVizConfig


# ---------------------------------------------------------------------------
# Gate Validators
# ---------------------------------------------------------------------------

class GateResult:
    """Result of a single gate validation."""

    def __init__(self, gate_name: str):
        self.gate_name = gate_name
        self.checks = []  # list of (check_name, passed, detail)

    def add_check(self, name: str, passed: bool, detail: str = ''):
        self.checks.append((name, passed, detail))

    @property
    def passed(self) -> bool:
        return all(ok for _, ok, _ in self.checks)

    @property
    def summary(self) -> str:
        n_pass = sum(1 for _, ok, _ in self.checks if ok)
        n_fail = sum(1 for _, ok, _ in self.checks if not ok)
        status = 'PASS' if self.passed else 'FAIL'
        return f"[{status}] {self.gate_name}: {n_pass} checks passed, {n_fail} failed"

    def detail_report(self) -> str:
        lines = [self.summary]
        for name, ok, detail in self.checks:
            mark = 'OK' if ok else 'FAIL'
            lines.append(f"    [{mark}] {name}")
            if detail and not ok:
                for dl in detail.strip().split('\n')[:3]:
                    lines.append(f"           {dl}")
        return '\n'.join(lines)


def validate_gate1_spike_raster() -> GateResult:
    """Gate 1: Spike Raster validation."""
    result = GateResult("Gate 1: Spike Raster")

    try:
        config = VizConfig(figsize=(8, 5), dpi=100)
        plotter = SpikeRasterPlotter(config)

        # Check 1: Figure creation
        spikes = (torch.rand(2, 50, 100) > 0.9).float()
        fig = plotter.plot_raster(spikes)
        result.add_check("figure_created", fig is not None)

        # Check 2: Has axes
        result.add_check("has_axes", len(fig.axes) > 0,
                         f"axes count: {len(fig.axes)}")

        # Check 3: X-axis label contains 'Time'
        ax = fig.axes[0]
        xl = ax.get_xlabel()
        has_time = 'Time' in xl or 'Step' in xl or 'time' in xl.lower()
        result.add_check("x_axis_time", has_time,
                         f"xlabel='{xl}', expected 'Time' or 'Step'")

        # Check 4: Y-axis label contains 'Neuron'
        yl = ax.get_ylabel()
        has_neuron = 'Neuron' in yl or 'neuron' in yl.lower()
        result.add_check("y_axis_neuron", has_neuron,
                         f"ylabel='{yl}', expected 'Neuron'")

        # Check 5: Exported PNG is non-empty
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            fig.savefig(f.name, dpi=100, bbox_inches='tight')
            plt.close(fig)
            size = os.path.getsize(f.name)
            os.unlink(f.name)
        result.add_check("png_non_empty", size > 1000,
                         f"PNG size: {size} bytes")

        # Check 6: Valid PNG header
        spikes2 = (torch.rand(1, 30, 60) > 0.9).float()
        fig2 = plotter.plot_raster(spikes2)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            fig2.savefig(f.name, dpi=100)
            plt.close(fig2)
            with open(f.name, 'rb') as fp:
                header = fp.read(8)
            os.unlink(f.name)
        result.add_check("valid_png_header", header[:4] == b'\x89PNG',
                         f"header bytes: {header[:4]}")

        # Check 7: Raster with empty spikes
        fig3 = plotter.plot_raster(torch.zeros(1, 50, 100))
        result.add_check("empty_spikes_handled", fig3 is not None)
        plt.close(fig3)

        # Check 8: Raster with 2D input
        spikes_2d = (torch.rand(30, 60) > 0.9).float()
        fig4 = plotter.plot_raster(spikes_2d)
        result.add_check("2d_input_accepted", fig4 is not None)
        plt.close(fig4)

        # Check 9: Large tensor subsampling
        large_spikes = (torch.rand(1, 500, 4096) > 0.95).float()
        fig5 = plotter.plot_raster(large_spikes)
        result.add_check("large_tensor_handled", fig5 is not None)
        plt.close(fig5)

        # Check 10: Numpy input
        np_spikes = (np.random.rand(1, 30, 40) > 0.9).astype(float)
        fig6 = plotter.plot_raster(np_spikes)
        result.add_check("numpy_input_accepted", fig6 is not None)
        plt.close(fig6)

    except Exception as e:
        result.add_check("unhandled_exception", False,
                         f"{e}\n{traceback.format_exc()}")

    return result


def validate_gate2_attention_heatmap() -> GateResult:
    """Gate 2: Attention Heatmap validation."""
    result = GateResult("Gate 2: Attention Heatmap")

    try:
        config = AttVizConfig(figsize=(8, 5), dpi=100)
        hm = AttentionHeatmapper(config)

        # Check 1: Figure creation
        weights = torch.softmax(torch.randn(8, 8), dim=-1)
        fig = hm.plot_attention(weights)
        result.add_check("figure_created", fig is not None)

        # Check 2: Has axes
        result.add_check("has_axes", len(fig.axes) > 0,
                         f"axes count: {len(fig.axes)}")

        # Check 3: Colorbar present (extra axes)
        result.add_check("colorbar_present", len(fig.axes) >= 2,
                         f"axes count: {len(fig.axes)}, expected >= 2 for colorbar")

        # Check 4: X-axis label correct
        ax = fig.axes[0]
        xl = ax.get_xlabel()
        has_key = 'Key' in xl or 'Position' in xl or 'key' in xl.lower()
        result.add_check("x_axis_key", has_key,
                         f"xlabel='{xl}', expected 'Key' or 'Position'")

        # Check 5: Y-axis label correct
        yl = ax.get_ylabel()
        has_query = 'Query' in yl or 'Position' in yl or 'query' in yl.lower()
        result.add_check("y_axis_query", has_query,
                         f"ylabel='{yl}', expected 'Query' or 'Position'")

        # Check 6: Exported PNG non-empty
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            fig.savefig(f.name, dpi=100, bbox_inches='tight')
            plt.close(fig)
            size = os.path.getsize(f.name)
            os.unlink(f.name)
        result.add_check("png_non_empty", size > 1000,
                         f"PNG size: {size} bytes")

        # Check 7: With labels
        labels = ['A', 'B', 'C', 'D', 'E']
        w2 = torch.softmax(torch.randn(5, 5), dim=-1)
        fig2 = hm.plot_attention(w2, labels=labels)
        result.add_check("labels_accepted", fig2 is not None)
        plt.close(fig2)

        # Check 8: 1x1 attention
        fig3 = hm.plot_attention(torch.tensor([[1.0]]))
        result.add_check("1x1_handled", fig3 is not None)
        plt.close(fig3)

        # Check 9: Zero attention
        fig4 = hm.plot_attention(torch.zeros(5, 5))
        result.add_check("zero_attention_handled", fig4 is not None)
        plt.close(fig4)

        # Check 10: Valid PNG header
        w5 = torch.softmax(torch.randn(4, 4), dim=-1)
        fig5 = hm.plot_attention(w5)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            fig5.savefig(f.name, dpi=100)
            plt.close(fig5)
            with open(f.name, 'rb') as fp:
                header = fp.read(8)
            os.unlink(f.name)
        result.add_check("valid_png_header", header[:4] == b'\x89PNG',
                         f"header bytes: {header[:4]}")

    except Exception as e:
        result.add_check("unhandled_exception", False,
                         f"{e}\n{traceback.format_exc()}")

    return result


def validate_gate3_dashboard_report() -> GateResult:
    """Gate 3: Dashboard Report validation."""
    result = GateResult("Gate 3: Dashboard Report")

    try:
        with tempfile.TemporaryDirectory() as td:
            config = DashVizConfig(dpi=72)
            dashboard = TrainingDashboard(log_dir=td, config=config)

            metrics = {
                'loss': [1.0, 0.8, 0.6, 0.4, 0.3],
                'accuracy': [0.2, 0.4, 0.6, 0.7, 0.8],
            }
            neuro = {
                'DA': [0.5, 0.6, 0.7, 0.8, 0.9],
                'ACh': [0.3, 0.4, 0.5, 0.6, 0.7],
                'NE': [0.8, 0.7, 0.6, 0.5, 0.4],
                '5-HT': [0.4, 0.4, 0.5, 0.5, 0.6],
            }
            phase_boundaries = [2, 4]

            # Generate report
            report_path = dashboard.generate_report(
                output_dir=td,
                metrics=metrics,
                neuromodulator_levels=neuro,
                phase_boundaries=phase_boundaries,
            )

            # Check 1: File exists
            result.add_check("file_exists", os.path.exists(report_path),
                             f"path: {report_path}")

            # Check 2: File non-empty
            size = os.path.getsize(report_path) if os.path.exists(report_path) else 0
            result.add_check("file_non_empty", size > 500,
                             f"size: {size} bytes")

            # Read content
            content = ''
            if os.path.exists(report_path):
                with open(report_path, 'r') as f:
                    content = f.read()

            # Check 3: Valid HTML
            is_html = '<html' in content.lower() or '<!doctype' in content.lower()
            result.add_check("valid_html", is_html,
                             "No <html> or <!DOCTYPE> tag found")

            # Check 4: Contains loss curves section
            has_loss = 'loss' in content.lower() or 'Loss' in content
            result.add_check("has_loss_section", has_loss,
                             "No 'loss' or 'Loss' found in report")

            # Check 5: Contains neuromodulator section
            has_neuro = ('neuromodulator' in content.lower() or
                         'DA' in content or 'Neuromodulator' in content)
            result.add_check("has_neuro_section", has_neuro,
                             "No neuromodulator section found")

            # Check 6: Contains phase boundaries
            has_phase = ('phase' in content.lower() or
                         'boundary' in content.lower() or
                         str(phase_boundaries) in content or
                         '[2, 4]' in content)
            result.add_check("has_phase_section", has_phase,
                             "No phase transition section found")

            # Check 7: Contains embedded images
            has_images = 'data:image/png;base64,' in content
            result.add_check("has_embedded_images", has_images,
                             "No base64 images found")

            # Check 8: Loss curves figure works standalone
            fig = dashboard.plot_loss_curves(metrics)
            result.add_check("loss_fig_created", fig is not None)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                fig.savefig(f.name, dpi=72)
                plt.close(fig)
                fig_size = os.path.getsize(f.name)
                os.unlink(f.name)
            result.add_check("loss_fig_non_empty", fig_size > 500,
                             f"size: {fig_size}")

            # Check 9: Neuromodulator figure works standalone
            fig2 = dashboard.plot_neuromodulator_levels(neuro)
            result.add_check("neuro_fig_created", fig2 is not None)
            plt.close(fig2)

            # Check 10: Phase transitions figure works standalone
            fig3 = dashboard.plot_phase_transitions(metrics, phase_boundaries)
            result.add_check("phase_fig_created", fig3 is not None)
            plt.close(fig3)

    except Exception as e:
        result.add_check("unhandled_exception", False,
                         f"{e}\n{traceback.format_exc()}")

    return result


# ---------------------------------------------------------------------------
# Agg backend check
# ---------------------------------------------------------------------------

def validate_agg_backend() -> GateResult:
    """Verify Agg backend is active."""
    result = GateResult("Pre-check: Agg Backend")
    backend = matplotlib.get_backend().lower()
    result.add_check("agg_backend", backend == 'agg',
                     f"backend='{backend}', expected 'agg'")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Visualization & Interpretability -- Done-When Gate Validation")
    print("=" * 70)
    print()

    gates = [
        validate_agg_backend(),
        validate_gate1_spike_raster(),
        validate_gate2_attention_heatmap(),
        validate_gate3_dashboard_report(),
    ]

    all_passed = True
    for gate in gates:
        print(gate.detail_report())
        print()
        if not gate.passed:
            all_passed = False

    # Summary
    print("=" * 70)
    total_checks = sum(len(g.checks) for g in gates)
    passed_checks = sum(
        sum(1 for _, ok, _ in g.checks if ok)
        for g in gates
    )
    failed_checks = total_checks - passed_checks
    gates_passed = sum(1 for g in gates if g.passed)
    gates_failed = len(gates) - gates_passed

    if all_passed:
        status = "ALL GATES PASSED"
    else:
        status = "SOME GATES FAILED"

    print(f"  {status}")
    print(f"  Gates: {gates_passed}/{len(gates)} passed")
    print(f"  Checks: {passed_checks}/{total_checks} passed, {failed_checks} failed")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
