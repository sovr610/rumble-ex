#!/usr/bin/env python3
"""
Generate 100+ pytest test cases for visualization modules.

Produces a comprehensive test suite covering all six visualization classes:
  - SpikeRasterPlotter
  - AttentionHeatmapper
  - WorkspaceVisualizer
  - ReasoningTraceVisualizer
  - EmbeddingProjector
  - TrainingDashboard

Test categories (per testing-matrix.md):
  Cat 1: Figure non-empty
  Cat 2: Correct axes
  Cat 3: Deterministic rendering
  Cat 4: Headless backend
  Cat 5: File export (PNG valid)
  Cat 6: Edge cases
  Cat 7: Configuration
  Cat 8: Integration

Usage:
    python scripts/gen_viz_tests.py                 # Print to stdout
    python scripts/gen_viz_tests.py -o test_viz.py  # Write to file
    python scripts/gen_viz_tests.py --run            # Generate and run

Requires: matplotlib, numpy, torch, pytest (for --run)
"""

import matplotlib
matplotlib.use('Agg')

import sys
import os
import argparse
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import torch
import traceback

# ---------------------------------------------------------------------------
# Resolve asset imports
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SKILL_DIR = os.path.dirname(SCRIPT_DIR)
ASSETS_DIR = os.path.join(SKILL_DIR, 'assets')

if ASSETS_DIR not in sys.path:
    sys.path.insert(0, ASSETS_DIR)

from spike_raster_template import SpikeRasterPlotter
from spike_raster_template import VizConfig as SpikeVizConfig
from attention_heatmap_template import AttentionHeatmapper
from attention_heatmap_template import VizConfig as AttnVizConfig
from workspace_viz_template import WorkspaceVisualizer
from workspace_viz_template import VizConfig as WsVizConfig
from reasoning_trace_template import ReasoningTraceVisualizer
from reasoning_trace_template import VizConfig as ReasonVizConfig
from embedding_projector_template import EmbeddingProjector
from embedding_projector_template import VizConfig as EmbVizConfig
from viz_config_template import TrainingDashboard, VizConfig


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def assert_fig_valid(fig, label=''):
    """Assert figure is non-None, has axes, and saves to non-empty PNG."""
    assert fig is not None, f"Figure is None ({label})"
    assert len(fig.axes) > 0, f"No axes in figure ({label})"
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        fig.savefig(f.name, dpi=72, bbox_inches='tight')
        plt.close(fig)
        size = os.path.getsize(f.name)
        os.unlink(f.name)
    assert size > 500, f"PNG too small: {size} bytes ({label})"


def assert_png_header(fig, label=''):
    """Assert saved figure has valid PNG magic bytes."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        fig.savefig(f.name, dpi=72)
        plt.close(fig)
        with open(f.name, 'rb') as fp:
            header = fp.read(8)
        os.unlink(f.name)
    assert header[:4] == b'\x89PNG', f"Invalid PNG header ({label}): {header[:4]}"


# ---------------------------------------------------------------------------
# All test functions
# ---------------------------------------------------------------------------

def _collect_tests():
    """Return list of (test_name, test_func) tuples. 100+ tests."""
    tests = []

    # Shared configs
    spike_cfg = SpikeVizConfig(figsize=(6, 4), dpi=72, max_neurons=50,
                                max_timesteps=100)
    attn_cfg = AttnVizConfig(figsize=(6, 4), dpi=72)
    ws_cfg = WsVizConfig(figsize=(6, 4), dpi=72)
    reason_cfg = ReasonVizConfig(figsize=(6, 4), dpi=72)
    emb_cfg = EmbVizConfig(figsize=(6, 4), dpi=72, max_points=200)
    dash_cfg = VizConfig(figsize=(6, 4), dpi=72)

    spike_p = SpikeRasterPlotter(spike_cfg)
    attn_h = AttentionHeatmapper(attn_cfg)
    ws_v = WorkspaceVisualizer(ws_cfg)
    reason_v = ReasoningTraceVisualizer(reason_cfg)
    emb_p = EmbeddingProjector(emb_cfg)
    dash = TrainingDashboard(config=dash_cfg)

    # ======================================================================
    # SPIKE RASTER PLOTTER (22 tests)
    # ======================================================================

    # Cat 1: Non-empty
    def sr_01():
        fig = spike_p.plot_raster((torch.rand(2, 50, 100) > 0.9).float())
        assert_fig_valid(fig, 'sr basic')
    tests.append(('sr_01_raster_non_empty', sr_01))

    def sr_02():
        fig = spike_p.plot_raster((torch.rand(1, 30, 200) > 0.85).float(),
                                   neuron_ids=[0, 50, 100, 199])
        assert_fig_valid(fig, 'sr neuron_ids')
    tests.append(('sr_02_raster_neuron_ids', sr_02))

    def sr_03():
        fig = spike_p.plot_firing_rates((torch.rand(1, 50, 80) > 0.88).float(), window=5)
        assert_fig_valid(fig, 'sr rates')
    tests.append(('sr_03_firing_rates', sr_03))

    def sr_04():
        fig = spike_p.plot_membrane_potential(torch.randn(1, 50, 100) * 0.5, threshold=1.0)
        assert_fig_valid(fig, 'sr membrane')
    tests.append(('sr_04_membrane', sr_04))

    def sr_05():
        fig = spike_p.plot_spike_count_distribution((torch.rand(1, 50, 100) > 0.9).float())
        assert_fig_valid(fig, 'sr count dist')
    tests.append(('sr_05_count_dist', sr_05))

    def sr_06():
        fig = spike_p.plot_population_activity((torch.rand(1, 50, 100) > 0.9).float())
        assert_fig_valid(fig, 'sr pop activity')
    tests.append(('sr_06_pop_activity', sr_06))

    # Cat 2: Axes
    def sr_07():
        fig = spike_p.plot_raster((torch.rand(1, 20, 30) > 0.9).float())
        ax = fig.axes[0]
        assert 'Time' in ax.get_xlabel(), f"X: {ax.get_xlabel()}"
        assert 'Neuron' in ax.get_ylabel(), f"Y: {ax.get_ylabel()}"
        plt.close(fig)
    tests.append(('sr_07_axes_labels', sr_07))

    def sr_08():
        fig = spike_p.plot_firing_rates((torch.rand(1, 40, 50) > 0.9).float())
        ax = fig.axes[0]
        assert 'Time' in ax.get_xlabel()
        plt.close(fig)
    tests.append(('sr_08_rates_axes', sr_08))

    def sr_09():
        fig = spike_p.plot_membrane_potential(torch.randn(1, 30, 50), threshold=1.0)
        ax = fig.axes[0]
        assert 'time' in ax.get_xlabel().lower() or 'step' in ax.get_xlabel().lower()
        plt.close(fig)
    tests.append(('sr_09_membrane_axes', sr_09))

    # Cat 3: Deterministic
    def sr_10():
        spikes = (torch.rand(1, 30, 50) > 0.9).float()
        sizes = []
        for _ in range(2):
            fig = spike_p.plot_raster(spikes)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                fig.savefig(f.name, dpi=72); plt.close(fig)
                sizes.append(os.path.getsize(f.name)); os.unlink(f.name)
        assert sizes[0] == sizes[1]
    tests.append(('sr_10_deterministic', sr_10))

    def sr_11():
        spikes = (torch.rand(1, 30, 50) > 0.9).float()
        sizes = []
        for _ in range(2):
            fig = spike_p.plot_firing_rates(spikes)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                fig.savefig(f.name, dpi=72); plt.close(fig)
                sizes.append(os.path.getsize(f.name)); os.unlink(f.name)
        assert sizes[0] == sizes[1]
    tests.append(('sr_11_rates_deterministic', sr_11))

    # Cat 4: Backend
    def sr_12():
        assert matplotlib.get_backend().lower() == 'agg'
    tests.append(('sr_12_agg_backend', sr_12))

    # Cat 5: PNG
    def sr_13():
        fig = spike_p.plot_raster((torch.rand(1, 20, 30) > 0.9).float())
        assert_png_header(fig, 'sr png')
    tests.append(('sr_13_png_header', sr_13))

    def sr_14():
        fig = spike_p.plot_firing_rates((torch.rand(1, 30, 40) > 0.9).float())
        assert_png_header(fig, 'sr rates png')
    tests.append(('sr_14_rates_png', sr_14))

    # Cat 6: Edge cases
    def sr_15():
        fig = spike_p.plot_raster(torch.zeros(1, 50, 100))
        assert_fig_valid(fig, 'sr zeros')
    tests.append(('sr_15_empty_spikes', sr_15))

    def sr_16():
        fig = spike_p.plot_raster((torch.rand(1, 50, 1) > 0.8).float())
        assert_fig_valid(fig, 'sr single neuron')
    tests.append(('sr_16_single_neuron', sr_16))

    def sr_17():
        fig = spike_p.plot_raster((torch.rand(1, 1, 50) > 0.5).float())
        assert_fig_valid(fig, 'sr single timestep')
    tests.append(('sr_17_single_timestep', sr_17))

    def sr_18():
        fig = spike_p.plot_raster((torch.rand(1, 500, 4096) > 0.95).float())
        assert_fig_valid(fig, 'sr large')
    tests.append(('sr_18_large_subsample', sr_18))

    # Cat 7: Config
    def sr_19():
        fig = spike_p.plot_raster((torch.rand(1, 20, 30) > 0.9).float(),
                                   title='Custom')
        assert 'Custom' in fig.axes[0].get_title(); plt.close(fig)
    tests.append(('sr_19_custom_title', sr_19))

    def sr_20():
        fig = spike_p.plot_raster((np.random.rand(1, 30, 40) > 0.9).astype(float))
        assert_fig_valid(fig, 'sr numpy')
    tests.append(('sr_20_numpy_input', sr_20))

    # Cat 8: Integration
    def sr_21():
        spikes = (torch.rand(1, 50, 80) > 0.88).float()
        fig = spike_p.plot_raster(spikes, layer_boundaries=[20, 40, 60])
        assert_fig_valid(fig, 'sr layers')
    tests.append(('sr_21_layer_boundaries', sr_21))

    def sr_22():
        m = torch.randn(1, 30, 50)
        s = (m > 0.8).float()
        fig = spike_p.plot_membrane_potential(m, threshold=0.8, spikes=s)
        assert_fig_valid(fig, 'sr membrane+spikes')
    tests.append(('sr_22_membrane_spikes', sr_22))

    # ======================================================================
    # ATTENTION HEATMAPPER (22 tests)
    # ======================================================================

    def ah_01():
        fig = attn_h.plot_attention(torch.softmax(torch.randn(8, 8), dim=-1))
        assert_fig_valid(fig, 'ah basic')
    tests.append(('ah_01_basic', ah_01))

    def ah_02():
        fig = attn_h.plot_attention(torch.softmax(torch.randn(5, 5), dim=-1),
                                     labels=['A', 'B', 'C', 'D', 'E'])
        assert_fig_valid(fig, 'ah labels')
    tests.append(('ah_02_labels', ah_02))

    def ah_03():
        fig = attn_h.plot_multi_head_attention(
            torch.softmax(torch.randn(8, 6, 6), dim=-1))
        assert_fig_valid(fig, 'ah multi-head')
    tests.append(('ah_03_multi_head', ah_03))

    def ah_04():
        fig = attn_h.plot_head_entropy(torch.softmax(torch.randn(8, 6, 6), dim=-1))
        assert_fig_valid(fig, 'ah entropy')
    tests.append(('ah_04_head_entropy', ah_04))

    def ah_05():
        w = {'vision->text': torch.tensor(0.6), 'text->vision': torch.tensor(0.4)}
        fig = attn_h.plot_cross_modal_attention(w)
        assert_fig_valid(fig, 'ah cross-modal')
    tests.append(('ah_05_cross_modal', ah_05))

    def ah_06():
        fig = attn_h.plot_workspace_competition(torch.tensor([0.5, 0.3, 0.15, 0.05]))
        assert_fig_valid(fig, 'ah competition')
    tests.append(('ah_06_competition', ah_06))

    def ah_07():
        fig = attn_h.plot_attention(torch.softmax(torch.randn(4, 6), dim=-1))
        ax = fig.axes[0]
        assert 'Key' in ax.get_xlabel(); plt.close(fig)
    tests.append(('ah_07_axes', ah_07))

    def ah_08():
        fig = attn_h.plot_attention(torch.softmax(torch.randn(6, 6), dim=-1))
        assert len(fig.axes) >= 2; plt.close(fig)
    tests.append(('ah_08_colorbar', ah_08))

    def ah_09():
        fig = attn_h.plot_attention(torch.softmax(torch.randn(6, 6), dim=-1))
        sizes = []
        for _ in range(2):
            w = torch.softmax(torch.randn(4, 4), dim=-1)
            fig = attn_h.plot_attention(w)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                fig.savefig(f.name, dpi=72); plt.close(fig)
                sizes.append(os.path.getsize(f.name)); os.unlink(f.name)
        # Two different inputs, just check both produce valid PNGs
        assert sizes[0] > 500 and sizes[1] > 500
    tests.append(('ah_09_export', ah_09))

    def ah_10():
        fig = attn_h.plot_attention(torch.softmax(torch.randn(4, 4), dim=-1))
        assert_png_header(fig, 'ah png')
    tests.append(('ah_10_png_header', ah_10))

    def ah_11():
        fig = attn_h.plot_attention(torch.tensor([[1.0]]))
        assert_fig_valid(fig, 'ah 1x1')
    tests.append(('ah_11_1x1', ah_11))

    def ah_12():
        fig = attn_h.plot_attention(torch.zeros(5, 5))
        assert_fig_valid(fig, 'ah zeros')
    tests.append(('ah_12_zeros', ah_12))

    def ah_13():
        fig = attn_h.plot_attention(torch.ones(6, 6) / 6.0)
        assert_fig_valid(fig, 'ah uniform')
    tests.append(('ah_13_uniform', ah_13))

    def ah_14():
        fig = attn_h.plot_attention(torch.softmax(torch.randn(50, 50), dim=-1))
        assert_fig_valid(fig, 'ah large')
    tests.append(('ah_14_large', ah_14))

    def ah_15():
        w = np.random.rand(5, 5); w = w / w.sum(axis=-1, keepdims=True)
        fig = attn_h.plot_attention(w)
        assert_fig_valid(fig, 'ah numpy')
    tests.append(('ah_15_numpy', ah_15))

    def ah_16():
        fig = attn_h.plot_workspace_competition(
            torch.tensor([0.4, 0.35, 0.2, 0.05]),
            modality_names=['Vision', 'Text', 'Audio', 'Sensor'])
        assert_fig_valid(fig, 'ah comp named')
    tests.append(('ah_16_competition_named', ah_16))

    def ah_17():
        fig = attn_h.plot_workspace_competition(torch.tensor([0.9]))
        assert_fig_valid(fig, 'ah single mod')
    tests.append(('ah_17_single_modality', ah_17))

    def ah_18():
        h = torch.tensor([[0.25, 0.25, 0.25, 0.25], [0.6, 0.2, 0.12, 0.08]])
        fig = attn_h.plot_competition_evolution(h)
        assert_fig_valid(fig, 'ah evolution')
    tests.append(('ah_18_evolution', ah_18))

    def ah_19():
        fig = attn_h.plot_multi_head_attention(
            torch.softmax(torch.randn(16, 4, 4), dim=-1), num_heads=4)
        assert_fig_valid(fig, 'ah head subset')
    tests.append(('ah_19_head_subset', ah_19))

    def ah_20():
        fig = attn_h.plot_attention(torch.softmax(torch.randn(3, 3), dim=-1),
                                     title='My Title')
        assert 'My Title' in fig.axes[0].get_title(); plt.close(fig)
    tests.append(('ah_20_custom_title', ah_20))

    def ah_21():
        fig = attn_h.plot_workspace_competition(torch.tensor([0.6, 0.2, 0.1, 0.1]),
                                                  threshold=0.3)
        ax = fig.axes[0]
        has_thresh = any('Threshold' in str(l.get_label()) for l in ax.get_lines())
        assert has_thresh; plt.close(fig)
    tests.append(('ah_21_threshold_line', ah_21))

    def ah_22():
        fig = attn_h.plot_attention(torch.softmax(torch.randn(5), dim=-1))
        assert_fig_valid(fig, 'ah 1d')
    tests.append(('ah_22_1d', ah_22))

    # ======================================================================
    # WORKSPACE VISUALIZER (19 tests)
    # ======================================================================

    def ws_01():
        h = [torch.tensor([0.25, 0.25, 0.25, 0.25]),
             torch.tensor([0.6, 0.2, 0.12, 0.08])]
        fig = ws_v.plot_competition_dynamics(h)
        assert_fig_valid(fig, 'ws comp')
    tests.append(('ws_01_competition', ws_01))

    def ws_02():
        fig = ws_v.plot_broadcast_map(torch.tensor([0.8, 0.6, 0.9, 0.5]))
        assert_fig_valid(fig, 'ws broadcast')
    tests.append(('ws_02_broadcast', ws_02))

    def ws_03():
        fig = ws_v.plot_working_memory_slots(torch.randn(7, 64))
        assert_fig_valid(fig, 'ws wm')
    tests.append(('ws_03_wm_slots', ws_03))

    def ws_04():
        fig = ws_v.plot_slot_dynamics(torch.randn(20, 7, 64))
        assert_fig_valid(fig, 'ws slot dyn')
    tests.append(('ws_04_slot_dynamics', ws_04))

    def ws_05():
        flow = {'Encoders': 0.9, 'SNN Core': 0.7, 'HTM': 0.5}
        fig = ws_v.plot_information_flow(flow)
        assert_fig_valid(fig, 'ws flow')
    tests.append(('ws_05_info_flow', ws_05))

    def ws_06():
        h = torch.tensor([[0.5, 0.5], [0.7, 0.3]])
        fig = ws_v.plot_competition_dynamics(h)
        ax = fig.axes[0]
        assert 'Round' in ax.get_xlabel(); plt.close(fig)
    tests.append(('ws_06_axes', ws_06))

    def ws_07():
        fig = ws_v.plot_broadcast_map(torch.tensor([0.7, 0.5, 0.3]),
                                       module_names=['SNN Core', 'HTM', 'Reasoning'],
                                       source_name='Vision')
        assert_fig_valid(fig, 'ws broadcast named')
    tests.append(('ws_07_broadcast_named', ws_07))

    def ws_08():
        fig = ws_v.plot_broadcast_map(torch.rand(3, 4))
        assert_fig_valid(fig, 'ws broadcast 2d')
    tests.append(('ws_08_broadcast_2d', ws_08))

    def ws_09():
        fig = ws_v.plot_working_memory_slots(torch.randn(4, 32),
                                              labels=['V', 'T', 'A', 'C'])
        assert_fig_valid(fig, 'ws wm labeled')
    tests.append(('ws_09_wm_labeled', ws_09))

    def ws_10():
        fig = ws_v.plot_working_memory_slots(torch.randn(1, 128))
        assert_fig_valid(fig, 'ws single slot')
    tests.append(('ws_10_single_slot', ws_10))

    def ws_11():
        fig = ws_v.plot_working_memory_slots(torch.randn(5, 64), show_similarity=False)
        assert_fig_valid(fig, 'ws no sim')
    tests.append(('ws_11_no_similarity', ws_11))

    def ws_12():
        fig = ws_v.plot_working_memory_slots(torch.zeros(7, 64))
        assert_fig_valid(fig, 'ws zeros')
    tests.append(('ws_12_zeros', ws_12))

    def ws_13():
        fig = ws_v.plot_slot_dynamics(torch.rand(15, 5))
        assert_fig_valid(fig, 'ws slot 2d')
    tests.append(('ws_13_slot_2d', ws_13))

    def ws_14():
        h = np.array([[0.3, 0.3, 0.2, 0.2], [0.6, 0.2, 0.12, 0.08]])
        fig = ws_v.plot_competition_dynamics(h)
        assert_fig_valid(fig, 'ws numpy')
    tests.append(('ws_14_numpy', ws_14))

    def ws_15():
        h = [torch.tensor([0.5, 0.5]), torch.tensor([0.7, 0.3])]
        fig = ws_v.plot_competition_dynamics(h, title='My Title')
        assert 'My Title' in fig.axes[0].get_title(); plt.close(fig)
    tests.append(('ws_15_custom_title', ws_15))

    def ws_16():
        fig = ws_v.plot_working_memory_slots(torch.randn(7, 4096),
                                              show_similarity=False)
        assert_fig_valid(fig, 'ws large dim')
    tests.append(('ws_16_large_dim', ws_16))

    def ws_17():
        fig = ws_v.plot_broadcast_map(torch.tensor([0.5, 0.3, 0.2]))
        assert_png_header(fig, 'ws png')
    tests.append(('ws_17_png', ws_17))

    def ws_18():
        h = [torch.tensor([0.3, 0.3]), torch.tensor([0.6, 0.4])]
        fig = ws_v.plot_competition_dynamics(h, threshold=0.5)
        ax = fig.axes[0]
        has_t = any('Threshold' in str(l.get_label()) for l in ax.get_lines())
        assert has_t; plt.close(fig)
    tests.append(('ws_18_threshold', ws_18))

    def ws_19():
        mem = torch.randn(4, 32)
        fig = ws_v.plot_working_memory_slots(mem, show_similarity=False)
        with tempfile.TemporaryDirectory() as td:
            wv = WorkspaceVisualizer(WsVizConfig(save_dir=td, dpi=72))
            path = wv.save_figure(fig, 'test_wm')
            assert os.path.exists(path)
    tests.append(('ws_19_save', ws_19))

    # ======================================================================
    # REASONING TRACE VISUALIZER (19 tests)
    # ======================================================================

    def rt_01():
        fig = reason_v.plot_routing_decision(0.65, threshold=0.8)
        assert_fig_valid(fig, 'rt routing')
    tests.append(('rt_01_routing', rt_01))

    def rt_02():
        fig = reason_v.plot_routing_decision(0.92, threshold=0.8)
        assert 'System 1' in fig.axes[0].get_title(); plt.close(fig)
    tests.append(('rt_02_sys1', rt_02))

    def rt_03():
        fig = reason_v.plot_routing_decision(0.5, threshold=0.8)
        assert 'System 2' in fig.axes[0].get_title(); plt.close(fig)
    tests.append(('rt_03_sys2', rt_03))

    def rt_04():
        trace = [{'confidence': 0.4}, {'confidence': 0.6}, {'confidence': 0.85}]
        fig = reason_v.plot_system2_steps(trace, threshold=0.8)
        assert_fig_valid(fig, 'rt sys2 steps')
    tests.append(('rt_04_sys2_steps', rt_04))

    def rt_05():
        trace = [{'confidence': 0.4, 'residual_norm': 2.1},
                 {'confidence': 0.6, 'residual_norm': 1.3},
                 {'confidence': 0.85, 'residual_norm': 0.3}]
        fig = reason_v.plot_system2_steps(trace)
        assert_fig_valid(fig, 'rt sys2 resid')
    tests.append(('rt_05_sys2_residuals', rt_05))

    def rt_06():
        trace = [{'confidence': 0.3}, {'confidence': 0.35}]
        fig = reason_v.plot_system2_steps(trace, threshold=0.8)
        assert_fig_valid(fig, 'rt no converge')
    tests.append(('rt_06_no_converge', rt_06))

    def rt_07():
        rules = ['rule_a', 'rule_b', 'rule_c', 'rule_d', 'rule_e']
        acts = torch.tensor([0.92, 0.78, 0.65, 0.31, 0.12])
        fig = reason_v.plot_rule_activation(rules, acts)
        assert_fig_valid(fig, 'rt rules')
    tests.append(('rt_07_rules', rt_07))

    def rt_08():
        rules = ['A', 'B']; acts = torch.tensor([0.8, 0.3])
        fig = reason_v.plot_rule_activation(rules, acts)
        ax = fig.axes[0]
        assert 'activation' in ax.get_xlabel().lower(); plt.close(fig)
    tests.append(('rt_08_rule_axes', rt_08))

    def rt_09():
        fig = reason_v.plot_routing_decision(0.0, threshold=0.8)
        assert_fig_valid(fig, 'rt zero conf')
    tests.append(('rt_09_zero_conf', rt_09))

    def rt_10():
        fig = reason_v.plot_routing_decision(1.0, threshold=0.8)
        assert_fig_valid(fig, 'rt max conf')
    tests.append(('rt_10_max_conf', rt_10))

    def rt_11():
        rules = [f'Rule {i}' for i in range(20)]
        acts = torch.rand(20)
        fig = reason_v.plot_rule_activation(rules, acts, top_k=5)
        assert_fig_valid(fig, 'rt topk')
    tests.append(('rt_11_topk', rt_11))

    def rt_12():
        rules = ['A', 'B', 'C']; acts = torch.zeros(3)
        fig = reason_v.plot_rule_activation(rules, acts)
        assert_fig_valid(fig, 'rt zeros')
    tests.append(('rt_12_zeros', rt_12))

    def rt_13():
        confs = torch.rand(100)
        fig = reason_v.plot_routing_distribution(confs, threshold=0.5)
        assert_fig_valid(fig, 'rt dist')
    tests.append(('rt_13_distribution', rt_13))

    def rt_14():
        stages = ['Enc', 'WS', 'Pre-R', 'Out']
        confs = [0.4, 0.6, 0.75, 0.9]
        fig = reason_v.plot_confidence_waterfall(stages, confs)
        assert_fig_valid(fig, 'rt waterfall')
    tests.append(('rt_14_waterfall', rt_14))

    def rt_15():
        trace = [{'confidence': 0.4, 'residual_norm': 2.0},
                 {'confidence': 0.7, 'residual_norm': 0.5}]
        fig = reason_v.plot_trace_tree(trace)
        assert_fig_valid(fig, 'rt tree')
    tests.append(('rt_15_tree', rt_15))

    def rt_16():
        acts = torch.rand(50, 6)
        fig = reason_v.plot_rule_coactivation(acts, [f'R{i}' for i in range(6)])
        assert_fig_valid(fig, 'rt coact')
    tests.append(('rt_16_coactivation', rt_16))

    def rt_17():
        fig = reason_v.plot_routing_decision(0.5)
        assert_png_header(fig, 'rt png')
    tests.append(('rt_17_png', rt_17))

    def rt_18():
        confs = torch.rand(20) * 0.5 + 0.4
        fig = reason_v.plot_routing_over_time(confs, threshold=0.7)
        assert_fig_valid(fig, 'rt over time')
    tests.append(('rt_18_over_time', rt_18))

    def rt_19():
        rules = ['A', 'B', 'C']; acts = np.array([0.9, 0.5, 0.1])
        fig = reason_v.plot_rule_activation(rules, acts)
        assert_fig_valid(fig, 'rt numpy')
    tests.append(('rt_19_numpy', rt_19))

    # ======================================================================
    # EMBEDDING PROJECTOR (17 tests)
    # ======================================================================

    def ep_01():
        fig = emb_p.plot_pca(torch.randn(100, 64))
        assert_fig_valid(fig, 'ep pca')
    tests.append(('ep_01_pca', ep_01))

    def ep_02():
        fig = emb_p.plot_pca(torch.randn(100, 64), labels=torch.randint(0, 5, (100,)))
        assert_fig_valid(fig, 'ep pca labels')
    tests.append(('ep_02_pca_labels', ep_02))

    def ep_03():
        fig = emb_p.plot_pca(torch.randn(50, 32))
        ax = fig.axes[0]
        assert 'PC' in ax.get_xlabel(); plt.close(fig)
    tests.append(('ep_03_pca_axes', ep_03))

    def ep_04():
        fig = emb_p.plot_tsne(torch.randn(80, 32), perplexity=10)
        assert_fig_valid(fig, 'ep tsne')
    tests.append(('ep_04_tsne', ep_04))

    def ep_05():
        fig = emb_p.plot_tsne(torch.randn(80, 32), labels=torch.randint(0, 3, (80,)),
                               perplexity=10)
        assert_fig_valid(fig, 'ep tsne labels')
    tests.append(('ep_05_tsne_labels', ep_05))

    def ep_06():
        fig = emb_p.plot_tsne(torch.randn(3, 32), perplexity=30)
        assert_fig_valid(fig, 'ep tsne fallback')
    tests.append(('ep_06_tsne_fallback', ep_06))

    def ep_07():
        fig = emb_p.plot_pca(torch.randn(1, 32))
        assert_fig_valid(fig, 'ep single')
    tests.append(('ep_07_single_point', ep_07))

    def ep_08():
        fig = emb_p.plot_pca(torch.zeros(50, 32))
        assert_fig_valid(fig, 'ep zeros')
    tests.append(('ep_08_zeros', ep_08))

    def ep_09():
        e = torch.randn(50, 32); e[10, 5] = float('nan')
        fig = emb_p.plot_pca(e)
        assert_fig_valid(fig, 'ep nan')
    tests.append(('ep_09_nan', ep_09))

    def ep_10():
        fig = emb_p.plot_pca_variance(torch.randn(100, 64), max_components=10)
        assert_fig_valid(fig, 'ep scree')
    tests.append(('ep_10_scree', ep_10))

    def ep_11():
        fig = emb_p.plot_embedding_norms(torch.randn(100, 64))
        assert_fig_valid(fig, 'ep norms')
    tests.append(('ep_11_norms', ep_11))

    def ep_12():
        fig = emb_p.plot_cosine_similarity_matrix(torch.randn(20, 32))
        assert_fig_valid(fig, 'ep cosine')
    tests.append(('ep_12_cosine', ep_12))

    def ep_13():
        fig = emb_p.plot_nearest_neighbors(torch.randn(50, 32), query_idx=0, k=5)
        assert_fig_valid(fig, 'ep nn')
    tests.append(('ep_13_nn', ep_13))

    def ep_14():
        fig = emb_p.plot_pca(np.random.randn(80, 32))
        assert_fig_valid(fig, 'ep numpy')
    tests.append(('ep_14_numpy', ep_14))

    def ep_15():
        fig = emb_p.plot_pca(torch.randn(50, 4096))
        assert_fig_valid(fig, 'ep high dim')
    tests.append(('ep_15_high_dim', ep_15))

    def ep_16():
        fig = emb_p.plot_pca(torch.randn(50, 32))
        assert_png_header(fig, 'ep png')
    tests.append(('ep_16_png', ep_16))

    def ep_17():
        fig = emb_p.plot_pca(torch.randn(50, 32), title='My Emb')
        assert 'My Emb' in fig.axes[0].get_title(); plt.close(fig)
    tests.append(('ep_17_custom_title', ep_17))

    # ======================================================================
    # TRAINING DASHBOARD (16 tests)
    # ======================================================================

    def td_01():
        fig = dash.plot_loss_curves({'loss': [1.0, 0.8, 0.6, 0.4, 0.3]})
        assert_fig_valid(fig, 'td loss')
    tests.append(('td_01_loss', td_01))

    def td_02():
        fig = dash.plot_loss_curves({'loss': [1.0, 0.5], 'accuracy': [0.2, 0.8]})
        assert_fig_valid(fig, 'td multi')
    tests.append(('td_02_multi_metric', td_02))

    def td_03():
        fig = dash.plot_loss_curves({'loss': [1.0, 0.5]})
        ax = fig.axes[0]
        assert 'Step' in ax.get_xlabel(); plt.close(fig)
    tests.append(('td_03_axes', td_03))

    def td_04():
        neuro = {'DA': [0.5, 0.6], 'ACh': [0.3, 0.4],
                 'NE': [0.7, 0.6], '5-HT': [0.4, 0.5]}
        fig = dash.plot_neuromodulator_levels(neuro)
        assert_fig_valid(fig, 'td neuro')
    tests.append(('td_04_neuro', td_04))

    def td_05():
        neuro = {'DA': [0.5, 0.6], 'ACh': [0.3, 0.4]}
        fig = dash.plot_neuromodulator_levels(neuro)
        ax = fig.axes[0]
        assert len(ax.get_lines()) >= 2; plt.close(fig)
    tests.append(('td_05_neuro_lines', td_05))

    def td_06():
        fig = dash.plot_phase_transitions(
            {'loss': [1.0, 0.8, 0.6, 0.4, 0.3, 0.2]}, [2, 4])
        assert_fig_valid(fig, 'td phase')
    tests.append(('td_06_phase', td_06))

    def td_07():
        with tempfile.TemporaryDirectory() as td_dir:
            db = TrainingDashboard(config=VizConfig(dpi=72))
            path = db.generate_report(
                td_dir,
                metrics={'loss': [1.0, 0.5], 'accuracy': [0.2, 0.8]},
                neuromodulator_levels={'DA': [0.5, 0.9], 'ACh': [0.3, 0.7],
                                       'NE': [0.8, 0.4], '5-HT': [0.4, 0.6]},
                phase_boundaries=[1],
            )
            assert os.path.exists(path)
            with open(path) as f:
                content = f.read()
            assert '<html' in content.lower()
            assert 'loss' in content.lower()
    tests.append(('td_07_report', td_07))

    def td_08():
        with tempfile.TemporaryDirectory() as td_dir:
            db = TrainingDashboard(config=VizConfig(dpi=72))
            path = db.generate_report(td_dir, metrics={'loss': [1.0, 0.5]})
            with open(path) as f:
                content = f.read()
            assert 'data:image/png;base64,' in content
    tests.append(('td_08_report_images', td_08))

    def td_09():
        fig = dash.plot_loss_curves({'loss': np.random.rand(50).tolist()},
                                     smooth_window=5)
        assert_fig_valid(fig, 'td smooth')
    tests.append(('td_09_smooth', td_09))

    def td_10():
        lr = [0.001 * (0.99 ** i) for i in range(100)]
        fig = dash.plot_learning_rate_schedule(lr)
        assert_fig_valid(fig, 'td lr')
    tests.append(('td_10_lr', td_10))

    def td_11():
        losses = {'SNN': [0.3, 0.2, 0.15], 'HTM': [0.2, 0.15, 0.1]}
        fig = dash.plot_module_loss_breakdown(losses)
        assert_fig_valid(fig, 'td breakdown')
    tests.append(('td_11_breakdown', td_11))

    def td_12():
        grads = {'SNN': [1.0, 0.8, 0.6], 'HTM': [0.5, 0.4, 0.3]}
        fig = dash.plot_gradient_norms(grads)
        assert_fig_valid(fig, 'td grads')
    tests.append(('td_12_grads', td_12))

    def td_13():
        fig = dash.plot_loss_curves({'loss': [1.0, 0.5]})
        assert_png_header(fig, 'td png')
    tests.append(('td_13_png', td_13))

    def td_14():
        fig = dash.plot_loss_curves({'loss': torch.tensor([1.0, 0.5, 0.3])})
        assert_fig_valid(fig, 'td torch')
    tests.append(('td_14_torch', td_14))

    def td_15():
        fig = dash.plot_loss_curves({'loss': [1.0, 0.5]}, title='My Loss')
        assert 'My Loss' in fig.axes[0].get_title(); plt.close(fig)
    tests.append(('td_15_custom_title', td_15))

    def td_16():
        with tempfile.TemporaryDirectory() as td_dir:
            db = TrainingDashboard(config=VizConfig(dpi=72))
            path = db.generate_report(td_dir)
            assert os.path.exists(path)
            with open(path) as f:
                content = f.read()
            assert '<html' in content.lower()
    tests.append(('td_16_empty_report', td_16))

    return tests


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_tests():
    """Run all tests and print results. Returns (passed, failed)."""
    tests = _collect_tests()
    results = []

    for name, func in tests:
        try:
            func()
            results.append((name, True, ''))
        except Exception as e:
            results.append((name, False, f'{e}\n{traceback.format_exc()}'))

    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    total = len(results)

    print(f"\n{'='*70}")
    print(f"Visualization Test Suite: {passed}/{total} passed, {failed} failed")
    print(f"{'='*70}")

    # Group by module
    modules = {}
    for name, ok, err in results:
        prefix = name.split('_')[0]
        if prefix not in modules:
            modules[prefix] = []
        modules[prefix].append((name, ok, err))

    module_names = {
        'sr': 'SpikeRasterPlotter',
        'ah': 'AttentionHeatmapper',
        'ws': 'WorkspaceVisualizer',
        'rt': 'ReasoningTraceVisualizer',
        'ep': 'EmbeddingProjector',
        'td': 'TrainingDashboard',
    }

    for prefix, test_results in modules.items():
        mod_pass = sum(1 for _, ok, _ in test_results if ok)
        mod_fail = sum(1 for _, ok, _ in test_results if not ok)
        mod_name = module_names.get(prefix, prefix)
        status = 'PASS' if mod_fail == 0 else 'FAIL'
        print(f"\n  [{status}] {mod_name}: {mod_pass}/{len(test_results)}")
        for name, ok, err in test_results:
            mark = 'OK  ' if ok else 'FAIL'
            print(f"    [{mark}] {name}")
            if err and not ok:
                for line in err.strip().split('\n')[:2]:
                    print(f"             {line}")

    print(f"\n{'='*70}")
    print(f"Total: {passed} passed, {failed} failed out of {total} tests")
    print(f"{'='*70}")

    return passed, failed


def generate_pytest_file() -> str:
    """Generate pytest-compatible test file content."""
    tests = _collect_tests()

    lines = [
        '"""Auto-generated pytest test suite for visualization modules."""',
        '',
        'import matplotlib',
        "matplotlib.use('Agg')",
        '',
        'import sys',
        'import os',
        'import tempfile',
        '',
        'import matplotlib.pyplot as plt',
        'import numpy as np',
        'import torch',
        '',
        '# Add assets to path',
        'SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))',
        'SKILL_DIR = os.path.dirname(SCRIPT_DIR)',
        "ASSETS_DIR = os.path.join(SKILL_DIR, 'assets')",
        'if ASSETS_DIR not in sys.path:',
        '    sys.path.insert(0, ASSETS_DIR)',
        '',
        'from spike_raster_template import SpikeRasterPlotter',
        'from spike_raster_template import VizConfig as SpikeVizConfig',
        'from attention_heatmap_template import AttentionHeatmapper',
        'from attention_heatmap_template import VizConfig as AttnVizConfig',
        'from workspace_viz_template import WorkspaceVisualizer',
        'from workspace_viz_template import VizConfig as WsVizConfig',
        'from reasoning_trace_template import ReasoningTraceVisualizer',
        'from reasoning_trace_template import VizConfig as ReasonVizConfig',
        'from embedding_projector_template import EmbeddingProjector',
        'from embedding_projector_template import VizConfig as EmbVizConfig',
        'from viz_config_template import TrainingDashboard, VizConfig',
        '',
        '',
        'def assert_fig_valid(fig, label=""):',
        '    assert fig is not None, f"Figure is None ({label})"',
        '    assert len(fig.axes) > 0, f"No axes ({label})"',
        '    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:',
        '        fig.savefig(f.name, dpi=72, bbox_inches="tight")',
        '        plt.close(fig)',
        '        size = os.path.getsize(f.name)',
        '        os.unlink(f.name)',
        '    assert size > 500, f"PNG too small: {size} ({label})"',
        '',
        '',
    ]

    for name, _ in tests:
        lines.append(f'def test_{name}():')
        lines.append(f'    """Test {name}."""')
        lines.append(f'    from gen_viz_tests import _collect_tests')
        lines.append(f'    tests = dict(_collect_tests())')
        lines.append(f'    tests["{name}"]()')
        lines.append('')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Generate visualization test suite')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Write pytest file to this path')
    parser.add_argument('--run', action='store_true',
                        help='Run tests immediately')
    args = parser.parse_args()

    if args.output:
        content = generate_pytest_file()
        with open(args.output, 'w') as f:
            f.write(content)
        print(f"Wrote {len(_collect_tests())} tests to {args.output}")
        return 0

    # Default: run tests
    passed, failed = run_tests()
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
