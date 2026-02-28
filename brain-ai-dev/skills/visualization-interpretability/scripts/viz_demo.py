#!/usr/bin/env python3
"""
Visualization demo script for brain_ai.

Produces sample PNG visualizations from synthetic data for all six
visualization modules. Output directory defaults to /tmp/viz_demo/.

Usage:
    python scripts/viz_demo.py                     # Default output to /tmp/viz_demo/
    python scripts/viz_demo.py --output-dir ./demo  # Custom output directory
    python scripts/viz_demo.py --module spike       # Only spike raster demos

Requires: matplotlib, numpy, torch
Backend: Agg (headless, no display required)
"""

import matplotlib
matplotlib.use('Agg')

import sys
import os
import argparse
import time

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
# Synthetic data generators
# ---------------------------------------------------------------------------

def make_spike_data(n_neurons=200, n_timesteps=100, spike_rate=0.08,
                    batch_size=1, seed=42):
    """Generate synthetic spike trains with temporal structure."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Base random spikes
    spikes = (torch.rand(batch_size, n_timesteps, n_neurons) < spike_rate).float()

    # Add a burst pattern (neurons 20-40 fire together at steps 30-40)
    spikes[:, 30:40, 20:40] = (torch.rand(batch_size, 10, 20) < 0.6).float()

    # Add a wave pattern (neurons fire in sequence)
    for t in range(50, 70):
        neuron_idx = (t - 50) * (n_neurons // 20)
        if neuron_idx < n_neurons:
            spikes[:, t, max(0, neuron_idx - 5):min(n_neurons, neuron_idx + 5)] = 1.0

    return spikes


def make_attention_data(seq_len=12, n_heads=8, seed=42):
    """Generate synthetic attention weights with structure."""
    torch.manual_seed(seed)
    # Create causal-ish attention patterns
    weights = torch.randn(n_heads, seq_len, seq_len)
    # Add causal mask tendency
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1) * -5
    weights = weights + mask.unsqueeze(0)
    weights = torch.softmax(weights, dim=-1)
    return weights


def make_workspace_data(n_modalities=4, n_rounds=5, seed=42):
    """Generate synthetic workspace competition data."""
    torch.manual_seed(seed)
    # Start equal, one modality gradually wins
    history = []
    scores = torch.ones(n_modalities) / n_modalities
    for r in range(n_rounds):
        # Amplify winner
        noise = torch.randn(n_modalities) * 0.05
        scores = scores + noise
        scores[0] += 0.1  # Vision tends to win
        scores = torch.clamp(scores, 0, 1)
        scores = scores / scores.sum()
        history.append(scores.clone())
    return history


def make_reasoning_trace(n_steps=8, initial_conf=0.35, seed=42):
    """Generate synthetic System 2 reasoning trace."""
    np.random.seed(seed)
    trace = []
    conf = initial_conf
    for i in range(n_steps):
        delta = np.random.uniform(0.05, 0.15)
        conf = min(conf + delta, 0.98)
        residual = max(0.1, 3.0 * np.exp(-0.5 * i))
        entropy = max(0.5, 2.5 * np.exp(-0.3 * i))
        trace.append({
            'confidence': conf,
            'residual_norm': residual,
            'attention_entropy': entropy,
        })
    return trace


def make_embedding_data(n_samples=300, n_features=128, n_classes=5, seed=42):
    """Generate synthetic clustered embedding data."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    embeddings = []
    labels = []

    for c in range(n_classes):
        center = torch.randn(n_features) * 2
        n_per_class = n_samples // n_classes
        cluster = center + torch.randn(n_per_class, n_features) * 0.5
        embeddings.append(cluster)
        labels.extend([c] * n_per_class)

    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.tensor(labels)
    return embeddings, labels


def make_training_metrics(n_steps=200, seed=42):
    """Generate synthetic training metrics."""
    np.random.seed(seed)

    steps = np.arange(n_steps)
    # Loss: exponential decay with noise
    loss = 2.0 * np.exp(-steps / 50) + np.random.randn(n_steps) * 0.05
    loss = np.maximum(loss, 0.01)

    # Accuracy: sigmoid growth
    accuracy = 1.0 / (1.0 + np.exp(-(steps - 80) / 20)) + np.random.randn(n_steps) * 0.02
    accuracy = np.clip(accuracy, 0, 1)

    # Neuromodulators
    da = 0.5 + 0.3 * np.sin(steps / 30) + np.random.randn(n_steps) * 0.03
    ach = 0.4 + 0.2 * np.cos(steps / 25) + np.random.randn(n_steps) * 0.02
    ne = 0.8 * np.exp(-steps / 80) + 0.2 + np.random.randn(n_steps) * 0.03
    sht = 0.3 + 0.15 * np.sin(steps / 40 + 1) + np.random.randn(n_steps) * 0.02

    da = np.clip(da, 0, 1)
    ach = np.clip(ach, 0, 1)
    ne = np.clip(ne, 0, 1)
    sht = np.clip(sht, 0, 1)

    metrics = {'loss': loss.tolist(), 'accuracy': accuracy.tolist()}
    neuro = {'DA': da.tolist(), 'ACh': ach.tolist(),
             'NE': ne.tolist(), '5-HT': sht.tolist()}
    return metrics, neuro


# ---------------------------------------------------------------------------
# Demo functions (one per module)
# ---------------------------------------------------------------------------

def demo_spike_raster(output_dir: str):
    """Generate spike raster demo visualizations."""
    print("  Generating spike raster demos...")
    config = SpikeVizConfig(figsize=(12, 8), dpi=150, save_dir=output_dir)
    plotter = SpikeRasterPlotter(config)

    spikes = make_spike_data(n_neurons=200, n_timesteps=100)
    membrane = torch.randn(1, 100, 200) * 0.6

    # Raster plot
    fig = plotter.plot_raster(spikes, layer_boundaries=[50, 100, 150])
    path = plotter.save_figure(fig, 'spike_raster')
    print(f"    -> {path} ({os.path.getsize(path)} bytes)")

    # Firing rate heatmap
    fig = plotter.plot_firing_rates(spikes, window=5)
    path = plotter.save_figure(fig, 'firing_rates')
    print(f"    -> {path} ({os.path.getsize(path)} bytes)")

    # Membrane potential
    fig = plotter.plot_membrane_potential(membrane, threshold=1.0,
                                          spikes=spikes, max_traces=4)
    path = plotter.save_figure(fig, 'membrane_potential')
    print(f"    -> {path} ({os.path.getsize(path)} bytes)")

    # Population activity
    fig = plotter.plot_population_activity(spikes)
    path = plotter.save_figure(fig, 'population_activity')
    print(f"    -> {path} ({os.path.getsize(path)} bytes)")

    # Spike count distribution
    fig = plotter.plot_spike_count_distribution(spikes, target_rate=0.08)
    path = plotter.save_figure(fig, 'spike_count_dist')
    print(f"    -> {path} ({os.path.getsize(path)} bytes)")


def demo_attention_heatmap(output_dir: str):
    """Generate attention heatmap demo visualizations."""
    print("  Generating attention heatmap demos...")
    config = AttnVizConfig(figsize=(10, 8), dpi=150, save_dir=output_dir)
    hm = AttentionHeatmapper(config)

    weights = make_attention_data(seq_len=12, n_heads=8)

    # Single head attention
    fig = hm.plot_attention(weights[0],
                             labels=[f'Tok{i}' for i in range(12)])
    path = hm.save_figure(fig, 'attention_single_head')
    print(f"    -> {path} ({os.path.getsize(path)} bytes)")

    # Multi-head grid
    fig = hm.plot_multi_head_attention(weights)
    path = hm.save_figure(fig, 'attention_multi_head')
    print(f"    -> {path} ({os.path.getsize(path)} bytes)")

    # Head entropy
    fig = hm.plot_head_entropy(weights)
    path = hm.save_figure(fig, 'attention_entropy')
    print(f"    -> {path} ({os.path.getsize(path)} bytes)")

    # Cross-modal attention
    cross_modal = {
        'vision->text': torch.tensor(0.65),
        'text->vision': torch.tensor(0.45),
        'vision->audio': torch.tensor(0.25),
        'audio->vision': torch.tensor(0.30),
        'text->audio': torch.tensor(0.15),
        'audio->text': torch.tensor(0.20),
    }
    fig = hm.plot_cross_modal_attention(cross_modal)
    path = hm.save_figure(fig, 'cross_modal_attention')
    print(f"    -> {path} ({os.path.getsize(path)} bytes)")

    # Workspace competition
    scores = torch.tensor([0.45, 0.30, 0.15, 0.10])
    fig = hm.plot_workspace_competition(
        scores, modality_names=['vision', 'text', 'audio', 'sensor'],
        threshold=0.3)
    path = hm.save_figure(fig, 'workspace_competition')
    print(f"    -> {path} ({os.path.getsize(path)} bytes)")


def demo_workspace(output_dir: str):
    """Generate workspace visualization demos."""
    print("  Generating workspace demos...")
    config = WsVizConfig(figsize=(12, 8), dpi=150, save_dir=output_dir)
    viz = WorkspaceVisualizer(config)

    history = make_workspace_data(n_modalities=4, n_rounds=6)

    # Competition dynamics
    fig = viz.plot_competition_dynamics(
        history, modality_names=['vision', 'text', 'audio', 'sensor'],
        threshold=0.3)
    path = viz.save_figure(fig, 'competition_dynamics')
    print(f"    -> {path} ({os.path.getsize(path)} bytes)")

    # Broadcast map
    broadcast = torch.tensor([0.9, 0.7, 0.8, 0.5, 0.6, 0.3])
    fig = viz.plot_broadcast_map(
        broadcast,
        module_names=['SNN Core', 'HTM', 'Workspace', 'Reasoning', 'Decision', 'Meta'],
        source_name='Vision')
    path = viz.save_figure(fig, 'broadcast_map')
    print(f"    -> {path} ({os.path.getsize(path)} bytes)")

    # Working memory slots
    torch.manual_seed(42)
    memory = torch.randn(7, 64) * 0.5
    # Make some slots more active
    memory[0] *= 3.0  # Strong visual slot
    memory[1] *= 2.0  # Active text slot
    fig = viz.plot_working_memory_slots(
        memory, labels=['Visual', 'Textual', 'Auditory', 'Context',
                        'Engram', 'Goal', 'Buffer'])
    path = viz.save_figure(fig, 'working_memory')
    print(f"    -> {path} ({os.path.getsize(path)} bytes)")

    # Slot dynamics
    slot_history = torch.randn(30, 7, 64) * 0.3
    # Slot 0 becomes more active over time
    for t in range(30):
        slot_history[t, 0] *= (1 + t * 0.1)
    fig = viz.plot_slot_dynamics(slot_history)
    path = viz.save_figure(fig, 'slot_dynamics')
    print(f"    -> {path} ({os.path.getsize(path)} bytes)")

    # Information flow
    flow = {
        'Encoders': 0.95,
        'SNN Core': 0.82,
        'HTM': 0.68,
        'Workspace': 0.90,
        'Reasoning': 0.75,
        'Decision': 0.60,
        'Meta': 0.45,
    }
    fig = viz.plot_information_flow(flow)
    path = viz.save_figure(fig, 'information_flow')
    print(f"    -> {path} ({os.path.getsize(path)} bytes)")


def demo_reasoning_trace(output_dir: str):
    """Generate reasoning trace demo visualizations."""
    print("  Generating reasoning trace demos...")
    config = ReasonVizConfig(figsize=(12, 6), dpi=150, save_dir=output_dir)
    viz = ReasoningTraceVisualizer(config)

    # Routing decision (System 2)
    fig = viz.plot_routing_decision(0.65, threshold=0.8)
    path = viz.save_figure(fig, 'routing_sys2')
    print(f"    -> {path} ({os.path.getsize(path)} bytes)")

    # Routing decision (System 1)
    fig = viz.plot_routing_decision(0.92, threshold=0.8)
    path = viz.save_figure(fig, 'routing_sys1')
    print(f"    -> {path} ({os.path.getsize(path)} bytes)")

    # Routing distribution
    torch.manual_seed(42)
    confs = torch.rand(200) * 0.6 + 0.3  # range [0.3, 0.9]
    fig = viz.plot_routing_distribution(confs, threshold=0.7)
    path = viz.save_figure(fig, 'routing_distribution')
    print(f"    -> {path} ({os.path.getsize(path)} bytes)")

    # System 2 steps
    trace = make_reasoning_trace(n_steps=8, initial_conf=0.35)
    fig = viz.plot_system2_steps(trace, threshold=0.8)
    path = viz.save_figure(fig, 'system2_steps')
    print(f"    -> {path} ({os.path.getsize(path)} bytes)")

    # Trace tree
    fig = viz.plot_trace_tree(trace, threshold=0.8)
    path = viz.save_figure(fig, 'trace_tree')
    print(f"    -> {path} ({os.path.getsize(path)} bytes)")

    # Rule activations
    rules = [
        'is_animal(x) AND has_fur(x)',
        'is_moving(x) AND is_fast(x)',
        'is_large(x) AND visible(x)',
        'is_dangerous(x)',
        'is_domestic(x)',
        'has_pattern(x) AND striped(x)',
        'near_water(x)',
        'is_predator(x)',
    ]
    acts = torch.tensor([0.92, 0.78, 0.65, 0.42, 0.18, 0.55, 0.08, 0.71])
    fig = viz.plot_rule_activation(rules, acts)
    path = viz.save_figure(fig, 'rule_activations')
    print(f"    -> {path} ({os.path.getsize(path)} bytes)")

    # Rule activation over time
    torch.manual_seed(42)
    rule_hist = torch.rand(50, 8)
    fig = viz.plot_rule_activation_over_time(rules, rule_hist, top_k=5)
    path = viz.save_figure(fig, 'rule_over_time')
    print(f"    -> {path} ({os.path.getsize(path)} bytes)")

    # Confidence waterfall
    stages = ['Encoder', 'Workspace', 'Pre-Reasoning', 'S2-Step1',
              'S2-Step3', 'Output']
    confs_pipeline = [0.35, 0.52, 0.60, 0.72, 0.85, 0.93]
    fig = viz.plot_confidence_waterfall(stages, confs_pipeline, threshold=0.8)
    path = viz.save_figure(fig, 'confidence_waterfall')
    print(f"    -> {path} ({os.path.getsize(path)} bytes)")


def demo_embedding_projector(output_dir: str):
    """Generate embedding projector demo visualizations."""
    print("  Generating embedding projector demos...")
    config = EmbVizConfig(figsize=(10, 8), dpi=150, save_dir=output_dir,
                           max_points=1000)
    proj = EmbeddingProjector(config)

    embeddings, labels = make_embedding_data(n_samples=300, n_features=128,
                                              n_classes=5)

    # PCA projection
    fig = proj.plot_pca(embeddings, labels=labels)
    path = proj.save_figure(fig, 'pca_projection')
    print(f"    -> {path} ({os.path.getsize(path)} bytes)")

    # t-SNE projection
    fig = proj.plot_tsne(embeddings, labels=labels, perplexity=20,
                          random_state=42)
    path = proj.save_figure(fig, 'tsne_projection')
    print(f"    -> {path} ({os.path.getsize(path)} bytes)")

    # PCA variance scree plot
    fig = proj.plot_pca_variance(embeddings, max_components=15)
    path = proj.save_figure(fig, 'pca_variance')
    print(f"    -> {path} ({os.path.getsize(path)} bytes)")

    # Embedding norms
    fig = proj.plot_embedding_norms(embeddings, labels=labels)
    path = proj.save_figure(fig, 'embedding_norms')
    print(f"    -> {path} ({os.path.getsize(path)} bytes)")

    # Cosine similarity matrix (small subset)
    fig = proj.plot_cosine_similarity_matrix(
        embeddings[:20],
        labels=[f'S{i} (C{int(labels[i])})' for i in range(20)])
    path = proj.save_figure(fig, 'cosine_similarity')
    print(f"    -> {path} ({os.path.getsize(path)} bytes)")

    # Nearest neighbors
    fig = proj.plot_nearest_neighbors(embeddings, query_idx=0, k=10)
    path = proj.save_figure(fig, 'nearest_neighbors')
    print(f"    -> {path} ({os.path.getsize(path)} bytes)")


def demo_training_dashboard(output_dir: str):
    """Generate training dashboard demo visualizations."""
    print("  Generating training dashboard demos...")
    config = VizConfig(figsize=(12, 8), dpi=150, save_dir=output_dir)
    dashboard = TrainingDashboard(log_dir=output_dir, config=config)

    metrics, neuro = make_training_metrics(n_steps=200)
    phase_boundaries = [40, 80, 120, 160]
    phase_names = ['SNN Core', 'Encoders', 'HTM', 'Workspace', 'Integration']

    # Loss curves
    fig = dashboard.plot_loss_curves(metrics)
    path = dashboard.save_figure(fig, 'loss_curves')
    print(f"    -> {path} ({os.path.getsize(path)} bytes)")

    # Loss curves smoothed
    fig = dashboard.plot_loss_curves(metrics, smooth_window=10,
                                      title='Smoothed Training Curves')
    path = dashboard.save_figure(fig, 'loss_curves_smooth')
    print(f"    -> {path} ({os.path.getsize(path)} bytes)")

    # Neuromodulator levels
    fig = dashboard.plot_neuromodulator_levels(neuro)
    path = dashboard.save_figure(fig, 'neuromodulator_levels')
    print(f"    -> {path} ({os.path.getsize(path)} bytes)")

    # Phase transitions
    fig = dashboard.plot_phase_transitions(metrics, phase_boundaries,
                                            phase_names=phase_names)
    path = dashboard.save_figure(fig, 'phase_transitions')
    print(f"    -> {path} ({os.path.getsize(path)} bytes)")

    # Learning rate schedule
    lr = [0.001 * (0.95 ** (i // 10)) for i in range(200)]
    fig = dashboard.plot_learning_rate_schedule(lr)
    path = dashboard.save_figure(fig, 'lr_schedule')
    print(f"    -> {path} ({os.path.getsize(path)} bytes)")

    # Module loss breakdown
    np.random.seed(42)
    module_losses = {
        'SNN': (0.5 * np.exp(-np.arange(200) / 60) +
                np.random.randn(200) * 0.02).clip(0).tolist(),
        'HTM': (0.3 * np.exp(-np.arange(200) / 70) +
                np.random.randn(200) * 0.015).clip(0).tolist(),
        'Workspace': (0.2 * np.exp(-np.arange(200) / 80) +
                      np.random.randn(200) * 0.01).clip(0).tolist(),
    }
    fig = dashboard.plot_module_loss_breakdown(module_losses)
    path = dashboard.save_figure(fig, 'module_loss_breakdown')
    print(f"    -> {path} ({os.path.getsize(path)} bytes)")

    # Gradient norms
    grad_norms = {
        'SNN': (np.exp(-np.arange(200) / 100) * 2 +
                np.random.randn(200) * 0.1).clip(0.01).tolist(),
        'HTM': (np.exp(-np.arange(200) / 120) * 1.5 +
                np.random.randn(200) * 0.08).clip(0.01).tolist(),
    }
    fig = dashboard.plot_gradient_norms(grad_norms)
    path = dashboard.save_figure(fig, 'gradient_norms')
    print(f"    -> {path} ({os.path.getsize(path)} bytes)")

    # Full HTML report
    report_path = dashboard.generate_report(
        output_dir=output_dir,
        metrics=metrics,
        neuromodulator_levels=neuro,
        phase_boundaries=phase_boundaries,
        phase_names=phase_names,
        title='Brain AI Training Report (Demo)',
    )
    print(f"    -> {report_path} ({os.path.getsize(report_path)} bytes)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEMOS = {
    'spike': ('Spike Raster', demo_spike_raster),
    'attention': ('Attention Heatmap', demo_attention_heatmap),
    'workspace': ('Workspace', demo_workspace),
    'reasoning': ('Reasoning Trace', demo_reasoning_trace),
    'embedding': ('Embedding Projector', demo_embedding_projector),
    'dashboard': ('Training Dashboard', demo_training_dashboard),
}


def main():
    parser = argparse.ArgumentParser(
        description='Generate sample visualization PNGs from synthetic data')
    parser.add_argument('--output-dir', type=str, default='/tmp/viz_demo',
                        help='Output directory for demo PNGs (default: /tmp/viz_demo)')
    parser.add_argument('--module', type=str, default=None,
                        choices=list(DEMOS.keys()) + ['all'],
                        help='Which module to demo (default: all)')
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    modules = list(DEMOS.keys()) if args.module in (None, 'all') else [args.module]

    print("=" * 60)
    print("Visualization Demo -- Generating Sample PNGs")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    start = time.time()
    total_files = 0

    for mod_key in modules:
        mod_name, demo_func = DEMOS[mod_key]
        print(f"\n[{mod_name}]")
        before = len([f for f in os.listdir(output_dir)
                      if f.endswith(('.png', '.html'))])
        try:
            demo_func(output_dir)
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
        after = len([f for f in os.listdir(output_dir)
                     if f.endswith(('.png', '.html'))])
        n_new = after - before
        total_files += n_new

    elapsed = time.time() - start

    # Summary
    all_files = sorted([f for f in os.listdir(output_dir)
                        if f.endswith(('.png', '.html'))])
    total_size = sum(os.path.getsize(os.path.join(output_dir, f))
                     for f in all_files)

    print(f"\n{'='*60}")
    print(f"Demo complete!")
    print(f"  Files generated: {len(all_files)}")
    print(f"  Total size: {total_size / 1024:.1f} KB")
    print(f"  Time elapsed: {elapsed:.1f}s")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    if all_files:
        print("\nGenerated files:")
        for f in all_files:
            size = os.path.getsize(os.path.join(output_dir, f))
            print(f"  {f:40s}  {size:>8,d} bytes")

    return 0


if __name__ == '__main__':
    sys.exit(main())
