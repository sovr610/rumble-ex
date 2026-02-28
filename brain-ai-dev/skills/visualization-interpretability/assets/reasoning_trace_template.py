"""
Reasoning Trace Visualization Template for brain_ai.

Provides ReasoningTraceVisualizer class for rendering dual-process
reasoning traces: routing decisions, System 2 step traces, rule
activation bar charts, confidence evolution, and trace trees.

Usage:
    from reasoning_trace_template import ReasoningTraceVisualizer, VizConfig
    config = VizConfig()
    viz = ReasoningTraceVisualizer(config)
    fig = viz.plot_routing_decision(confidence=0.65, threshold=0.8)

Requires: matplotlib, numpy, torch
Backend: Agg (headless, no display required)
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import torch
import os
import tempfile
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Union


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class VizConfig:
    """Visualization configuration."""
    backend: str = "matplotlib"
    output_format: str = "png"
    dpi: int = 150
    figsize: Tuple[int, int] = (12, 8)
    colormap: str = "viridis"
    dark_mode: bool = False
    save_dir: str = "viz/"
    max_neurons: int = 100
    max_timesteps: int = 500
    tsne_perplexity: int = 30
    umap_n_neighbors: int = 15
    max_points: int = 5000


# ---------------------------------------------------------------------------
# Color Constants
# ---------------------------------------------------------------------------

SYSTEM1_COLOR = '#00bcd4'   # Cyan
SYSTEM2_COLOR = '#e91e63'   # Magenta
THRESHOLD_COLOR = '#f44336'  # Red
CONVERGED_COLOR = '#4caf50'  # Green
WARNING_COLOR = '#ff9800'    # Orange

RULE_COLORS = {
    'high': '#2ca02c',     # Green (activation > 0.8)
    'medium': '#ff7f0e',   # Orange (0.5 - 0.8)
    'low': '#e57373',      # Light red (0.2 - 0.5)
    'inactive': '#bdbdbd', # Gray (< 0.2)
}

NEUROMODULATOR_COLORS = {
    'DA': '#d62728',
    'ACh': '#2ca02c',
    'NE': '#1f77b4',
    '5-HT': '#bcbd22',
}

CONFIDENCE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'confidence',
    ['#d32f2f', '#ff9800', '#cddc39', '#4caf50'],
    N=256,
)


# ---------------------------------------------------------------------------
# ReasoningTraceVisualizer
# ---------------------------------------------------------------------------

class ReasoningTraceVisualizer:
    """Render dual-process reasoning traces as plots and graphs.

    Visualizes routing decisions (System 1 vs System 2), iterative
    System 2 reasoning step traces, and symbolic rule activations.

    Args:
        config: VizConfig instance controlling appearance.
    """

    def __init__(self, config: Optional[VizConfig] = None):
        self.config = config or VizConfig()

    def _to_numpy(self, tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Convert tensor to numpy, detaching if needed."""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return np.asarray(tensor)

    def _get_rule_color(self, activation: float) -> str:
        """Return color for a rule based on its activation level."""
        if activation > 0.8:
            return RULE_COLORS['high']
        elif activation > 0.5:
            return RULE_COLORS['medium']
        elif activation > 0.2:
            return RULE_COLORS['low']
        else:
            return RULE_COLORS['inactive']

    # -- Public API ----------------------------------------------------------

    def plot_routing_decision(
        self,
        confidence: float,
        threshold: float = 0.8,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a gauge-style routing decision between System 1 and System 2.

        Args:
            confidence: Current confidence value in [0, 1].
            threshold: Routing threshold. If confidence >= threshold,
                       routes to System 1; otherwise System 2.
            title: Custom plot title.

        Returns:
            matplotlib Figure.
        """
        confidence = float(confidence)
        threshold = float(threshold)
        route = 'System 1' if confidence >= threshold else 'System 2'
        bar_color = SYSTEM1_COLOR if confidence >= threshold else SYSTEM2_COLOR

        fig, ax = plt.subplots(figsize=(self.config.figsize[0], 3))

        # Draw gauge bar
        ax.barh(0, confidence, height=0.5, color=bar_color,
                label=f'Confidence = {confidence:.3f}')
        ax.barh(0, 1.0 - confidence, left=confidence, height=0.5,
                color='#e0e0e0')

        # Threshold line
        ax.axvline(x=threshold, color=THRESHOLD_COLOR, linestyle='--',
                   linewidth=2.5, label=f'Threshold ({threshold})')

        # Zone labels
        ax.text(threshold / 2, -0.5, 'System 2\n(slow, deliberative)',
                ha='center', va='top', fontsize=9, color=SYSTEM2_COLOR,
                fontweight='bold')
        ax.text((threshold + 1.0) / 2, -0.5, 'System 1\n(fast, automatic)',
                ha='center', va='top', fontsize=9, color=SYSTEM1_COLOR,
                fontweight='bold')

        # Confidence annotation
        ax.text(confidence, 0, f'  {confidence:.3f}', va='center',
                fontsize=12, fontweight='bold', color=bar_color)

        ax.set_xlim(0, 1)
        ax.set_ylim(-1.0, 0.8)
        ax.set_xlabel('Confidence')
        ax.set_yticks([])
        ax.set_title(title or f'Routing Decision: {route}')
        ax.legend(loc='upper right', fontsize=9)
        fig.tight_layout()
        return fig

    def plot_routing_distribution(
        self,
        confidences: Union[torch.Tensor, np.ndarray],
        threshold: float = 0.8,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Plot histogram of routing decisions for a batch.

        Args:
            confidences: 1D tensor/array of confidence values.
            threshold: Routing threshold.
            title: Custom title.

        Returns:
            matplotlib Figure.
        """
        data = self._to_numpy(confidences).ravel()
        n_total = len(data)
        n_sys1 = int(np.sum(data >= threshold))
        n_sys2 = n_total - n_sys1
        frac1 = n_sys1 / max(n_total, 1) * 100
        frac2 = n_sys2 / max(n_total, 1) * 100

        fig, ax = plt.subplots(figsize=self.config.figsize)

        bins = np.linspace(0, 1, 21)
        counts, edges, patches = ax.hist(data, bins=bins, edgecolor='white',
                                         linewidth=0.8)

        # Color bars by routing
        for patch, left_edge in zip(patches, edges[:-1]):
            mid = left_edge + (edges[1] - edges[0]) / 2
            if mid >= threshold:
                patch.set_facecolor(SYSTEM1_COLOR)
            else:
                patch.set_facecolor(SYSTEM2_COLOR)

        ax.axvline(x=threshold, color=THRESHOLD_COLOR, linestyle='--',
                   linewidth=2, label=f'Threshold ({threshold})')

        # Annotations
        ax.text(0.02, 0.95,
                f'System 1: {frac1:.1f}% ({n_sys1}/{n_total})',
                transform=ax.transAxes, fontsize=10, color=SYSTEM1_COLOR,
                fontweight='bold', va='top')
        ax.text(0.02, 0.88,
                f'System 2: {frac2:.1f}% ({n_sys2}/{n_total})',
                transform=ax.transAxes, fontsize=10, color=SYSTEM2_COLOR,
                fontweight='bold', va='top')

        ax.set_xlabel('Confidence')
        ax.set_ylabel('Count')
        ax.set_title(title or 'Routing Decision Distribution')
        ax.set_xlim(0, 1)
        ax.legend(loc='upper right')
        fig.tight_layout()
        return fig

    def plot_system2_steps(
        self,
        trace: List[Dict],
        threshold: float = 0.8,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Plot System 2 reasoning step trace with confidence evolution.

        Args:
            trace: List of dicts per step, each containing at minimum:
                   - 'confidence': float
                   Optionally:
                   - 'residual_norm': float (update magnitude)
                   - 'attention_entropy': float
            threshold: Convergence threshold.
            title: Custom title.

        Returns:
            matplotlib Figure.
        """
        n_steps = len(trace)
        steps = np.arange(n_steps)
        confidences = np.array([s.get('confidence', 0.0) for s in trace])
        has_residuals = any('residual_norm' in s for s in trace)
        has_entropy = any('attention_entropy' in s for s in trace)

        n_panels = 1 + int(has_residuals) + int(has_entropy)
        fig, axes = plt.subplots(n_panels, 1,
                                 figsize=(self.config.figsize[0],
                                          3.5 * n_panels),
                                 squeeze=False)
        axes = axes.ravel()

        # Panel 1: Confidence evolution
        ax = axes[0]
        colors = [CONFIDENCE_CMAP(c) for c in confidences]
        ax.plot(steps, confidences, '-o', color=SYSTEM2_COLOR, linewidth=2,
                markersize=8, zorder=3)
        for i, (s, c) in enumerate(zip(steps, confidences)):
            ax.scatter(s, c, color=CONFIDENCE_CMAP(c), s=80, zorder=4,
                       edgecolors='black', linewidths=0.5)

        ax.axhline(y=threshold, color=THRESHOLD_COLOR, linestyle='--',
                   linewidth=1.5, alpha=0.8, label=f'Threshold ({threshold})')
        ax.fill_between(steps, 0, confidences, alpha=0.1, color=SYSTEM2_COLOR)

        # Mark convergence
        converged = np.where(confidences >= threshold)[0]
        if len(converged) > 0:
            conv_step = converged[0]
            ax.axvline(x=conv_step, color=CONVERGED_COLOR, linestyle=':',
                       linewidth=2, label=f'Converged at step {conv_step}')
            ax.scatter(conv_step, confidences[conv_step], marker='*',
                       s=200, color=CONVERGED_COLOR, zorder=5,
                       edgecolors='black', linewidths=0.5)
        else:
            ax.text(0.5, 0.05, 'DID NOT CONVERGE',
                    transform=ax.transAxes, ha='center', fontsize=12,
                    color=WARNING_COLOR, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff3e0',
                              edgecolor=WARNING_COLOR))

        ax.set_xlabel('System 2 Step')
        ax.set_ylabel('Confidence')
        ax.set_title('Confidence Evolution')
        ax.set_xlim(-0.3, n_steps - 0.7)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(steps)
        ax.set_xticklabels([f'S{i}' for i in steps])
        ax.legend(loc='lower right', fontsize=9)

        panel_idx = 1

        # Panel 2: Residual norms
        if has_residuals:
            ax = axes[panel_idx]
            residuals = np.array([s.get('residual_norm', 0.0) for s in trace])
            ax.bar(steps, residuals, color=SYSTEM2_COLOR, edgecolor='black',
                   linewidth=0.5, alpha=0.8)
            ax.set_xlabel('System 2 Step')
            ax.set_ylabel('Residual Norm')
            ax.set_title('Update Magnitude per Step')
            ax.set_xticks(steps)
            ax.set_xticklabels([f'S{i}' for i in steps])
            panel_idx += 1

        # Panel 3: Attention entropy
        if has_entropy:
            ax = axes[panel_idx]
            entropies = np.array([s.get('attention_entropy', 0.0) for s in trace])
            ax.plot(steps, entropies, '-s', color='#6A5ACD', linewidth=2,
                    markersize=7)
            ax.set_xlabel('System 2 Step')
            ax.set_ylabel('Attention Entropy')
            ax.set_title('Attention Entropy per Step')
            ax.set_xticks(steps)
            ax.set_xticklabels([f'S{i}' for i in steps])
            panel_idx += 1

        fig.suptitle(title or 'System 2 Reasoning Trace', fontsize=13, y=1.01)
        fig.tight_layout()
        return fig

    def plot_confidence_waterfall(
        self,
        stage_names: List[str],
        confidences: Union[torch.Tensor, np.ndarray, List[float]],
        threshold: float = 0.8,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Plot confidence at each pipeline stage as a waterfall chart.

        Args:
            stage_names: List of stage names (e.g., ['Encoder', 'WS', ...]).
            confidences: Confidence at each stage.
            threshold: Routing/convergence threshold.
            title: Custom title.

        Returns:
            matplotlib Figure.
        """
        if isinstance(confidences, (list, tuple)):
            data = np.array(confidences, dtype=float)
        else:
            data = self._to_numpy(confidences).ravel()

        n = len(stage_names)
        assert len(data) == n, f"Stage count ({n}) != confidence count ({len(data)})"

        fig, ax = plt.subplots(figsize=(max(8, n * 1.5), 5))

        colors = [CONFIDENCE_CMAP(c) for c in data]
        bars = ax.bar(range(n), data, color=colors, edgecolor='black',
                      linewidth=0.8)

        # Connect bars with lines
        for i in range(n - 1):
            ax.plot([i, i + 1], [data[i], data[i + 1]], 'k-', linewidth=1,
                    alpha=0.4)

        ax.axhline(y=threshold, color=THRESHOLD_COLOR, linestyle='--',
                   linewidth=1.5, alpha=0.7, label=f'Threshold ({threshold})')

        # Value annotations
        for bar, val in zip(bars, data):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9,
                    fontweight='bold')

        ax.set_xticks(range(n))
        ax.set_xticklabels(stage_names, rotation=30, ha='right')
        ax.set_xlabel('Pipeline Stage')
        ax.set_ylabel('Confidence')
        ax.set_title(title or 'Confidence Through Pipeline')
        ax.set_ylim(0, 1.15)
        ax.legend(loc='lower right', fontsize=9)
        fig.tight_layout()
        return fig

    def plot_rule_activation(
        self,
        rules: List[str],
        activations: Union[torch.Tensor, np.ndarray],
        title: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> plt.Figure:
        """Plot symbolic rule activations as a horizontal bar chart.

        Args:
            rules: List of rule name strings.
            activations: 1D tensor of activation values in [0, 1].
            title: Custom title.
            top_k: If set, show only top-K most active rules.

        Returns:
            matplotlib Figure.
        """
        data = self._to_numpy(activations).ravel()
        n = len(rules)
        assert len(data) == n, f"Rule count ({n}) != activation count ({len(data)})"

        # Sort by activation descending
        order = np.argsort(data)[::-1]
        if top_k is not None and top_k < n:
            order = order[:top_k]

        sorted_rules = [rules[i] for i in order]
        sorted_acts = data[order]
        n_display = len(order)

        fig, ax = plt.subplots(figsize=(self.config.figsize[0],
                                        max(4, n_display * 0.5)))
        y_pos = np.arange(n_display)
        colors = [self._get_rule_color(a) for a in sorted_acts]

        bars = ax.barh(y_pos, sorted_acts, color=colors, edgecolor='black',
                       linewidth=0.5, height=0.7)

        # Value annotations
        for bar, val in zip(bars, sorted_acts):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', fontsize=9)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_rules, fontsize=9)
        ax.set_xlabel('Activation (Truth Value)')
        ax.set_ylabel('Rule')
        ax.set_title(title or 'Rule Activations')
        ax.set_xlim(0, 1.15)
        ax.invert_yaxis()

        # Legend
        legend_patches = [
            mpatches.Patch(color=RULE_COLORS['high'], label='High (>0.8)'),
            mpatches.Patch(color=RULE_COLORS['medium'], label='Medium (0.5-0.8)'),
            mpatches.Patch(color=RULE_COLORS['low'], label='Low (0.2-0.5)'),
            mpatches.Patch(color=RULE_COLORS['inactive'], label='Inactive (<0.2)'),
        ]
        ax.legend(handles=legend_patches, loc='lower right', fontsize=8)

        fig.tight_layout()
        return fig

    def plot_rule_activation_over_time(
        self,
        rules: List[str],
        activations: Union[torch.Tensor, np.ndarray],
        top_k: int = 10,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Plot rule activations over time as line chart.

        Args:
            rules: List of rule name strings.
            activations: 2D tensor (timesteps, num_rules).
            top_k: Show only the top-K most variable rules.
            title: Custom title.

        Returns:
            matplotlib Figure.
        """
        data = self._to_numpy(activations)
        assert data.ndim == 2, f"Expected 2D (time, rules), got {data.ndim}D"
        n_time, n_rules = data.shape

        # Select top-K by variance
        variances = np.var(data, axis=0)
        top_indices = np.argsort(variances)[-top_k:][::-1]

        fig, ax = plt.subplots(figsize=self.config.figsize)
        cmap = plt.cm.tab10

        for rank, idx in enumerate(top_indices):
            color = cmap(rank / max(len(top_indices), 1))
            label = rules[idx] if idx < len(rules) else f'Rule {idx}'
            ax.plot(range(n_time), data[:, idx], '-', label=label,
                    color=color, linewidth=1.5, alpha=0.85)

        ax.set_xlabel('Time Step')
        ax.set_ylabel('Activation')
        ax.set_title(title or f'Rule Activations Over Time (top {len(top_indices)})')
        ax.set_xlim(0, n_time - 1)
        ax.set_ylim(0, 1.05)
        ax.legend(loc='best', fontsize=7, ncol=2)
        fig.tight_layout()
        return fig

    def plot_rule_coactivation(
        self,
        activations: Union[torch.Tensor, np.ndarray],
        rules: Optional[List[str]] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Plot rule co-activation correlation matrix.

        Args:
            activations: 2D tensor (samples, num_rules) of activation values.
            rules: Optional rule names.
            title: Custom title.

        Returns:
            matplotlib Figure.
        """
        data = self._to_numpy(activations)
        assert data.ndim == 2, f"Expected 2D, got {data.ndim}D"
        n_samples, n_rules = data.shape

        if rules is None:
            rules = [f'Rule {i}' for i in range(n_rules)]

        # Compute correlation matrix
        # Handle constant columns gracefully
        std = np.std(data, axis=0)
        valid = std > 1e-10
        corr = np.zeros((n_rules, n_rules))
        if np.any(valid):
            normed = np.zeros_like(data)
            normed[:, valid] = (data[:, valid] - np.mean(data[:, valid], axis=0)) / std[valid]
            corr_valid = normed.T @ normed / max(n_samples - 1, 1)
            corr = corr_valid
            np.fill_diagonal(corr, 1.0)

        fig, ax = plt.subplots(figsize=(max(6, n_rules * 0.8),
                                        max(5, n_rules * 0.7)))
        im = ax.imshow(corr, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

        ax.set_xticks(range(n_rules))
        ax.set_xticklabels(rules[:n_rules], rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(n_rules))
        ax.set_yticklabels(rules[:n_rules], fontsize=8)
        ax.set_title(title or 'Rule Co-Activation Correlation')

        # Annotate cells for small matrices
        if n_rules <= 12:
            for i in range(n_rules):
                for j in range(n_rules):
                    color = 'white' if abs(corr[i, j]) > 0.5 else 'black'
                    ax.text(j, i, f'{corr[i, j]:.2f}', ha='center',
                            va='center', color=color, fontsize=8)

        fig.colorbar(im, ax=ax, label='Correlation')
        fig.tight_layout()
        return fig

    def plot_trace_tree(
        self,
        trace: List[Dict],
        threshold: float = 0.8,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Render System 2 trace as a vertical tree.

        Args:
            trace: List of dicts per step with 'confidence' and
                   optionally 'residual_norm'.
            threshold: Convergence threshold.
            title: Custom title.

        Returns:
            matplotlib Figure.
        """
        n_steps = len(trace)
        confidences = [s.get('confidence', 0.0) for s in trace]
        residuals = [s.get('residual_norm', 0.0) for s in trace]

        fig, ax = plt.subplots(figsize=(6, max(4, n_steps * 1.0)))

        x_center = 0.5
        y_positions = np.linspace(0.9, 0.1, n_steps + 1)
        node_radius = 0.03

        # Root node
        root_color = CONFIDENCE_CMAP(confidences[0] if confidences else 0)
        root_y = y_positions[0]
        root_circle = plt.Circle((x_center, root_y), node_radius,
                                  color=root_color, ec='black', lw=1.5,
                                  transform=ax.transAxes, zorder=5)
        ax.add_patch(root_circle)
        ax.text(x_center + 0.06, root_y,
                f'Root (conf={confidences[0]:.2f})' if confidences else 'Root',
                transform=ax.transAxes, fontsize=9, va='center')

        # Steps
        converged = False
        for i in range(n_steps):
            y = y_positions[i + 1]
            prev_y = y_positions[i]
            conf = confidences[i]
            color = CONFIDENCE_CMAP(conf)

            # Edge
            ax.plot([x_center, x_center], [prev_y - node_radius, y + node_radius],
                    'k-', linewidth=1.5, transform=ax.transAxes, zorder=2)

            # Residual annotation on edge
            if residuals[i] > 0:
                mid_y = (prev_y + y) / 2
                ax.text(x_center - 0.08, mid_y, f'd={residuals[i]:.2f}',
                        transform=ax.transAxes, fontsize=7, color='gray',
                        va='center', ha='right')

            # Node
            ec = CONVERGED_COLOR if conf >= threshold else 'black'
            lw = 2.5 if conf >= threshold else 1.0
            circle = plt.Circle((x_center, y), node_radius,
                                color=color, ec=ec, lw=lw,
                                transform=ax.transAxes, zorder=5)
            ax.add_patch(circle)

            label = f'Step {i} (conf={conf:.2f})'
            if conf >= threshold and not converged:
                label += ' CONVERGED'
                converged = True
            ax.text(x_center + 0.06, y, label,
                    transform=ax.transAxes, fontsize=9, va='center')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()
        ax.set_title(title or 'System 2 Trace Tree', fontsize=12, pad=15)
        fig.tight_layout()
        return fig

    def plot_routing_over_time(
        self,
        confidences: Union[torch.Tensor, np.ndarray],
        threshold: float = 0.8,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Plot confidence as a time series with System 1/2 background.

        Args:
            confidences: 1D tensor of confidence values over time.
            threshold: Routing threshold.
            title: Custom title.

        Returns:
            matplotlib Figure.
        """
        data = self._to_numpy(confidences).ravel()
        n = len(data)
        steps = np.arange(n)

        fig, ax = plt.subplots(figsize=self.config.figsize)

        # Background zones
        ax.axhspan(threshold, 1.05, alpha=0.08, color=SYSTEM1_COLOR,
                   label='System 1 zone')
        ax.axhspan(0, threshold, alpha=0.08, color=SYSTEM2_COLOR,
                   label='System 2 zone')

        ax.axhline(y=threshold, color=THRESHOLD_COLOR, linestyle='--',
                   linewidth=1.5, alpha=0.8, label=f'Threshold ({threshold})')

        # Plot confidence
        for i in range(n - 1):
            color = SYSTEM1_COLOR if data[i] >= threshold else SYSTEM2_COLOR
            ax.plot([steps[i], steps[i + 1]], [data[i], data[i + 1]],
                    '-', color=color, linewidth=2)

        # Points
        for i in range(n):
            color = SYSTEM1_COLOR if data[i] >= threshold else SYSTEM2_COLOR
            ax.scatter(steps[i], data[i], color=color, s=40, zorder=4,
                       edgecolors='black', linewidths=0.5)

        ax.set_xlabel('Time Step')
        ax.set_ylabel('Confidence')
        ax.set_title(title or 'Routing Over Time')
        ax.set_xlim(-0.3, n - 0.7)
        ax.set_ylim(0, 1.05)
        ax.legend(loc='lower right', fontsize=9)
        fig.tight_layout()
        return fig

    def save_figure(self, fig: plt.Figure, filename: str, close: bool = True) -> str:
        """Save figure to configured directory."""
        os.makedirs(self.config.save_dir, exist_ok=True)
        if not filename.endswith(f'.{self.config.output_format}'):
            filename = f'{filename}.{self.config.output_format}'
        path = os.path.join(self.config.save_dir, filename)
        fig.savefig(path, dpi=self.config.dpi, bbox_inches='tight')
        if close:
            plt.close(fig)
        return path


# ---------------------------------------------------------------------------
# Self-Tests
# ---------------------------------------------------------------------------

def _run_self_tests():
    """Run self-tests. Returns (passed, failed)."""
    import traceback

    results = []

    def test(name, func):
        try:
            func()
            results.append((name, True, ''))
        except Exception as e:
            results.append((name, False, f'{e}\n{traceback.format_exc()}'))

    config = VizConfig(figsize=(8, 5), dpi=72)
    viz = ReasoningTraceVisualizer(config)

    def assert_fig(fig, label=''):
        assert fig is not None, f"Figure is None ({label})"
        assert len(fig.axes) > 0, f"No axes ({label})"
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            fig.savefig(f.name, dpi=72)
            plt.close(fig)
            size = os.path.getsize(f.name)
            os.unlink(f.name)
        assert size > 500, f"PNG too small: {size} ({label})"

    # ---- 1: Basic routing decision System 2 ----
    def t01():
        fig = viz.plot_routing_decision(0.65, threshold=0.8)
        assert_fig(fig, 'routing sys2')
    test('01_routing_sys2', t01)

    # ---- 2: Routing decision System 1 ----
    def t02():
        fig = viz.plot_routing_decision(0.92, threshold=0.8)
        assert_fig(fig, 'routing sys1')
    test('02_routing_sys1', t02)

    # ---- 3: Routing axes labels ----
    def t03():
        fig = viz.plot_routing_decision(0.5, threshold=0.8)
        ax = fig.axes[0]
        xl = ax.get_xlabel().lower()
        assert 'confidence' in xl, f"X label: {ax.get_xlabel()}"
        plt.close(fig)
    test('03_routing_axes', t03)

    # ---- 4: Routing title contains System ----
    def t04():
        fig = viz.plot_routing_decision(0.5, threshold=0.8)
        ax = fig.axes[0]
        assert 'System' in ax.get_title(), f"Title: {ax.get_title()}"
        plt.close(fig)
    test('04_routing_title', t04)

    # ---- 5: Routing zero confidence ----
    def t05():
        fig = viz.plot_routing_decision(0.0, threshold=0.8)
        assert_fig(fig, 'routing zero')
    test('05_routing_zero', t05)

    # ---- 6: Routing confidence = 1.0 ----
    def t06():
        fig = viz.plot_routing_decision(1.0, threshold=0.8)
        ax = fig.axes[0]
        assert 'System 1' in ax.get_title()
        plt.close(fig)
    test('06_routing_max', t06)

    # ---- 7: Routing custom title ----
    def t07():
        fig = viz.plot_routing_decision(0.5, title='My Routing')
        ax = fig.axes[0]
        assert 'My Routing' in ax.get_title()
        plt.close(fig)
    test('07_routing_custom_title', t07)

    # ---- 8: Routing distribution ----
    def t08():
        confs = torch.rand(100)
        fig = viz.plot_routing_distribution(confs, threshold=0.5)
        assert_fig(fig, 'routing distribution')
    test('08_routing_distribution', t08)

    # ---- 9: System 2 steps basic ----
    def t09():
        trace = [
            {'confidence': 0.4},
            {'confidence': 0.55},
            {'confidence': 0.7},
            {'confidence': 0.82},
            {'confidence': 0.88},
        ]
        fig = viz.plot_system2_steps(trace, threshold=0.8)
        assert_fig(fig, 'sys2 basic')
    test('09_sys2_basic', t09)

    # ---- 10: System 2 with residuals ----
    def t10():
        trace = [
            {'confidence': 0.4, 'residual_norm': 2.1},
            {'confidence': 0.6, 'residual_norm': 1.3},
            {'confidence': 0.75, 'residual_norm': 0.8},
            {'confidence': 0.85, 'residual_norm': 0.3},
        ]
        fig = viz.plot_system2_steps(trace, threshold=0.8)
        assert_fig(fig, 'sys2 residuals')
    test('10_sys2_residuals', t10)

    # ---- 11: System 2 with entropy ----
    def t11():
        trace = [
            {'confidence': 0.5, 'attention_entropy': 2.0},
            {'confidence': 0.7, 'attention_entropy': 1.5},
            {'confidence': 0.85, 'attention_entropy': 1.0},
        ]
        fig = viz.plot_system2_steps(trace, threshold=0.8)
        assert_fig(fig, 'sys2 entropy')
    test('11_sys2_entropy', t11)

    # ---- 12: System 2 non-convergence ----
    def t12():
        trace = [
            {'confidence': 0.3},
            {'confidence': 0.35},
            {'confidence': 0.4},
        ]
        fig = viz.plot_system2_steps(trace, threshold=0.8)
        assert_fig(fig, 'sys2 no converge')
    test('12_sys2_no_converge', t12)

    # ---- 13: System 2 axes labels ----
    def t13():
        trace = [{'confidence': 0.5}, {'confidence': 0.7}]
        fig = viz.plot_system2_steps(trace)
        ax = fig.axes[0]
        xl = ax.get_xlabel().lower()
        yl = ax.get_ylabel().lower()
        assert 'step' in xl or 'system' in xl, f"X: {ax.get_xlabel()}"
        assert 'confidence' in yl, f"Y: {ax.get_ylabel()}"
        plt.close(fig)
    test('13_sys2_axes', t13)

    # ---- 14: Rule activation basic ----
    def t14():
        rules = ['is_animal AND has_fur', 'is_moving AND fast',
                 'is_large', 'is_dangerous', 'is_domestic']
        acts = torch.tensor([0.92, 0.78, 0.65, 0.31, 0.12])
        fig = viz.plot_rule_activation(rules, acts)
        assert_fig(fig, 'rule basic')
    test('14_rule_basic', t14)

    # ---- 15: Rule activation axes ----
    def t15():
        rules = ['Rule A', 'Rule B']
        acts = torch.tensor([0.8, 0.3])
        fig = viz.plot_rule_activation(rules, acts)
        ax = fig.axes[0]
        xl = ax.get_xlabel().lower()
        assert 'activation' in xl, f"X: {ax.get_xlabel()}"
        plt.close(fig)
    test('15_rule_axes', t15)

    # ---- 16: Rule activation top-k ----
    def t16():
        rules = [f'Rule {i}' for i in range(20)]
        acts = torch.rand(20)
        fig = viz.plot_rule_activation(rules, acts, top_k=5)
        assert_fig(fig, 'rule top-k')
    test('16_rule_topk', t16)

    # ---- 17: Rule activation zero activations ----
    def t17():
        rules = ['A', 'B', 'C']
        acts = torch.zeros(3)
        fig = viz.plot_rule_activation(rules, acts)
        assert_fig(fig, 'rule zeros')
    test('17_rule_zeros', t17)

    # ---- 18: Rule activation single rule ----
    def t18():
        rules = ['OnlyRule']
        acts = torch.tensor([0.95])
        fig = viz.plot_rule_activation(rules, acts)
        assert_fig(fig, 'rule single')
    test('18_rule_single', t18)

    # ---- 19: Confidence waterfall ----
    def t19():
        stages = ['Encoder', 'WS', 'Pre-R', 'S2-1', 'S2-2', 'Output']
        confs = [0.4, 0.5, 0.65, 0.75, 0.85, 0.92]
        fig = viz.plot_confidence_waterfall(stages, confs, threshold=0.8)
        assert_fig(fig, 'waterfall')
    test('19_waterfall', t19)

    # ---- 20: Trace tree ----
    def t20():
        trace = [
            {'confidence': 0.4, 'residual_norm': 2.1},
            {'confidence': 0.55, 'residual_norm': 1.5},
            {'confidence': 0.72, 'residual_norm': 0.9},
            {'confidence': 0.85, 'residual_norm': 0.3},
        ]
        fig = viz.plot_trace_tree(trace, threshold=0.8)
        assert_fig(fig, 'trace tree')
    test('20_trace_tree', t20)

    # ---- 21: Rule co-activation ----
    def t21():
        acts = torch.rand(50, 6)
        rules = [f'Rule {i}' for i in range(6)]
        fig = viz.plot_rule_coactivation(acts, rules)
        assert_fig(fig, 'co-activation')
    test('21_coactivation', t21)

    # ---- 22: Rule activation over time ----
    def t22():
        rules = [f'Rule {i}' for i in range(15)]
        acts = torch.rand(30, 15)
        fig = viz.plot_rule_activation_over_time(rules, acts, top_k=5)
        assert_fig(fig, 'rule over time')
    test('22_rule_over_time', t22)

    # ---- 23: Routing over time ----
    def t23():
        confs = torch.rand(20) * 0.5 + 0.4  # range [0.4, 0.9]
        fig = viz.plot_routing_over_time(confs, threshold=0.7)
        assert_fig(fig, 'routing over time')
    test('23_routing_over_time', t23)

    # ---- 24: Deterministic rendering ----
    def t24():
        sizes = []
        for _ in range(2):
            fig = viz.plot_routing_decision(0.65, threshold=0.8)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                fig.savefig(f.name, dpi=72)
                plt.close(fig)
                sizes.append(os.path.getsize(f.name))
                os.unlink(f.name)
        assert sizes[0] == sizes[1], f"Non-deterministic: {sizes}"
    test('24_deterministic', t24)

    # ---- 25: PNG header valid ----
    def t25():
        fig = viz.plot_routing_decision(0.5)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            fig.savefig(f.name, dpi=72)
            plt.close(fig)
            with open(f.name, 'rb') as fp:
                header = fp.read(8)
            os.unlink(f.name)
        assert header[:4] == b'\x89PNG', "Invalid PNG"
    test('25_png_header', t25)

    # ---- 26: Save figure ----
    def t26():
        fig = viz.plot_routing_decision(0.5)
        with tempfile.TemporaryDirectory() as td:
            viz_local = ReasoningTraceVisualizer(VizConfig(save_dir=td, dpi=72))
            path = viz_local.save_figure(fig, 'test_routing')
            assert os.path.exists(path)
            assert os.path.getsize(path) > 500
    test('26_save_figure', t26)

    # ---- 27: Numpy input rule activation ----
    def t27():
        rules = ['A', 'B', 'C']
        acts = np.array([0.9, 0.5, 0.1])
        fig = viz.plot_rule_activation(rules, acts)
        assert_fig(fig, 'numpy rule')
    test('27_numpy_rule', t27)

    # Print
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{'='*60}")
    print(f"ReasoningTraceVisualizer Self-Tests: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    for name, ok, err in results:
        status = 'PASS' if ok else 'FAIL'
        print(f"  [{status}] {name}")
        if err:
            for line in err.strip().split('\n')[:3]:
                print(f"         {line}")
    print()
    return passed, failed


if __name__ == '__main__':
    passed, failed = _run_self_tests()
    exit(0 if failed == 0 else 1)
