"""
Visualization Config & Training Dashboard Template for brain_ai.

Provides VizConfig dataclass and TrainingDashboard class for aggregate
training curve visualization, neuromodulator level tracking, phase
transition display, and HTML report generation.

Usage:
    from viz_config_template import VizConfig, TrainingDashboard
    config = VizConfig()
    dashboard = TrainingDashboard(log_dir='logs/', config=config)
    fig = dashboard.plot_loss_curves(metrics)
    report = dashboard.generate_report(output_dir='reports/')

Requires: matplotlib, numpy, torch
Backend: Agg (headless, no display required)
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import os
import io
import base64
import tempfile
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Union


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class VizConfig:
    """Master visualization configuration for all brain_ai viz modules.

    Attributes:
        backend: Rendering backend ('matplotlib' or 'plotly').
        output_format: Default output format for saved figures.
        dpi: Dots per inch for raster output.
        figsize: Default figure size (width, height) in inches.
        colormap: Default matplotlib colormap name.
        dark_mode: Whether to use dark mode styling.
        save_dir: Default directory for saved figures.
        max_neurons: Maximum neurons to display (subsample if more).
        max_timesteps: Maximum timesteps to display.
        tsne_perplexity: Default t-SNE perplexity.
        umap_n_neighbors: Default UMAP n_neighbors.
        max_points: Maximum points for embedding projections.
    """
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

    def to_dict(self) -> Dict:
        """Convert config to dict."""
        return {
            'backend': self.backend,
            'output_format': self.output_format,
            'dpi': self.dpi,
            'figsize': self.figsize,
            'colormap': self.colormap,
            'dark_mode': self.dark_mode,
            'save_dir': self.save_dir,
            'max_neurons': self.max_neurons,
            'max_timesteps': self.max_timesteps,
            'tsne_perplexity': self.tsne_perplexity,
            'umap_n_neighbors': self.umap_n_neighbors,
            'max_points': self.max_points,
        }

    @staticmethod
    def from_dict(d: Dict) -> 'VizConfig':
        """Create VizConfig from dict."""
        return VizConfig(**{k: v for k, v in d.items()
                           if k in VizConfig.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Color Constants
# ---------------------------------------------------------------------------

MODALITY_COLORS = {
    'vision': '#1f77b4',
    'text': '#2ca02c',
    'audio': '#ff7f0e',
    'sensor': '#9467bd',
    'engram': '#8c564b',
}

NEUROMODULATOR_COLORS = {
    'DA': '#d62728',
    'ACh': '#2ca02c',
    'NE': '#1f77b4',
    '5-HT': '#bcbd22',
}

SYSTEM_COLORS = {
    'System1': '#00bcd4',
    'System2': '#e91e63',
}

PHASE_COLORS = ['#1f77b4', '#2196f3', '#4caf50', '#ff9800',
                '#f44336', '#9c27b0', '#e91e63']

METRIC_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                 '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                 '#bcbd22', '#17becf']


# ---------------------------------------------------------------------------
# TrainingDashboard
# ---------------------------------------------------------------------------

class TrainingDashboard:
    """Aggregate training curves and module-specific diagnostics.

    Provides methods for plotting loss curves, neuromodulator levels,
    phase transitions, and generating comprehensive HTML reports.

    Args:
        log_dir: Directory containing training logs.
        config: VizConfig instance controlling appearance.
    """

    def __init__(self, log_dir: str = "logs/", config: Optional[VizConfig] = None):
        self.log_dir = log_dir
        self.config = config or VizConfig()

    def _to_numpy(self, tensor: Union[torch.Tensor, np.ndarray, List]) -> np.ndarray:
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        if isinstance(tensor, list):
            return np.array(tensor, dtype=float)
        return np.asarray(tensor)

    def _fig_to_base64(self, fig: plt.Figure) -> str:
        """Convert figure to base64-encoded PNG string."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=self.config.dpi, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    # -- Public API ----------------------------------------------------------

    def plot_loss_curves(
        self,
        metrics: Dict[str, Union[List[float], np.ndarray, torch.Tensor]],
        smooth_window: int = 1,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Plot training loss/metric curves.

        Args:
            metrics: Dict mapping metric names to value lists/arrays.
                     e.g., {'loss': [1.0, 0.8, ...], 'accuracy': [0.2, 0.4, ...]}.
            smooth_window: Smoothing window size (1 = no smoothing).
            title: Custom title.

        Returns:
            matplotlib Figure.
        """
        n_metrics = len(metrics)
        has_loss = any('loss' in k.lower() for k in metrics)
        has_acc = any('acc' in k.lower() or 'accuracy' in k.lower() for k in metrics)

        # Split into loss and non-loss for dual y-axis if both exist
        use_dual = has_loss and has_acc and n_metrics > 1

        fig, ax1 = plt.subplots(figsize=self.config.figsize)
        ax2 = ax1.twinx() if use_dual else None

        color_idx = 0
        for name, values in metrics.items():
            data = self._to_numpy(values).ravel()

            # Smooth
            if smooth_window > 1 and len(data) >= smooth_window:
                kernel = np.ones(smooth_window) / smooth_window
                smoothed = np.convolve(data, kernel, mode='valid')
                x = np.arange(len(smoothed))
            else:
                smoothed = data
                x = np.arange(len(smoothed))

            color = METRIC_COLORS[color_idx % len(METRIC_COLORS)]
            color_idx += 1

            is_loss = 'loss' in name.lower()
            ax = ax1 if (is_loss or not use_dual) else ax2

            ax.plot(x, smoothed, '-', color=color, linewidth=1.8,
                    label=name, alpha=0.9)

            # Show raw data as faint background if smoothed
            if smooth_window > 1 and len(data) >= smooth_window:
                ax.plot(np.arange(len(data)), data, '-', color=color,
                        linewidth=0.4, alpha=0.25)

        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss' if has_loss else 'Metric Value')
        if use_dual and ax2:
            ax2.set_ylabel('Accuracy / Other Metrics')

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        if use_dual and ax2:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2,
                       loc='best', fontsize=9)
        else:
            ax1.legend(loc='best', fontsize=9)

        ax1.set_title(title or 'Training Curves')
        ax1.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_neuromodulator_levels(
        self,
        levels: Dict[str, Union[List[float], np.ndarray, torch.Tensor]],
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Plot neuromodulator level trajectories over training.

        Args:
            levels: Dict mapping neuromodulator names to value lists.
                    e.g., {'DA': [0.5, 0.6, ...], 'ACh': [0.3, 0.4, ...]}.
            title: Custom title.

        Returns:
            matplotlib Figure.
        """
        n_neuro = len(levels)

        fig, axes = plt.subplots(1, 2, figsize=(self.config.figsize[0] + 4,
                                                  self.config.figsize[1]),
                                  gridspec_kw={'width_ratios': [3, 1]})

        # Panel 1: Time series
        ax = axes[0]
        for name, values in levels.items():
            data = self._to_numpy(values).ravel()
            color = NEUROMODULATOR_COLORS.get(name, '#808080')
            ax.plot(range(len(data)), data, '-', color=color,
                    linewidth=2, label=name, alpha=0.9)

        ax.set_xlabel('Training Step')
        ax.set_ylabel('Level')
        ax.set_title('Neuromodulator Trajectories')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Panel 2: Final state bar chart
        ax2 = axes[1]
        names = list(levels.keys())
        final_vals = []
        colors = []
        for name in names:
            data = self._to_numpy(levels[name]).ravel()
            final_vals.append(data[-1] if len(data) > 0 else 0.0)
            colors.append(NEUROMODULATOR_COLORS.get(name, '#808080'))

        ax2.barh(range(len(names)), final_vals, color=colors,
                 edgecolor='black', linewidth=0.8, height=0.6)
        ax2.set_yticks(range(len(names)))
        ax2.set_yticklabels(names)
        ax2.set_xlabel('Final Level')
        ax2.set_title('Current State')
        ax2.set_xlim(0, max(max(final_vals) * 1.15, 0.1) if final_vals else 1.0)

        for i, val in enumerate(final_vals):
            ax2.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9)

        fig.suptitle(title or 'Neuromodulator Levels', fontsize=13)
        fig.tight_layout()
        return fig

    def plot_phase_transitions(
        self,
        metrics: Dict[str, Union[List[float], np.ndarray, torch.Tensor]],
        phase_boundaries: List[int],
        phase_names: Optional[List[str]] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Plot metrics with phase boundaries highlighted.

        Args:
            metrics: Dict of metric name -> values.
            phase_boundaries: Step indices where phases change.
            phase_names: Optional names for each phase.
            title: Custom title.

        Returns:
            matplotlib Figure.
        """
        if phase_names is None:
            n_phases = len(phase_boundaries) + 1
            phase_names = [f'Phase {i+1}' for i in range(n_phases)]

        fig, ax = plt.subplots(figsize=self.config.figsize)

        # Plot metrics
        color_idx = 0
        max_len = 0
        for name, values in metrics.items():
            data = self._to_numpy(values).ravel()
            max_len = max(max_len, len(data))
            color = METRIC_COLORS[color_idx % len(METRIC_COLORS)]
            ax.plot(range(len(data)), data, '-', color=color, linewidth=1.8,
                    label=name, alpha=0.9)
            color_idx += 1

        # Phase boundaries and background coloring
        all_bounds = [0] + sorted(phase_boundaries) + [max_len]
        for i in range(len(all_bounds) - 1):
            start, end = all_bounds[i], all_bounds[i + 1]
            color = PHASE_COLORS[i % len(PHASE_COLORS)]
            ax.axvspan(start, end, alpha=0.06, color=color)
            mid = (start + end) / 2
            pname = phase_names[i] if i < len(phase_names) else f'P{i+1}'
            ax.text(mid, ax.get_ylim()[1] * 0.97, pname,
                    ha='center', va='top', fontsize=9, fontweight='bold',
                    color=color, alpha=0.8)

        for boundary in phase_boundaries:
            ax.axvline(x=boundary, color='gray', linestyle='--',
                       linewidth=1.5, alpha=0.6)

        ax.set_xlabel('Training Step')
        ax.set_ylabel('Metric Value')
        ax.set_title(title or 'Training Progress with Phase Transitions')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        return fig

    def plot_learning_rate_schedule(
        self,
        lr_values: Union[List[float], np.ndarray, torch.Tensor],
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Plot learning rate schedule over training.

        Args:
            lr_values: Learning rate at each step.
            title: Custom title.

        Returns:
            matplotlib Figure.
        """
        data = self._to_numpy(lr_values).ravel()

        fig, ax = plt.subplots(figsize=self.config.figsize)
        ax.plot(range(len(data)), data, '-', color='#d62728', linewidth=1.8)
        ax.fill_between(range(len(data)), data, alpha=0.15, color='#d62728')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Learning Rate')
        ax.set_title(title or 'Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_module_loss_breakdown(
        self,
        module_losses: Dict[str, Union[List[float], np.ndarray]],
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Plot stacked area chart of per-module loss contributions.

        Args:
            module_losses: Dict mapping module names to loss value lists.
            title: Custom title.

        Returns:
            matplotlib Figure.
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)

        names = list(module_losses.keys())
        arrays = [self._to_numpy(module_losses[n]).ravel() for n in names]
        min_len = min(len(a) for a in arrays) if arrays else 0
        arrays = [a[:min_len] for a in arrays]
        stacked = np.array(arrays)
        x = np.arange(min_len)

        colors = [METRIC_COLORS[i % len(METRIC_COLORS)] for i in range(len(names))]
        ax.stackplot(x, stacked, labels=names, colors=colors, alpha=0.75)

        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_title(title or 'Module Loss Breakdown')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        return fig

    def plot_gradient_norms(
        self,
        grad_norms: Dict[str, Union[List[float], np.ndarray]],
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Plot gradient norm trajectories per module.

        Args:
            grad_norms: Dict mapping module names to gradient norm lists.
            title: Custom title.

        Returns:
            matplotlib Figure.
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)

        color_idx = 0
        for name, values in grad_norms.items():
            data = self._to_numpy(values).ravel()
            color = METRIC_COLORS[color_idx % len(METRIC_COLORS)]
            ax.plot(range(len(data)), data, '-', color=color, linewidth=1.5,
                    label=name, alpha=0.85)
            color_idx += 1

        ax.set_xlabel('Training Step')
        ax.set_ylabel('Gradient L2 Norm')
        ax.set_title(title or 'Gradient Norms by Module')
        ax.legend(loc='best', fontsize=8, ncol=2)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def generate_report(
        self,
        output_dir: str,
        metrics: Optional[Dict[str, Union[List[float], np.ndarray]]] = None,
        neuromodulator_levels: Optional[Dict[str, Union[List[float], np.ndarray]]] = None,
        phase_boundaries: Optional[List[int]] = None,
        phase_names: Optional[List[str]] = None,
        title: str = "Brain AI Training Report",
    ) -> str:
        """Generate a comprehensive HTML report with embedded plots.

        Args:
            output_dir: Directory to write the report HTML.
            metrics: Training metrics dict (loss, accuracy, etc.).
            neuromodulator_levels: Neuromodulator level trajectories.
            phase_boundaries: Phase transition step indices.
            phase_names: Phase names.
            title: Report title.

        Returns:
            Path to the generated HTML file.
        """
        os.makedirs(output_dir, exist_ok=True)

        sections = []
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

        # Section 1: Loss curves
        if metrics:
            fig = self.plot_loss_curves(metrics)
            img_b64 = self._fig_to_base64(fig)
            metric_summary = ', '.join([
                f'{k}: {self._to_numpy(v).ravel()[-1]:.4f}'
                for k, v in metrics.items()
                if len(self._to_numpy(v).ravel()) > 0
            ])
            sections.append(f"""
        <div class="section">
            <h2>Training Curves</h2>
            <p>Final values: {metric_summary}</p>
            <img src="data:image/png;base64,{img_b64}" alt="Loss Curves" />
        </div>""")

        # Section 2: Neuromodulator levels
        if neuromodulator_levels:
            fig = self.plot_neuromodulator_levels(neuromodulator_levels)
            img_b64 = self._fig_to_base64(fig)
            neuro_summary = ', '.join([
                f'{k}: {self._to_numpy(v).ravel()[-1]:.3f}'
                for k, v in neuromodulator_levels.items()
                if len(self._to_numpy(v).ravel()) > 0
            ])
            sections.append(f"""
        <div class="section">
            <h2>Neuromodulator Levels</h2>
            <p>Final levels: {neuro_summary}</p>
            <img src="data:image/png;base64,{img_b64}" alt="Neuromodulator Levels" />
        </div>""")

        # Section 3: Phase transitions
        if metrics and phase_boundaries:
            fig = self.plot_phase_transitions(metrics, phase_boundaries,
                                              phase_names=phase_names)
            img_b64 = self._fig_to_base64(fig)
            n_phases = len(phase_boundaries) + 1
            sections.append(f"""
        <div class="section">
            <h2>Phase Transitions</h2>
            <p>{n_phases} training phases, boundaries at steps: {phase_boundaries}</p>
            <img src="data:image/png;base64,{img_b64}" alt="Phase Transitions" />
        </div>""")

        # Section 4: Configuration summary
        config_rows = '\n'.join([
            f'            <tr><td>{k}</td><td>{v}</td></tr>'
            for k, v in self.config.to_dict().items()
        ])
        sections.append(f"""
        <div class="section">
            <h2>Configuration</h2>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
{config_rows}
            </table>
        </div>""")

        sections_html = '\n'.join(sections)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: {'#1a1a2e' if self.config.dark_mode else '#fafafa'};
            color: {'#e0e0e0' if self.config.dark_mode else '#333'};
        }}
        h1 {{
            border-bottom: 2px solid {'#444' if self.config.dark_mode else '#ddd'};
            padding-bottom: 10px;
        }}
        .section {{
            background: {'#16213e' if self.config.dark_mode else 'white'};
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin-top: 10px;
        }}
        th, td {{
            border: 1px solid {'#444' if self.config.dark_mode else '#ddd'};
            padding: 8px 12px;
            text-align: left;
        }}
        th {{
            background: {'#1a1a3e' if self.config.dark_mode else '#f5f5f5'};
        }}
        .timestamp {{
            color: {'#888' if self.config.dark_mode else '#999'};
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p class="timestamp">Generated: {timestamp}</p>
{sections_html}
    <p class="timestamp">End of report.</p>
</body>
</html>"""

        report_path = os.path.join(output_dir, 'training_report.html')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)

        return report_path

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

    def assert_fig(fig, label=''):
        assert fig is not None, f"Figure is None ({label})"
        assert len(fig.axes) > 0, f"No axes ({label})"
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            fig.savefig(f.name, dpi=72)
            plt.close(fig)
            size = os.path.getsize(f.name)
            os.unlink(f.name)
        assert size > 500, f"PNG too small: {size} ({label})"

    # ---- 1: VizConfig defaults ----
    def t01():
        c = VizConfig()
        assert c.backend == 'matplotlib'
        assert c.dpi == 150
        assert c.figsize == (12, 8)
        assert c.max_neurons == 100
    test('01_config_defaults', t01)

    # ---- 2: VizConfig to_dict ----
    def t02():
        c = VizConfig(dpi=72)
        d = c.to_dict()
        assert d['dpi'] == 72
        assert 'backend' in d
    test('02_config_to_dict', t02)

    # ---- 3: VizConfig from_dict ----
    def t03():
        d = {'dpi': 300, 'dark_mode': True, 'extra_key': 'ignored'}
        c = VizConfig.from_dict(d)
        assert c.dpi == 300
        assert c.dark_mode is True
    test('03_config_from_dict', t03)

    # ---- 4: VizConfig round-trip ----
    def t04():
        c1 = VizConfig(dpi=96, colormap='plasma')
        c2 = VizConfig.from_dict(c1.to_dict())
        assert c2.dpi == c1.dpi
        assert c2.colormap == c1.colormap
    test('04_config_roundtrip', t04)

    # ---- 5: Basic loss curves ----
    def t05():
        dashboard = TrainingDashboard(config=config)
        metrics = {'loss': [1.0, 0.8, 0.6, 0.4, 0.3]}
        fig = dashboard.plot_loss_curves(metrics)
        assert_fig(fig, 'loss basic')
    test('05_loss_basic', t05)

    # ---- 6: Loss curves multiple metrics ----
    def t06():
        dashboard = TrainingDashboard(config=config)
        metrics = {
            'loss': [1.0, 0.8, 0.6, 0.4, 0.3],
            'accuracy': [0.2, 0.4, 0.6, 0.7, 0.8],
        }
        fig = dashboard.plot_loss_curves(metrics)
        assert_fig(fig, 'loss multi')
    test('06_loss_multi', t06)

    # ---- 7: Loss curves axes ----
    def t07():
        dashboard = TrainingDashboard(config=config)
        metrics = {'loss': [1.0, 0.5, 0.3]}
        fig = dashboard.plot_loss_curves(metrics)
        ax = fig.axes[0]
        assert 'Step' in ax.get_xlabel(), f"X: {ax.get_xlabel()}"
        assert 'Loss' in ax.get_ylabel() or 'Metric' in ax.get_ylabel(), \
            f"Y: {ax.get_ylabel()}"
        plt.close(fig)
    test('07_loss_axes', t07)

    # ---- 8: Loss curves with smoothing ----
    def t08():
        dashboard = TrainingDashboard(config=config)
        metrics = {'loss': np.random.rand(50).tolist()}
        fig = dashboard.plot_loss_curves(metrics, smooth_window=5)
        assert_fig(fig, 'loss smooth')
    test('08_loss_smooth', t08)

    # ---- 9: Neuromodulator levels basic ----
    def t09():
        dashboard = TrainingDashboard(config=config)
        levels = {
            'DA': [0.5, 0.6, 0.7, 0.8, 0.9],
            'ACh': [0.3, 0.4, 0.5, 0.6, 0.7],
            'NE': [0.8, 0.7, 0.6, 0.5, 0.4],
            '5-HT': [0.4, 0.4, 0.5, 0.5, 0.6],
        }
        fig = dashboard.plot_neuromodulator_levels(levels)
        assert_fig(fig, 'neuro basic')
    test('09_neuro_basic', t09)

    # ---- 10: Neuromodulator levels axes ----
    def t10():
        dashboard = TrainingDashboard(config=config)
        levels = {'DA': [0.5, 0.6], 'ACh': [0.3, 0.4]}
        fig = dashboard.plot_neuromodulator_levels(levels)
        ax = fig.axes[0]
        xl = ax.get_xlabel().lower()
        yl = ax.get_ylabel().lower()
        assert 'step' in xl or 'time' in xl, f"X: {ax.get_xlabel()}"
        assert 'level' in yl, f"Y: {ax.get_ylabel()}"
        plt.close(fig)
    test('10_neuro_axes', t10)

    # ---- 11: Neuromodulator 4 colors present ----
    def t11():
        dashboard = TrainingDashboard(config=config)
        levels = {
            'DA': [0.5, 0.6],
            'ACh': [0.3, 0.4],
            'NE': [0.7, 0.6],
            '5-HT': [0.4, 0.5],
        }
        fig = dashboard.plot_neuromodulator_levels(levels)
        ax = fig.axes[0]
        lines = ax.get_lines()
        assert len(lines) >= 4, f"Expected 4 lines, got {len(lines)}"
        plt.close(fig)
    test('11_neuro_colors', t11)

    # ---- 12: Phase transitions basic ----
    def t12():
        dashboard = TrainingDashboard(config=config)
        metrics = {'loss': [1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1]}
        fig = dashboard.plot_phase_transitions(metrics, [2, 5])
        assert_fig(fig, 'phase basic')
    test('12_phase_basic', t12)

    # ---- 13: Phase transitions with names ----
    def t13():
        dashboard = TrainingDashboard(config=config)
        metrics = {'loss': [1.0, 0.8, 0.6, 0.4, 0.3]}
        fig = dashboard.plot_phase_transitions(metrics, [2],
                                                phase_names=['SNN', 'HTM'])
        assert_fig(fig, 'phase names')
    test('13_phase_names', t13)

    # ---- 14: Generate report basic ----
    def t14():
        with tempfile.TemporaryDirectory() as td:
            dashboard = TrainingDashboard(config=VizConfig(dpi=72))
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
            path = dashboard.generate_report(
                output_dir=td,
                metrics=metrics,
                neuromodulator_levels=neuro,
                phase_boundaries=[2, 4],
            )
            assert os.path.exists(path), f"Report not created: {path}"
            with open(path) as f:
                content = f.read()
            assert '<html' in content.lower() or '<!doctype' in content.lower()
    test('14_report_basic', t14)

    # ---- 15: Report contains loss section ----
    def t15():
        with tempfile.TemporaryDirectory() as td:
            dashboard = TrainingDashboard(config=VizConfig(dpi=72))
            metrics = {'loss': [1.0, 0.5]}
            path = dashboard.generate_report(td, metrics=metrics)
            with open(path) as f:
                content = f.read()
            assert 'loss' in content.lower(), "No 'loss' in report"
    test('15_report_loss_section', t15)

    # ---- 16: Report contains neuromodulator section ----
    def t16():
        with tempfile.TemporaryDirectory() as td:
            dashboard = TrainingDashboard(config=VizConfig(dpi=72))
            neuro = {'DA': [0.5, 0.6], 'ACh': [0.3, 0.4]}
            path = dashboard.generate_report(td, neuromodulator_levels=neuro)
            with open(path) as f:
                content = f.read()
            assert 'neuromodulator' in content.lower() or 'DA' in content
    test('16_report_neuro_section', t16)

    # ---- 17: Report contains images ----
    def t17():
        with tempfile.TemporaryDirectory() as td:
            dashboard = TrainingDashboard(config=VizConfig(dpi=72))
            metrics = {'loss': [1.0, 0.5]}
            path = dashboard.generate_report(td, metrics=metrics)
            with open(path) as f:
                content = f.read()
            assert 'data:image/png;base64,' in content
    test('17_report_images', t17)

    # ---- 18: Report HTML file size ----
    def t18():
        with tempfile.TemporaryDirectory() as td:
            dashboard = TrainingDashboard(config=VizConfig(dpi=72))
            metrics = {'loss': [1.0, 0.5]}
            path = dashboard.generate_report(td, metrics=metrics)
            size = os.path.getsize(path)
            assert size > 500, f"Report too small: {size}"
    test('18_report_size', t18)

    # ---- 19: Learning rate schedule ----
    def t19():
        dashboard = TrainingDashboard(config=config)
        lr = [0.001 * (0.99 ** i) for i in range(100)]
        fig = dashboard.plot_learning_rate_schedule(lr)
        assert_fig(fig, 'lr schedule')
    test('19_lr_schedule', t19)

    # ---- 20: Module loss breakdown ----
    def t20():
        dashboard = TrainingDashboard(config=config)
        losses = {
            'SNN': [0.3, 0.25, 0.2, 0.15],
            'HTM': [0.2, 0.18, 0.15, 0.12],
            'Workspace': [0.15, 0.12, 0.1, 0.08],
        }
        fig = dashboard.plot_module_loss_breakdown(losses)
        assert_fig(fig, 'module loss')
    test('20_module_loss', t20)

    # ---- 21: Gradient norms ----
    def t21():
        dashboard = TrainingDashboard(config=config)
        grads = {
            'SNN': [1.0, 0.8, 0.6, 0.5],
            'HTM': [0.5, 0.4, 0.3, 0.25],
        }
        fig = dashboard.plot_gradient_norms(grads)
        assert_fig(fig, 'grad norms')
    test('21_grad_norms', t21)

    # ---- 22: Deterministic loss curves ----
    def t22():
        dashboard = TrainingDashboard(config=config)
        metrics = {'loss': [1.0, 0.5, 0.3]}
        sizes = []
        for _ in range(2):
            fig = dashboard.plot_loss_curves(metrics)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                fig.savefig(f.name, dpi=72)
                plt.close(fig)
                sizes.append(os.path.getsize(f.name))
                os.unlink(f.name)
        assert sizes[0] == sizes[1], f"Non-deterministic: {sizes}"
    test('22_deterministic', t22)

    # ---- 23: PNG header valid ----
    def t23():
        dashboard = TrainingDashboard(config=config)
        metrics = {'loss': [1.0, 0.5]}
        fig = dashboard.plot_loss_curves(metrics)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            fig.savefig(f.name, dpi=72)
            plt.close(fig)
            with open(f.name, 'rb') as fp:
                header = fp.read(8)
            os.unlink(f.name)
        assert header[:4] == b'\x89PNG', "Invalid PNG"
    test('23_png_header', t23)

    # ---- 24: Save figure ----
    def t24():
        dashboard = TrainingDashboard(config=config)
        metrics = {'loss': [1.0, 0.5]}
        fig = dashboard.plot_loss_curves(metrics)
        with tempfile.TemporaryDirectory() as td:
            db_local = TrainingDashboard(config=VizConfig(save_dir=td, dpi=72))
            path = db_local.save_figure(fig, 'test_loss')
            assert os.path.exists(path)
            assert os.path.getsize(path) > 500
    test('24_save_figure', t24)

    # ---- 25: Numpy input ----
    def t25():
        dashboard = TrainingDashboard(config=config)
        metrics = {'loss': np.array([1.0, 0.5, 0.3])}
        fig = dashboard.plot_loss_curves(metrics)
        assert_fig(fig, 'numpy input')
    test('25_numpy_input', t25)

    # ---- 26: Torch tensor input ----
    def t26():
        dashboard = TrainingDashboard(config=config)
        metrics = {'loss': torch.tensor([1.0, 0.5, 0.3])}
        fig = dashboard.plot_loss_curves(metrics)
        assert_fig(fig, 'torch input')
    test('26_torch_input', t26)

    # ---- 27: Custom title loss curves ----
    def t27():
        dashboard = TrainingDashboard(config=config)
        metrics = {'loss': [1.0, 0.5]}
        fig = dashboard.plot_loss_curves(metrics, title='My Loss')
        ax = fig.axes[0]
        assert 'My Loss' in ax.get_title()
        plt.close(fig)
    test('27_custom_title', t27)

    # ---- 28: Report with phase names ----
    def t28():
        with tempfile.TemporaryDirectory() as td:
            dashboard = TrainingDashboard(config=VizConfig(dpi=72))
            metrics = {'loss': [1.0, 0.8, 0.6, 0.4, 0.3]}
            path = dashboard.generate_report(
                td, metrics=metrics,
                phase_boundaries=[2],
                phase_names=['SNN Core', 'HTM Temporal'],
            )
            assert os.path.exists(path)
            with open(path) as f:
                content = f.read()
            assert len(content) > 500
    test('28_report_phase_names', t28)

    # ---- 29: Single metric loss ----
    def t29():
        dashboard = TrainingDashboard(config=config)
        metrics = {'validation_loss': [2.0, 1.5, 1.0, 0.8]}
        fig = dashboard.plot_loss_curves(metrics)
        assert_fig(fig, 'single val loss')
    test('29_single_val_loss', t29)

    # ---- 30: Empty report ----
    def t30():
        with tempfile.TemporaryDirectory() as td:
            dashboard = TrainingDashboard(config=VizConfig(dpi=72))
            path = dashboard.generate_report(td)
            assert os.path.exists(path)
            with open(path) as f:
                content = f.read()
            assert '<html' in content.lower()
    test('30_empty_report', t30)

    # Print
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{'='*60}")
    print(f"VizConfig & TrainingDashboard Self-Tests: {passed} passed, {failed} failed")
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
