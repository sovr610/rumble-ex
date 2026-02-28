"""
Spike Raster Visualization Template for brain_ai.

Provides SpikeRasterPlotter class for visualizing SNN spike patterns,
firing rate heatmaps, and membrane potential traces.

Usage:
    from spike_raster_template import SpikeRasterPlotter, VizConfig
    config = VizConfig()
    plotter = SpikeRasterPlotter(config)
    fig = plotter.plot_raster(spikes)
    fig.savefig('raster.png')

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
import tempfile
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union


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

LAYER_COLORS = ['#4682B4', '#3CB371', '#FF7F50', '#9370DB']

MODALITY_COLORS = {
    'vision': '#1f77b4',
    'text': '#2ca02c',
    'audio': '#ff7f0e',
    'sensor': '#9467bd',
    'engram': '#8c564b',
}


# ---------------------------------------------------------------------------
# SpikeRasterPlotter
# ---------------------------------------------------------------------------

class SpikeRasterPlotter:
    """Visualize SNN spike patterns across neurons and time steps.

    Handles spike raster plots, firing rate heatmaps, and membrane
    potential traces for the brain_ai SNN core.

    Args:
        config: VizConfig instance controlling appearance.
    """

    def __init__(self, config: Optional[VizConfig] = None):
        self.config = config or VizConfig()
        self._apply_style()

    # -- style helpers -------------------------------------------------------

    def _apply_style(self):
        """Apply dark/light mode globally for this plotter."""
        if self.config.dark_mode:
            plt.rcParams.update({
                'figure.facecolor': '#1a1a2e',
                'axes.facecolor': '#1a1a2e',
                'text.color': '#e0e0e0',
                'axes.labelcolor': '#e0e0e0',
                'xtick.color': '#e0e0e0',
                'ytick.color': '#e0e0e0',
            })
        else:
            plt.rcParams.update(plt.rcParamsDefault)
            matplotlib.use('Agg')  # re-set after defaults reset

    def _to_numpy(self, tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Convert tensor to numpy, detaching if needed."""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return np.asarray(tensor)

    def _subsample_neurons(
        self,
        data: np.ndarray,
        neuron_ids: Optional[List[int]] = None,
        axis: int = -1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Subsample neurons to max_neurons. Returns (data, indices)."""
        n_neurons = data.shape[axis]
        if neuron_ids is not None:
            idx = np.array(neuron_ids)
            return np.take(data, idx, axis=axis), idx
        if n_neurons <= self.config.max_neurons:
            return data, np.arange(n_neurons)
        idx = np.linspace(0, n_neurons - 1, self.config.max_neurons, dtype=int)
        return np.take(data, idx, axis=axis), idx

    def _subsample_timesteps(self, data: np.ndarray, axis: int = -2) -> Tuple[np.ndarray, np.ndarray]:
        """Subsample timesteps to max_timesteps. Returns (data, indices)."""
        n_steps = data.shape[axis]
        if n_steps <= self.config.max_timesteps:
            return data, np.arange(n_steps)
        idx = np.linspace(0, n_steps - 1, self.config.max_timesteps, dtype=int)
        return np.take(data, idx, axis=axis), idx

    # -- public API ----------------------------------------------------------

    def plot_raster(
        self,
        spikes: Union[torch.Tensor, np.ndarray],
        neuron_ids: Optional[List[int]] = None,
        batch_idx: int = 0,
        title: Optional[str] = None,
        layer_boundaries: Optional[List[int]] = None,
    ) -> plt.Figure:
        """Create a spike raster plot.

        Args:
            spikes: Tensor of shape (batch, timesteps, neurons) with binary values.
            neuron_ids: Optional list of neuron indices to plot.
            batch_idx: Which batch element to visualize.
            title: Custom title (auto-generated if None).
            layer_boundaries: Neuron indices where SNN layers start.

        Returns:
            matplotlib Figure object.
        """
        data = self._to_numpy(spikes)
        if data.ndim == 3:
            data = data[batch_idx]  # (timesteps, neurons)
        elif data.ndim == 2:
            pass  # already (timesteps, neurons)
        else:
            raise ValueError(f"Expected 2D or 3D spike tensor, got {data.ndim}D")

        data, t_idx = self._subsample_timesteps(data, axis=0)
        data, n_idx = self._subsample_neurons(data, neuron_ids, axis=1)

        fig, ax = plt.subplots(figsize=self.config.figsize)

        # Find spike locations
        times, neurons = np.where(data > 0.5)
        if len(times) > 0:
            ax.scatter(
                t_idx[times], n_idx[neurons],
                s=1.5, c='black' if not self.config.dark_mode else '#00ff88',
                marker='|', linewidths=0.7, rasterized=(len(times) > 5000),
            )
        else:
            ax.text(
                0.5, 0.5, 'No spikes detected',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=14, color='gray',
            )

        # Layer boundaries
        if layer_boundaries is not None:
            for boundary in layer_boundaries:
                ax.axhline(y=boundary, color='red', linestyle='--', alpha=0.4, linewidth=0.8)

        ax.set_xlabel('Time Step')
        ax.set_ylabel('Neuron Index')
        total_neurons = spikes.shape[-1] if hasattr(spikes, 'shape') else data.shape[-1]
        subsample_note = f' ({len(n_idx)} of {total_neurons})' if len(n_idx) < total_neurons else ''
        ax.set_title(title or f'Spike Raster Plot{subsample_note}')
        ax.set_xlim(t_idx[0], t_idx[-1])
        if len(n_idx) > 0:
            ax.set_ylim(n_idx[0] - 1, n_idx[-1] + 1)
        fig.tight_layout()
        return fig

    def plot_firing_rates(
        self,
        spikes: Union[torch.Tensor, np.ndarray],
        window: int = 10,
        batch_idx: int = 0,
        neuron_ids: Optional[List[int]] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Create a firing rate heatmap using a sliding window.

        Args:
            spikes: Tensor of shape (batch, timesteps, neurons).
            window: Sliding window size for rate computation.
            batch_idx: Which batch element to visualize.
            neuron_ids: Optional neuron subset.
            title: Custom title.

        Returns:
            matplotlib Figure.
        """
        data = self._to_numpy(spikes)
        if data.ndim == 3:
            data = data[batch_idx]
        data, n_idx = self._subsample_neurons(data, neuron_ids, axis=1)
        data, t_idx = self._subsample_timesteps(data, axis=0)

        # Compute firing rates via convolution
        kernel = np.ones(window) / window
        n_time, n_neurons = data.shape
        rates = np.zeros_like(data, dtype=float)
        for n in range(n_neurons):
            if n_time >= window:
                rates[:, n] = np.convolve(data[:, n], kernel, mode='same')
            else:
                rates[:, n] = data[:, n].mean()

        fig, axes = plt.subplots(1, 2, figsize=(self.config.figsize[0] + 4, self.config.figsize[1]),
                                 gridspec_kw={'width_ratios': [4, 1]})

        # Heatmap
        im = axes[0].imshow(
            rates.T, aspect='auto', origin='lower',
            cmap=self.config.colormap, vmin=0,
            extent=[t_idx[0], t_idx[-1], n_idx[0], n_idx[-1]],
        )
        axes[0].set_xlabel('Time Step')
        axes[0].set_ylabel('Neuron Index')
        axes[0].set_title(title or f'Firing Rate Heatmap (window={window})')
        fig.colorbar(im, ax=axes[0], label='Firing Rate (spikes/step)')

        # Marginal histogram of mean rates
        mean_rates = rates.mean(axis=0)
        axes[1].barh(np.arange(len(mean_rates)), mean_rates, color='steelblue', height=0.8)
        axes[1].set_xlabel('Mean Rate')
        axes[1].set_ylabel('Neuron')
        axes[1].set_title('Distribution')
        axes[1].set_ylim(-0.5, len(mean_rates) - 0.5)

        fig.tight_layout()
        return fig

    def plot_membrane_potential(
        self,
        membrane: Union[torch.Tensor, np.ndarray],
        threshold: float = 1.0,
        neuron_ids: Optional[List[int]] = None,
        batch_idx: int = 0,
        max_traces: int = 5,
        spikes: Optional[Union[torch.Tensor, np.ndarray]] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Plot membrane potential traces for selected neurons.

        Args:
            membrane: Tensor (batch, timesteps, neurons) of membrane potentials.
            threshold: Spike threshold value.
            neuron_ids: Specific neurons to plot. If None, picks top-max_traces active.
            batch_idx: Batch element index.
            max_traces: Max number of neuron traces to overlay.
            spikes: Optional spike tensor to mark spike times.
            title: Custom title.

        Returns:
            matplotlib Figure.
        """
        data = self._to_numpy(membrane)
        if data.ndim == 3:
            data = data[batch_idx]  # (timesteps, neurons)

        n_time, n_neurons = data.shape

        # Select neurons
        if neuron_ids is not None:
            selected = np.array(neuron_ids[:max_traces])
        else:
            # Pick neurons with highest variance (most interesting dynamics)
            variance = np.nanvar(data, axis=0)
            selected = np.argsort(variance)[-max_traces:][::-1]

        colors = plt.cm.tab10(np.linspace(0, 1, len(selected)))

        fig, ax = plt.subplots(figsize=self.config.figsize)

        for i, nid in enumerate(selected):
            trace = data[:, nid]
            # Replace NaN with 0 for plotting
            trace = np.nan_to_num(trace, nan=0.0)
            ax.plot(trace, color=colors[i], linewidth=1.2, label=f'Neuron {nid}', alpha=0.85)

            # Mark spikes
            if spikes is not None:
                sp = self._to_numpy(spikes)
                if sp.ndim == 3:
                    sp = sp[batch_idx]
                spike_times = np.where(sp[:, nid] > 0.5)[0]
                for st in spike_times:
                    ax.axvline(x=st, color=colors[i], linestyle=':', alpha=0.3, linewidth=0.5)

        # Threshold line
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5,
                    alpha=0.7, label=f'Threshold ({threshold})')

        ax.set_xlabel('Time Step')
        ax.set_ylabel('Membrane Potential')
        ax.set_title(title or 'Membrane Potential Traces')
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.set_xlim(0, n_time)
        fig.tight_layout()
        return fig

    def plot_spike_count_distribution(
        self,
        spikes: Union[torch.Tensor, np.ndarray],
        batch_idx: int = 0,
        target_rate: float = 0.1,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Plot distribution of total spike counts across neurons.

        Args:
            spikes: Spike tensor (batch, timesteps, neurons).
            batch_idx: Batch element index.
            target_rate: Expected firing rate for annotation.
            title: Custom title.

        Returns:
            matplotlib Figure.
        """
        data = self._to_numpy(spikes)
        if data.ndim == 3:
            data = data[batch_idx]

        n_time = data.shape[0]
        counts = data.sum(axis=0)  # per-neuron total
        expected_count = target_rate * n_time

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(counts, bins=30, color='steelblue', edgecolor='white', alpha=0.85)
        ax.axvline(x=expected_count, color='red', linestyle='--', linewidth=2,
                    label=f'Expected ({expected_count:.1f})')
        ax.set_xlabel('Spike Count')
        ax.set_ylabel('Number of Neurons')
        ax.set_title(title or 'Spike Count Distribution')
        ax.legend()
        fig.tight_layout()
        return fig

    def plot_population_activity(
        self,
        spikes: Union[torch.Tensor, np.ndarray],
        batch_idx: int = 0,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Plot total population activity over time.

        Args:
            spikes: Spike tensor (batch, timesteps, neurons).
            batch_idx: Batch element.
            title: Custom title.

        Returns:
            matplotlib Figure.
        """
        data = self._to_numpy(spikes)
        if data.ndim == 3:
            data = data[batch_idx]

        pop_activity = data.sum(axis=1)  # sum across neurons at each timestep

        fig, ax = plt.subplots(figsize=self.config.figsize)
        ax.fill_between(range(len(pop_activity)), pop_activity, alpha=0.4, color='steelblue')
        ax.plot(pop_activity, color='steelblue', linewidth=1.5)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Active Neuron Count')
        ax.set_title(title or 'Population Activity Over Time')
        ax.set_xlim(0, len(pop_activity) - 1)
        fig.tight_layout()
        return fig

    def save_figure(self, fig: plt.Figure, filename: str, close: bool = True) -> str:
        """Save figure to configured directory.

        Args:
            fig: Figure to save.
            filename: Filename (extension optional).
            close: Whether to close figure after saving.

        Returns:
            Full path to saved file.
        """
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
    """Run self-tests verifying figure generation. Returns (passed, failed) counts."""
    import traceback

    results = []

    def test(name, func):
        try:
            func()
            results.append((name, True, ''))
        except Exception as e:
            results.append((name, False, f'{e}\n{traceback.format_exc()}'))

    config = VizConfig(figsize=(8, 5), dpi=72, max_neurons=50, max_timesteps=100)
    plotter = SpikeRasterPlotter(config)

    # Helper: save figure and check non-empty
    def assert_figure_valid(fig, label=''):
        assert fig is not None, f"Figure is None ({label})"
        assert len(fig.axes) > 0, f"No axes in figure ({label})"
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            fig.savefig(f.name, dpi=72)
            plt.close(fig)
            size = os.path.getsize(f.name)
            os.unlink(f.name)
        assert size > 500, f"PNG too small: {size} bytes ({label})"

    # ---- Test 1: Basic raster plot ----
    def t01():
        spikes = (torch.rand(2, 50, 100) > 0.9).float()
        fig = plotter.plot_raster(spikes)
        assert_figure_valid(fig, 'basic raster')
    test('01_basic_raster', t01)

    # ---- Test 2: Raster with neuron_ids ----
    def t02():
        spikes = (torch.rand(1, 30, 200) > 0.85).float()
        fig = plotter.plot_raster(spikes, neuron_ids=[0, 10, 50, 100, 199])
        assert_figure_valid(fig, 'raster neuron_ids')
    test('02_raster_neuron_ids', t02)

    # ---- Test 3: Raster with layer boundaries ----
    def t03():
        spikes = (torch.rand(1, 40, 80) > 0.9).float()
        fig = plotter.plot_raster(spikes, layer_boundaries=[20, 40, 60])
        assert_figure_valid(fig, 'raster layer boundaries')
    test('03_raster_layer_boundaries', t03)

    # ---- Test 4: Raster axes labels ----
    def t04():
        spikes = (torch.rand(1, 20, 30) > 0.9).float()
        fig = plotter.plot_raster(spikes)
        ax = fig.axes[0]
        assert 'Time' in ax.get_xlabel(), f"X label missing 'Time': {ax.get_xlabel()}"
        assert 'Neuron' in ax.get_ylabel(), f"Y label missing 'Neuron': {ax.get_ylabel()}"
        plt.close(fig)
    test('04_raster_axes_labels', t04)

    # ---- Test 5: Raster with empty spikes ----
    def t05():
        spikes = torch.zeros(1, 50, 100)
        fig = plotter.plot_raster(spikes)
        assert_figure_valid(fig, 'empty spikes')
    test('05_empty_spikes_raster', t05)

    # ---- Test 6: Raster 2D input ----
    def t06():
        spikes = (torch.rand(30, 60) > 0.9).float()
        fig = plotter.plot_raster(spikes)
        assert_figure_valid(fig, '2D raster')
    test('06_raster_2d', t06)

    # ---- Test 7: Basic firing rate heatmap ----
    def t07():
        spikes = (torch.rand(1, 50, 80) > 0.88).float()
        fig = plotter.plot_firing_rates(spikes, window=5)
        assert_figure_valid(fig, 'firing rates')
    test('07_firing_rates_basic', t07)

    # ---- Test 8: Firing rate axes labels ----
    def t08():
        spikes = (torch.rand(1, 40, 50) > 0.9).float()
        fig = plotter.plot_firing_rates(spikes)
        ax = fig.axes[0]
        assert 'Time' in ax.get_xlabel(), f"X label: {ax.get_xlabel()}"
        assert 'Neuron' in ax.get_ylabel(), f"Y label: {ax.get_ylabel()}"
        plt.close(fig)
    test('08_firing_rates_axes', t08)

    # ---- Test 9: Firing rates with neuron subset ----
    def t09():
        spikes = (torch.rand(1, 30, 200) > 0.9).float()
        fig = plotter.plot_firing_rates(spikes, neuron_ids=[0, 50, 100, 150])
        assert_figure_valid(fig, 'firing rates subset')
    test('09_firing_rates_subset', t09)

    # ---- Test 10: Firing rates colorbar present ----
    def t10():
        spikes = (torch.rand(1, 50, 60) > 0.9).float()
        fig = plotter.plot_firing_rates(spikes)
        # Colorbar adds an extra axes
        assert len(fig.axes) >= 2, f"Expected colorbar axes, got {len(fig.axes)} axes"
        plt.close(fig)
    test('10_firing_rates_colorbar', t10)

    # ---- Test 11: Basic membrane potential ----
    def t11():
        membrane = torch.randn(1, 50, 100) * 0.5
        fig = plotter.plot_membrane_potential(membrane, threshold=1.0)
        assert_figure_valid(fig, 'membrane basic')
    test('11_membrane_basic', t11)

    # ---- Test 12: Membrane with spikes overlay ----
    def t12():
        membrane = torch.randn(1, 40, 80) * 0.6
        spikes = (membrane > 0.8).float()
        fig = plotter.plot_membrane_potential(membrane, threshold=0.8, spikes=spikes)
        assert_figure_valid(fig, 'membrane with spikes')
    test('12_membrane_with_spikes', t12)

    # ---- Test 13: Membrane axes labels ----
    def t13():
        membrane = torch.randn(1, 30, 50)
        fig = plotter.plot_membrane_potential(membrane, threshold=1.0)
        ax = fig.axes[0]
        xl = ax.get_xlabel().lower()
        yl = ax.get_ylabel().lower()
        assert 'time' in xl or 'step' in xl, f"X label: {ax.get_xlabel()}"
        assert 'membrane' in yl or 'potential' in yl, f"Y label: {ax.get_ylabel()}"
        plt.close(fig)
    test('13_membrane_axes', t13)

    # ---- Test 14: Membrane with specific neurons ----
    def t14():
        membrane = torch.randn(1, 50, 100) * 0.4
        fig = plotter.plot_membrane_potential(membrane, threshold=1.0, neuron_ids=[0, 25, 50, 75, 99])
        assert_figure_valid(fig, 'membrane selected neurons')
    test('14_membrane_selected_neurons', t14)

    # ---- Test 15: Membrane threshold line present ----
    def t15():
        membrane = torch.randn(1, 20, 30)
        fig = plotter.plot_membrane_potential(membrane, threshold=0.7)
        ax = fig.axes[0]
        lines = ax.get_lines()
        has_threshold = any('Threshold' in str(l.get_label()) for l in lines)
        assert has_threshold, "Threshold line not labeled"
        plt.close(fig)
    test('15_membrane_threshold', t15)

    # ---- Test 16: Spike count distribution ----
    def t16():
        spikes = (torch.rand(1, 50, 100) > 0.9).float()
        fig = plotter.plot_spike_count_distribution(spikes, target_rate=0.1)
        assert_figure_valid(fig, 'spike count dist')
    test('16_spike_count_dist', t16)

    # ---- Test 17: Population activity ----
    def t17():
        spikes = (torch.rand(1, 50, 100) > 0.9).float()
        fig = plotter.plot_population_activity(spikes)
        assert_figure_valid(fig, 'population activity')
    test('17_population_activity', t17)

    # ---- Test 18: Large tensor subsampling ----
    def t18():
        spikes = (torch.rand(1, 500, 4096) > 0.95).float()
        fig = plotter.plot_raster(spikes)
        assert_figure_valid(fig, 'large tensor')
    test('18_large_tensor_subsample', t18)

    # ---- Test 19: Save figure to tempdir ----
    def t19():
        spikes = (torch.rand(1, 20, 30) > 0.9).float()
        fig = plotter.plot_raster(spikes)
        with tempfile.TemporaryDirectory() as td:
            plotter_local = SpikeRasterPlotter(VizConfig(save_dir=td, dpi=72))
            path = plotter_local.save_figure(fig, 'test_save')
            assert os.path.exists(path), f"File not saved: {path}"
            assert os.path.getsize(path) > 500, "Saved file too small"
    test('19_save_figure', t19)

    # ---- Test 20: Dark mode raster ----
    def t20():
        dark_config = VizConfig(dark_mode=True, figsize=(8, 5), dpi=72)
        dark_plotter = SpikeRasterPlotter(dark_config)
        spikes = (torch.rand(1, 40, 60) > 0.9).float()
        fig = dark_plotter.plot_raster(spikes)
        assert_figure_valid(fig, 'dark mode raster')
        # Reset style
        plt.rcParams.update(plt.rcParamsDefault)
        matplotlib.use('Agg')
    test('20_dark_mode', t20)

    # ---- Test 21: Deterministic raster rendering ----
    def t21():
        spikes = (torch.rand(1, 30, 50) > 0.9).float()
        sizes = []
        for _ in range(2):
            fig = plotter.plot_raster(spikes)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                fig.savefig(f.name, dpi=72)
                plt.close(fig)
                sizes.append(os.path.getsize(f.name))
                os.unlink(f.name)
        # Sizes should be identical for same input
        assert sizes[0] == sizes[1], f"Non-deterministic: {sizes}"
    test('21_deterministic', t21)

    # ---- Test 22: PNG valid header ----
    def t22():
        spikes = (torch.rand(1, 20, 30) > 0.9).float()
        fig = plotter.plot_raster(spikes)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            fig.savefig(f.name, dpi=72)
            plt.close(fig)
            with open(f.name, 'rb') as fp:
                header = fp.read(8)
            os.unlink(f.name)
        assert header[:4] == b'\x89PNG', f"Invalid PNG header: {header[:4]}"
    test('22_png_header', t22)

    # ---- Test 23: NaN handling in membrane ----
    def t23():
        membrane = torch.randn(1, 30, 50)
        membrane[0, 10, 5] = float('nan')
        fig = plotter.plot_membrane_potential(membrane, threshold=1.0)
        assert_figure_valid(fig, 'nan membrane')
    test('23_nan_membrane', t23)

    # ---- Test 24: Single neuron raster ----
    def t24():
        spikes = (torch.rand(1, 50, 1) > 0.8).float()
        fig = plotter.plot_raster(spikes)
        assert_figure_valid(fig, 'single neuron')
    test('24_single_neuron', t24)

    # ---- Test 25: Single timestep raster ----
    def t25():
        spikes = (torch.rand(1, 1, 50) > 0.5).float()
        fig = plotter.plot_raster(spikes)
        assert_figure_valid(fig, 'single timestep')
    test('25_single_timestep', t25)

    # ---- Test 26: Custom title ----
    def t26():
        spikes = (torch.rand(1, 20, 30) > 0.9).float()
        fig = plotter.plot_raster(spikes, title='Custom Title Test')
        ax = fig.axes[0]
        assert 'Custom Title Test' in ax.get_title()
        plt.close(fig)
    test('26_custom_title', t26)

    # ---- Test 27: Numpy array input ----
    def t27():
        spikes = (np.random.rand(1, 30, 40) > 0.9).astype(float)
        fig = plotter.plot_raster(spikes)
        assert_figure_valid(fig, 'numpy input')
    test('27_numpy_input', t27)

    # Print results
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{'='*60}")
    print(f"SpikeRasterPlotter Self-Tests: {passed} passed, {failed} failed")
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
