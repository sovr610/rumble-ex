"""
Workspace Visualization Template for brain_ai.

Provides WorkspaceVisualizer class for visualizing global workspace
competition dynamics, broadcast patterns, and working memory slots.

Usage:
    from workspace_viz_template import WorkspaceVisualizer, VizConfig
    viz = WorkspaceVisualizer(VizConfig())
    fig = viz.plot_competition_dynamics(history)

Requires: matplotlib, numpy, torch
Backend: Agg (headless, no display required)
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
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

MODALITY_COLORS = {
    'vision': '#1f77b4',
    'text': '#2ca02c',
    'audio': '#ff7f0e',
    'sensor': '#9467bd',
    'engram': '#8c564b',
}

MODULE_COLORS = {
    'SNN Core': '#4682B4',
    'HTM': '#2E8B57',
    'Workspace': '#FFD700',
    'Reasoning': '#DA70D6',
    'Decision': '#FF6347',
    'Meta': '#6A5ACD',
    'Encoders': '#17becf',
}

SLOT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
               '#9467bd', '#8c564b', '#e377c2']


# ---------------------------------------------------------------------------
# WorkspaceVisualizer
# ---------------------------------------------------------------------------

class WorkspaceVisualizer:
    """Visualize global workspace state, broadcast patterns, and working memory.

    Args:
        config: VizConfig instance.
    """

    def __init__(self, config: Optional[VizConfig] = None):
        self.config = config or VizConfig()

    def _to_numpy(self, tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return np.asarray(tensor)

    def plot_competition_dynamics(
        self,
        history: Union[List[Union[torch.Tensor, np.ndarray]], torch.Tensor, np.ndarray],
        modality_names: Optional[List[str]] = None,
        threshold: float = 0.3,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Plot competition scores over selection rounds.

        Args:
            history: List of 1D tensors (per round) or 2D tensor (rounds, modalities).
            modality_names: Names for each modality.
            threshold: Ignition threshold.
            title: Plot title.

        Returns:
            matplotlib Figure.
        """
        # Normalize input to 2D numpy
        if isinstance(history, list):
            data = np.array([self._to_numpy(h).ravel() for h in history])
        else:
            data = self._to_numpy(history)
            if data.ndim == 1:
                data = data.reshape(1, -1)

        n_rounds, n_mod = data.shape

        if modality_names is None:
            defaults = ['vision', 'text', 'audio', 'sensor', 'engram']
            modality_names = defaults[:n_mod] if n_mod <= len(defaults) else [f'Mod{i}' for i in range(n_mod)]

        fig, ax = plt.subplots(figsize=self.config.figsize)
        rounds = np.arange(n_rounds)

        for i in range(n_mod):
            name = modality_names[i] if i < len(modality_names) else f'Mod{i}'
            color = MODALITY_COLORS.get(name, f'C{i}')
            ax.plot(rounds, data[:, i], '-o', label=name, color=color,
                    linewidth=2, markersize=7)

            # Mark ignition point
            ignited = np.where(data[:, i] >= threshold)[0]
            if len(ignited) > 0:
                first = ignited[0]
                ax.plot(first, data[first, i], '*', color=color,
                        markersize=14, markeredgecolor='black', markeredgewidth=0.5)

        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5,
                    alpha=0.7, label=f'Ignition Threshold ({threshold})')

        # Mark winner at final round
        winner = int(np.argmax(data[-1]))
        ax.annotate(
            f'Winner: {modality_names[winner] if winner < len(modality_names) else f"Mod{winner}"}',
            xy=(n_rounds - 1, data[-1, winner]),
            xytext=(n_rounds - 1 - 0.5, data[-1, winner] + 0.08),
            fontsize=9, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
        )

        ax.set_xlabel('Selection Round')
        ax.set_ylabel('Competition Score')
        ax.set_title(title or 'Workspace Competition Dynamics')
        ax.set_ylim(0, 1.05)
        ax.set_xlim(-0.2, n_rounds - 0.8)
        ax.set_xticks(rounds)
        ax.legend(loc='upper left', fontsize=9)
        fig.tight_layout()
        return fig

    def plot_broadcast_map(
        self,
        broadcast: Union[torch.Tensor, np.ndarray],
        module_names: Optional[List[str]] = None,
        source_name: Optional[str] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Plot broadcast strength from workspace to each module.

        Args:
            broadcast: 1D tensor of broadcast strengths to each module,
                       or 2D tensor (sources, targets).
            module_names: Names of receiving modules.
            source_name: Name of the broadcast source.
            title: Plot title.

        Returns:
            matplotlib Figure.
        """
        data = self._to_numpy(broadcast)

        if data.ndim == 1:
            # Single source -> multiple targets
            n_modules = len(data)
            if module_names is None:
                defaults = list(MODULE_COLORS.keys())
                module_names = defaults[:n_modules] if n_modules <= len(defaults) else [f'Module{i}' for i in range(n_modules)]

            fig, ax = plt.subplots(figsize=(max(8, n_modules * 1.2), 6))
            colors = [MODULE_COLORS.get(m, '#808080') for m in module_names]
            bars = ax.bar(range(n_modules), data, color=colors, edgecolor='black', linewidth=0.8)

            ax.set_xticks(range(n_modules))
            ax.set_xticklabels(module_names, rotation=30, ha='right')
            ax.set_xlabel('Target Module')
            ax.set_ylabel('Broadcast Strength')
            source_label = f' from {source_name}' if source_name else ''
            ax.set_title(title or f'Broadcast Map{source_label}')
            ax.set_ylim(0, max(float(np.max(data)) * 1.15, 0.1))

            # Annotate bars
            for bar, val in zip(bars, data):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=9)

        elif data.ndim == 2:
            # Multiple sources -> multiple targets
            n_src, n_tgt = data.shape
            if module_names is None:
                defaults = list(MODULE_COLORS.keys())
                module_names = defaults[:n_tgt] if n_tgt <= len(defaults) else [f'Mod{i}' for i in range(n_tgt)]
            src_names = [f'Source {i}' for i in range(n_src)]

            fig, ax = plt.subplots(figsize=(max(8, n_tgt * 2), max(5, n_src * 1.5)))
            im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0,
                            vmax=max(float(np.nanmax(data)), 1e-8))
            ax.set_xticks(range(n_tgt))
            ax.set_xticklabels(module_names[:n_tgt], rotation=30, ha='right')
            ax.set_yticks(range(n_src))
            ax.set_yticklabels(src_names)
            ax.set_xlabel('Target Module')
            ax.set_ylabel('Broadcast Source')
            ax.set_title(title or 'Broadcast Strength Matrix')

            for i in range(n_src):
                for j in range(n_tgt):
                    color = 'white' if data[i, j] > np.nanmax(data) / 2 else 'black'
                    ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center',
                            color=color, fontsize=9)

            fig.colorbar(im, ax=ax, label='Broadcast Strength')
        else:
            raise ValueError(f"Expected 1D or 2D broadcast tensor, got {data.ndim}D")

        fig.tight_layout()
        return fig

    def plot_working_memory_slots(
        self,
        memory: Union[torch.Tensor, np.ndarray],
        labels: Optional[List[str]] = None,
        title: Optional[str] = None,
        show_similarity: bool = True,
    ) -> plt.Figure:
        """Visualize working memory slot contents.

        Args:
            memory: 2D tensor (num_slots, dim) of slot contents.
            labels: Optional labels for each slot.
            title: Plot title.
            show_similarity: Whether to include a slot similarity heatmap.

        Returns:
            matplotlib Figure.
        """
        data = self._to_numpy(memory)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        n_slots, dim = data.shape

        if labels is None:
            labels = [f'Slot {i}' for i in range(n_slots)]

        ncols = 3 if show_similarity else 2
        fig, axes = plt.subplots(1, ncols, figsize=(ncols * 5, 6))

        # Panel 1: Slot norms
        norms = np.linalg.norm(data, axis=1)
        colors = [SLOT_COLORS[i % len(SLOT_COLORS)] for i in range(n_slots)]
        axes[0].bar(range(n_slots), norms, color=colors, edgecolor='black', linewidth=0.8)
        axes[0].set_xticks(range(n_slots))
        axes[0].set_xticklabels(labels[:n_slots], rotation=30, ha='right', fontsize=8)
        axes[0].set_xlabel('Memory Slot')
        axes[0].set_ylabel('L2 Norm')
        axes[0].set_title('Slot Norms')

        # Panel 2: Slot content mini-heatmaps
        # Reshape to grid for visualization
        grid_side = int(np.ceil(np.sqrt(dim)))
        padded = np.zeros((n_slots, grid_side * grid_side))
        padded[:, :dim] = data
        content_grid = padded.reshape(n_slots, grid_side, grid_side)

        # Show as a single row of mini heatmaps
        combined = np.concatenate([content_grid[i] for i in range(n_slots)], axis=1)
        im = axes[1].imshow(combined, cmap=self.config.colormap, aspect='auto')
        axes[1].set_title('Slot Contents (reshaped)')
        axes[1].set_xlabel('Dimension (tiled per slot)')
        axes[1].set_ylabel('Dimension')

        # Add vertical lines between slots
        for s in range(1, n_slots):
            axes[1].axvline(x=s * grid_side - 0.5, color='white', linewidth=2)

        fig.colorbar(im, ax=axes[1], label='Activation', shrink=0.8)

        # Panel 3: Pairwise similarity
        if show_similarity and ncols == 3:
            # Cosine similarity
            norms_safe = np.maximum(norms, 1e-8)
            normed = data / norms_safe[:, None]
            sim = normed @ normed.T

            im2 = axes[2].imshow(sim, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            axes[2].set_xticks(range(n_slots))
            axes[2].set_xticklabels(labels[:n_slots], rotation=45, ha='right', fontsize=8)
            axes[2].set_yticks(range(n_slots))
            axes[2].set_yticklabels(labels[:n_slots], fontsize=8)
            axes[2].set_title('Slot Cosine Similarity')

            for i in range(n_slots):
                for j in range(n_slots):
                    color = 'white' if abs(sim[i, j]) > 0.5 else 'black'
                    axes[2].text(j, i, f'{sim[i, j]:.2f}', ha='center', va='center',
                                 color=color, fontsize=max(6, 10 - n_slots))

            fig.colorbar(im2, ax=axes[2], label='Cosine Similarity', shrink=0.8)

        fig.suptitle(title or 'Working Memory Slots', fontsize=13)
        fig.tight_layout()
        return fig

    def plot_slot_dynamics(
        self,
        slot_history: Union[torch.Tensor, np.ndarray],
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Plot working memory slot norms over time.

        Args:
            slot_history: 3D tensor (timesteps, num_slots, dim) or
                          2D tensor (timesteps, num_slots) of norms.
            title: Plot title.

        Returns:
            matplotlib Figure.
        """
        data = self._to_numpy(slot_history)
        if data.ndim == 3:
            # Compute norms
            norms = np.linalg.norm(data, axis=-1)  # (timesteps, num_slots)
        elif data.ndim == 2:
            norms = data
        else:
            raise ValueError(f"Expected 2D or 3D, got {data.ndim}D")

        n_time, n_slots = norms.shape

        fig, ax = plt.subplots(figsize=self.config.figsize)
        for s in range(n_slots):
            color = SLOT_COLORS[s % len(SLOT_COLORS)]
            ax.plot(range(n_time), norms[:, s], '-', label=f'Slot {s}',
                    color=color, linewidth=1.5, alpha=0.85)

        ax.set_xlabel('Time Step')
        ax.set_ylabel('Slot Norm')
        ax.set_title(title or 'Working Memory Slot Dynamics')
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.set_xlim(0, n_time - 1)
        fig.tight_layout()
        return fig

    def plot_information_flow(
        self,
        flow_values: Dict[str, float],
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Plot information flow through the pipeline as a horizontal bar chart.

        Args:
            flow_values: Dict mapping module names to scalar flow values.
            title: Plot title.

        Returns:
            matplotlib Figure.
        """
        names = list(flow_values.keys())
        values = [flow_values[n] for n in names]
        n = len(names)

        fig, ax = plt.subplots(figsize=(10, max(4, n * 0.8)))
        colors = [MODULE_COLORS.get(name, '#808080') for name in names]
        y_pos = np.arange(n)
        ax.barh(y_pos, values, color=colors, edgecolor='black', linewidth=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.set_xlabel('Information Flow (a.u.)')
        ax.set_title(title or 'Information Flow Through Pipeline')

        for i, val in enumerate(values):
            ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9)

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
    viz = WorkspaceVisualizer(config)

    def assert_fig(fig, label=''):
        assert fig is not None, f"Figure is None ({label})"
        assert len(fig.axes) > 0, f"No axes ({label})"
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            fig.savefig(f.name, dpi=72)
            plt.close(fig)
            size = os.path.getsize(f.name)
            os.unlink(f.name)
        assert size > 500, f"PNG too small: {size} ({label})"

    # ---- 1: Competition dynamics basic ----
    def t01():
        history = [torch.tensor([0.25, 0.25, 0.25, 0.25]),
                   torch.tensor([0.4, 0.3, 0.2, 0.1]),
                   torch.tensor([0.6, 0.2, 0.12, 0.08])]
        fig = viz.plot_competition_dynamics(history)
        assert_fig(fig, 'competition basic')
    test('01_competition_basic', t01)

    # ---- 2: Competition with 2D tensor ----
    def t02():
        history = torch.tensor([
            [0.3, 0.3, 0.2, 0.2],
            [0.5, 0.25, 0.15, 0.1],
            [0.7, 0.15, 0.1, 0.05],
        ])
        fig = viz.plot_competition_dynamics(history)
        assert_fig(fig, 'competition 2D')
    test('02_competition_2d', t02)

    # ---- 3: Competition axes labels ----
    def t03():
        history = torch.tensor([[0.5, 0.5], [0.7, 0.3]])
        fig = viz.plot_competition_dynamics(history)
        ax = fig.axes[0]
        assert 'Round' in ax.get_xlabel(), f"X: {ax.get_xlabel()}"
        assert 'Score' in ax.get_ylabel(), f"Y: {ax.get_ylabel()}"
        plt.close(fig)
    test('03_competition_axes', t03)

    # ---- 4: Competition with custom names ----
    def t04():
        history = [torch.tensor([0.6, 0.4]), torch.tensor([0.8, 0.2])]
        fig = viz.plot_competition_dynamics(history, modality_names=['Vision', 'Text'])
        assert_fig(fig, 'competition custom names')
    test('04_competition_custom_names', t04)

    # ---- 5: Competition threshold line ----
    def t05():
        history = [torch.tensor([0.3, 0.3]), torch.tensor([0.6, 0.4])]
        fig = viz.plot_competition_dynamics(history, threshold=0.5)
        ax = fig.axes[0]
        lines = ax.get_lines()
        has_threshold = any('Threshold' in str(l.get_label()) for l in lines)
        assert has_threshold, "Threshold line missing"
        plt.close(fig)
    test('05_competition_threshold', t05)

    # ---- 6: Broadcast map 1D ----
    def t06():
        broadcast = torch.tensor([0.8, 0.6, 0.9, 0.5, 0.3])
        fig = viz.plot_broadcast_map(broadcast)
        assert_fig(fig, 'broadcast 1D')
    test('06_broadcast_1d', t06)

    # ---- 7: Broadcast map with names ----
    def t07():
        broadcast = torch.tensor([0.7, 0.5, 0.3])
        fig = viz.plot_broadcast_map(broadcast,
                                     module_names=['SNN Core', 'HTM', 'Reasoning'],
                                     source_name='Vision')
        assert_fig(fig, 'broadcast named')
    test('07_broadcast_named', t07)

    # ---- 8: Broadcast map 2D matrix ----
    def t08():
        broadcast = torch.rand(3, 4)
        fig = viz.plot_broadcast_map(broadcast)
        assert_fig(fig, 'broadcast 2D')
    test('08_broadcast_2d', t08)

    # ---- 9: Working memory slots basic ----
    def t09():
        memory = torch.randn(7, 64)
        fig = viz.plot_working_memory_slots(memory)
        assert_fig(fig, 'wm basic')
    test('09_wm_basic', t09)

    # ---- 10: Working memory with labels ----
    def t10():
        memory = torch.randn(4, 32)
        labels = ['Visual', 'Textual', 'Auditory', 'Context']
        fig = viz.plot_working_memory_slots(memory, labels=labels)
        assert_fig(fig, 'wm labels')
    test('10_wm_labels', t10)

    # ---- 11: Working memory single slot ----
    def t11():
        memory = torch.randn(1, 128)
        fig = viz.plot_working_memory_slots(memory)
        assert_fig(fig, 'wm single slot')
    test('11_wm_single_slot', t11)

    # ---- 12: Working memory without similarity ----
    def t12():
        memory = torch.randn(5, 64)
        fig = viz.plot_working_memory_slots(memory, show_similarity=False)
        assert_fig(fig, 'wm no similarity')
    test('12_wm_no_similarity', t12)

    # ---- 13: Working memory zero slots ----
    def t13():
        memory = torch.zeros(7, 64)
        fig = viz.plot_working_memory_slots(memory)
        assert_fig(fig, 'wm zeros')
    test('13_wm_zeros', t13)

    # ---- 14: Slot dynamics 3D ----
    def t14():
        history = torch.randn(20, 7, 64)
        fig = viz.plot_slot_dynamics(history)
        assert_fig(fig, 'slot dynamics 3D')
    test('14_slot_dynamics_3d', t14)

    # ---- 15: Slot dynamics 2D ----
    def t15():
        norms = torch.rand(15, 5)
        fig = viz.plot_slot_dynamics(norms)
        assert_fig(fig, 'slot dynamics 2D')
    test('15_slot_dynamics_2d', t15)

    # ---- 16: Information flow ----
    def t16():
        flow = {'Encoders': 0.9, 'SNN Core': 0.7, 'HTM': 0.5,
                'Workspace': 0.8, 'Reasoning': 0.6, 'Decision': 0.4}
        fig = viz.plot_information_flow(flow)
        assert_fig(fig, 'info flow')
    test('16_info_flow', t16)

    # ---- 17: Deterministic rendering ----
    def t17():
        history = [torch.tensor([0.4, 0.3, 0.2, 0.1]),
                   torch.tensor([0.6, 0.2, 0.12, 0.08])]
        sizes = []
        for _ in range(2):
            fig = viz.plot_competition_dynamics(history)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                fig.savefig(f.name, dpi=72)
                plt.close(fig)
                sizes.append(os.path.getsize(f.name))
                os.unlink(f.name)
        assert sizes[0] == sizes[1], f"Non-deterministic: {sizes}"
    test('17_deterministic', t17)

    # ---- 18: PNG header valid ----
    def t18():
        broadcast = torch.tensor([0.5, 0.3, 0.2])
        fig = viz.plot_broadcast_map(broadcast)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            fig.savefig(f.name, dpi=72)
            plt.close(fig)
            with open(f.name, 'rb') as fp:
                header = fp.read(8)
            os.unlink(f.name)
        assert header[:4] == b'\x89PNG', "Invalid PNG"
    test('18_png_header', t18)

    # ---- 19: Save figure ----
    def t19():
        memory = torch.randn(4, 32)
        fig = viz.plot_working_memory_slots(memory, show_similarity=False)
        with tempfile.TemporaryDirectory() as td:
            viz_local = WorkspaceVisualizer(VizConfig(save_dir=td, dpi=72))
            path = viz_local.save_figure(fig, 'test_wm')
            assert os.path.exists(path)
            assert os.path.getsize(path) > 500
    test('19_save_figure', t19)

    # ---- 20: Numpy input ----
    def t20():
        history = np.array([[0.3, 0.3, 0.2, 0.2], [0.6, 0.2, 0.12, 0.08]])
        fig = viz.plot_competition_dynamics(history)
        assert_fig(fig, 'numpy input')
    test('20_numpy_input', t20)

    # ---- 21: Custom title ----
    def t21():
        history = [torch.tensor([0.5, 0.5]), torch.tensor([0.7, 0.3])]
        fig = viz.plot_competition_dynamics(history, title='My Title')
        ax = fig.axes[0]
        assert 'My Title' in ax.get_title()
        plt.close(fig)
    test('21_custom_title', t21)

    # ---- 22: Large workspace dim ----
    def t22():
        memory = torch.randn(7, 4096)
        fig = viz.plot_working_memory_slots(memory, show_similarity=False)
        assert_fig(fig, 'large dim')
    test('22_large_dim', t22)

    # Print
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{'='*60}")
    print(f"WorkspaceVisualizer Self-Tests: {passed} passed, {failed} failed")
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
