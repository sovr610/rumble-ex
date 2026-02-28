"""
Attention Heatmap Visualization Template for brain_ai.

Provides AttentionHeatmapper class for visualizing attention weights,
cross-modal attention patterns, and workspace competition scores.

Usage:
    from attention_heatmap_template import AttentionHeatmapper
    config = VizConfig()
    heatmapper = AttentionHeatmapper(config)
    fig = heatmapper.plot_attention(weights)

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

NEUROMODULATOR_COLORS = {
    'DA': '#d62728',
    'ACh': '#2ca02c',
    'NE': '#1f77b4',
    '5-HT': '#bcbd22',
}


# ---------------------------------------------------------------------------
# AttentionHeatmapper
# ---------------------------------------------------------------------------

class AttentionHeatmapper:
    """Visualize attention weights from workspace competition and cross-modal attention.

    Args:
        config: VizConfig instance controlling appearance.
    """

    def __init__(self, config: Optional[VizConfig] = None):
        self.config = config or VizConfig()

    def _to_numpy(self, tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return np.asarray(tensor)

    def plot_attention(
        self,
        weights: Union[torch.Tensor, np.ndarray],
        labels: Optional[List[str]] = None,
        title: Optional[str] = None,
        annotate: bool = True,
        vmin: float = 0.0,
        vmax: Optional[float] = None,
    ) -> plt.Figure:
        """Render a single attention weight matrix as a heatmap.

        Args:
            weights: 2D tensor of shape (query_len, key_len).
            labels: Optional axis labels (used for both axes if square, or provide tuple).
            title: Plot title.
            annotate: Whether to annotate cells with values (for small matrices).
            vmin: Minimum color scale value.
            vmax: Maximum color scale value (defaults to data max).

        Returns:
            matplotlib Figure.
        """
        data = self._to_numpy(weights)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        assert data.ndim == 2, f"Expected 2D weights, got {data.ndim}D"

        rows, cols = data.shape
        vmax = vmax if vmax is not None else float(np.nanmax(data))

        fig, ax = plt.subplots(figsize=self.config.figsize)
        im = ax.imshow(data, cmap=self.config.colormap, aspect='auto',
                        vmin=vmin, vmax=max(vmax, 1e-8))

        # Annotate cells for small matrices
        if annotate and rows <= 12 and cols <= 12:
            for i in range(rows):
                for j in range(cols):
                    val = data[i, j]
                    color = 'white' if val > (vmin + vmax) / 2 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            color=color, fontsize=max(6, 10 - max(rows, cols) // 2))

        # Labels
        if labels is not None:
            if rows <= 20:
                ax.set_yticks(range(rows))
                ax.set_yticklabels(labels[:rows] if len(labels) >= rows else
                                   [f'Q{i}' for i in range(rows)])
            if cols <= 20:
                ax.set_xticks(range(cols))
                ax.set_xticklabels(labels[:cols] if len(labels) >= cols else
                                   [f'K{i}' for i in range(cols)],
                                   rotation=45, ha='right')

        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        ax.set_title(title or 'Attention Weights')
        fig.colorbar(im, ax=ax, label='Attention Weight')
        fig.tight_layout()
        return fig

    def plot_multi_head_attention(
        self,
        weights: Union[torch.Tensor, np.ndarray],
        num_heads: Optional[int] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Plot attention weights for multiple heads in a grid.

        Args:
            weights: 3D tensor (num_heads, query_len, key_len).
            num_heads: Number of heads to display (defaults to all).
            title: Overall title.

        Returns:
            matplotlib Figure.
        """
        data = self._to_numpy(weights)
        assert data.ndim == 3, f"Expected 3D (heads, Q, K), got {data.ndim}D"

        n_heads = data.shape[0]
        if num_heads is not None:
            n_heads = min(num_heads, n_heads)
            data = data[:n_heads]

        ncols = min(8, n_heads)
        nrows = (n_heads + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 2.5, nrows * 2.5),
                                 squeeze=False)

        vmax = float(np.nanmax(data))
        for i in range(n_heads):
            r, c = divmod(i, ncols)
            im = axes[r][c].imshow(data[i], cmap=self.config.colormap,
                                    aspect='auto', vmin=0, vmax=max(vmax, 1e-8))
            axes[r][c].set_title(f'Head {i}', fontsize=8)
            axes[r][c].set_xticks([])
            axes[r][c].set_yticks([])

        # Hide unused subplots
        for i in range(n_heads, nrows * ncols):
            r, c = divmod(i, ncols)
            axes[r][c].set_visible(False)

        fig.suptitle(title or f'Multi-Head Attention ({n_heads} heads)', fontsize=12)
        fig.tight_layout()
        return fig

    def plot_head_entropy(
        self,
        weights: Union[torch.Tensor, np.ndarray],
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Plot entropy of each attention head.

        Args:
            weights: 3D tensor (num_heads, query_len, key_len).
            title: Plot title.

        Returns:
            matplotlib Figure.
        """
        data = self._to_numpy(weights)
        assert data.ndim == 3, f"Expected 3D, got {data.ndim}D"

        n_heads = data.shape[0]
        eps = 1e-10
        # Mean entropy across queries per head
        entropies = []
        for h in range(n_heads):
            head_attn = data[h] + eps
            head_attn = head_attn / head_attn.sum(axis=-1, keepdims=True)
            ent = -np.sum(head_attn * np.log(head_attn + eps), axis=-1)
            entropies.append(np.mean(ent))
        entropies = np.array(entropies)

        fig, ax = plt.subplots(figsize=(max(8, n_heads * 0.4), 5))
        colors = plt.cm.RdYlGn_r(entropies / (np.max(entropies) + 1e-8))
        ax.bar(range(n_heads), entropies, color=colors)
        ax.set_xlabel('Head Index')
        ax.set_ylabel('Mean Entropy')
        ax.set_title(title or 'Attention Head Entropy')
        ax.set_xticks(range(n_heads))
        fig.tight_layout()
        return fig

    def plot_cross_modal_attention(
        self,
        weights: Dict[str, Union[torch.Tensor, np.ndarray]],
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Plot cross-modal attention as a summary matrix.

        Args:
            weights: Dict mapping 'src->tgt' keys to attention tensors or scalars.
                     E.g., {'vision->text': tensor, 'text->vision': tensor}.
                     If tensors, the mean is taken for the summary cell.
            title: Plot title.

        Returns:
            matplotlib Figure.
        """
        # Parse modality names from keys
        modalities = set()
        for key in weights:
            parts = key.split('->')
            if len(parts) == 2:
                modalities.add(parts[0].strip())
                modalities.add(parts[1].strip())
        modalities = sorted(modalities)

        n = len(modalities)
        matrix = np.zeros((n, n))
        mod_idx = {m: i for i, m in enumerate(modalities)}

        for key, val in weights.items():
            parts = key.split('->')
            if len(parts) == 2:
                src, tgt = parts[0].strip(), parts[1].strip()
                v = self._to_numpy(val)
                matrix[mod_idx[src], mod_idx[tgt]] = float(np.nanmean(v))

        fig, ax = plt.subplots(figsize=(max(6, n * 1.5), max(5, n * 1.2)))
        im = ax.imshow(matrix, cmap='Blues', aspect='auto', vmin=0,
                        vmax=max(float(np.nanmax(matrix)), 1e-8))

        ax.set_xticks(range(n))
        ax.set_xticklabels(modalities, rotation=45, ha='right')
        ax.set_yticks(range(n))
        ax.set_yticklabels(modalities)
        ax.set_xlabel('Target Modality')
        ax.set_ylabel('Source Modality')
        ax.set_title(title or 'Cross-Modal Attention')

        # Annotate
        for i in range(n):
            for j in range(n):
                color = 'white' if matrix[i, j] > np.nanmax(matrix) / 2 else 'black'
                ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center',
                        color=color, fontsize=10)

        fig.colorbar(im, ax=ax, label='Mean Attention Weight')
        fig.tight_layout()
        return fig

    def plot_workspace_competition(
        self,
        scores: Union[torch.Tensor, np.ndarray],
        winner_idx: Optional[int] = None,
        modality_names: Optional[List[str]] = None,
        threshold: float = 0.3,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Plot workspace competition scores as a bar chart.

        Args:
            scores: 1D tensor of competition scores per modality.
            winner_idx: Index of the winner (auto-detected if None).
            modality_names: Names of modalities.
            threshold: Ignition threshold.
            title: Plot title.

        Returns:
            matplotlib Figure.
        """
        data = self._to_numpy(scores).ravel()
        n = len(data)

        if modality_names is None:
            default_names = ['vision', 'text', 'audio', 'sensor', 'engram']
            modality_names = default_names[:n] if n <= len(default_names) else [f'Mod{i}' for i in range(n)]

        if winner_idx is None:
            winner_idx = int(np.argmax(data))

        colors = []
        for i in range(n):
            if i == winner_idx:
                colors.append('#FFD700')  # Gold for winner
            else:
                name = modality_names[i] if i < len(modality_names) else 'unknown'
                colors.append(MODALITY_COLORS.get(name, '#808080'))

        fig, ax = plt.subplots(figsize=(max(6, n * 1.5), 5))
        bars = ax.bar(range(n), data, color=colors, edgecolor='black', linewidth=1.2)

        # Highlight winner
        bars[winner_idx].set_edgecolor('#FF4500')
        bars[winner_idx].set_linewidth(3)

        # Threshold line
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5,
                    alpha=0.7, label=f'Ignition Threshold ({threshold})')

        ax.set_xticks(range(n))
        ax.set_xticklabels(modality_names[:n], rotation=30, ha='right')
        ax.set_xlabel('Modality')
        ax.set_ylabel('Competition Score')
        ax.set_title(title or 'Workspace Competition Scores')
        ax.set_ylim(0, max(float(np.max(data)) * 1.15, threshold * 1.5))
        ax.legend(loc='upper right')
        fig.tight_layout()
        return fig

    def plot_competition_evolution(
        self,
        scores_history: Union[torch.Tensor, np.ndarray],
        modality_names: Optional[List[str]] = None,
        threshold: float = 0.3,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Plot competition score evolution over selection rounds.

        Args:
            scores_history: 2D tensor (rounds, modalities).
            modality_names: Modality names.
            threshold: Ignition threshold.
            title: Plot title.

        Returns:
            matplotlib Figure.
        """
        data = self._to_numpy(scores_history)
        assert data.ndim == 2, f"Expected 2D, got {data.ndim}D"
        n_rounds, n_mod = data.shape

        if modality_names is None:
            default_names = ['vision', 'text', 'audio', 'sensor', 'engram']
            modality_names = default_names[:n_mod] if n_mod <= len(default_names) else [f'Mod{i}' for i in range(n_mod)]

        fig, ax = plt.subplots(figsize=self.config.figsize)
        for i in range(n_mod):
            name = modality_names[i] if i < len(modality_names) else f'Mod{i}'
            color = MODALITY_COLORS.get(name, f'C{i}')
            ax.plot(range(n_rounds), data[:, i], '-o', label=name,
                    color=color, linewidth=2, markersize=6)

        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5,
                    alpha=0.7, label=f'Threshold ({threshold})')
        ax.set_xlabel('Selection Round')
        ax.set_ylabel('Competition Score')
        ax.set_title(title or 'Competition Score Evolution')
        ax.set_ylim(0, 1.05)
        ax.legend(loc='best', fontsize=9)
        ax.set_xticks(range(n_rounds))
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
    hm = AttentionHeatmapper(config)

    def assert_fig(fig, label=''):
        assert fig is not None, f"Figure is None ({label})"
        assert len(fig.axes) > 0, f"No axes ({label})"
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            fig.savefig(f.name, dpi=72)
            plt.close(fig)
            size = os.path.getsize(f.name)
            os.unlink(f.name)
        assert size > 500, f"PNG too small: {size} ({label})"

    # ---- 1: Basic attention heatmap ----
    def t01():
        w = torch.softmax(torch.randn(8, 8), dim=-1)
        fig = hm.plot_attention(w)
        assert_fig(fig, 'basic attention')
    test('01_basic_attention', t01)

    # ---- 2: Attention with labels ----
    def t02():
        w = torch.softmax(torch.randn(5, 5), dim=-1)
        labels = ['A', 'B', 'C', 'D', 'E']
        fig = hm.plot_attention(w, labels=labels)
        assert_fig(fig, 'attention labels')
    test('02_attention_labels', t02)

    # ---- 3: Attention axes labels ----
    def t03():
        w = torch.softmax(torch.randn(4, 6), dim=-1)
        fig = hm.plot_attention(w)
        ax = fig.axes[0]
        assert 'Key' in ax.get_xlabel(), f"X: {ax.get_xlabel()}"
        assert 'Query' in ax.get_ylabel(), f"Y: {ax.get_ylabel()}"
        plt.close(fig)
    test('03_axes_labels', t03)

    # ---- 4: Colorbar present ----
    def t04():
        w = torch.softmax(torch.randn(6, 6), dim=-1)
        fig = hm.plot_attention(w)
        assert len(fig.axes) >= 2, f"No colorbar, axes count: {len(fig.axes)}"
        plt.close(fig)
    test('04_colorbar_present', t04)

    # ---- 5: Large attention no annotations ----
    def t05():
        w = torch.softmax(torch.randn(50, 50), dim=-1)
        fig = hm.plot_attention(w, annotate=True)  # should auto-skip
        assert_fig(fig, 'large attention')
    test('05_large_no_annotate', t05)

    # ---- 6: 1x1 attention ----
    def t06():
        w = torch.tensor([[1.0]])
        fig = hm.plot_attention(w)
        assert_fig(fig, '1x1 attention')
    test('06_1x1_attention', t06)

    # ---- 7: 1D attention (single query) ----
    def t07():
        w = torch.softmax(torch.randn(5), dim=-1)
        fig = hm.plot_attention(w)
        assert_fig(fig, '1d attention')
    test('07_1d_attention', t07)

    # ---- 8: Multi-head attention grid ----
    def t08():
        w = torch.softmax(torch.randn(8, 6, 6), dim=-1)
        fig = hm.plot_multi_head_attention(w)
        assert_fig(fig, 'multi-head')
    test('08_multi_head', t08)

    # ---- 9: Multi-head subset ----
    def t09():
        w = torch.softmax(torch.randn(16, 4, 4), dim=-1)
        fig = hm.plot_multi_head_attention(w, num_heads=4)
        assert_fig(fig, 'multi-head subset')
    test('09_multi_head_subset', t09)

    # ---- 10: Head entropy ----
    def t10():
        w = torch.softmax(torch.randn(8, 6, 6), dim=-1)
        fig = hm.plot_head_entropy(w)
        assert_fig(fig, 'head entropy')
    test('10_head_entropy', t10)

    # ---- 11: Cross-modal attention ----
    def t11():
        weights = {
            'vision->text': torch.tensor(0.6),
            'text->vision': torch.tensor(0.4),
            'vision->audio': torch.tensor(0.2),
            'audio->vision': torch.tensor(0.3),
            'text->audio': torch.tensor(0.1),
            'audio->text': torch.tensor(0.15),
        }
        fig = hm.plot_cross_modal_attention(weights)
        assert_fig(fig, 'cross-modal')
    test('11_cross_modal', t11)

    # ---- 12: Cross-modal with tensor values ----
    def t12():
        weights = {
            'vision->text': torch.randn(4, 4).abs(),
            'text->vision': torch.randn(4, 4).abs(),
        }
        fig = hm.plot_cross_modal_attention(weights)
        assert_fig(fig, 'cross-modal tensors')
    test('12_cross_modal_tensor', t12)

    # ---- 13: Workspace competition basic ----
    def t13():
        scores = torch.tensor([0.5, 0.3, 0.15, 0.05])
        fig = hm.plot_workspace_competition(scores)
        assert_fig(fig, 'competition basic')
    test('13_competition_basic', t13)

    # ---- 14: Workspace competition with names ----
    def t14():
        scores = torch.tensor([0.4, 0.35, 0.2, 0.05])
        fig = hm.plot_workspace_competition(
            scores, modality_names=['Vision', 'Text', 'Audio', 'Sensor'])
        assert_fig(fig, 'competition named')
    test('14_competition_named', t14)

    # ---- 15: Competition threshold line ----
    def t15():
        scores = torch.tensor([0.6, 0.2, 0.1, 0.1])
        fig = hm.plot_workspace_competition(scores, threshold=0.3)
        ax = fig.axes[0]
        lines = ax.get_lines()
        has_threshold = any('Threshold' in str(l.get_label()) for l in lines)
        assert has_threshold, "Threshold line missing"
        plt.close(fig)
    test('15_competition_threshold', t15)

    # ---- 16: Competition winner highlight ----
    def t16():
        scores = torch.tensor([0.1, 0.8, 0.05, 0.05])
        fig = hm.plot_workspace_competition(scores, winner_idx=1)
        assert_fig(fig, 'competition winner')
    test('16_competition_winner', t16)

    # ---- 17: Competition evolution ----
    def t17():
        history = torch.tensor([
            [0.25, 0.25, 0.25, 0.25],
            [0.4, 0.3, 0.2, 0.1],
            [0.6, 0.2, 0.12, 0.08],
        ])
        fig = hm.plot_competition_evolution(history)
        assert_fig(fig, 'competition evolution')
    test('17_competition_evolution', t17)

    # ---- 18: Evolution axes ----
    def t18():
        history = torch.tensor([[0.5, 0.5], [0.7, 0.3]])
        fig = hm.plot_competition_evolution(history)
        ax = fig.axes[0]
        assert 'Round' in ax.get_xlabel(), f"X: {ax.get_xlabel()}"
        assert 'Score' in ax.get_ylabel(), f"Y: {ax.get_ylabel()}"
        plt.close(fig)
    test('18_evolution_axes', t18)

    # ---- 19: All-uniform attention ----
    def t19():
        w = torch.ones(6, 6) / 6.0
        fig = hm.plot_attention(w)
        assert_fig(fig, 'uniform attention')
    test('19_uniform_attention', t19)

    # ---- 20: Zero attention ----
    def t20():
        w = torch.zeros(5, 5)
        fig = hm.plot_attention(w)
        assert_fig(fig, 'zero attention')
    test('20_zero_attention', t20)

    # ---- 21: Deterministic rendering ----
    def t21():
        w = torch.softmax(torch.randn(6, 6), dim=-1)
        sizes = []
        for _ in range(2):
            fig = hm.plot_attention(w)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                fig.savefig(f.name, dpi=72)
                plt.close(fig)
                sizes.append(os.path.getsize(f.name))
                os.unlink(f.name)
        assert sizes[0] == sizes[1], f"Non-deterministic: {sizes}"
    test('21_deterministic', t21)

    # ---- 22: PNG valid header ----
    def t22():
        w = torch.softmax(torch.randn(4, 4), dim=-1)
        fig = hm.plot_attention(w)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            fig.savefig(f.name, dpi=72)
            plt.close(fig)
            with open(f.name, 'rb') as fp:
                header = fp.read(8)
            os.unlink(f.name)
        assert header[:4] == b'\x89PNG', "Invalid PNG"
    test('22_png_header', t22)

    # ---- 23: Numpy input ----
    def t23():
        w = np.random.rand(5, 5)
        w = w / w.sum(axis=-1, keepdims=True)
        fig = hm.plot_attention(w)
        assert_fig(fig, 'numpy input')
    test('23_numpy_input', t23)

    # ---- 24: Custom title ----
    def t24():
        w = torch.softmax(torch.randn(3, 3), dim=-1)
        fig = hm.plot_attention(w, title='My Custom Title')
        ax = fig.axes[0]
        assert 'My Custom Title' in ax.get_title()
        plt.close(fig)
    test('24_custom_title', t24)

    # ---- 25: Save figure ----
    def t25():
        w = torch.softmax(torch.randn(4, 4), dim=-1)
        fig = hm.plot_attention(w)
        with tempfile.TemporaryDirectory() as td:
            hm_local = AttentionHeatmapper(VizConfig(save_dir=td, dpi=72))
            path = hm_local.save_figure(fig, 'test_heatmap')
            assert os.path.exists(path)
            assert os.path.getsize(path) > 500
    test('25_save_figure', t25)

    # ---- 26: Single modality competition ----
    def t26():
        scores = torch.tensor([0.9])
        fig = hm.plot_workspace_competition(scores, modality_names=['vision'])
        assert_fig(fig, 'single modality competition')
    test('26_single_modality', t26)

    # ---- 27: Five modality competition ----
    def t27():
        scores = torch.tensor([0.3, 0.25, 0.2, 0.15, 0.1])
        fig = hm.plot_workspace_competition(
            scores, modality_names=['vision', 'text', 'audio', 'sensor', 'engram'])
        assert_fig(fig, 'five modality competition')
    test('27_five_modalities', t27)

    # Print
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{'='*60}")
    print(f"AttentionHeatmapper Self-Tests: {passed} passed, {failed} failed")
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
