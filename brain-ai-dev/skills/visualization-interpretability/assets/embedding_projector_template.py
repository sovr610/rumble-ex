"""
Embedding Projector Template for brain_ai.

Provides EmbeddingProjector class for dimensionality reduction and
visualization of workspace representations using t-SNE, PCA, and
optional UMAP projections.

Usage:
    from embedding_projector_template import EmbeddingProjector, VizConfig
    config = VizConfig()
    proj = EmbeddingProjector(config)
    fig = proj.plot_pca(embeddings, n_components=2)

Requires: matplotlib, numpy, torch
Optional: scikit-learn (for t-SNE); falls back to PCA if unavailable.
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
import warnings
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Union

# Optional imports with fallback
_HAS_SKLEARN = False
try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA as SklearnPCA
    _HAS_SKLEARN = True
except ImportError:
    pass


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

PHASE_COLORS = ['#1f77b4', '#2196f3', '#4caf50', '#ff9800',
                '#f44336', '#9c27b0', '#e91e63']


# ---------------------------------------------------------------------------
# Numpy PCA fallback
# ---------------------------------------------------------------------------

def _pca_numpy(data: np.ndarray, n_components: int = 2
               ) -> Tuple[np.ndarray, np.ndarray]:
    """Pure numpy PCA implementation (fallback when sklearn unavailable).

    Args:
        data: 2D array (n_samples, n_features).
        n_components: Number of components.

    Returns:
        (projected, explained_variance_ratio)
    """
    n_samples, n_features = data.shape
    n_components = min(n_components, n_features, n_samples)

    # Center data
    mean = np.mean(data, axis=0)
    centered = data - mean

    # SVD-based PCA
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    components = Vt[:n_components]
    projected = centered @ components.T

    # Explained variance
    total_var = np.sum(S ** 2) / max(n_samples - 1, 1)
    explained_var = (S[:n_components] ** 2) / max(n_samples - 1, 1)
    if total_var > 0:
        explained_ratio = explained_var / total_var
    else:
        explained_ratio = np.zeros(n_components)

    return projected, explained_ratio


# ---------------------------------------------------------------------------
# EmbeddingProjector
# ---------------------------------------------------------------------------

class EmbeddingProjector:
    """Dimensionality reduction and visualization for workspace representations.

    Supports t-SNE (via sklearn with PCA fallback), PCA, and basic
    UMAP if the umap-learn package is available.

    Args:
        config: VizConfig instance controlling appearance.
    """

    def __init__(self, config: Optional[VizConfig] = None):
        self.config = config or VizConfig()

    def _to_numpy(self, tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().float().numpy()
        return np.asarray(tensor, dtype=float)

    def _subsample(self, data: np.ndarray,
                   labels: Optional[np.ndarray] = None
                   ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Subsample to max_points if needed."""
        n = data.shape[0]
        if n <= self.config.max_points:
            return data, labels
        idx = np.random.choice(n, self.config.max_points, replace=False)
        idx.sort()
        sub_data = data[idx]
        sub_labels = labels[idx] if labels is not None else None
        return sub_data, sub_labels

    def _prepare_labels(self, labels: Optional[Union[torch.Tensor, np.ndarray, List]],
                        n: int) -> Optional[np.ndarray]:
        """Convert labels to numpy array."""
        if labels is None:
            return None
        if isinstance(labels, torch.Tensor):
            return labels.detach().cpu().numpy().ravel()
        return np.asarray(labels).ravel()

    def _scatter_with_labels(
        self,
        ax: plt.Axes,
        points: np.ndarray,
        labels: Optional[np.ndarray],
        xlabel: str,
        ylabel: str,
        title: str,
    ) -> plt.Figure:
        """Shared scatter plot logic."""
        if labels is not None and len(np.unique(labels)) <= 20:
            unique = np.unique(labels)
            cmap = plt.cm.tab20 if len(unique) > 10 else plt.cm.tab10
            for i, lbl in enumerate(unique):
                mask = labels == lbl
                color = cmap(i / max(len(unique), 1))
                ax.scatter(points[mask, 0], points[mask, 1],
                           c=[color], s=15, alpha=0.7, label=f'Class {lbl}',
                           edgecolors='none', rasterized=(mask.sum() > 1000))
            ax.legend(loc='best', fontsize=7, ncol=2, markerscale=2)
        elif labels is not None:
            sc = ax.scatter(points[:, 0], points[:, 1], c=labels,
                            cmap=self.config.colormap, s=15, alpha=0.7,
                            edgecolors='none',
                            rasterized=(len(points) > 1000))
            plt.colorbar(sc, ax=ax, label='Label', shrink=0.8)
        else:
            ax.scatter(points[:, 0], points[:, 1], c='steelblue',
                       s=15, alpha=0.7, edgecolors='none',
                       rasterized=(len(points) > 1000))

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        return ax.get_figure()

    # -- Public API ----------------------------------------------------------

    def plot_pca(
        self,
        embeddings: Union[torch.Tensor, np.ndarray],
        labels: Optional[Union[torch.Tensor, np.ndarray, List]] = None,
        n_components: int = 2,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Project embeddings via PCA and create a scatter plot.

        Args:
            embeddings: 2D tensor (n_samples, n_features).
            labels: Optional labels for coloring.
            n_components: Number of PCA components (2 or 3, plots 2D).
            title: Custom title.

        Returns:
            matplotlib Figure.
        """
        data = self._to_numpy(embeddings)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        assert data.ndim == 2, f"Expected 2D, got {data.ndim}D"

        label_arr = self._prepare_labels(labels, data.shape[0])
        data, label_arr = self._subsample(data, label_arr)

        # Handle degenerate case
        n_samples, n_features = data.shape
        if n_samples < 2 or n_features < 2:
            fig, ax = plt.subplots(figsize=self.config.figsize)
            if n_samples == 1:
                ax.scatter([0], [0], c='steelblue', s=50)
                ax.set_title(title or 'PCA Projection (single point)')
            else:
                ax.text(0.5, 0.5, 'Insufficient dimensions for PCA',
                        transform=ax.transAxes, ha='center', fontsize=12)
                ax.set_title(title or 'PCA Projection')
            ax.set_xlabel('PC 1')
            ax.set_ylabel('PC 2')
            fig.tight_layout()
            return fig

        # Replace NaN
        data = np.nan_to_num(data, nan=0.0)

        # Compute PCA
        if _HAS_SKLEARN:
            pca = SklearnPCA(n_components=min(n_components, n_features, n_samples))
            projected = pca.fit_transform(data)
            explained = pca.explained_variance_ratio_
        else:
            projected, explained = _pca_numpy(data, n_components)

        fig, ax = plt.subplots(figsize=self.config.figsize)

        # Build subtitle with variance info
        var_info = ', '.join([f'PC{i+1}: {v*100:.1f}%' for i, v in enumerate(explained[:2])])
        full_title = title or 'PCA Projection'
        full_title += f'\n({var_info})'

        self._scatter_with_labels(ax, projected[:, :2], label_arr,
                                  'PC 1', 'PC 2', full_title)

        fig.tight_layout()
        return fig

    def plot_tsne(
        self,
        embeddings: Union[torch.Tensor, np.ndarray],
        labels: Optional[Union[torch.Tensor, np.ndarray, List]] = None,
        perplexity: Optional[int] = None,
        random_state: int = 42,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Project embeddings via t-SNE and create a scatter plot.

        Falls back to PCA if sklearn is not available.

        Args:
            embeddings: 2D tensor (n_samples, n_features).
            labels: Optional labels for coloring.
            perplexity: t-SNE perplexity (defaults to config.tsne_perplexity).
            random_state: Random seed for reproducibility.
            title: Custom title.

        Returns:
            matplotlib Figure.
        """
        data = self._to_numpy(embeddings)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        assert data.ndim == 2, f"Expected 2D, got {data.ndim}D"

        label_arr = self._prepare_labels(labels, data.shape[0])
        data, label_arr = self._subsample(data, label_arr)

        # Replace NaN
        data = np.nan_to_num(data, nan=0.0)

        n_samples, n_features = data.shape
        perplexity = perplexity or self.config.tsne_perplexity

        # Need at least perplexity + 1 samples for t-SNE
        if not _HAS_SKLEARN or n_samples < max(perplexity + 1, 4):
            if not _HAS_SKLEARN:
                warnings.warn("sklearn not available; falling back to PCA for t-SNE")
            else:
                warnings.warn(f"Too few samples ({n_samples}) for t-SNE "
                              f"perplexity={perplexity}; falling back to PCA")
            return self.plot_pca(embeddings, labels,
                                title=title or 't-SNE (PCA fallback)')

        # Run t-SNE (sklearn >= 1.6 renamed n_iter to max_iter)
        tsne_kwargs = dict(
            n_components=2,
            perplexity=min(perplexity, n_samples - 1),
            random_state=random_state,
        )
        import inspect
        tsne_params = inspect.signature(TSNE.__init__).parameters
        if 'max_iter' in tsne_params:
            tsne_kwargs['max_iter'] = 500
        elif 'n_iter' in tsne_params:
            tsne_kwargs['n_iter'] = 500
        tsne = TSNE(**tsne_kwargs)
        projected = tsne.fit_transform(data)

        fig, ax = plt.subplots(figsize=self.config.figsize)
        self._scatter_with_labels(ax, projected, label_arr,
                                  't-SNE 1', 't-SNE 2',
                                  title or 't-SNE Projection')
        fig.tight_layout()
        return fig

    def plot_pca_variance(
        self,
        embeddings: Union[torch.Tensor, np.ndarray],
        max_components: int = 20,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Plot PCA explained variance (scree plot).

        Args:
            embeddings: 2D tensor (n_samples, n_features).
            max_components: Maximum number of components to show.
            title: Custom title.

        Returns:
            matplotlib Figure.
        """
        data = self._to_numpy(embeddings)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        data = np.nan_to_num(data, nan=0.0)
        n_samples, n_features = data.shape
        n_comp = min(max_components, n_features, n_samples)

        if _HAS_SKLEARN:
            pca = SklearnPCA(n_components=n_comp)
            pca.fit(data)
            explained = pca.explained_variance_ratio_
        else:
            _, explained = _pca_numpy(data, n_comp)

        cumulative = np.cumsum(explained)

        fig, ax = plt.subplots(figsize=self.config.figsize)
        components = np.arange(1, len(explained) + 1)

        ax.bar(components, explained, alpha=0.7, color='steelblue',
               label='Individual', edgecolor='white')
        ax.plot(components, cumulative, '-o', color='#d62728', linewidth=2,
                markersize=5, label='Cumulative')
        ax.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5,
                   label='95% threshold')

        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Explained Variance Ratio')
        ax.set_title(title or 'PCA Explained Variance')
        ax.set_xlim(0.5, len(explained) + 0.5)
        ax.set_ylim(0, 1.05)
        ax.legend(loc='center right', fontsize=9)
        fig.tight_layout()
        return fig

    def plot_embedding_norms(
        self,
        embeddings: Union[torch.Tensor, np.ndarray],
        labels: Optional[Union[torch.Tensor, np.ndarray, List]] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Plot histogram of embedding L2 norms.

        Args:
            embeddings: 2D tensor (n_samples, n_features).
            labels: Optional labels for group coloring.
            title: Custom title.

        Returns:
            matplotlib Figure.
        """
        data = self._to_numpy(embeddings)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        norms = np.linalg.norm(data, axis=1)

        fig, ax = plt.subplots(figsize=self.config.figsize)

        if labels is not None:
            label_arr = self._prepare_labels(labels, data.shape[0])
            unique = np.unique(label_arr)
            if len(unique) <= 10:
                cmap = plt.cm.tab10
                for i, lbl in enumerate(unique):
                    mask = label_arr == lbl
                    ax.hist(norms[mask], bins=30, alpha=0.6,
                            color=cmap(i / max(len(unique), 1)),
                            label=f'Class {lbl}', edgecolor='white')
                ax.legend(fontsize=8)
            else:
                ax.hist(norms, bins=30, color='steelblue', edgecolor='white',
                        alpha=0.85)
        else:
            ax.hist(norms, bins=30, color='steelblue', edgecolor='white',
                    alpha=0.85)

        mean_norm = np.mean(norms)
        ax.axvline(x=mean_norm, color='red', linestyle='--', linewidth=2,
                   label=f'Mean ({mean_norm:.2f})')
        ax.set_xlabel('L2 Norm')
        ax.set_ylabel('Count')
        ax.set_title(title or 'Embedding Norm Distribution')
        ax.legend(loc='upper right', fontsize=9)
        fig.tight_layout()
        return fig

    def plot_cosine_similarity_matrix(
        self,
        embeddings: Union[torch.Tensor, np.ndarray],
        labels: Optional[List[str]] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Plot pairwise cosine similarity matrix.

        Args:
            embeddings: 2D tensor (n_samples, n_features).
            labels: Optional labels for axis annotations.
            title: Custom title.

        Returns:
            matplotlib Figure.
        """
        data = self._to_numpy(embeddings)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        # Subsample for visualization
        n = data.shape[0]
        max_display = 50
        if n > max_display:
            idx = np.linspace(0, n - 1, max_display, dtype=int)
            data = data[idx]
            if labels is not None:
                labels = [labels[i] for i in idx]
            n = max_display

        # Cosine similarity
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normed = data / norms
        sim = normed @ normed.T

        fig, ax = plt.subplots(figsize=(max(6, n * 0.3), max(5, n * 0.25)))
        im = ax.imshow(sim, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

        if labels is not None and n <= 30:
            ax.set_xticks(range(n))
            ax.set_xticklabels(labels[:n], rotation=45, ha='right', fontsize=7)
            ax.set_yticks(range(n))
            ax.set_yticklabels(labels[:n], fontsize=7)

        ax.set_title(title or 'Cosine Similarity Matrix')
        fig.colorbar(im, ax=ax, label='Cosine Similarity', shrink=0.8)
        fig.tight_layout()
        return fig

    def plot_nearest_neighbors(
        self,
        embeddings: Union[torch.Tensor, np.ndarray],
        query_idx: int = 0,
        k: int = 10,
        labels: Optional[List[str]] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Plot nearest neighbors of a query embedding in PCA space.

        Args:
            embeddings: 2D tensor (n_samples, n_features).
            query_idx: Index of the query point.
            k: Number of nearest neighbors.
            labels: Optional point labels.
            title: Custom title.

        Returns:
            matplotlib Figure.
        """
        data = self._to_numpy(embeddings)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        data = np.nan_to_num(data, nan=0.0)

        n = data.shape[0]
        k = min(k, n - 1)

        # Compute distances
        query = data[query_idx]
        dists = np.linalg.norm(data - query, axis=1)
        nn_indices = np.argsort(dists)[1:k + 1]  # exclude self

        # PCA for visualization
        if _HAS_SKLEARN:
            pca = SklearnPCA(n_components=2)
            proj = pca.fit_transform(data)
        else:
            proj, _ = _pca_numpy(data, 2)

        fig, ax = plt.subplots(figsize=self.config.figsize)

        # All points gray
        ax.scatter(proj[:, 0], proj[:, 1], c='#cccccc', s=10, alpha=0.4,
                   edgecolors='none', rasterized=(n > 1000))

        # Neighbors highlighted
        ax.scatter(proj[nn_indices, 0], proj[nn_indices, 1],
                   c='#ff7f0e', s=40, alpha=0.9, edgecolors='black',
                   linewidths=0.5, label=f'Top-{k} NN', zorder=4)

        # Query point
        ax.scatter(proj[query_idx, 0], proj[query_idx, 1],
                   c='#d62728', s=100, marker='*', edgecolors='black',
                   linewidths=1, label='Query', zorder=5)

        # Draw lines from query to neighbors
        for ni in nn_indices:
            ax.plot([proj[query_idx, 0], proj[ni, 0]],
                    [proj[query_idx, 1], proj[ni, 1]],
                    'k-', alpha=0.15, linewidth=0.5)

        query_label = labels[query_idx] if labels and query_idx < len(labels) else f'#{query_idx}'
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_title(title or f'Nearest Neighbors of {query_label}')
        ax.legend(loc='upper right', fontsize=9)
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

    config = VizConfig(figsize=(8, 5), dpi=72, max_points=200)
    proj = EmbeddingProjector(config)

    def assert_fig(fig, label=''):
        assert fig is not None, f"Figure is None ({label})"
        assert len(fig.axes) > 0, f"No axes ({label})"
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            fig.savefig(f.name, dpi=72)
            plt.close(fig)
            size = os.path.getsize(f.name)
            os.unlink(f.name)
        assert size > 500, f"PNG too small: {size} ({label})"

    # ---- 1: Basic PCA ----
    def t01():
        emb = torch.randn(100, 64)
        fig = proj.plot_pca(emb)
        assert_fig(fig, 'pca basic')
    test('01_pca_basic', t01)

    # ---- 2: PCA with labels ----
    def t02():
        emb = torch.randn(100, 64)
        labels = torch.randint(0, 5, (100,))
        fig = proj.plot_pca(emb, labels=labels)
        assert_fig(fig, 'pca labels')
    test('02_pca_labels', t02)

    # ---- 3: PCA axes ----
    def t03():
        emb = torch.randn(50, 32)
        fig = proj.plot_pca(emb)
        ax = fig.axes[0]
        assert 'PC' in ax.get_xlabel(), f"X: {ax.get_xlabel()}"
        assert 'PC' in ax.get_ylabel(), f"Y: {ax.get_ylabel()}"
        plt.close(fig)
    test('03_pca_axes', t03)

    # ---- 4: PCA single point ----
    def t04():
        emb = torch.randn(1, 32)
        fig = proj.plot_pca(emb)
        assert_fig(fig, 'pca single')
    test('04_pca_single', t04)

    # ---- 5: PCA all zeros ----
    def t05():
        emb = torch.zeros(50, 32)
        fig = proj.plot_pca(emb)
        assert_fig(fig, 'pca zeros')
    test('05_pca_zeros', t05)

    # ---- 6: PCA NaN handling ----
    def t06():
        emb = torch.randn(50, 32)
        emb[10, 5] = float('nan')
        fig = proj.plot_pca(emb)
        assert_fig(fig, 'pca nan')
    test('06_pca_nan', t06)

    # ---- 7: PCA numpy input ----
    def t07():
        emb = np.random.randn(80, 32)
        fig = proj.plot_pca(emb)
        assert_fig(fig, 'pca numpy')
    test('07_pca_numpy', t07)

    # ---- 8: PCA variance info in title ----
    def t08():
        emb = torch.randn(100, 64)
        fig = proj.plot_pca(emb)
        ax = fig.axes[0]
        title = ax.get_title()
        assert 'PC1' in title or '%' in title, f"No variance info: {title}"
        plt.close(fig)
    test('08_pca_variance_title', t08)

    # ---- 9: t-SNE basic ----
    def t09():
        emb = torch.randn(80, 32)
        fig = proj.plot_tsne(emb, perplexity=10)
        assert_fig(fig, 'tsne basic')
    test('09_tsne_basic', t09)

    # ---- 10: t-SNE with labels ----
    def t10():
        emb = torch.randn(80, 32)
        labels = torch.randint(0, 3, (80,))
        fig = proj.plot_tsne(emb, labels=labels, perplexity=10)
        assert_fig(fig, 'tsne labels')
    test('10_tsne_labels', t10)

    # ---- 11: t-SNE axes ----
    def t11():
        emb = torch.randn(50, 16)
        fig = proj.plot_tsne(emb, perplexity=5)
        ax = fig.axes[0]
        xl = ax.get_xlabel()
        assert 't-SNE' in xl or 'PC' in xl, f"X: {xl}"
        plt.close(fig)
    test('11_tsne_axes', t11)

    # ---- 12: t-SNE too few samples falls back to PCA ----
    def t12():
        emb = torch.randn(3, 32)
        fig = proj.plot_tsne(emb, perplexity=30)
        assert_fig(fig, 'tsne fallback')
    test('12_tsne_fallback', t12)

    # ---- 13: PCA scree plot ----
    def t13():
        emb = torch.randn(100, 64)
        fig = proj.plot_pca_variance(emb, max_components=10)
        assert_fig(fig, 'pca scree')
    test('13_pca_scree', t13)

    # ---- 14: Embedding norms ----
    def t14():
        emb = torch.randn(100, 64)
        fig = proj.plot_embedding_norms(emb)
        assert_fig(fig, 'norms')
    test('14_norms', t14)

    # ---- 15: Embedding norms with labels ----
    def t15():
        emb = torch.randn(100, 64)
        labels = torch.randint(0, 3, (100,))
        fig = proj.plot_embedding_norms(emb, labels=labels)
        assert_fig(fig, 'norms labels')
    test('15_norms_labels', t15)

    # ---- 16: Cosine similarity matrix ----
    def t16():
        emb = torch.randn(20, 32)
        fig = proj.plot_cosine_similarity_matrix(emb)
        assert_fig(fig, 'cosine sim')
    test('16_cosine_sim', t16)

    # ---- 17: Cosine similarity with labels ----
    def t17():
        emb = torch.randn(10, 32)
        labels = [f'P{i}' for i in range(10)]
        fig = proj.plot_cosine_similarity_matrix(emb, labels=labels)
        assert_fig(fig, 'cosine sim labels')
    test('17_cosine_sim_labels', t17)

    # ---- 18: Nearest neighbors ----
    def t18():
        emb = torch.randn(50, 32)
        fig = proj.plot_nearest_neighbors(emb, query_idx=0, k=5)
        assert_fig(fig, 'nearest neighbors')
    test('18_nearest_neighbors', t18)

    # ---- 19: Large embedding subsampling ----
    def t19():
        emb = torch.randn(500, 64)
        fig = proj.plot_pca(emb)
        assert_fig(fig, 'large subsampled')
    test('19_large_subsample', t19)

    # ---- 20: Custom title PCA ----
    def t20():
        emb = torch.randn(50, 32)
        fig = proj.plot_pca(emb, title='My PCA Plot')
        ax = fig.axes[0]
        assert 'My PCA Plot' in ax.get_title()
        plt.close(fig)
    test('20_custom_title', t20)

    # ---- 21: Deterministic PCA ----
    def t21():
        emb = torch.randn(50, 32)
        sizes = []
        for _ in range(2):
            fig = proj.plot_pca(emb)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                fig.savefig(f.name, dpi=72)
                plt.close(fig)
                sizes.append(os.path.getsize(f.name))
                os.unlink(f.name)
        assert sizes[0] == sizes[1], f"Non-deterministic: {sizes}"
    test('21_deterministic_pca', t21)

    # ---- 22: PNG header valid ----
    def t22():
        emb = torch.randn(50, 32)
        fig = proj.plot_pca(emb)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            fig.savefig(f.name, dpi=72)
            plt.close(fig)
            with open(f.name, 'rb') as fp:
                header = fp.read(8)
            os.unlink(f.name)
        assert header[:4] == b'\x89PNG', "Invalid PNG"
    test('22_png_header', t22)

    # ---- 23: Save figure ----
    def t23():
        emb = torch.randn(50, 32)
        fig = proj.plot_pca(emb)
        with tempfile.TemporaryDirectory() as td:
            proj_local = EmbeddingProjector(VizConfig(save_dir=td, dpi=72))
            path = proj_local.save_figure(fig, 'test_emb')
            assert os.path.exists(path)
            assert os.path.getsize(path) > 500
    test('23_save_figure', t23)

    # ---- 24: PCA many labels (continuous) ----
    def t24():
        emb = torch.randn(100, 32)
        labels = torch.arange(100).float()
        fig = proj.plot_pca(emb, labels=labels)
        assert_fig(fig, 'pca continuous labels')
    test('24_continuous_labels', t24)

    # ---- 25: t-SNE deterministic with seed ----
    def t25():
        emb = torch.randn(50, 16)
        sizes = []
        for _ in range(2):
            fig = proj.plot_tsne(emb, perplexity=5, random_state=42)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                fig.savefig(f.name, dpi=72)
                plt.close(fig)
                sizes.append(os.path.getsize(f.name))
                os.unlink(f.name)
        assert sizes[0] == sizes[1], f"Non-deterministic t-SNE: {sizes}"
    test('25_tsne_deterministic', t25)

    # ---- 26: PCA high-dim input ----
    def t26():
        emb = torch.randn(50, 4096)
        fig = proj.plot_pca(emb)
        assert_fig(fig, 'pca high dim')
    test('26_pca_high_dim', t26)

    # ---- 27: Nearest neighbors with labels ----
    def t27():
        emb = torch.randn(30, 16)
        labels = [f'Sample_{i}' for i in range(30)]
        fig = proj.plot_nearest_neighbors(emb, query_idx=5, k=3, labels=labels)
        assert_fig(fig, 'nn labels')
    test('27_nn_labels', t27)

    # Print
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{'='*60}")
    print(f"EmbeddingProjector Self-Tests: {passed} passed, {failed} failed")
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
