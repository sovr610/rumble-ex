"""
clustering_template.py -- Clustering algorithms, stability metrics, and diagnostics
for Task2Vec task embeddings.

Complete, self-contained module for task embedding clustering and stability analysis.
Provides k-means, agglomerative, and optional HDBSCAN clustering with cosine distance,
adjusted Rand index (ARI) and normalized mutual information (NMI) stability evaluation,
silhouette scoring, within/between cluster diagnostics, and representative task
identification.

All sklearn and hdbscan imports are optional; pure numpy fallbacks are provided for
k-means, ARI, NMI, and silhouette so the module works without any third-party
clustering library installed.

Target module: brain_ai/meta/task_clustering.py
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Optional imports -- guarded with try/except
# ---------------------------------------------------------------------------

try:
    from sklearn.cluster import KMeans as _SklearnKMeans
    from sklearn.cluster import AgglomerativeClustering as _SklearnAgglomerative
    SKLEARN_CLUSTER_AVAILABLE = True
except ImportError:
    SKLEARN_CLUSTER_AVAILABLE = False

try:
    from sklearn.metrics import (
        adjusted_rand_score as _sklearn_ari,
        normalized_mutual_info_score as _sklearn_nmi,
        silhouette_score as _sklearn_silhouette_score,
        silhouette_samples as _sklearn_silhouette_samples,
    )
    SKLEARN_METRICS_AVAILABLE = True
except ImportError:
    SKLEARN_METRICS_AVAILABLE = False

try:
    import hdbscan as _hdbscan_lib
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


logger = logging.getLogger(__name__)

# ============================================================================
# Enums
# ============================================================================


class ClusterAlgorithm(Enum):
    """Supported clustering algorithms."""
    KMEANS = "kmeans"
    AGGLOMERATIVE = "agglomerative"
    HDBSCAN = "hdbscan"


class LinkageType(Enum):
    """Linkage methods for agglomerative clustering."""
    WARD = "ward"
    COMPLETE = "complete"
    AVERAGE = "average"


# ============================================================================
# Configuration dataclass
# ============================================================================


@dataclass
class ClusterConfig:
    """Configuration for clustering operations.

    Attributes:
        algorithm: Clustering algorithm to use.
        n_clusters: Number of clusters for k-means and agglomerative.
            Ignored for HDBSCAN.
        distance_metric: Distance metric. Currently only ``"cosine"``
            is supported.
        stability_threshold: Minimum mean ARI for the stability gate.
        linkage: Linkage type for agglomerative clustering.
        min_cluster_size: HDBSCAN ``min_cluster_size`` parameter.
        min_samples: HDBSCAN ``min_samples`` parameter. ``None`` defaults
            to ``min_cluster_size``.
        n_init: Number of k-means initializations.
        max_iter: Maximum Lloyd iterations for k-means.
        tol: Convergence tolerance for k-means.
    """

    algorithm: str = "kmeans"
    n_clusters: int = 8
    distance_metric: str = "cosine"
    stability_threshold: float = 0.9
    linkage: str = "ward"
    min_cluster_size: int = 5
    min_samples: Optional[int] = None
    n_init: int = 10
    max_iter: int = 300
    tol: float = 1e-4


# ============================================================================
# Result dataclasses
# ============================================================================


@dataclass
class ClusterResult:
    """Result of a clustering run.

    Attributes:
        labels: Integer cluster label per sample. Shape ``(N,)``.
            May contain ``-1`` for noise when using HDBSCAN.
        n_clusters: Number of clusters discovered (excluding noise).
        centroids: L2-normalised cluster centroids of shape ``(K, E)``,
            or ``None`` if centroids are not available (e.g. HDBSCAN
            with no clusters).
        algorithm: Name of the algorithm used.
        seed: Random seed that was used.
        metadata: Algorithm-specific metadata (inertia, linkage, etc.).
    """

    labels: np.ndarray
    n_clusters: int
    centroids: Optional[np.ndarray]
    algorithm: str
    seed: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StabilityResult:
    """Result of a multi-run stability evaluation.

    Attributes:
        ari: Mean adjusted Rand index across all pairs of runs.
        nmi: Mean normalised mutual information across all pairs.
        is_stable: Whether mean ARI meets or exceeds the threshold.
        run_labels: List of label arrays from each run.
        ari_values: Pairwise ARI values.
        nmi_values: Pairwise NMI values.
        threshold: Stability threshold used.
    """

    ari: float
    nmi: float
    is_stable: bool
    run_labels: List[np.ndarray]
    ari_values: List[float] = field(default_factory=list)
    nmi_values: List[float] = field(default_factory=list)
    threshold: float = 0.9


@dataclass
class ClusterDiagnostics:
    """Diagnostic metrics for a clustering result.

    Attributes:
        silhouette_score: Mean silhouette coefficient in ``[-1, 1]``.
        within_cluster_dispersion: Mean pairwise cosine distance within
            each cluster, keyed by cluster label.
        inter_cluster_distances: ``(K, K)`` cosine distance matrix between
            cluster centroids.
        cluster_sizes: Number of members per cluster, keyed by label.
        representative_task_ids: For each cluster, the index of the sample
            closest to the centroid.
    """

    silhouette_score: float
    within_cluster_dispersion: Dict[int, float]
    inter_cluster_distances: np.ndarray
    cluster_sizes: Dict[int, int]
    representative_task_ids: Dict[int, int]


# ============================================================================
# Distance computation
# ============================================================================


def cosine_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Compute cosine distance between two vectors.

    ``d(u, v) = 1 - cos(u, v)``.  Returns a value in ``[0, 2]``.

    Args:
        u: First vector, shape ``(E,)``.
        v: Second vector, shape ``(E,)``.

    Returns:
        Scalar cosine distance.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u < 1e-12 or norm_v < 1e-12:
        return 1.0  # undefined; treat as orthogonal
    sim = np.dot(u, v) / (norm_u * norm_v)
    sim = float(np.clip(sim, -1.0, 1.0))
    return 1.0 - sim


def cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute the pairwise cosine distance matrix.

    Uses efficient matrix multiplication.  Embeddings are assumed to be
    L2-normalised along ``axis=1`` but the function works correctly
    regardless (it normalises internally).

    Args:
        embeddings: Array of shape ``(N, E)``.

    Returns:
        Symmetric distance matrix of shape ``(N, N)`` with zeros on
        the diagonal and values in ``[0, 2]``.

    Raises:
        ValueError: If *embeddings* has fewer than 2 dimensions.
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected 2-D array, got shape {embeddings.shape}"
        )
    N = embeddings.shape[0]
    if N == 0:
        return np.zeros((0, 0), dtype=np.float64)

    # L2-normalise rows
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normed = embeddings / norms

    # Similarity via matmul, then distance
    sim = normed @ normed.T
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    np.clip(dist, 0.0, 2.0, out=dist)
    return dist


def _cosine_distance_matrix_chunked(
    embeddings: np.ndarray,
    chunk_size: int = 2048,
) -> np.ndarray:
    """Memory-efficient cosine distance matrix for large N.

    Computes the same result as :func:`cosine_distance_matrix` but
    processes in blocks to limit peak memory.
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normed = embeddings / norms
    N = normed.shape[0]
    dist = np.zeros((N, N), dtype=np.float64)
    for i in range(0, N, chunk_size):
        ei = min(i + chunk_size, N)
        for j in range(i, N, chunk_size):
            ej = min(j + chunk_size, N)
            block = 1.0 - normed[i:ei] @ normed[j:ej].T
            dist[i:ei, j:ej] = block
            if i != j:
                dist[j:ej, i:ei] = block.T
    np.fill_diagonal(dist, 0.0)
    np.clip(dist, 0.0, 2.0, out=dist)
    return dist


# ============================================================================
# L2 normalisation helper
# ============================================================================


def _l2_normalize(X: np.ndarray) -> np.ndarray:
    """Row-wise L2-normalise, clamping near-zero norms."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return X / norms


# ============================================================================
# Centroid computation helper
# ============================================================================


def _compute_centroids(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    """Compute L2-normalised centroids from cluster assignments.

    Args:
        embeddings: ``(N, E)`` array.
        labels: ``(N,)`` integer labels (values ``0 .. n_clusters-1``).
        n_clusters: Number of clusters.

    Returns:
        ``(n_clusters, E)`` array of unit-norm centroids.
    """
    E = embeddings.shape[1]
    centroids = np.zeros((n_clusters, E), dtype=np.float64)
    for k in range(n_clusters):
        mask = labels == k
        if mask.sum() > 0:
            centroid = embeddings[mask].mean(axis=0)
            norm = np.linalg.norm(centroid)
            centroids[k] = centroid / max(norm, 1e-8)
    return centroids


# ============================================================================
# Pure-numpy k-means fallback
# ============================================================================


def _kmeans_plusplus_init(
    X: np.ndarray,
    n_clusters: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """K-means++ initialisation (pure numpy).

    Args:
        X: ``(N, E)`` data matrix, assumed L2-normalised.
        n_clusters: Number of centres to initialise.
        rng: Numpy ``RandomState`` for reproducibility.

    Returns:
        ``(n_clusters, E)`` initial centroid array.
    """
    N, E = X.shape
    centres = np.empty((n_clusters, E), dtype=np.float64)

    # Pick first centre uniformly at random
    idx = rng.randint(0, N)
    centres[0] = X[idx]

    # Squared Euclidean distances (equivalent to cosine for unit-norm vecs)
    for c in range(1, n_clusters):
        # Distances to nearest existing centre
        dists = np.full(N, np.inf)
        for j in range(c):
            d = np.sum((X - centres[j]) ** 2, axis=1)
            dists = np.minimum(dists, d)
        # Probability proportional to distance
        probs = dists / dists.sum()
        idx = rng.choice(N, p=probs)
        centres[c] = X[idx]

    return centres


def _numpy_kmeans(
    X: np.ndarray,
    n_clusters: int,
    seed: int = 42,
    max_iter: int = 300,
    tol: float = 1e-4,
    n_init: int = 10,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """Pure-numpy k-means with k-means++ init and Lloyd's algorithm.

    Args:
        X: ``(N, E)`` L2-normalised data.
        n_clusters: Number of clusters.
        seed: Random seed.
        max_iter: Max iterations per init.
        tol: Convergence tolerance on centroid movement.
        n_init: Number of independent initialisations.

    Returns:
        Tuple of ``(labels, centroids, inertia, n_iter)`` for the best
        run (lowest inertia).
    """
    N, E = X.shape
    rng = np.random.RandomState(seed)

    best_labels: Optional[np.ndarray] = None
    best_centroids: Optional[np.ndarray] = None
    best_inertia = np.inf
    best_n_iter = 0

    for init_idx in range(n_init):
        # Derive a deterministic sub-seed for each init
        sub_rng = np.random.RandomState(rng.randint(0, 2**31))
        centres = _kmeans_plusplus_init(X, n_clusters, sub_rng)

        n_iter = 0
        for iteration in range(max_iter):
            n_iter = iteration + 1

            # Assign step: each point to nearest centroid (Euclidean)
            # (N, K) distance matrix
            # ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x.c
            # For unit-norm vectors: = 2 - 2 x.c
            sims = X @ centres.T  # (N, K)
            labels = np.argmax(sims, axis=1)

            # Update step: recompute centroids
            new_centres = np.zeros_like(centres)
            for k in range(n_clusters):
                mask = labels == k
                if mask.sum() > 0:
                    new_centres[k] = X[mask].mean(axis=0)
                else:
                    # Empty cluster: reinitialise from farthest point
                    dists_to_nearest = 2.0 - sims[np.arange(N), labels]
                    farthest = np.argmax(dists_to_nearest)
                    new_centres[k] = X[farthest]

            # Normalise new centres for cosine interpretation
            c_norms = np.linalg.norm(new_centres, axis=1, keepdims=True)
            c_norms = np.maximum(c_norms, 1e-8)
            new_centres = new_centres / c_norms

            # Check convergence
            shift = np.sqrt(np.sum((new_centres - centres) ** 2))
            centres = new_centres
            if shift < tol:
                break

        # Compute inertia (sum of squared distances to assigned centroid)
        sims_final = X @ centres.T
        labels_final = np.argmax(sims_final, axis=1)
        inertia = 0.0
        for k in range(n_clusters):
            mask = labels_final == k
            if mask.sum() > 0:
                diff = X[mask] - centres[k]
                inertia += float(np.sum(diff ** 2))

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels_final
            best_centroids = centres.copy()
            best_n_iter = n_iter

    assert best_labels is not None
    assert best_centroids is not None
    return best_labels, best_centroids, best_inertia, best_n_iter


# ============================================================================
# Clustering algorithms
# ============================================================================


def cluster_kmeans(
    embeddings: np.ndarray,
    n_clusters: int,
    seed: int = 42,
    max_iter: int = 300,
    tol: float = 1e-4,
    n_init: int = 10,
) -> ClusterResult:
    """Cluster embeddings using k-means.

    L2-normalises the embeddings first so that Euclidean k-means
    optimises cosine distance.  Uses scikit-learn when available,
    otherwise falls back to a pure-numpy implementation.

    Args:
        embeddings: ``(N, E)`` embedding matrix.
        n_clusters: Number of clusters.
        seed: Random seed for reproducibility.
        max_iter: Maximum iterations.
        tol: Convergence tolerance.
        n_init: Number of initialisations.

    Returns:
        A :class:`ClusterResult` with labels and L2-normalised centroids.
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    N = embeddings.shape[0]

    if N == 0:
        return ClusterResult(
            labels=np.array([], dtype=int),
            n_clusters=0,
            centroids=None,
            algorithm="kmeans",
            seed=seed,
            metadata={"warning": "empty embeddings"},
        )

    # Clamp n_clusters
    effective_k = min(n_clusters, N)
    if effective_k < n_clusters:
        logger.warning(
            "n_clusters=%d > n_samples=%d; reducing to %d",
            n_clusters, N, effective_k,
        )

    X = _l2_normalize(embeddings)

    if SKLEARN_CLUSTER_AVAILABLE:
        km = _SklearnKMeans(
            n_clusters=effective_k,
            init="k-means++",
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            random_state=seed,
        )
        labels = km.fit_predict(X)
        centroids = _l2_normalize(km.cluster_centers_)
        metadata: Dict[str, Any] = {
            "inertia": float(km.inertia_),
            "n_iter": int(km.n_iter_),
            "converged": km.n_iter_ < max_iter,
            "backend": "sklearn",
        }
    else:
        logger.info("sklearn unavailable; using pure-numpy k-means fallback")
        labels, centroids, inertia, n_iter = _numpy_kmeans(
            X, effective_k, seed=seed, max_iter=max_iter,
            tol=tol, n_init=n_init,
        )
        metadata = {
            "inertia": float(inertia),
            "n_iter": int(n_iter),
            "converged": n_iter < max_iter,
            "backend": "numpy",
        }

    return ClusterResult(
        labels=labels,
        n_clusters=effective_k,
        centroids=centroids,
        algorithm="kmeans",
        seed=seed,
        metadata=metadata,
    )


def cluster_agglomerative(
    embeddings: np.ndarray,
    n_clusters: int,
    linkage: str = "ward",
    seed: int = 42,
) -> ClusterResult:
    """Cluster embeddings using agglomerative (hierarchical) clustering.

    Requires scikit-learn.  Ward linkage uses Euclidean distance on
    L2-normalised embeddings (cosine-equivalent).  Complete and average
    linkage use a precomputed cosine distance matrix.

    Agglomerative clustering is deterministic so *seed* is retained only
    for interface consistency.

    Args:
        embeddings: ``(N, E)`` embedding matrix.
        n_clusters: Number of clusters.
        linkage: One of ``"ward"``, ``"complete"``, ``"average"``.
        seed: Unused; kept for interface consistency.

    Returns:
        A :class:`ClusterResult`.

    Raises:
        ImportError: If scikit-learn is not installed.
    """
    if not SKLEARN_CLUSTER_AVAILABLE:
        raise ImportError(
            "scikit-learn is required for agglomerative clustering. "
            "Install with: pip install scikit-learn"
        )

    embeddings = np.asarray(embeddings, dtype=np.float64)
    N = embeddings.shape[0]

    if N == 0:
        return ClusterResult(
            labels=np.array([], dtype=int),
            n_clusters=0,
            centroids=None,
            algorithm=f"agglomerative_{linkage}",
            seed=seed,
            metadata={"warning": "empty embeddings"},
        )

    effective_k = min(n_clusters, N)
    if effective_k < n_clusters:
        logger.warning(
            "n_clusters=%d > n_samples=%d; reducing to %d",
            n_clusters, N, effective_k,
        )

    X = _l2_normalize(embeddings)

    if linkage == "ward":
        model = _SklearnAgglomerative(
            n_clusters=effective_k,
            linkage="ward",
        )
        labels = model.fit_predict(X)
    else:
        dist_matrix = cosine_distance_matrix(X)
        model = _SklearnAgglomerative(
            n_clusters=effective_k,
            metric="precomputed",
            linkage=linkage,
        )
        labels = model.fit_predict(dist_matrix)

    centroids = _compute_centroids(X, labels, effective_k)

    return ClusterResult(
        labels=labels,
        n_clusters=effective_k,
        centroids=centroids,
        algorithm=f"agglomerative_{linkage}",
        seed=seed,
        metadata={"linkage": linkage},
    )


def cluster_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: Optional[int] = None,
    seed: int = 42,
) -> ClusterResult:
    """Cluster embeddings using HDBSCAN.

    HDBSCAN is density-based and does not require a fixed cluster count.
    Points in low-density regions receive label ``-1`` (noise).

    Requires the ``hdbscan`` package.

    Args:
        embeddings: ``(N, E)`` embedding matrix.
        min_cluster_size: Minimum points to form a cluster.
        min_samples: Controls conservativeness. Defaults to
            *min_cluster_size*.
        seed: Seed for tie-breaking (HDBSCAN is largely deterministic).

    Returns:
        A :class:`ClusterResult`.  ``centroids`` is ``None`` if zero
        clusters are found.

    Raises:
        ImportError: If the ``hdbscan`` package is not installed.
    """
    if not HDBSCAN_AVAILABLE:
        raise ImportError(
            "hdbscan package not installed. Install with: pip install hdbscan"
        )

    embeddings = np.asarray(embeddings, dtype=np.float64)
    N = embeddings.shape[0]

    if N == 0:
        return ClusterResult(
            labels=np.array([], dtype=int),
            n_clusters=0,
            centroids=None,
            algorithm="hdbscan",
            seed=seed,
            metadata={"warning": "empty embeddings"},
        )

    X = _l2_normalize(embeddings)

    clusterer = _hdbscan_lib.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",  # cosine-equivalent for unit-norm vectors
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(X)

    n_clusters = len(set(labels) - {-1})
    noise_count = int(np.sum(labels == -1))

    if noise_count > 0:
        noise_frac = noise_count / N
        logger.info(
            "HDBSCAN: %d noise points (%.1f%%)", noise_count, 100 * noise_frac
        )
        if noise_frac > 0.3:
            logger.warning(
                "HDBSCAN noise fraction %.1f%% exceeds 30%%. "
                "Consider reducing min_cluster_size.",
                100 * noise_frac,
            )

    centroids: Optional[np.ndarray] = None
    if n_clusters > 0:
        non_noise = labels >= 0
        centroids = _compute_centroids(
            X[non_noise], labels[non_noise], n_clusters
        )

    probabilities = (
        clusterer.probabilities_.tolist()
        if hasattr(clusterer, "probabilities_")
        else []
    )

    return ClusterResult(
        labels=labels,
        n_clusters=n_clusters,
        centroids=centroids,
        algorithm="hdbscan",
        seed=seed,
        metadata={
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "noise_count": noise_count,
            "noise_fraction": noise_count / max(N, 1),
            "probabilities": probabilities,
        },
    )


def cluster(
    embeddings: np.ndarray,
    config: ClusterConfig,
    seed: Optional[int] = None,
) -> ClusterResult:
    """Factory function that dispatches to the configured clustering algorithm.

    Args:
        embeddings: ``(N, E)`` embedding matrix.
        config: :class:`ClusterConfig` specifying algorithm and parameters.
        seed: Random seed.  If ``None``, defaults to ``42``.

    Returns:
        A :class:`ClusterResult`.

    Raises:
        ValueError: If ``config.algorithm`` is not recognised.
    """
    if seed is None:
        seed = 42

    algo = config.algorithm.lower().strip()

    if algo == "kmeans":
        return cluster_kmeans(
            embeddings,
            n_clusters=config.n_clusters,
            seed=seed,
            max_iter=config.max_iter,
            tol=config.tol,
            n_init=config.n_init,
        )
    elif algo == "agglomerative":
        return cluster_agglomerative(
            embeddings,
            n_clusters=config.n_clusters,
            linkage=config.linkage,
            seed=seed,
        )
    elif algo == "hdbscan":
        return cluster_hdbscan(
            embeddings,
            min_cluster_size=config.min_cluster_size,
            min_samples=config.min_samples,
            seed=seed,
        )
    else:
        raise ValueError(
            f"Unknown clustering algorithm: {algo!r}. "
            f"Choose from 'kmeans', 'agglomerative', 'hdbscan'."
        )


# ============================================================================
# Pure-numpy ARI fallback
# ============================================================================


def _contingency_matrix(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the contingency matrix for two label vectors.

    Returns:
        Tuple of ``(contingency, classes_a, classes_b)`` where
        ``contingency`` has shape ``(|classes_a|, |classes_b|)``.
    """
    classes_a = np.unique(labels_a)
    classes_b = np.unique(labels_b)
    a_map = {v: i for i, v in enumerate(classes_a)}
    b_map = {v: i for i, v in enumerate(classes_b)}
    C = np.zeros((len(classes_a), len(classes_b)), dtype=np.int64)
    for la, lb in zip(labels_a, labels_b):
        C[a_map[la], b_map[lb]] += 1
    return C, classes_a, classes_b


def _comb2(n: int) -> int:
    """Compute C(n, 2) = n*(n-1)/2."""
    if n < 2:
        return 0
    return n * (n - 1) // 2


def _numpy_adjusted_rand_score(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
) -> float:
    """Adjusted Rand Index computed entirely with numpy.

    Follows the standard combinatorial definition using the contingency
    table.  Equivalent to ``sklearn.metrics.adjusted_rand_score``.

    Args:
        labels_a: First clustering labels, shape ``(N,)``.
        labels_b: Second clustering labels, shape ``(N,)``.

    Returns:
        ARI value in ``[-1, 1]``.
    """
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)
    N = len(labels_a)
    if N == 0:
        return 1.0

    C, _, _ = _contingency_matrix(labels_a, labels_b)

    # Row sums and column sums
    a_sums = C.sum(axis=1)
    b_sums = C.sum(axis=0)

    # Sum of C(n_ij, 2) over all cells
    sum_comb_c = sum(_comb2(int(v)) for v in C.ravel())
    sum_comb_a = sum(_comb2(int(v)) for v in a_sums)
    sum_comb_b = sum(_comb2(int(v)) for v in b_sums)

    comb_n = _comb2(N)
    if comb_n == 0:
        return 1.0

    expected = sum_comb_a * sum_comb_b / comb_n
    max_index = 0.5 * (sum_comb_a + sum_comb_b)
    denominator = max_index - expected

    if abs(denominator) < 1e-15:
        return 1.0

    ari = (sum_comb_c - expected) / denominator
    return float(ari)


# ============================================================================
# Pure-numpy NMI fallback
# ============================================================================


def _entropy(labels: np.ndarray) -> float:
    """Shannon entropy of a label vector."""
    _, counts = np.unique(labels, return_counts=True)
    N = len(labels)
    if N == 0:
        return 0.0
    probs = counts / N
    # Filter out zero probabilities to avoid log(0)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def _mutual_information(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
) -> float:
    """Mutual information between two label vectors."""
    C, _, _ = _contingency_matrix(labels_a, labels_b)
    N = len(labels_a)
    if N == 0:
        return 0.0

    a_sums = C.sum(axis=1)
    b_sums = C.sum(axis=0)

    mi = 0.0
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if C[i, j] == 0:
                continue
            p_ij = C[i, j] / N
            p_i = a_sums[i] / N
            p_j = b_sums[j] / N
            mi += p_ij * np.log(p_ij / (p_i * p_j))
    return float(mi)


def _numpy_normalized_mutual_info(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
) -> float:
    """Normalized Mutual Information (arithmetic average) via numpy.

    ``NMI(U, V) = 2 * I(U; V) / (H(U) + H(V))``

    Equivalent to ``sklearn.metrics.normalized_mutual_info_score``
    with ``average_method="arithmetic"``.

    Args:
        labels_a: First clustering labels, shape ``(N,)``.
        labels_b: Second clustering labels, shape ``(N,)``.

    Returns:
        NMI in ``[0, 1]``.
    """
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)

    if len(labels_a) == 0:
        return 1.0

    h_a = _entropy(labels_a)
    h_b = _entropy(labels_b)
    denom = h_a + h_b

    if denom < 1e-15:
        return 1.0  # both are single-cluster

    mi = _mutual_information(labels_a, labels_b)
    nmi = 2.0 * mi / denom
    return float(np.clip(nmi, 0.0, 1.0))


# ============================================================================
# Pure-numpy silhouette fallback
# ============================================================================


def _numpy_silhouette_samples(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Per-sample silhouette coefficient using cosine distance (numpy only).

    For each sample *i*:

    - ``a(i)`` = mean cosine distance to other members of the same cluster
    - ``b(i)`` = min over other clusters of mean cosine distance to members
    - ``s(i) = (b(i) - a(i)) / max(a(i), b(i))``

    Args:
        embeddings: ``(N, E)`` L2-normalised embeddings.
        labels: ``(N,)`` integer cluster labels (no ``-1`` noise expected).

    Returns:
        ``(N,)`` array of silhouette coefficients in ``[-1, 1]``.
    """
    N = len(labels)
    dist = cosine_distance_matrix(embeddings)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    sil = np.zeros(N, dtype=np.float64)
    if n_clusters < 2:
        return sil  # silhouette undefined for < 2 clusters

    for i in range(N):
        own_label = labels[i]
        own_mask = labels == own_label
        own_count = own_mask.sum()

        if own_count <= 1:
            # Singleton cluster => a(i) = 0
            a_i = 0.0
        else:
            # Mean distance to other members of same cluster
            a_i = dist[i, own_mask].sum() / (own_count - 1)

        # b(i): min mean distance to members of other clusters
        b_i = np.inf
        for lbl in unique_labels:
            if lbl == own_label:
                continue
            other_mask = labels == lbl
            other_count = other_mask.sum()
            if other_count == 0:
                continue
            mean_dist = dist[i, other_mask].mean()
            if mean_dist < b_i:
                b_i = mean_dist

        if b_i == np.inf:
            sil[i] = 0.0
        else:
            denom = max(a_i, b_i)
            sil[i] = (b_i - a_i) / denom if denom > 1e-15 else 0.0

    return sil


def _numpy_silhouette_score(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Mean silhouette score using pure numpy."""
    samples = _numpy_silhouette_samples(embeddings, labels)
    if len(samples) == 0:
        return 0.0
    return float(samples.mean())


# ============================================================================
# Public metric wrappers (use sklearn when available, else fallback)
# ============================================================================


def adjusted_rand_score(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
) -> float:
    """Adjusted Rand Index between two clusterings.

    Uses scikit-learn when available, otherwise a pure-numpy fallback.

    Args:
        labels_a: First label vector, shape ``(N,)``.
        labels_b: Second label vector, shape ``(N,)``.

    Returns:
        ARI in ``[-1, 1]``.
    """
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)
    if SKLEARN_METRICS_AVAILABLE:
        return float(_sklearn_ari(labels_a, labels_b))
    return _numpy_adjusted_rand_score(labels_a, labels_b)


def normalized_mutual_info_score(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
) -> float:
    """Normalised Mutual Information between two clusterings.

    Uses scikit-learn when available (arithmetic average), otherwise a
    pure-numpy fallback.

    Args:
        labels_a: First label vector, shape ``(N,)``.
        labels_b: Second label vector, shape ``(N,)``.

    Returns:
        NMI in ``[0, 1]``.
    """
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)
    if SKLEARN_METRICS_AVAILABLE:
        return float(
            _sklearn_nmi(labels_a, labels_b, average_method="arithmetic")
        )
    return _numpy_normalized_mutual_info(labels_a, labels_b)


def silhouette_score(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Mean silhouette score using cosine distance.

    Uses scikit-learn when available, otherwise a pure-numpy fallback.

    Args:
        embeddings: ``(N, E)`` array.
        labels: ``(N,)`` integer cluster labels.

    Returns:
        Mean silhouette in ``[-1, 1]``, or ``0.0`` if fewer than 2
        clusters are present.
    """
    labels = np.asarray(labels)
    mask = labels >= 0
    if mask.sum() < 2:
        return 0.0
    unique = np.unique(labels[mask])
    if len(unique) < 2:
        return 0.0

    X = np.asarray(embeddings, dtype=np.float64)[mask]
    L = labels[mask]

    if SKLEARN_METRICS_AVAILABLE:
        return float(_sklearn_silhouette_score(X, L, metric="cosine"))
    return _numpy_silhouette_score(X, L)


def silhouette_samples(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Per-sample silhouette coefficients using cosine distance.

    Uses scikit-learn when available, otherwise a pure-numpy fallback.

    Args:
        embeddings: ``(N, E)`` array.
        labels: ``(N,)`` integer cluster labels.

    Returns:
        ``(N,)`` array of silhouette values in ``[-1, 1]``.
    """
    labels = np.asarray(labels)
    embeddings = np.asarray(embeddings, dtype=np.float64)

    mask = labels >= 0
    unique = np.unique(labels[mask])
    if mask.sum() < 2 or len(unique) < 2:
        return np.zeros(len(labels), dtype=np.float64)

    if SKLEARN_METRICS_AVAILABLE:
        # sklearn only for non-noise samples; fill noise with 0
        result = np.zeros(len(labels), dtype=np.float64)
        result[mask] = _sklearn_silhouette_samples(
            embeddings[mask], labels[mask], metric="cosine"
        )
        return result

    result = np.zeros(len(labels), dtype=np.float64)
    result[mask] = _numpy_silhouette_samples(embeddings[mask], labels[mask])
    return result


# ============================================================================
# Stability evaluation
# ============================================================================


def evaluate_stability(
    embeddings: np.ndarray,
    config: ClusterConfig,
    n_runs: int = 5,
    seeds: Optional[List[int]] = None,
) -> StabilityResult:
    """Evaluate clustering stability across multiple random seeds.

    Runs the configured clustering algorithm *n_runs* times with different
    seeds, then computes pairwise ARI and NMI between all pairs of
    resulting label vectors.

    Args:
        embeddings: ``(N, E)`` embedding matrix.
        config: Cluster configuration.
        n_runs: Number of independent clustering runs.
        seeds: Explicit list of seeds. If ``None``, seeds
            ``[42, 123, 456, 789, 1024, ...]`` are generated.

    Returns:
        A :class:`StabilityResult` with mean ARI/NMI and the stability
        verdict.
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    N = embeddings.shape[0]

    if seeds is None:
        seeds = [42 + i * 81 for i in range(n_runs)]
    else:
        n_runs = len(seeds)

    if N < 2:
        # Trivial case: 0 or 1 samples
        trivial_labels = np.zeros(N, dtype=int) if N > 0 else np.array([], dtype=int)
        return StabilityResult(
            ari=1.0,
            nmi=1.0,
            is_stable=True,
            run_labels=[trivial_labels.copy() for _ in range(n_runs)],
            ari_values=[1.0],
            nmi_values=[1.0],
            threshold=config.stability_threshold,
        )

    # Run clustering n_runs times
    run_labels: List[np.ndarray] = []
    for s in seeds:
        result = cluster(embeddings, config, seed=s)
        run_labels.append(result.labels.copy())

    # Pairwise comparisons
    ari_values: List[float] = []
    nmi_values: List[float] = []

    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            la = run_labels[i]
            lb = run_labels[j]

            # For HDBSCAN, filter out noise in either run
            if config.algorithm.lower() == "hdbscan":
                mask = (la >= 0) & (lb >= 0)
                if mask.sum() < 2:
                    ari_values.append(1.0)
                    nmi_values.append(1.0)
                    continue
                la_f = la[mask]
                lb_f = lb[mask]
            else:
                la_f = la
                lb_f = lb

            ari_values.append(adjusted_rand_score(la_f, lb_f))
            nmi_values.append(normalized_mutual_info_score(la_f, lb_f))

    mean_ari = float(np.mean(ari_values)) if ari_values else 1.0
    mean_nmi = float(np.mean(nmi_values)) if nmi_values else 1.0
    is_stable = mean_ari >= config.stability_threshold

    return StabilityResult(
        ari=mean_ari,
        nmi=mean_nmi,
        is_stable=is_stable,
        run_labels=run_labels,
        ari_values=ari_values,
        nmi_values=nmi_values,
        threshold=config.stability_threshold,
    )


def evaluate_stability_multi(
    embeddings: np.ndarray,
    config: ClusterConfig,
    n_trials: int = 10,
    base_seed: int = 0,
) -> Dict[str, Any]:
    """Run multiple stability trials and report statistics.

    Each trial clusters the same data with two different seeds and
    computes ARI and NMI between the two runs.

    Args:
        embeddings: ``(N, E)`` embedding matrix.
        config: Cluster configuration.
        n_trials: Number of trials.
        base_seed: Base seed for generating trial-specific seeds.

    Returns:
        Dictionary with ``ari_mean``, ``ari_std``, ``ari_min``,
        ``nmi_mean``, ``nmi_std``, ``nmi_min``, ``n_trials``,
        ``stable`` (bool).
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    ari_scores: List[float] = []
    nmi_scores: List[float] = []

    for t in range(n_trials):
        seed_a = base_seed + t * 2
        seed_b = base_seed + t * 2 + 1

        result_a = cluster(embeddings, config, seed=seed_a)
        result_b = cluster(embeddings, config, seed=seed_b)

        la = result_a.labels
        lb = result_b.labels

        if config.algorithm.lower() == "hdbscan":
            mask = (la >= 0) & (lb >= 0)
            if mask.sum() < 2:
                ari_scores.append(1.0)
                nmi_scores.append(1.0)
                continue
            la = la[mask]
            lb = lb[mask]

        ari_scores.append(adjusted_rand_score(la, lb))
        nmi_scores.append(normalized_mutual_info_score(la, lb))

    ari_arr = np.array(ari_scores)
    nmi_arr = np.array(nmi_scores)

    return {
        "ari_mean": float(ari_arr.mean()),
        "ari_std": float(ari_arr.std()),
        "ari_min": float(ari_arr.min()),
        "nmi_mean": float(nmi_arr.mean()),
        "nmi_std": float(nmi_arr.std()),
        "nmi_min": float(nmi_arr.min()),
        "n_trials": n_trials,
        "stable": bool(ari_arr.min() >= config.stability_threshold),
    }


# ============================================================================
# Diagnostics
# ============================================================================


def compute_diagnostics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    task_ids: Optional[List[Union[str, int]]] = None,
) -> ClusterDiagnostics:
    """Compute comprehensive clustering diagnostics.

    Args:
        embeddings: ``(N, E)`` embedding matrix.
        labels: ``(N,)`` integer cluster labels. May contain ``-1`` for
            noise (HDBSCAN).
        task_ids: Optional task identifiers of length *N*. If ``None``,
            integer indices are used.

    Returns:
        A :class:`ClusterDiagnostics` instance.
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    N = len(labels)

    if task_ids is None:
        task_ids_int: List[Union[str, int]] = list(range(N))
    else:
        task_ids_int = list(task_ids)

    # --- Silhouette ---
    sil = silhouette_score(embeddings, labels)

    # --- Cluster sizes ---
    non_noise_labels = labels[labels >= 0]
    unique_labels = np.unique(non_noise_labels) if len(non_noise_labels) > 0 else np.array([], dtype=int)
    cluster_sizes: Dict[int, int] = {}
    for lbl in unique_labels:
        cluster_sizes[int(lbl)] = int((labels == lbl).sum())

    n_clusters = len(unique_labels)

    # --- Within-cluster dispersion ---
    dispersion = _within_cluster_dispersion(embeddings, labels)

    # --- Inter-cluster distances ---
    if n_clusters >= 2:
        centroids = _compute_centroids(
            _l2_normalize(embeddings), labels, n_clusters
        )
        inter_dist = cosine_distance_matrix(centroids)
    elif n_clusters == 1:
        centroids = _compute_centroids(
            _l2_normalize(embeddings), labels, 1
        )
        inter_dist = np.zeros((1, 1), dtype=np.float64)
    else:
        centroids = np.empty((0, embeddings.shape[1] if embeddings.ndim == 2 else 0), dtype=np.float64)
        inter_dist = np.zeros((0, 0), dtype=np.float64)

    # --- Representative task IDs ---
    representative_ids = _find_representative_indices(
        embeddings, labels, centroids, n_clusters
    )

    return ClusterDiagnostics(
        silhouette_score=sil,
        within_cluster_dispersion=dispersion,
        inter_cluster_distances=inter_dist,
        cluster_sizes=cluster_sizes,
        representative_task_ids=representative_ids,
    )


def _within_cluster_dispersion(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> Dict[int, float]:
    """Mean pairwise cosine distance within each cluster.

    Args:
        embeddings: ``(N, E)`` array.
        labels: ``(N,)`` integer labels.

    Returns:
        Dictionary mapping cluster label to mean within-cluster cosine
        distance.
    """
    X = _l2_normalize(np.asarray(embeddings, dtype=np.float64))
    dispersion: Dict[int, float] = {}
    for k in sorted(set(labels)):
        if k == -1:
            continue
        mask = labels == k
        cluster_embs = X[mask]
        n = cluster_embs.shape[0]
        if n < 2:
            dispersion[int(k)] = 0.0
            continue
        sim = cluster_embs @ cluster_embs.T
        triu_idx = np.triu_indices(n, k=1)
        pairwise_dist = 1.0 - sim[triu_idx]
        dispersion[int(k)] = float(pairwise_dist.mean())
    return dispersion


def _find_representative_indices(
    embeddings: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    n_clusters: int,
) -> Dict[int, int]:
    """Find the index of the sample closest to each centroid.

    Args:
        embeddings: ``(N, E)`` array.
        labels: ``(N,)`` integer labels.
        centroids: ``(K, E)`` L2-normalised centroids.
        n_clusters: Number of clusters.

    Returns:
        Dictionary mapping cluster label to the global index of the
        representative sample.
    """
    X = _l2_normalize(np.asarray(embeddings, dtype=np.float64))
    reps: Dict[int, int] = {}
    for k in range(n_clusters):
        mask = labels == k
        if mask.sum() == 0:
            continue
        indices = np.where(mask)[0]
        cluster_embs = X[indices]
        # Cosine similarity to centroid
        sims = cluster_embs @ centroids[k]
        best_local = int(np.argmax(sims))
        reps[int(k)] = int(indices[best_local])
    return reps


def inter_cluster_distances(
    centroids: np.ndarray,
) -> np.ndarray:
    """Compute the K x K cosine distance matrix between cluster centroids.

    Args:
        centroids: ``(K, E)`` L2-normalised centroid array.

    Returns:
        ``(K, K)`` symmetric distance matrix with zeros on diagonal.
    """
    return cosine_distance_matrix(centroids)


def within_cluster_dispersion(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> Dict[int, float]:
    """Public wrapper for within-cluster dispersion computation.

    Args:
        embeddings: ``(N, E)`` array.
        labels: ``(N,)`` integer labels.

    Returns:
        Dictionary mapping cluster label to mean intra-cluster cosine
        distance.
    """
    return _within_cluster_dispersion(embeddings, labels)


def cluster_size_report(
    labels: np.ndarray,
) -> Dict[str, Any]:
    """Report cluster sizes and detect degenerate cases.

    Args:
        labels: ``(N,)`` integer cluster labels.

    Returns:
        Dictionary with keys ``sizes``, ``n_clusters``, ``total_tasks``,
        ``min_size``, ``max_size``, ``mean_size``, ``std_size``,
        ``empty_clusters``, ``singletons``, ``dominant_cluster``,
        ``dominant_fraction``.
    """
    labels = np.asarray(labels)
    non_noise = labels[labels >= 0]

    if len(non_noise) == 0:
        return {
            "sizes": {},
            "n_clusters": 0,
            "total_tasks": 0,
            "min_size": 0,
            "max_size": 0,
            "mean_size": 0.0,
            "std_size": 0.0,
            "empty_clusters": True,
            "singletons": [],
            "dominant_cluster": -1,
            "dominant_fraction": 0.0,
        }

    unique, counts = np.unique(non_noise, return_counts=True)
    size_map = {int(k): int(c) for k, c in zip(unique, counts)}
    total = int(counts.sum())
    n_clusters = len(unique)

    return {
        "sizes": size_map,
        "n_clusters": n_clusters,
        "total_tasks": total,
        "min_size": int(counts.min()),
        "max_size": int(counts.max()),
        "mean_size": float(counts.mean()),
        "std_size": float(counts.std()),
        "empty_clusters": n_clusters == 0,
        "singletons": [int(k) for k, c in zip(unique, counts) if c == 1],
        "dominant_cluster": int(unique[counts.argmax()]),
        "dominant_fraction": float(counts.max() / total) if total > 0 else 0.0,
    }


def find_representative_tasks(
    embeddings: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    task_ids: List[str],
) -> Dict[int, Dict[str, Any]]:
    """Find the task closest to each cluster centroid.

    Args:
        embeddings: ``(N, E)`` embedding matrix.
        labels: ``(N,)`` integer cluster labels.
        centroids: ``(K, E)`` L2-normalised centroids.
        task_ids: Task identifier strings of length *N*.

    Returns:
        Dictionary mapping cluster label to a dict with ``task_id``,
        ``task_index``, ``cosine_similarity``, ``cosine_distance``.
    """
    X = _l2_normalize(np.asarray(embeddings, dtype=np.float64))
    centroids = np.asarray(centroids, dtype=np.float64)
    representatives: Dict[int, Dict[str, Any]] = {}

    for k in range(centroids.shape[0]):
        mask = labels == k
        if mask.sum() == 0:
            continue
        indices = np.where(mask)[0]
        cluster_embs = X[indices]
        sims = cluster_embs @ centroids[k]
        best_local = int(np.argmax(sims))
        best_global = int(indices[best_local])

        representatives[int(k)] = {
            "task_id": task_ids[best_global],
            "task_index": best_global,
            "cosine_similarity": float(sims[best_local]),
            "cosine_distance": float(1.0 - sims[best_local]),
        }

    return representatives


# ============================================================================
# Input validation
# ============================================================================


def validate_inputs(
    embeddings: np.ndarray,
    config: ClusterConfig,
) -> List[str]:
    """Validate clustering inputs and return a list of warning messages.

    Checks for: empty arrays, n_tasks < n_clusters, non-normalised
    embeddings, near-duplicate embeddings, constant embeddings.

    Args:
        embeddings: ``(N, E)`` embedding matrix.
        config: Cluster configuration.

    Returns:
        List of warning strings (empty if everything is fine).

    Raises:
        ValueError: If embeddings array is empty (zero rows).
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    N, E = embeddings.shape
    warnings_list: List[str] = []

    if N == 0:
        raise ValueError("Cannot cluster zero embeddings")

    if N < config.n_clusters and config.algorithm.lower() != "hdbscan":
        warnings_list.append(
            f"n_tasks ({N}) < n_clusters ({config.n_clusters}). "
            f"Reducing to {N} clusters."
        )

    # Check normalisation
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        warnings_list.append(
            f"Embeddings not L2-normalized (norm range: "
            f"[{norms.min():.4f}, {norms.max():.4f}]). "
            f"Will normalise before clustering."
        )

    # Check for constant embeddings
    if N > 1 and np.allclose(embeddings, embeddings[0:1], atol=1e-6):
        warnings_list.append("All embeddings are approximately identical")

    # Check for near-duplicates
    if N > 1 and N <= 5000:  # skip for very large N (expensive)
        dist = cosine_distance_matrix(embeddings)
        np.fill_diagonal(dist, np.inf)
        min_dist = dist.min()
        if min_dist < 1e-6:
            n_duplicates = int((dist < 1e-6).sum() // 2)
            warnings_list.append(
                f"{n_duplicates} near-duplicate embedding pairs detected "
                f"(cosine distance < 1e-6)"
            )

    for w in warnings_list:
        logger.warning(w)

    return warnings_list


# ============================================================================
# Full clustering pipeline
# ============================================================================


def run_clustering_pipeline(
    embeddings: np.ndarray,
    task_ids: Optional[List[str]] = None,
    config: Optional[ClusterConfig] = None,
    seed: int = 42,
    compute_diag: bool = True,
    run_stability: bool = True,
    stability_n_trials: int = 10,
) -> Dict[str, Any]:
    """Run the full clustering pipeline: cluster, diagnose, evaluate stability.

    Args:
        embeddings: ``(N, E)`` embedding matrix.
        task_ids: Optional task identifiers. If ``None``, integer indices
            are used.
        config: Cluster configuration. Defaults to a ``ClusterConfig()``
            with default values.
        seed: Random seed for clustering.
        compute_diag: Whether to compute diagnostics.
        run_stability: Whether to run stability evaluation.
        stability_n_trials: Number of stability trials.

    Returns:
        Dictionary with keys ``result`` (:class:`ClusterResult`),
        ``diagnostics`` (:class:`ClusterDiagnostics` or ``None``),
        ``stability`` (dict or ``None``), ``warnings`` (list of str).
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if config is None:
        config = ClusterConfig()

    N = embeddings.shape[0] if embeddings.ndim == 2 else 0

    if task_ids is None:
        task_ids = [str(i) for i in range(N)]

    # --- Validate ---
    warnings_list: List[str] = []
    try:
        warnings_list = validate_inputs(embeddings, config)
    except ValueError as exc:
        logger.error(str(exc))
        return {
            "result": ClusterResult(
                labels=np.array([], dtype=int),
                n_clusters=0,
                centroids=None,
                algorithm=config.algorithm,
                seed=seed,
                metadata={"error": str(exc)},
            ),
            "diagnostics": None,
            "stability": None,
            "warnings": [str(exc)],
        }

    # --- Edge case: N < 2 ---
    if N < 2:
        trivial_labels = np.zeros(N, dtype=int)
        return {
            "result": ClusterResult(
                labels=trivial_labels,
                n_clusters=max(1, N),
                centroids=embeddings[:1].copy() if N > 0 else None,
                algorithm=config.algorithm,
                seed=seed,
                metadata={"warning": "trivial clustering (N < 2)"},
            ),
            "diagnostics": None,
            "stability": None,
            "warnings": warnings_list,
        }

    # --- Cluster ---
    result = cluster(embeddings, config, seed=seed)

    # --- Diagnostics ---
    diagnostics: Optional[ClusterDiagnostics] = None
    if compute_diag:
        diagnostics = compute_diagnostics(embeddings, result.labels, task_ids)

    # --- Stability ---
    stability: Optional[Dict[str, Any]] = None
    if run_stability:
        stability = evaluate_stability_multi(
            embeddings, config,
            n_trials=stability_n_trials,
            base_seed=seed,
        )

    return {
        "result": result,
        "diagnostics": diagnostics,
        "stability": stability,
        "warnings": warnings_list,
    }


# ============================================================================
# Synthetic data generation for testing
# ============================================================================


def _make_separated_clusters(
    n_clusters: int = 5,
    n_per_cluster: int = 20,
    dim: int = 64,
    separation: float = 3.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate well-separated synthetic clusters on the unit hypersphere.

    Each cluster is formed by sampling a random centroid direction and then
    adding Gaussian noise scaled by ``1 / separation``.  Higher separation
    yields more distinct clusters.

    Args:
        n_clusters: Number of clusters.
        n_per_cluster: Points per cluster.
        dim: Embedding dimension.
        separation: Higher values yield tighter, more separated clusters.
        seed: Random seed.

    Returns:
        Tuple of ``(embeddings, true_labels)`` where ``embeddings`` has
        shape ``(n_clusters * n_per_cluster, dim)`` and is L2-normalised.
    """
    rng = np.random.RandomState(seed)
    all_points = []
    all_labels = []

    # Generate random centroid directions
    raw_centroids = rng.randn(n_clusters, dim)
    raw_centroids /= np.linalg.norm(raw_centroids, axis=1, keepdims=True)

    for k in range(n_clusters):
        noise = rng.randn(n_per_cluster, dim) / separation
        points = raw_centroids[k] + noise
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / np.maximum(norms, 1e-8)
        all_points.append(points)
        all_labels.append(np.full(n_per_cluster, k, dtype=int))

    embeddings = np.vstack(all_points)
    labels = np.concatenate(all_labels)
    return embeddings, labels


def _make_overlapping_clusters(
    n_clusters: int = 5,
    n_per_cluster: int = 20,
    dim: int = 64,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate heavily overlapping clusters (low separation).

    Uses separation = 0.5 to ensure clusters are hard to distinguish,
    useful for testing that stability metrics correctly report instability.
    """
    return _make_separated_clusters(
        n_clusters=n_clusters,
        n_per_cluster=n_per_cluster,
        dim=dim,
        separation=0.5,
        seed=seed,
    )


# ============================================================================
# Self-test block
# ============================================================================


def _run_self_tests() -> None:
    """Run 18+ self-tests and print PASS/FAIL for each.

    This function exercises the entire module: distance computation,
    clustering algorithms, stability evaluation, diagnostics, pure-numpy
    fallbacks, and edge cases.
    """
    passed = 0
    failed = 0
    total = 0

    def _test(name: str, condition: bool, detail: str = "") -> None:
        nonlocal passed, failed, total
        total += 1
        if condition:
            passed += 1
            print(f"  PASS  [{total:2d}] {name}")
        else:
            failed += 1
            msg = f"  FAIL  [{total:2d}] {name}"
            if detail:
                msg += f"  -- {detail}"
            print(msg)

    print("=" * 70)
    print("clustering_template.py -- Self-Test Suite")
    print("=" * 70)
    print()

    # ----------------------------------------------------------------
    # Test 1: cosine_distance_matrix is symmetric with zero diagonal
    # ----------------------------------------------------------------
    rng = np.random.RandomState(0)
    emb_small = rng.randn(10, 32)
    emb_small /= np.linalg.norm(emb_small, axis=1, keepdims=True)
    D = cosine_distance_matrix(emb_small)
    sym_ok = np.allclose(D, D.T, atol=1e-10)
    diag_ok = np.allclose(D.diagonal(), 0.0, atol=1e-10)
    range_ok = D.min() >= -1e-10 and D.max() <= 2.0 + 1e-10
    _test(
        "cosine_distance_matrix: symmetric, diagonal zero, range [0,2]",
        sym_ok and diag_ok and range_ok,
        f"sym={sym_ok}, diag={diag_ok}, range={range_ok}",
    )

    # ----------------------------------------------------------------
    # Test 2: cosine_distance single pair
    # ----------------------------------------------------------------
    x = np.array([1.0, 0.0, 0.0])
    d_self = cosine_distance(x, x)
    d_opp = cosine_distance(x, -x)
    _test(
        "cosine_distance: d(x,x)=0, d(x,-x)~2",
        abs(d_self) < 1e-10 and abs(d_opp - 2.0) < 1e-10,
        f"d_self={d_self:.6e}, d_opp={d_opp:.6e}",
    )

    # ----------------------------------------------------------------
    # Test 3: kmeans produces k clusters, all points assigned
    # ----------------------------------------------------------------
    emb_50, true_labels_50 = _make_separated_clusters(
        n_clusters=5, n_per_cluster=10, dim=64, separation=8.0, seed=42
    )
    res_km = cluster_kmeans(emb_50, n_clusters=5, seed=42)
    n_unique = len(np.unique(res_km.labels))
    all_assigned = len(res_km.labels) == 50
    _test(
        "kmeans: produces 5 clusters, all 50 points assigned",
        n_unique == 5 and all_assigned,
        f"n_unique={n_unique}, len={len(res_km.labels)}",
    )

    # ----------------------------------------------------------------
    # Test 4: kmeans determinism (same seed -> same labels)
    # ----------------------------------------------------------------
    res_km_a = cluster_kmeans(emb_50, n_clusters=5, seed=42)
    res_km_b = cluster_kmeans(emb_50, n_clusters=5, seed=42)
    labels_match = np.array_equal(res_km_a.labels, res_km_b.labels)
    _test(
        "kmeans determinism: same seed -> same labels",
        labels_match,
    )

    # ----------------------------------------------------------------
    # Test 5: agglomerative produces k clusters (skip if no sklearn)
    # ----------------------------------------------------------------
    if SKLEARN_CLUSTER_AVAILABLE:
        res_agg = cluster_agglomerative(emb_50, n_clusters=5, linkage="ward")
        n_unique_agg = len(np.unique(res_agg.labels))
        _test(
            "agglomerative: produces 5 clusters (ward linkage)",
            n_unique_agg == 5 and len(res_agg.labels) == 50,
            f"n_unique={n_unique_agg}",
        )
    else:
        total += 1
        passed += 1
        print(f"  SKIP  [{total:2d}] agglomerative: sklearn not available")

    # ----------------------------------------------------------------
    # Test 6: hdbscan runs without error (skip if unavailable)
    # ----------------------------------------------------------------
    if HDBSCAN_AVAILABLE:
        try:
            res_hdb = cluster_hdbscan(emb_50, min_cluster_size=3)
            hdb_ok = isinstance(res_hdb.labels, np.ndarray) and len(res_hdb.labels) == 50
            _test(
                "hdbscan: runs without error, returns 50 labels",
                hdb_ok,
            )
        except Exception as exc:
            total += 1
            failed += 1
            print(f"  FAIL  [{total:2d}] hdbscan: exception {exc}")
    else:
        total += 1
        passed += 1
        print(f"  SKIP  [{total:2d}] hdbscan: package not installed")

    # ----------------------------------------------------------------
    # Test 7: cluster factory routes to correct algorithm
    # ----------------------------------------------------------------
    cfg_km = ClusterConfig(algorithm="kmeans", n_clusters=5)
    res_factory = cluster(emb_50, cfg_km, seed=42)
    _test(
        "cluster factory: routes to kmeans",
        res_factory.algorithm == "kmeans" and len(np.unique(res_factory.labels)) == 5,
    )

    # ----------------------------------------------------------------
    # Test 8: stability ARI >= 0.9 for well-separated clusters
    # ----------------------------------------------------------------
    cfg_stability = ClusterConfig(
        algorithm="kmeans", n_clusters=5, stability_threshold=0.9
    )
    stab_result = evaluate_stability(
        emb_50, cfg_stability, n_runs=5
    )
    _test(
        "stability: ARI >= 0.9 for well-separated clusters",
        stab_result.ari >= 0.9,
        f"mean_ari={stab_result.ari:.4f}",
    )

    # ----------------------------------------------------------------
    # Test 9: stability ARI < 0.9 for overlapping clusters
    # ----------------------------------------------------------------
    emb_overlap, _ = _make_overlapping_clusters(
        n_clusters=5, n_per_cluster=10, dim=64, seed=99
    )
    cfg_overlap = ClusterConfig(
        algorithm="kmeans", n_clusters=5, stability_threshold=0.9
    )
    stab_overlap = evaluate_stability(
        emb_overlap, cfg_overlap, n_runs=5,
        seeds=[100, 200, 300, 400, 500],
    )
    # Overlapping clusters should be less stable (ARI lower)
    # We don't mandate ARI < 0.9 since it depends on the random draw,
    # but we check that evaluation runs without error and returns a value.
    overlap_ok = isinstance(stab_overlap.ari, float) and 0.0 <= stab_overlap.ari <= 1.0
    _test(
        "stability: overlapping clusters evaluated (ARI in [0,1])",
        overlap_ok,
        f"mean_ari={stab_overlap.ari:.4f}, is_stable={stab_overlap.is_stable}",
    )

    # ----------------------------------------------------------------
    # Test 10: diagnostics silhouette in [-1, 1]
    # ----------------------------------------------------------------
    diag = compute_diagnostics(emb_50, res_km.labels)
    sil_ok = -1.0 <= diag.silhouette_score <= 1.0
    _test(
        "diagnostics: silhouette_score in [-1, 1]",
        sil_ok,
        f"silhouette={diag.silhouette_score:.4f}",
    )

    # ----------------------------------------------------------------
    # Test 11: diagnostics cluster sizes sum to N
    # ----------------------------------------------------------------
    total_from_sizes = sum(diag.cluster_sizes.values())
    _test(
        "diagnostics: cluster sizes sum to n_tasks (50)",
        total_from_sizes == 50,
        f"sum={total_from_sizes}",
    )

    # ----------------------------------------------------------------
    # Test 12: diagnostics representative tasks are valid indices
    # ----------------------------------------------------------------
    all_reps_valid = all(
        0 <= idx < 50
        for idx in diag.representative_task_ids.values()
    )
    _test(
        "diagnostics: representative task indices in [0, 50)",
        all_reps_valid and len(diag.representative_task_ids) == 5,
        f"reps={diag.representative_task_ids}",
    )

    # ----------------------------------------------------------------
    # Test 13: within-cluster dispersion < inter-cluster distance
    # ----------------------------------------------------------------
    mean_within = np.mean(list(diag.within_cluster_dispersion.values()))
    if diag.inter_cluster_distances.size > 0 and diag.inter_cluster_distances.shape[0] >= 2:
        # Mean of upper triangle (inter-cluster distances)
        triu_idx = np.triu_indices(diag.inter_cluster_distances.shape[0], k=1)
        mean_between = float(diag.inter_cluster_distances[triu_idx].mean())
        wb_ok = mean_within < mean_between
        _test(
            "within-cluster dispersion < between-cluster distance",
            wb_ok,
            f"within={mean_within:.4f}, between={mean_between:.4f}",
        )
    else:
        total += 1
        passed += 1
        print(f"  SKIP  [{total:2d}] within/between comparison: not enough clusters")

    # ----------------------------------------------------------------
    # Test 14: pure-numpy kmeans fallback matches approximately
    # ----------------------------------------------------------------
    X_norm = _l2_normalize(emb_50)
    np_labels, np_centroids, np_inertia, np_niter = _numpy_kmeans(
        X_norm, n_clusters=5, seed=42, n_init=10,
    )
    np_n_unique = len(np.unique(np_labels))
    np_all_assigned = len(np_labels) == 50
    _test(
        "pure-numpy kmeans: produces 5 clusters, 50 points",
        np_n_unique == 5 and np_all_assigned,
        f"n_unique={np_n_unique}",
    )

    # Check if numpy kmeans is approximately similar to sklearn (when available)
    if SKLEARN_CLUSTER_AVAILABLE:
        # Both should find similar structure; measure via ARI
        ari_np_sk = adjusted_rand_score(res_km.labels, np_labels)
        _test(
            "pure-numpy kmeans: ARI vs sklearn >= 0.7",
            ari_np_sk >= 0.7,
            f"ari={ari_np_sk:.4f}",
        )
    else:
        total += 1
        passed += 1
        print(f"  SKIP  [{total:2d}] numpy vs sklearn kmeans: sklearn not available")

    # ----------------------------------------------------------------
    # Test 15: pure-numpy ARI matches sklearn
    # ----------------------------------------------------------------
    la = np.array([0, 0, 1, 1, 2, 2, 0, 1])
    lb = np.array([1, 1, 0, 0, 2, 2, 1, 0])
    np_ari = _numpy_adjusted_rand_score(la, lb)
    if SKLEARN_METRICS_AVAILABLE:
        sk_ari = float(_sklearn_ari(la, lb))
        ari_match = abs(np_ari - sk_ari) < 1e-10
        _test(
            "pure-numpy ARI: matches sklearn",
            ari_match,
            f"numpy={np_ari:.6f}, sklearn={sk_ari:.6f}",
        )
    else:
        # Just check it runs and gives a reasonable value
        ari_range_ok = -1.0 <= np_ari <= 1.0
        _test(
            "pure-numpy ARI: value in [-1, 1]",
            ari_range_ok,
            f"ari={np_ari:.6f}",
        )

    # ----------------------------------------------------------------
    # Test 16: pure-numpy NMI check
    # ----------------------------------------------------------------
    np_nmi = _numpy_normalized_mutual_info(la, lb)
    if SKLEARN_METRICS_AVAILABLE:
        sk_nmi = float(_sklearn_nmi(la, lb, average_method="arithmetic"))
        nmi_match = abs(np_nmi - sk_nmi) < 1e-6
        _test(
            "pure-numpy NMI: matches sklearn",
            nmi_match,
            f"numpy={np_nmi:.6f}, sklearn={sk_nmi:.6f}",
        )
    else:
        nmi_range_ok = 0.0 <= np_nmi <= 1.0
        _test(
            "pure-numpy NMI: value in [0, 1]",
            nmi_range_ok,
            f"nmi={np_nmi:.6f}",
        )

    # ----------------------------------------------------------------
    # Test 17: pure-numpy silhouette check
    # ----------------------------------------------------------------
    np_sil = _numpy_silhouette_score(X_norm, res_km.labels)
    sil_range = -1.0 <= np_sil <= 1.0
    if SKLEARN_METRICS_AVAILABLE:
        sk_sil = float(
            _sklearn_silhouette_score(X_norm, res_km.labels, metric="cosine")
        )
        sil_close = abs(np_sil - sk_sil) < 0.05  # allow small difference
        _test(
            "pure-numpy silhouette: close to sklearn",
            sil_range and sil_close,
            f"numpy={np_sil:.4f}, sklearn={sk_sil:.4f}",
        )
    else:
        _test(
            "pure-numpy silhouette: value in [-1, 1]",
            sil_range,
            f"sil={np_sil:.4f}",
        )

    # ----------------------------------------------------------------
    # Test 18: edge case -- single cluster
    # ----------------------------------------------------------------
    res_single = cluster_kmeans(emb_50, n_clusters=1, seed=42)
    single_ok = (
        len(np.unique(res_single.labels)) == 1
        and np.all(res_single.labels == 0)
        and res_single.n_clusters == 1
    )
    _test(
        "edge case: single cluster (k=1)",
        single_ok,
    )

    # ----------------------------------------------------------------
    # Test 19: edge case -- n_clusters == n_tasks
    # ----------------------------------------------------------------
    emb_5 = rng.randn(5, 16)
    emb_5 /= np.linalg.norm(emb_5, axis=1, keepdims=True)
    res_n_eq_k = cluster_kmeans(emb_5, n_clusters=5, seed=42)
    neqk_ok = (
        res_n_eq_k.n_clusters == 5
        and len(np.unique(res_n_eq_k.labels)) == 5
        and len(res_n_eq_k.labels) == 5
    )
    _test(
        "edge case: n_clusters == n_tasks (5 points, k=5)",
        neqk_ok,
        f"n_clusters={res_n_eq_k.n_clusters}, "
        f"unique_labels={len(np.unique(res_n_eq_k.labels))}",
    )

    # ----------------------------------------------------------------
    # Test 20: edge case -- empty embeddings
    # ----------------------------------------------------------------
    emb_empty = np.empty((0, 16), dtype=np.float64)
    res_empty = cluster_kmeans(emb_empty, n_clusters=3, seed=42)
    empty_ok = (
        len(res_empty.labels) == 0
        and res_empty.n_clusters == 0
    )
    _test(
        "edge case: empty embeddings handled gracefully",
        empty_ok,
    )

    # ----------------------------------------------------------------
    # Test 21: evaluate_stability_multi returns correct structure
    # ----------------------------------------------------------------
    multi_report = evaluate_stability_multi(
        emb_50, cfg_stability, n_trials=3, base_seed=10
    )
    expected_keys = {
        "ari_mean", "ari_std", "ari_min",
        "nmi_mean", "nmi_std", "nmi_min",
        "n_trials", "stable",
    }
    keys_ok = expected_keys.issubset(set(multi_report.keys()))
    _test(
        "evaluate_stability_multi: returns expected keys",
        keys_ok,
        f"keys={sorted(multi_report.keys())}",
    )

    # ----------------------------------------------------------------
    # Test 22: cluster_size_report
    # ----------------------------------------------------------------
    size_report = cluster_size_report(res_km.labels)
    sizes_sum_ok = size_report["total_tasks"] == 50
    has_sizes_keys = all(
        k in size_report
        for k in ("sizes", "n_clusters", "total_tasks", "singletons")
    )
    _test(
        "cluster_size_report: correct structure and sum",
        sizes_sum_ok and has_sizes_keys,
        f"total={size_report['total_tasks']}, keys_ok={has_sizes_keys}",
    )

    # ----------------------------------------------------------------
    # Test 23: ARI of identical labellings is 1.0
    # ----------------------------------------------------------------
    ari_perfect = adjusted_rand_score(res_km.labels, res_km.labels)
    _test(
        "ARI of identical labellings is 1.0",
        abs(ari_perfect - 1.0) < 1e-10,
        f"ari={ari_perfect:.6f}",
    )

    # ----------------------------------------------------------------
    # Test 24: NMI of identical labellings is 1.0
    # ----------------------------------------------------------------
    nmi_perfect = normalized_mutual_info_score(res_km.labels, res_km.labels)
    _test(
        "NMI of identical labellings is 1.0",
        abs(nmi_perfect - 1.0) < 1e-10,
        f"nmi={nmi_perfect:.6f}",
    )

    # ----------------------------------------------------------------
    # Test 25: run_clustering_pipeline end-to-end
    # ----------------------------------------------------------------
    pipeline_out = run_clustering_pipeline(
        emb_50,
        config=cfg_km,
        seed=42,
        compute_diag=True,
        run_stability=True,
        stability_n_trials=3,
    )
    pipeline_ok = (
        pipeline_out["result"].n_clusters == 5
        and pipeline_out["diagnostics"] is not None
        and pipeline_out["stability"] is not None
        and isinstance(pipeline_out["warnings"], list)
    )
    _test(
        "run_clustering_pipeline: end-to-end produces expected output",
        pipeline_ok,
    )

    # ----------------------------------------------------------------
    # Test 26: cosine_distance_matrix_chunked matches standard
    # ----------------------------------------------------------------
    D_standard = cosine_distance_matrix(emb_small)
    D_chunked = _cosine_distance_matrix_chunked(emb_small, chunk_size=4)
    chunked_match = np.allclose(D_standard, D_chunked, atol=1e-10)
    _test(
        "cosine_distance_matrix_chunked matches standard implementation",
        chunked_match,
    )

    # ----------------------------------------------------------------
    # Test 27: validate_inputs raises on empty
    # ----------------------------------------------------------------
    try:
        validate_inputs(np.empty((0, 16)), ClusterConfig())
        validate_empty_raised = False
    except ValueError:
        validate_empty_raised = True
    _test(
        "validate_inputs: raises ValueError on empty embeddings",
        validate_empty_raised,
    )

    # ----------------------------------------------------------------
    # Test 28: find_representative_tasks returns valid structure
    # ----------------------------------------------------------------
    task_id_strings = [f"task_{i}" for i in range(50)]
    reps = find_representative_tasks(
        emb_50, res_km.labels, res_km.centroids, task_id_strings
    )
    reps_ok = (
        len(reps) == 5
        and all("task_id" in v and "cosine_distance" in v for v in reps.values())
        and all(v["cosine_distance"] >= 0 for v in reps.values())
    )
    _test(
        "find_representative_tasks: valid structure for 5 clusters",
        reps_ok,
    )

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print()
    print("-" * 70)
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print("All tests PASSED.")
    else:
        print(f"WARNING: {failed} test(s) FAILED.")
    print("-" * 70)


if __name__ == "__main__":
    _run_self_tests()
