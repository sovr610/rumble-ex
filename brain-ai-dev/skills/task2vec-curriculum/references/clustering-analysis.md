# Clustering Analysis Reference

This document specifies distance metrics, clustering algorithms, stability evaluation
protocols, cluster diagnostics, and visualization utilities for Task2Vec task embeddings.
All clustering operates on L2-normalized Fisher-derived embeddings of dimension E (default
512). The target module is `brain_ai/meta/task_clustering.py`, implemented from the template
at `assets/clustering_template.py`.

---

## 1. Distance Metrics

### 1.1 Cosine Distance as Primary Metric

Define cosine distance between two embedding vectors u and v as:

```
d(u, v) = 1 - cos(u, v) = 1 - (u . v) / (||u|| * ||v||)
```

Because all Task2Vec embeddings are L2-normalized (||u|| = ||v|| = 1), this simplifies to:

```
d(u, v) = 1 - u . v
```

Range: [0, 2]. Identical vectors yield d = 0. Orthogonal vectors yield d = 1. Opposite
vectors yield d = 2 (rare for Fisher embeddings which are non-negative after log1p).

Use cosine distance as the default for all clustering, stability, and diagnostic
computations. Store the string `"cosine"` in `ClusterConfig.distance_metric`.

### 1.2 Why Cosine Over Euclidean

Fisher-derived task embeddings capture the relative importance of probe parameters for a
given task. The magnitude of the raw Fisher diagonal varies with support set size, number of
gradient samples, and numerical scale. After log1p + L2-normalization, embeddings lie on the
unit hypersphere. On the unit hypersphere, cosine distance and Euclidean distance are
monotonically related:

```
||u - v||^2 = 2 * (1 - cos(u, v)) = 2 * d_cosine(u, v)
```

This means that for L2-normalized vectors, running k-means with Euclidean distance is
equivalent to optimizing cosine distance. This is the key implementation insight: normalize
embeddings to unit length, then use standard Euclidean-based algorithms (scikit-learn
KMeans, AgglomerativeClustering with ward linkage). No custom distance kernels required.

However, when using precomputed distance matrices (agglomerative with complete/average
linkage, HDBSCAN), compute the cosine distance matrix explicitly. Do not rely on the
Euclidean equivalence in that case.

### 1.3 Pairwise Distance Matrix Computation

Compute the full N x N pairwise cosine distance matrix using batched matrix multiplication.
Avoid Python loops over pairs.

```python
import torch
import numpy as np

def cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine distance matrix.

    Args:
        embeddings: (N, E) array, assumed L2-normalized along axis=1.

    Returns:
        (N, N) symmetric distance matrix with zeros on diagonal.
    """
    # Similarity matrix via single matmul
    sim = embeddings @ embeddings.T       # (N, N), values in [-1, 1]
    dist = 1.0 - sim                      # cosine distance
    np.fill_diagonal(dist, 0.0)           # enforce exact zero on diagonal
    np.clip(dist, 0.0, 2.0, out=dist)     # clip numerical noise
    return dist
```

For large N (> 10,000), compute in chunks to avoid memory pressure:

```python
def cosine_distance_matrix_chunked(
    embeddings: np.ndarray,
    chunk_size: int = 2048
) -> np.ndarray:
    N = embeddings.shape[0]
    dist = np.zeros((N, N), dtype=np.float32)
    for i in range(0, N, chunk_size):
        end_i = min(i + chunk_size, N)
        for j in range(i, N, chunk_size):
            end_j = min(j + chunk_size, N)
            block = 1.0 - embeddings[i:end_i] @ embeddings[j:end_j].T
            dist[i:end_i, j:end_j] = block
            if i != j:
                dist[j:end_j, i:end_i] = block.T
    np.fill_diagonal(dist, 0.0)
    np.clip(dist, 0.0, 2.0, out=dist)
    return dist
```

When using PyTorch tensors on GPU, keep embeddings as `torch.Tensor` and compute
`1 - embeddings @ embeddings.T` directly. Convert to NumPy only when passing to
scikit-learn.

---

## 2. Clustering Algorithms

All clustering functions accept the same interface:

```python
def cluster_embeddings(
    embeddings: np.ndarray,          # (N, E), L2-normalized
    config: ClusterConfig,
    seed: int = 42
) -> ClusterResult:
    ...
```

`ClusterResult` is a dataclass containing:

| Field | Type | Description |
|---|---|---|
| `labels` | `np.ndarray` (N,) int | Cluster assignment per task (-1 for noise in HDBSCAN) |
| `n_clusters` | `int` | Number of clusters found |
| `centroids` | `np.ndarray` (K, E) or None | Cluster centroids (L2-normalized), None for HDBSCAN |
| `algorithm` | `str` | Algorithm name used |
| `seed` | `int` | Random seed used |
| `metadata` | `Dict[str, Any]` | Algorithm-specific metadata (inertia, linkage type, etc.) |

### 2.1 K-Means

Use scikit-learn `KMeans` with explicit `random_state` for full reproducibility. Because
embeddings are L2-normalized, Euclidean k-means optimizes cosine distance implicitly.

#### Initialization

Use k-means++ initialization (`init="k-means++"`). Always pass `random_state=seed` to make
initialization and centroid updates deterministic. Set `n_init=1` when reproducibility
matters (multiple inits with the same seed produce the same result, but `n_init > 1` with
different internal seeds adds variance). For production, use `n_init=10` with the same
`random_state` to get better optima while remaining reproducible.

#### Convergence

Set `max_iter=300` (default) and `tol=1e-4`. These defaults work for E=512 with up to
10,000 tasks. If the algorithm does not converge, log a warning but return the result. Store
`n_iter_` and `inertia_` in `ClusterResult.metadata`.

#### Implementation

```python
from sklearn.cluster import KMeans

def _cluster_kmeans(
    embeddings: np.ndarray,
    n_clusters: int,
    seed: int
) -> ClusterResult:
    # Ensure L2 normalization (defensive)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    X = embeddings / norms

    km = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        max_iter=300,
        tol=1e-4,
        random_state=seed,
    )
    labels = km.fit_predict(X)

    # Normalize centroids to unit sphere for cosine interpretation
    centroids = km.cluster_centers_
    centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.maximum(centroid_norms, 1e-8)

    return ClusterResult(
        labels=labels,
        n_clusters=n_clusters,
        centroids=centroids,
        algorithm="kmeans",
        seed=seed,
        metadata={
            "inertia": float(km.inertia_),
            "n_iter": int(km.n_iter_),
            "converged": km.n_iter_ < 300,
        },
    )
```

#### Post-Normalization of Centroids

After k-means converges, centroids are the Euclidean mean of assigned points. These means
do not generally lie on the unit sphere. L2-normalize centroids before computing cosine
distances to them. This is essential for the difficulty proxy in curriculum ordering, where
task difficulty is measured as cosine distance from the "easy cluster" centroid.

### 2.2 Agglomerative Clustering

Use scikit-learn `AgglomerativeClustering`. This algorithm builds a tree (dendrogram) of
merges and cuts it at the desired number of clusters.

#### Linkage Options

Support three linkage methods:

| Linkage | Distance Definition | When to Use |
|---|---|---|
| `ward` | Minimize within-cluster variance | Default; produces compact, equal-sized clusters. Requires Euclidean metric (use normalized embeddings). |
| `complete` | Maximum pairwise distance between clusters | Produces tight clusters; sensitive to outliers. Works with precomputed distance matrices. |
| `average` | Mean pairwise distance between clusters | Balanced between ward and complete. Works with precomputed distance matrices. |

Ward linkage requires `metric="euclidean"` (the default). Because embeddings are
L2-normalized, this is equivalent to cosine. Complete and average linkage can accept a
precomputed cosine distance matrix via `metric="precomputed"` and `affinity="precomputed"`.

#### Dendrogram-Based Cluster Count Selection

When the number of clusters is not known in advance, use the dendrogram to select k.
Compute the full linkage matrix using `scipy.cluster.hierarchy.linkage`, then inspect the
gap between merge distances. The largest gap suggests a natural cluster count.

```python
from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np

def select_k_from_dendrogram(
    embeddings: np.ndarray,
    linkage_method: str = "ward",
    max_k: int = 20
) -> int:
    """Select cluster count from dendrogram gap heuristic.

    Returns the k corresponding to the largest gap in merge distances,
    bounded by max_k.
    """
    Z = linkage(embeddings, method=linkage_method)
    # Z[:, 2] contains merge distances in ascending order
    merge_distances = Z[:, 2]
    # Compute gaps between consecutive merge distances
    gaps = np.diff(merge_distances)
    # Look at the last max_k gaps (top of dendrogram)
    candidate_gaps = gaps[-(max_k - 1):]
    # Largest gap index from the end
    best_gap_idx = np.argmax(candidate_gaps)
    # Convert to cluster count: gap at position i from end means k = max_k - i
    k = len(candidate_gaps) - best_gap_idx
    return int(np.clip(k, 2, max_k))
```

This is a heuristic. Validate the selected k with silhouette score (Section 4.1).

#### Precomputed Distance Matrix Mode

For complete and average linkage, pass the cosine distance matrix directly:

```python
from sklearn.cluster import AgglomerativeClustering

def _cluster_agglomerative(
    embeddings: np.ndarray,
    n_clusters: int,
    linkage: str = "ward",
    seed: int = 42           # not used by agglomerative, kept for interface consistency
) -> ClusterResult:
    if linkage == "ward":
        # Ward requires Euclidean; normalized embeddings make it cosine-equivalent
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        X = embeddings / np.maximum(norms, 1e-8)
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage="ward",
        )
        labels = model.fit_predict(X)
    else:
        # Complete or average: use precomputed cosine distance
        dist_matrix = cosine_distance_matrix(embeddings)
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="precomputed",
            linkage=linkage,
        )
        labels = model.fit_predict(dist_matrix)

    # Compute centroids as normalized mean of assigned embeddings
    centroids = _compute_centroids(embeddings, labels, n_clusters)

    return ClusterResult(
        labels=labels,
        n_clusters=n_clusters,
        centroids=centroids,
        algorithm=f"agglomerative_{linkage}",
        seed=seed,
        metadata={"linkage": linkage},
    )
```

Agglomerative clustering is deterministic for a given input (no random initialization), so
the seed parameter is not functionally used. Keep it in the interface for consistency with
the stability protocol.

### 2.3 HDBSCAN (Optional)

HDBSCAN is a density-based algorithm that does not require a fixed cluster count. It
discovers clusters of varying density and labels low-density points as noise (-1). Guard
the import with try/except because `hdbscan` is an optional dependency.

```python
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
```

#### Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `min_cluster_size` | 5 | Minimum points to form a cluster. Lower values find smaller clusters. |
| `min_samples` | `None` (= min_cluster_size) | Controls conservativeness. Higher = fewer, denser clusters. |
| `metric` | `"euclidean"` | Use `"euclidean"` with normalized embeddings for cosine equivalence, or pass a precomputed distance matrix. |
| `cluster_selection_method` | `"eom"` | `"eom"` (Excess of Mass) or `"leaf"`. EOM finds variable-density clusters; leaf finds fine-grained clusters. |

#### Handling Noise Points

HDBSCAN assigns label -1 to points it considers noise. Handle these in downstream code:

1. For curriculum ordering, assign noise points to the nearest cluster centroid.
2. For stability metrics (ARI/NMI), exclude noise points or treat -1 as its own cluster.
3. Log the fraction of noise points. If > 30% of tasks are noise, the embeddings may lack
   structure or `min_cluster_size` is too high.

```python
def _cluster_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: int = None,
    seed: int = 42             # HDBSCAN is largely deterministic; seed for tie-breaking
) -> ClusterResult:
    if not HDBSCAN_AVAILABLE:
        raise ImportError(
            "hdbscan package not installed. "
            "Install with: pip install hdbscan"
        )

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    X = embeddings / np.maximum(norms, 1e-8)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",       # cosine-equivalent for normalized vectors
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(X)

    n_clusters = len(set(labels) - {-1})
    noise_count = int(np.sum(labels == -1))

    # Centroids from non-noise points
    centroids = None
    if n_clusters > 0:
        centroids = _compute_centroids(
            X[labels >= 0], labels[labels >= 0], n_clusters
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
            "noise_fraction": noise_count / len(labels),
            "probabilities": clusterer.probabilities_.tolist(),
        },
    )
```

#### Centroid Computation Utility

Shared across algorithms:

```python
def _compute_centroids(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_clusters: int
) -> np.ndarray:
    """Compute L2-normalized centroids from cluster assignments."""
    E = embeddings.shape[1]
    centroids = np.zeros((n_clusters, E), dtype=np.float32)
    for k in range(n_clusters):
        mask = labels == k
        if mask.sum() > 0:
            centroid = embeddings[mask].mean(axis=0)
            norm = np.linalg.norm(centroid)
            centroids[k] = centroid / max(norm, 1e-8)
    return centroids
```

---

## 3. Stability Metrics

Clustering stability measures whether the algorithm produces consistent assignments under
perturbation. For deterministic algorithms with fixed seeds, stability should be perfect.
The stability protocol tests reproducibility across seeds and quantifies sensitivity.

### 3.1 Adjusted Rand Index (ARI)

ARI measures the agreement between two clusterings of the same data, adjusted for chance.

**Definition**: Given two label vectors `labels_a` and `labels_b` of the same N points,
compute the contingency table of pair co-assignments and adjust for the expected value
under random permutation.

```
ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)
```

where RI is the Rand Index (fraction of pairs that are either co-clustered in both or
separated in both).

**Range**: [-1, 1].
- ARI = 1.0: Perfect agreement (identical partitions up to label permutation).
- ARI = 0.0: Agreement equal to chance.
- ARI < 0.0: Agreement worse than chance.

**Interpretation for stability**: For k-means with the same seed on the same data, ARI must
be exactly 1.0. When comparing across different seeds, ARI >= 0.9 indicates stable
structure. ARI < 0.7 signals that the clustering is sensitive to initialization and the
data may not have well-separated clusters.

```python
from sklearn.metrics import adjusted_rand_score

ari = adjusted_rand_score(labels_a, labels_b)
```

### 3.2 Normalized Mutual Information (NMI)

NMI measures the information-theoretic agreement between two clusterings.

**Definition**: Given two clusterings U and V:

```
NMI(U, V) = 2 * I(U; V) / (H(U) + H(V))
```

where I(U; V) is the mutual information and H(.) is the entropy.

**Range**: [0, 1].
- NMI = 1.0: Perfect agreement.
- NMI = 0.0: Independent clusterings.

NMI is always non-negative, which makes it easier to interpret as a "quality fraction" but
less sensitive to detecting anti-correlation than ARI. Use both metrics and report both.

```python
from sklearn.metrics import normalized_mutual_info_score

nmi = normalized_mutual_info_score(labels_a, labels_b, average_method="arithmetic")
```

Always specify `average_method="arithmetic"` for the symmetric normalization variant.

### 3.3 Stability Protocol

The stability protocol evaluates whether clustering assignments are robust to
initialization randomness. This is the primary gate for the "cluster stability" done-when
criterion (Gate (b) in SKILL.md).

#### Single-Trial Stability

Run clustering twice on the same data with two different seeds. Compare assignments.

```python
def evaluate_stability_single(
    embeddings: np.ndarray,
    config: ClusterConfig,
    seed_a: int = 42,
    seed_b: int = 123
) -> Dict[str, float]:
    """Run two clusterings and compare with ARI and NMI."""
    result_a = cluster_embeddings(embeddings, config, seed=seed_a)
    result_b = cluster_embeddings(embeddings, config, seed=seed_b)

    # For HDBSCAN: filter out noise points present in both
    if config.algorithm == "hdbscan":
        mask = (result_a.labels >= 0) & (result_b.labels >= 0)
        la, lb = result_a.labels[mask], result_b.labels[mask]
    else:
        la, lb = result_a.labels, result_b.labels

    ari = adjusted_rand_score(la, lb)
    nmi = normalized_mutual_info_score(la, lb, average_method="arithmetic")

    return {"ari": ari, "nmi": nmi}
```

#### Multi-Trial Stability for Statistical Confidence

A single pair of seeds may be unrepresentative. Run T trials (T >= 5) with different seed
pairs and report mean and standard deviation of ARI and NMI.

```python
def evaluate_stability_multi(
    embeddings: np.ndarray,
    config: ClusterConfig,
    n_trials: int = 10,
    base_seed: int = 0
) -> Dict[str, Any]:
    """Run multiple stability trials and report statistics."""
    ari_scores = []
    nmi_scores = []

    for t in range(n_trials):
        seed_a = base_seed + t * 2
        seed_b = base_seed + t * 2 + 1
        scores = evaluate_stability_single(
            embeddings, config, seed_a, seed_b
        )
        ari_scores.append(scores["ari"])
        nmi_scores.append(scores["nmi"])

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
```

**Target thresholds**:
- K-means with same seed: ARI = 1.0 exactly (deterministic given same random_state).
- K-means across seeds: ARI >= 0.9 (mean), ARI_min >= 0.8.
- Agglomerative (all linkages): ARI = 1.0 exactly (deterministic, no random state).
- HDBSCAN: ARI >= 0.85 (largely deterministic, minor tie-breaking variance).

If stability falls below threshold, consider: reducing n_clusters, increasing the number
of tasks, checking whether embedding quality is sufficient (see diagnostics below).

#### Stability as a Gate

The stability check is a pass/fail gate. Wrap it in a function that raises on failure:

```python
def assert_cluster_stability(
    embeddings: np.ndarray,
    config: ClusterConfig,
    n_trials: int = 10
) -> Dict[str, Any]:
    """Assert clustering meets stability threshold. Raise if not."""
    report = evaluate_stability_multi(embeddings, config, n_trials=n_trials)
    if not report["stable"]:
        raise AssertionError(
            f"Clustering stability below threshold: "
            f"ARI_min={report['ari_min']:.4f} < {config.stability_threshold}. "
            f"ARI_mean={report['ari_mean']:.4f}, ARI_std={report['ari_std']:.4f}"
        )
    return report
```

---

## 4. Cluster Diagnostics

After clustering, compute diagnostic metrics to assess cluster quality, detect degenerate
configurations, and identify representative tasks.

### 4.1 Silhouette Score

The silhouette coefficient measures how similar a sample is to its own cluster compared to
the nearest neighboring cluster. Computed per sample and then averaged.

For sample i assigned to cluster C_i:
- a(i) = mean distance from i to all other points in C_i (intra-cluster distance)
- b(i) = min over all clusters C != C_i of mean distance from i to points in C (nearest-cluster distance)
- s(i) = (b(i) - a(i)) / max(a(i), b(i))

Range: [-1, 1]. s(i) near 1 means well-clustered. Near 0 means on boundary. Negative means
likely misassigned.

```python
from sklearn.metrics import silhouette_score, silhouette_samples

def compute_silhouette(
    embeddings: np.ndarray,
    labels: np.ndarray
) -> Dict[str, Any]:
    """Compute silhouette score and per-sample coefficients."""
    # Filter out noise labels if present
    mask = labels >= 0
    if mask.sum() < 2:
        return {"mean_silhouette": 0.0, "per_sample": np.array([])}

    X = embeddings[mask]
    L = labels[mask]

    # Check for at least 2 clusters
    n_clusters = len(set(L))
    if n_clusters < 2:
        return {"mean_silhouette": 0.0, "per_sample": np.array([])}

    # Use cosine metric for silhouette
    per_sample = silhouette_samples(X, L, metric="cosine")
    mean_score = float(silhouette_score(X, L, metric="cosine"))

    return {
        "mean_silhouette": mean_score,
        "per_sample": per_sample,
        "per_cluster_mean": {
            int(k): float(per_sample[L == k].mean())
            for k in sorted(set(L))
        },
    }
```

**Interpretation guidelines**:
- Mean silhouette > 0.5: strong cluster structure.
- Mean silhouette 0.25-0.5: reasonable structure, some overlap.
- Mean silhouette < 0.25: weak structure, consider reducing k or examining embedding quality.
- Any cluster with negative mean silhouette: likely a spurious cluster.

### 4.2 Within-Cluster Cosine Dispersion

Measure how spread out each cluster is by computing the mean pairwise cosine distance among
its members. Lower dispersion = tighter cluster.

```python
def within_cluster_dispersion(
    embeddings: np.ndarray,
    labels: np.ndarray
) -> Dict[int, float]:
    """Mean pairwise cosine distance within each cluster."""
    dispersion = {}
    for k in sorted(set(labels)):
        if k == -1:
            continue
        mask = labels == k
        cluster_embs = embeddings[mask]
        n = cluster_embs.shape[0]
        if n < 2:
            dispersion[int(k)] = 0.0
            continue
        sim = cluster_embs @ cluster_embs.T
        # Extract upper triangle (exclude diagonal)
        triu_indices = np.triu_indices(n, k=1)
        pairwise_dist = 1.0 - sim[triu_indices]
        dispersion[int(k)] = float(pairwise_dist.mean())
    return dispersion
```

Combine with cluster size to detect degenerate clusters: a cluster with high dispersion and
many members may need splitting. A cluster with near-zero dispersion and one member is a
singleton (see below).

### 4.3 Cluster Size Distribution

Report the number of tasks per cluster. Flag degenerate cases:

- **Empty clusters**: k-means can produce empty clusters if initialized poorly. This should
  not happen with k-means++ but check anyway. Remove empty clusters from downstream use.
- **Singleton clusters**: clusters with exactly one task. These provide no diversity benefit.
  Flag them but do not remove (the task may be genuinely unique).
- **Imbalanced clusters**: one cluster contains > 50% of all tasks. This suggests k is too
  high or embeddings lack sufficient diversity.

```python
def cluster_size_report(
    labels: np.ndarray
) -> Dict[str, Any]:
    """Report cluster sizes and detect degenerate cases."""
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    size_map = {int(k): int(c) for k, c in zip(unique, counts)}
    total = int(counts.sum())
    n_clusters = len(unique)

    return {
        "sizes": size_map,
        "n_clusters": n_clusters,
        "total_tasks": total,
        "min_size": int(counts.min()) if len(counts) > 0 else 0,
        "max_size": int(counts.max()) if len(counts) > 0 else 0,
        "mean_size": float(counts.mean()) if len(counts) > 0 else 0.0,
        "std_size": float(counts.std()) if len(counts) > 0 else 0.0,
        "empty_clusters": n_clusters == 0,
        "singletons": [int(k) for k, c in zip(unique, counts) if c == 1],
        "dominant_cluster": int(unique[counts.argmax()]) if len(counts) > 0 else -1,
        "dominant_fraction": float(counts.max() / total) if total > 0 else 0.0,
    }
```

### 4.4 Representative Tasks

For each cluster, identify the task whose embedding is closest to the cluster centroid.
This is useful for human inspection and for the "easy cluster" definition in curriculum
ordering.

```python
def find_representative_tasks(
    embeddings: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    task_ids: List[str]
) -> Dict[int, Dict[str, Any]]:
    """Find the task closest to each cluster centroid."""
    representatives = {}
    for k in range(centroids.shape[0]):
        mask = labels == k
        if mask.sum() == 0:
            continue
        cluster_embs = embeddings[mask]
        cluster_ids = [task_ids[i] for i in np.where(mask)[0]]
        # Cosine similarity to centroid
        sims = cluster_embs @ centroids[k]
        best_idx = int(np.argmax(sims))
        representatives[int(k)] = {
            "task_id": cluster_ids[best_idx],
            "cosine_similarity": float(sims[best_idx]),
            "cosine_distance": float(1.0 - sims[best_idx]),
        }
    return representatives
```

### 4.5 Inter-Cluster Distances

Compute the K x K centroid-to-centroid cosine distance matrix. This reveals how well
separated the clusters are and identifies potentially redundant clusters (those with very
small inter-cluster distance).

```python
def inter_cluster_distances(
    centroids: np.ndarray
) -> np.ndarray:
    """K x K cosine distance matrix between cluster centroids."""
    sim = centroids @ centroids.T
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    np.clip(dist, 0.0, 2.0, out=dist)
    return dist
```

Flag cluster pairs with inter-cluster distance < 0.1 as candidates for merging. If many
clusters are close together, reduce k.

---

## 5. Visualization Utilities

Visualization functions are optional utilities for offline analysis. They should not be
called in training loops. Guard matplotlib/plotting imports with try/except.

### 5.1 t-SNE / UMAP Projection

Project E-dimensional embeddings to 2D for scatter plots, colored by cluster assignment.

```python
def plot_embedding_clusters(
    embeddings: np.ndarray,
    labels: np.ndarray,
    method: str = "tsne",
    seed: int = 42,
    title: str = "Task Embedding Clusters",
    save_path: str = None
):
    """2D scatter plot of embeddings colored by cluster.

    Args:
        embeddings: (N, E) L2-normalized embeddings.
        labels: (N,) cluster labels.
        method: 'tsne' or 'umap'.
        seed: Random seed for projection.
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    import matplotlib.pyplot as plt

    if method == "tsne":
        from sklearn.manifold import TSNE
        projector = TSNE(
            n_components=2,
            metric="cosine",
            random_state=seed,
            perplexity=min(30, len(embeddings) - 1),
        )
        coords = projector.fit_transform(embeddings)
    elif method == "umap":
        try:
            import umap
        except ImportError:
            raise ImportError("umap-learn not installed. pip install umap-learn")
        projector = umap.UMAP(
            n_components=2,
            metric="cosine",
            random_state=seed,
        )
        coords = projector.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown projection method: {method}")

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    unique_labels = sorted(set(labels))
    cmap = plt.cm.get_cmap("tab20", len(unique_labels))

    for i, k in enumerate(unique_labels):
        mask = labels == k
        label_str = f"Cluster {k}" if k >= 0 else "Noise"
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[cmap(i)], label=label_str,
            s=30, alpha=0.7, edgecolors="none"
        )

    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.set_xlabel(f"{method.upper()} dim 1")
    ax.set_ylabel(f"{method.upper()} dim 2")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

**Notes on t-SNE vs UMAP**:
- t-SNE preserves local structure better; use for inspecting within-cluster relationships.
- UMAP preserves more global structure; use for seeing inter-cluster relationships.
- Both are stochastic; always set the random seed. The projection is for visualization only
  and must not be used for clustering or distance computation.

### 5.2 Cluster Size Histogram

```python
def plot_cluster_sizes(
    labels: np.ndarray,
    title: str = "Cluster Size Distribution",
    save_path: str = None
):
    """Bar chart of cluster sizes."""
    import matplotlib.pyplot as plt

    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(unique.astype(str), counts, color="steelblue")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Tasks")
    ax.set_title(title)
    for i, (u, c) in enumerate(zip(unique, counts)):
        ax.text(i, c + 0.5, str(c), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

### 5.3 Silhouette Plot

Display per-sample silhouette coefficients, sorted by cluster, to identify misassigned
samples and weak clusters.

```python
def plot_silhouette(
    embeddings: np.ndarray,
    labels: np.ndarray,
    title: str = "Silhouette Analysis",
    save_path: str = None
):
    """Silhouette plot showing per-sample coefficients by cluster."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import silhouette_samples, silhouette_score

    mask = labels >= 0
    X, L = embeddings[mask], labels[mask]
    n_clusters = len(set(L))
    if n_clusters < 2:
        return

    sample_scores = silhouette_samples(X, L, metric="cosine")
    mean_score = silhouette_score(X, L, metric="cosine")

    fig, ax = plt.subplots(figsize=(8, 6))
    y_lower = 10

    for k in sorted(set(L)):
        cluster_scores = sample_scores[L == k]
        cluster_scores.sort()
        size_k = len(cluster_scores)
        y_upper = y_lower + size_k

        color = plt.cm.nipy_spectral(float(k) / n_clusters)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0, cluster_scores,
            facecolor=color, edgecolor=color, alpha=0.7,
        )
        ax.text(-0.05, y_lower + 0.5 * size_k, str(k), fontsize=9)
        y_lower = y_upper + 10

    ax.axvline(x=mean_score, color="red", linestyle="--",
               label=f"Mean: {mean_score:.3f}")
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster (sorted samples)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

### 5.4 Distance Heatmap

Display the pairwise cosine distance matrix or the inter-cluster centroid distance matrix
as a heatmap.

```python
def plot_distance_heatmap(
    dist_matrix: np.ndarray,
    labels: List[str] = None,
    title: str = "Cosine Distance Matrix",
    save_path: str = None
):
    """Heatmap of a distance matrix."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(dist_matrix, cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax, label="Cosine Distance")

    if labels is not None:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)

    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

---

## 6. Implementation Patterns

### 6.1 Full Clustering Pipeline

End-to-end flow from embeddings to cluster result with diagnostics.

```python
def run_clustering_pipeline(
    embeddings: np.ndarray,
    task_ids: List[str],
    config: ClusterConfig,
    seed: int = 42,
    compute_diagnostics: bool = True,
    save_plots: str = None         # directory path for optional plots
) -> Dict[str, Any]:
    """Full clustering pipeline: cluster, diagnose, optionally visualize.

    Args:
        embeddings: (N, E) L2-normalized task embeddings.
        task_ids: N task identifier strings.
        config: ClusterConfig with algorithm, n_clusters, distance_metric, etc.
        seed: Random seed for clustering.
        compute_diagnostics: Whether to compute silhouette, dispersion, etc.
        save_plots: If not None, directory to save visualization PNGs.

    Returns:
        Dictionary with cluster result, diagnostics, and stability report.
    """
    N, E = embeddings.shape

    # --- Edge case: too few tasks ---
    if N < config.n_clusters:
        # Assign each task to its own cluster
        return {
            "result": ClusterResult(
                labels=np.arange(N),
                n_clusters=N,
                centroids=embeddings.copy(),
                algorithm=config.algorithm,
                seed=seed,
                metadata={"warning": "n_tasks < n_clusters, one task per cluster"},
            ),
            "diagnostics": None,
            "stability": None,
        }

    if N < 2:
        return {
            "result": ClusterResult(
                labels=np.zeros(N, dtype=int),
                n_clusters=1,
                centroids=embeddings[:1].copy(),
                algorithm=config.algorithm,
                seed=seed,
                metadata={"warning": "only 1 task, trivial clustering"},
            ),
            "diagnostics": None,
            "stability": None,
        }

    # --- Cluster ---
    result = cluster_embeddings(embeddings, config, seed=seed)

    # --- Diagnostics ---
    diagnostics = None
    if compute_diagnostics:
        sil = compute_silhouette(embeddings, result.labels)
        disp = within_cluster_dispersion(embeddings, result.labels)
        sizes = cluster_size_report(result.labels)
        reps = find_representative_tasks(
            embeddings, result.labels, result.centroids, task_ids
        ) if result.centroids is not None else {}
        inter = inter_cluster_distances(result.centroids).tolist() \
            if result.centroids is not None else []

        diagnostics = {
            "silhouette": sil,
            "within_cluster_dispersion": disp,
            "cluster_sizes": sizes,
            "representatives": reps,
            "inter_cluster_distances": inter,
        }

    # --- Stability ---
    stability = evaluate_stability_multi(embeddings, config, n_trials=10, base_seed=seed)

    # --- Visualization ---
    if save_plots is not None:
        import os
        os.makedirs(save_plots, exist_ok=True)
        plot_embedding_clusters(
            embeddings, result.labels,
            save_path=os.path.join(save_plots, "clusters_tsne.png"),
        )
        plot_cluster_sizes(
            result.labels,
            save_path=os.path.join(save_plots, "cluster_sizes.png"),
        )
        if diagnostics and diagnostics["silhouette"]["per_sample"].size > 0:
            plot_silhouette(
                embeddings, result.labels,
                save_path=os.path.join(save_plots, "silhouette.png"),
            )
        if result.centroids is not None:
            inter_dist = inter_cluster_distances(result.centroids)
            plot_distance_heatmap(
                inter_dist,
                labels=[f"C{k}" for k in range(result.n_clusters)],
                title="Inter-Cluster Centroid Distances",
                save_path=os.path.join(save_plots, "inter_cluster_heatmap.png"),
            )

    return {
        "result": result,
        "diagnostics": diagnostics,
        "stability": stability,
    }
```

### 6.2 Stability Evaluation Loop

Standalone script pattern for evaluating stability across different configurations.

```python
def sweep_stability(
    embeddings: np.ndarray,
    algorithms: List[str] = ["kmeans", "agglomerative"],
    k_values: List[int] = [4, 8, 12, 16],
    n_trials: int = 10
) -> List[Dict[str, Any]]:
    """Sweep over algorithms and k values, reporting stability for each."""
    results = []
    for algo in algorithms:
        for k in k_values:
            config = ClusterConfig(
                algorithm=algo,
                n_clusters=k,
                distance_metric="cosine",
                stability_threshold=0.9,
            )
            try:
                stability = evaluate_stability_multi(
                    embeddings, config, n_trials=n_trials
                )
                sil = compute_silhouette(
                    embeddings,
                    cluster_embeddings(embeddings, config, seed=42).labels
                )
                results.append({
                    "algorithm": algo,
                    "k": k,
                    "ari_mean": stability["ari_mean"],
                    "ari_std": stability["ari_std"],
                    "ari_min": stability["ari_min"],
                    "nmi_mean": stability["nmi_mean"],
                    "silhouette": sil["mean_silhouette"],
                    "stable": stability["stable"],
                })
            except Exception as e:
                results.append({
                    "algorithm": algo,
                    "k": k,
                    "error": str(e),
                })
    return results
```

Print the results as a table to select the best (algorithm, k) combination. Prefer the
configuration with the highest silhouette score among those passing the stability gate.

### 6.3 Diagnostic Report Generation

Generate a structured report suitable for logging or checkpoint metadata.

```python
def generate_diagnostic_report(
    pipeline_output: Dict[str, Any]
) -> Dict[str, Any]:
    """Produce a JSON-serializable diagnostic report from pipeline output."""
    result = pipeline_output["result"]
    diag = pipeline_output["diagnostics"]
    stab = pipeline_output["stability"]

    report = {
        "clustering": {
            "algorithm": result.algorithm,
            "n_clusters": result.n_clusters,
            "seed": result.seed,
            "metadata": result.metadata,
        },
        "quality": {},
        "stability": {},
        "warnings": [],
    }

    if diag is not None:
        report["quality"] = {
            "mean_silhouette": diag["silhouette"]["mean_silhouette"],
            "per_cluster_silhouette": diag["silhouette"].get("per_cluster_mean", {}),
            "within_cluster_dispersion": diag["within_cluster_dispersion"],
            "cluster_sizes": diag["cluster_sizes"]["sizes"],
            "representatives": {
                k: v["task_id"] for k, v in diag["representatives"].items()
            },
        }

        # Warnings
        sizes = diag["cluster_sizes"]
        if sizes["dominant_fraction"] > 0.5:
            report["warnings"].append(
                f"Cluster {sizes['dominant_cluster']} contains "
                f"{sizes['dominant_fraction']:.0%} of tasks"
            )
        if sizes["singletons"]:
            report["warnings"].append(
                f"Singleton clusters: {sizes['singletons']}"
            )
        if diag["silhouette"]["mean_silhouette"] < 0.25:
            report["warnings"].append(
                f"Low silhouette score: {diag['silhouette']['mean_silhouette']:.3f}"
            )

        # Check per-cluster silhouette for negative values
        for k, v in diag["silhouette"].get("per_cluster_mean", {}).items():
            if v < 0:
                report["warnings"].append(
                    f"Cluster {k} has negative mean silhouette: {v:.3f}"
                )

    if stab is not None:
        report["stability"] = {
            "ari_mean": stab["ari_mean"],
            "ari_std": stab["ari_std"],
            "ari_min": stab["ari_min"],
            "nmi_mean": stab["nmi_mean"],
            "nmi_std": stab["nmi_std"],
            "n_trials": stab["n_trials"],
            "passed": stab["stable"],
        }
        if not stab["stable"]:
            report["warnings"].append(
                f"Stability gate FAILED: ARI_min={stab['ari_min']:.4f}"
            )

    return report
```

### 6.4 Handling Edge Cases

Summarize all edge cases and required handling:

| Condition | Detection | Action |
|---|---|---|
| N < 2 | Check N before clustering | Return single cluster with all tasks, skip diagnostics |
| N < n_clusters | N < config.n_clusters | Assign one task per cluster, log warning |
| Empty cluster after k-means | Any cluster with 0 members | Remove from centroid list, renumber labels, log warning |
| All embeddings identical | Pairwise distance matrix is all zeros | Return single cluster, silhouette undefined, log warning |
| HDBSCAN finds 0 clusters | n_clusters == 0 after HDBSCAN | Fall back to k-means with k=1, log warning |
| HDBSCAN noise > 30% | noise_fraction > 0.3 | Log warning, suggest reducing min_cluster_size |
| Silhouette undefined | Fewer than 2 clusters after filtering | Return 0.0, skip silhouette plot |
| Singular covariance (whitening) | Eigenvalues near zero in embedding covariance | Add regularization (1e-6 * I) before inverting |

Implement edge case handling at the top of each function with early returns and explicit
warning messages. Never let edge cases produce silent failures or NaN values.

```python
def _validate_inputs(
    embeddings: np.ndarray,
    config: ClusterConfig
) -> List[str]:
    """Validate clustering inputs and return list of warnings."""
    warnings = []
    N, E = embeddings.shape

    if N == 0:
        raise ValueError("Cannot cluster zero embeddings")

    if N < config.n_clusters and config.algorithm != "hdbscan":
        warnings.append(
            f"n_tasks ({N}) < n_clusters ({config.n_clusters}). "
            f"Reducing to {N} clusters."
        )

    # Check normalization
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        warnings.append(
            f"Embeddings not L2-normalized (norm range: "
            f"[{norms.min():.4f}, {norms.max():.4f}]). "
            f"Normalizing before clustering."
        )

    # Check for duplicate embeddings
    dist = cosine_distance_matrix(embeddings)
    np.fill_diagonal(dist, np.inf)
    min_dist = dist.min()
    if min_dist < 1e-6:
        n_duplicates = int((dist < 1e-6).sum() // 2)
        warnings.append(
            f"{n_duplicates} near-duplicate embedding pairs detected "
            f"(cosine distance < 1e-6)"
        )

    # Check for constant embeddings
    if np.allclose(embeddings, embeddings[0:1], atol=1e-6):
        warnings.append("All embeddings are approximately identical")

    return warnings
```

---

## Summary of Thresholds and Defaults

| Parameter | Default | Notes |
|---|---|---|
| Distance metric | Cosine | `1 - cos(u,v)` on L2-normalized embeddings |
| K-means n_init | 10 | Multiple initializations for better optima |
| K-means max_iter | 300 | Default scikit-learn setting |
| K-means tol | 1e-4 | Convergence tolerance |
| Agglomerative linkage | ward | Default; complete/average for precomputed distance |
| HDBSCAN min_cluster_size | 5 | Lower for small task sets |
| Stability threshold (ARI) | 0.9 | Gate (b) in SKILL.md |
| Stability n_trials | 10 | Minimum for statistical confidence |
| Silhouette "strong" | > 0.5 | Indicates well-separated clusters |
| Silhouette "weak" | < 0.25 | Suggests poor cluster structure |
| Noise fraction warning | > 0.3 | HDBSCAN-specific |
| Near-cluster merge threshold | < 0.1 | Inter-cluster centroid cosine distance |
| Near-duplicate detection | < 1e-6 | Cosine distance between task pairs |

All thresholds are configurable via `ClusterConfig` or function parameters. The values
listed here are defaults that work for the typical case of 50-1000 task embeddings in
E=512 dimensions. Adjust for significantly different scales.
