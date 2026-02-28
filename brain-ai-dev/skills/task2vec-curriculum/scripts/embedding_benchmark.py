#!/usr/bin/env python3
"""Task2Vec Embedding Pipeline -- Performance Benchmark

Self-contained benchmark script that measures throughput, latency, and memory
usage of the key operations in the Task2Vec embedding pipeline: embedding
extraction (probe + Fisher), clustering (K-means, distance matrix), curriculum
ordering (easy-to-hard, diversity batches, Kendall tau), registry I/O
(save/load at scale), and peak memory profiling.

All stubs are inline -- no brain_ai imports required. The script uses a Conv4-
like probe network, synthetic episodes, and simplified Fisher computation to
benchmark the pipeline's computational characteristics independently of the
rest of the cognitive architecture.

Usage:
    python embedding_benchmark.py                           # Run all suites
    python embedding_benchmark.py --suite embedding         # Single suite
    python embedding_benchmark.py --device cuda             # GPU benchmarks
    python embedding_benchmark.py --json results.json       # Save JSON output
    python embedding_benchmark.py --quick                   # Reduced sizes
    python embedding_benchmark.py --list                    # List suites

Benchmark Suites:
    embedding          Embedding extraction throughput (episodes/sec, ms/ep)
    fisher             Fisher computation speed (samples/sec, ms/sample)
    clustering         K-means and distance matrix speed (ms/clustering)
    curriculum         Curriculum ordering and batch construction (ms/ordering)
    registry_io        Registry save/load throughput (ms/save, ms/load)
    memory             Peak memory for extraction and registry (MB)
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Deterministic seeding
# ---------------------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALL_SUITES = [
    "embedding",
    "fisher",
    "clustering",
    "curriculum",
    "registry_io",
    "memory",
]

_SEP = "=" * 76


# ============================================================================
# Benchmark Infrastructure
# ============================================================================

@dataclass
class BenchmarkResult:
    """A single benchmark measurement.

    Attributes:
        name: Human-readable name for the measurement.
        metric: Unit string (e.g., ``"ms"``, ``"MB"``, ``"episodes/sec"``).
        value: The measured numeric value.
        details: Extra metadata (std, min, max, repeats, config, etc.).
    """
    name: str
    metric: str
    value: float
    details: Dict[str, Any] = field(default_factory=dict)


class BenchmarkSuite:
    """Core benchmarking harness with warmup/repeat timing and memory tracking.

    Provides ``bench()`` for timing callables with warmup, ``record()`` for
    pre-computed results, and ``peak_mem_mb()`` / ``reset_mem()`` for memory
    tracking on both CPU and CUDA.

    Args:
        name: Name of this benchmark suite.
        device: Torch device string (``"cpu"`` or ``"cuda"``).
    """

    def __init__(self, name: str, device: str = "cpu"):
        self.name = name
        self.device = torch.device(device)
        self.results: List[BenchmarkResult] = []

    def bench(
        self,
        fn,
        name: str,
        metric: str = "ms",
        warmup: int = 3,
        repeats: int = 10,
    ) -> float:
        """Time a callable with warmup iterations and averaging.

        Args:
            fn: Zero-argument callable to benchmark.
            name: Human-readable name for this measurement.
            metric: Unit string for reporting.
            warmup: Number of warmup calls (results discarded).
            repeats: Number of timed calls.

        Returns:
            Mean time in seconds.
        """
        for _ in range(warmup):
            fn()
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        times: List[float] = []
        for _ in range(repeats):
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        mean_t = sum(times) / len(times)
        std_t = (sum((t - mean_t) ** 2 for t in times) / len(times)) ** 0.5
        self.results.append(BenchmarkResult(
            name=name,
            metric=metric,
            value=mean_t * 1000,
            details={
                "std_ms": std_t * 1000,
                "min_ms": min(times) * 1000,
                "max_ms": max(times) * 1000,
                "repeats": repeats,
            },
        ))
        return mean_t

    def record(
        self,
        name: str,
        metric: str,
        value: float,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a pre-computed benchmark result.

        Args:
            name: Human-readable name.
            metric: Unit string.
            value: Measured value.
            details: Optional extra metadata.
        """
        self.results.append(BenchmarkResult(
            name=name, metric=metric, value=value, details=details or {},
        ))

    def peak_mem_mb(self) -> float:
        """Get peak memory in MB.

        On CUDA, returns ``torch.cuda.max_memory_allocated``.
        On CPU, returns ``resource.getrusage`` RSS if available, else 0.
        """
        if self.device.type == "cuda":
            return torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        except Exception:
            return 0.0

    def reset_mem(self) -> None:
        """Reset GPU peak memory stats and run garbage collection."""
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()


# ============================================================================
# Inline Stubs: Conv4 Probe Network
# ============================================================================

class Conv4Probe(nn.Module):
    """Conv4-like probe network for Task2Vec embedding extraction.

    Four convolutional blocks with 3x3 kernels, batch normalization (no
    running stats for determinism), ReLU activation, and 2x2 max pooling
    for the first three blocks plus adaptive average pooling in the fourth.
    Followed by a linear classifier head.

    This is a simplified version of the probe described in the Task2Vec
    paper. In the real pipeline, the probe is pretrained and frozen; here
    it is randomly initialized for benchmarking purposes.

    Args:
        in_channels: Number of input channels (1 for grayscale, 3 for RGB).
        num_filters: Number of convolutional filters per block.
        num_classes: Number of output classes for the classifier head.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_filters: int = 64,
        num_classes: int = 5,
    ):
        super().__init__()
        nf = num_filters
        # Block 1
        self.conv1 = nn.Conv2d(in_channels, nf, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(nf, track_running_stats=False)
        # Block 2
        self.conv2 = nn.Conv2d(nf, nf, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(nf, track_running_stats=False)
        # Block 3
        self.conv3 = nn.Conv2d(nf, nf, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(nf, track_running_stats=False)
        # Block 4
        self.conv4 = nn.Conv2d(nf, nf, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(nf, track_running_stats=False)
        # Classifier
        self.classifier = nn.Linear(nf, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all four blocks and the classifier.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.

        Returns:
            Logits tensor of shape ``(B, num_classes)``.
        """
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), 2)
        x = F.adaptive_avg_pool2d(F.relu(self.bn4(self.conv4(x))), 1)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def get_block_params(self, subset: str = "last_block") -> List[nn.Parameter]:
        """Return parameters for the specified layer subset.

        Args:
            subset: One of ``"last_block"``, ``"per_stage"``, ``"all"``.

        Returns:
            List of parameter tensors from the selected layers.
        """
        if subset == "last_block":
            return list(self.conv4.parameters()) + list(self.bn4.parameters())
        elif subset == "per_stage":
            params = []
            for block in [self.conv1, self.conv2, self.conv3, self.conv4]:
                params.extend(list(block.parameters()))
            return params
        else:  # "all"
            return list(self.parameters())


# ============================================================================
# Inline Stubs: Synthetic Episode Generation
# ============================================================================

def make_synthetic_episodes(
    n_episodes: int,
    n_way: int = 5,
    k_shot: int = 5,
    q_query: int = 15,
    img_size: int = 28,
    in_channels: int = 1,
    device: str = "cpu",
    seed: int = 42,
) -> List[Dict[str, torch.Tensor]]:
    """Generate synthetic few-shot episodes for benchmarking.

    Each episode contains a support set and query set of random images
    with random class labels. The images are Gaussian noise -- sufficient
    for measuring compute throughput without requiring real data.

    Args:
        n_episodes: Number of episodes to generate.
        n_way: Number of classes per episode.
        k_shot: Support samples per class.
        q_query: Query samples per class.
        img_size: Spatial dimension of input images.
        in_channels: Number of input channels.
        device: Torch device string.
        seed: RNG seed for reproducibility.

    Returns:
        List of episode dicts, each with keys ``"support_x"``,
        ``"support_y"``, ``"query_x"``, ``"query_y"``.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    dev = torch.device(device)
    episodes = []

    for _ in range(n_episodes):
        s_total = n_way * k_shot
        q_total = n_way * q_query

        support_x = torch.randn(
            s_total, in_channels, img_size, img_size,
            generator=gen,
        ).to(dev)
        support_y = torch.arange(n_way, device=dev).repeat_interleave(k_shot)

        query_x = torch.randn(
            q_total, in_channels, img_size, img_size,
            generator=gen,
        ).to(dev)
        query_y = torch.arange(n_way, device=dev).repeat_interleave(q_query)

        episodes.append({
            "support_x": support_x,
            "support_y": support_y,
            "query_x": query_x,
            "query_y": query_y,
        })
    return episodes


# ============================================================================
# Inline Stubs: Diagonal Fisher Computation
# ============================================================================

def compute_diagonal_fisher(
    probe: Conv4Probe,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    subset: str = "last_block",
) -> torch.Tensor:
    """Compute the diagonal of the Fisher information matrix.

    For each sample in the support set, computes the per-parameter
    gradient of the cross-entropy loss, squares it, and accumulates.
    The result is a flattened vector of diagonal Fisher values.

    This is the core computation in Task2Vec embedding extraction.
    The probe must be in inference mode with frozen weights.

    Args:
        probe: The Conv4Probe network (inference mode, frozen).
        support_x: Support images, shape ``(N, C, H, W)``.
        support_y: Support labels, shape ``(N,)``.
        subset: Parameter subset (``"last_block"``, ``"per_stage"``, ``"all"``).

    Returns:
        Flattened diagonal Fisher vector of shape ``(D,)`` where D is the
        total number of parameters in the selected subset.
    """
    params = probe.get_block_params(subset)
    n_params = sum(p.numel() for p in params)
    fisher = torch.zeros(n_params, device=support_x.device)

    # Put probe in inference mode
    was_training = probe.training
    probe.train(False)
    n_samples = support_x.size(0)

    for i in range(n_samples):
        probe.zero_grad()
        logits = probe(support_x[i : i + 1])
        loss = F.cross_entropy(logits, support_y[i : i + 1])
        loss.backward()

        offset = 0
        for p in params:
            if p.grad is not None:
                g = p.grad.detach().flatten()
                fisher[offset : offset + g.numel()] += g ** 2
                offset += g.numel()

    fisher /= n_samples

    # Restore training mode
    if was_training:
        probe.train(True)

    return fisher.detach()


def compute_diagonal_fisher_batch(
    probe: Conv4Probe,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    subset: str = "last_block",
) -> torch.Tensor:
    """Compute diagonal Fisher using a single batch forward pass.

    More efficient than per-sample computation when the batch fits in
    memory. Uses the full-batch gradient as an approximation.

    Args:
        probe: The Conv4Probe network (inference mode, frozen).
        support_x: Support images, shape ``(N, C, H, W)``.
        support_y: Support labels, shape ``(N,)``.
        subset: Parameter subset.

    Returns:
        Flattened diagonal Fisher vector of shape ``(D,)``.
    """
    params = probe.get_block_params(subset)

    was_training = probe.training
    probe.train(False)
    probe.zero_grad()

    logits = probe(support_x)
    loss = F.cross_entropy(logits, support_y)
    loss.backward()

    fisher_parts = []
    for p in params:
        if p.grad is not None:
            fisher_parts.append((p.grad.detach().flatten()) ** 2)
        else:
            fisher_parts.append(torch.zeros(p.numel(), device=support_x.device))

    if was_training:
        probe.train(True)

    return torch.cat(fisher_parts)


# ============================================================================
# Inline Stubs: Embedding Projection
# ============================================================================

def project_fisher_to_embedding(
    fisher: torch.Tensor,
    embedding_dim: int = 512,
    normalize: str = "log1p_l2",
) -> np.ndarray:
    """Project a diagonal Fisher vector to a fixed-dimensional embedding.

    Applies log1p transformation, then projects to ``embedding_dim`` via
    a deterministic random projection matrix (seeded from the Fisher
    dimension), and finally L2-normalizes.

    Args:
        fisher: Diagonal Fisher vector of shape ``(D,)``.
        embedding_dim: Target embedding dimensionality.
        normalize: Normalization mode (``"log1p_l2"``, ``"l2"``, ``"whiten"``).

    Returns:
        L2-normalized embedding as a float32 numpy array of shape ``(E,)``.
    """
    f = fisher.detach().cpu().float()

    # Apply log1p if requested
    if normalize in ("log1p_l2",):
        f = torch.log1p(f)

    # Deterministic random projection
    d = f.shape[0]
    gen = torch.Generator()
    gen.manual_seed(d)  # seed from input dimension for reproducibility
    proj = torch.randn(d, embedding_dim, generator=gen) / math.sqrt(d)
    emb = f @ proj

    # L2 normalize
    norm = emb.norm(2)
    if norm > 1e-12:
        emb = emb / norm

    return emb.numpy().astype(np.float32)


# ============================================================================
# Inline Stubs: Full Embedding Extraction Pipeline
# ============================================================================

def extract_embedding(
    probe: Conv4Probe,
    episode: Dict[str, torch.Tensor],
    embedding_dim: int = 512,
    subset: str = "last_block",
    normalize: str = "log1p_l2",
) -> np.ndarray:
    """Run the full embedding extraction pipeline for one episode.

    Steps:
        1. Forward pass through probe on support set.
        2. Compute diagonal Fisher information.
        3. Project Fisher to embedding dimension.
        4. L2-normalize.

    Args:
        probe: Frozen Conv4Probe in inference mode.
        episode: Dict with ``"support_x"`` and ``"support_y"`` keys.
        embedding_dim: Target embedding dimension.
        subset: Parameter subset for Fisher computation.
        normalize: Normalization method.

    Returns:
        L2-normalized embedding, shape ``(embedding_dim,)``.
    """
    fisher = compute_diagonal_fisher(
        probe, episode["support_x"], episode["support_y"], subset=subset,
    )
    return project_fisher_to_embedding(fisher, embedding_dim, normalize)


# ============================================================================
# Inline Stubs: K-Means Clustering
# ============================================================================

def kmeans(
    embeddings: np.ndarray,
    k: int = 8,
    max_iter: int = 100,
    seed: int = 42,
    distance: str = "cosine",
) -> Tuple[np.ndarray, np.ndarray]:
    """K-means clustering on embedding vectors.

    Supports cosine and euclidean distance. For cosine distance, vectors
    are L2-normalized before centroid computation and assignment.

    Args:
        embeddings: Array of shape ``(N, E)`` with N embedding vectors.
        k: Number of clusters.
        max_iter: Maximum number of iterations.
        seed: Random seed for centroid initialization.
        distance: Distance metric (``"cosine"`` or ``"euclidean"``).

    Returns:
        Tuple of ``(assignments, centroids)`` where ``assignments`` has
        shape ``(N,)`` and ``centroids`` has shape ``(k, E)``.
    """
    rng = np.random.RandomState(seed)
    n, e = embeddings.shape

    if distance == "cosine":
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        data = embeddings / norms
    else:
        data = embeddings.copy()

    # Initialize centroids via k-means++
    indices = [rng.randint(n)]
    for _ in range(1, k):
        centroids_so_far = data[indices]
        if distance == "cosine":
            sims = data @ centroids_so_far.T
            dists = 1.0 - sims
        else:
            diffs = data[:, None, :] - centroids_so_far[None, :, :]
            dists = np.sqrt((diffs ** 2).sum(axis=2))
        min_dists = dists.min(axis=1)
        min_dists = np.maximum(min_dists, 0.0)
        probs = min_dists / (min_dists.sum() + 1e-12)
        indices.append(rng.choice(n, p=probs))

    centroids = data[indices].copy()
    assignments = np.zeros(n, dtype=np.int64)

    for _ in range(max_iter):
        # Assignment step
        if distance == "cosine":
            c_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
            c_norms = np.maximum(c_norms, 1e-12)
            c_normed = centroids / c_norms
            sims = data @ c_normed.T
            new_assignments = sims.argmax(axis=1)
        else:
            diffs = data[:, None, :] - centroids[None, :, :]
            sq_dists = (diffs ** 2).sum(axis=2)
            new_assignments = sq_dists.argmin(axis=1)

        if np.array_equal(assignments, new_assignments):
            break
        assignments = new_assignments

        # Update step
        for ci in range(k):
            mask = assignments == ci
            if mask.sum() > 0:
                centroids[ci] = data[mask].mean(axis=0)
                if distance == "cosine":
                    cn = np.linalg.norm(centroids[ci])
                    if cn > 1e-12:
                        centroids[ci] /= cn

    return assignments, centroids


# ============================================================================
# Inline Stubs: Cosine Distance Matrix
# ============================================================================

def cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute the pairwise cosine distance matrix.

    Cosine distance is defined as ``1 - cosine_similarity``. The diagonal
    is exactly zero.

    Args:
        embeddings: Array of shape ``(N, E)``.

    Returns:
        Symmetric ``(N, N)`` float32 distance matrix.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normed = embeddings / norms
    cos_sim = normed @ normed.T
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return (1.0 - cos_sim).astype(np.float32)


# ============================================================================
# Inline Stubs: Curriculum Ordering
# ============================================================================

def easy_to_hard_ordering(
    embeddings: np.ndarray,
    cluster_assignments: np.ndarray,
    easy_cluster: int = 0,
    centroids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Order tasks from easy to hard based on distance from the easy cluster.

    Tasks closest to the easy cluster centroid come first. Distance is
    measured as cosine distance in embedding space.

    Args:
        embeddings: Array of shape ``(N, E)``.
        cluster_assignments: Cluster labels of shape ``(N,)``.
        easy_cluster: Index of the cluster considered "easy".
        centroids: Optional precomputed centroids of shape ``(k, E)``.

    Returns:
        Index array of shape ``(N,)`` sorted from easiest to hardest.
    """
    # Compute easy centroid
    if centroids is not None:
        easy_centroid = centroids[easy_cluster]
    else:
        mask = cluster_assignments == easy_cluster
        if mask.sum() > 0:
            easy_centroid = embeddings[mask].mean(axis=0)
        else:
            easy_centroid = embeddings.mean(axis=0)

    # Normalize
    ec_norm = np.linalg.norm(easy_centroid)
    if ec_norm > 1e-12:
        easy_centroid = easy_centroid / ec_norm

    e_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    e_norms = np.maximum(e_norms, 1e-12)
    normed = embeddings / e_norms

    # Cosine distance to easy centroid
    similarities = normed @ easy_centroid
    distances = 1.0 - similarities

    # Sort by ascending distance (easy first)
    order = np.argsort(distances)
    return order


def diversity_batch_construction(
    embeddings: np.ndarray,
    batch_size: int = 16,
    min_distance: float = 0.3,
    seed: int = 42,
) -> np.ndarray:
    """Construct a diversity-constrained meta-batch.

    Greedy selection: start with a random task, then iteratively add the
    task whose minimum distance to already-selected tasks is maximized,
    subject to the ``min_distance`` threshold.

    Args:
        embeddings: Array of shape ``(N, E)``.
        batch_size: Desired batch size.
        min_distance: Minimum pairwise cosine distance.
        seed: RNG seed.

    Returns:
        Index array of selected tasks, shape ``(B,)`` where B <= batch_size.
    """
    rng = np.random.RandomState(seed)
    n = embeddings.shape[0]
    if n == 0:
        return np.array([], dtype=np.int64)

    dist_mat = cosine_distance_matrix(embeddings)
    selected = [rng.randint(n)]
    available = set(range(n)) - {selected[0]}

    while len(selected) < batch_size and available:
        best_idx = -1
        best_min_dist = -1.0

        for idx in available:
            min_d = min(dist_mat[idx, s] for s in selected)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_idx = idx

        if best_min_dist < min_distance and len(selected) >= 2:
            break
        if best_idx < 0:
            break

        selected.append(best_idx)
        available.discard(best_idx)

    return np.array(selected, dtype=np.int64)


def kendall_tau(order_a: np.ndarray, order_b: np.ndarray) -> float:
    """Compute the Kendall tau distance between two orderings.

    Counts the number of pairwise discordances between the two rankings,
    normalized to the range [-1, 1].

    Args:
        order_a: First permutation of indices, shape ``(N,)``.
        order_b: Second permutation of indices, shape ``(N,)``.

    Returns:
        Kendall tau correlation coefficient in [-1, 1].
    """
    n = len(order_a)
    if n < 2:
        return 1.0

    # Convert orders to rank arrays
    rank_a = np.zeros(n, dtype=np.int64)
    rank_b = np.zeros(n, dtype=np.int64)
    for i, idx in enumerate(order_a):
        rank_a[idx] = i
    for i, idx in enumerate(order_b):
        rank_b[idx] = i

    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            diff_a = rank_a[i] - rank_a[j]
            diff_b = rank_b[i] - rank_b[j]
            if diff_a * diff_b > 0:
                concordant += 1
            elif diff_a * diff_b < 0:
                discordant += 1

    total_pairs = n * (n - 1) // 2
    if total_pairs == 0:
        return 1.0
    return (concordant - discordant) / total_pairs


# ============================================================================
# Inline Stubs: Registry I/O (Simplified)
# ============================================================================

def create_registry(
    n_entries: int,
    embedding_dim: int = 512,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Create a synthetic registry with random embeddings and metadata.

    Produces a list of metadata dicts and a contiguous embedding matrix,
    mimicking the data structures used by TaskEmbeddingRegistry.

    Args:
        n_entries: Number of registry entries.
        embedding_dim: Dimensionality of each embedding.
        seed: RNG seed.

    Returns:
        Tuple of (metadata_list, embedding_matrix) where the matrix
        has shape ``(n_entries, embedding_dim)``.
    """
    rng = np.random.RandomState(seed)
    emb_matrix = rng.randn(n_entries, embedding_dim).astype(np.float32)
    # L2 normalize
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    emb_matrix = emb_matrix / norms

    metadata = []
    datasets = ["omniglot", "mini-imagenet", "tiered-imagenet", "cifar-fs"]
    splits = ["train", "val", "test"]
    for i in range(n_entries):
        metadata.append({
            "task_id": f"task_{i:06d}",
            "dataset": datasets[i % len(datasets)],
            "split": splits[i % len(splits)],
            "n_way": 5,
            "k_shot": int(rng.choice([1, 5])),
            "q_query": 15,
            "class_ids": sorted(rng.choice(100, size=5, replace=False).tolist()),
            "support_indices": list(range(5)),
            "probe_signature": "conv4_last_block_v1",
            "extraction_seed": 42,
            "extraction_timestamp": time.time(),
            "code_version": "1.0",
            "diagnostics": {
                "fisher_norm": float(rng.uniform(1.0, 100.0)),
                "sparsity": float(rng.uniform(0.0, 1.0)),
            },
            "row_index": i,
        })

    return metadata, emb_matrix


def save_registry(
    path: Path,
    metadata: List[Dict[str, Any]],
    emb_matrix: np.ndarray,
    embedding_dim: int = 512,
) -> None:
    """Save a registry to JSONL + NPZ format.

    Args:
        path: Directory to write files into.
        metadata: List of metadata dicts, one per entry.
        emb_matrix: Embedding matrix of shape ``(N, E)``.
        embedding_dim: Dimensionality (for the header).
    """
    path.mkdir(parents=True, exist_ok=True)

    # Write JSONL
    jsonl_path = path / "task2vec_registry.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        header = {
            "_version": "1.0",
            "_probe_signature": "conv4_last_block_v1",
            "_created": time.time(),
            "_n_entries": len(metadata),
            "_embedding_dim": embedding_dim,
        }
        f.write(json.dumps(header, separators=(",", ":")) + "\n")
        for entry in metadata:
            f.write(json.dumps(entry, separators=(",", ":")) + "\n")

    # Write NPZ
    npz_path = path / "embeddings.npz"
    np.savez_compressed(str(npz_path), embeddings=emb_matrix)


def load_registry(path: Path) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Load a registry from JSONL + NPZ format.

    Args:
        path: Directory containing ``task2vec_registry.jsonl`` and
            ``embeddings.npz``.

    Returns:
        Tuple of (metadata_list, embedding_matrix).
    """
    jsonl_path = path / "task2vec_registry.jsonl"
    npz_path = path / "embeddings.npz"

    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")

    # Skip header (first line)
    metadata = []
    for line in lines[1:]:
        metadata.append(json.loads(line))

    with np.load(str(npz_path)) as npz:
        emb_matrix = npz["embeddings"].astype(np.float32)

    return metadata, emb_matrix


# ============================================================================
# Suite 1: Embedding Extraction Throughput
# ============================================================================

def bench_embedding(device: str = "cpu", quick: bool = False) -> BenchmarkSuite:
    """Benchmark embedding extraction throughput.

    Measures time for single-episode extraction, batch extraction at
    varying scales, and the effect of embedding dimensionality on
    extraction speed.

    Sub-benchmarks:
        - Single extraction (1 episode) for each embedding_dim (128, 256, 512)
        - Batch extraction (10, 50, 100 episodes) at dim=256
        - Throughput reporting: episodes/sec, ms/episode

    Args:
        device: Torch device string.
        quick: If True, use reduced batch sizes.

    Returns:
        Populated BenchmarkSuite with all measurements.
    """
    suite = BenchmarkSuite("embedding", device)
    print("  Running embedding extraction throughput benchmarks...")

    batch_sizes = [10, 50, 100] if not quick else [5, 10, 20]
    embedding_dims = [128, 256, 512]

    # ------------------------------------------------------------------
    # 1a. Single extraction, varying embedding_dim
    # ------------------------------------------------------------------
    for edim in embedding_dims:
        probe = Conv4Probe(
            in_channels=1, num_filters=64, num_classes=5,
        ).to(device)
        probe.train(False)
        episodes = make_synthetic_episodes(
            1, n_way=5, k_shot=5, device=device, seed=100 + edim,
        )
        ep = episodes[0]

        def _single(probe=probe, ep=ep, edim=edim):
            extract_embedding(probe, ep, embedding_dim=edim)

        t = suite.bench(
            _single,
            f"single_extract_dim{edim}",
            "ms/episode",
            warmup=2,
            repeats=8,
        )
        suite.results[-1].details.update(
            embedding_dim=edim,
            episodes_per_sec=1.0 / t if t > 0 else 0,
        )
        del probe; gc.collect()

    # ------------------------------------------------------------------
    # 1b. Batch extraction, varying episode count (at dim=256)
    # ------------------------------------------------------------------
    edim = 256
    for n_ep in batch_sizes:
        probe = Conv4Probe(
            in_channels=1, num_filters=64, num_classes=5,
        ).to(device)
        probe.train(False)
        episodes = make_synthetic_episodes(
            n_ep, n_way=5, k_shot=5, device=device, seed=200 + n_ep,
        )

        def _batch(probe=probe, episodes=episodes, edim=edim):
            for ep in episodes:
                extract_embedding(probe, ep, embedding_dim=edim)

        t = suite.bench(
            _batch,
            f"batch_extract_{n_ep}ep_dim{edim}",
            "ms",
            warmup=1,
            repeats=5,
        )
        per_ep_ms = (t * 1000) / n_ep
        suite.results[-1].details.update(
            n_episodes=n_ep,
            embedding_dim=edim,
            ms_per_episode=per_ep_ms,
            episodes_per_sec=n_ep / t if t > 0 else 0,
        )
        del probe, episodes; gc.collect()

    # ------------------------------------------------------------------
    # 1c. Throughput summary at each dim with fixed batch of 10
    # ------------------------------------------------------------------
    for edim in embedding_dims:
        probe = Conv4Probe(
            in_channels=1, num_filters=64, num_classes=5,
        ).to(device)
        probe.train(False)
        n_ep = 10
        episodes = make_synthetic_episodes(
            n_ep, n_way=5, k_shot=5, device=device, seed=300 + edim,
        )

        def _throughput(probe=probe, episodes=episodes, edim=edim):
            for ep in episodes:
                extract_embedding(probe, ep, embedding_dim=edim)

        t = suite.bench(
            _throughput,
            f"throughput_10ep_dim{edim}",
            "ms",
            warmup=1,
            repeats=5,
        )
        eps_sec = n_ep / t if t > 0 else 0
        suite.record(
            f"throughput_dim{edim}_ep_per_sec",
            "episodes/sec",
            eps_sec,
            {"embedding_dim": edim, "total_ms": t * 1000},
        )
        del probe, episodes; gc.collect()

    return suite


# ============================================================================
# Suite 2: Fisher Computation
# ============================================================================

def bench_fisher(device: str = "cpu", quick: bool = False) -> BenchmarkSuite:
    """Benchmark Fisher computation for varying support set sizes and subsets.

    Measures:
        - Per-sample Fisher speed for support sizes 5, 25, 50, 100
        - Fisher computation for different parameter subsets
        - Batch vs per-sample Fisher comparison

    Args:
        device: Torch device string.
        quick: If True, use reduced support sizes.

    Returns:
        Populated BenchmarkSuite.
    """
    suite = BenchmarkSuite("fisher", device)
    print("  Running Fisher computation benchmarks...")

    support_sizes = [5, 25, 50, 100] if not quick else [5, 10, 25]
    subsets = ["last_block", "per_stage", "all"]

    # ------------------------------------------------------------------
    # 2a. Varying support set size (last_block subset)
    # ------------------------------------------------------------------
    for ss in support_sizes:
        probe = Conv4Probe(
            in_channels=1, num_filters=64, num_classes=5,
        ).to(device)
        probe.train(False)

        # Generate one episode with enough support samples
        episodes = make_synthetic_episodes(
            1, n_way=5, k_shot=ss, device=device, seed=400 + ss,
        )
        ep = episodes[0]

        def _fisher(probe=probe, ep=ep):
            compute_diagonal_fisher(
                probe, ep["support_x"], ep["support_y"], subset="last_block",
            )

        t = suite.bench(
            _fisher,
            f"fisher_ss{ss}_last_block",
            "ms",
            warmup=2,
            repeats=6,
        )
        total_samples = 5 * ss
        suite.results[-1].details.update(
            support_size=total_samples,
            samples_per_sec=total_samples / t if t > 0 else 0,
            ms_per_sample=(t * 1000) / total_samples if total_samples > 0 else 0,
        )
        del probe; gc.collect()

    # ------------------------------------------------------------------
    # 2b. Varying parameter subset (fixed support size = 25)
    # ------------------------------------------------------------------
    ss_fixed = 25 if not quick else 10
    for subset in subsets:
        probe = Conv4Probe(
            in_channels=1, num_filters=64, num_classes=5,
        ).to(device)
        probe.train(False)
        n_params = sum(p.numel() for p in probe.get_block_params(subset))

        episodes = make_synthetic_episodes(
            1, n_way=5, k_shot=ss_fixed, device=device, seed=500,
        )
        ep = episodes[0]

        def _fisher_sub(probe=probe, ep=ep, subset=subset):
            compute_diagonal_fisher(
                probe, ep["support_x"], ep["support_y"], subset=subset,
            )

        t = suite.bench(
            _fisher_sub,
            f"fisher_subset_{subset}",
            "ms",
            warmup=2,
            repeats=6,
        )
        total_samples = 5 * ss_fixed
        suite.results[-1].details.update(
            subset=subset,
            n_params=n_params,
            support_size=total_samples,
            samples_per_sec=total_samples / t if t > 0 else 0,
            ms_per_sample=(t * 1000) / total_samples if total_samples > 0 else 0,
        )
        del probe; gc.collect()

    # ------------------------------------------------------------------
    # 2c. Batch vs per-sample Fisher comparison
    # ------------------------------------------------------------------
    ss_cmp = 25 if not quick else 10
    probe = Conv4Probe(
        in_channels=1, num_filters=64, num_classes=5,
    ).to(device)
    probe.train(False)
    episodes = make_synthetic_episodes(
        1, n_way=5, k_shot=ss_cmp, device=device, seed=600,
    )
    ep = episodes[0]

    def _per_sample():
        compute_diagonal_fisher(
            probe, ep["support_x"], ep["support_y"], subset="last_block",
        )

    def _batch_fisher():
        compute_diagonal_fisher_batch(
            probe, ep["support_x"], ep["support_y"], subset="last_block",
        )

    t_ps = suite.bench(
        _per_sample, "fisher_per_sample", "ms", warmup=2, repeats=6,
    )
    t_batch = suite.bench(
        _batch_fisher, "fisher_batch", "ms", warmup=2, repeats=6,
    )
    speedup = t_ps / t_batch if t_batch > 0 else 0
    suite.record(
        "fisher_batch_speedup", "x", speedup,
        {"per_sample_ms": t_ps * 1000, "batch_ms": t_batch * 1000},
    )
    del probe; gc.collect()

    return suite


# ============================================================================
# Suite 3: Clustering Speed
# ============================================================================

def bench_clustering(device: str = "cpu", quick: bool = False) -> BenchmarkSuite:
    """Benchmark K-means clustering and distance matrix computation.

    Measures:
        - K-means for varying number of tasks (50, 200, 1000)
        - K-means for varying k (4, 8, 16, 32)
        - Distance matrix computation for varying N

    Args:
        device: Torch device string (clustering runs on CPU/numpy).
        quick: If True, use reduced task counts.

    Returns:
        Populated BenchmarkSuite.
    """
    suite = BenchmarkSuite("clustering", device)
    print("  Running clustering speed benchmarks...")

    n_tasks_list = [50, 200, 1000] if not quick else [20, 50, 100]
    k_list = [4, 8, 16, 32] if not quick else [4, 8, 16]
    edim = 256

    # ------------------------------------------------------------------
    # 3a. K-means for varying n_tasks (k=8 fixed)
    # ------------------------------------------------------------------
    for nt in n_tasks_list:
        rng = np.random.RandomState(700 + nt)
        embs = rng.randn(nt, edim).astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / np.maximum(norms, 1e-12)

        def _km(embs=embs):
            kmeans(embs, k=8, max_iter=50, seed=42)

        suite.bench(
            _km,
            f"kmeans_n{nt}_k8",
            "ms/clustering",
            warmup=1,
            repeats=5,
        )
        suite.results[-1].details.update(n_tasks=nt, k=8, embedding_dim=edim)

    # ------------------------------------------------------------------
    # 3b. K-means for varying k (n_tasks=200 fixed)
    # ------------------------------------------------------------------
    nt_fixed = 200 if not quick else 50
    rng = np.random.RandomState(800)
    embs_fixed = rng.randn(nt_fixed, edim).astype(np.float32)
    norms = np.linalg.norm(embs_fixed, axis=1, keepdims=True)
    embs_fixed = embs_fixed / np.maximum(norms, 1e-12)

    for k in k_list:
        def _km_k(embs=embs_fixed, k=k):
            kmeans(embs, k=k, max_iter=50, seed=42)

        suite.bench(
            _km_k,
            f"kmeans_n{nt_fixed}_k{k}",
            "ms/clustering",
            warmup=1,
            repeats=5,
        )
        suite.results[-1].details.update(n_tasks=nt_fixed, k=k, embedding_dim=edim)

    # ------------------------------------------------------------------
    # 3c. Distance matrix computation for varying N
    # ------------------------------------------------------------------
    for nt in n_tasks_list:
        rng = np.random.RandomState(900 + nt)
        embs = rng.randn(nt, edim).astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / np.maximum(norms, 1e-12)

        def _dist(embs=embs):
            cosine_distance_matrix(embs)

        suite.bench(
            _dist,
            f"dist_matrix_n{nt}",
            "ms/distance_matrix",
            warmup=2,
            repeats=8,
        )
        suite.results[-1].details.update(
            n_tasks=nt,
            embedding_dim=edim,
            matrix_elements=nt * nt,
        )

    return suite


# ============================================================================
# Suite 4: Curriculum Ordering
# ============================================================================

def bench_curriculum(device: str = "cpu", quick: bool = False) -> BenchmarkSuite:
    """Benchmark curriculum ordering, diversity batches, and Kendall tau.

    Measures:
        - Easy-to-hard ordering for varying task counts (50, 200, 1000)
        - Diversity batch construction
        - Kendall tau computation
        - Full pipeline: cluster -> order -> build batch

    Args:
        device: Torch device string (curriculum runs on CPU/numpy).
        quick: If True, use reduced task counts.

    Returns:
        Populated BenchmarkSuite.
    """
    suite = BenchmarkSuite("curriculum", device)
    print("  Running curriculum ordering benchmarks...")

    n_tasks_list = [50, 200, 1000] if not quick else [20, 50, 100]
    edim = 256

    # ------------------------------------------------------------------
    # 4a. Easy-to-hard ordering for varying task counts
    # ------------------------------------------------------------------
    for nt in n_tasks_list:
        rng = np.random.RandomState(1000 + nt)
        embs = rng.randn(nt, edim).astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / np.maximum(norms, 1e-12)

        # Pre-cluster
        assignments, centroids = kmeans(embs, k=8, seed=42)

        def _e2h(embs=embs, assignments=assignments, centroids=centroids):
            easy_to_hard_ordering(embs, assignments, easy_cluster=0,
                                  centroids=centroids)

        suite.bench(
            _e2h,
            f"easy_to_hard_n{nt}",
            "ms/ordering",
            warmup=2,
            repeats=10,
        )
        suite.results[-1].details.update(n_tasks=nt, embedding_dim=edim)

    # ------------------------------------------------------------------
    # 4b. Diversity batch construction
    # ------------------------------------------------------------------
    for nt in n_tasks_list:
        rng = np.random.RandomState(1100 + nt)
        embs = rng.randn(nt, edim).astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / np.maximum(norms, 1e-12)

        batch_size = min(16, nt)

        def _div(embs=embs, bs=batch_size):
            diversity_batch_construction(embs, batch_size=bs, min_distance=0.3)

        suite.bench(
            _div,
            f"diversity_batch_n{nt}_b{batch_size}",
            "ms/batch",
            warmup=2,
            repeats=10,
        )
        suite.results[-1].details.update(
            n_tasks=nt, batch_size=batch_size, embedding_dim=edim,
        )

    # ------------------------------------------------------------------
    # 4c. Kendall tau computation
    # ------------------------------------------------------------------
    for nt in n_tasks_list:
        order_a = np.arange(nt)
        rng = np.random.RandomState(1200 + nt)
        order_b = rng.permutation(nt)

        def _tau(a=order_a, b=order_b):
            kendall_tau(a, b)

        suite.bench(
            _tau,
            f"kendall_tau_n{nt}",
            "ms",
            warmup=2,
            repeats=8,
        )
        suite.results[-1].details.update(n_tasks=nt)

    # ------------------------------------------------------------------
    # 4d. Full curriculum pipeline: cluster -> order -> build batch
    # ------------------------------------------------------------------
    for nt in n_tasks_list:
        rng = np.random.RandomState(1300 + nt)
        embs = rng.randn(nt, edim).astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / np.maximum(norms, 1e-12)

        def _full_pipeline(embs=embs, nt=nt):
            assignments, centroids = kmeans(embs, k=min(8, nt // 4 + 1), seed=42)
            order = easy_to_hard_ordering(
                embs, assignments, easy_cluster=0, centroids=centroids,
            )
            batch = diversity_batch_construction(
                embs, batch_size=min(16, nt), min_distance=0.3,
            )
            return order, batch

        suite.bench(
            _full_pipeline,
            f"full_pipeline_n{nt}",
            "ms",
            warmup=1,
            repeats=5,
        )
        suite.results[-1].details.update(n_tasks=nt, embedding_dim=edim)

    return suite


# ============================================================================
# Suite 5: Registry I/O
# ============================================================================

def bench_registry_io(device: str = "cpu", quick: bool = False) -> BenchmarkSuite:
    """Benchmark registry save/load throughput and file sizes.

    Measures:
        - Save time for varying registry sizes (100, 1000, 10000)
        - Load time for varying registry sizes
        - File sizes (JSONL + NPZ)
        - Bytes per entry

    Args:
        device: Torch device string (I/O is CPU-bound).
        quick: If True, use reduced registry sizes.

    Returns:
        Populated BenchmarkSuite.
    """
    suite = BenchmarkSuite("registry_io", device)
    print("  Running registry I/O benchmarks...")

    sizes = [100, 1000, 10000] if not quick else [50, 200, 1000]
    edim = 512

    for n_entries in sizes:
        metadata, emb_matrix = create_registry(n_entries, edim, seed=1400 + n_entries)

        # ------------------------------------------------------------------
        # 5a. Save time
        # ------------------------------------------------------------------
        with tempfile.TemporaryDirectory() as tmpdir:
            reg_path = Path(tmpdir) / "registry"

            def _save(md=metadata, em=emb_matrix, p=reg_path, ed=edim):
                save_registry(p, md, em, ed)

            t_save = suite.bench(
                _save,
                f"registry_save_n{n_entries}",
                "ms/save",
                warmup=1,
                repeats=5,
            )
            suite.results[-1].details.update(
                n_entries=n_entries,
                embedding_dim=edim,
                ms_per_entry=(t_save * 1000) / n_entries if n_entries > 0 else 0,
            )

            # Ensure registry is saved for load benchmark and file size
            save_registry(reg_path, metadata, emb_matrix, edim)

            # ------------------------------------------------------------------
            # 5b. Load time
            # ------------------------------------------------------------------
            def _load(p=reg_path):
                load_registry(p)

            t_load = suite.bench(
                _load,
                f"registry_load_n{n_entries}",
                "ms/load",
                warmup=1,
                repeats=5,
            )
            suite.results[-1].details.update(
                n_entries=n_entries,
                embedding_dim=edim,
                ms_per_entry=(t_load * 1000) / n_entries if n_entries > 0 else 0,
            )

            # ------------------------------------------------------------------
            # 5c. File sizes
            # ------------------------------------------------------------------
            jsonl_path = reg_path / "task2vec_registry.jsonl"
            npz_path = reg_path / "embeddings.npz"
            jsonl_size = jsonl_path.stat().st_size if jsonl_path.exists() else 0
            npz_size = npz_path.stat().st_size if npz_path.exists() else 0
            total_size = jsonl_size + npz_size
            bytes_per_entry = total_size / n_entries if n_entries > 0 else 0

            suite.record(
                f"registry_size_n{n_entries}",
                "bytes",
                total_size,
                {
                    "n_entries": n_entries,
                    "embedding_dim": edim,
                    "jsonl_bytes": jsonl_size,
                    "npz_bytes": npz_size,
                    "bytes_per_entry": bytes_per_entry,
                    "jsonl_pct": jsonl_size / total_size * 100 if total_size > 0 else 0,
                    "npz_pct": npz_size / total_size * 100 if total_size > 0 else 0,
                },
            )

    return suite


# ============================================================================
# Suite 6: Memory Profile
# ============================================================================

def bench_memory(device: str = "cpu", quick: bool = False) -> BenchmarkSuite:
    """Profile peak memory usage for embedding extraction and registries.

    Measures:
        - Peak memory during single embedding extraction (varying dim)
        - Peak memory during batch extraction (varying batch size)
        - Memory for registry storage at varying sizes
        - Memory per 1000 entries summary
        - Probe model parameter sizes

    Args:
        device: Torch device string.
        quick: If True, use reduced sizes.

    Returns:
        Populated BenchmarkSuite.
    """
    suite = BenchmarkSuite("memory", device)
    print("  Running memory profiling benchmarks...")

    # ------------------------------------------------------------------
    # 6a. Peak memory during single extraction
    # ------------------------------------------------------------------
    for edim in [128, 256, 512]:
        probe = Conv4Probe(
            in_channels=1, num_filters=64, num_classes=5,
        ).to(device)
        probe.train(False)
        episodes = make_synthetic_episodes(
            1, n_way=5, k_shot=5, device=device, seed=1500 + edim,
        )
        ep = episodes[0]

        suite.reset_mem()
        # Fisher computation requires gradients -- do not wrap in no_grad
        extract_embedding(probe, ep, embedding_dim=edim)
        peak = suite.peak_mem_mb()

        suite.record(
            f"mem_extract_dim{edim}", "MB", peak,
            {"embedding_dim": edim, "support_size": 25, "n_way": 5, "k_shot": 5},
        )
        del probe; gc.collect()

    # ------------------------------------------------------------------
    # 6b. Peak memory during batch extraction (varying batch size)
    # ------------------------------------------------------------------
    batch_sizes_mem = [5, 20, 50] if not quick else [3, 10, 20]
    edim = 256
    for n_ep in batch_sizes_mem:
        probe = Conv4Probe(
            in_channels=1, num_filters=64, num_classes=5,
        ).to(device)
        probe.train(False)
        episodes = make_synthetic_episodes(
            n_ep, n_way=5, k_shot=5, device=device, seed=1600 + n_ep,
        )

        suite.reset_mem()
        # Fisher computation requires gradients -- do not wrap in no_grad
        for ep in episodes:
            extract_embedding(probe, ep, embedding_dim=edim)
        peak = suite.peak_mem_mb()

        suite.record(
            f"mem_batch_{n_ep}ep_dim{edim}", "MB", peak,
            {"n_episodes": n_ep, "embedding_dim": edim},
        )
        del probe, episodes; gc.collect()

    # ------------------------------------------------------------------
    # 6c. Memory for registry at varying sizes
    # ------------------------------------------------------------------
    registry_sizes = [100, 1000, 10000] if not quick else [50, 200, 1000]
    edim_reg = 512

    for n_entries in registry_sizes:
        suite.reset_mem()
        metadata, emb_matrix = create_registry(n_entries, edim_reg, seed=1700 + n_entries)

        # Measure memory holding the full registry
        mem_baseline = suite.peak_mem_mb()
        emb_bytes = emb_matrix.nbytes

        suite.record(
            f"mem_registry_n{n_entries}", "MB", mem_baseline,
            {
                "n_entries": n_entries,
                "embedding_dim": edim_reg,
                "emb_matrix_bytes": emb_bytes,
                "emb_matrix_MB": emb_bytes / (1024 * 1024),
                "estimated_per_1000_entries_MB": (
                    (emb_bytes / (1024 * 1024)) * 1000 / max(n_entries, 1)
                ),
            },
        )
        del metadata, emb_matrix; gc.collect()

    # ------------------------------------------------------------------
    # 6d. Memory per 1000 entries summary
    # ------------------------------------------------------------------
    for n_entries in registry_sizes:
        # Pure embedding matrix size
        emb_mb = (n_entries * edim_reg * 4) / (1024 * 1024)  # float32
        per_1000 = emb_mb * 1000 / max(n_entries, 1)
        suite.record(
            f"mem_per_1000_entries_n{n_entries}",
            "MB/1000_entries",
            per_1000,
            {
                "n_entries": n_entries,
                "embedding_dim": edim_reg,
                "total_emb_MB": emb_mb,
            },
        )

    # ------------------------------------------------------------------
    # 6e. Probe model size
    # ------------------------------------------------------------------
    for nf in [32, 64, 128]:
        probe = Conv4Probe(in_channels=1, num_filters=nf, num_classes=5)
        param_count = sum(p.numel() for p in probe.parameters())
        param_mb = sum(p.numel() * 4 for p in probe.parameters()) / (1024 * 1024)
        suite.record(
            f"mem_probe_nf{nf}",
            "MB",
            param_mb,
            {"num_filters": nf, "param_count": param_count},
        )
        del probe; gc.collect()

    return suite


# ============================================================================
# Runner: Execute Suites
# ============================================================================

SUITE_REG: Dict[str, Any] = {
    "embedding": bench_embedding,
    "fisher": bench_fisher,
    "clustering": bench_clustering,
    "curriculum": bench_curriculum,
    "registry_io": bench_registry_io,
    "memory": bench_memory,
}


def run_suites(
    names: List[str],
    device: str = "cpu",
    quick: bool = False,
) -> Dict[str, BenchmarkSuite]:
    """Run the specified benchmark suites, catching crashes gracefully.

    Args:
        names: List of suite names to execute.
        device: Torch device string.
        quick: Whether to use reduced sizes.

    Returns:
        Dictionary mapping suite name to its BenchmarkSuite.
    """
    results: Dict[str, BenchmarkSuite] = {}
    for name in names:
        if name not in SUITE_REG:
            print(f"  [WARNING] Unknown suite: {name}, skipping.")
            continue
        print(f"\n--- Suite: {name} ---")
        try:
            results[name] = SUITE_REG[name](device, quick)
        except Exception as e:
            print(f"  [FATAL] Suite '{name}' crashed: {e}")
            traceback.print_exc()
            s = BenchmarkSuite(name, device)
            s.record(f"{name}_error", "error", 0.0, {"error": str(e)})
            results[name] = s
    return results


# ============================================================================
# Output Formatting
# ============================================================================

def format_report(suites: Dict[str, BenchmarkSuite], device: str) -> str:
    """Format all benchmark results into a human-readable table report.

    Each suite is printed as a separate table with aligned columns.
    Results include the measurement name, value, unit, and standard
    deviation (where applicable).

    Args:
        suites: Dictionary of suite_name -> BenchmarkSuite.
        device: Device string for the report header.

    Returns:
        Multi-line formatted report string.
    """
    lines = [
        "",
        _SEP,
        "  Task2Vec Embedding Pipeline -- Benchmark Report",
        _SEP,
        f"  Device:    {device}",
        f"  PyTorch:   {torch.__version__}",
        f"  NumPy:     {np.__version__}",
        f"  Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    for sn, suite in suites.items():
        lines.append(f"  --- Suite: {sn} {'=' * (60 - len(sn))}")
        lines.append("")

        # Table header
        lines.append(f"  {'Benchmark':<50s} {'Value':>12s}  {'Unit':<20s}")
        lines.append(f"  {'-' * 50} {'-' * 12}  {'-' * 20}")

        for r in suite.results:
            std = r.details.get("std_ms", 0.0)
            u = r.metric

            if u in (
                "ms", "ms/episode", "ms/step", "ms/clustering",
                "ms/distance_matrix", "ms/ordering", "ms/batch",
                "ms/save", "ms/load",
            ):
                tail = f"  (+/-{std:.2f})" if std > 0 else ""
                lines.append(
                    f"  {r.name:<50s} {r.value:>12.2f}  {u:<20s}{tail}"
                )
            elif u == "MB":
                lines.append(
                    f"  {r.name:<50s} {r.value:>12.2f}  {u:<20s}"
                )
            elif u == "MB/1000_entries":
                lines.append(
                    f"  {r.name:<50s} {r.value:>12.4f}  {u:<20s}"
                )
            elif u == "MB/step":
                lines.append(
                    f"  {r.name:<50s} {r.value:>12.4f}  {u:<20s}"
                )
            elif u == "episodes/sec":
                lines.append(
                    f"  {r.name:<50s} {r.value:>12.1f}  {u:<20s}"
                )
            elif u == "x":
                lines.append(
                    f"  {r.name:<50s} {r.value:>12.2f}x"
                )
            elif u == "bytes":
                if r.value > 1024 * 1024:
                    display = f"{r.value / (1024 * 1024):.2f} MB"
                elif r.value > 1024:
                    display = f"{r.value / 1024:.1f} KB"
                else:
                    display = f"{r.value:.0f} B"
                bpe = r.details.get("bytes_per_entry", 0)
                lines.append(
                    f"  {r.name:<50s} {display:>12s}  {u:<20s}"
                    f"  ({bpe:.0f} bytes/entry)"
                )
            elif u == "error":
                err = r.details.get("error", "unknown")
                lines.append(
                    f"  {r.name:<50s} {'ERROR':>12s}  {err}"
                )
            else:
                lines.append(
                    f"  {r.name:<50s} {r.value:>12.2f}  {u:<20s}"
                )

        # Suite summary
        n_results = len(suite.results)
        lines.append("")
        lines.append(f"  ({n_results} measurements)")
        lines.append("")

    total = sum(len(s.results) for s in suites.values())
    lines += [
        _SEP,
        f"  Total measurements: {total}",
        _SEP,
        "",
    ]
    return "\n".join(lines)


def suites_to_json(
    suites: Dict[str, BenchmarkSuite],
    device: str,
) -> dict:
    """Convert all suite results to a JSON-serializable dictionary.

    Args:
        suites: Dictionary of suite_name -> BenchmarkSuite.
        device: Device string.

    Returns:
        JSON-serializable dictionary with device, timestamp, and suite data.
    """
    return {
        "device": device,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "suites": {
            name: [asdict(r) for r in suite.results]
            for name, suite in suites.items()
        },
        "summary": {
            "total_measurements": sum(len(s.results) for s in suites.values()),
            "suites_run": list(suites.keys()),
        },
    }


# ============================================================================
# CLI Entry Point
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Task2Vec Embedding Pipeline -- Performance Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--suite",
        type=str,
        default=None,
        choices=ALL_SUITES,
        help="Run a specific benchmark suite (default: all).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run benchmarks on (default: cpu).",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        dest="json_output",
        metavar="PATH",
        help="Path to save JSON results.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use reduced sizes for faster runs.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_suites",
        help="List all available benchmark suites and exit.",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point: parse args, run suites, print report, save JSON."""
    args = parse_args()

    # ------------------------------------------------------------------
    # List mode
    # ------------------------------------------------------------------
    if args.list_suites:
        descs = {
            "embedding": "Embedding extraction throughput (episodes/sec, ms/ep)",
            "fisher": "Fisher computation speed (samples/sec, ms/sample)",
            "clustering": "K-means and distance matrix speed (ms/clustering)",
            "curriculum": "Curriculum ordering and batch construction (ms/ordering)",
            "registry_io": "Registry save/load throughput (ms/save, ms/load)",
            "memory": "Peak memory for extraction and registry (MB)",
        }
        print("\nAvailable benchmark suites:\n")
        for name in ALL_SUITES:
            print(f"  {name:20s}  {descs.get(name, '')}")
        print()
        return

    # ------------------------------------------------------------------
    # Validate CUDA
    # ------------------------------------------------------------------
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    # ------------------------------------------------------------------
    # Determine suites
    # ------------------------------------------------------------------
    names = [args.suite] if args.suite else list(ALL_SUITES)

    # ------------------------------------------------------------------
    # Print header
    # ------------------------------------------------------------------
    print(_SEP)
    print("  Task2Vec Embedding Pipeline -- Benchmark")
    print(_SEP)
    print(f"  Device:    {device}")
    print(f"  PyTorch:   {torch.__version__}")
    print(f"  NumPy:     {np.__version__}")
    print(f"  Quick:     {args.quick}")
    print(f"  Suites:    {', '.join(names)}")
    print(_SEP)

    # ------------------------------------------------------------------
    # Run benchmarks
    # ------------------------------------------------------------------
    suites = run_suites(names, device, quick=args.quick)

    # ------------------------------------------------------------------
    # Print report
    # ------------------------------------------------------------------
    report = format_report(suites, device)
    print(report)

    # ------------------------------------------------------------------
    # Save JSON results
    # ------------------------------------------------------------------
    if args.json_output:
        output_dir = os.path.dirname(os.path.abspath(args.json_output))
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        with open(args.json_output, "w") as f:
            json.dump(suites_to_json(suites, device), f, indent=2, default=str)
        print(f"  Results saved to: {args.json_output}")

    print("  Benchmark complete.")


if __name__ == "__main__":
    main()
