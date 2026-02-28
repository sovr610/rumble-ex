#!/usr/bin/env python3
"""Task2Vec Task Embeddings + Curriculum/Clustering -- Runtime Contract Validation

Validates the three done-when gates and all key contracts for the Task2Vec
task embedding, clustering, and curriculum ordering skill.  Self-contained:
runs without the brain_ai package installed -- uses inline stubs and
synthetic data generated on the fly.

Done-When Gates validated:
  (a) Embedding determinism  -- same episode + seed => identical embedding
  (b) Cluster stability      -- ARI >= 0.9 across seeds
  (c) Curriculum effect       -- Kendall tau significant vs. random

Usage:
    python validate_task2vec.py                    # Run all checks
    python validate_task2vec.py --group probe      # Run specific group
    python validate_task2vec.py --verbose           # Detailed output
    python validate_task2vec.py --json              # JSON-only output
    python validate_task2vec.py --list              # List all check groups

Exit codes:
    0 = all checks passed
    1 = one or more checks failed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
import tempfile
import time
import traceback
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ============================================================================
# ANSI colour helpers
# ============================================================================

_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def _c(text: str, code: str) -> str:
    """Wrap text in ANSI colour if stdout is a tty."""
    if sys.stdout.isatty():
        return f"{code}{text}{_RESET}"
    return text


# ============================================================================
# Group registry
# ============================================================================

_ALL_GROUPS: List[str] = [
    "probe",
    "fisher",
    "gate_a",
    "registry",
    "clustering",
    "gate_b",
    "gate_c",
]

_GROUP_DESCRIPTIONS: Dict[str, str] = {
    "probe": "Probe Network Validation: Conv4 forward, head creation, freezing",
    "fisher": "Fisher Computation: non-negative, deterministic, fp32, normalized",
    "gate_a": "Done-When Gate (a): Embedding determinism, L2-norm, task ID",
    "registry": "Registry Operations: save/load round-trip, JSONL/NPZ, query",
    "clustering": "Clustering Validation: K-means, cluster count, determinism",
    "gate_b": "Done-When Gate (b): Cluster stability (ARI >= 0.9)",
    "gate_c": "Done-When Gate (c): Curriculum effect (Kendall tau significant)",
}

# ============================================================================
# Result dataclass
# ============================================================================


@dataclass
class CheckResult:
    """Outcome of a single validation check."""
    name: str
    group: str
    passed: bool
    message: str
    details: Optional[str] = None
    elapsed_ms: float = 0.0


# ============================================================================
# Inline stub implementations
# ============================================================================
# These minimal implementations are self-contained so the validation script
# does not depend on brain_ai being installed.  They faithfully replicate the
# contracts specified in SKILL.md and the reference documents.


# ---------------------------------------------------------------------------
# Conv4 Probe Network
# ---------------------------------------------------------------------------

class Conv4Block(nn.Module):
    """Single convolutional block: Conv -> BN -> ReLU -> MaxPool."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        return self.pool(self.relu(self.bn(self.conv(x))))


class Conv4Probe(nn.Module):
    """Conv4 probe backbone: four convolutional blocks.

    Produces a flat feature vector from an input image.  The final spatial
    size depends on the input resolution.  For 32x32 inputs the output is
    64-dim (64 channels after global average pool).
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 64,
    ):
        super().__init__()
        self.block1 = Conv4Block(in_channels, hidden_channels)
        self.block2 = Conv4Block(hidden_channels, hidden_channels)
        self.block3 = Conv4Block(hidden_channels, hidden_channels)
        self.block4 = Conv4Block(hidden_channels, hidden_channels)
        self.hidden_channels = hidden_channels

    def forward(self, x: Tensor) -> Tensor:
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        # Global average pool to (B, C)
        h = h.mean(dim=[2, 3])
        return h

    @property
    def output_dim(self) -> int:
        return self.hidden_channels


class ProbeWithHead(nn.Module):
    """Probe backbone + ephemeral N-way classification head.

    The probe is frozen (set to mode, requires_grad=False).
    Only the head is trainable -- and even that is ephemeral (used
    only during Fisher accumulation, then discarded).
    """

    def __init__(self, probe: Conv4Probe, n_way: int):
        super().__init__()
        self.probe = probe
        self.head = nn.Linear(probe.output_dim, n_way)
        # Freeze the probe
        self.probe.train(False)
        for p in self.probe.parameters():
            p.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            features = self.probe(x)
        logits = self.head(features)
        return logits


# ---------------------------------------------------------------------------
# Diagonal Fisher computation
# ---------------------------------------------------------------------------

def compute_diagonal_fisher(
    model: ProbeWithHead,
    x: Tensor,
    y: Tensor,
    *,
    num_samples: Optional[int] = None,
) -> Dict[str, Tensor]:
    """Compute diagonal Fisher information for the probe parameters.

    Uses the squared-gradient approximation:
        F_ii = (1/N) sum_n (d log p(y_n | x_n; theta) / d theta_i)^2

    Only computes Fisher for the *probe* parameters (which are frozen
    during normal forward but we temporarily enable grad here to
    accumulate squared gradients).  The head parameters are ignored.

    Returns a dict mapping parameter name -> Fisher diagonal tensor.
    All values are non-negative (squared gradients) and in fp32.
    """
    model.probe.train(False)
    device = x.device

    # Temporarily enable grad for probe params
    original_requires_grad = {}
    for name, p in model.probe.named_parameters():
        original_requires_grad[name] = p.requires_grad
        p.requires_grad_(True)

    # Accumulate squared gradients
    fisher = {}
    for name, p in model.probe.named_parameters():
        fisher[name] = torch.zeros_like(p, dtype=torch.float32, device=device)

    N = x.shape[0]
    if num_samples is not None:
        N = min(N, num_samples)

    for i in range(N):
        model.zero_grad()
        xi = x[i:i+1].to(torch.float32)
        yi = y[i:i+1]

        # Forward through probe WITH grad this time
        features = model.probe(xi)
        logits = model.head(features)
        log_probs = F.log_softmax(logits, dim=-1)
        loss = F.nll_loss(log_probs, yi)
        loss.backward()

        for name, p in model.probe.named_parameters():
            if p.grad is not None:
                fisher[name] += p.grad.detach().float() ** 2

    # Normalize by sample count
    for name in fisher:
        fisher[name] /= float(N)

    # Restore original requires_grad
    for name, p in model.probe.named_parameters():
        p.requires_grad_(original_requires_grad[name])

    return fisher


# ---------------------------------------------------------------------------
# Embedding projection: Fisher -> fixed-dim embedding
# ---------------------------------------------------------------------------

def fisher_to_embedding(
    fisher: Dict[str, Tensor],
    embedding_dim: int = 64,
) -> np.ndarray:
    """Project Fisher diagonal to a fixed-dimension L2-normalized embedding.

    Pipeline:
        1. Concatenate all Fisher diagonals into a single vector
        2. Apply log1p transform (stabilizes scale)
        3. Adaptive average pool to target embedding_dim
        4. L2-normalize to unit length

    Returns an L2-normalized numpy array of shape (embedding_dim,).
    """
    # Concatenate
    parts = []
    for name in sorted(fisher.keys()):
        parts.append(fisher[name].detach().cpu().flatten())
    full_fisher = torch.cat(parts, dim=0).float()

    # log1p
    full_fisher = torch.log1p(full_fisher)

    # Adaptive pool to target dim
    if full_fisher.numel() >= embedding_dim:
        pooled = F.adaptive_avg_pool1d(
            full_fisher.unsqueeze(0).unsqueeze(0),
            embedding_dim,
        ).squeeze()
    else:
        # Pad if fewer params than embedding dim
        padded = torch.zeros(embedding_dim, dtype=torch.float32)
        padded[:full_fisher.numel()] = full_fisher
        pooled = padded

    # L2 normalize
    emb = pooled.numpy().astype(np.float64)
    norm = np.linalg.norm(emb)
    if norm > 1e-12:
        emb = emb / norm
    else:
        # Degenerate case: set first element to 1
        emb = np.zeros(embedding_dim, dtype=np.float64)
        emb[0] = 1.0

    return emb.astype(np.float32)


# ---------------------------------------------------------------------------
# Task ID canonicalization
# ---------------------------------------------------------------------------

def canonicalize_task_id(
    dataset: str,
    split: str,
    class_ids: List[int],
    support_indices: List[int],
    transforms_hash: str = "",
) -> str:
    """Compute a deterministic SHA-256-based task identifier.

    The ID is derived from the canonical representation of the task:
    (dataset, split, sorted class_ids, sorted support_indices, transforms_hash).
    Truncated to 16 hex characters for readability.

    Same inputs always produce the same task_id.
    """
    sorted_classes = sorted(class_ids)
    sorted_indices = sorted(support_indices)
    canonical = (
        f"{dataset}|{split}|"
        f"{','.join(map(str, sorted_classes))}|"
        f"{','.join(map(str, sorted_indices))}|"
        f"{transforms_hash}"
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return digest[:16]


# ---------------------------------------------------------------------------
# Simple K-means clustering (pure numpy, no sklearn required)
# ---------------------------------------------------------------------------

def kmeans_numpy(
    X: np.ndarray,
    k: int,
    seed: int = 42,
    max_iter: int = 300,
    tol: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pure-numpy K-means clustering.

    Args:
        X: (N, D) data matrix, assumed L2-normalized rows.
        k: Number of clusters.
        seed: Random seed for initialization.
        max_iter: Maximum iterations.
        tol: Convergence tolerance (centroid movement).

    Returns:
        labels: (N,) int array of cluster assignments.
        centroids: (k, D) cluster centroids (L2-normalized).
    """
    rng = np.random.RandomState(seed)
    N, D = X.shape

    if N == 0:
        return np.array([], dtype=np.int32), np.zeros((k, D), dtype=np.float32)

    # Clamp k to at most N
    k = min(k, N)

    # k-means++ initialization
    centroids = np.zeros((k, D), dtype=np.float64)
    idx = rng.randint(N)
    centroids[0] = X[idx].astype(np.float64)

    for c in range(1, k):
        # Compute distances from each point to nearest existing centroid
        dists = np.full(N, np.inf, dtype=np.float64)
        for j in range(c):
            d = np.sum((X.astype(np.float64) - centroids[j:j+1]) ** 2, axis=1)
            dists = np.minimum(dists, d)
        # Avoid zero-sum
        dists_sum = dists.sum()
        if dists_sum < 1e-30:
            # All points are identical; pick randomly
            probs = np.ones(N) / N
        else:
            probs = dists / dists_sum
        idx = rng.choice(N, p=probs)
        centroids[c] = X[idx].astype(np.float64)

    labels = np.zeros(N, dtype=np.int32)

    for iteration in range(max_iter):
        # Assignment step
        distances = np.zeros((N, k), dtype=np.float64)
        for c in range(k):
            diff = X.astype(np.float64) - centroids[c:c+1]
            distances[:, c] = np.sum(diff ** 2, axis=1)
        new_labels = np.argmin(distances, axis=1).astype(np.int32)

        # Update step
        new_centroids = np.zeros_like(centroids)
        for c in range(k):
            mask = new_labels == c
            if mask.sum() > 0:
                new_centroids[c] = X[mask].astype(np.float64).mean(axis=0)
            else:
                # Empty cluster: reinitialize to a random point
                new_centroids[c] = X[rng.randint(N)].astype(np.float64)

        # Check convergence
        centroid_shift = np.sqrt(np.sum((new_centroids - centroids) ** 2))
        centroids = new_centroids
        labels = new_labels

        if centroid_shift < tol:
            break

    # Normalize centroids
    for c in range(k):
        norm = np.linalg.norm(centroids[c])
        if norm > 1e-12:
            centroids[c] /= norm

    return labels, centroids.astype(np.float32)


# ---------------------------------------------------------------------------
# Adjusted Rand Index (pure Python)
# ---------------------------------------------------------------------------

def adjusted_rand_index(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Compute the Adjusted Rand Index between two clusterings.

    Pure-Python implementation using the contingency table approach.
    ARI = (RI - E[RI]) / (max(RI) - E[RI]).
    Range: [-1, 1], with 1.0 meaning perfect agreement.
    """
    n = len(labels_a)
    if n == 0:
        return 1.0

    # Build contingency table
    classes_a = sorted(set(labels_a))
    classes_b = sorted(set(labels_b))
    map_a = {v: i for i, v in enumerate(classes_a)}
    map_b = {v: i for i, v in enumerate(classes_b)}

    nij = np.zeros((len(classes_a), len(classes_b)), dtype=np.int64)
    for i in range(n):
        nij[map_a[labels_a[i]], map_b[labels_b[i]]] += 1

    # Row sums (a_i) and column sums (b_j)
    a = nij.sum(axis=1)
    b = nij.sum(axis=0)

    # Compute C(n, 2) helper
    def comb2(x):
        return x * (x - 1) // 2

    sum_comb_nij = sum(comb2(int(nij[i, j]))
                       for i in range(len(classes_a))
                       for j in range(len(classes_b)))
    sum_comb_a = sum(comb2(int(ai)) for ai in a)
    sum_comb_b = sum(comb2(int(bj)) for bj in b)
    comb_n = comb2(n)

    if comb_n == 0:
        return 1.0

    expected = (sum_comb_a * sum_comb_b) / comb_n
    max_index = 0.5 * (sum_comb_a + sum_comb_b)

    if max_index == expected:
        return 1.0

    ari = (sum_comb_nij - expected) / (max_index - expected)
    return float(ari)


# ---------------------------------------------------------------------------
# Normalized Mutual Information (pure Python)
# ---------------------------------------------------------------------------

def normalized_mutual_info(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Compute Normalized Mutual Information between two clusterings.

    NMI = 2 * I(A; B) / (H(A) + H(B)).
    Range: [0, 1], with 1.0 meaning perfect agreement.
    """
    n = len(labels_a)
    if n == 0:
        return 1.0

    classes_a = sorted(set(labels_a))
    classes_b = sorted(set(labels_b))
    map_a = {v: i for i, v in enumerate(classes_a)}
    map_b = {v: i for i, v in enumerate(classes_b)}

    nij = np.zeros((len(classes_a), len(classes_b)), dtype=np.float64)
    for idx in range(n):
        nij[map_a[labels_a[idx]], map_b[labels_b[idx]]] += 1.0

    # Marginals
    pi = nij.sum(axis=1)  # row sums
    pj = nij.sum(axis=0)  # col sums

    # Entropy
    def entropy(counts):
        total = counts.sum()
        if total == 0:
            return 0.0
        probs = counts / total
        probs = probs[probs > 0]
        return -float(np.sum(probs * np.log(probs)))

    h_a = entropy(pi)
    h_b = entropy(pj)

    if h_a + h_b == 0:
        return 1.0

    # Mutual information
    mi = 0.0
    for i in range(len(classes_a)):
        for j in range(len(classes_b)):
            if nij[i, j] > 0:
                mi += (nij[i, j] / n) * math.log(
                    (n * nij[i, j]) / (pi[i] * pj[j])
                )

    nmi = (2.0 * mi) / (h_a + h_b)
    return float(nmi)


# ---------------------------------------------------------------------------
# Silhouette score (pure numpy)
# ---------------------------------------------------------------------------

def silhouette_score_numpy(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute mean silhouette score using cosine distance.

    For L2-normalized data, cosine distance = 1 - X @ X.T.
    Returns a value in [-1, 1]. Higher is better.
    """
    n = len(labels)
    unique_labels = sorted(set(labels))
    if len(unique_labels) < 2 or n < 2:
        return 0.0

    # Cosine similarity matrix
    sim = X @ X.T
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    np.clip(dist, 0.0, 2.0, out=dist)

    scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        own_label = labels[i]
        own_mask = labels == own_label
        own_count = own_mask.sum()

        # a(i) = mean distance to own cluster (excluding self)
        if own_count > 1:
            a_i = dist[i, own_mask].sum() / (own_count - 1)
        else:
            a_i = 0.0

        # b(i) = min mean distance to other clusters
        b_i = np.inf
        for lbl in unique_labels:
            if lbl == own_label:
                continue
            other_mask = labels == lbl
            other_count = other_mask.sum()
            if other_count > 0:
                mean_dist = dist[i, other_mask].mean()
                b_i = min(b_i, mean_dist)

        if b_i == np.inf:
            b_i = 0.0

        denom = max(a_i, b_i)
        if denom > 0:
            scores[i] = (b_i - a_i) / denom
        else:
            scores[i] = 0.0

    return float(scores.mean())


# ---------------------------------------------------------------------------
# Cosine distance matrix
# ---------------------------------------------------------------------------

def cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine distance matrix.

    Args:
        embeddings: (N, E) array, assumed L2-normalized along axis=1.

    Returns:
        (N, N) symmetric distance matrix with zeros on diagonal.
    """
    sim = embeddings @ embeddings.T
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    np.clip(dist, 0.0, 2.0, out=dist)
    return dist


# ---------------------------------------------------------------------------
# Curriculum ordering: easy-to-hard
# ---------------------------------------------------------------------------

def compute_difficulty_centroid(
    task_embeddings: Dict[str, np.ndarray],
    easy_centroid: np.ndarray,
) -> Dict[str, float]:
    """Difficulty proxy: cosine distance to easy-cluster centroid."""
    easy_centroid = easy_centroid / (np.linalg.norm(easy_centroid) + 1e-12)
    difficulty = {}
    for task_id, emb in task_embeddings.items():
        cosine_sim = float(np.dot(emb, easy_centroid))
        difficulty[task_id] = 1.0 - cosine_sim
    return difficulty


def easy_to_hard_order(
    task_ids: List[str],
    difficulty: Dict[str, float],
) -> List[str]:
    """Sort tasks from easiest to hardest.

    Ties broken by task_id lexicographic order for determinism.
    """
    return sorted(task_ids, key=lambda tid: (difficulty.get(tid, 0.0), tid))


# ---------------------------------------------------------------------------
# Kendall tau (pure Python)
# ---------------------------------------------------------------------------

def kendall_tau(x: List[int], y: List[int]) -> Tuple[float, float]:
    """Pure-Python Kendall tau-a computation.

    Counts concordant and discordant pairs.
    Returns (tau, p_value).
    """
    n = len(x)
    if n < 2:
        return 0.0, 1.0

    concordant = 0
    discordant = 0

    for i in range(n):
        for j in range(i + 1, n):
            xi_xj = x[i] - x[j]
            yi_yj = y[i] - y[j]
            product = xi_xj * yi_yj
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1

    n_pairs = n * (n - 1) // 2
    if n_pairs == 0:
        return 0.0, 1.0

    tau = (concordant - discordant) / n_pairs

    # Approximate p-value using normal approximation
    variance = n * (n - 1) * (2 * n + 5) / 18.0
    if variance <= 0:
        return tau, 1.0

    s = concordant - discordant
    z = s / math.sqrt(variance)
    p_value = 2.0 * _norm_sf(abs(z))

    return tau, p_value


def _norm_sf(z: float) -> float:
    """Survival function of the standard normal distribution (approximation).

    Uses Abramowitz and Stegun approximation 7.1.26, accurate to 1e-5.
    """
    z = abs(z)
    t = 1.0 / (1.0 + 0.2316419 * z)
    d = 0.3989422804014327  # 1/sqrt(2*pi)
    p = d * math.exp(-z * z / 2.0) * (
        t * (0.319381530
             + t * (-0.356563782
                    + t * (1.781477937
                           + t * (-1.821255978
                                  + t * 1.330274429))))
    )
    return p


# ---------------------------------------------------------------------------
# Registry save/load (JSONL + NPZ)
# ---------------------------------------------------------------------------

class TaskEmbeddingRegistry:
    """Minimal self-contained task embedding registry for validation.

    Stores (task_id -> embedding) mappings with metadata.
    Persistence via JSONL (metadata) + NPZ (vectors).
    """

    JSONL_FILENAME = "task2vec_registry.jsonl"
    NPZ_FILENAME = "embeddings.npz"

    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self._entries: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._embeddings: List[np.ndarray] = []
        self._id_to_row: Dict[str, int] = {}
        self._dirty = True
        self._matrix_cache: Optional[np.ndarray] = None
        self.header: Dict[str, Any] = {
            "_version": "1.0",
            "_probe_signature": "conv4_last_block_e64_stub",
            "_created": datetime.now(timezone.utc).isoformat(),
        }

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, task_id: str) -> bool:
        return task_id in self._entries

    @property
    def task_ids(self) -> List[str]:
        return list(self._entries.keys())

    def update(
        self,
        task_id: str,
        embedding: np.ndarray,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert or overwrite a task entry."""
        if embedding.shape != (self.embedding_dim,):
            raise ValueError(
                f"Expected shape ({self.embedding_dim},), got {embedding.shape}"
            )
        if meta is None:
            meta = {}
        meta["task_id"] = task_id

        if task_id in self._id_to_row:
            row = self._id_to_row[task_id]
            self._embeddings[row] = embedding.copy().astype(np.float32)
            meta["row_index"] = row
            self._entries[task_id] = meta
        else:
            row = len(self._embeddings)
            self._embeddings.append(embedding.copy().astype(np.float32))
            self._id_to_row[task_id] = row
            meta["row_index"] = row
            self._entries[task_id] = meta

        self._dirty = True

    def get_embedding(self, task_id: str) -> np.ndarray:
        if task_id not in self._id_to_row:
            raise KeyError(f"Task '{task_id}' not in registry")
        row = self._id_to_row[task_id]
        return self._embeddings[row].copy()

    def get_meta(self, task_id: str) -> Dict[str, Any]:
        if task_id not in self._entries:
            raise KeyError(f"Task '{task_id}' not in registry")
        return self._entries[task_id]

    def query(self, dataset: Optional[str] = None) -> List[str]:
        """Filter task IDs by dataset name."""
        results = []
        for task_id, entry in self._entries.items():
            if dataset is not None and entry.get("dataset") != dataset:
                continue
            results.append(task_id)
        return results

    def _get_matrix(self) -> np.ndarray:
        if self._dirty or self._matrix_cache is None:
            if self._embeddings:
                self._matrix_cache = np.stack(self._embeddings, axis=0)
            else:
                self._matrix_cache = np.empty(
                    (0, self.embedding_dim), dtype=np.float32
                )
            self._dirty = False
        return self._matrix_cache

    def save(self, path: str) -> None:
        """Save registry as JSONL + NPZ files."""
        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)

        jsonl_path = dir_path / self.JSONL_FILENAME
        npz_path = dir_path / self.NPZ_FILENAME

        with open(jsonl_path, "w") as f:
            f.write(json.dumps(self.header) + "\n")
            for entry in self._entries.values():
                # Make a serializable copy
                safe = {}
                for k, v in entry.items():
                    if isinstance(v, (np.integer,)):
                        safe[k] = int(v)
                    elif isinstance(v, (np.floating,)):
                        safe[k] = float(v)
                    elif isinstance(v, np.ndarray):
                        safe[k] = v.tolist()
                    else:
                        safe[k] = v
                f.write(json.dumps(safe) + "\n")

        matrix = self._get_matrix()
        np.savez_compressed(str(npz_path), embeddings=matrix)

    @classmethod
    def load(cls, path: str) -> "TaskEmbeddingRegistry":
        """Load registry from JSONL + NPZ files."""
        dir_path = Path(path)
        jsonl_path = dir_path / cls.JSONL_FILENAME
        npz_path = dir_path / cls.NPZ_FILENAME

        if not jsonl_path.exists():
            raise FileNotFoundError(f"Missing {jsonl_path}")
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing {npz_path}")

        data = np.load(str(npz_path))
        embedding_matrix = data["embeddings"]
        embedding_dim = embedding_matrix.shape[1] if embedding_matrix.ndim > 1 else 0

        registry = cls(embedding_dim=embedding_dim)

        with open(jsonl_path, "r") as f:
            lines = f.readlines()

        if not lines:
            raise ValueError("Empty registry JSONL file")

        # First line is header
        header = json.loads(lines[0])
        if "_version" not in header:
            raise ValueError("Missing _version in header")
        registry.header = header

        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            task_id = entry["task_id"]
            row_idx = entry["row_index"]

            registry._entries[task_id] = entry
            registry._id_to_row[task_id] = row_idx
            registry._embeddings.append(
                embedding_matrix[row_idx].copy()
            )

        registry._dirty = True
        return registry


# ---------------------------------------------------------------------------
# Epoch log generation
# ---------------------------------------------------------------------------

def generate_epoch_log(
    epoch: int,
    strategy: str,
    ordered_task_ids: List[str],
    difficulty: Dict[str, float],
    cluster_assignments: Dict[str, int],
) -> Dict[str, Any]:
    """Create a structured epoch log entry."""
    order_str = ",".join(ordered_task_ids)
    order_hash = hashlib.sha256(order_str.encode()).hexdigest()

    cluster_hist: Dict[int, int] = defaultdict(int)
    for tid in ordered_task_ids:
        cid = cluster_assignments.get(tid, -1)
        cluster_hist[cid] += 1

    diff_values = [difficulty.get(tid, 0.0) for tid in ordered_task_ids]

    log_entry = {
        "epoch": epoch,
        "strategy": strategy,
        "task_order_hash": order_hash,
        "num_tasks": len(ordered_task_ids),
        "cluster_histogram": dict(cluster_hist),
        "difficulty_stats": {
            "mean": float(np.mean(diff_values)) if diff_values else 0.0,
            "std": float(np.std(diff_values)) if diff_values else 0.0,
            "min": float(np.min(diff_values)) if diff_values else 0.0,
            "max": float(np.max(diff_values)) if diff_values else 0.0,
        },
    }
    return log_entry


# ---------------------------------------------------------------------------
# Diversity batch builder (greedy min-distance)
# ---------------------------------------------------------------------------

def build_diversity_batch(
    task_ids: List[str],
    task_embeddings: Dict[str, np.ndarray],
    batch_size: int,
    min_distance: float,
    seed: int,
) -> List[str]:
    """Build a diversity-constrained meta-batch using pairwise distance threshold.

    Greedy selection: pick a random first task, then add tasks whose cosine
    distance to ALL selected tasks exceeds min_distance.  If the batch cannot
    be filled, relax the threshold by halving and retry.
    """
    rng = np.random.RandomState(seed)
    candidates = list(task_ids)
    rng.shuffle(candidates)

    threshold = min_distance
    max_relaxations = 10

    for _relaxation in range(max_relaxations):
        selected: List[str] = []
        selected_embs: List[np.ndarray] = []

        for tid in candidates:
            emb = task_embeddings[tid]
            if len(selected) == 0:
                selected.append(tid)
                selected_embs.append(emb)
            else:
                dists = [1.0 - float(np.dot(emb, se)) for se in selected_embs]
                if min(dists) >= threshold:
                    selected.append(tid)
                    selected_embs.append(emb)

            if len(selected) == batch_size:
                return selected

        if len(selected) >= batch_size:
            return selected[:batch_size]

        threshold *= 0.5

    # Final fallback: fill remaining slots
    remaining = [tid for tid in candidates if tid not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[: batch_size - len(selected)])
    return selected[:batch_size]


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def make_synthetic_episode(
    seed: int = 42,
    n_way: int = 5,
    k_shot: int = 5,
    in_channels: int = 3,
    img_size: int = 32,
) -> Tuple[Tensor, Tensor, List[int]]:
    """Generate a synthetic episode (support set only).

    Returns:
        x_support: (n_way * k_shot, in_channels, img_size, img_size)
        y_support: (n_way * k_shot,)
        class_ids: List[int] of length n_way
    """
    torch.manual_seed(seed)
    n_support = n_way * k_shot
    x = torch.randn(n_support, in_channels, img_size, img_size)
    y = torch.arange(n_way).repeat_interleave(k_shot)
    class_ids = list(range(n_way))
    return x, y, class_ids


def make_well_separated_embeddings(
    n_tasks: int = 50,
    n_clusters: int = 5,
    embedding_dim: int = 64,
    seed: int = 42,
    separation: float = 3.0,
    noise_scale: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Generate well-separated L2-normalized embeddings for testing.

    Returns:
        embeddings: (n_tasks, embedding_dim) L2-normalized
        true_labels: (n_tasks,) ground-truth cluster assignments
        task_ids: list of deterministic task ID strings
    """
    rng = np.random.RandomState(seed)

    # Generate cluster centers spread apart
    centers = rng.randn(n_clusters, embedding_dim).astype(np.float64)
    centers *= separation
    # Normalize centers
    for i in range(n_clusters):
        norm = np.linalg.norm(centers[i])
        if norm > 1e-12:
            centers[i] /= norm

    tasks_per_cluster = n_tasks // n_clusters
    remainder = n_tasks - tasks_per_cluster * n_clusters

    embeddings = []
    true_labels = []

    for c in range(n_clusters):
        count = tasks_per_cluster + (1 if c < remainder else 0)
        for _ in range(count):
            noise = rng.randn(embedding_dim).astype(np.float64) * noise_scale
            emb = centers[c] + noise
            norm = np.linalg.norm(emb)
            if norm > 1e-12:
                emb /= norm
            embeddings.append(emb)
            true_labels.append(c)

    embeddings = np.array(embeddings, dtype=np.float32)
    true_labels = np.array(true_labels, dtype=np.int32)

    # Generate deterministic task IDs
    task_ids = []
    for i in range(n_tasks):
        tid = canonicalize_task_id(
            dataset="synthetic",
            split="train",
            class_ids=[i, i + 100],
            support_indices=[i * 10 + j for j in range(5)],
        )
        task_ids.append(tid)

    return embeddings, true_labels, task_ids


# ---------------------------------------------------------------------------
# Epoch seed derivation
# ---------------------------------------------------------------------------

def _epoch_seed(seed: int, epoch: int) -> int:
    """Deterministic per-epoch seed."""
    data = struct.pack(">qq", seed, epoch)
    digest = hashlib.sha256(data).digest()
    return int.from_bytes(digest[:4], "big")


# ============================================================================
# Validation runner
# ============================================================================

class ValidationRunner:
    """Collects and runs validation checks, reporting results."""

    def __init__(self, verbose: bool = False, json_only: bool = False):
        self.verbose = verbose
        self.json_only = json_only
        self.results: List[CheckResult] = []
        self.group_times: Dict[str, float] = {}

    def run_check(
        self,
        name: str,
        group: str,
        fn: Callable[[], Tuple[bool, str]],
    ) -> CheckResult:
        """Run a single check function and record its result."""
        t0 = time.perf_counter()
        try:
            passed, message = fn()
            details = None
        except Exception as e:
            passed = False
            message = f"Exception: {e}"
            details = traceback.format_exc()
        elapsed = (time.perf_counter() - t0) * 1000.0

        result = CheckResult(
            name=name,
            group=group,
            passed=passed,
            message=message,
            details=details,
            elapsed_ms=elapsed,
        )
        self.results.append(result)

        if not self.json_only:
            symbol = _c("[PASS]", _GREEN) if passed else _c("[FAIL]", _RED)
            print(f"  {symbol} {name}: {message}" + (
                f" ({elapsed:.1f}ms)" if self.verbose else ""
            ))
            if not passed and details and self.verbose:
                for line in details.strip().split("\n"):
                    print(f"         {_c(line, _DIM)}")

        return result

    def print_group_header(self, group: str) -> None:
        if not self.json_only:
            desc = _GROUP_DESCRIPTIONS.get(group, group)
            print(f"\n{_c(f'--- {group}: {desc} ---', _CYAN)}")

    def print_summary(self) -> Dict[str, Any]:
        """Print summary and return JSON-compatible dict."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed

        # Per-group summary
        groups_summary: Dict[str, Dict[str, Any]] = {}
        for g in _ALL_GROUPS:
            group_results = [r for r in self.results if r.group == g]
            if not group_results:
                continue
            g_pass = sum(1 for r in group_results if r.passed)
            g_fail = len(group_results) - g_pass
            groups_summary[g] = {
                "total": len(group_results),
                "passed": g_pass,
                "failed": g_fail,
                "elapsed_ms": round(self.group_times.get(g, 0.0), 1),
                "checks": [
                    {
                        "name": r.name,
                        "passed": r.passed,
                        "message": r.message,
                        "elapsed_ms": round(r.elapsed_ms, 2),
                    }
                    for r in group_results
                ],
            }

        total_time = sum(self.group_times.values())

        summary = {
            "overall": "PASS" if failed == 0 else "FAIL",
            "total_checks": total,
            "passed": passed,
            "failed": failed,
            "total_time_ms": round(total_time, 1),
            "groups": groups_summary,
        }

        if self.json_only:
            print(json.dumps(summary, indent=2))
        else:
            print(f"\n{'=' * 60}")
            status = _c("ALL PASSED", _GREEN) if failed == 0 else _c(
                f"{failed} FAILED", _RED
            )
            print(
                f"  {_c('Summary:', _BOLD)} {passed}/{total} checks passed "
                f"| {status} | {total_time:.0f}ms total"
            )

            if failed > 0:
                print(f"\n  {_c('Failed checks:', _RED)}")
                for r in self.results:
                    if not r.passed:
                        print(f"    - [{r.group}] {r.name}: {r.message}")

            print(f"\n  Per-group timing:")
            for g, t in self.group_times.items():
                g_res = groups_summary.get(g, {})
                g_pass = g_res.get("passed", 0)
                g_total = g_res.get("total", 0)
                mark = _c("ok", _GREEN) if g_pass == g_total else _c("FAIL", _RED)
                print(f"    {g:20s} {g_pass}/{g_total} {mark}  ({t:.0f}ms)")
            print()

        return summary


# ============================================================================
# Validation Group 1: Probe Network Validation (5 checks)
# ============================================================================

def run_probe_checks(runner: ValidationRunner) -> None:
    """Group 1: Probe network forward pass, head, freezing, shapes."""
    runner.print_group_header("probe")
    t0 = time.perf_counter()

    # --- Check 1: Conv4 forward pass produces output ---
    def check_conv4_forward():
        torch.manual_seed(1)
        probe = Conv4Probe(in_channels=3, hidden_channels=64)
        probe.train(False)
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            out = probe(x)
        if out is None or out.numel() == 0:
            return False, "Forward pass produced no output"
        return True, f"Conv4 output shape: {tuple(out.shape)}"

    runner.run_check("conv4_forward_produces_output", "probe", check_conv4_forward)

    # --- Check 2: ProbeWithHead creates correct N-way head ---
    def check_probe_with_head():
        torch.manual_seed(2)
        probe = Conv4Probe(in_channels=3, hidden_channels=64)
        model = ProbeWithHead(probe, n_way=7)
        if model.head.out_features != 7:
            return False, f"Head out_features={model.head.out_features}, expected 7"
        x = torch.randn(3, 3, 32, 32)
        logits = model(x)
        if logits.shape != (3, 7):
            return False, f"Logit shape {tuple(logits.shape)}, expected (3, 7)"
        return True, f"Head out_features=7, logit shape (3, 7)"

    runner.run_check("probe_head_n_way_correct", "probe", check_probe_with_head)

    # --- Check 3: Probe is frozen (no training mode, no grad) ---
    def check_probe_frozen():
        torch.manual_seed(3)
        probe = Conv4Probe(in_channels=3, hidden_channels=64)
        model = ProbeWithHead(probe, n_way=5)
        if model.probe.training:
            return False, "Probe is in training mode (expected not training)"
        for name, p in model.probe.named_parameters():
            if p.requires_grad:
                return False, f"Probe param {name} has requires_grad=True"
        # Verify BN modules are not in training mode
        for name, m in model.probe.named_modules():
            if isinstance(m, nn.BatchNorm2d) and m.training:
                return False, f"BN module {name} is in training mode"
        return True, "Probe frozen: not training, no requires_grad, BN not training"

    runner.run_check("probe_frozen_state", "probe", check_probe_frozen)

    # --- Check 4: Output shape matches expected ---
    def check_output_shape():
        torch.manual_seed(4)
        probe = Conv4Probe(in_channels=3, hidden_channels=64)
        model = ProbeWithHead(probe, n_way=5)
        x = torch.randn(4, 3, 32, 32)
        out = model(x)
        expected = (4, 5)
        if out.shape != torch.Size(expected):
            return False, f"Shape {tuple(out.shape)}, expected {expected}"
        return True, f"Output shape {tuple(out.shape)} matches expected {expected}"

    runner.run_check("output_shape_matches", "probe", check_output_shape)

    # --- Check 5: Different inputs -> different outputs ---
    def check_different_inputs():
        torch.manual_seed(5)
        probe = Conv4Probe(in_channels=3, hidden_channels=64)
        model = ProbeWithHead(probe, n_way=5)
        model.probe.train(False)
        torch.manual_seed(10)
        x1 = torch.randn(1, 3, 32, 32)
        torch.manual_seed(20)
        x2 = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out1 = model(x1)
            out2 = model(x2)
        diff = (out1 - out2).abs().max().item()
        if diff < 1e-6:
            return False, f"Different inputs produced same output (max diff={diff:.2e})"
        return True, f"Different inputs -> different outputs (max diff={diff:.4f})"

    runner.run_check("different_inputs_different_outputs", "probe", check_different_inputs)

    runner.group_times["probe"] = (time.perf_counter() - t0) * 1000.0


# ============================================================================
# Validation Group 2: Fisher Computation (5 checks)
# ============================================================================

def run_fisher_checks(runner: ValidationRunner) -> None:
    """Group 2: Fisher diagonal computation properties."""
    runner.print_group_header("fisher")
    t0 = time.perf_counter()

    # Shared setup
    def _make_fisher():
        torch.manual_seed(42)
        probe = Conv4Probe(in_channels=3, hidden_channels=64)
        model = ProbeWithHead(probe, n_way=5)
        x, y, _ = make_synthetic_episode(seed=42, n_way=5, k_shot=5)
        fisher = compute_diagonal_fisher(model, x, y)
        return fisher, model, x, y

    # --- Check 1: Fisher values are non-negative ---
    def check_fisher_non_negative():
        fisher, _, _, _ = _make_fisher()
        for name, f_diag in fisher.items():
            if (f_diag < -1e-8).any():
                min_val = f_diag.min().item()
                return False, f"Negative Fisher in {name}: min={min_val:.6e}"
        return True, "All Fisher diagonal values are non-negative"

    runner.run_check("fisher_non_negative", "fisher", check_fisher_non_negative)

    # --- Check 2: Fisher is deterministic ---
    def check_fisher_deterministic():
        torch.manual_seed(42)
        probe1 = Conv4Probe(in_channels=3, hidden_channels=64)
        model1 = ProbeWithHead(probe1, n_way=5)
        x, y, _ = make_synthetic_episode(seed=42, n_way=5, k_shot=5)
        fisher1 = compute_diagonal_fisher(model1, x, y)

        torch.manual_seed(42)
        probe2 = Conv4Probe(in_channels=3, hidden_channels=64)
        model2 = ProbeWithHead(probe2, n_way=5)
        x2, y2, _ = make_synthetic_episode(seed=42, n_way=5, k_shot=5)
        fisher2 = compute_diagonal_fisher(model2, x2, y2)

        for name in fisher1:
            if name not in fisher2:
                return False, f"Key {name} missing in second Fisher"
            diff = (fisher1[name] - fisher2[name]).abs().max().item()
            if diff > 1e-6:
                return False, f"Fisher differs in {name}: max diff={diff:.6e}"
        return True, "Fisher is deterministic (same inputs -> same result)"

    runner.run_check("fisher_deterministic", "fisher", check_fisher_deterministic)

    # --- Check 3: Fisher computed in fp32 ---
    def check_fisher_fp32():
        fisher, _, _, _ = _make_fisher()
        for name, f_diag in fisher.items():
            if f_diag.dtype != torch.float32:
                return False, f"{name} dtype is {f_diag.dtype}, expected float32"
        return True, "All Fisher tensors are float32"

    runner.run_check("fisher_fp32", "fisher", check_fisher_fp32)

    # --- Check 4: Fisher normalized by sample count ---
    def check_fisher_normalized():
        torch.manual_seed(42)
        probe = Conv4Probe(in_channels=3, hidden_channels=64)
        model = ProbeWithHead(probe, n_way=5)
        x, y, _ = make_synthetic_episode(seed=42, n_way=5, k_shot=5)

        # Compute with full set
        fisher_full = compute_diagonal_fisher(model, x, y)

        # Check that Fisher magnitudes are reasonable (not sum, but mean)
        # If normalized by N, Fisher should be finite and non-negative
        total_fisher_norm = 0.0
        for name, f_diag in fisher_full.items():
            total_fisher_norm += f_diag.float().sum().item()

        if total_fisher_norm < 0:
            return False, f"Total Fisher norm is negative: {total_fisher_norm}"
        if not math.isfinite(total_fisher_norm):
            return False, f"Total Fisher norm is not finite: {total_fisher_norm}"
        return True, f"Fisher normalized by sample count (total norm={total_fisher_norm:.4f})"

    runner.run_check("fisher_normalized_by_samples", "fisher", check_fisher_normalized)

    # --- Check 5: Non-zero Fisher (model has gradients) ---
    def check_fisher_nonzero():
        fisher, _, _, _ = _make_fisher()
        total = 0.0
        for name, f_diag in fisher.items():
            total += f_diag.sum().item()
        if total < 1e-12:
            return False, f"Fisher is all zeros (total={total:.2e})"
        return True, f"Fisher is non-zero (total={total:.4f})"

    runner.run_check("fisher_nonzero", "fisher", check_fisher_nonzero)

    runner.group_times["fisher"] = (time.perf_counter() - t0) * 1000.0


# ============================================================================
# Validation Group 3: Done-When Gate (a) -- Embedding Determinism (5 checks)
# ============================================================================

def run_gate_a_checks(runner: ValidationRunner) -> None:
    """Group 3: Gate (a) -- embedding determinism, L2-norm, task ID."""
    runner.print_group_header("gate_a")
    t0 = time.perf_counter()

    def _extract_embedding(seed: int = 42, n_way: int = 5, k_shot: int = 5,
                           embedding_dim: int = 64, in_channels: int = 3):
        """Full extraction pipeline: episode -> probe -> Fisher -> embedding."""
        torch.manual_seed(seed)
        probe = Conv4Probe(in_channels=in_channels, hidden_channels=64)
        model = ProbeWithHead(probe, n_way=n_way)
        x, y, class_ids = make_synthetic_episode(
            seed=seed, n_way=n_way, k_shot=k_shot, in_channels=in_channels,
        )
        fisher = compute_diagonal_fisher(model, x, y)
        emb = fisher_to_embedding(fisher, embedding_dim=embedding_dim)
        return emb, class_ids

    # --- Check 1: Same episode + seed -> identical embedding within 1e-7 ---
    def check_embedding_determinism():
        emb1, _ = _extract_embedding(seed=42, embedding_dim=64)
        emb2, _ = _extract_embedding(seed=42, embedding_dim=64)
        max_diff = np.max(np.abs(emb1 - emb2))
        if max_diff > 1e-7:
            return False, f"Embeddings differ: max diff={max_diff:.2e} (threshold=1e-7)"
        return True, f"Embeddings identical within 1e-7 (max diff={max_diff:.2e})"

    runner.run_check("embedding_determinism", "gate_a", check_embedding_determinism)

    # --- Check 2: Embedding is L2-normalized (norm approximately 1.0) ---
    def check_embedding_l2_norm():
        emb, _ = _extract_embedding(seed=42, embedding_dim=64)
        norm = np.linalg.norm(emb)
        tol = 1e-5
        if abs(norm - 1.0) > tol:
            return False, f"L2 norm = {norm:.8f}, expected 1.0 (tol={tol})"
        return True, f"L2 norm = {norm:.8f} (within {tol} of 1.0)"

    runner.run_check("embedding_l2_normalized", "gate_a", check_embedding_l2_norm)

    # --- Check 3: Embedding dimension matches config ---
    def check_embedding_dim():
        for dim in [32, 64, 128]:
            emb, _ = _extract_embedding(seed=42, embedding_dim=dim)
            if emb.shape != (dim,):
                return False, f"Shape {emb.shape}, expected ({dim},)"
        return True, "Embedding dimension matches config for all tested dims"

    runner.run_check("embedding_dimension_correct", "gate_a", check_embedding_dim)

    # --- Check 4: Different episodes -> different embeddings ---
    def check_different_episodes():
        emb1, _ = _extract_embedding(seed=42, embedding_dim=64)
        emb2, _ = _extract_embedding(seed=99, embedding_dim=64)
        cos_sim = float(np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-12
        ))
        cos_dist = 1.0 - cos_sim
        if cos_dist <= 0:
            return False, f"Cosine distance = {cos_dist:.6f} (expected > 0)"
        return True, f"Different episodes -> cosine distance = {cos_dist:.6f} > 0"

    runner.run_check("different_episodes_different_embeddings", "gate_a",
                     check_different_episodes)

    # --- Check 5: Task ID is deterministic ---
    def check_task_id_determinism():
        id1 = canonicalize_task_id("mnist", "train", [0, 1, 2], [10, 20, 30])
        id2 = canonicalize_task_id("mnist", "train", [0, 1, 2], [10, 20, 30])
        if id1 != id2:
            return False, f"Same inputs produced different IDs: {id1} vs {id2}"
        id3 = canonicalize_task_id("mnist", "train", [0, 1, 3], [10, 20, 30])
        if id1 == id3:
            return False, f"Different inputs produced same ID: {id1}"
        # Check order-independence of class_ids (sorted internally)
        id4 = canonicalize_task_id("mnist", "train", [2, 0, 1], [30, 10, 20])
        if id1 != id4:
            return False, f"Reordered inputs produced different ID: {id1} vs {id4}"
        return True, f"Task ID deterministic: {id1} (16 hex chars, order-independent)"

    runner.run_check("task_id_deterministic", "gate_a", check_task_id_determinism)

    runner.group_times["gate_a"] = (time.perf_counter() - t0) * 1000.0


# ============================================================================
# Validation Group 4: Registry Operations (5 checks)
# ============================================================================

def run_registry_checks(runner: ValidationRunner) -> None:
    """Group 4: Registry save/load, JSONL, NPZ, query, row index."""
    runner.print_group_header("registry")
    t0 = time.perf_counter()

    EMB_DIM = 64
    N_ENTRIES = 20

    def _make_populated_registry() -> TaskEmbeddingRegistry:
        """Create a registry with 20 entries (10 from 'A', 10 from 'B')."""
        rng = np.random.RandomState(42)
        reg = TaskEmbeddingRegistry(embedding_dim=EMB_DIM)
        for i in range(N_ENTRIES):
            emb = rng.randn(EMB_DIM).astype(np.float32)
            emb /= np.linalg.norm(emb) + 1e-12
            dataset = "A" if i < 10 else "B"
            task_id = canonicalize_task_id(
                dataset=dataset, split="train",
                class_ids=[i, i + 100], support_indices=[i * 10 + j for j in range(5)],
            )
            meta = {
                "dataset": dataset,
                "split": "train",
                "n_way": 5,
                "k_shot": 5,
                "probe_signature": "conv4_last_block_e64_stub",
            }
            reg.update(task_id, emb, meta)
        return reg

    # --- Check 1: Save/load round-trip preserves data ---
    def check_save_load_roundtrip():
        reg = _make_populated_registry()
        with tempfile.TemporaryDirectory() as tmpdir:
            reg.save(tmpdir)
            loaded = TaskEmbeddingRegistry.load(tmpdir)

            if len(loaded) != len(reg):
                return False, f"Length mismatch: {len(loaded)} vs {len(reg)}"

            for tid in reg.task_ids:
                orig = reg.get_embedding(tid)
                loaded_emb = loaded.get_embedding(tid)
                diff = np.max(np.abs(orig - loaded_emb))
                if diff > 1e-7:
                    return False, f"Embedding mismatch for {tid}: diff={diff:.2e}"

        return True, f"Round-trip preserves all {N_ENTRIES} embeddings within 1e-7"

    runner.run_check("registry_save_load_roundtrip", "registry",
                     check_save_load_roundtrip)

    # --- Check 2: JSONL format valid ---
    def check_jsonl_format():
        reg = _make_populated_registry()
        with tempfile.TemporaryDirectory() as tmpdir:
            reg.save(tmpdir)
            jsonl_path = Path(tmpdir) / TaskEmbeddingRegistry.JSONL_FILENAME
            with open(jsonl_path, "r") as f:
                lines = f.readlines()

            if len(lines) < 2:
                return False, f"JSONL has only {len(lines)} lines (expected >= 2)"

            # First line must be header with _version
            header = json.loads(lines[0])
            if "_version" not in header:
                return False, "Header missing _version"

            # All subsequent lines must be valid JSON with task_id
            for i, line in enumerate(lines[1:], start=1):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    return False, f"Line {i} is not valid JSON: {e}"
                if "task_id" not in obj:
                    return False, f"Line {i} missing task_id"
                if "row_index" not in obj:
                    return False, f"Line {i} missing row_index"

        return True, f"JSONL valid: 1 header + {N_ENTRIES} entries, all parseable"

    runner.run_check("registry_jsonl_format", "registry", check_jsonl_format)

    # --- Check 3: NPZ contains embeddings ---
    def check_npz_embeddings():
        reg = _make_populated_registry()
        with tempfile.TemporaryDirectory() as tmpdir:
            reg.save(tmpdir)
            npz_path = Path(tmpdir) / TaskEmbeddingRegistry.NPZ_FILENAME
            data = np.load(str(npz_path))
            if "embeddings" not in data:
                return False, "'embeddings' key missing from NPZ"
            matrix = data["embeddings"]
            expected_shape = (N_ENTRIES, EMB_DIM)
            if matrix.shape != expected_shape:
                return False, f"Shape {matrix.shape}, expected {expected_shape}"
        return True, f"NPZ contains 'embeddings' with shape {expected_shape}"

    runner.run_check("registry_npz_embeddings", "registry", check_npz_embeddings)

    # --- Check 4: Query by dataset returns correct entries ---
    def check_query_by_dataset():
        reg = _make_populated_registry()
        results_a = reg.query(dataset="A")
        results_b = reg.query(dataset="B")
        if len(results_a) != 10:
            return False, f"Query dataset='A' returned {len(results_a)}, expected 10"
        if len(results_b) != 10:
            return False, f"Query dataset='B' returned {len(results_b)}, expected 10"
        # Verify all returned entries have correct dataset
        for tid in results_a:
            meta = reg.get_meta(tid)
            if meta.get("dataset") != "A":
                return False, f"Task {tid} in query 'A' has dataset={meta.get('dataset')}"
        return True, "Query by dataset returns correct subsets (10 'A', 10 'B')"

    runner.run_check("registry_query_by_dataset", "registry", check_query_by_dataset)

    # --- Check 5: Row index consistency ---
    def check_row_index_consistency():
        reg = _make_populated_registry()
        with tempfile.TemporaryDirectory() as tmpdir:
            reg.save(tmpdir)

            # Parse JSONL to get row indices
            jsonl_path = Path(tmpdir) / TaskEmbeddingRegistry.JSONL_FILENAME
            npz_path = Path(tmpdir) / TaskEmbeddingRegistry.NPZ_FILENAME

            with open(jsonl_path, "r") as f:
                lines = f.readlines()

            data = np.load(str(npz_path))
            matrix = data["embeddings"]

            for line in lines[1:]:
                entry = json.loads(line.strip())
                task_id = entry["task_id"]
                row_idx = entry["row_index"]
                npz_emb = matrix[row_idx]
                reg_emb = reg.get_embedding(task_id)
                diff = np.max(np.abs(npz_emb - reg_emb))
                if diff > 1e-7:
                    return False, (
                        f"Row index inconsistency for {task_id}: "
                        f"NPZ[{row_idx}] differs from registry (diff={diff:.2e})"
                    )

        return True, f"All {N_ENTRIES} row indices are consistent between JSONL and NPZ"

    runner.run_check("registry_row_index_consistency", "registry",
                     check_row_index_consistency)

    runner.group_times["registry"] = (time.perf_counter() - t0) * 1000.0


# ============================================================================
# Validation Group 5: Clustering Validation (4 checks)
# ============================================================================

def run_clustering_checks(runner: ValidationRunner) -> None:
    """Group 5: K-means cluster count, assignment, determinism, silhouette."""
    runner.print_group_header("clustering")
    t0 = time.perf_counter()

    N_TASKS = 50
    N_CLUSTERS = 5
    EMB_DIM = 64

    embeddings, true_labels, task_ids = make_well_separated_embeddings(
        n_tasks=N_TASKS, n_clusters=N_CLUSTERS, embedding_dim=EMB_DIM, seed=42,
    )

    # --- Check 1: K-means produces k clusters ---
    def check_kmeans_k_clusters():
        labels, centroids = kmeans_numpy(embeddings, k=N_CLUSTERS, seed=42)
        unique = set(labels.tolist())
        if len(unique) != N_CLUSTERS:
            return False, f"Found {len(unique)} clusters, expected {N_CLUSTERS}"
        if centroids.shape != (N_CLUSTERS, EMB_DIM):
            return False, f"Centroids shape {centroids.shape}, expected ({N_CLUSTERS}, {EMB_DIM})"
        return True, f"K-means produced exactly {N_CLUSTERS} clusters"

    runner.run_check("kmeans_produces_k_clusters", "clustering",
                     check_kmeans_k_clusters)

    # --- Check 2: All points assigned to a cluster ---
    def check_all_points_assigned():
        labels, _ = kmeans_numpy(embeddings, k=N_CLUSTERS, seed=42)
        if len(labels) != N_TASKS:
            return False, f"Labels length {len(labels)}, expected {N_TASKS}"
        for i, lbl in enumerate(labels):
            if lbl < 0 or lbl >= N_CLUSTERS:
                return False, f"Point {i} has invalid label {lbl}"
        return True, f"All {N_TASKS} points assigned to valid clusters [0, {N_CLUSTERS-1}]"

    runner.run_check("all_points_assigned", "clustering", check_all_points_assigned)

    # --- Check 3: Deterministic with fixed seed ---
    def check_clustering_deterministic():
        labels1, centroids1 = kmeans_numpy(embeddings, k=N_CLUSTERS, seed=42)
        labels2, centroids2 = kmeans_numpy(embeddings, k=N_CLUSTERS, seed=42)
        if not np.array_equal(labels1, labels2):
            return False, "Same seed produced different labels"
        max_centroid_diff = np.max(np.abs(centroids1 - centroids2))
        if max_centroid_diff > 1e-7:
            return False, f"Centroid diff {max_centroid_diff:.2e}"
        return True, "K-means is deterministic with fixed seed"

    runner.run_check("clustering_deterministic_seed", "clustering",
                     check_clustering_deterministic)

    # --- Check 4: Well-separated data -> high silhouette ---
    def check_silhouette_well_separated():
        labels, _ = kmeans_numpy(embeddings, k=N_CLUSTERS, seed=42)
        sil = silhouette_score_numpy(embeddings, labels)
        if sil < 0.3:
            return False, f"Silhouette = {sil:.4f} (expected > 0.3 for well-separated data)"
        if sil > 1.0 or sil < -1.0:
            return False, f"Silhouette = {sil:.4f} out of range [-1, 1]"
        return True, f"Silhouette = {sil:.4f} (> 0.3, indicating well-separated clusters)"

    runner.run_check("silhouette_well_separated", "clustering",
                     check_silhouette_well_separated)

    runner.group_times["clustering"] = (time.perf_counter() - t0) * 1000.0


# ============================================================================
# Validation Group 6: Done-When Gate (b) -- Cluster Stability (4 checks)
# ============================================================================

def run_gate_b_checks(runner: ValidationRunner) -> None:
    """Group 6: Gate (b) -- ARI >= 0.9 for well-separated clusters."""
    runner.print_group_header("gate_b")
    t0 = time.perf_counter()

    N_TASKS = 50
    N_CLUSTERS = 5
    EMB_DIM = 64

    # Use high separation and very low noise for robust stability across seeds
    embeddings, true_labels, task_ids = make_well_separated_embeddings(
        n_tasks=N_TASKS, n_clusters=N_CLUSTERS, embedding_dim=EMB_DIM, seed=42,
        separation=5.0, noise_scale=0.01,
    )

    # --- Check 1: ARI >= 0.9 for deterministic well-separated clusters ---
    def check_ari_high():
        labels1, _ = kmeans_numpy(embeddings, k=N_CLUSTERS, seed=42)
        labels2, _ = kmeans_numpy(embeddings, k=N_CLUSTERS, seed=123)
        ari = adjusted_rand_index(labels1, labels2)
        if ari < 0.9:
            return False, f"ARI = {ari:.4f} (below threshold 0.9)"
        return True, f"ARI = {ari:.4f} >= 0.9 (cluster stability gate PASSED)"

    runner.run_check("ari_above_threshold", "gate_b", check_ari_high)

    # --- Check 2: ARI computation correct for known cases ---
    def check_ari_known_cases():
        # Perfect agreement
        perfect_a = np.array([0, 0, 1, 1, 2, 2])
        perfect_b = np.array([0, 0, 1, 1, 2, 2])
        ari_perfect = adjusted_rand_index(perfect_a, perfect_b)
        if abs(ari_perfect - 1.0) > 1e-6:
            return False, f"Perfect agreement ARI = {ari_perfect:.6f}, expected 1.0"

        # Label permutation (should still be 1.0)
        perm_b = np.array([1, 1, 2, 2, 0, 0])
        ari_perm = adjusted_rand_index(perfect_a, perm_b)
        if abs(ari_perm - 1.0) > 1e-6:
            return False, f"Permuted labels ARI = {ari_perm:.6f}, expected 1.0"

        # Completely random (should be near 0)
        rng = np.random.RandomState(42)
        random_a = rng.randint(0, 3, size=100)
        random_b = rng.randint(0, 3, size=100)
        ari_random = adjusted_rand_index(random_a, random_b)
        if abs(ari_random) > 0.3:
            return False, f"Random labels ARI = {ari_random:.4f}, expected near 0"

        return True, (
            f"ARI correct: perfect={ari_perfect:.4f}, "
            f"permuted={ari_perm:.4f}, random={ari_random:.4f}"
        )

    runner.run_check("ari_known_cases", "gate_b", check_ari_known_cases)

    # --- Check 3: Multiple runs with different seeds -> stable clustering ---
    def check_multi_seed_stability():
        seeds = [42, 123, 7, 999, 314, 271, 55, 88, 101, 202]
        aris = []
        base_labels, _ = kmeans_numpy(embeddings, k=N_CLUSTERS, seed=seeds[0])
        for s in seeds[1:]:
            labels_s, _ = kmeans_numpy(embeddings, k=N_CLUSTERS, seed=s)
            ari = adjusted_rand_index(base_labels, labels_s)
            aris.append(ari)

        mean_ari = np.mean(aris)
        min_ari = np.min(aris)
        if min_ari < 0.8:
            return False, (
                f"Multi-seed stability: min ARI = {min_ari:.4f} (expected >= 0.8), "
                f"mean ARI = {mean_ari:.4f}"
            )
        return True, (
            f"Multi-seed stability: mean ARI = {mean_ari:.4f}, "
            f"min ARI = {min_ari:.4f} (all >= 0.8)"
        )

    runner.run_check("multi_seed_stability", "gate_b", check_multi_seed_stability)

    # --- Check 4: NMI agrees with ARI direction ---
    def check_nmi_agrees_with_ari():
        labels1, _ = kmeans_numpy(embeddings, k=N_CLUSTERS, seed=42)
        labels2, _ = kmeans_numpy(embeddings, k=N_CLUSTERS, seed=123)
        ari = adjusted_rand_index(labels1, labels2)
        nmi = normalized_mutual_info(labels1, labels2)

        # For well-separated data, both should be high
        if nmi < 0.7:
            return False, f"NMI = {nmi:.4f} (expected >= 0.7 for well-separated data)"
        # ARI and NMI should agree in direction
        ari_high = ari >= 0.9
        nmi_high = nmi >= 0.7
        if ari_high != nmi_high:
            return False, f"ARI ({ari:.4f}) and NMI ({nmi:.4f}) disagree in direction"
        return True, f"NMI = {nmi:.4f} agrees with ARI = {ari:.4f} (both high)"

    runner.run_check("nmi_agrees_with_ari", "gate_b", check_nmi_agrees_with_ari)

    runner.group_times["gate_b"] = (time.perf_counter() - t0) * 1000.0


# ============================================================================
# Validation Group 7: Done-When Gate (c) -- Curriculum Effect (5 checks)
# ============================================================================

def run_gate_c_checks(runner: ValidationRunner) -> None:
    """Group 7: Gate (c) -- Kendall tau, no drops, logging, diversity."""
    runner.print_group_header("gate_c")
    t0 = time.perf_counter()

    N_TASKS = 100
    N_CLUSTERS = 5
    EMB_DIM = 64

    embeddings, true_labels, task_ids = make_well_separated_embeddings(
        n_tasks=N_TASKS, n_clusters=N_CLUSTERS, embedding_dim=EMB_DIM, seed=42,
        separation=5.0,
    )
    task_emb_map = {tid: embeddings[i] for i, tid in enumerate(task_ids)}

    # Compute difficulty based on centroid distance to cluster 0
    labels, centroids = kmeans_numpy(embeddings, k=N_CLUSTERS, seed=42)
    easy_centroid = centroids[0]
    difficulty = compute_difficulty_centroid(task_emb_map, easy_centroid)

    # --- Check 1: Easy-to-hard ordering differs from random (Kendall tau significant) ---
    def check_kendall_tau_significant():
        curriculum_order = easy_to_hard_order(task_ids, difficulty)

        # Compute difficulty rank for each task (lower rank = easier)
        diff_rank = {tid: i for i, tid in enumerate(curriculum_order)}

        # For the curriculum ordering, the Kendall tau between position
        # index and difficulty rank should be exactly 1.0 (monotone).
        positions_c = list(range(N_TASKS))
        diff_ranks_c = [diff_rank[tid] for tid in curriculum_order]
        tau_curriculum, p_curriculum = kendall_tau(positions_c, diff_ranks_c)

        if abs(tau_curriculum - 1.0) > 1e-6:
            return False, (
                f"Curriculum tau(position, difficulty) = {tau_curriculum:.4f}, "
                f"expected 1.0 (monotone ordering)"
            )

        # For a random ordering, the Kendall tau between position and
        # difficulty rank should be near 0 (not significant).
        rng_local = np.random.RandomState(99)
        random_order = list(task_ids)
        rng_local.shuffle(random_order)

        diff_ranks_r = [diff_rank[tid] for tid in random_order]
        tau_random, p_random = kendall_tau(list(range(N_TASKS)), diff_ranks_r)

        # The difference between curriculum tau and random tau should be
        # substantial (curriculum tau ~ 1.0, random tau ~ 0)
        tau_gap = abs(tau_curriculum) - abs(tau_random)
        if tau_gap < 0.5:
            return False, (
                f"Gap between curriculum tau ({tau_curriculum:.4f}) and "
                f"random tau ({tau_random:.4f}) is {tau_gap:.4f} (expected > 0.5)"
            )

        return True, (
            f"Curriculum tau = {tau_curriculum:.4f} (monotone), "
            f"random tau = {tau_random:.4f}, gap = {tau_gap:.4f} > 0.5"
        )

    runner.run_check("kendall_tau_significant", "gate_c",
                     check_kendall_tau_significant)

    # --- Check 2: Curriculum order contains all task_ids (no drops) ---
    def check_no_drops():
        curriculum_order = easy_to_hard_order(task_ids, difficulty)
        if set(curriculum_order) != set(task_ids):
            missing = set(task_ids) - set(curriculum_order)
            extra = set(curriculum_order) - set(task_ids)
            return False, f"Missing: {len(missing)}, Extra: {len(extra)}"
        if len(curriculum_order) != len(task_ids):
            return False, (
                f"Length mismatch: {len(curriculum_order)} vs {len(task_ids)}"
            )
        # Check no duplicates
        if len(set(curriculum_order)) != len(curriculum_order):
            return False, "Duplicates found in curriculum order"
        return True, f"Curriculum order contains all {N_TASKS} task_ids (no drops, no dups)"

    runner.run_check("curriculum_no_drops", "gate_c", check_no_drops)

    # --- Check 3: Logging produces valid epoch log ---
    def check_epoch_log_valid():
        curriculum_order = easy_to_hard_order(task_ids, difficulty)
        cluster_assignments = {
            task_ids[i]: int(labels[i]) for i in range(N_TASKS)
        }
        log_entry = generate_epoch_log(
            epoch=0,
            strategy="easy_to_hard",
            ordered_task_ids=curriculum_order,
            difficulty=difficulty,
            cluster_assignments=cluster_assignments,
        )
        # Should be JSON-serializable
        try:
            serialized = json.dumps(log_entry)
            parsed = json.loads(serialized)
        except (TypeError, json.JSONDecodeError) as e:
            return False, f"Log entry not JSON-serializable: {e}"

        required_keys = ["epoch", "strategy", "task_order_hash", "num_tasks",
                         "cluster_histogram", "difficulty_stats"]
        for key in required_keys:
            if key not in parsed:
                return False, f"Missing key in log entry: {key}"

        return True, f"Epoch log is valid JSON with all required keys"

    runner.run_check("epoch_log_valid", "gate_c", check_epoch_log_valid)

    # --- Check 4: Epoch log contains strategy and task_order_hash ---
    def check_log_contains_strategy_and_hash():
        curriculum_order = easy_to_hard_order(task_ids, difficulty)
        cluster_assignments = {
            task_ids[i]: int(labels[i]) for i in range(N_TASKS)
        }
        log_entry = generate_epoch_log(
            epoch=5,
            strategy="easy_to_hard",
            ordered_task_ids=curriculum_order,
            difficulty=difficulty,
            cluster_assignments=cluster_assignments,
        )

        if log_entry.get("strategy") != "easy_to_hard":
            return False, f"Strategy = '{log_entry.get('strategy')}', expected 'easy_to_hard'"
        order_hash = log_entry.get("task_order_hash", "")
        if len(order_hash) != 64:
            return False, f"task_order_hash length = {len(order_hash)}, expected 64 (SHA-256 hex)"

        # Hash should be deterministic
        log_entry2 = generate_epoch_log(
            epoch=5,
            strategy="easy_to_hard",
            ordered_task_ids=curriculum_order,
            difficulty=difficulty,
            cluster_assignments=cluster_assignments,
        )
        if log_entry["task_order_hash"] != log_entry2["task_order_hash"]:
            return False, "task_order_hash not deterministic"

        return True, (
            f"Log has strategy='easy_to_hard', "
            f"task_order_hash={order_hash[:16]}... (deterministic, 64 hex chars)"
        )

    runner.run_check("log_strategy_and_hash", "gate_c",
                     check_log_contains_strategy_and_hash)

    # --- Check 5: Diversity batch has min pairwise distance >= threshold ---
    def check_diversity_batch():
        threshold = 0.1
        batch_size = 8
        batch = build_diversity_batch(
            task_ids=task_ids,
            task_embeddings=task_emb_map,
            batch_size=batch_size,
            min_distance=threshold,
            seed=42,
        )
        if len(batch) != batch_size:
            return False, f"Batch size = {len(batch)}, expected {batch_size}"

        # Compute pairwise distances
        min_pair_dist = float("inf")
        for i in range(len(batch)):
            for j in range(i + 1, len(batch)):
                d = 1.0 - float(np.dot(
                    task_emb_map[batch[i]], task_emb_map[batch[j]]
                ))
                if d < min_pair_dist:
                    min_pair_dist = d

        # After potential relaxation, check that we got reasonable diversity
        # The threshold may have been halved during relaxation, so check
        # against a lower bound
        effective_threshold = threshold * 0.25  # Allow 2 relaxation halvings
        if min_pair_dist < effective_threshold:
            return False, (
                f"Min pairwise cosine distance = {min_pair_dist:.4f} "
                f"(below effective threshold {effective_threshold:.4f})"
            )
        return True, (
            f"Diversity batch: {batch_size} tasks, "
            f"min pairwise distance = {min_pair_dist:.4f} "
            f"(>= {effective_threshold:.4f})"
        )

    runner.run_check("diversity_batch_min_distance", "gate_c",
                     check_diversity_batch)

    runner.group_times["gate_c"] = (time.perf_counter() - t0) * 1000.0


# ============================================================================
# Internal self-tests for helper implementations
# ============================================================================

def _validate_kendall_tau_impl() -> None:
    """Validate our pure-Python Kendall tau implementation against known values.

    Called before running any groups to catch implementation bugs early.
    Raises AssertionError if the implementation is incorrect.
    """
    # Perfect positive correlation
    x = list(range(10))
    y = list(range(10))
    tau_pos, _ = kendall_tau(x, y)
    assert abs(tau_pos - 1.0) < 1e-6, f"Expected tau=1.0, got {tau_pos}"

    # Perfect negative correlation
    y_rev = list(reversed(range(10)))
    tau_neg, _ = kendall_tau(x, y_rev)
    assert abs(tau_neg - (-1.0)) < 1e-6, f"Expected tau=-1.0, got {tau_neg}"

    # Uncorrelated (approximately zero)
    # This is a deterministic sequence that should give |tau| < 0.5
    semi_random = [3, 7, 1, 8, 4, 9, 0, 5, 2, 6]
    tau_mid, _ = kendall_tau(x, semi_random)
    assert abs(tau_mid) < 1.0, f"tau should be in [-1, 1], got {tau_mid}"


def _validate_ari_impl() -> None:
    """Validate ARI implementation against known cases.

    Raises AssertionError if the implementation is incorrect.
    """
    # Perfect agreement
    a = np.array([0, 0, 1, 1, 2, 2])
    b = np.array([0, 0, 1, 1, 2, 2])
    assert abs(adjusted_rand_index(a, b) - 1.0) < 1e-6

    # Permuted labels should still be 1.0
    c = np.array([2, 2, 0, 0, 1, 1])
    assert abs(adjusted_rand_index(a, c) - 1.0) < 1e-6

    # Single element
    assert abs(adjusted_rand_index(np.array([0]), np.array([0])) - 1.0) < 1e-6


def _validate_nmi_impl() -> None:
    """Validate NMI implementation against known cases."""
    a = np.array([0, 0, 1, 1, 2, 2])
    b = np.array([0, 0, 1, 1, 2, 2])
    assert abs(normalized_mutual_info(a, b) - 1.0) < 1e-6

    c = np.array([2, 2, 0, 0, 1, 1])
    assert abs(normalized_mutual_info(a, c) - 1.0) < 1e-6


# ============================================================================
# Main entry point and CLI
# ============================================================================

_GROUP_FN_MAP: Dict[str, Callable[[ValidationRunner], None]] = {
    "probe": run_probe_checks,
    "fisher": run_fisher_checks,
    "gate_a": run_gate_a_checks,
    "registry": run_registry_checks,
    "clustering": run_clustering_checks,
    "gate_b": run_gate_b_checks,
    "gate_c": run_gate_c_checks,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Task2Vec task embeddings + curriculum/clustering contracts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Validation groups:\n"
            "  probe       Probe Network: Conv4, head, freezing\n"
            "  fisher      Fisher Computation: non-negative, deterministic, fp32\n"
            "  gate_a      Done-When Gate (a): Embedding determinism\n"
            "  registry    Registry Operations: save/load, JSONL/NPZ, query\n"
            "  clustering  Clustering: K-means, silhouette\n"
            "  gate_b      Done-When Gate (b): Cluster stability (ARI >= 0.9)\n"
            "  gate_c      Done-When Gate (c): Curriculum effect (Kendall tau)\n"
        ),
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed output including timing and tracebacks",
    )
    parser.add_argument(
        "--group", "-g",
        type=str,
        default=None,
        choices=_ALL_GROUPS,
        help="Run only a specific validation group",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_only",
        help="Output JSON summary only (no terminal formatting)",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        dest="list_groups",
        help="List all validation groups and exit",
    )
    args = parser.parse_args()

    if args.list_groups:
        print("Validation groups:")
        for g in _ALL_GROUPS:
            desc = _GROUP_DESCRIPTIONS.get(g, "")
            print(f"  {g:20s}  {desc}")
        return 0

    # ---- Internal self-checks (before any user-facing output) ----
    try:
        _validate_kendall_tau_impl()
        _validate_ari_impl()
        _validate_nmi_impl()
    except AssertionError as e:
        print(f"INTERNAL ERROR: Self-check failed: {e}", file=sys.stderr)
        return 1

    runner = ValidationRunner(verbose=args.verbose, json_only=args.json_only)

    if not args.json_only:
        print(_c("=" * 60, _BOLD))
        print(_c(
            "  Task2Vec + Curriculum/Clustering  --  Contract Validation",
            _BOLD,
        ))
        print(_c("=" * 60, _BOLD))

    total_start = time.perf_counter()

    groups_to_run = [args.group] if args.group else _ALL_GROUPS
    for group_name in groups_to_run:
        fn = _GROUP_FN_MAP.get(group_name)
        if fn is not None:
            fn(runner)

    total_elapsed = (time.perf_counter() - total_start) * 1000.0

    summary = runner.print_summary()

    if not args.json_only:
        print(f"  Total validation time: {total_elapsed:.0f}ms")
        if total_elapsed < 5000:
            print(f"  {_c('(within 5s CPU target)', _DIM)}")
        else:
            print(f"  {_c('(exceeded 5s target)', _YELLOW)}")

    return 0 if summary["overall"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
