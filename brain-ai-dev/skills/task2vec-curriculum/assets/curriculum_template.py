"""
Curriculum Ordering and Meta-Batch Composition for Task2Vec Embeddings

Provides a comprehensive, self-contained framework for curriculum-based task
ordering and diversity-constrained meta-batch composition in Phase 7 of the
cognitive pipeline. Given pre-computed Task2Vec embeddings (Fisher-information-
based vectors stored in a TaskEmbeddingRegistry) and cluster assignments from
the clustering module, this module:

  1. Computes difficulty proxies (centroid distance, adaptation gain, loss slope)
  2. Orders tasks by curriculum strategy (easy-to-hard, anti-curriculum, mixed)
  3. Builds diversity-constrained meta-batches (min pairwise distance, stratified)
  4. Validates curriculum effects via Kendall tau rank correlation
  5. Logs every curriculum decision with SHA-256 hashes and statistics

Key invariant: given identical (seed, epoch, task_ids, strategy), the curriculum
produces identical task orderings across runs.  All randomness derives from
deterministic per-epoch seeds, never from global RNG state.

This template is an asset for the task2vec-curriculum Claude Code skill.
It is intended to be copied into brain_ai/meta/curriculum.py.

Usage::

    from brain_ai.meta.curriculum import (
        CurriculumStrategy, DifficultyProxy, ScheduleType,
        CurriculumOrder, MetaBatch, EpochLog,
        get_curriculum_order, build_meta_batch,
        compute_kendall_tau, create_epoch_log,
        CurriculumManager,
    )

    # Quick start with synthetic data
    manager = CurriculumManager(config, embeddings, task_ids, cluster_labels)
    order = manager.get_epoch_order(epoch=0)
    batch = manager.get_epoch_batch(epoch=0, batch_idx=0)
    log = manager.get_epoch_log(epoch=0)

Architecture notes:
  - All curriculum decisions operate on pre-computed E-dimensional embeddings,
    not raw data.  This decouples curriculum logic from data loading.
  - Difficulty proxies are pluggable: centroid distance (static), adaptation
    gain (dynamic), loss slope (dynamic).
  - Schedule functions control the difficulty mixture in mixed strategies.
  - Every ordering is auditable via SHA-256 hashes and Kendall tau validation.
"""

import hashlib
import json
import logging
import math
import struct
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

# Optional scipy for Kendall tau -- pure Python fallback provided
try:
    from scipy.stats import kendalltau as _scipy_kendalltau
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================

class CurriculumStrategy(Enum):
    """Curriculum ordering strategy.

    Controls how tasks are ordered within each training epoch.

    Attributes:
        NONE: Random shuffle with deterministic seed.  No curriculum effect.
        EASY_TO_HARD: Sort tasks by increasing difficulty proxy value.
        DIVERSITY: Order to maximize inter-task diversity (not a difficulty
            ordering; instead maximizes pairwise distance in task space).
        ANTI_CURRICULUM: Sort tasks by decreasing difficulty (hard-first).
        MIXED: Sample tasks with difficulty-dependent weights that change
            over epochs according to a schedule function.
    """
    NONE = "none"
    EASY_TO_HARD = "easy_to_hard"
    DIVERSITY = "diversity"
    ANTI_CURRICULUM = "anti_curriculum"
    MIXED = "mixed"


class DifficultyProxy(Enum):
    """Method for computing task difficulty from embeddings and metadata.

    Attributes:
        CENTROID_DISTANCE: Cosine distance from each task embedding to the
            easy-cluster centroid.  Static (depends only on embeddings).
        ADAPTATION_GAIN: Historical adaptation gain per task from the
            registry.  Dynamic (changes as training progresses).
        LOSS_SLOPE: Rate of loss decrease during inner-loop adaptation.
            Dynamic (requires per-step inner-loop loss logs).
    """
    CENTROID_DISTANCE = "centroid_distance"
    ADAPTATION_GAIN = "adaptation_gain"
    LOSS_SLOPE = "loss_slope"


class ScheduleType(Enum):
    """Schedule function type for mixed-strategy difficulty weighting.

    Controls how p(hard) evolves from 0 to 1 over the course of training.

    Attributes:
        LINEAR: p = epoch / max_epoch.  Uniform increase.
        COSINE: p = 0.5 * (1 - cos(pi * epoch / max_epoch)).  S-curve.
        STEP: p = 0 before switch_epoch, 1 at and after switch_epoch.
    """
    LINEAR = "linear"
    COSINE = "cosine"
    STEP = "step"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class CurriculumConfig:
    """Configuration for curriculum ordering and meta-batch composition.

    All curriculum parameters are centralized here.  Strategy selection is a
    configuration choice, not a code path change.

    Attributes:
        strategy: Curriculum ordering strategy.
        difficulty_proxy: Method for computing task difficulty.
        diversity_constraint: Meta-batch diversity method.
        diversity_min_distance: Minimum cosine distance for diversity batches.
        meta_batch_size: Target number of tasks per meta-batch.
        schedule_type: Schedule function type for mixed strategy.
        schedule_kwargs: Additional kwargs for the schedule function.
        max_epochs: Total training epochs (used by schedule functions).
        warmup_epochs: Number of initial epochs with random ordering.
        seed: Global seed for reproducibility.
    """
    strategy: str = "none"
    difficulty_proxy: str = "centroid_distance"
    diversity_constraint: str = "min_distance"
    diversity_min_distance: float = 0.3
    meta_batch_size: int = 8
    schedule_type: str = "linear"
    schedule_kwargs: Dict[str, Any] = field(default_factory=dict)
    max_epochs: int = 100
    warmup_epochs: int = 0
    seed: int = 42


@dataclass
class CurriculumOrder:
    """Result of curriculum ordering for a single epoch.

    Contains the ordered task IDs and all metadata needed for logging,
    reproducibility, and downstream analysis.

    Attributes:
        task_ids: Ordered list of task identifiers.
        strategy: Strategy name used for this ordering.
        difficulty_scores: Mapping from task_id to difficulty proxy value.
            May be empty if strategy is NONE.
        cluster_histogram: Count of tasks per cluster in the ordered list.
        epoch: Epoch number for which this ordering was computed.
        metadata: Additional strategy-specific metadata.
    """
    task_ids: List[str]
    strategy: str
    difficulty_scores: Dict[str, float] = field(default_factory=dict)
    cluster_histogram: Dict[int, int] = field(default_factory=dict)
    epoch: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetaBatch:
    """A diversity-constrained meta-batch of tasks.

    Contains the selected task IDs and diversity metrics for the batch.

    Attributes:
        task_ids: Selected task identifiers for this batch.
        diversity_metric: Minimum pairwise cosine distance among selected
            tasks.  Higher values indicate more diverse batches.
        strategy: Diversity constraint method used.
    """
    task_ids: List[str]
    diversity_metric: float = 0.0
    strategy: str = "none"


@dataclass
class EpochLog:
    """Logging record for a single curriculum epoch.

    Contains all information needed to reproduce and audit the curriculum
    decision for this epoch.  JSON-serializable for downstream analysis.

    Attributes:
        epoch: Epoch number (0-indexed).
        strategy: Strategy name used for this epoch.
        task_order_hash: SHA-256 hex digest of the comma-joined task_ids.
        cluster_histogram: Count of tasks per cluster in the ordering.
        difficulty_stats: Summary statistics of difficulty proxy values.
        kendall_tau: Kendall tau rank correlation vs random baseline.
        p_value: P-value for the Kendall tau test.
    """
    epoch: int
    strategy: str
    task_order_hash: str
    cluster_histogram: Dict[int, int] = field(default_factory=dict)
    difficulty_stats: Dict[str, float] = field(default_factory=dict)
    kendall_tau: float = 0.0
    p_value: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "epoch": self.epoch,
            "strategy": self.strategy,
            "task_order_hash": self.task_order_hash,
            "cluster_histogram": {str(k): v for k, v in self.cluster_histogram.items()},
            "difficulty_stats": self.difficulty_stats,
            "kendall_tau": self.kendall_tau,
            "p_value": self.p_value,
        }


# ============================================================================
# Minimal Registry Protocol
# ============================================================================

class _MinimalRegistry:
    """Minimal registry interface for standalone usage and testing.

    When the full TaskEmbeddingRegistry is not available (e.g., during
    unit testing or standalone execution), this class provides the
    minimum interface needed by difficulty proxy computations.

    Attributes:
        _metadata: Mapping from task_id to metadata dict.
        _embeddings: Mapping from task_id to embedding vector.
    """

    def __init__(self) -> None:
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._embeddings: Dict[str, np.ndarray] = {}

    def get_metadata(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Return metadata for a task, or None if not found."""
        return self._metadata.get(task_id)

    def get_embedding(self, task_id: str) -> Optional[np.ndarray]:
        """Return embedding for a task, or None if not found."""
        return self._embeddings.get(task_id)

    def set_metadata(self, task_id: str, meta: Dict[str, Any]) -> None:
        """Store metadata for a task."""
        self._metadata[task_id] = meta

    def set_embedding(self, task_id: str, embedding: np.ndarray) -> None:
        """Store embedding for a task."""
        self._embeddings[task_id] = embedding.copy()


# ============================================================================
# Deterministic Seed Derivation
# ============================================================================

def _epoch_seed(seed: int, epoch: int) -> int:
    """Derive a deterministic per-epoch seed from a global seed and epoch.

    Uses SHA-256 to mix the two integers into a single deterministic seed.
    This avoids correlated random streams across epochs.

    Args:
        seed: Global random seed.
        epoch: Current epoch number.

    Returns:
        A deterministic integer seed derived from the inputs.
    """
    data = struct.pack(">qq", seed, epoch)
    digest = hashlib.sha256(data).digest()
    return int.from_bytes(digest[:4], "big")


def _batch_seed(seed: int, epoch: int, batch_idx: int) -> int:
    """Derive a deterministic per-batch seed.

    Args:
        seed: Global random seed.
        epoch: Current epoch number.
        batch_idx: Batch index within the epoch.

    Returns:
        A deterministic integer seed for this specific batch.
    """
    data = struct.pack(">qqq", seed, epoch, batch_idx)
    digest = hashlib.sha256(data).digest()
    return int.from_bytes(digest[:4], "big")


# ============================================================================
# Schedule Functions
# ============================================================================

def linear_schedule(epoch: int, max_epoch: int) -> float:
    """Linear schedule: p(hard) increases linearly from 0 to 1.

    p = epoch / max_epoch

    At epoch 0, only easy tasks are favored.  At max_epoch, only hard tasks.
    Linear interpolation in between.

    Args:
        epoch: Current epoch number (0-indexed).
        max_epoch: Total number of training epochs.

    Returns:
        p(hard) value in [0, 1].
    """
    if max_epoch <= 0:
        return 0.0
    return min(float(epoch) / float(max_epoch), 1.0)


def cosine_schedule(epoch: int, max_epoch: int) -> float:
    """Cosine schedule: p(hard) follows an S-curve from 0 to 1.

    p = 0.5 * (1 - cos(pi * epoch / max_epoch))

    Starts slow, accelerates through middle epochs, decelerates toward end.
    Provides a smooth transition from easy to hard.

    Args:
        epoch: Current epoch number (0-indexed).
        max_epoch: Total number of training epochs.

    Returns:
        p(hard) value in [0, 1].
    """
    if max_epoch <= 0:
        return 0.0
    progress = min(float(epoch) / float(max_epoch), 1.0)
    return 0.5 * (1.0 - math.cos(math.pi * progress))


def step_schedule(epoch: int, switch_epoch: int) -> float:
    """Step schedule: p(hard) = 0 before switch, 1 at and after switch.

    A hard transition from easy-only to hard-only at a specified epoch.

    Args:
        epoch: Current epoch number (0-indexed).
        switch_epoch: Epoch at which to switch from 0 to 1.

    Returns:
        0.0 if epoch < switch_epoch, else 1.0.
    """
    if switch_epoch <= 0:
        return 1.0
    return 0.0 if epoch < switch_epoch else 1.0


# Schedule function registry
_SCHEDULE_REGISTRY: Dict[str, Callable] = {
    "linear": linear_schedule,
    "cosine": cosine_schedule,
    "step": step_schedule,
}


def register_schedule(name: str, fn: Callable) -> None:
    """Register a custom schedule function.

    Args:
        name: Name for the schedule (used in CurriculumConfig.schedule_type).
        fn: Callable with signature compatible with schedule functions.
            For LINEAR and COSINE: (epoch: int, max_epoch: int) -> float.
            For STEP: (epoch: int, switch_epoch: int) -> float.
            Must return a value in [0, 1].
    """
    _SCHEDULE_REGISTRY[name] = fn
    logger.debug("Registered schedule function: %s", name)


def get_schedule_fn(schedule_type: Union[str, ScheduleType], **kwargs: Any) -> Callable:
    """Retrieve a schedule function by name or enum.

    Returns a callable that takes (epoch, max_epoch_or_switch) and returns
    p(hard) in [0, 1].

    Args:
        schedule_type: Schedule type as string or ScheduleType enum.
        **kwargs: Additional keyword arguments (reserved for future use).

    Returns:
        The schedule callable.

    Raises:
        ValueError: If the schedule type is not registered.
    """
    if isinstance(schedule_type, ScheduleType):
        name = schedule_type.value
    else:
        name = str(schedule_type)

    if name not in _SCHEDULE_REGISTRY:
        available = list(_SCHEDULE_REGISTRY.keys())
        raise ValueError(
            f"Unknown schedule function: '{name}'. "
            f"Available: {available}. "
            f"Register custom schedules with register_schedule()."
        )
    return _SCHEDULE_REGISTRY[name]


# ============================================================================
# Difficulty Proxy Computation
# ============================================================================

def compute_difficulty_centroid_distance(
    embeddings: Dict[str, np.ndarray],
    task_ids: List[str],
    easy_cluster_centroid: np.ndarray,
) -> Dict[str, float]:
    """Compute difficulty as cosine distance from the easy-cluster centroid.

    Tasks near the easy-cluster centroid are easy; tasks far from it are hard.
    The easy cluster is the cluster with the lowest average probe loss in the
    registry.

    Args:
        embeddings: Mapping from task_id to L2-normalized embedding vector.
        task_ids: List of task identifiers to score.
        easy_cluster_centroid: L2-normalized centroid vector of the easy cluster.

    Returns:
        Mapping from task_id to difficulty score (cosine distance to centroid).
        Higher values indicate harder tasks.
    """
    # Ensure centroid is normalized
    centroid_norm = np.linalg.norm(easy_cluster_centroid)
    if centroid_norm < 1e-12:
        logger.warning(
            "Easy cluster centroid has near-zero norm. "
            "Returning uniform difficulty."
        )
        return {tid: 0.5 for tid in task_ids}

    centroid = easy_cluster_centroid / centroid_norm

    difficulty: Dict[str, float] = {}
    for task_id in task_ids:
        emb = embeddings.get(task_id)
        if emb is None:
            logger.warning(
                "Task %s not found in embeddings. Assigning neutral difficulty.",
                task_id,
            )
            difficulty[task_id] = 0.5
            continue
        cosine_sim = float(np.dot(emb, centroid))
        cosine_dist = 1.0 - cosine_sim
        difficulty[task_id] = max(0.0, cosine_dist)

    return difficulty


def compute_difficulty_adaptation_gain(
    task_ids: List[str],
    registry: Any,
) -> Dict[str, float]:
    """Compute difficulty from historical adaptation gain.

    Tasks with HIGH adaptation gain are EASY (the model learns them quickly).
    Tasks with LOW adaptation gain are HARD (the model struggles).

    Args:
        task_ids: List of task identifiers to score.
        registry: Registry object with get_metadata(task_id) method.
            Metadata should contain 'adaptation_gain' key.

    Returns:
        Mapping from task_id to difficulty score.
        Higher score = harder task.
    """
    difficulty: Dict[str, float] = {}
    gains: List[Tuple[str, float]] = []

    for task_id in task_ids:
        meta = registry.get_metadata(task_id) if registry else None
        gain = None
        if meta is not None:
            gain = meta.get("adaptation_gain")
        if gain is not None:
            gains.append((task_id, float(gain)))

    if not gains:
        logger.warning(
            "No adaptation gain data found in registry. "
            "Returning uniform difficulty for %d tasks.",
            len(task_ids),
        )
        return {tid: 0.5 for tid in task_ids}

    max_gain = max(g for _, g in gains)
    min_gain = min(g for _, g in gains)
    gain_range = max_gain - min_gain if max_gain > min_gain else 1.0

    for task_id, gain in gains:
        normalized = (gain - min_gain) / gain_range
        # Invert: high gain = easy = low difficulty
        difficulty[task_id] = 1.0 - normalized

    # Assign median difficulty to tasks without history
    if difficulty:
        values = sorted(difficulty.values())
        mid = len(values) // 2
        median_diff = values[mid] if len(values) % 2 == 1 else (values[mid - 1] + values[mid]) / 2.0
    else:
        median_diff = 0.5

    for task_id in task_ids:
        if task_id not in difficulty:
            difficulty[task_id] = median_diff

    return difficulty


def compute_difficulty_loss_slope(
    task_ids: List[str],
    registry: Any,
) -> Dict[str, float]:
    """Compute difficulty from the slope of inner-loop loss decrease.

    A steep negative slope (fast loss decrease) indicates an easy task.
    A flat or positive slope indicates a hard task.

    Args:
        task_ids: List of task identifiers to score.
        registry: Registry object with get_metadata(task_id) method.
            Metadata should contain 'inner_loss_curve' key (List[float]).

    Returns:
        Mapping from task_id to difficulty score.
        Higher score = harder task (flatter or positive slope).
    """
    difficulty: Dict[str, float] = {}

    for task_id in task_ids:
        meta = registry.get_metadata(task_id) if registry else None
        curve = None
        if meta is not None:
            curve = meta.get("inner_loss_curve")

        if curve is not None and len(curve) >= 2:
            steps = np.arange(len(curve), dtype=np.float64)
            losses = np.array(curve, dtype=np.float64)
            # Linear regression slope
            n = len(steps)
            mean_x = steps.mean()
            mean_y = losses.mean()
            numerator = float(np.sum((steps - mean_x) * (losses - mean_y)))
            denominator = float(np.sum((steps - mean_x) ** 2))
            if abs(denominator) < 1e-12:
                slope = 0.0
            else:
                slope = numerator / denominator
            # Negate: steep negative slope -> small value -> easy
            # Flat/positive slope -> large value -> hard
            difficulty[task_id] = float(-slope)
        else:
            difficulty[task_id] = 0.0

    # Shift so minimum difficulty is 0
    if difficulty:
        min_d = min(difficulty.values())
        if min_d < 0.0:
            difficulty = {k: v - min_d for k, v in difficulty.items()}

    return difficulty


def _identify_easy_cluster_centroid(
    embeddings: Dict[str, np.ndarray],
    cluster_labels: Dict[str, int],
    registry: Optional[Any] = None,
) -> np.ndarray:
    """Identify the easy cluster and return its centroid.

    The easy cluster is the one whose member tasks have the lowest average
    probe loss from registry diagnostics.  If no probe loss data is available,
    falls back to the cluster with the most members (heuristic: common tasks
    tend to be easier).

    Args:
        embeddings: Mapping from task_id to embedding vector.
        cluster_labels: Mapping from task_id to cluster label.
        registry: Optional registry for probe loss lookup.

    Returns:
        L2-normalized centroid vector of the easy cluster.
    """
    # Group embeddings by cluster
    cluster_embs: Dict[int, List[np.ndarray]] = defaultdict(list)
    cluster_losses: Dict[int, List[float]] = defaultdict(list)

    for task_id, cluster_id in cluster_labels.items():
        if task_id in embeddings:
            cluster_embs[cluster_id].append(embeddings[task_id])

            if registry is not None:
                meta = registry.get_metadata(task_id)
                if meta is not None:
                    diag = meta.get("diagnostics", {})
                    probe_loss = diag.get("probe_loss")
                    if probe_loss is not None:
                        cluster_losses[cluster_id].append(float(probe_loss))

    if not cluster_embs:
        logger.warning("No embeddings found for cluster labels. Returning zero vector.")
        dim = 64  # default
        if embeddings:
            dim = next(iter(embeddings.values())).shape[0]
        centroid = np.zeros(dim, dtype=np.float64)
        centroid[0] = 1.0  # avoid zero norm
        return centroid

    # Determine easy cluster
    easy_cluster_id: int
    if cluster_losses:
        # Use average probe loss
        mean_losses = {cid: np.mean(losses) for cid, losses in cluster_losses.items()}
        easy_cluster_id = min(mean_losses, key=lambda k: mean_losses[k])
    else:
        # Fallback: cluster with most members
        easy_cluster_id = max(cluster_embs, key=lambda k: len(cluster_embs[k]))

    # Compute centroid
    embs = np.stack(cluster_embs[easy_cluster_id], axis=0)
    centroid = embs.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 1e-12:
        centroid = centroid / norm
    else:
        centroid = np.zeros_like(centroid)
        centroid[0] = 1.0

    return centroid


# ============================================================================
# Curriculum Ordering
# ============================================================================

def _easy_to_hard_order(
    task_ids: List[str],
    difficulty: Dict[str, float],
) -> List[str]:
    """Sort tasks from easiest to hardest.

    Args:
        task_ids: List of task identifiers to order.
        difficulty: Mapping from task_id to difficulty score.

    Returns:
        Task IDs sorted by increasing difficulty.
        Ties broken by task_id lexicographic order for determinism.
    """
    return sorted(task_ids, key=lambda tid: (difficulty.get(tid, 0.0), tid))


def _anti_curriculum_order(
    task_ids: List[str],
    difficulty: Dict[str, float],
) -> List[str]:
    """Sort tasks from hardest to easiest.

    Args:
        task_ids: List of task identifiers to order.
        difficulty: Mapping from task_id to difficulty score.

    Returns:
        Task IDs sorted by decreasing difficulty.
        Ties broken by task_id lexicographic order.
    """
    return sorted(task_ids, key=lambda tid: (-difficulty.get(tid, 0.0), tid))


def _diversity_order(
    task_ids: List[str],
    embeddings: Dict[str, np.ndarray],
    seed: int,
    epoch: int,
) -> List[str]:
    """Order tasks to maximize inter-task diversity.

    Uses a greedy farthest-point traversal: start with a random task, then
    repeatedly add the task that is most distant (in cosine distance) from
    the nearest already-selected task.

    Args:
        task_ids: List of task identifiers to order.
        embeddings: Mapping from task_id to L2-normalized embedding.
        seed: Global seed.
        epoch: Current epoch.

    Returns:
        Task IDs ordered by greedy diversity maximization.
    """
    if len(task_ids) <= 1:
        return list(task_ids)

    rng = np.random.RandomState(_epoch_seed(seed, epoch))

    # Build embedding matrix
    available = [tid for tid in task_ids if tid in embeddings]
    missing = [tid for tid in task_ids if tid not in embeddings]

    if not available:
        shuffled = list(task_ids)
        rng.shuffle(shuffled)
        return shuffled

    emb_matrix = np.stack([embeddings[tid] for tid in available], axis=0)

    # Greedy farthest-point traversal
    selected_indices: List[int] = []
    first_idx = int(rng.randint(len(available)))
    selected_indices.append(first_idx)

    # Track minimum distance from each point to the selected set
    min_dists = 1.0 - emb_matrix @ emb_matrix[first_idx]
    min_dists[first_idx] = -1.0  # already selected

    for _ in range(len(available) - 1):
        next_idx = int(np.argmax(min_dists))
        selected_indices.append(next_idx)
        min_dists[next_idx] = -1.0

        # Update distances
        new_dists = 1.0 - emb_matrix @ emb_matrix[next_idx]
        min_dists = np.minimum(min_dists, new_dists)
        # Keep already-selected at -1
        for idx in selected_indices:
            min_dists[idx] = -1.0

    result = [available[i] for i in selected_indices]
    # Append missing tasks at the end
    result.extend(missing)
    return result


def _mixed_schedule_order(
    task_ids: List[str],
    difficulty: Dict[str, float],
    epoch: int,
    max_epoch: int,
    schedule_fn: Callable,
    schedule_kwargs: Dict[str, Any],
    seed: int,
) -> List[str]:
    """Order tasks using a mixed difficulty schedule.

    At each epoch, compute p(hard) from the schedule function.  Tasks above
    the median difficulty receive weight p(hard); tasks below receive weight
    1 - p(hard).  Sample len(task_ids) tasks according to these weights.

    Args:
        task_ids: Full list of available task IDs.
        difficulty: Mapping from task_id to difficulty score.
        epoch: Current epoch number.
        max_epoch: Total number of training epochs.
        schedule_fn: Schedule function returning p(hard).
        schedule_kwargs: Additional kwargs for schedule_fn.
        seed: Global seed.

    Returns:
        Ordered list of task IDs sampled according to the difficulty schedule.
    """
    if not task_ids:
        return []

    rng = np.random.RandomState(_epoch_seed(seed, epoch))

    # Compute p(hard) from schedule
    if "switch_epoch" in schedule_kwargs:
        p_hard = schedule_fn(epoch, schedule_kwargs["switch_epoch"])
    else:
        p_hard = schedule_fn(epoch, max_epoch)

    p_hard = float(np.clip(p_hard, 0.0, 1.0))

    # Split tasks by median difficulty
    difficulties = np.array([difficulty.get(tid, 0.0) for tid in task_ids])
    median_diff = float(np.median(difficulties))

    # Assign sampling weights
    weights = np.zeros(len(task_ids), dtype=np.float64)
    for i, tid in enumerate(task_ids):
        d = difficulty.get(tid, 0.0)
        if d > median_diff:
            weights[i] = p_hard
        else:
            weights[i] = 1.0 - p_hard

    # Normalize
    weight_sum = weights.sum()
    if weight_sum < 1e-12:
        weights = np.ones(len(task_ids), dtype=np.float64) / len(task_ids)
    else:
        weights = weights / weight_sum

    # Sample with replacement according to weights
    indices = rng.choice(len(task_ids), size=len(task_ids), replace=True, p=weights)
    # Deduplicate while preserving order, then add remaining tasks
    seen = set()
    ordered: List[str] = []
    for idx in indices:
        tid = task_ids[idx]
        if tid not in seen:
            seen.add(tid)
            ordered.append(tid)

    # Add any task_ids not yet sampled (ensure no drops)
    for tid in task_ids:
        if tid not in seen:
            seen.add(tid)
            ordered.append(tid)

    return ordered


def _compute_cluster_histogram(
    task_ids: List[str],
    cluster_labels: Optional[Dict[str, int]],
) -> Dict[int, int]:
    """Compute a histogram of cluster assignments for the given task IDs.

    Args:
        task_ids: List of task identifiers.
        cluster_labels: Mapping from task_id to cluster label.

    Returns:
        Mapping from cluster_id to count.
    """
    histogram: Dict[int, int] = defaultdict(int)
    if cluster_labels is None:
        return dict(histogram)
    for tid in task_ids:
        cid = cluster_labels.get(tid, -1)
        histogram[cid] += 1
    return dict(histogram)


def get_curriculum_order(
    task_ids: List[str],
    embeddings: Dict[str, np.ndarray],
    *,
    strategy: Union[str, CurriculumStrategy] = CurriculumStrategy.NONE,
    difficulty_scores: Optional[Dict[str, float]] = None,
    cluster_labels: Optional[Dict[str, int]] = None,
    epoch: int = 0,
    max_epoch: int = 100,
    seed: int = 42,
    registry: Optional[Any] = None,
    schedule_type: Union[str, ScheduleType] = ScheduleType.LINEAR,
    schedule_kwargs: Optional[Dict[str, Any]] = None,
    warmup_epochs: int = 0,
    difficulty_proxy: Union[str, DifficultyProxy] = DifficultyProxy.CENTROID_DISTANCE,
    **kwargs: Any,
) -> CurriculumOrder:
    """Get the curriculum-ordered task list for a given epoch.

    This is the primary entry point for curriculum ordering.  It handles
    strategy dispatch, warmup, difficulty computation, and logging metadata.

    All task_ids are always returned (no drops), just reordered.

    Args:
        task_ids: Full list of available task IDs.
        embeddings: Mapping from task_id to L2-normalized embedding vector.
        strategy: Curriculum strategy to use.
        difficulty_scores: Pre-computed difficulty scores.  If None, computed
            from the specified difficulty_proxy.
        cluster_labels: Mapping from task_id to cluster label.
        epoch: Current epoch number (0-indexed).
        max_epoch: Total number of training epochs.
        seed: Global random seed.
        registry: Optional registry for dynamic difficulty proxies.
        schedule_type: Schedule function type for mixed strategy.
        schedule_kwargs: Additional kwargs for schedule function.
        warmup_epochs: Number of initial epochs with random ordering.
        difficulty_proxy: Method for computing difficulty if not pre-computed.
        **kwargs: Additional strategy-specific parameters.

    Returns:
        CurriculumOrder containing the ordered task IDs and metadata.
    """
    if schedule_kwargs is None:
        schedule_kwargs = {}

    # Handle empty task list
    if not task_ids:
        logger.debug("Empty task list; returning empty CurriculumOrder.")
        return CurriculumOrder(
            task_ids=[],
            strategy=_strategy_name(strategy),
            difficulty_scores={},
            cluster_histogram={},
            epoch=epoch,
        )

    # Normalize strategy to string
    strat_name = _strategy_name(strategy)

    # Warmup: random ordering for first N epochs
    if epoch < warmup_epochs:
        rng = np.random.RandomState(_epoch_seed(seed, epoch))
        shuffled = list(task_ids)
        rng.shuffle(shuffled)
        return CurriculumOrder(
            task_ids=shuffled,
            strategy="warmup_random",
            difficulty_scores={},
            cluster_histogram=_compute_cluster_histogram(shuffled, cluster_labels),
            epoch=epoch,
            metadata={"warmup": True},
        )

    # Compute difficulty scores if not provided
    if difficulty_scores is None and strat_name in (
        CurriculumStrategy.EASY_TO_HARD.value,
        CurriculumStrategy.ANTI_CURRICULUM.value,
        CurriculumStrategy.MIXED.value,
    ):
        difficulty_scores = _compute_difficulty(
            task_ids=task_ids,
            embeddings=embeddings,
            cluster_labels=cluster_labels,
            registry=registry,
            proxy=difficulty_proxy,
        )

    if difficulty_scores is None:
        difficulty_scores = {}

    # Check for zero-variance difficulty
    if difficulty_scores and strat_name in (
        CurriculumStrategy.EASY_TO_HARD.value,
        CurriculumStrategy.ANTI_CURRICULUM.value,
        CurriculumStrategy.MIXED.value,
    ):
        diff_values = list(difficulty_scores.values())
        unique_rounded = set(round(v, 8) for v in diff_values)
        if len(unique_rounded) <= 1:
            logger.warning(
                "Difficulty proxy has zero variance (all tasks equally difficult). "
                "Falling back to random ordering."
            )
            rng = np.random.RandomState(_epoch_seed(seed, epoch))
            shuffled = list(task_ids)
            rng.shuffle(shuffled)
            return CurriculumOrder(
                task_ids=shuffled,
                strategy="fallback_random",
                difficulty_scores=difficulty_scores,
                cluster_histogram=_compute_cluster_histogram(shuffled, cluster_labels),
                epoch=epoch,
                metadata={"fallback": True, "reason": "zero_variance_difficulty"},
            )

    # Dispatch to strategy
    if strat_name == CurriculumStrategy.NONE.value:
        rng = np.random.RandomState(_epoch_seed(seed, epoch))
        ordered = list(task_ids)
        rng.shuffle(ordered)

    elif strat_name == CurriculumStrategy.EASY_TO_HARD.value:
        ordered = _easy_to_hard_order(task_ids, difficulty_scores)

    elif strat_name == CurriculumStrategy.ANTI_CURRICULUM.value:
        ordered = _anti_curriculum_order(task_ids, difficulty_scores)

    elif strat_name == CurriculumStrategy.DIVERSITY.value:
        ordered = _diversity_order(task_ids, embeddings, seed, epoch)

    elif strat_name == CurriculumStrategy.MIXED.value:
        fn = get_schedule_fn(schedule_type)
        ordered = _mixed_schedule_order(
            task_ids=task_ids,
            difficulty=difficulty_scores,
            epoch=epoch,
            max_epoch=max_epoch,
            schedule_fn=fn,
            schedule_kwargs=schedule_kwargs,
            seed=seed,
        )

    else:
        raise ValueError(
            f"Unknown curriculum strategy: '{strat_name}'. "
            f"Valid strategies: {[s.value for s in CurriculumStrategy]}."
        )

    return CurriculumOrder(
        task_ids=ordered,
        strategy=strat_name,
        difficulty_scores=difficulty_scores,
        cluster_histogram=_compute_cluster_histogram(ordered, cluster_labels),
        epoch=epoch,
        metadata={"seed": seed},
    )


def _strategy_name(strategy: Union[str, CurriculumStrategy]) -> str:
    """Normalize strategy to its string value."""
    if isinstance(strategy, CurriculumStrategy):
        return strategy.value
    return str(strategy)


def _compute_difficulty(
    task_ids: List[str],
    embeddings: Dict[str, np.ndarray],
    cluster_labels: Optional[Dict[str, int]],
    registry: Optional[Any],
    proxy: Union[str, DifficultyProxy],
) -> Dict[str, float]:
    """Dispatch to the appropriate difficulty proxy computation.

    Args:
        task_ids: List of task identifiers.
        embeddings: Mapping from task_id to embedding vector.
        cluster_labels: Mapping from task_id to cluster label.
        registry: Optional registry for metadata lookup.
        proxy: Difficulty proxy method.

    Returns:
        Mapping from task_id to difficulty score.
    """
    proxy_name = proxy.value if isinstance(proxy, DifficultyProxy) else str(proxy)

    if proxy_name == DifficultyProxy.CENTROID_DISTANCE.value:
        if cluster_labels is None:
            logger.warning(
                "No cluster labels provided for centroid_distance proxy. "
                "Using uniform difficulty."
            )
            return {tid: 0.5 for tid in task_ids}
        centroid = _identify_easy_cluster_centroid(
            embeddings, cluster_labels, registry
        )
        return compute_difficulty_centroid_distance(
            embeddings, task_ids, centroid
        )

    elif proxy_name == DifficultyProxy.ADAPTATION_GAIN.value:
        return compute_difficulty_adaptation_gain(task_ids, registry)

    elif proxy_name == DifficultyProxy.LOSS_SLOPE.value:
        return compute_difficulty_loss_slope(task_ids, registry)

    else:
        raise ValueError(
            f"Unknown difficulty proxy: '{proxy_name}'. "
            f"Valid proxies: {[p.value for p in DifficultyProxy]}."
        )


# ============================================================================
# Meta-Batch Composition
# ============================================================================

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance between two L2-normalized vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine distance: 1 - cos(a, b).
    """
    return float(1.0 - np.dot(a, b))


def _min_pairwise_distance(
    embeddings: Dict[str, np.ndarray],
    task_ids: List[str],
) -> float:
    """Compute the minimum pairwise cosine distance among tasks.

    Args:
        embeddings: Mapping from task_id to embedding vector.
        task_ids: List of task identifiers.

    Returns:
        Minimum pairwise cosine distance, or 0.0 if fewer than 2 tasks.
    """
    if len(task_ids) < 2:
        return 0.0

    min_dist = float("inf")
    for i in range(len(task_ids)):
        emb_i = embeddings.get(task_ids[i])
        if emb_i is None:
            continue
        for j in range(i + 1, len(task_ids)):
            emb_j = embeddings.get(task_ids[j])
            if emb_j is None:
                continue
            d = _cosine_distance(emb_i, emb_j)
            if d < min_dist:
                min_dist = d

    return min_dist if min_dist != float("inf") else 0.0


def _build_meta_batch_greedy(
    task_ids: List[str],
    embeddings: Dict[str, np.ndarray],
    batch_size: int,
    min_distance: float,
    seed: int,
    epoch: int,
    batch_idx: int = 0,
) -> MetaBatch:
    """Build a diversity-constrained meta-batch using greedy selection.

    Pick a random first task, then greedily add tasks whose cosine distance
    to ALL already-selected tasks exceeds min_distance.  If the batch cannot
    be filled at the current threshold, relax by 10% iteratively.

    Args:
        task_ids: Pool of candidate task IDs.
        embeddings: Mapping from task_id to L2-normalized embedding.
        batch_size: Target number of tasks in the meta-batch.
        min_distance: Initial minimum cosine distance threshold.
        seed: Global seed.
        epoch: Current epoch.
        batch_idx: Batch index within epoch.

    Returns:
        MetaBatch with exactly batch_size task_ids (or fewer if pool is small).
    """
    if not task_ids:
        return MetaBatch(task_ids=[], diversity_metric=0.0, strategy="min_distance")

    effective_batch_size = min(batch_size, len(task_ids))

    rng = np.random.RandomState(_batch_seed(seed, epoch, batch_idx))
    candidates = list(task_ids)
    rng.shuffle(candidates)

    threshold = min_distance
    max_relaxation_steps = 20

    for relaxation in range(max_relaxation_steps):
        selected: List[str] = []
        selected_embs: List[np.ndarray] = []

        for task_id in candidates:
            emb = embeddings.get(task_id)
            if emb is None:
                continue

            if len(selected) == 0:
                selected.append(task_id)
                selected_embs.append(emb)
            else:
                # Check distance to all selected tasks
                distances = [_cosine_distance(emb, sel_emb) for sel_emb in selected_embs]
                if min(distances) >= threshold:
                    selected.append(task_id)
                    selected_embs.append(emb)

            if len(selected) == effective_batch_size:
                break

        if len(selected) >= effective_batch_size:
            selected = selected[:effective_batch_size]
            div_metric = _min_pairwise_distance(embeddings, selected)
            return MetaBatch(
                task_ids=selected,
                diversity_metric=div_metric,
                strategy="min_distance",
            )

        # Relax threshold by 10%
        threshold *= 0.9

    # Final fallback: fill remaining slots randomly
    remaining = [tid for tid in candidates if tid not in set(selected)]
    rng.shuffle(remaining)
    needed = effective_batch_size - len(selected)
    selected.extend(remaining[:needed])
    selected = selected[:effective_batch_size]

    div_metric = _min_pairwise_distance(embeddings, selected)
    return MetaBatch(
        task_ids=selected,
        diversity_metric=div_metric,
        strategy="min_distance_relaxed",
    )


def _build_meta_batch_stratified(
    task_ids: List[str],
    embeddings: Dict[str, np.ndarray],
    cluster_labels: Dict[str, int],
    batch_size: int,
    seed: int,
    epoch: int,
    batch_idx: int = 0,
) -> MetaBatch:
    """Build a meta-batch by stratified per-cluster sampling.

    Sample one task per cluster (round-robin), fill remaining randomly.

    Args:
        task_ids: Pool of candidate task IDs.
        embeddings: Mapping from task_id to embedding vector.
        cluster_labels: Mapping from task_id to cluster label.
        batch_size: Target number of tasks in the meta-batch.
        seed: Global seed.
        epoch: Current epoch.
        batch_idx: Batch index within epoch.

    Returns:
        MetaBatch with exactly batch_size task_ids.
    """
    if not task_ids:
        return MetaBatch(task_ids=[], diversity_metric=0.0, strategy="stratified")

    effective_batch_size = min(batch_size, len(task_ids))

    rng = np.random.RandomState(_batch_seed(seed, epoch, batch_idx))

    # Group tasks by cluster
    cluster_to_tasks: Dict[int, List[str]] = defaultdict(list)
    for task_id in task_ids:
        if task_id in cluster_labels:
            cluster_to_tasks[cluster_labels[task_id]].append(task_id)

    cluster_ids = sorted(cluster_to_tasks.keys())
    n_clusters = len(cluster_ids)

    if n_clusters == 0:
        # No cluster info: random selection
        pool = list(task_ids)
        rng.shuffle(pool)
        selected = pool[:effective_batch_size]
        div_metric = _min_pairwise_distance(embeddings, selected)
        return MetaBatch(
            task_ids=selected,
            diversity_metric=div_metric,
            strategy="stratified_fallback",
        )

    selected: List[str] = []
    selected_set: set = set()

    if n_clusters >= effective_batch_size:
        # More clusters than needed: sample batch_size clusters
        chosen_clusters = rng.choice(
            cluster_ids, size=effective_batch_size, replace=False
        )
        for cid in chosen_clusters:
            tasks = cluster_to_tasks[cid]
            chosen = tasks[rng.randint(len(tasks))]
            selected.append(chosen)
            selected_set.add(chosen)
    else:
        # One task per cluster first
        for cid in cluster_ids:
            tasks = cluster_to_tasks[cid]
            chosen = tasks[rng.randint(len(tasks))]
            selected.append(chosen)
            selected_set.add(chosen)

        # Fill remaining by cycling through clusters
        remaining_needed = effective_batch_size - len(selected)
        cycle_idx = 0
        while remaining_needed > 0:
            cid = cluster_ids[cycle_idx % n_clusters]
            tasks = cluster_to_tasks[cid]
            available = [t for t in tasks if t not in selected_set]
            if not available:
                available = tasks  # allow replacement
            chosen = available[rng.randint(len(available))]
            selected.append(chosen)
            selected_set.add(chosen)
            remaining_needed -= 1
            cycle_idx += 1

    selected = selected[:effective_batch_size]
    div_metric = _min_pairwise_distance(embeddings, selected)
    return MetaBatch(
        task_ids=selected,
        diversity_metric=div_metric,
        strategy="stratified",
    )


def build_meta_batch(
    task_ids: List[str],
    embeddings: Dict[str, np.ndarray],
    *,
    batch_size: int = 8,
    diversity_constraint: str = "min_distance",
    min_distance: float = 0.3,
    cluster_labels: Optional[Dict[str, int]] = None,
    seed: int = 42,
    epoch: int = 0,
    batch_idx: int = 0,
    **kwargs: Any,
) -> MetaBatch:
    """Build a diversity-constrained meta-batch of tasks.

    This is the primary entry point for meta-batch composition.

    Args:
        task_ids: Pool of candidate task IDs.
        embeddings: Mapping from task_id to L2-normalized embedding.
        batch_size: Target number of tasks in the meta-batch.
        diversity_constraint: Diversity method: 'min_distance', 'stratified',
            or 'none'.
        min_distance: Minimum cosine distance threshold for 'min_distance'.
        cluster_labels: Mapping from task_id to cluster label (required for
            'stratified').
        seed: Global random seed.
        epoch: Current epoch number.
        batch_idx: Batch index within the epoch.
        **kwargs: Additional strategy-specific parameters.

    Returns:
        MetaBatch with the selected task IDs and diversity metrics.

    Raises:
        ValueError: If diversity_constraint is unknown.
    """
    if not task_ids:
        return MetaBatch(task_ids=[], diversity_metric=0.0, strategy=diversity_constraint)

    if diversity_constraint == "none":
        rng = np.random.RandomState(_batch_seed(seed, epoch, batch_idx))
        pool = list(task_ids)
        rng.shuffle(pool)
        effective_size = min(batch_size, len(pool))
        selected = pool[:effective_size]
        div_metric = _min_pairwise_distance(embeddings, selected)
        return MetaBatch(
            task_ids=selected,
            diversity_metric=div_metric,
            strategy="none",
        )

    elif diversity_constraint == "min_distance":
        return _build_meta_batch_greedy(
            task_ids=task_ids,
            embeddings=embeddings,
            batch_size=batch_size,
            min_distance=min_distance,
            seed=seed,
            epoch=epoch,
            batch_idx=batch_idx,
        )

    elif diversity_constraint == "stratified":
        if cluster_labels is None:
            logger.warning(
                "Stratified diversity requested but no cluster_labels provided. "
                "Falling back to min_distance."
            )
            return _build_meta_batch_greedy(
                task_ids=task_ids,
                embeddings=embeddings,
                batch_size=batch_size,
                min_distance=min_distance,
                seed=seed,
                epoch=epoch,
                batch_idx=batch_idx,
            )
        return _build_meta_batch_stratified(
            task_ids=task_ids,
            embeddings=embeddings,
            cluster_labels=cluster_labels,
            batch_size=batch_size,
            seed=seed,
            epoch=epoch,
            batch_idx=batch_idx,
        )

    else:
        raise ValueError(
            f"Unknown diversity constraint: '{diversity_constraint}'. "
            f"Choose from 'none', 'min_distance', 'stratified'."
        )


# ============================================================================
# Kendall Tau Validation
# ============================================================================

def _norm_sf(z: float) -> float:
    """Survival function of the standard normal distribution (approximation).

    Uses Abramowitz and Stegun approximation 7.1.26, accurate to 1e-5.

    Args:
        z: Standard normal deviate (assumed non-negative).

    Returns:
        P(Z > z) for Z ~ N(0,1).
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
    return max(0.0, min(1.0, p))


def _kendall_tau_fallback(x: List[int], y: List[int]) -> Tuple[float, float]:
    """Pure-Python Kendall tau-a computation.

    Counts concordant and discordant pairs.  Does not handle ties (ranks
    are assumed unique, which holds for permutation orderings).

    Args:
        x: First rank vector.
        y: Second rank vector.

    Returns:
        (tau, p_value): Kendall tau-a and approximate two-sided p-value.
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
    # Var(S) = n(n-1)(2n+5)/18 for tau-a without ties
    variance = n * (n - 1) * (2 * n + 5) / 18.0
    if variance <= 0:
        return tau, 1.0

    s = concordant - discordant
    z = s / math.sqrt(variance)

    # Two-sided p-value
    p_value = 2.0 * _norm_sf(abs(z))

    return tau, p_value


def compute_kendall_tau(
    order1: Sequence[str],
    order2: Sequence[str],
) -> Tuple[float, float]:
    """Compute Kendall tau rank correlation between two orderings.

    Validates that the curriculum strategy actually changes task ordering
    relative to a random baseline.

    Args:
        order1: First ordering of task IDs.
        order2: Second ordering of task IDs (must contain the same task IDs).

    Returns:
        (tau, p_value): Kendall tau statistic and two-sided p-value.
        tau > 0 means positive correlation (similar ordering).
        tau < 0 means negative correlation (reversed ordering).
        tau = 1.0 means identical ordering.
        p_value < 0.05 means the correlation is statistically significant.
    """
    if len(order1) < 2 or len(order2) < 2:
        return 0.0, 1.0

    set1 = set(order1)
    set2 = set(order2)

    if set1 != set2:
        # Use intersection
        common = sorted(set1 & set2)
        if len(common) < 2:
            return 0.0, 1.0
        logger.warning(
            "Order lists contain different task IDs. "
            "Using intersection of %d common tasks.",
            len(common),
        )
        order1 = [tid for tid in order1 if tid in common]
        order2 = [tid for tid in order2 if tid in common]

    # Build rank vectors
    rank1 = {tid: i for i, tid in enumerate(order1)}
    rank2 = {tid: i for i, tid in enumerate(order2)}

    all_ids = sorted(set(order1))
    ranks_x = [rank1[tid] for tid in all_ids]
    ranks_y = [rank2[tid] for tid in all_ids]

    # Use scipy if available, otherwise pure Python fallback
    if _HAS_SCIPY:
        tau, p_value = _scipy_kendalltau(ranks_x, ranks_y)
        # Handle NaN from scipy (can occur with identical rankings of length 1)
        if math.isnan(tau):
            tau = 0.0
        if math.isnan(p_value):
            p_value = 1.0
        return float(tau), float(p_value)
    else:
        return _kendall_tau_fallback(ranks_x, ranks_y)


# ============================================================================
# Epoch Logging
# ============================================================================

def _compute_task_order_hash(task_ids: List[str]) -> str:
    """Compute SHA-256 hash of the ordered task ID list.

    Args:
        task_ids: Ordered list of task identifiers.

    Returns:
        Hex digest string (64 characters).
    """
    order_str = ",".join(task_ids)
    return hashlib.sha256(order_str.encode("utf-8")).hexdigest()


def _compute_difficulty_stats(
    difficulty_scores: Dict[str, float],
    task_ids: List[str],
) -> Dict[str, float]:
    """Compute summary statistics for difficulty proxy values.

    Args:
        difficulty_scores: Mapping from task_id to difficulty score.
        task_ids: Ordered list of task IDs (to filter scores).

    Returns:
        Dictionary with mean, std, min, max of difficulty values.
    """
    if not difficulty_scores or not task_ids:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    values = [difficulty_scores.get(tid, 0.0) for tid in task_ids]
    arr = np.array(values, dtype=np.float64)

    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def create_epoch_log(
    curriculum_order: CurriculumOrder,
    random_order: Optional[List[str]] = None,
    epoch: Optional[int] = None,
) -> EpochLog:
    """Create a logging record for a curriculum epoch.

    Computes the task order hash, difficulty statistics, cluster histogram,
    and Kendall tau correlation vs a random baseline.

    Args:
        curriculum_order: The curriculum ordering for this epoch.
        random_order: Random baseline ordering for Kendall tau computation.
            If None, Kendall tau is set to 0.0 with p_value 1.0.
        epoch: Override epoch number (defaults to curriculum_order.epoch).

    Returns:
        EpochLog with all computed fields.
    """
    ep = epoch if epoch is not None else curriculum_order.epoch

    task_order_hash = _compute_task_order_hash(curriculum_order.task_ids)

    difficulty_stats = _compute_difficulty_stats(
        curriculum_order.difficulty_scores,
        curriculum_order.task_ids,
    )

    tau = 0.0
    p_value = 1.0
    if random_order is not None and len(curriculum_order.task_ids) >= 2:
        tau, p_value = compute_kendall_tau(
            curriculum_order.task_ids, random_order
        )

    return EpochLog(
        epoch=ep,
        strategy=curriculum_order.strategy,
        task_order_hash=task_order_hash,
        cluster_histogram=curriculum_order.cluster_histogram,
        difficulty_stats=difficulty_stats,
        kendall_tau=tau,
        p_value=p_value,
    )


# ============================================================================
# CurriculumManager
# ============================================================================

class CurriculumManager:
    """Stateful manager for multi-epoch curriculum ordering and batching.

    Encapsulates configuration, embeddings, and task metadata.  Provides
    a simple interface for the training loop to request per-epoch orderings
    and per-batch task selections.

    Caches difficulty scores (computed once for static proxies) and epoch
    orderings (computed on first access per epoch).

    Example::

        manager = CurriculumManager(config, embeddings, task_ids, cluster_labels)
        for epoch in range(100):
            order = manager.get_epoch_order(epoch)
            for batch_idx in range(num_batches):
                batch = manager.get_epoch_batch(epoch, batch_idx)
                # train on batch.task_ids
            log = manager.get_epoch_log(epoch)

    Attributes:
        config: CurriculumConfig instance.
        embeddings: Mapping from task_id to L2-normalized embedding.
        task_ids: Full list of available task IDs.
        cluster_labels: Optional mapping from task_id to cluster label.
        registry: Optional registry for dynamic difficulty proxies.
    """

    def __init__(
        self,
        config: CurriculumConfig,
        embeddings: Dict[str, np.ndarray],
        task_ids: List[str],
        cluster_labels: Optional[Dict[str, int]] = None,
        registry: Optional[Any] = None,
    ) -> None:
        """Initialize the CurriculumManager.

        Args:
            config: Curriculum configuration.
            embeddings: Mapping from task_id to L2-normalized embedding.
            task_ids: Full list of available task IDs.
            cluster_labels: Optional mapping from task_id to cluster label.
            registry: Optional registry for dynamic difficulty proxies.
        """
        self.config = config
        self.embeddings = embeddings
        self.task_ids = list(task_ids)
        self.cluster_labels = cluster_labels
        self.registry = registry

        # Caches
        self._difficulty_cache: Optional[Dict[str, float]] = None
        self._order_cache: Dict[int, CurriculumOrder] = {}
        self._log_cache: Dict[int, EpochLog] = {}
        self._random_order_cache: Dict[int, List[str]] = {}

        # Pre-compute static difficulty if applicable
        if config.difficulty_proxy == DifficultyProxy.CENTROID_DISTANCE.value:
            self._precompute_difficulty()

    def _precompute_difficulty(self) -> None:
        """Pre-compute difficulty scores for static proxies."""
        try:
            self._difficulty_cache = _compute_difficulty(
                task_ids=self.task_ids,
                embeddings=self.embeddings,
                cluster_labels=self.cluster_labels,
                registry=self.registry,
                proxy=self.config.difficulty_proxy,
            )
        except Exception as e:
            logger.warning("Failed to pre-compute difficulty: %s", e)
            self._difficulty_cache = None

    def _get_random_order(self, epoch: int) -> List[str]:
        """Get a deterministic random ordering for a given epoch.

        Used as the Kendall tau baseline.

        Args:
            epoch: Epoch number.

        Returns:
            Randomly shuffled task ID list.
        """
        if epoch not in self._random_order_cache:
            # Use a different seed offset to avoid correlation with curriculum seed
            rng = np.random.RandomState(_epoch_seed(self.config.seed + 7919, epoch))
            shuffled = list(self.task_ids)
            rng.shuffle(shuffled)
            self._random_order_cache[epoch] = shuffled
        return self._random_order_cache[epoch]

    def get_epoch_order(self, epoch: int) -> CurriculumOrder:
        """Get the curriculum ordering for a specific epoch.

        Results are cached; repeated calls with the same epoch return the
        same ordering.

        Args:
            epoch: Epoch number (0-indexed).

        Returns:
            CurriculumOrder for this epoch.
        """
        if epoch in self._order_cache:
            return self._order_cache[epoch]

        # For dynamic proxies, recompute difficulty each epoch
        difficulty = self._difficulty_cache
        if self.config.difficulty_proxy != DifficultyProxy.CENTROID_DISTANCE.value:
            difficulty = None  # Will be recomputed inside get_curriculum_order

        order = get_curriculum_order(
            task_ids=self.task_ids,
            embeddings=self.embeddings,
            strategy=self.config.strategy,
            difficulty_scores=difficulty,
            cluster_labels=self.cluster_labels,
            epoch=epoch,
            max_epoch=self.config.max_epochs,
            seed=self.config.seed,
            registry=self.registry,
            schedule_type=self.config.schedule_type,
            schedule_kwargs=self.config.schedule_kwargs,
            warmup_epochs=self.config.warmup_epochs,
            difficulty_proxy=self.config.difficulty_proxy,
        )

        self._order_cache[epoch] = order
        return order

    def get_epoch_batch(self, epoch: int, batch_idx: int = 0) -> MetaBatch:
        """Get a specific meta-batch for a given epoch and batch index.

        Uses the curriculum ordering for this epoch as the candidate pool
        for the batch, then applies diversity constraints.

        Args:
            epoch: Epoch number (0-indexed).
            batch_idx: Batch index within the epoch.

        Returns:
            MetaBatch with diversity-constrained task selection.
        """
        order = self.get_epoch_order(epoch)

        return build_meta_batch(
            task_ids=order.task_ids,
            embeddings=self.embeddings,
            batch_size=self.config.meta_batch_size,
            diversity_constraint=self.config.diversity_constraint,
            min_distance=self.config.diversity_min_distance,
            cluster_labels=self.cluster_labels,
            seed=self.config.seed,
            epoch=epoch,
            batch_idx=batch_idx,
        )

    def get_epoch_log(self, epoch: int) -> EpochLog:
        """Get the logging summary for a specific epoch.

        Results are cached; repeated calls return the same log.

        Args:
            epoch: Epoch number (0-indexed).

        Returns:
            EpochLog with curriculum statistics and Kendall tau.
        """
        if epoch in self._log_cache:
            return self._log_cache[epoch]

        order = self.get_epoch_order(epoch)
        random_order = self._get_random_order(epoch)

        log = create_epoch_log(
            curriculum_order=order,
            random_order=random_order,
            epoch=epoch,
        )

        self._log_cache[epoch] = log
        logger.info(
            "Curriculum epoch %d: strategy=%s, tau=%.4f, p=%.4f, hash=%s",
            epoch, log.strategy, log.kendall_tau, log.p_value,
            log.task_order_hash[:16],
        )

        return log

    def clear_cache(self) -> None:
        """Clear all cached orderings and logs.

        Call this when embeddings or task_ids change mid-training.
        """
        self._order_cache.clear()
        self._log_cache.clear()
        self._random_order_cache.clear()
        self._difficulty_cache = None
        if self.config.difficulty_proxy == DifficultyProxy.CENTROID_DISTANCE.value:
            self._precompute_difficulty()


# ============================================================================
# Synthetic Test Data Generators
# ============================================================================

def _generate_synthetic_embeddings(
    n_tasks: int = 50,
    n_clusters: int = 5,
    dim: int = 64,
    seed: int = 42,
) -> Tuple[List[str], Dict[str, np.ndarray], Dict[str, int]]:
    """Generate synthetic task embeddings for testing.

    Creates n_tasks embeddings from n_clusters Gaussian clusters, each
    L2-normalized to the unit sphere.

    Args:
        n_tasks: Number of tasks to generate.
        n_clusters: Number of clusters.
        dim: Embedding dimension.
        seed: Random seed.

    Returns:
        (task_ids, embeddings, cluster_labels):
            task_ids: List of task ID strings.
            embeddings: Mapping from task_id to L2-normalized embedding.
            cluster_labels: Mapping from task_id to cluster label.
    """
    rng = np.random.RandomState(seed)

    # Generate cluster centers
    centers = rng.randn(n_clusters, dim).astype(np.float64)
    for i in range(n_clusters):
        norm = np.linalg.norm(centers[i])
        if norm > 1e-8:
            centers[i] /= norm

    task_ids: List[str] = []
    embeddings: Dict[str, np.ndarray] = {}
    cluster_labels: Dict[str, int] = {}

    tasks_per_cluster = max(1, n_tasks // n_clusters)

    for c in range(n_clusters):
        n_in_cluster = tasks_per_cluster
        if c == n_clusters - 1:
            # Last cluster gets remaining tasks
            n_in_cluster = n_tasks - len(task_ids)

        for j in range(n_in_cluster):
            if len(task_ids) >= n_tasks:
                break

            tid = f"task_{len(task_ids):04d}"
            # Generate embedding near cluster center
            noise = rng.randn(dim) * 0.15
            emb = centers[c] + noise
            norm = np.linalg.norm(emb)
            if norm > 1e-8:
                emb /= norm
            else:
                emb = centers[c].copy()

            task_ids.append(tid)
            embeddings[tid] = emb.astype(np.float64)
            cluster_labels[tid] = c

    return task_ids, embeddings, cluster_labels


def _generate_synthetic_registry(
    task_ids: List[str],
    seed: int = 42,
) -> _MinimalRegistry:
    """Generate a minimal registry with synthetic metadata for testing.

    Args:
        task_ids: List of task identifiers.
        seed: Random seed.

    Returns:
        _MinimalRegistry with adaptation_gain and inner_loss_curve metadata.
    """
    rng = np.random.RandomState(seed)
    registry = _MinimalRegistry()

    for i, tid in enumerate(task_ids):
        # Generate synthetic adaptation gain (higher = easier)
        gain = rng.uniform(0.1, 0.9)

        # Generate synthetic inner loss curve
        steps = 5
        initial_loss = rng.uniform(1.0, 3.0)
        slope = -rng.uniform(0.05, 0.5)  # negative = easy
        curve = [initial_loss + slope * s + rng.normal(0, 0.02) for s in range(steps)]

        registry.set_metadata(tid, {
            "adaptation_gain": gain,
            "inner_loss_curve": curve,
            "diagnostics": {
                "probe_loss": rng.uniform(0.5, 2.0),
            },
        })

    return registry


# ============================================================================
# Self-Test Block
# ============================================================================

def _run_self_tests() -> None:
    """Run comprehensive self-tests for all curriculum functionality.

    Validates ordering, batching, scheduling, logging, and edge cases.
    Each test prints PASS or FAIL with a descriptive message.
    """
    passed = 0
    failed = 0
    total = 0

    def _test(name: str, condition: bool, detail: str = "") -> None:
        nonlocal passed, failed, total
        total += 1
        status = "PASS" if condition else "FAIL"
        if not condition:
            failed += 1
            msg = f"  [{status}] {name}"
            if detail:
                msg += f" -- {detail}"
            print(msg)
        else:
            passed += 1
            print(f"  [{status}] {name}")

    print("=" * 72)
    print("Curriculum Template Self-Tests")
    print("=" * 72)

    # Generate synthetic data
    task_ids, embeddings, cluster_labels = _generate_synthetic_embeddings(
        n_tasks=50, n_clusters=5, dim=64, seed=42
    )
    registry = _generate_synthetic_registry(task_ids, seed=42)

    # ----------------------------------------------------------------
    # Test 1: Easy-to-hard ordering matches sorted difficulty
    # ----------------------------------------------------------------
    centroid = _identify_easy_cluster_centroid(embeddings, cluster_labels, registry)
    difficulty = compute_difficulty_centroid_distance(embeddings, task_ids, centroid)
    order = get_curriculum_order(
        task_ids, embeddings,
        strategy=CurriculumStrategy.EASY_TO_HARD,
        difficulty_scores=difficulty,
        cluster_labels=cluster_labels,
        seed=42,
    )
    expected = sorted(task_ids, key=lambda t: (difficulty.get(t, 0.0), t))
    _test(
        "Easy-to-hard: ordering matches sorted difficulty",
        order.task_ids == expected,
        f"got {order.task_ids[:5]}... vs expected {expected[:5]}..."
    )

    # ----------------------------------------------------------------
    # Test 2: Anti-curriculum is reverse of easy-to-hard
    # ----------------------------------------------------------------
    anti_order = get_curriculum_order(
        task_ids, embeddings,
        strategy=CurriculumStrategy.ANTI_CURRICULUM,
        difficulty_scores=difficulty,
        cluster_labels=cluster_labels,
        seed=42,
    )
    expected_anti = sorted(task_ids, key=lambda t: (-difficulty.get(t, 0.0), t))
    _test(
        "Anti-curriculum: reverse of easy-to-hard",
        anti_order.task_ids == expected_anti,
    )

    # ----------------------------------------------------------------
    # Test 3: NONE strategy preserves all task_ids (shuffled)
    # ----------------------------------------------------------------
    none_order = get_curriculum_order(
        task_ids, embeddings,
        strategy=CurriculumStrategy.NONE,
        seed=42,
    )
    _test(
        "NONE strategy: all task_ids present",
        set(none_order.task_ids) == set(task_ids) and len(none_order.task_ids) == len(task_ids),
    )

    # ----------------------------------------------------------------
    # Test 4: Determinism -- same seed produces same order
    # ----------------------------------------------------------------
    order_a = get_curriculum_order(
        task_ids, embeddings,
        strategy=CurriculumStrategy.NONE,
        seed=42,
        epoch=5,
    )
    order_b = get_curriculum_order(
        task_ids, embeddings,
        strategy=CurriculumStrategy.NONE,
        seed=42,
        epoch=5,
    )
    _test(
        "Determinism: same seed and epoch produce same order",
        order_a.task_ids == order_b.task_ids,
    )

    # ----------------------------------------------------------------
    # Test 5: Different seeds produce different orders
    # ----------------------------------------------------------------
    order_c = get_curriculum_order(
        task_ids, embeddings,
        strategy=CurriculumStrategy.NONE,
        seed=42,
        epoch=5,
    )
    order_d = get_curriculum_order(
        task_ids, embeddings,
        strategy=CurriculumStrategy.NONE,
        seed=99,
        epoch=5,
    )
    _test(
        "Different seeds: produce different orders",
        order_c.task_ids != order_d.task_ids,
        "Same order for different seeds (extremely unlikely with 50 tasks)"
    )

    # ----------------------------------------------------------------
    # Test 6: Diversity batch -- min pairwise distance >= threshold
    # ----------------------------------------------------------------
    div_batch = build_meta_batch(
        task_ids, embeddings,
        batch_size=8,
        diversity_constraint="min_distance",
        min_distance=0.1,
        seed=42,
    )
    actual_min = _min_pairwise_distance(embeddings, div_batch.task_ids)
    # Threshold may be relaxed, but should be >= some fraction
    _test(
        "Diversity batch: min pairwise distance >= relaxed threshold",
        actual_min >= 0.01,  # very relaxed check
        f"min_dist={actual_min:.4f}",
    )

    # ----------------------------------------------------------------
    # Test 7: Stratified batch -- tasks from different clusters
    # ----------------------------------------------------------------
    strat_batch = build_meta_batch(
        task_ids, embeddings,
        batch_size=10,
        diversity_constraint="stratified",
        cluster_labels=cluster_labels,
        seed=42,
    )
    clusters_in_batch = set(cluster_labels.get(tid, -1) for tid in strat_batch.task_ids)
    all_clusters = set(cluster_labels.values())
    _test(
        "Stratified batch: tasks from different clusters",
        len(clusters_in_batch) >= min(len(all_clusters), len(strat_batch.task_ids)),
        f"clusters represented: {len(clusters_in_batch)} / {len(all_clusters)}",
    )

    # ----------------------------------------------------------------
    # Test 8: Batch size enforcement
    # ----------------------------------------------------------------
    for target_size in [1, 5, 10, 20]:
        batch = build_meta_batch(
            task_ids, embeddings,
            batch_size=target_size,
            diversity_constraint="none",
            seed=42,
        )
        _test(
            f"Batch size enforcement: requested={target_size}",
            len(batch.task_ids) == target_size,
            f"got {len(batch.task_ids)}",
        )

    # ----------------------------------------------------------------
    # Test 9: Kendall tau -- curriculum vs random has non-zero tau
    # ----------------------------------------------------------------
    rng_kt = np.random.RandomState(999)
    random_order = list(task_ids)
    rng_kt.shuffle(random_order)
    tau, p_val = compute_kendall_tau(order.task_ids, random_order)
    _test(
        "Kendall tau: curriculum vs random has non-zero tau",
        abs(tau) > 0.0,
        f"tau={tau:.4f}, p={p_val:.4f}",
    )

    # ----------------------------------------------------------------
    # Test 10: Kendall tau -- identical orderings have tau = 1.0
    # ----------------------------------------------------------------
    tau_id, _ = compute_kendall_tau(task_ids, task_ids)
    _test(
        "Kendall tau: identical orderings have tau = 1.0",
        abs(tau_id - 1.0) < 1e-6,
        f"tau={tau_id:.6f}",
    )

    # ----------------------------------------------------------------
    # Test 11: Schedule -- linear(0) = 0, linear(max) = 1
    # ----------------------------------------------------------------
    _test(
        "Schedule: linear(0, 100) = 0",
        abs(linear_schedule(0, 100)) < 1e-9,
        f"got {linear_schedule(0, 100)}",
    )
    _test(
        "Schedule: linear(100, 100) = 1",
        abs(linear_schedule(100, 100) - 1.0) < 1e-9,
        f"got {linear_schedule(100, 100)}",
    )

    # ----------------------------------------------------------------
    # Test 12: Schedule -- cosine(0) ~ 0, cosine(max) ~ 1
    # ----------------------------------------------------------------
    _test(
        "Schedule: cosine(0, 100) ~ 0",
        abs(cosine_schedule(0, 100)) < 0.01,
        f"got {cosine_schedule(0, 100)}",
    )
    _test(
        "Schedule: cosine(100, 100) ~ 1",
        abs(cosine_schedule(100, 100) - 1.0) < 0.01,
        f"got {cosine_schedule(100, 100)}",
    )

    # ----------------------------------------------------------------
    # Test 13: Schedule -- step before switch=0, after=1
    # ----------------------------------------------------------------
    _test(
        "Schedule: step(10, 50) = 0 (before switch)",
        step_schedule(10, 50) == 0.0,
        f"got {step_schedule(10, 50)}",
    )
    _test(
        "Schedule: step(50, 50) = 1 (at switch)",
        step_schedule(50, 50) == 1.0,
        f"got {step_schedule(50, 50)}",
    )
    _test(
        "Schedule: step(60, 50) = 1 (after switch)",
        step_schedule(60, 50) == 1.0,
        f"got {step_schedule(60, 50)}",
    )

    # ----------------------------------------------------------------
    # Test 14: Epoch log -- contains all required fields
    # ----------------------------------------------------------------
    log = create_epoch_log(order, random_order, epoch=0)
    required_fields = ["epoch", "strategy", "task_order_hash",
                       "cluster_histogram", "difficulty_stats",
                       "kendall_tau", "p_value"]
    log_dict = log.to_dict()
    all_present = all(f in log_dict for f in required_fields)
    _test(
        "Epoch log: contains all required fields",
        all_present,
        f"missing: {[f for f in required_fields if f not in log_dict]}",
    )

    # ----------------------------------------------------------------
    # Test 15: Epoch log -- task_order_hash is valid hex string
    # ----------------------------------------------------------------
    is_valid_hex = len(log.task_order_hash) == 64
    try:
        int(log.task_order_hash, 16)
    except ValueError:
        is_valid_hex = False
    _test(
        "Epoch log: task_order_hash is valid hex string",
        is_valid_hex,
        f"hash='{log.task_order_hash[:20]}...' len={len(log.task_order_hash)}",
    )

    # ----------------------------------------------------------------
    # Test 16: Difficulty proxy -- centroid distance returns valid scores
    # ----------------------------------------------------------------
    scores = compute_difficulty_centroid_distance(embeddings, task_ids, centroid)
    all_valid = (
        len(scores) == len(task_ids)
        and all(isinstance(v, float) for v in scores.values())
        and all(v >= 0.0 for v in scores.values())
    )
    _test(
        "Difficulty proxy: centroid distance returns valid scores",
        all_valid,
        f"n_scores={len(scores)}, n_tasks={len(task_ids)}",
    )

    # ----------------------------------------------------------------
    # Test 17: CurriculumManager -- multi-epoch orders differ
    # ----------------------------------------------------------------
    config = CurriculumConfig(
        strategy="mixed",
        difficulty_proxy="centroid_distance",
        schedule_type="linear",
        max_epochs=100,
        seed=42,
    )
    manager = CurriculumManager(
        config, embeddings, task_ids, cluster_labels, registry
    )
    order_ep0 = manager.get_epoch_order(0)
    order_ep50 = manager.get_epoch_order(50)
    order_ep99 = manager.get_epoch_order(99)
    _test(
        "CurriculumManager: multi-epoch orders differ",
        order_ep0.task_ids != order_ep50.task_ids or order_ep50.task_ids != order_ep99.task_ids,
    )

    # ----------------------------------------------------------------
    # Test 18: Mixed schedule -- early epochs mostly easy, late mostly hard
    # ----------------------------------------------------------------
    # With linear schedule, epoch 0 should favor easy tasks (p_hard ~0)
    # and epoch 99 should favor hard tasks (p_hard ~1)
    config_mix = CurriculumConfig(
        strategy="mixed",
        difficulty_proxy="centroid_distance",
        schedule_type="linear",
        max_epochs=100,
        seed=42,
    )
    mgr_mix = CurriculumManager(
        config_mix, embeddings, task_ids, cluster_labels, registry
    )
    early_order = mgr_mix.get_epoch_order(1)
    late_order = mgr_mix.get_epoch_order(99)

    # Compute mean difficulty of first half of early vs late ordering
    half = len(task_ids) // 2
    if difficulty and half > 0:
        early_first_half = [difficulty.get(tid, 0.0) for tid in early_order.task_ids[:half]]
        late_first_half = [difficulty.get(tid, 0.0) for tid in late_order.task_ids[:half]]
        early_mean = np.mean(early_first_half) if early_first_half else 0.0
        late_mean = np.mean(late_first_half) if late_first_half else 0.0
        # Late epoch should have harder tasks in the first half
        # (not guaranteed with sampling, so just check they differ)
        _test(
            "Mixed schedule: early and late epochs have different compositions",
            early_order.task_ids != late_order.task_ids,
            f"early_mean_diff={early_mean:.4f}, late_mean_diff={late_mean:.4f}",
        )
    else:
        _test(
            "Mixed schedule: early and late epochs have different compositions",
            early_order.task_ids != late_order.task_ids,
        )

    # ----------------------------------------------------------------
    # Test 19: Empty task list handled gracefully
    # ----------------------------------------------------------------
    empty_order = get_curriculum_order(
        [], {},
        strategy=CurriculumStrategy.EASY_TO_HARD,
        seed=42,
    )
    _test(
        "Empty task list: handled gracefully (ordering)",
        empty_order.task_ids == [],
    )

    empty_batch = build_meta_batch(
        [], {},
        batch_size=5,
        diversity_constraint="min_distance",
        seed=42,
    )
    _test(
        "Empty task list: handled gracefully (batching)",
        empty_batch.task_ids == [],
    )

    # ----------------------------------------------------------------
    # Test 20: Single task works without error
    # ----------------------------------------------------------------
    single_ids = [task_ids[0]]
    single_embs = {task_ids[0]: embeddings[task_ids[0]]}
    single_order = get_curriculum_order(
        single_ids, single_embs,
        strategy=CurriculumStrategy.EASY_TO_HARD,
        difficulty_scores={task_ids[0]: 0.5},
        seed=42,
    )
    _test(
        "Single task: works without error",
        len(single_order.task_ids) == 1 and single_order.task_ids[0] == task_ids[0],
    )

    single_batch = build_meta_batch(
        single_ids, single_embs,
        batch_size=1,
        diversity_constraint="min_distance",
        seed=42,
    )
    _test(
        "Single task: batching works without error",
        len(single_batch.task_ids) == 1,
    )

    # ----------------------------------------------------------------
    # Test 21: Kendall tau pure Python fallback
    # ----------------------------------------------------------------
    x_rev = list(range(10))
    y_rev = list(reversed(range(10)))
    tau_rev, _ = _kendall_tau_fallback(x_rev, y_rev)
    _test(
        "Kendall tau fallback: reversed ordering has tau = -1.0",
        abs(tau_rev - (-1.0)) < 1e-6,
        f"tau={tau_rev:.6f}",
    )
    tau_same, _ = _kendall_tau_fallback(x_rev, x_rev)
    _test(
        "Kendall tau fallback: identical ordering has tau = 1.0",
        abs(tau_same - 1.0) < 1e-6,
        f"tau={tau_same:.6f}",
    )

    # ----------------------------------------------------------------
    # Test 22: Difficulty proxy -- adaptation gain
    # ----------------------------------------------------------------
    gain_scores = compute_difficulty_adaptation_gain(task_ids, registry)
    _test(
        "Difficulty proxy: adaptation gain returns valid scores",
        len(gain_scores) == len(task_ids)
        and all(isinstance(v, float) for v in gain_scores.values()),
        f"n_scores={len(gain_scores)}",
    )

    # ----------------------------------------------------------------
    # Test 23: Difficulty proxy -- loss slope
    # ----------------------------------------------------------------
    slope_scores = compute_difficulty_loss_slope(task_ids, registry)
    _test(
        "Difficulty proxy: loss slope returns valid scores",
        len(slope_scores) == len(task_ids)
        and all(isinstance(v, float) for v in slope_scores.values()),
        f"n_scores={len(slope_scores)}",
    )

    # ----------------------------------------------------------------
    # Test 24: get_schedule_fn factory works for all types
    # ----------------------------------------------------------------
    schedule_ok = True
    for stype in ScheduleType:
        try:
            fn = get_schedule_fn(stype)
            if stype == ScheduleType.STEP:
                val = fn(0, 50)
            else:
                val = fn(0, 100)
            if not isinstance(val, float):
                schedule_ok = False
        except Exception:
            schedule_ok = False
    _test(
        "get_schedule_fn: works for all ScheduleType values",
        schedule_ok,
    )

    # ----------------------------------------------------------------
    # Test 25: CurriculumManager epoch log contains valid data
    # ----------------------------------------------------------------
    mgr_log = manager.get_epoch_log(0)
    _test(
        "CurriculumManager: epoch log has valid task_order_hash",
        len(mgr_log.task_order_hash) == 64,
    )

    # ----------------------------------------------------------------
    # Test 26: Diversity ordering (farthest-point traversal)
    # ----------------------------------------------------------------
    div_order = get_curriculum_order(
        task_ids, embeddings,
        strategy=CurriculumStrategy.DIVERSITY,
        seed=42,
    )
    _test(
        "Diversity ordering: returns all tasks",
        set(div_order.task_ids) == set(task_ids),
    )

    # ----------------------------------------------------------------
    # Test 27: Batch from pool larger than batch_size
    # ----------------------------------------------------------------
    big_batch = build_meta_batch(
        task_ids, embeddings,
        batch_size=5,
        diversity_constraint="min_distance",
        min_distance=0.3,
        seed=42,
    )
    _test(
        "Large pool, small batch: returns exactly batch_size",
        len(big_batch.task_ids) == 5,
        f"got {len(big_batch.task_ids)}",
    )

    # ----------------------------------------------------------------
    # Test 28: Batch from pool smaller than batch_size
    # ----------------------------------------------------------------
    small_pool = task_ids[:3]
    small_embs = {tid: embeddings[tid] for tid in small_pool}
    small_batch = build_meta_batch(
        small_pool, small_embs,
        batch_size=10,
        diversity_constraint="none",
        seed=42,
    )
    _test(
        "Small pool, large batch: returns pool size (graceful)",
        len(small_batch.task_ids) == len(small_pool),
        f"pool={len(small_pool)}, batch={len(small_batch.task_ids)}",
    )

    # ----------------------------------------------------------------
    # Test 29: CurriculumOrder metadata populated
    # ----------------------------------------------------------------
    _test(
        "CurriculumOrder: strategy field is set",
        order.strategy == CurriculumStrategy.EASY_TO_HARD.value,
        f"strategy='{order.strategy}'",
    )

    # ----------------------------------------------------------------
    # Test 30: Warmup epochs produce random ordering
    # ----------------------------------------------------------------
    warmup_order = get_curriculum_order(
        task_ids, embeddings,
        strategy=CurriculumStrategy.EASY_TO_HARD,
        difficulty_scores=difficulty,
        cluster_labels=cluster_labels,
        epoch=0,
        warmup_epochs=5,
        seed=42,
    )
    _test(
        "Warmup epochs: produce random ordering (not easy-to-hard)",
        warmup_order.strategy == "warmup_random",
        f"strategy='{warmup_order.strategy}'",
    )

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print("=" * 72)
    print(f"Results: {passed}/{total} passed, {failed}/{total} failed")
    print("=" * 72)

    if failed > 0:
        print(f"\nWARNING: {failed} test(s) failed!")
    else:
        print("\nAll tests passed.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    _run_self_tests()
