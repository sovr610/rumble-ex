"""
Task2Vec + Curriculum Configuration Dataclasses.

Complete, self-contained configuration for the Task2Vec embedding pipeline,
task clustering, curriculum ordering, and embedding registry. Designed to
integrate with the main BrainAIConfig system as an extension for Phase 7
meta-learning.

Configuration Hierarchy:
========================
  Task2VecFullConfig (aggregate)
    |-- Task2VecConfig      (probe + Fisher embedding settings)
    |-- ClusterConfig        (clustering algorithm settings)
    |-- CurriculumConfig     (curriculum ordering settings)
    |-- RegistryConfig       (embedding registry settings)

Presets:
  - Task2VecFullConfig.minimal()     ~128-dim, 4 clusters, no curriculum (tests)
  - Task2VecFullConfig.dev()         ~256-dim, 6 clusters, easy-to-hard (dev)
  - Task2VecFullConfig.production()  ~512-dim, 8 clusters, diversity (production)

Usage:
  from task2vec_config_template import Task2VecFullConfig
  cfg = Task2VecFullConfig.production()
  print(cfg.task2vec.embedding_dim)   # 512
  print(cfg.curriculum.strategy)      # "diversity"

  # Serialize / deserialize
  d = cfg.to_dict()
  cfg2 = Task2VecFullConfig.from_dict(d)
  assert cfg == cfg2
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field, fields, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants: valid choice sets
# ---------------------------------------------------------------------------

VALID_PROBE_MODELS: Tuple[str, ...] = ("conv4", "resnet12", "vit_tiny")
VALID_LAYER_SUBSETS: Tuple[str, ...] = ("last_block", "per_stage", "all")
VALID_AGGREGATIONS: Tuple[str, ...] = ("per_channel", "per_head", "per_layer")
VALID_NORMALIZATIONS: Tuple[str, ...] = ("log1p_l2", "l2", "whiten")
VALID_CLUSTER_ALGORITHMS: Tuple[str, ...] = ("kmeans", "agglomerative", "hdbscan")
VALID_DISTANCE_METRICS: Tuple[str, ...] = ("cosine", "euclidean", "manhattan")
VALID_LINKAGE_METHODS: Tuple[str, ...] = ("ward", "complete", "average", "single")
VALID_CURRICULUM_STRATEGIES: Tuple[str, ...] = (
    "none",
    "easy_to_hard",
    "diversity",
    "anti_curriculum",
    "mixed",
)
VALID_DIFFICULTY_PROXIES: Tuple[str, ...] = (
    "centroid_distance",
    "adaptation_gain",
    "loss_slope",
)
VALID_DIVERSITY_METHODS: Tuple[str, ...] = ("min_distance", "stratified")
VALID_SCHEDULE_FNS: Tuple[str, ...] = ("linear", "cosine", "step")
VALID_REGISTRY_FORMATS: Tuple[str, ...] = ("jsonl_npz", "parquet")


# ---------------------------------------------------------------------------
# Helper: generic choice validation
# ---------------------------------------------------------------------------

def _validate_choice(
    value: str,
    valid_choices: Tuple[str, ...],
    field_name: str,
    config_name: str,
) -> None:
    """Validate that *value* is one of *valid_choices*.

    Raises:
        ValueError: If *value* is not in the allowed set.
    """
    if value not in valid_choices:
        raise ValueError(
            f"{config_name}.{field_name} must be one of {valid_choices}, "
            f"got {value!r}"
        )


def _validate_positive(
    value: Union[int, float],
    field_name: str,
    config_name: str,
    *,
    allow_zero: bool = False,
) -> None:
    """Validate that *value* is a positive number (or non-negative if allow_zero).

    Raises:
        ValueError: If the constraint is violated.
    """
    if allow_zero:
        if value < 0:
            raise ValueError(
                f"{config_name}.{field_name} must be >= 0, got {value}"
            )
    else:
        if value <= 0:
            raise ValueError(
                f"{config_name}.{field_name} must be > 0, got {value}"
            )


def _validate_range(
    value: Union[int, float],
    low: Union[int, float],
    high: Union[int, float],
    field_name: str,
    config_name: str,
    *,
    inclusive_low: bool = True,
    inclusive_high: bool = True,
) -> None:
    """Validate that *value* is within [low, high] (bounds optionally exclusive).

    Raises:
        ValueError: If out of range.
    """
    below = (value < low) if inclusive_low else (value <= low)
    above = (value > high) if inclusive_high else (value >= high)
    lb = "[" if inclusive_low else "("
    rb = "]" if inclusive_high else ")"
    if below or above:
        raise ValueError(
            f"{config_name}.{field_name} must be in {lb}{low}, {high}{rb}, "
            f"got {value}"
        )


# ---------------------------------------------------------------------------
# TypeVar for from_dict classmethods
# ---------------------------------------------------------------------------

T = TypeVar("T")


# =========================================================================
# 1. Task2VecConfig -- probe network + Fisher embedding settings
# =========================================================================

@dataclass
class Task2VecConfig:
    """Configuration for the Task2Vec probe and Fisher embedding pipeline.

    The probe network is a fixed, pretrained reference model used to compute
    diagonal Fisher information for each task. The resulting embedding is a
    fixed-dimensional vector that captures task structure independently of
    the meta-learner being trained.

    Attributes:
        probe_model: Backbone architecture for the probe network.
            - ``"conv4"``: 4-layer ConvNet, lightweight, fast extraction.
            - ``"resnet12"``: 12-layer ResNet, richer features, moderate cost.
            - ``"vit_tiny"``: Vision Transformer (tiny), attention-based features.
        layer_subset: Which layers of the probe to use for Fisher computation.
            - ``"last_block"``: Only the final convolutional/transformer block.
            - ``"per_stage"``: One representative layer per stage/block.
            - ``"all"``: All parameter layers (highest-dimensional, slowest).
        embedding_dim: Target dimensionality of the output embedding vector.
            Must be positive. Typical values: 128 (minimal), 256 (dev), 512 (prod).
        aggregation: How to aggregate per-parameter Fisher values within layers.
            - ``"per_channel"``: Aggregate per output channel (default, balanced).
            - ``"per_head"``: Aggregate per attention head (ViT probes).
            - ``"per_layer"``: Single scalar per layer (most compressed).
        normalize: Normalization applied after Fisher aggregation.
            - ``"log1p_l2"``: log(1 + F) then L2 normalize (default, scale-invariant).
            - ``"l2"``: Raw L2 normalization.
            - ``"whiten"``: ZCA whitening using a reference covariance matrix.
        num_fisher_samples: Number of support samples used for Fisher estimation.
            ``None`` means use all available support samples in the episode.
        use_fp32: Whether to force fp32 precision during Fisher computation.
            Recommended ``True`` for determinism; ``False`` may speed up on GPU.
        cache_probe: Whether to cache the probe model in memory between extractions
            rather than reloading each time.
    """

    probe_model: str = "conv4"
    layer_subset: str = "last_block"
    embedding_dim: int = 512
    aggregation: str = "per_channel"
    normalize: str = "log1p_l2"
    num_fisher_samples: Optional[int] = None
    use_fp32: bool = True
    cache_probe: bool = True

    def __post_init__(self) -> None:
        """Validate all fields on construction."""
        _validate_choice(
            self.probe_model, VALID_PROBE_MODELS,
            "probe_model", "Task2VecConfig",
        )
        _validate_choice(
            self.layer_subset, VALID_LAYER_SUBSETS,
            "layer_subset", "Task2VecConfig",
        )
        _validate_positive(
            self.embedding_dim, "embedding_dim", "Task2VecConfig",
        )
        _validate_choice(
            self.aggregation, VALID_AGGREGATIONS,
            "aggregation", "Task2VecConfig",
        )
        _validate_choice(
            self.normalize, VALID_NORMALIZATIONS,
            "normalize", "Task2VecConfig",
        )
        if self.num_fisher_samples is not None:
            _validate_positive(
                self.num_fisher_samples,
                "num_fisher_samples",
                "Task2VecConfig",
            )

    # -- Serialization -----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task2VecConfig":
        """Deserialize from a plain dictionary.

        Unknown keys are silently ignored so that forward-compatible configs
        can be loaded by older code.
        """
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


# =========================================================================
# 2. ClusterConfig -- clustering algorithm settings
# =========================================================================

@dataclass
class ClusterConfig:
    """Configuration for task embedding clustering.

    Clustering groups tasks by similarity in the embedding space, enabling
    stratified meta-batch composition and curriculum design. Stability is
    measured via adjusted Rand index (ARI) across repeated runs.

    Attributes:
        algorithm: Clustering algorithm.
            - ``"kmeans"``: K-means (seeded, deterministic).
            - ``"agglomerative"``: Agglomerative hierarchical clustering.
            - ``"hdbscan"``: Density-based clustering (ignores ``n_clusters``).
        n_clusters: Number of clusters (ignored by HDBSCAN). Must be > 0.
        distance_metric: Distance metric for computing task similarity.
            - ``"cosine"``: 1 - cos(u, v). Default and recommended.
            - ``"euclidean"``: L2 distance.
            - ``"manhattan"``: L1 distance.
        linkage: Linkage criterion for agglomerative clustering.
            - ``"ward"``: Minimize within-cluster variance.
            - ``"complete"``: Maximum pairwise distance.
            - ``"average"``: Mean pairwise distance.
            - ``"single"``: Minimum pairwise distance.
        min_cluster_size: Minimum cluster size for HDBSCAN.
        min_samples: Minimum samples for HDBSCAN core points.
        stability_threshold: Minimum ARI between repeated clustering runs
            to consider the clustering stable. Range [0, 1].
        stability_n_runs: Number of repeated clustering runs for stability
            estimation.
    """

    algorithm: str = "kmeans"
    n_clusters: int = 8
    distance_metric: str = "cosine"
    linkage: str = "ward"
    min_cluster_size: int = 5
    min_samples: int = 3
    stability_threshold: float = 0.9
    stability_n_runs: int = 5

    def __post_init__(self) -> None:
        """Validate all fields on construction."""
        _validate_choice(
            self.algorithm, VALID_CLUSTER_ALGORITHMS,
            "algorithm", "ClusterConfig",
        )
        _validate_positive(
            self.n_clusters, "n_clusters", "ClusterConfig",
        )
        _validate_choice(
            self.distance_metric, VALID_DISTANCE_METRICS,
            "distance_metric", "ClusterConfig",
        )
        _validate_choice(
            self.linkage, VALID_LINKAGE_METHODS,
            "linkage", "ClusterConfig",
        )
        _validate_positive(
            self.min_cluster_size, "min_cluster_size", "ClusterConfig",
        )
        _validate_positive(
            self.min_samples, "min_samples", "ClusterConfig",
        )
        _validate_range(
            self.stability_threshold, 0.0, 1.0,
            "stability_threshold", "ClusterConfig",
        )
        _validate_positive(
            self.stability_n_runs, "stability_n_runs", "ClusterConfig",
        )

    # -- Serialization -----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClusterConfig":
        """Deserialize from a plain dictionary."""
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


# =========================================================================
# 3. CurriculumConfig -- curriculum ordering settings
# =========================================================================

@dataclass
class CurriculumConfig:
    """Configuration for curriculum-based task ordering.

    Controls how tasks are ordered during meta-training epochs. Strategies
    range from no ordering (random) to difficulty-based curricula and
    diversity-constrained meta-batch composition.

    Attributes:
        strategy: Curriculum strategy for task ordering.
            - ``"none"``: Random ordering (baseline).
            - ``"easy_to_hard"``: Order tasks by increasing difficulty.
            - ``"diversity"``: Maximize pairwise diversity within meta-batches.
            - ``"anti_curriculum"``: Hard-first, or probability-mixed schedule.
            - ``"mixed"``: Blend strategies according to ``schedule_fn``.
        difficulty_proxy: How to estimate task difficulty.
            - ``"centroid_distance"``: Distance from easy-cluster centroid.
            - ``"adaptation_gain"``: Meta-learner loss improvement on the task.
            - ``"loss_slope"``: Slope of inner-loop loss over adaptation steps.
        diversity_min_distance: Minimum pairwise cosine distance enforced when
            building diversity-constrained meta-batches. Range [0.0, 2.0]
            (cosine distance is in [0, 2]).
        diversity_method: Method for diversity-constrained sampling.
            - ``"min_distance"``: Greedy selection maximizing minimum distance.
            - ``"stratified"``: Stratified sampling from each cluster.
        schedule_fn: Schedule function for ``"mixed"`` strategy blending.
            - ``"linear"``: Linearly interpolate from one strategy to another.
            - ``"cosine"``: Cosine annealing between strategies.
            - ``"step"``: Hard switch at ``schedule_switch_epoch``.
        schedule_switch_epoch: Epoch at which to switch strategies in ``"step"``
            schedule. Must be >= 0.
        warmup_epochs: Number of initial epochs with random ordering before
            the curriculum takes effect. Must be >= 0.
        log_task_order: Whether to log the ordered list of task identifiers
            at each epoch.
        log_full_ids: Whether to log full task IDs or only their truncated
            hashes. Defaults to ``False`` (hash only) to keep logs compact.
    """

    strategy: str = "none"
    difficulty_proxy: str = "centroid_distance"
    diversity_min_distance: float = 0.3
    diversity_method: str = "min_distance"
    schedule_fn: str = "linear"
    schedule_switch_epoch: int = 50
    warmup_epochs: int = 0
    log_task_order: bool = True
    log_full_ids: bool = False

    def __post_init__(self) -> None:
        """Validate all fields on construction."""
        _validate_choice(
            self.strategy, VALID_CURRICULUM_STRATEGIES,
            "strategy", "CurriculumConfig",
        )
        _validate_choice(
            self.difficulty_proxy, VALID_DIFFICULTY_PROXIES,
            "difficulty_proxy", "CurriculumConfig",
        )
        _validate_range(
            self.diversity_min_distance, 0.0, 2.0,
            "diversity_min_distance", "CurriculumConfig",
        )
        _validate_choice(
            self.diversity_method, VALID_DIVERSITY_METHODS,
            "diversity_method", "CurriculumConfig",
        )
        _validate_choice(
            self.schedule_fn, VALID_SCHEDULE_FNS,
            "schedule_fn", "CurriculumConfig",
        )
        _validate_positive(
            self.schedule_switch_epoch,
            "schedule_switch_epoch",
            "CurriculumConfig",
            allow_zero=True,
        )
        _validate_positive(
            self.warmup_epochs,
            "warmup_epochs",
            "CurriculumConfig",
            allow_zero=True,
        )

    # -- Serialization -----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CurriculumConfig":
        """Deserialize from a plain dictionary."""
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


# =========================================================================
# 4. RegistryConfig -- embedding registry settings
# =========================================================================

@dataclass
class RegistryConfig:
    """Configuration for the task embedding registry.

    The registry is a persistent artifact that maps task IDs to their
    embeddings and metadata. It is stored alongside training checkpoints
    and can be used for offline analysis, curriculum planning, and
    reproducibility.

    Attributes:
        format: Storage format for the registry.
            - ``"jsonl_npz"``: JSONL for metadata + NPZ for embeddings (default).
              Compact, human-readable metadata, efficient binary embeddings.
            - ``"parquet"``: Single Parquet file with embedded arrays.
              Good for large-scale analytics with pandas/polars.
        save_with_checkpoint: Whether to save the registry automatically
            whenever a training checkpoint is written.
        max_entries: Maximum number of entries in the registry. ``None`` means
            unlimited. When exceeded, oldest entries are evicted (FIFO).
        auto_extract: Whether to automatically extract and register embeddings
            for new episodes encountered during training. If ``False``,
            extraction must be triggered explicitly.
        dedup_on_task_id: Whether to deduplicate entries by task_id. If
            ``True``, re-extracting an embedding for an existing task_id
            will overwrite the previous entry. If ``False``, all extractions
            are appended.
        compression: Whether to compress the NPZ file (gzip). Reduces disk
            usage at the cost of slightly slower I/O.
        version: Registry schema version string for forward compatibility.
    """

    format: str = "jsonl_npz"
    save_with_checkpoint: bool = True
    max_entries: Optional[int] = None
    auto_extract: bool = False
    dedup_on_task_id: bool = True
    compression: bool = False
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        """Validate all fields on construction."""
        _validate_choice(
            self.format, VALID_REGISTRY_FORMATS,
            "format", "RegistryConfig",
        )
        if self.max_entries is not None:
            _validate_positive(
                self.max_entries, "max_entries", "RegistryConfig",
            )

    # -- Serialization -----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegistryConfig":
        """Deserialize from a plain dictionary."""
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


# =========================================================================
# 5. Task2VecFullConfig -- aggregate configuration
# =========================================================================

@dataclass
class Task2VecFullConfig:
    """Aggregate configuration for the entire Task2Vec pipeline.

    Combines probe/Fisher, clustering, curriculum, and registry configs
    into a single top-level object with a master feature flag.

    Attributes:
        task2vec: Probe and Fisher embedding configuration.
        cluster: Clustering algorithm configuration.
        curriculum: Curriculum ordering configuration.
        registry: Embedding registry configuration.
        use_task2vec: Master feature flag. When ``False``, the entire
            Task2Vec pipeline is disabled and training uses random
            task ordering.
        seed: Global seed for reproducibility across extraction,
            clustering, and curriculum ordering.
        log_level: Logging level for Task2Vec-related messages.
            One of ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``.
    """

    task2vec: Task2VecConfig = field(default_factory=Task2VecConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    registry: RegistryConfig = field(default_factory=RegistryConfig)
    use_task2vec: bool = True
    seed: int = 42
    log_level: str = "INFO"

    # -- Valid log levels --------------------------------------------------

    _VALID_LOG_LEVELS: Tuple[str, ...] = (
        "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL",
    )

    def __post_init__(self) -> None:
        """Validate aggregate-level fields and propagate sub-config validation.

        Sub-config __post_init__ methods are called automatically by their
        own dataclass constructors, so we only need to validate fields that
        belong to this level.
        """
        if self.log_level.upper() not in self._VALID_LOG_LEVELS:
            raise ValueError(
                f"Task2VecFullConfig.log_level must be one of "
                f"{self._VALID_LOG_LEVELS}, got {self.log_level!r}"
            )
        # Normalize log_level to uppercase
        self.log_level = self.log_level.upper()

        if not isinstance(self.seed, int):
            raise TypeError(
                f"Task2VecFullConfig.seed must be int, got {type(self.seed).__name__}"
            )

    # -- Presets -----------------------------------------------------------

    @classmethod
    def minimal(cls) -> "Task2VecFullConfig":
        """Minimal configuration for unit tests.

        Small embedding dimensionality (128), few clusters (4), no
        curriculum ordering. Fast extraction with conv4 probe.

        Returns:
            A Task2VecFullConfig suitable for unit testing.
        """
        return cls(
            task2vec=Task2VecConfig(
                probe_model="conv4",
                layer_subset="last_block",
                embedding_dim=128,
                aggregation="per_layer",
                normalize="l2",
                num_fisher_samples=16,
                use_fp32=True,
                cache_probe=False,
            ),
            cluster=ClusterConfig(
                algorithm="kmeans",
                n_clusters=4,
                distance_metric="cosine",
                stability_threshold=0.8,
                stability_n_runs=3,
                min_cluster_size=3,
                min_samples=2,
            ),
            curriculum=CurriculumConfig(
                strategy="none",
                log_task_order=False,
                log_full_ids=False,
            ),
            registry=RegistryConfig(
                format="jsonl_npz",
                save_with_checkpoint=False,
                max_entries=1000,
                auto_extract=False,
            ),
            use_task2vec=True,
            seed=0,
            log_level="WARNING",
        )

    @classmethod
    def dev(cls) -> "Task2VecFullConfig":
        """Development configuration for iteration and debugging.

        Mid-range embedding dimensionality (256), moderate clusters (6),
        easy-to-hard curriculum for observing ordering effects.

        Returns:
            A Task2VecFullConfig suitable for development workflows.
        """
        return cls(
            task2vec=Task2VecConfig(
                probe_model="conv4",
                layer_subset="per_stage",
                embedding_dim=256,
                aggregation="per_channel",
                normalize="log1p_l2",
                num_fisher_samples=64,
                use_fp32=True,
                cache_probe=True,
            ),
            cluster=ClusterConfig(
                algorithm="kmeans",
                n_clusters=6,
                distance_metric="cosine",
                stability_threshold=0.85,
                stability_n_runs=5,
                min_cluster_size=4,
                min_samples=3,
            ),
            curriculum=CurriculumConfig(
                strategy="easy_to_hard",
                difficulty_proxy="centroid_distance",
                diversity_min_distance=0.3,
                schedule_fn="linear",
                warmup_epochs=5,
                log_task_order=True,
                log_full_ids=False,
            ),
            registry=RegistryConfig(
                format="jsonl_npz",
                save_with_checkpoint=True,
                max_entries=50000,
                auto_extract=True,
            ),
            use_task2vec=True,
            seed=42,
            log_level="INFO",
        )

    @classmethod
    def production(cls) -> "Task2VecFullConfig":
        """Production configuration for full-scale training.

        Full embedding dimensionality (512), 8 clusters, diversity
        curriculum for meta-batch composition, full logging.

        Returns:
            A Task2VecFullConfig suitable for production training.
        """
        return cls(
            task2vec=Task2VecConfig(
                probe_model="resnet12",
                layer_subset="per_stage",
                embedding_dim=512,
                aggregation="per_channel",
                normalize="log1p_l2",
                num_fisher_samples=None,
                use_fp32=True,
                cache_probe=True,
            ),
            cluster=ClusterConfig(
                algorithm="kmeans",
                n_clusters=8,
                distance_metric="cosine",
                linkage="ward",
                min_cluster_size=5,
                min_samples=3,
                stability_threshold=0.9,
                stability_n_runs=5,
            ),
            curriculum=CurriculumConfig(
                strategy="diversity",
                difficulty_proxy="centroid_distance",
                diversity_min_distance=0.3,
                diversity_method="min_distance",
                schedule_fn="cosine",
                schedule_switch_epoch=50,
                warmup_epochs=10,
                log_task_order=True,
                log_full_ids=True,
            ),
            registry=RegistryConfig(
                format="jsonl_npz",
                save_with_checkpoint=True,
                max_entries=None,
                auto_extract=True,
                dedup_on_task_id=True,
                compression=True,
            ),
            use_task2vec=True,
            seed=42,
            log_level="INFO",
        )

    # -- Serialization -----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full config to a nested dictionary.

        Returns:
            A dictionary with sub-config keys and a flat ``use_task2vec`` flag.
        """
        return {
            "task2vec": self.task2vec.to_dict(),
            "cluster": self.cluster.to_dict(),
            "curriculum": self.curriculum.to_dict(),
            "registry": self.registry.to_dict(),
            "use_task2vec": self.use_task2vec,
            "seed": self.seed,
            "log_level": self.log_level,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task2VecFullConfig":
        """Deserialize from a nested dictionary.

        Sub-configs are reconstructed from their respective sub-dicts.
        Unknown top-level keys are silently ignored.

        Args:
            data: Dictionary produced by :meth:`to_dict` or equivalent.

        Returns:
            A fully validated Task2VecFullConfig instance.
        """
        task2vec = Task2VecConfig.from_dict(data.get("task2vec", {}))
        cluster = ClusterConfig.from_dict(data.get("cluster", {}))
        curriculum = CurriculumConfig.from_dict(data.get("curriculum", {}))
        registry = RegistryConfig.from_dict(data.get("registry", {}))
        return cls(
            task2vec=task2vec,
            cluster=cluster,
            curriculum=curriculum,
            registry=registry,
            use_task2vec=data.get("use_task2vec", True),
            seed=data.get("seed", 42),
            log_level=data.get("log_level", "INFO"),
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialize the full config to a JSON string.

        Args:
            indent: Number of spaces for indentation.

        Returns:
            A JSON string representation.
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "Task2VecFullConfig":
        """Deserialize from a JSON string.

        Args:
            json_str: JSON string produced by :meth:`to_json` or equivalent.

        Returns:
            A fully validated Task2VecFullConfig instance.
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    # -- Utility methods ---------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of the configuration.

        Returns:
            Multi-line string summarizing key settings.
        """
        lines = [
            "Task2Vec Pipeline Configuration",
            "=" * 40,
            f"  Feature flag:       {'ENABLED' if self.use_task2vec else 'DISABLED'}",
            f"  Seed:               {self.seed}",
            f"  Log level:          {self.log_level}",
            "",
            "  [Probe / Embedding]",
            f"    Probe model:      {self.task2vec.probe_model}",
            f"    Layer subset:     {self.task2vec.layer_subset}",
            f"    Embedding dim:    {self.task2vec.embedding_dim}",
            f"    Aggregation:      {self.task2vec.aggregation}",
            f"    Normalize:        {self.task2vec.normalize}",
            f"    Fisher samples:   {self.task2vec.num_fisher_samples or 'all'}",
            f"    FP32:             {self.task2vec.use_fp32}",
            f"    Cache probe:      {self.task2vec.cache_probe}",
            "",
            "  [Clustering]",
            f"    Algorithm:        {self.cluster.algorithm}",
            f"    n_clusters:       {self.cluster.n_clusters}",
            f"    Distance metric:  {self.cluster.distance_metric}",
            f"    Linkage:          {self.cluster.linkage}",
            f"    Stability thresh: {self.cluster.stability_threshold}",
            f"    Stability runs:   {self.cluster.stability_n_runs}",
            "",
            "  [Curriculum]",
            f"    Strategy:         {self.curriculum.strategy}",
            f"    Difficulty proxy: {self.curriculum.difficulty_proxy}",
            f"    Diversity dist:   {self.curriculum.diversity_min_distance}",
            f"    Diversity method: {self.curriculum.diversity_method}",
            f"    Schedule fn:      {self.curriculum.schedule_fn}",
            f"    Switch epoch:     {self.curriculum.schedule_switch_epoch}",
            f"    Warmup epochs:    {self.curriculum.warmup_epochs}",
            f"    Log task order:   {self.curriculum.log_task_order}",
            f"    Log full IDs:     {self.curriculum.log_full_ids}",
            "",
            "  [Registry]",
            f"    Format:           {self.registry.format}",
            f"    Save w/ ckpt:     {self.registry.save_with_checkpoint}",
            f"    Max entries:      {self.registry.max_entries or 'unlimited'}",
            f"    Auto extract:     {self.registry.auto_extract}",
            f"    Dedup on ID:      {self.registry.dedup_on_task_id}",
            f"    Compression:      {self.registry.compression}",
            f"    Version:          {self.registry.version}",
        ]
        return "\n".join(lines)

    def validate(self) -> List[str]:
        """Run cross-field validation across sub-configs.

        Returns warnings (not errors) for potentially problematic
        combinations that are technically valid but may cause issues.

        Returns:
            List of warning messages (empty if no issues detected).
        """
        warnings: List[str] = []

        # Warn if diversity curriculum but diversity_min_distance is very low
        if (
            self.curriculum.strategy == "diversity"
            and self.curriculum.diversity_min_distance < 0.1
        ):
            warnings.append(
                "Diversity curriculum with diversity_min_distance < 0.1 may "
                "not provide meaningful diversity constraints."
            )

        # Warn if HDBSCAN but n_clusters is set (it will be ignored)
        if self.cluster.algorithm == "hdbscan" and self.cluster.n_clusters != 8:
            warnings.append(
                f"ClusterConfig.n_clusters={self.cluster.n_clusters} is set "
                f"but will be ignored by HDBSCAN. HDBSCAN determines cluster "
                f"count automatically."
            )

        # Warn if whiten normalization with small embedding dim
        if (
            self.task2vec.normalize == "whiten"
            and self.task2vec.embedding_dim < 64
        ):
            warnings.append(
                "Whitening with embedding_dim < 64 may produce singular "
                "covariance matrices. Consider using 'log1p_l2' instead."
            )

        # Warn if per_head aggregation with non-ViT probe
        if (
            self.task2vec.aggregation == "per_head"
            and self.task2vec.probe_model != "vit_tiny"
        ):
            warnings.append(
                f"per_head aggregation with probe_model='{self.task2vec.probe_model}' "
                f"may not be meaningful. per_head is designed for ViT probes."
            )

        # Warn if auto_extract is on but use_task2vec is off
        if self.registry.auto_extract and not self.use_task2vec:
            warnings.append(
                "registry.auto_extract is True but use_task2vec is False. "
                "Extraction will not run unless use_task2vec is enabled."
            )

        # Warn if no curriculum but warmup_epochs > 0
        if (
            self.curriculum.strategy == "none"
            and self.curriculum.warmup_epochs > 0
        ):
            warnings.append(
                "curriculum.warmup_epochs > 0 has no effect when strategy "
                "is 'none' (already random ordering)."
            )

        # Warn if step schedule but switch epoch is 0
        if (
            self.curriculum.schedule_fn == "step"
            and self.curriculum.schedule_switch_epoch == 0
        ):
            warnings.append(
                "Step schedule with switch_epoch=0 means the second strategy "
                "is used from the start. This is equivalent to using only "
                "the second strategy."
            )

        return warnings

    def copy(self) -> "Task2VecFullConfig":
        """Create a deep copy of this configuration.

        Returns:
            An independent copy that can be modified without affecting
            the original.
        """
        return copy.deepcopy(self)

    def with_overrides(self, **kwargs: Any) -> "Task2VecFullConfig":
        """Create a copy with top-level field overrides.

        Supports dotted paths for nested overrides:
            cfg.with_overrides(**{"task2vec.embedding_dim": 256})

        Args:
            **kwargs: Field names (or dotted paths) to override.

        Returns:
            A new Task2VecFullConfig with the specified overrides.

        Raises:
            KeyError: If a top-level key is unknown.
        """
        new_cfg = self.copy()
        for key, value in kwargs.items():
            parts = key.split(".")
            if len(parts) == 1:
                if not hasattr(new_cfg, key):
                    raise KeyError(f"Unknown top-level field: {key!r}")
                setattr(new_cfg, key, value)
            elif len(parts) == 2:
                sub_name, field_name = parts
                sub_cfg = getattr(new_cfg, sub_name, None)
                if sub_cfg is None:
                    raise KeyError(f"Unknown sub-config: {sub_name!r}")
                if not hasattr(sub_cfg, field_name):
                    raise KeyError(
                        f"Unknown field {field_name!r} in {sub_name}"
                    )
                setattr(sub_cfg, field_name, value)
            else:
                raise KeyError(
                    f"Dotted path {key!r} has too many levels (max 2)"
                )
        # Re-validate after overrides
        new_cfg.__post_init__()
        new_cfg.task2vec.__post_init__()
        new_cfg.cluster.__post_init__()
        new_cfg.curriculum.__post_init__()
        new_cfg.registry.__post_init__()
        return new_cfg

    def merge(self, other: "Task2VecFullConfig") -> "Task2VecFullConfig":
        """Merge another config into this one, preferring *other*'s non-default values.

        This is a shallow merge: if a field in *other* differs from the
        default, it overrides the corresponding field in *self*. Sub-configs
        are merged field-by-field.

        Args:
            other: Configuration whose non-default values take precedence.

        Returns:
            A new merged Task2VecFullConfig.
        """
        defaults = Task2VecFullConfig()
        result = self.copy()

        # Merge top-level flags
        if other.use_task2vec != defaults.use_task2vec:
            result.use_task2vec = other.use_task2vec
        if other.seed != defaults.seed:
            result.seed = other.seed
        if other.log_level != defaults.log_level:
            result.log_level = other.log_level

        # Merge sub-configs field by field
        for sub_name in ("task2vec", "cluster", "curriculum", "registry"):
            self_sub = getattr(result, sub_name)
            other_sub = getattr(other, sub_name)
            default_sub = getattr(defaults, sub_name)
            for f in fields(other_sub):
                other_val = getattr(other_sub, f.name)
                default_val = getattr(default_sub, f.name)
                if other_val != default_val:
                    setattr(self_sub, f.name, other_val)

        return result

    def diff(self, other: "Task2VecFullConfig") -> Dict[str, Tuple[Any, Any]]:
        """Compute the difference between this config and another.

        Args:
            other: Configuration to compare against.

        Returns:
            Dictionary mapping field paths to (self_value, other_value) tuples
            for all fields that differ.
        """
        differences: Dict[str, Tuple[Any, Any]] = {}

        # Compare top-level fields
        for name in ("use_task2vec", "seed", "log_level"):
            self_val = getattr(self, name)
            other_val = getattr(other, name)
            if self_val != other_val:
                differences[name] = (self_val, other_val)

        # Compare sub-config fields
        for sub_name in ("task2vec", "cluster", "curriculum", "registry"):
            self_sub = getattr(self, sub_name)
            other_sub = getattr(other, sub_name)
            for f in fields(self_sub):
                self_val = getattr(self_sub, f.name)
                other_val = getattr(other_sub, f.name)
                if self_val != other_val:
                    differences[f"{sub_name}.{f.name}"] = (self_val, other_val)

        return differences


# =========================================================================
# 6. Convenience functions
# =========================================================================

def load_task2vec_config(path: str) -> Task2VecFullConfig:
    """Load a Task2VecFullConfig from a JSON file.

    Args:
        path: Path to a JSON configuration file.

    Returns:
        A validated Task2VecFullConfig instance.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        ValueError: If the config fails validation.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Task2VecFullConfig.from_dict(data)


def save_task2vec_config(config: Task2VecFullConfig, path: str) -> None:
    """Save a Task2VecFullConfig to a JSON file.

    Args:
        config: Configuration to save.
        path: Output file path.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(config.to_json(indent=2))
        f.write("\n")
    logger.info("Saved Task2Vec config to %s", path)


def get_preset(name: str) -> Task2VecFullConfig:
    """Get a named configuration preset.

    Args:
        name: One of ``"minimal"``, ``"dev"``, ``"production"``.

    Returns:
        The corresponding Task2VecFullConfig preset.

    Raises:
        ValueError: If *name* is not a recognized preset.
    """
    presets = {
        "minimal": Task2VecFullConfig.minimal,
        "dev": Task2VecFullConfig.dev,
        "production": Task2VecFullConfig.production,
    }
    if name not in presets:
        raise ValueError(
            f"Unknown preset {name!r}. Available: {list(presets.keys())}"
        )
    return presets[name]()


def list_presets() -> List[str]:
    """List available configuration preset names.

    Returns:
        Sorted list of preset name strings.
    """
    return ["dev", "minimal", "production"]


# =========================================================================
# 7. Self-Test Block
# =========================================================================

if __name__ == "__main__":
    import sys
    import traceback

    passed = 0
    failed = 0
    total = 0

    def run_test(name: str, test_fn):
        """Run a single test and report PASS/FAIL."""
        global passed, failed, total
        total += 1
        try:
            test_fn()
            print(f"  PASS  [{total:2d}] {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  [{total:2d}] {name}")
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print("Task2Vec Config Template -- Self-Test Suite")
    print("=" * 60)
    print()

    # ----------------------------------------------------------------
    # Test 1: Default construction
    # ----------------------------------------------------------------
    def test_default_construction():
        cfg = Task2VecFullConfig()
        assert cfg.task2vec.probe_model == "conv4"
        assert cfg.task2vec.embedding_dim == 512
        assert cfg.cluster.n_clusters == 8
        assert cfg.curriculum.strategy == "none"
        assert cfg.registry.format == "jsonl_npz"

    run_test("Default construction with expected values", test_default_construction)

    # ----------------------------------------------------------------
    # Test 2: Minimal preset
    # ----------------------------------------------------------------
    def test_minimal_preset():
        cfg = Task2VecFullConfig.minimal()
        assert cfg.task2vec.embedding_dim == 128
        assert cfg.cluster.n_clusters == 4
        assert cfg.curriculum.strategy == "none"
        assert cfg.registry.max_entries == 1000

    run_test("Minimal preset creates valid config", test_minimal_preset)

    # ----------------------------------------------------------------
    # Test 3: Dev preset
    # ----------------------------------------------------------------
    def test_dev_preset():
        cfg = Task2VecFullConfig.dev()
        assert cfg.task2vec.embedding_dim == 256
        assert cfg.cluster.n_clusters == 6
        assert cfg.curriculum.strategy == "easy_to_hard"
        assert cfg.registry.auto_extract is True

    run_test("Dev preset creates valid config", test_dev_preset)

    # ----------------------------------------------------------------
    # Test 4: Production preset
    # ----------------------------------------------------------------
    def test_production_preset():
        cfg = Task2VecFullConfig.production()
        assert cfg.task2vec.embedding_dim == 512
        assert cfg.task2vec.probe_model == "resnet12"
        assert cfg.cluster.n_clusters == 8
        assert cfg.curriculum.strategy == "diversity"
        assert cfg.curriculum.log_full_ids is True
        assert cfg.registry.compression is True

    run_test("Production preset creates valid config", test_production_preset)

    # ----------------------------------------------------------------
    # Test 5: to_dict / from_dict round-trip
    # ----------------------------------------------------------------
    def test_dict_roundtrip():
        for preset_name, preset_fn in [
            ("default", Task2VecFullConfig),
            ("minimal", Task2VecFullConfig.minimal),
            ("dev", Task2VecFullConfig.dev),
            ("production", Task2VecFullConfig.production),
        ]:
            original = preset_fn()
            d = original.to_dict()
            restored = Task2VecFullConfig.from_dict(d)
            # Compare all fields
            assert original.use_task2vec == restored.use_task2vec, \
                f"{preset_name}: use_task2vec mismatch"
            assert original.seed == restored.seed, \
                f"{preset_name}: seed mismatch"
            assert original.task2vec.to_dict() == restored.task2vec.to_dict(), \
                f"{preset_name}: task2vec sub-config mismatch"
            assert original.cluster.to_dict() == restored.cluster.to_dict(), \
                f"{preset_name}: cluster sub-config mismatch"
            assert original.curriculum.to_dict() == restored.curriculum.to_dict(), \
                f"{preset_name}: curriculum sub-config mismatch"
            assert original.registry.to_dict() == restored.registry.to_dict(), \
                f"{preset_name}: registry sub-config mismatch"

    run_test("to_dict / from_dict round-trip preserves all fields", test_dict_roundtrip)

    # ----------------------------------------------------------------
    # Test 6: JSON serialization round-trip
    # ----------------------------------------------------------------
    def test_json_roundtrip():
        original = Task2VecFullConfig.production()
        json_str = original.to_json()
        restored = Task2VecFullConfig.from_json(json_str)
        assert original.to_dict() == restored.to_dict()
        # Verify it is valid JSON
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert "task2vec" in parsed
        assert "cluster" in parsed
        assert "curriculum" in parsed
        assert "registry" in parsed

    run_test("JSON serialization round-trip", test_json_roundtrip)

    # ----------------------------------------------------------------
    # Test 7: Invalid probe_model raises ValueError
    # ----------------------------------------------------------------
    def test_invalid_probe_model():
        try:
            Task2VecConfig(probe_model="invalid_model")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "probe_model" in str(e)
            assert "invalid_model" in str(e)

    run_test("Invalid probe_model raises ValueError", test_invalid_probe_model)

    # ----------------------------------------------------------------
    # Test 8: Negative embedding_dim raises ValueError
    # ----------------------------------------------------------------
    def test_negative_embedding_dim():
        try:
            Task2VecConfig(embedding_dim=-1)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "embedding_dim" in str(e)

    run_test("Negative embedding_dim raises ValueError", test_negative_embedding_dim)

    # ----------------------------------------------------------------
    # Test 9: Zero embedding_dim raises ValueError
    # ----------------------------------------------------------------
    def test_zero_embedding_dim():
        try:
            Task2VecConfig(embedding_dim=0)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "embedding_dim" in str(e)

    run_test("Zero embedding_dim raises ValueError", test_zero_embedding_dim)

    # ----------------------------------------------------------------
    # Test 10: stability_threshold > 1 raises ValueError
    # ----------------------------------------------------------------
    def test_stability_threshold_over_1():
        try:
            ClusterConfig(stability_threshold=1.5)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "stability_threshold" in str(e)

    run_test("stability_threshold > 1 raises ValueError", test_stability_threshold_over_1)

    # ----------------------------------------------------------------
    # Test 11: stability_threshold < 0 raises ValueError
    # ----------------------------------------------------------------
    def test_stability_threshold_negative():
        try:
            ClusterConfig(stability_threshold=-0.1)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "stability_threshold" in str(e)

    run_test("stability_threshold < 0 raises ValueError", test_stability_threshold_negative)

    # ----------------------------------------------------------------
    # Test 12: Invalid curriculum strategy raises ValueError
    # ----------------------------------------------------------------
    def test_invalid_strategy():
        try:
            CurriculumConfig(strategy="unknown_strategy")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "strategy" in str(e)

    run_test("Invalid curriculum strategy raises ValueError", test_invalid_strategy)

    # ----------------------------------------------------------------
    # Test 13: Invalid registry format raises ValueError
    # ----------------------------------------------------------------
    def test_invalid_registry_format():
        try:
            RegistryConfig(format="sqlite")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "format" in str(e)

    run_test("Invalid registry format raises ValueError", test_invalid_registry_format)

    # ----------------------------------------------------------------
    # Test 14: Preset values -- minimal has smaller dims than production
    # ----------------------------------------------------------------
    def test_preset_ordering():
        minimal = Task2VecFullConfig.minimal()
        dev = Task2VecFullConfig.dev()
        prod = Task2VecFullConfig.production()
        assert minimal.task2vec.embedding_dim < dev.task2vec.embedding_dim, \
            "minimal embedding_dim should be < dev"
        assert dev.task2vec.embedding_dim < prod.task2vec.embedding_dim, \
            "dev embedding_dim should be < production"
        assert minimal.cluster.n_clusters < dev.cluster.n_clusters, \
            "minimal n_clusters should be < dev"
        assert dev.cluster.n_clusters < prod.cluster.n_clusters, \
            "dev n_clusters should be < production"

    run_test("Preset ordering: minimal < dev < production", test_preset_ordering)

    # ----------------------------------------------------------------
    # Test 15: Feature flag defaults to True
    # ----------------------------------------------------------------
    def test_feature_flag_default():
        cfg = Task2VecFullConfig()
        assert cfg.use_task2vec is True

    run_test("Feature flag use_task2vec defaults to True", test_feature_flag_default)

    # ----------------------------------------------------------------
    # Test 16: Deep copy -- modifying copy doesn't affect original
    # ----------------------------------------------------------------
    def test_deep_copy():
        original = Task2VecFullConfig.production()
        copied = original.copy()
        # Mutate the copy
        copied.task2vec.embedding_dim = 9999
        copied.cluster.n_clusters = 999
        copied.curriculum.strategy = "none"
        copied.use_task2vec = False
        copied.seed = 12345
        # Original should be unchanged
        assert original.task2vec.embedding_dim == 512
        assert original.cluster.n_clusters == 8
        assert original.curriculum.strategy == "diversity"
        assert original.use_task2vec is True
        assert original.seed == 42

    run_test("Deep copy: modifying copy doesn't affect original", test_deep_copy)

    # ----------------------------------------------------------------
    # Test 17: Sub-config from_dict with unknown keys
    # ----------------------------------------------------------------
    def test_unknown_keys_ignored():
        data = {
            "probe_model": "conv4",
            "embedding_dim": 256,
            "future_field": "should_be_ignored",
            "another_unknown": 999,
        }
        cfg = Task2VecConfig.from_dict(data)
        assert cfg.probe_model == "conv4"
        assert cfg.embedding_dim == 256
        assert not hasattr(cfg, "future_field")

    run_test("from_dict ignores unknown keys (forward compat)", test_unknown_keys_ignored)

    # ----------------------------------------------------------------
    # Test 18: diversity_min_distance out of range raises ValueError
    # ----------------------------------------------------------------
    def test_diversity_distance_range():
        try:
            CurriculumConfig(diversity_min_distance=2.5)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "diversity_min_distance" in str(e)
        try:
            CurriculumConfig(diversity_min_distance=-0.1)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "diversity_min_distance" in str(e)

    run_test("diversity_min_distance out of [0,2] raises ValueError", test_diversity_distance_range)

    # ----------------------------------------------------------------
    # Test 19: Invalid cluster algorithm raises ValueError
    # ----------------------------------------------------------------
    def test_invalid_cluster_algorithm():
        try:
            ClusterConfig(algorithm="spectral")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "algorithm" in str(e)

    run_test("Invalid cluster algorithm raises ValueError", test_invalid_cluster_algorithm)

    # ----------------------------------------------------------------
    # Test 20: with_overrides creates modified copy
    # ----------------------------------------------------------------
    def test_with_overrides():
        original = Task2VecFullConfig.production()
        modified = original.with_overrides(
            use_task2vec=False,
            seed=99,
            **{"task2vec.embedding_dim": 256, "cluster.n_clusters": 4},
        )
        # Modified should have new values
        assert modified.use_task2vec is False
        assert modified.seed == 99
        assert modified.task2vec.embedding_dim == 256
        assert modified.cluster.n_clusters == 4
        # Original should be unchanged
        assert original.use_task2vec is True
        assert original.seed == 42
        assert original.task2vec.embedding_dim == 512
        assert original.cluster.n_clusters == 8

    run_test("with_overrides creates modified copy", test_with_overrides)

    # ----------------------------------------------------------------
    # Test 21: validate() returns warnings for suspicious combos
    # ----------------------------------------------------------------
    def test_validate_warnings():
        cfg = Task2VecFullConfig(
            curriculum=CurriculumConfig(
                strategy="diversity",
                diversity_min_distance=0.05,
            ),
        )
        warnings = cfg.validate()
        assert any("diversity_min_distance" in w for w in warnings), \
            "Expected warning about low diversity_min_distance"

    run_test("validate() returns warnings for suspicious combos", test_validate_warnings)

    # ----------------------------------------------------------------
    # Test 22: summary() returns non-empty string
    # ----------------------------------------------------------------
    def test_summary():
        cfg = Task2VecFullConfig.production()
        s = cfg.summary()
        assert isinstance(s, str)
        assert len(s) > 100
        assert "Task2Vec" in s
        assert "512" in s
        assert "diversity" in s

    run_test("summary() returns informative string", test_summary)

    # ----------------------------------------------------------------
    # Test 23: diff() detects differences
    # ----------------------------------------------------------------
    def test_diff():
        a = Task2VecFullConfig.minimal()
        b = Task2VecFullConfig.production()
        d = a.diff(b)
        assert "task2vec.embedding_dim" in d
        assert d["task2vec.embedding_dim"] == (128, 512)
        assert "cluster.n_clusters" in d
        assert "curriculum.strategy" in d

    run_test("diff() detects differences between configs", test_diff)

    # ----------------------------------------------------------------
    # Test 24: get_preset convenience function
    # ----------------------------------------------------------------
    def test_get_preset():
        for name in ["minimal", "dev", "production"]:
            cfg = get_preset(name)
            assert isinstance(cfg, Task2VecFullConfig)
        try:
            get_preset("nonexistent")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    run_test("get_preset() returns valid configs", test_get_preset)

    # ----------------------------------------------------------------
    # Test 25: list_presets returns sorted names
    # ----------------------------------------------------------------
    def test_list_presets():
        names = list_presets()
        assert names == ["dev", "minimal", "production"]

    run_test("list_presets() returns sorted names", test_list_presets)

    # ----------------------------------------------------------------
    # Test 26: negative n_clusters raises ValueError
    # ----------------------------------------------------------------
    def test_negative_n_clusters():
        try:
            ClusterConfig(n_clusters=-1)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "n_clusters" in str(e)

    run_test("Negative n_clusters raises ValueError", test_negative_n_clusters)

    # ----------------------------------------------------------------
    # Test 27: negative warmup_epochs raises ValueError
    # ----------------------------------------------------------------
    def test_negative_warmup_epochs():
        try:
            CurriculumConfig(warmup_epochs=-1)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "warmup_epochs" in str(e)

    run_test("Negative warmup_epochs raises ValueError", test_negative_warmup_epochs)

    # ----------------------------------------------------------------
    # Test 28: merge combines two configs
    # ----------------------------------------------------------------
    def test_merge():
        base = Task2VecFullConfig.minimal()
        override = Task2VecFullConfig()
        # Modify override to have non-default embedding_dim
        override.task2vec.embedding_dim = 1024
        override.curriculum.strategy = "easy_to_hard"
        merged = base.merge(override)
        # Override's non-default values should win
        assert merged.task2vec.embedding_dim == 1024
        assert merged.curriculum.strategy == "easy_to_hard"
        # Base's non-default values (from minimal preset) that override
        # didn't change should remain
        assert merged.cluster.n_clusters == base.cluster.n_clusters

    run_test("merge() combines two configs correctly", test_merge)

    # ----------------------------------------------------------------
    # Test 29: Invalid log_level raises ValueError
    # ----------------------------------------------------------------
    def test_invalid_log_level():
        try:
            Task2VecFullConfig(log_level="TRACE")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "log_level" in str(e)

    run_test("Invalid log_level raises ValueError", test_invalid_log_level)

    # ----------------------------------------------------------------
    # Test 30: All sub-configs independently instantiate
    # ----------------------------------------------------------------
    def test_independent_sub_configs():
        t2v = Task2VecConfig()
        assert t2v.probe_model == "conv4"
        cl = ClusterConfig()
        assert cl.algorithm == "kmeans"
        cur = CurriculumConfig()
        assert cur.strategy == "none"
        reg = RegistryConfig()
        assert reg.format == "jsonl_npz"

    run_test("All sub-configs instantiate independently", test_independent_sub_configs)

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {total} total")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
    else:
        print("All tests passed.")
        sys.exit(0)
