---
name: Task2Vec Task Embeddings + Curriculum/Clustering
description: >-
  This skill should be used when the user asks to "implement Task2Vec embeddings",
  "add Fisher task embeddings", "implement curriculum ordering", "add task diversity",
  "implement meta-batch composition", "add task clustering", "implement embedding registry",
  "add curriculum learning", "implement easy-to-hard curriculum", "add diversity-constrained batches",
  "implement task similarity", "add probe network embedding", "implement diagonal Fisher",
  "add anti-curriculum", "implement task embedding extraction", "add cluster stability metrics",
  "implement ARI/NMI evaluation", "add silhouette scoring", "implement stratified meta-batches",
  "add task difficulty proxy", "implement embedding whitening",
  or mentions Task2Vec, Fisher information embeddings, task-space clustering,
  curriculum meta-learning, meta-batch diversity constraints, or task embedding registries
  in the cognitive pipeline.
version: 0.1.0
---

# Task2Vec Task Embeddings + Curriculum/Clustering

## Purpose

This skill standardizes the "task embedding" pipeline: given episodic few-shot tasks from
Phase 7, compute fixed-dimensional Fisher-information-based embeddings (Task2Vec style)
that enable curriculum ordering, diversity-constrained meta-batch composition, and offline
task-space analysis. The non-negotiable goals are deterministic embedding extraction and
measurable curriculum effects on task ordering.

## Key Files

| Target Module | Template Asset | Purpose |
|---|---|---|
| `brain_ai/meta/task2vec.py` | `assets/task2vec_extractor_template.py` | Task2VecExtractor: probe network, diagonal Fisher, embedding pipeline |
| `brain_ai/meta/embedding_registry.py` | `assets/embedding_registry_template.py` | TaskEmbeddingRegistry: persistent storage, querying, checkpoint integration |
| `brain_ai/meta/task_clustering.py` | `assets/clustering_template.py` | Clustering algorithms, stability metrics (ARI/NMI/silhouette), diagnostics |
| `brain_ai/meta/curriculum.py` | `assets/curriculum_template.py` | Curriculum ordering, meta-batch diversity, schedule functions |
| `brain_ai/config.py` (extend) | `assets/task2vec_config_template.py` | Task2VecConfig, ClusterConfig, CurriculumConfig, RegistryConfig |

## Public Contract

```python
extract_task_embedding(episode, *, seed, device) -> TaskEmbedding
update_registry(task_id, embedding, meta) -> None
get_curriculum_order(task_ids, strategy, seed) -> ordered_task_ids
build_meta_batch(task_ids, diversity_constraint, seed) -> batch_task_ids
```

`episode` is a `TaskEpisode` with canonicalized `task_id`, support set `(x_support, y_support)`,
and metadata. `TaskEmbedding` wraps the E-dimensional vector plus extraction diagnostics.

## TaskEmbedding Contract

| Field | Shape / Type | Description |
|---|---|---|
| `embedding` | `(E,)` | L2-normalized Fisher-derived task vector |
| `task_id` | `str` | Deterministic hash of (dataset, split, class_ids, support_indices, transforms) |
| `diagnostics` | `Dict[str, float]` | Fisher norm, sparsity, probe loss, extraction time |
| `probe_signature` | `str` | Hash of (model name, layer subset, preprocessing) |

**Hard invariants**:
- Given identical `(seed, dataset split, class/sample indices)`, embeddings are identical within 1e-7 on same device.
- Cross-device (CPU/CUDA) cosine similarity > 0.9999.
- Probe network is **frozen** during extraction (eval mode, no dropout, fixed BN).

## Embedding Computation Pipeline

Three-stage process: probe forward pass, diagonal Fisher accumulation, projection to E dimensions.

| Stage | Operation | Key Detail |
|---|---|---|
| 1. Probe forward | Fixed pretrained backbone + ephemeral N-way head | Eval mode, frozen weights |
| 2. Fisher estimate | Per-sample gradient squared, accumulated | Diagonal approximation, fp32 |
| 3. Projection | Layer subset selection, per-group aggregation, log1p + L2-norm | Configurable E dims |

The probe network is **not** the meta-learner — it is a fixed reference model. Embeddings are
relative to this probe, making them comparable across different meta-learning runs.

See `references/fisher-embedding.md` for probe selection, Fisher computation details, parameter
subset strategies, and normalization/whitening options.

## Clustering and Stability

Supported algorithms: k-means (seeded), agglomerative (fixed linkage), optional HDBSCAN.
Distance metric: cosine distance `d(u,v) = 1 - cos(u,v)`.

Stability is measured via adjusted Rand index (ARI) or normalized mutual information (NMI)
between clustering runs with different seeds. Target: ARI >= 0.9 for deterministic configs.

See `references/clustering-analysis.md` for algorithm details, stability protocols, and diagnostics.

## Curriculum Strategies

| Strategy | Mechanism | Use Case |
|---|---|---|
| Easy-to-hard | Order by difficulty proxy (distance to easy-cluster centroid) | Gradual difficulty increase |
| Diversity-constrained | Min pairwise cosine distance or stratified-per-cluster sampling | Prevent meta-update collapse |
| Anti-curriculum | Hard-first or mixed schedule with epoch-varying p(hard) | Robustness training |

Every epoch logs: ordered task_ids (or hash summary), strategy name, cluster histogram.
Kendall tau distance between curriculum and random ordering must be significant.

See `references/curriculum-strategies.md` for strategy implementations, schedule functions, and logging.

## Task Embedding Registry

Persistent artifact saved alongside checkpoints:

| Field | Type | Description |
|---|---|---|
| `task_id` | `str` | Deterministic task hash |
| `embedding` | `ndarray(E,)` | L2-normalized Fisher vector |
| `dataset` | `str` | Dataset name |
| `split` | `str` | Train/val/test |
| `n_way`, `k_shot`, `q_query` | `int` | Episode parameters |
| `class_ids` | `List[int]` | Selected classes |
| `probe_signature` | `str` | Probe model + layer subset hash |
| `extraction_seed` | `int` | Seed used for extraction |

Format: `task2vec_registry.jsonl` (metadata) + `embeddings.npz` (vectors).

See `references/registry-format.md` for schema details, versioning, and checkpoint integration.

## Configuration Surface

### Task2VecConfig

| Field | Default | Purpose |
|---|---|---|
| `probe_model` | `"conv4"` | Probe backbone: `"conv4"`, `"resnet12"`, `"vit_tiny"` |
| `layer_subset` | `"last_block"` | Which layers for Fisher: `"last_block"`, `"per_stage"`, `"all"` |
| `embedding_dim` | 512 | Target embedding dimension E |
| `aggregation` | `"per_channel"` | Fisher aggregation: `"per_channel"`, `"per_head"`, `"per_layer"` |
| `normalize` | `"log1p_l2"` | Normalization: `"log1p_l2"`, `"l2"`, `"whiten"` |
| `num_fisher_samples` | `None` | Samples for Fisher (None = all support) |

### ClusterConfig

| Field | Default | Purpose |
|---|---|---|
| `algorithm` | `"kmeans"` | `"kmeans"`, `"agglomerative"`, `"hdbscan"` |
| `n_clusters` | 8 | Number of clusters (ignored for HDBSCAN) |
| `distance_metric` | `"cosine"` | Distance metric |
| `stability_threshold` | 0.9 | Min ARI for stability gate |

### CurriculumConfig

| Field | Default | Purpose |
|---|---|---|
| `strategy` | `"none"` | `"none"`, `"easy_to_hard"`, `"diversity"`, `"anti_curriculum"`, `"mixed"` |
| `difficulty_proxy` | `"centroid_distance"` | Difficulty measure |
| `diversity_min_distance` | 0.3 | Min cosine distance for diversity batches |
| `schedule_fn` | `"linear"` | `"linear"`, `"cosine"`, `"step"` for mixed strategies |
| `log_task_order` | True | Log ordered task_ids each epoch |

Presets: `Task2VecFullConfig.minimal()`, `.dev()`, `.production()`.

## Done-When Gates

| Gate | Test | Threshold |
|---|---|---|
| **(a) Embedding determinism** | Fixed synthetic dataset + episode; extract twice; assert equality within 1e-7 on CPU; cross-device cosine > 0.9999 | Exact / tight tolerance |
| **(b) Cluster stability** | Cluster twice with different seeds; ARI/NMI >= threshold | ARI >= 0.9 |
| **(c) Curriculum changes ordering** | 50 task embeddings; random vs curriculum ordering; Kendall tau significant; logs contain strategy + order | Tau significant + logged |

## Common Failure Modes

| Symptom | Cause | Fix |
|---|---|---|
| Embeddings differ across runs | Probe not frozen or non-deterministic ops | Ensure eval mode, deterministic torch settings |
| All embeddings similar | Fisher dominated by one layer | Use per-stage subset, check layer balance |
| Clustering unstable | Too few tasks or too many clusters | Reduce k, increase task sample, check silhouette |
| Curriculum has no effect | Strategy not actually reordering | Verify Kendall tau, check difficulty proxy variance |
| Registry bloats disk | Storing full Fisher per task | Store only projected E-dim embedding, not raw Fisher |
| Cross-device mismatch | Float precision differences | Use fp32, accept cosine > 0.9999 tolerance |
| Whitening fails on small sets | Covariance matrix singular | Require min reference tasks, add regularization |

## Anti-Patterns

- **Using meta-learner as probe** -- probe must be fixed reference, not the model being trained
- **Raw Fisher without projection** -- too high-dimensional, not comparable across architectures
- **Exact equality across devices** -- floating-point differs; use cosine similarity threshold
- **Curriculum without logging** -- ordering decisions must be auditable and reproducible
- **Hardcoded cluster count** -- parameterize via ClusterConfig, not magic numbers
- **Updating probe during extraction** -- probe must be eval + frozen for determinism
- **Ignoring Fisher normalization** -- log1p + L2-norm essential for scale-invariant embeddings

## Additional Resources

### Reference Files

- **`references/fisher-embedding.md`** -- Probe network selection, diagonal Fisher computation, parameter subset, normalization/whitening
- **`references/clustering-analysis.md`** -- K-means/agglomerative/HDBSCAN, ARI/NMI, silhouette, stability protocols
- **`references/curriculum-strategies.md`** -- Easy-to-hard, diversity-constrained, anti-curriculum, schedule functions, logging
- **`references/registry-format.md`** -- Registry schema, JSONL + NPZ format, versioning, checkpoint integration
- **`references/testing-matrix.md`** -- All test cases: determinism, clustering stability, curriculum ordering

### Asset Templates

- **`assets/task2vec_extractor_template.py`** -- Task2VecExtractor, probe network, Fisher computation, embedding pipeline, self-test
- **`assets/embedding_registry_template.py`** -- TaskEmbeddingRegistry, JSONL/NPZ I/O, querying, self-test
- **`assets/clustering_template.py`** -- Clustering algorithms, stability metrics, diagnostics, self-test
- **`assets/curriculum_template.py`** -- Curriculum strategies, meta-batch composition, schedule functions, self-test
- **`assets/task2vec_config_template.py`** -- All configs, presets, serialization, self-test

### Scripts

- **`scripts/validate_task2vec.py`** -- Runtime contract validation (embedding determinism, cluster stability, curriculum effect)
- **`scripts/gen_task2vec_tests.py`** -- Generates `tests/test_task2vec.py` (~70+ test cases)
- **`scripts/embedding_benchmark.py`** -- Benchmark extraction throughput, clustering speed, curriculum overhead
