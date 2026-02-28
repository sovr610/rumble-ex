# Curriculum Strategies for Task2Vec Meta-Batch Composition

## Overview

Curriculum strategies control the order in which episodic tasks are presented to the meta-learner during Phase 7 training. Given a set of Task2Vec embeddings (Fisher-information-based vectors computed by the Task2VecExtractor), curriculum strategies reorder tasks or compose meta-batches to improve meta-learning convergence, robustness, or generalization. The strategies operate on pre-computed embeddings stored in the TaskEmbeddingRegistry and on cluster assignments produced by the clustering module.

Three families of strategies are supported:

- **Easy-to-hard curriculum**: Present tasks in order of increasing difficulty, measured by a configurable difficulty proxy derived from embeddings and registry metadata.
- **Diversity-constrained meta-batches**: Compose each meta-batch to maximize task-space coverage, preventing meta-updates from collapsing into a narrow region of the task embedding space.
- **Anti-curriculum / mixed schedule**: Present hard tasks first, or mix difficulty levels according to an epoch-varying schedule function.

Every strategy must produce auditable logs: the ordered task list (or its hash), the strategy name, cluster histograms, and difficulty statistics. The Kendall tau rank correlation between the curriculum ordering and a random baseline must be computed each epoch to confirm the curriculum actually changes task presentation order.

### Terminology

| Term | Definition |
|---|---|
| **Task embedding** | L2-normalized E-dimensional Fisher-derived vector for a single episodic task |
| **Difficulty proxy** | Scalar measure of task difficulty derived from embeddings, probe loss, or adaptation history |
| **Easy cluster** | The cluster whose member tasks have the lowest average probe loss during embedding extraction |
| **Centroid** | Mean embedding vector of a cluster (re-normalized to unit length after averaging) |
| **Meta-batch** | A subset of tasks selected for a single meta-learning outer update |
| **Schedule function** | A callable mapping `(epoch, max_epoch) -> p(hard)` controlling difficulty mixture |
| **Kendall tau** | Rank correlation coefficient measuring agreement between two orderings |

### Design Principles

1. **Embedding-driven, not data-driven.** All curriculum decisions operate on pre-computed E-dimensional embeddings, not raw data. This decouples curriculum logic from data loading and makes curriculum computation fast (milliseconds per epoch for hundreds of tasks).
2. **Deterministic given seed.** Every ordering and selection operation derives randomness from a deterministic seed. Given identical `(seed, epoch, task_ids, strategy)`, the curriculum produces identical task orderings across runs.
3. **Strategy-agnostic training loop.** The training loop calls `get_curriculum_order()` or `build_meta_batch()` without strategy-specific branches. Strategy selection is a configuration choice, not a code path change.
4. **Logging is mandatory.** Every epoch logs the full ordering (or its SHA-256 hash), strategy metadata, and statistical summaries. Curriculum decisions must be reproducible and auditable after the fact.
5. **Graceful fallback.** If the difficulty proxy has zero variance (all tasks equally difficult) or the diversity constraint cannot be satisfied, fall back to random ordering with a logged warning.


## Easy-to-Hard Curriculum

### Concept

Present tasks in order of increasing difficulty within each epoch. The meta-learner first adapts to simple tasks (building a reliable base initialization), then encounters progressively harder tasks that stress-test and refine the initialization. This mirrors biological and pedagogical learning progressions.

### Difficulty Proxy Options

Three difficulty proxy functions are supported. Each takes a task embedding, the registry metadata for that task, and the clustering context, and returns a scalar difficulty value. Higher values indicate harder tasks.

#### Proxy 1: `centroid_distance`

Compute the cosine distance from each task embedding to the "easy cluster" centroid. The easy cluster is the cluster whose member tasks have the lowest average probe loss (recorded in the registry during embedding extraction). Tasks near this centroid are easy; tasks far from it are hard.

```python
def compute_centroid_distance(
    task_embeddings: Dict[str, np.ndarray],
    cluster_assignments: Dict[str, int],
    cluster_centroids: Dict[int, np.ndarray],
    easy_cluster_id: int,
) -> Dict[str, float]:
    """Compute cosine distance from each task to the easy-cluster centroid.

    Args:
        task_embeddings: Mapping from task_id to L2-normalized embedding vector.
        cluster_assignments: Mapping from task_id to cluster label.
        cluster_centroids: Mapping from cluster label to centroid vector.
        easy_cluster_id: ID of the cluster with lowest average probe loss.

    Returns:
        Mapping from task_id to difficulty score (cosine distance to easy centroid).
    """
    easy_centroid = cluster_centroids[easy_cluster_id]
    # Re-normalize centroid (mean of unit vectors is not necessarily unit)
    easy_centroid = easy_centroid / (np.linalg.norm(easy_centroid) + 1e-12)

    difficulty = {}
    for task_id, emb in task_embeddings.items():
        cosine_sim = np.dot(emb, easy_centroid)
        cosine_dist = 1.0 - cosine_sim
        difficulty[task_id] = float(cosine_dist)

    return difficulty
```

Identify the easy cluster by scanning registry metadata:

```python
def identify_easy_cluster(
    registry: 'TaskEmbeddingRegistry',
    cluster_assignments: Dict[str, int],
) -> int:
    """Find the cluster with the lowest average probe loss.

    Args:
        registry: TaskEmbeddingRegistry containing per-task diagnostics.
        cluster_assignments: Mapping from task_id to cluster label.

    Returns:
        Cluster ID with the lowest mean probe loss.
    """
    cluster_losses = defaultdict(list)
    for task_id, cluster_id in cluster_assignments.items():
        meta = registry.get_metadata(task_id)
        if meta is not None and 'probe_loss' in meta.get('diagnostics', {}):
            cluster_losses[cluster_id].append(meta['diagnostics']['probe_loss'])

    if not cluster_losses:
        raise ValueError(
            "No probe_loss found in registry diagnostics. "
            "Cannot identify easy cluster. Run embedding extraction first."
        )

    mean_losses = {cid: np.mean(losses) for cid, losses in cluster_losses.items()}
    easy_cluster_id = min(mean_losses, key=mean_losses.get)
    return easy_cluster_id
```

#### Proxy 2: `adaptation_gain`

Use historical meta-learning performance per cluster as a difficulty signal. Tasks from clusters where the meta-learner historically achieves high adaptation gain (large accuracy improvement from pre- to post-adaptation) are considered easy. Tasks from clusters with low adaptation gain are hard.

This proxy requires the training loop to record per-task adaptation metrics in the registry:

```python
def compute_adaptation_gain_difficulty(
    task_ids: List[str],
    registry: 'TaskEmbeddingRegistry',
) -> Dict[str, float]:
    """Compute difficulty from historical adaptation gain.

    Tasks with HIGH adaptation gain are EASY (the model learns them quickly).
    Tasks with LOW adaptation gain are HARD (the model struggles).

    Args:
        task_ids: List of task identifiers to score.
        registry: Registry containing historical 'adaptation_gain' per task.

    Returns:
        Mapping from task_id to difficulty score.
        Higher score = harder task.
    """
    difficulty = {}
    gains = []

    for task_id in task_ids:
        meta = registry.get_metadata(task_id)
        gain = meta.get('adaptation_gain', None) if meta else None
        if gain is not None:
            gains.append((task_id, gain))

    if not gains:
        # Fallback: uniform difficulty if no historical data
        return {tid: 0.5 for tid in task_ids}

    # Invert: high gain = low difficulty, low gain = high difficulty
    max_gain = max(g for _, g in gains)
    min_gain = min(g for _, g in gains)
    gain_range = max_gain - min_gain if max_gain > min_gain else 1.0

    for task_id, gain in gains:
        # Normalize to [0, 1] and invert
        normalized = (gain - min_gain) / gain_range
        difficulty[task_id] = 1.0 - normalized

    # Assign median difficulty to tasks without history
    median_diff = np.median(list(difficulty.values())) if difficulty else 0.5
    for task_id in task_ids:
        if task_id not in difficulty:
            difficulty[task_id] = median_diff

    return difficulty
```

#### Proxy 3: `loss_slope`

Measure the rate of loss decrease during inner-loop adaptation. Tasks where the loss drops quickly (steep negative slope) are easy. Tasks where the loss decreases slowly or not at all are hard. This proxy requires per-step inner-loop loss logs from recent training.

```python
def compute_loss_slope_difficulty(
    task_ids: List[str],
    registry: 'TaskEmbeddingRegistry',
) -> Dict[str, float]:
    """Compute difficulty from the slope of inner-loop loss decrease.

    A steep negative slope (fast loss decrease) indicates an easy task.
    A flat or positive slope indicates a hard task.

    Args:
        task_ids: List of task identifiers to score.
        registry: Registry containing 'inner_loss_curve' per task.
            Each curve is a list of loss values, one per inner step.

    Returns:
        Mapping from task_id to difficulty score.
        Higher score = harder task (flatter or positive slope).
    """
    difficulty = {}

    for task_id in task_ids:
        meta = registry.get_metadata(task_id)
        curve = meta.get('inner_loss_curve', None) if meta else None

        if curve is not None and len(curve) >= 2:
            # Simple linear regression slope: negative slope = easy
            steps = np.arange(len(curve), dtype=np.float64)
            losses = np.array(curve, dtype=np.float64)
            # slope = cov(steps, losses) / var(steps)
            slope = np.polyfit(steps, losses, 1)[0]
            # Negate: steep negative slope -> low difficulty value -> easy
            # Flat/positive slope -> high difficulty value -> hard
            difficulty[task_id] = float(-slope)  # Negate so positive = easy becomes negative
        else:
            difficulty[task_id] = 0.0  # Unknown; assign neutral

    # Shift so minimum difficulty is 0
    if difficulty:
        min_d = min(difficulty.values())
        difficulty = {k: v - min_d for k, v in difficulty.items()}

    return difficulty
```

### Ordering Algorithm

Sort tasks by increasing difficulty proxy value. Ties are broken by task_id lexicographic order to ensure determinism.

```python
def easy_to_hard_order(
    task_ids: List[str],
    difficulty: Dict[str, float],
) -> List[str]:
    """Sort tasks from easiest to hardest.

    Args:
        task_ids: List of task identifiers to order.
        difficulty: Mapping from task_id to difficulty score.

    Returns:
        Task IDs sorted by increasing difficulty.
        Ties broken by task_id lexicographic order.
    """
    return sorted(task_ids, key=lambda tid: (difficulty.get(tid, 0.0), tid))
```

### Epoch-Level Reordering

Recompute the ordering before each epoch. If the difficulty proxy depends on training history (e.g., `adaptation_gain` or `loss_slope`), the ordering may change between epochs as the model improves. If the proxy is static (e.g., `centroid_distance`), the ordering is fixed but the recomputation is still performed for logging consistency.

```python
def apply_curriculum_epoch(
    task_ids: List[str],
    strategy: str,
    epoch: int,
    registry: 'TaskEmbeddingRegistry',
    cluster_ctx: 'ClusterContext',
    config: 'CurriculumConfig',
    seed: int,
) -> List[str]:
    """Apply curriculum ordering for a single epoch.

    Args:
        task_ids: Full list of available task IDs.
        strategy: One of 'easy_to_hard', 'anti_curriculum', 'mixed', 'none'.
        epoch: Current epoch number.
        registry: TaskEmbeddingRegistry for metadata lookups.
        cluster_ctx: ClusterContext with assignments, centroids, easy_cluster_id.
        config: CurriculumConfig with strategy parameters.
        seed: Global seed for reproducibility.

    Returns:
        Ordered list of task IDs for this epoch.
    """
    if strategy == 'none':
        # Random shuffle, seeded
        rng = np.random.RandomState(_epoch_seed(seed, epoch))
        shuffled = list(task_ids)
        rng.shuffle(shuffled)
        return shuffled

    # Compute difficulty proxy
    difficulty = compute_difficulty(task_ids, config.difficulty_proxy, registry, cluster_ctx)

    if strategy == 'easy_to_hard':
        return easy_to_hard_order(task_ids, difficulty)
    elif strategy == 'anti_curriculum':
        return list(reversed(easy_to_hard_order(task_ids, difficulty)))
    elif strategy == 'mixed':
        return mixed_schedule_order(task_ids, difficulty, epoch, config, seed)
    else:
        raise ValueError(f"Unknown curriculum strategy: {strategy}")
```

### Warmup Phase

Optionally run random ordering for the first N epochs before the curriculum kicks in. This allows the meta-learner to build a reasonable initialization without being biased by the curriculum ordering. Set `config.warmup_epochs` to control the warmup duration.

```python
def get_curriculum_order(
    task_ids: List[str],
    strategy: str,
    epoch: int,
    seed: int,
    registry: 'TaskEmbeddingRegistry',
    cluster_ctx: 'ClusterContext',
    config: 'CurriculumConfig',
) -> List[str]:
    """Top-level curriculum ordering with warmup support.

    Args:
        task_ids: Full list of available task IDs.
        strategy: Curriculum strategy name.
        epoch: Current epoch number (0-indexed).
        seed: Global seed.
        registry: TaskEmbeddingRegistry.
        cluster_ctx: ClusterContext.
        config: CurriculumConfig.

    Returns:
        Ordered list of task IDs for this epoch.
    """
    # Warmup: random ordering for first N epochs
    if epoch < config.warmup_epochs:
        rng = np.random.RandomState(_epoch_seed(seed, epoch))
        shuffled = list(task_ids)
        rng.shuffle(shuffled)
        return shuffled

    # Apply curriculum strategy
    ordered = apply_curriculum_epoch(
        task_ids, strategy, epoch, registry, cluster_ctx, config, seed,
    )

    return ordered


def _epoch_seed(seed: int, epoch: int) -> int:
    """Deterministic per-epoch seed."""
    import hashlib, struct
    data = struct.pack('>qq', seed, epoch)
    digest = hashlib.sha256(data).digest()
    return int.from_bytes(digest[:4], 'big')
```


## Diversity-Constrained Meta-Batches

### Goal

Prevent meta-updates from collapsing into a narrow region of task-space. When the meta-batch contains only similar tasks, the outer gradient points in a task-specific direction that does not generalize. Diversity-constrained batches ensure the meta-gradient integrates information from multiple distinct regions of the task embedding space.

### Method A: Minimum Pairwise Cosine Distance Threshold

Select tasks greedily, ensuring every new task is sufficiently distant (in cosine distance) from all already-selected tasks.

```python
def build_meta_batch_distance(
    task_ids: List[str],
    task_embeddings: Dict[str, np.ndarray],
    batch_size: int,
    min_distance: float,
    seed: int,
    epoch: int,
) -> List[str]:
    """Build a diversity-constrained meta-batch using pairwise distance threshold.

    Greedy selection: pick a random first task, then add tasks one by one,
    accepting only those whose cosine distance to ALL already-selected tasks
    exceeds min_distance. If the batch cannot be filled at the current threshold,
    relax the threshold by halving and retry.

    Args:
        task_ids: Pool of candidate task IDs.
        task_embeddings: Mapping from task_id to L2-normalized embedding.
        batch_size: Target number of tasks in the meta-batch.
        min_distance: Minimum cosine distance threshold.
        seed: Global seed.
        epoch: Current epoch (for deterministic seed derivation).

    Returns:
        List of selected task IDs, length exactly batch_size.
    """
    rng = np.random.RandomState(_epoch_seed(seed, epoch))
    candidates = list(task_ids)
    rng.shuffle(candidates)

    threshold = min_distance
    max_relaxation_steps = 10

    for relaxation in range(max_relaxation_steps):
        selected = []
        selected_embeddings = []

        for task_id in candidates:
            emb = task_embeddings[task_id]

            if len(selected) == 0:
                # Always accept the first task
                selected.append(task_id)
                selected_embeddings.append(emb)
            else:
                # Check distance to all selected tasks
                distances = [
                    1.0 - np.dot(emb, sel_emb)
                    for sel_emb in selected_embeddings
                ]
                if min(distances) >= threshold:
                    selected.append(task_id)
                    selected_embeddings.append(emb)

            if len(selected) == batch_size:
                return selected

        # Batch not filled: relax threshold
        if len(selected) >= batch_size:
            return selected[:batch_size]

        threshold *= 0.5  # Halve the threshold and retry

    # Final fallback: fill remaining slots with random tasks not yet selected
    remaining = [tid for tid in candidates if tid not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[:batch_size - len(selected)])

    return selected[:batch_size]
```

### Method B: Stratified-Per-Cluster Sampling

Sample one task per cluster in round-robin fashion. This guarantees representation from every cluster in each meta-batch, providing structural diversity.

```python
def build_meta_batch_stratified(
    task_ids: List[str],
    cluster_assignments: Dict[str, int],
    batch_size: int,
    seed: int,
    epoch: int,
) -> List[str]:
    """Build a meta-batch by sampling one task per cluster (round-robin).

    If there are more clusters than batch_size, sample batch_size clusters
    uniformly. If there are fewer clusters than batch_size, sample multiple
    tasks per cluster to fill the batch.

    Args:
        task_ids: Pool of candidate task IDs.
        cluster_assignments: Mapping from task_id to cluster label.
        batch_size: Target number of tasks in the meta-batch.
        seed: Global seed.
        epoch: Current epoch.

    Returns:
        List of selected task IDs, length exactly batch_size.
    """
    rng = np.random.RandomState(_epoch_seed(seed, epoch))

    # Group tasks by cluster
    cluster_to_tasks = defaultdict(list)
    for task_id in task_ids:
        if task_id in cluster_assignments:
            cluster_to_tasks[cluster_assignments[task_id]].append(task_id)

    cluster_ids = sorted(cluster_to_tasks.keys())
    n_clusters = len(cluster_ids)

    if n_clusters == 0:
        # No cluster info: fall back to random
        pool = list(task_ids)
        rng.shuffle(pool)
        return pool[:batch_size]

    selected = []

    if n_clusters >= batch_size:
        # More clusters than needed: sample batch_size clusters, one task each
        chosen_clusters = rng.choice(cluster_ids, size=batch_size, replace=False)
        for cid in chosen_clusters:
            tasks_in_cluster = cluster_to_tasks[cid]
            chosen_task = tasks_in_cluster[rng.randint(len(tasks_in_cluster))]
            selected.append(chosen_task)
    else:
        # Fewer clusters than batch_size: round-robin with multiple samples per cluster
        # First pass: one task per cluster
        for cid in cluster_ids:
            tasks_in_cluster = cluster_to_tasks[cid]
            chosen_task = tasks_in_cluster[rng.randint(len(tasks_in_cluster))]
            selected.append(chosen_task)

        # Fill remaining slots: cycle through clusters
        remaining = batch_size - len(selected)
        cycle_idx = 0
        while remaining > 0:
            cid = cluster_ids[cycle_idx % n_clusters]
            tasks_in_cluster = cluster_to_tasks[cid]
            # Sample a task not already selected, if possible
            available = [t for t in tasks_in_cluster if t not in selected]
            if not available:
                # All tasks in this cluster already selected; allow replacement
                available = tasks_in_cluster
            chosen_task = available[rng.randint(len(available))]
            selected.append(chosen_task)
            remaining -= 1
            cycle_idx += 1

    return selected[:batch_size]
```

### Unified Meta-Batch Builder

Route to the appropriate method based on the diversity constraint configuration.

```python
def build_meta_batch(
    task_ids: List[str],
    diversity_constraint: str,
    seed: int,
    epoch: int,
    task_embeddings: Dict[str, np.ndarray],
    cluster_assignments: Dict[str, int],
    config: 'CurriculumConfig',
) -> List[str]:
    """Build a diversity-constrained meta-batch.

    Args:
        task_ids: Pool of candidate task IDs.
        diversity_constraint: One of 'min_distance', 'stratified', 'none'.
        seed: Global seed.
        epoch: Current epoch.
        task_embeddings: Mapping from task_id to embedding vector.
        cluster_assignments: Mapping from task_id to cluster label.
        config: CurriculumConfig with batch_size, diversity_min_distance, etc.

    Returns:
        List of selected task IDs for this meta-batch.
    """
    if diversity_constraint == 'none':
        rng = np.random.RandomState(_epoch_seed(seed, epoch))
        pool = list(task_ids)
        rng.shuffle(pool)
        return pool[:config.meta_batch_size]

    elif diversity_constraint == 'min_distance':
        return build_meta_batch_distance(
            task_ids=task_ids,
            task_embeddings=task_embeddings,
            batch_size=config.meta_batch_size,
            min_distance=config.diversity_min_distance,
            seed=seed,
            epoch=epoch,
        )

    elif diversity_constraint == 'stratified':
        return build_meta_batch_stratified(
            task_ids=task_ids,
            cluster_assignments=cluster_assignments,
            batch_size=config.meta_batch_size,
            seed=seed,
            epoch=epoch,
        )

    else:
        raise ValueError(
            f"Unknown diversity constraint: {diversity_constraint}. "
            f"Choose from 'none', 'min_distance', 'stratified'."
        )
```

### Batch Size Enforcement

Every meta-batch builder must return exactly `config.meta_batch_size` tasks. If the pool is smaller than the batch size, sample with replacement and log a warning. Never return a partial batch -- downstream meta-loss computation expects a fixed batch dimension for vectorized operations.


## Anti-Curriculum (Hard-First) and Mixed Schedule

### Hard-First Ordering

Reverse of easy-to-hard: present the most difficult tasks first. Compute the same difficulty proxy and sort in decreasing order. Use this strategy to stress-test the initialization early, forcing the meta-learner to develop robust representations before fine-tuning on easier tasks.

```python
def hard_first_order(
    task_ids: List[str],
    difficulty: Dict[str, float],
) -> List[str]:
    """Sort tasks from hardest to easiest.

    Args:
        task_ids: List of task identifiers to order.
        difficulty: Mapping from task_id to difficulty score.

    Returns:
        Task IDs sorted by decreasing difficulty.
        Ties broken by task_id lexicographic order (reversed).
    """
    return sorted(task_ids, key=lambda tid: (-difficulty.get(tid, 0.0), tid))
```

### Mixed Schedule with Epoch-Varying Difficulty

Instead of a fixed ordering, sample tasks with difficulty-dependent weights that change over the course of training. A schedule function `p_hard(epoch, max_epoch)` controls the probability of sampling hard tasks at each epoch.

At each epoch:
1. Compute the difficulty proxy for all tasks.
2. Evaluate `p_hard = schedule_fn(epoch, max_epoch)`.
3. Assign sampling weights: tasks with difficulty above the median receive weight `p_hard`; tasks below receive weight `1 - p_hard`.
4. Sample `len(task_ids)` tasks (with replacement allowed) according to these weights.

```python
def mixed_schedule_order(
    task_ids: List[str],
    difficulty: Dict[str, float],
    epoch: int,
    config: 'CurriculumConfig',
    seed: int,
) -> List[str]:
    """Order tasks using a mixed difficulty schedule.

    Args:
        task_ids: Full list of available task IDs.
        difficulty: Mapping from task_id to difficulty score.
        epoch: Current epoch number.
        config: CurriculumConfig with schedule_fn name and max_epochs.
        seed: Global seed.

    Returns:
        Ordered list of task IDs for this epoch, sampled according to
        the difficulty schedule.
    """
    rng = np.random.RandomState(_epoch_seed(seed, epoch))

    # Compute p(hard) from schedule
    schedule = get_schedule_fn(config.schedule_fn)
    p_hard = schedule(epoch, config.max_epochs)
    p_hard = np.clip(p_hard, 0.0, 1.0)

    # Split tasks into easy and hard by median difficulty
    difficulties = np.array([difficulty.get(tid, 0.0) for tid in task_ids])
    median_diff = np.median(difficulties)

    easy_ids = [tid for tid in task_ids if difficulty.get(tid, 0.0) <= median_diff]
    hard_ids = [tid for tid in task_ids if difficulty.get(tid, 0.0) > median_diff]

    # Handle edge case: all tasks on one side of the median
    if not hard_ids:
        hard_ids = easy_ids
    if not easy_ids:
        easy_ids = hard_ids

    # Assign sampling weights
    weights = np.zeros(len(task_ids))
    for i, tid in enumerate(task_ids):
        if difficulty.get(tid, 0.0) > median_diff:
            weights[i] = p_hard
        else:
            weights[i] = 1.0 - p_hard

    # Normalize weights to a probability distribution
    weight_sum = weights.sum()
    if weight_sum < 1e-12:
        weights = np.ones(len(task_ids)) / len(task_ids)
    else:
        weights = weights / weight_sum

    # Sample with replacement according to weights
    indices = rng.choice(len(task_ids), size=len(task_ids), replace=True, p=weights)
    ordered = [task_ids[i] for i in indices]

    return ordered
```

### Alternating Cluster Families

Cycle through cluster groups each epoch. In epoch 0, present tasks from clusters 0 and 1. In epoch 1, present tasks from clusters 2 and 3. And so on. This ensures the meta-learner experiences different regions of task space in different epochs, preventing overfitting to any one cluster family.

```python
def alternating_cluster_order(
    task_ids: List[str],
    cluster_assignments: Dict[str, int],
    epoch: int,
    clusters_per_epoch: int,
    seed: int,
) -> List[str]:
    """Select tasks from a rotating subset of clusters each epoch.

    Args:
        task_ids: Full list of available task IDs.
        cluster_assignments: Mapping from task_id to cluster label.
        epoch: Current epoch number.
        clusters_per_epoch: Number of clusters to include per epoch.
        seed: Global seed.

    Returns:
        Ordered list of task IDs drawn from the active clusters.
    """
    rng = np.random.RandomState(_epoch_seed(seed, epoch))

    cluster_ids = sorted(set(cluster_assignments.values()))
    n_clusters = len(cluster_ids)

    if clusters_per_epoch >= n_clusters:
        # All clusters active; fall back to random within-epoch shuffle
        pool = list(task_ids)
        rng.shuffle(pool)
        return pool

    # Rotate: select clusters_per_epoch clusters starting at epoch offset
    start = (epoch * clusters_per_epoch) % n_clusters
    active_clusters = set()
    for i in range(clusters_per_epoch):
        active_clusters.add(cluster_ids[(start + i) % n_clusters])

    # Gather tasks from active clusters
    active_tasks = [
        tid for tid in task_ids
        if cluster_assignments.get(tid) in active_clusters
    ]

    if not active_tasks:
        # Fallback: all tasks
        active_tasks = list(task_ids)

    rng.shuffle(active_tasks)
    return active_tasks
```


## Schedule Functions

Schedule functions map `(epoch, max_epoch)` to a value `p(hard)` in `[0, 1]` that controls the difficulty mixture in mixed-schedule strategies.

### Built-In Schedules

```python
def schedule_linear(epoch: int, max_epoch: int) -> float:
    """Linear increase in p(hard) from 0 to 1.

    p(hard) = epoch / max_epoch

    At epoch 0, only easy tasks. At max_epoch, only hard tasks.
    Linear interpolation in between.
    """
    if max_epoch <= 0:
        return 0.0
    return min(epoch / max_epoch, 1.0)


def schedule_cosine(epoch: int, max_epoch: int) -> float:
    """Cosine-shaped increase in p(hard) from 0 to 1.

    p(hard) = 0.5 * (1 - cos(pi * epoch / max_epoch))

    Starts slow, accelerates through the middle epochs, and
    decelerates toward the end. Provides a smooth S-curve transition
    from easy to hard.
    """
    import math
    if max_epoch <= 0:
        return 0.0
    progress = min(epoch / max_epoch, 1.0)
    return 0.5 * (1.0 - math.cos(math.pi * progress))


def schedule_step(epoch: int, max_epoch: int, switch_fraction: float = 0.5) -> float:
    """Step function: p(hard) = 0 before switch point, 1 after.

    Args:
        epoch: Current epoch.
        max_epoch: Maximum epoch count.
        switch_fraction: Fraction of training at which to switch.
            Default 0.5 (switch at midpoint).

    Returns:
        0.0 if epoch < switch_epoch, else 1.0.
    """
    switch_epoch = int(max_epoch * switch_fraction)
    return 0.0 if epoch < switch_epoch else 1.0
```

### Custom Schedule Functions

Register user-defined schedule functions via a callable that matches the `(epoch, max_epoch) -> float` signature.

```python
# Schedule function registry
_SCHEDULE_REGISTRY = {
    'linear': schedule_linear,
    'cosine': schedule_cosine,
    'step': schedule_step,
}


def register_schedule(name: str, fn) -> None:
    """Register a custom schedule function.

    Args:
        name: Name for the schedule (used in CurriculumConfig.schedule_fn).
        fn: Callable with signature (epoch: int, max_epoch: int) -> float.
            Must return a value in [0, 1].
    """
    _SCHEDULE_REGISTRY[name] = fn


def get_schedule_fn(name: str):
    """Retrieve a schedule function by name.

    Args:
        name: Registered schedule name.

    Returns:
        The schedule callable.

    Raises:
        ValueError: If the name is not registered.
    """
    if name not in _SCHEDULE_REGISTRY:
        raise ValueError(
            f"Unknown schedule function: '{name}'. "
            f"Available: {list(_SCHEDULE_REGISTRY.keys())}. "
            f"Register custom schedules with register_schedule()."
        )
    return _SCHEDULE_REGISTRY[name]
```

Example: register an exponential warmup schedule.

```python
def schedule_exponential(epoch: int, max_epoch: int) -> float:
    """Exponential ramp: fast initial increase, then saturation."""
    import math
    if max_epoch <= 0:
        return 0.0
    # tau controls the speed; 3.0 means ~95% at 100% of training
    tau = 3.0
    return 1.0 - math.exp(-tau * epoch / max_epoch)

register_schedule('exponential', schedule_exponential)
```


## Logging Requirements

### Per-Epoch Log Fields

Every epoch, regardless of strategy, log the following fields as a JSON object:

```python
def log_curriculum_epoch(
    epoch: int,
    strategy: str,
    ordered_task_ids: List[str],
    difficulty: Dict[str, float],
    cluster_assignments: Dict[str, int],
    random_baseline_ids: List[str],
    logger,
) -> Dict:
    """Log curriculum state for one epoch.

    Args:
        epoch: Current epoch number.
        strategy: Strategy name (e.g., 'easy_to_hard', 'mixed').
        ordered_task_ids: The curriculum-ordered list of task IDs.
        difficulty: Mapping from task_id to difficulty proxy value.
        cluster_assignments: Mapping from task_id to cluster label.
        random_baseline_ids: Random-ordered task list for Kendall tau.
        logger: Logger instance.

    Returns:
        Dict containing all logged fields (also written to logger).
    """
    import hashlib
    import json

    # Task order hash (SHA-256 of concatenated task IDs)
    order_str = ','.join(ordered_task_ids)
    order_hash = hashlib.sha256(order_str.encode()).hexdigest()

    # Cluster histogram
    cluster_hist = defaultdict(int)
    for tid in ordered_task_ids:
        cid = cluster_assignments.get(tid, -1)
        cluster_hist[cid] += 1

    # Difficulty proxy statistics
    diff_values = [difficulty.get(tid, 0.0) for tid in ordered_task_ids]
    diff_stats = {
        'mean': float(np.mean(diff_values)),
        'std': float(np.std(diff_values)),
        'min': float(np.min(diff_values)),
        'max': float(np.max(diff_values)),
    }

    # Kendall tau
    tau, p_value = compute_kendall_tau(ordered_task_ids, random_baseline_ids)

    log_entry = {
        'epoch': epoch,
        'strategy': strategy,
        'task_order_hash': order_hash,
        'num_tasks': len(ordered_task_ids),
        'cluster_histogram': dict(cluster_hist),
        'difficulty_stats': diff_stats,
        'kendall_tau': float(tau),
        'kendall_p_value': float(p_value),
    }

    logger.info(f"Curriculum epoch {epoch}: {json.dumps(log_entry)}")
    return log_entry
```

### Log Entry Schema

| Field | Type | Description |
|---|---|---|
| `epoch` | `int` | Epoch number (0-indexed) |
| `strategy` | `str` | Strategy name used for this epoch |
| `task_order_hash` | `str` | SHA-256 hex digest of the comma-joined ordered task_ids |
| `num_tasks` | `int` | Number of tasks in the ordered list |
| `cluster_histogram` | `Dict[int, int]` | Count of tasks per cluster in the ordered list |
| `difficulty_stats` | `Dict[str, float]` | Mean, std, min, max of difficulty proxy values |
| `kendall_tau` | `float` | Kendall tau rank correlation vs. random baseline |
| `kendall_p_value` | `float` | P-value for the Kendall tau test |

Write the full JSON log entry to a file `curriculum_log.jsonl` (one line per epoch) for downstream analysis.


## Kendall Tau Validation

### Purpose

Kendall tau rank correlation measures the agreement between two orderings of the same elements. A tau of +1.0 means the orderings are identical. A tau of -1.0 means one is the reverse of the other. A tau near 0.0 means the orderings are unrelated.

Use Kendall tau to validate that the curriculum strategy actually changes the task ordering relative to a random baseline. A curriculum that produces an ordering indistinguishable from random (tau near 0, p-value > 0.05) is not providing any curriculum signal and should be investigated.

### Implementation

Prefer `scipy.stats.kendalltau` when available. Provide a pure-Python fallback for environments without SciPy.

```python
def compute_kendall_tau(
    ordered_ids: List[str],
    baseline_ids: List[str],
) -> Tuple[float, float]:
    """Compute Kendall tau rank correlation between curriculum and baseline ordering.

    Args:
        ordered_ids: Curriculum-ordered list of task IDs.
        baseline_ids: Baseline (random) ordered list of task IDs.
            Must contain the same task IDs as ordered_ids.

    Returns:
        (tau, p_value): Kendall tau statistic and two-sided p-value.
        tau > 0 means positive correlation (similar ordering).
        tau < 0 means negative correlation (reversed ordering).
        p_value < 0.05 means the correlation is statistically significant.
    """
    # Build rank vectors
    if set(ordered_ids) != set(baseline_ids):
        raise ValueError(
            "ordered_ids and baseline_ids must contain the same task IDs"
        )

    # Assign ranks in curriculum ordering
    curriculum_rank = {tid: i for i, tid in enumerate(ordered_ids)}
    # Assign ranks in baseline ordering
    baseline_rank = {tid: i for i, tid in enumerate(baseline_ids)}

    # Build paired rank arrays in a consistent key order
    all_ids = sorted(set(ordered_ids))
    ranks_curriculum = [curriculum_rank[tid] for tid in all_ids]
    ranks_baseline = [baseline_rank[tid] for tid in all_ids]

    try:
        from scipy.stats import kendalltau
        tau, p_value = kendalltau(ranks_curriculum, ranks_baseline)
        return float(tau), float(p_value)
    except ImportError:
        # Pure-Python fallback
        return _kendall_tau_fallback(ranks_curriculum, ranks_baseline)


def _kendall_tau_fallback(x: List[int], y: List[int]) -> Tuple[float, float]:
    """Pure-Python Kendall tau-b computation.

    Counts concordant and discordant pairs. Does not handle ties
    (ranks are assumed unique, which holds for permutation orderings).

    Args:
        x: First rank vector.
        y: Second rank vector.

    Returns:
        (tau, p_value): Kendall tau-b and approximate p-value.
    """
    import math
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
            # product == 0 is a tie; ignored in tau-a

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

    # Two-sided p-value from standard normal
    # Using complementary error function approximation
    p_value = 2.0 * _norm_sf(abs(z))

    return tau, p_value


def _norm_sf(z: float) -> float:
    """Survival function of the standard normal distribution (approximation).

    Uses Abramowitz and Stegun approximation 7.1.26, accurate to 1e-5.
    """
    import math
    # Ensure z >= 0
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
```

### Validation Protocol

At each epoch:

1. Generate a random baseline ordering using a deterministic seed derived from `(global_seed, epoch, 'baseline')`.
2. Compute Kendall tau between the curriculum ordering and the baseline.
3. Log both tau and p_value.
4. Assert that for non-trivial curricula (strategy != 'none'), tau is significantly different from 0 (p < 0.05) for at least 80% of epochs after warmup. If this condition fails consistently, the curriculum is not providing a meaningful ordering signal.

```python
def validate_curriculum_effect(
    curriculum_log: List[Dict],
    min_significant_fraction: float = 0.8,
) -> bool:
    """Validate that the curriculum produces statistically significant orderings.

    Args:
        curriculum_log: List of per-epoch log entries (from log_curriculum_epoch).
        min_significant_fraction: Minimum fraction of post-warmup epochs
            that must have p < 0.05.

    Returns:
        True if the curriculum effect is validated.
    """
    significant_count = 0
    total_count = 0

    for entry in curriculum_log:
        if entry['strategy'] == 'none':
            continue  # Skip random epochs (warmup)
        total_count += 1
        if entry['kendall_p_value'] < 0.05:
            significant_count += 1

    if total_count == 0:
        return True  # No curriculum epochs to validate

    fraction = significant_count / total_count
    return fraction >= min_significant_fraction
```


## Difficulty Proxy Computation

Unified entry point for computing the difficulty proxy given a proxy name.

```python
def compute_difficulty(
    task_ids: List[str],
    proxy_name: str,
    registry: 'TaskEmbeddingRegistry',
    cluster_ctx: 'ClusterContext',
) -> Dict[str, float]:
    """Compute difficulty proxy values for a set of tasks.

    Args:
        task_ids: List of task identifiers.
        proxy_name: One of 'centroid_distance', 'adaptation_gain', 'loss_slope'.
        registry: TaskEmbeddingRegistry for metadata and embeddings.
        cluster_ctx: ClusterContext with assignments, centroids, easy_cluster_id.

    Returns:
        Mapping from task_id to difficulty score (higher = harder).

    Raises:
        ValueError: If proxy_name is unknown.
    """
    if proxy_name == 'centroid_distance':
        task_embeddings = {tid: registry.get_embedding(tid) for tid in task_ids}
        return compute_centroid_distance(
            task_embeddings=task_embeddings,
            cluster_assignments=cluster_ctx.assignments,
            cluster_centroids=cluster_ctx.centroids,
            easy_cluster_id=cluster_ctx.easy_cluster_id,
        )

    elif proxy_name == 'adaptation_gain':
        return compute_adaptation_gain_difficulty(task_ids, registry)

    elif proxy_name == 'loss_slope':
        return compute_loss_slope_difficulty(task_ids, registry)

    else:
        raise ValueError(
            f"Unknown difficulty proxy: '{proxy_name}'. "
            f"Choose from 'centroid_distance', 'adaptation_gain', 'loss_slope'."
        )
```


## Implementation Patterns

### Full `get_curriculum_order` Pipeline

Complete pipeline from task IDs to ordered output, including warmup, difficulty computation, strategy application, and logging.

```python
def get_curriculum_order_full(
    task_ids: List[str],
    strategy: str,
    seed: int,
    epoch: int,
    max_epochs: int,
    registry: 'TaskEmbeddingRegistry',
    cluster_ctx: 'ClusterContext',
    config: 'CurriculumConfig',
    logger,
) -> List[str]:
    """Full curriculum ordering pipeline.

    Steps:
        1. Check warmup phase; if active, return random ordering.
        2. Compute difficulty proxy for all tasks.
        3. Apply strategy-specific ordering.
        4. Generate random baseline for Kendall tau.
        5. Log all curriculum metadata.
        6. Return ordered task list.

    Args:
        task_ids: Full list of available task IDs.
        strategy: Curriculum strategy name.
        seed: Global seed.
        epoch: Current epoch number (0-indexed).
        max_epochs: Total number of training epochs.
        registry: TaskEmbeddingRegistry.
        cluster_ctx: ClusterContext.
        config: CurriculumConfig.
        logger: Logger instance for curriculum logging.

    Returns:
        Ordered list of task IDs for this epoch.
    """
    # Step 1: Warmup check
    if epoch < config.warmup_epochs:
        rng = np.random.RandomState(_epoch_seed(seed, epoch))
        shuffled = list(task_ids)
        rng.shuffle(shuffled)
        log_curriculum_epoch(
            epoch=epoch,
            strategy='warmup_random',
            ordered_task_ids=shuffled,
            difficulty={tid: 0.0 for tid in task_ids},
            cluster_assignments=cluster_ctx.assignments,
            random_baseline_ids=shuffled,
            logger=logger,
        )
        return shuffled

    # Step 2: Compute difficulty
    difficulty = compute_difficulty(
        task_ids, config.difficulty_proxy, registry, cluster_ctx,
    )

    # Check for zero variance (all tasks equally difficult)
    diff_values = list(difficulty.values())
    if len(set(round(v, 8) for v in diff_values)) <= 1:
        logger.warning(
            f"Difficulty proxy '{config.difficulty_proxy}' has zero variance. "
            f"Falling back to random ordering."
        )
        rng = np.random.RandomState(_epoch_seed(seed, epoch))
        shuffled = list(task_ids)
        rng.shuffle(shuffled)
        log_curriculum_epoch(
            epoch=epoch,
            strategy='fallback_random',
            ordered_task_ids=shuffled,
            difficulty=difficulty,
            cluster_assignments=cluster_ctx.assignments,
            random_baseline_ids=shuffled,
            logger=logger,
        )
        return shuffled

    # Step 3: Apply strategy
    if strategy == 'easy_to_hard':
        ordered = easy_to_hard_order(task_ids, difficulty)
    elif strategy == 'anti_curriculum':
        ordered = hard_first_order(task_ids, difficulty)
    elif strategy == 'mixed':
        ordered = mixed_schedule_order(task_ids, difficulty, epoch, config, seed)
    else:
        raise ValueError(f"Unknown curriculum strategy: {strategy}")

    # Step 4: Generate random baseline for Kendall tau
    rng = np.random.RandomState(_epoch_seed(seed, epoch))
    random_baseline = list(task_ids)
    rng.shuffle(random_baseline)

    # Step 5: Log
    log_curriculum_epoch(
        epoch=epoch,
        strategy=strategy,
        ordered_task_ids=ordered,
        difficulty=difficulty,
        cluster_assignments=cluster_ctx.assignments,
        random_baseline_ids=random_baseline,
        logger=logger,
    )

    # Step 6: Return
    return ordered
```

### Full `build_meta_batch` Pipeline

Complete meta-batch construction with diversity enforcement and logging.

```python
def build_meta_batch_full(
    task_ids: List[str],
    diversity_constraint: str,
    seed: int,
    epoch: int,
    batch_idx: int,
    registry: 'TaskEmbeddingRegistry',
    cluster_ctx: 'ClusterContext',
    config: 'CurriculumConfig',
    logger,
) -> List[str]:
    """Full meta-batch construction pipeline.

    Steps:
        1. Retrieve embeddings and cluster assignments.
        2. Apply diversity constraint to select batch.
        3. Verify batch size.
        4. Log batch composition.
        5. Return selected task IDs.

    Args:
        task_ids: Pool of candidate task IDs for this batch.
        diversity_constraint: 'min_distance', 'stratified', or 'none'.
        seed: Global seed.
        epoch: Current epoch number.
        batch_idx: Batch index within the epoch (for seed derivation).
        registry: TaskEmbeddingRegistry.
        cluster_ctx: ClusterContext.
        config: CurriculumConfig.
        logger: Logger instance.

    Returns:
        List of task IDs for this meta-batch.
    """
    # Step 1: Retrieve embeddings
    task_embeddings = {}
    for tid in task_ids:
        emb = registry.get_embedding(tid)
        if emb is not None:
            task_embeddings[tid] = emb

    # Derive batch-specific seed
    batch_seed = _epoch_seed(seed, epoch * 10000 + batch_idx)

    # Step 2: Apply diversity constraint
    batch = build_meta_batch(
        task_ids=list(task_embeddings.keys()),
        diversity_constraint=diversity_constraint,
        seed=batch_seed,
        epoch=epoch,
        task_embeddings=task_embeddings,
        cluster_assignments=cluster_ctx.assignments,
        config=config,
    )

    # Step 3: Verify batch size
    assert len(batch) == config.meta_batch_size, (
        f"Meta-batch size mismatch: got {len(batch)}, "
        f"expected {config.meta_batch_size}"
    )

    # Step 4: Log batch composition
    batch_clusters = [cluster_ctx.assignments.get(tid, -1) for tid in batch]
    cluster_hist = defaultdict(int)
    for cid in batch_clusters:
        cluster_hist[cid] += 1

    if task_embeddings and len(batch) > 1:
        # Compute mean pairwise cosine distance within batch
        batch_embs = [task_embeddings[tid] for tid in batch if tid in task_embeddings]
        pairwise_dists = []
        for i in range(len(batch_embs)):
            for j in range(i + 1, len(batch_embs)):
                d = 1.0 - np.dot(batch_embs[i], batch_embs[j])
                pairwise_dists.append(d)
        mean_dist = float(np.mean(pairwise_dists)) if pairwise_dists else 0.0
        min_dist = float(np.min(pairwise_dists)) if pairwise_dists else 0.0
    else:
        mean_dist = 0.0
        min_dist = 0.0

    logger.debug(
        f"Meta-batch epoch={epoch} idx={batch_idx}: "
        f"clusters={dict(cluster_hist)}, "
        f"mean_pairwise_dist={mean_dist:.4f}, "
        f"min_pairwise_dist={min_dist:.4f}"
    )

    # Step 5: Return
    return batch
```

### Epoch-Level Curriculum Application in Training Loop

Integrate curriculum ordering and diversity-constrained batching into the meta-training loop.

```python
def meta_train_with_curriculum(
    model,
    meta_algorithm,
    registry: 'TaskEmbeddingRegistry',
    cluster_ctx: 'ClusterContext',
    config,
    num_epochs: int,
    logger,
):
    """Meta-training loop with curriculum ordering and diversity batches.

    Pseudocode showing how curriculum functions integrate into the
    standard meta-training loop.
    """
    all_task_ids = registry.all_task_ids()

    for epoch in range(num_epochs):
        # 1. Get curriculum-ordered task list for this epoch
        ordered_tasks = get_curriculum_order_full(
            task_ids=all_task_ids,
            strategy=config.curriculum.strategy,
            seed=config.seed,
            epoch=epoch,
            max_epochs=num_epochs,
            registry=registry,
            cluster_ctx=cluster_ctx,
            config=config.curriculum,
            logger=logger,
        )

        # 2. Iterate over ordered tasks in meta-batch-sized chunks
        batch_size = config.curriculum.meta_batch_size
        num_batches = (len(ordered_tasks) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(ordered_tasks))
            batch_task_ids = ordered_tasks[start:end]

            # 3. Apply diversity constraint within the batch
            if config.curriculum.diversity_constraint != 'none':
                batch_task_ids = build_meta_batch_full(
                    task_ids=batch_task_ids,
                    diversity_constraint=config.curriculum.diversity_constraint,
                    seed=config.seed,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    registry=registry,
                    cluster_ctx=cluster_ctx,
                    config=config.curriculum,
                    logger=logger,
                )

            # 4. Load episodes for the selected tasks
            episodes = [registry.load_episode(tid) for tid in batch_task_ids]

            # 5. Run meta-learning step
            output = meta_algorithm.meta_step(model, episodes, config)

            # 6. Update registry with adaptation metrics (for dynamic proxies)
            for tid, ep_metrics in zip(batch_task_ids, output.per_task_metrics):
                registry.update_metadata(tid, {
                    'adaptation_gain': ep_metrics.get('adaptation_gain'),
                    'inner_loss_curve': ep_metrics.get('inner_loss_curve'),
                })

        # 7. End-of-epoch: update cluster context if needed
        #    (re-clustering with updated embeddings, etc.)
        logger.info(f"Epoch {epoch}/{num_epochs} complete.")
```


## Configuration Reference

All curriculum parameters are controlled by `CurriculumConfig` in `brain_ai/config.py`:

| Field | Type | Default | Description |
|---|---|---|---|
| `strategy` | `str` | `"none"` | Curriculum strategy: `"none"`, `"easy_to_hard"`, `"anti_curriculum"`, `"mixed"` |
| `difficulty_proxy` | `str` | `"centroid_distance"` | Difficulty measure: `"centroid_distance"`, `"adaptation_gain"`, `"loss_slope"` |
| `diversity_constraint` | `str` | `"none"` | Meta-batch diversity: `"none"`, `"min_distance"`, `"stratified"` |
| `diversity_min_distance` | `float` | `0.3` | Minimum cosine distance threshold for `min_distance` constraint |
| `meta_batch_size` | `int` | `8` | Target number of tasks per meta-batch |
| `schedule_fn` | `str` | `"linear"` | Schedule function for mixed strategy: `"linear"`, `"cosine"`, `"step"` |
| `max_epochs` | `int` | `200` | Total training epochs (used by schedule functions) |
| `warmup_epochs` | `int` | `0` | Number of initial epochs with random ordering |
| `log_task_order` | `bool` | `True` | Whether to log the full task order or just its hash |
| `clusters_per_epoch` | `int` | `4` | Number of clusters per epoch for alternating cluster strategy |


## Testing Patterns

### Test 1: Curriculum Changes Ordering

Verify that the curriculum strategy produces an ordering different from random, validated by Kendall tau.

```python
def test_curriculum_changes_ordering():
    """Curriculum ordering must differ significantly from random."""
    # Create 50 synthetic task embeddings in 2 clusters
    task_ids, embeddings, clusters = make_synthetic_tasks(n=50, n_clusters=2, dim=64)
    registry = build_test_registry(task_ids, embeddings, clusters)
    cluster_ctx = build_test_cluster_ctx(task_ids, embeddings, clusters)

    config = CurriculumConfig(
        strategy='easy_to_hard',
        difficulty_proxy='centroid_distance',
        warmup_epochs=0,
    )

    ordered = get_curriculum_order(
        task_ids=task_ids,
        strategy='easy_to_hard',
        epoch=0,
        seed=42,
        registry=registry,
        cluster_ctx=cluster_ctx,
        config=config,
    )

    # Random baseline
    rng = np.random.RandomState(99)
    random_order = list(task_ids)
    rng.shuffle(random_order)

    tau, p_value = compute_kendall_tau(ordered, random_order)
    assert abs(tau) > 0.1, f"Kendall tau too small: {tau}"
    assert p_value < 0.05, f"Kendall tau not significant: p={p_value}"
```

### Test 2: Easy-to-Hard Monotonicity

Verify that the difficulty proxy values are monotonically non-decreasing in the curriculum ordering.

```python
def test_easy_to_hard_monotonicity():
    """Easy-to-hard ordering must have non-decreasing difficulty."""
    task_ids, embeddings, clusters = make_synthetic_tasks(n=30, n_clusters=3, dim=64)
    registry = build_test_registry(task_ids, embeddings, clusters)
    cluster_ctx = build_test_cluster_ctx(task_ids, embeddings, clusters)

    difficulty = compute_centroid_distance(
        {tid: embeddings[tid] for tid in task_ids},
        cluster_ctx.assignments,
        cluster_ctx.centroids,
        cluster_ctx.easy_cluster_id,
    )

    ordered = easy_to_hard_order(task_ids, difficulty)

    for i in range(1, len(ordered)):
        d_prev = difficulty[ordered[i - 1]]
        d_curr = difficulty[ordered[i]]
        assert d_curr >= d_prev - 1e-9, (
            f"Difficulty decreased at position {i}: {d_prev:.4f} -> {d_curr:.4f}"
        )
```

### Test 3: Diversity Minimum Distance

Verify that the min-distance meta-batch builder respects the distance threshold.

```python
def test_diversity_min_distance():
    """All pairwise distances in the batch must exceed the threshold (or relaxed)."""
    task_ids, embeddings, _ = make_synthetic_tasks(n=50, n_clusters=5, dim=64)
    threshold = 0.2

    batch = build_meta_batch_distance(
        task_ids=task_ids,
        task_embeddings=embeddings,
        batch_size=8,
        min_distance=threshold,
        seed=42,
        epoch=0,
    )

    assert len(batch) == 8, f"Batch size mismatch: {len(batch)}"

    # Check pairwise distances
    for i in range(len(batch)):
        for j in range(i + 1, len(batch)):
            d = 1.0 - np.dot(embeddings[batch[i]], embeddings[batch[j]])
            # Allow some relaxation (threshold may have been halved)
            assert d >= threshold * 0.25, (
                f"Pairwise distance too small: d({batch[i]}, {batch[j]}) = {d:.4f}"
            )
```

### Test 4: Stratified Sampling Covers All Clusters

```python
def test_stratified_covers_clusters():
    """Stratified sampling must include at least one task from each cluster
    when batch_size >= n_clusters."""
    task_ids, _, clusters = make_synthetic_tasks(n=40, n_clusters=4, dim=64)
    cluster_assignments = {tid: clusters[tid] for tid in task_ids}

    batch = build_meta_batch_stratified(
        task_ids=task_ids,
        cluster_assignments=cluster_assignments,
        batch_size=8,
        seed=42,
        epoch=0,
    )

    represented_clusters = set(cluster_assignments[tid] for tid in batch)
    all_clusters = set(cluster_assignments.values())
    assert represented_clusters == all_clusters, (
        f"Missing clusters: {all_clusters - represented_clusters}"
    )
```

### Test 5: Schedule Function Boundary Values

```python
def test_schedule_functions():
    """Schedule functions must return 0 at epoch 0 and 1 at max_epoch."""
    max_epoch = 100

    for name in ['linear', 'cosine', 'step']:
        fn = get_schedule_fn(name)
        val_start = fn(0, max_epoch)
        val_end = fn(max_epoch, max_epoch)

        assert val_start <= 0.01, f"{name}: p(hard) at epoch 0 should be ~0, got {val_start}"
        assert val_end >= 0.99, f"{name}: p(hard) at max_epoch should be ~1, got {val_end}"

    # Cosine should be 0.5 at midpoint
    mid_val = schedule_cosine(50, 100)
    assert abs(mid_val - 0.5) < 0.01, f"Cosine midpoint should be ~0.5, got {mid_val}"
```

### Test 6: Determinism

```python
def test_curriculum_determinism():
    """Same inputs must produce identical orderings."""
    task_ids, embeddings, clusters = make_synthetic_tasks(n=50, n_clusters=5, dim=64)
    registry = build_test_registry(task_ids, embeddings, clusters)
    cluster_ctx = build_test_cluster_ctx(task_ids, embeddings, clusters)
    config = CurriculumConfig(strategy='easy_to_hard', difficulty_proxy='centroid_distance')

    order1 = get_curriculum_order(task_ids, 'easy_to_hard', 0, 42, registry, cluster_ctx, config)
    order2 = get_curriculum_order(task_ids, 'easy_to_hard', 0, 42, registry, cluster_ctx, config)

    assert order1 == order2, "Curriculum ordering is not deterministic"
```

### Test 7: Kendall Tau Fallback

```python
def test_kendall_tau_fallback():
    """Pure-Python Kendall tau must match scipy within tolerance."""
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    y = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]  # Perfect reversal

    tau_fallback, _ = _kendall_tau_fallback(x, y)
    assert abs(tau_fallback - (-1.0)) < 1e-6, (
        f"Expected tau=-1.0 for reversed ordering, got {tau_fallback}"
    )

    # Identity ordering
    tau_identity, _ = _kendall_tau_fallback(x, x)
    assert abs(tau_identity - 1.0) < 1e-6, (
        f"Expected tau=1.0 for identical ordering, got {tau_identity}"
    )

    # Compare with scipy if available
    try:
        from scipy.stats import kendalltau
        tau_scipy, _ = kendalltau(x, y)
        assert abs(tau_fallback - tau_scipy) < 1e-6, (
            f"Fallback tau={tau_fallback} differs from scipy tau={tau_scipy}"
        )
    except ImportError:
        pass  # Scipy not available; fallback-only test sufficient
```


## Common Failure Modes

| Symptom | Cause | Fix |
|---|---|---|
| Kendall tau near 0 every epoch | Difficulty proxy has no variance | Check that embeddings are diverse; verify probe is not trivial; try a different proxy |
| All tasks assigned same difficulty | Easy cluster centroid is equidistant from all tasks | Verify clustering quality (silhouette > 0.3); check if embeddings collapsed |
| Diversity batch always falls back to random | Min distance threshold too high for the embedding space | Lower `diversity_min_distance`; check embedding spread with histogram of pairwise distances |
| Mixed schedule shows no effect | Schedule function returns constant value | Verify `max_epochs` is set correctly; check schedule function registration |
| Curriculum improves early but hurts late | Fixed easy-to-hard ordering starves hard-task exposure early | Switch to mixed schedule or add warmup; consider anti-curriculum after plateau |
| Stratified sampling produces duplicates | Some clusters have very few tasks | Increase task pool size or reduce batch size; monitor cluster sizes |
| Non-deterministic ordering across runs | Using Python `hash()` without `PYTHONHASHSEED=0` | Use `hashlib.sha256` for seed derivation; set `PYTHONHASHSEED=0` |


## Anti-Patterns

- **Curriculum without logging** -- Every ordering decision must be logged with strategy name, order hash, and Kendall tau. Unlogged curricula cannot be debugged or reproduced.
- **Hardcoded difficulty thresholds** -- All thresholds (easy/hard split, distance minimums) must come from `CurriculumConfig`, not magic numbers in code.
- **Recomputing embeddings per epoch** -- Embeddings are pre-computed and stored in the registry. Curriculum functions read from the registry, never invoke the Task2VecExtractor.
- **Ignoring zero-variance fallback** -- When all tasks have identical difficulty, the curriculum is meaningless. Detect and fall back to random with a warning, rather than sorting a constant sequence.
- **Modifying task order after logging** -- The logged order must match the actual presentation order. Never reorder tasks between the logging call and the training loop consumption.
- **Using global RNG for curriculum** -- All randomness in curriculum functions (shuffles, sampling) must derive from deterministic per-epoch seeds, never from `np.random` or `random` global state.
