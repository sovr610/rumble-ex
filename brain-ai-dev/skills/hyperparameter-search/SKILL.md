---
name: Hyperparameter Search
description: >
  This skill should be used when the user asks to "tune hyperparameters",
  "find optimal learning rate", "run grid search", "run random search",
  "Bayesian optimization", "learning rate finder", "hyperparameter sweep",
  "tune model", "search space", "parameter optimization", "Optuna",
  "Ray Tune", "hyperband", "population based training", or needs guidance on
  hyperparameter search strategies, search space definition, trial management,
  or automated tuning for the brain_ai system.
version: 0.1.0
---

# Hyperparameter Search

## Overview

Guide implementation of hyperparameter search infrastructure for the brain_ai system. The 10-dataclass configuration system (`BrainAIConfig`) creates a vast search space. Cover grid search, random search, Bayesian optimization (via Optuna), learning rate finders, search space definition, trial management, early stopping (Hyperband/ASHA), and result analysis. Integrate with the training-orchestrator's manifest system for reproducibility.

## Public Contract

### SearchEngine

Unified interface for all search strategies.

```python
class SearchEngine:
    def __init__(self, config: SearchConfig): ...
    def search(self, objective_fn: Callable[[Dict], float], space: SearchSpace) -> SearchResult: ...
    def resume(self, study_path: str) -> SearchResult: ...
    def best_params(self) -> Dict[str, Any]: ...
    def best_value(self) -> float: ...
```

### SearchSpace

Declarative search space definition matching BrainAIConfig structure.

```python
class SearchSpace:
    def __init__(self): ...
    def add_float(self, name: str, low: float, high: float, log: bool = False): ...
    def add_int(self, name: str, low: int, high: int): ...
    def add_categorical(self, name: str, choices: List[Any]): ...
    def add_conditional(self, name: str, condition: str, space: "SearchSpace"): ...
    def from_config_class(self, config_cls: type, overrides: Dict) -> "SearchSpace": ...
    def sample(self) -> Dict[str, Any]: ...
```

### LearningRateFinder

Smith LR range test for finding optimal learning rate bounds.

```python
class LearningRateFinder:
    def __init__(self, model: nn.Module, optimizer_cls: type, config: LRFinderConfig): ...
    def find(self, train_loader: DataLoader, criterion: Callable) -> LRFinderResult: ...
    def plot(self) -> Figure: ...
    def suggested_lr(self) -> Tuple[float, float]: ...  # (min_lr, max_lr)
```

### TrialManager

Track, checkpoint, and analyze search trials.

```python
class TrialManager:
    def __init__(self, study_dir: str): ...
    def log_trial(self, params: Dict, metrics: Dict, trial_id: str) -> None: ...
    def get_trial(self, trial_id: str) -> TrialRecord: ...
    def get_best_trials(self, n: int = 5) -> List[TrialRecord]: ...
    def get_importance(self) -> Dict[str, float]: ...
    def export_csv(self, path: str) -> None: ...
    def plot_optimization_history(self) -> Figure: ...
    def plot_param_importances(self) -> Figure: ...
```

## Key Concepts

### Search Strategies

| Strategy | When to Use | Trials Needed | Quality |
|----------|------------|--------------|---------|
| Grid Search | <5 params, discrete values | Exhaustive | Thorough but slow |
| Random Search | >5 params, first exploration | 50-200 | Good baseline |
| Bayesian (TPE) | Expensive objective, focused tuning | 20-100 | Best for budget |
| Hyperband/ASHA | Large space, early stopping | 100+ (cheap) | Resource efficient |
| Population-Based | Long training, scheduling | 10-50 parallel | Dynamic schedules |

### Critical Hyperparameters by Phase

| Phase | Key Hyperparameters | Typical Range |
|-------|-------------------|---------------|
| 1 SNN | lr, threshold, tau_mem, surrogate_slope | lr: 1e-4–1e-2, tau: 5–50ms |
| 2 Encoders | lr, hidden_dim, dropout, warmup_steps | lr: 1e-5–1e-3 |
| 3 HTM | n_columns, cells_per_column, permanence_inc | columns: 512–4096 |
| 4 Workspace | n_competitors, broadcast_dim, temperature | temp: 0.1–2.0 |
| 5 Active Inf. | efe_weight, planning_horizon, lr | horizon: 1–10 |
| 6 Reasoning | confidence_threshold, max_system2_steps | threshold: 0.5–0.9 |
| 7 Meta | inner_lr, inner_steps, meta_lr | inner_lr: 0.01–0.5 |

### Integration with Training Orchestrator

- Each trial creates a run manifest via the training-orchestrator's `RunManifest`
- Trial params are recorded in the manifest's `hyperparameters` section
- All trials use deterministic seeding from the manifest's `SeedConfig`
- Results are stored in `runs/<study_id>/trials/<trial_id>/`

### Early Stopping Protocol

Successive Halving (ASHA):
1. Start N trials with minimum budget (e.g., 1 epoch)
2. Keep top 1/η fraction (default η=3)
3. Continue survivors with 3× budget
4. Repeat until 1 trial remains or max budget reached

## Configuration Surface

```python
@dataclass
class SearchConfig:
    strategy: str = "bayesian"           # grid | random | bayesian | hyperband
    n_trials: int = 100
    metric: str = "val_accuracy"
    direction: str = "maximize"          # maximize | minimize
    # Bayesian
    sampler: str = "tpe"                 # tpe | cmaes | random
    # Early stopping
    enable_pruning: bool = True
    pruner: str = "asha"                 # asha | hyperband | median
    min_resource: int = 1                # Minimum epochs before pruning
    reduction_factor: int = 3
    # Storage
    study_dir: str = "studies/"
    study_name: str = "brain_ai_search"
    # Parallelism
    n_jobs: int = 1                      # Parallel trials
    timeout_per_trial: int = 3600        # seconds

@dataclass
class LRFinderConfig:
    start_lr: float = 1e-7
    end_lr: float = 10.0
    num_steps: int = 100
    smooth_factor: float = 0.05
    divergence_threshold: float = 5.0
```

## Done-When Gates

1. **Search Execution** — `SearchEngine.search()` with random strategy completes 10 trials on a minimal-config model, returning `best_params()` that improve over the first trial's metric.
2. **LR Finder** — `LearningRateFinder.find()` produces a smooth loss-vs-LR curve; `suggested_lr()` returns reasonable bounds (not NaN, not at extremes).
3. **Trial Management** — `TrialManager.export_csv()` produces valid CSV with all trial params and metrics; `get_importance()` returns non-trivial importance scores for >1 parameter.

## Failure Modes

| Mode | Symptom | Fix |
|------|---------|-----|
| Search space too large | Trials never converge | Narrow ranges based on LR finder or domain knowledge |
| All trials pruned | No completed trials | Increase min_resource or disable pruning initially |
| Objective NaN | Bayesian sampler confused | Add NaN handling in objective; report as worst value |
| Storage full | Study database grows large | Set n_trials limit; prune old studies |
| Parallelism conflicts | GPU contention | Assign trials to specific GPUs; use n_jobs=1 per GPU |

## Anti-Patterns

- Searching all parameters at once — fix most, search 3-5 at a time
- Skipping LR finder — always start with LR range test
- No early stopping — wastes compute on clearly bad configurations
- Ignoring parameter importance — focus budget on high-importance params
- Not recording trial manifests — breaks reproducibility

## Resources

### Reference Files
- **`references/search-strategies.md`** — Grid, random, Bayesian, Hyperband algorithms in depth
- **`references/lr-finder.md`** — Smith LR range test, cyclic LR, warmup strategies
- **`references/bayesian-optimization.md`** — TPE, CMA-ES, Gaussian processes, acquisition functions
- **`references/search-spaces.md`** — Per-phase search spaces, conditional params, constraints
- **`references/testing-matrix.md`** — Test scenarios for search infrastructure

### Asset Files
- **`assets/search_engine_template.py`** — SearchEngine with grid/random/Bayesian backends
- **`assets/search_space_template.py`** — SearchSpace with config integration, self-tests
- **`assets/lr_finder_template.py`** — LearningRateFinder with plotting and suggestion
- **`assets/trial_manager_template.py`** — TrialManager with CSV export, importance analysis
- **`assets/early_stopping_template.py`** — ASHA, Hyperband, median pruner implementations
- **`assets/search_config_template.py`** — All config dataclasses + validation

### Scripts
- **`scripts/validate_search.py`** — Validates search infrastructure against done-when gates
- **`scripts/gen_search_tests.py`** — Generates 100+ pytest test cases
- **`scripts/search_benchmark.py`** — Benchmarks search efficiency (trials/hour, convergence speed)
