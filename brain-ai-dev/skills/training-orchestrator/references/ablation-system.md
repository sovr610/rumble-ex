# Ablation System Reference

This document specifies the automatic ablation system for the brain_ai training pipeline. It covers the YAML spec DSL, matrix generation algorithms (full and pairwise), run ID derivation, execution strategies, CSV output schema, report generation, and comparison utilities. All ablation logic lives in `brain_ai/training/ablation.py` and is invoked via `scripts/run_ablation.py` or programmatically through the `AblationRunner` API.

---

## 1. Ablation Spec DSL

Define ablation experiments as YAML files. Each spec declares the toggles to vary, the valid value ranges, constraints that prune invalid combinations, and execution parameters.

### Full Spec Format

```yaml
ablation:
  name: "workspace_engram_ablation"
  description: "Test workspace and engram module interactions"
  base_config: "configs/production_3b.yaml"  # path to YAML config or preset name ("dev", "minimal")
  phases: [4, 5]  # which training phases to run for each combination
  baseline_run_id: null  # optional: reuse Phase 1-3 boundary from an existing run

  toggles:
    use_engram: [false, true]
    engram_mode: ["encoder", "layer"]  # only meaningful when use_engram=true
    use_learnable_delays: [false, true]
    use_ltn: [false, true]
    workspace_ignition: [false, true]

  constraints:
    - if: {use_engram: false}
      then: {engram_mode: null}  # skip engram_mode permutations when engram is disabled

  mode: "full"  # "full" = Cartesian product, "pairwise" = covering array reduction
  seeds: [1337, 42, 7]  # run each valid combination with each seed for statistical power

  metrics:
    primary: "best_val_loss"  # metric used for ranking combinations
    secondary: ["best_val_acc", "final_train_loss", "final_val_loss"]

  resource_limits:
    max_concurrent: 4  # for parallel execution
    timeout_per_run: 86400  # seconds (24h)
    gpu_ids: [0, 1, 2, 3]  # available GPUs for parallel dispatch
```

### Spec Field Reference

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | string | yes | -- | Human-readable ablation name; used in ablation_id |
| `description` | string | no | `""` | Free-text description stored in reports |
| `base_config` | string | yes | -- | Path to base config YAML or preset name |
| `phases` | list[int] | yes | -- | Training phases to execute per combination |
| `baseline_run_id` | string | no | `null` | Existing run whose phase boundary to reuse |
| `toggles` | dict[str, list] | yes | -- | Toggle names mapped to lists of values |
| `constraints` | list[dict] | no | `[]` | If-then rules for pruning invalid combos |
| `mode` | string | no | `"full"` | `"full"` or `"pairwise"` |
| `seeds` | list[int] | no | `[1337]` | Seeds for multi-run statistical analysis |
| `metrics.primary` | string | no | `"best_val_loss"` | Primary metric for ranking |
| `metrics.secondary` | list[str] | no | `[]` | Additional metrics to record |
| `resource_limits` | dict | no | `{}` | Concurrency, timeout, GPU constraints |

### Toggle Naming Convention

Toggle names must correspond to fields in `BrainAIConfig` or its nested dataclass configs. Use dot notation for nested fields:

```yaml
toggles:
  use_engram: [false, true]                 # BrainAIConfig.use_engram
  snn.use_learnable_delays: [false, true]   # BrainAIConfig.snn.use_learnable_delays
  reasoning.use_ltn: [false, true]          # BrainAIConfig.reasoning.use_ltn
  workspace.ignition_threshold: [0.2, 0.3, 0.5]  # BrainAIConfig.workspace.ignition_threshold
```

The ablation runner resolves each toggle name against `BrainAIConfig` via `getattr` chains and raises `ValueError` for unknown fields.

---

## 2. Matrix Generation

### Full Mode (Cartesian Product)

Compute the Cartesian product of all toggle value lists, then filter through constraints.

```python
import itertools
import json
import hashlib

def generate_full_matrix(spec: dict) -> list[dict]:
    """Generate all valid toggle combinations from an ablation spec."""
    toggles = spec["toggles"]
    constraints = spec.get("constraints", [])

    # Cartesian product of all toggle values
    keys = sorted(toggles.keys())
    value_lists = [toggles[k] for k in keys]
    raw_combos = [dict(zip(keys, vals)) for vals in itertools.product(*value_lists)]

    # Filter by constraints
    valid_combos = [c for c in raw_combos if passes_constraints(c, constraints)]
    return valid_combos


def passes_constraints(combo: dict, constraints: list[dict]) -> bool:
    """Evaluate if-then constraint rules against a combination."""
    for rule in constraints:
        if_clause = rule["if"]
        then_clause = rule["then"]

        # Check if the 'if' condition matches
        if all(combo.get(k) == v for k, v in if_clause.items()):
            for k, v in then_clause.items():
                if v is None:
                    # 'null' means this toggle is irrelevant -- skip combos
                    # where the toggle has any non-default value.
                    # Pin it to its first listed value to avoid duplicates.
                    if combo.get(k) != spec["toggles"][k][0]:
                        return False
                elif combo.get(k) != v:
                    return False
    return True
```

For the example spec above with 5 toggles (2x2x2x2x2 = 32 raw combinations), the constraint `if use_engram=false then engram_mode=null` pins `engram_mode` to its first value (`"encoder"`) whenever `use_engram=false`, reducing the matrix from 32 to 24 valid combinations.

### Multi-Seed Expansion

Multiply each valid combination by the seed list:

```python
def expand_with_seeds(combos: list[dict], seeds: list[int]) -> list[dict]:
    """Expand combinations across multiple seeds."""
    runs = []
    for combo in combos:
        for seed in seeds:
            run = combo.copy()
            run["__seed__"] = seed
            runs.append(run)
    return runs
```

Total runs = `len(valid_combos) * len(seeds)`. For 24 combos and 3 seeds, that is 72 runs.

### Run Count Warning

After matrix generation, check the total count and warn if it exceeds 100:

```python
total_runs = len(valid_combos) * len(seeds)
if total_runs > 100:
    logger.warning(
        f"Ablation matrix has {total_runs} runs (>{100}). "
        f"Consider switching to mode='pairwise' to reduce to ~{estimate_pairwise_size(toggles)} runs."
    )
```

### Pairwise Mode (Covering Array)

See Section 7 for the full algorithm. In pairwise mode, `generate_matrix` calls the covering array generator instead of the Cartesian product:

```python
def generate_matrix(spec: dict) -> list[dict]:
    if spec.get("mode", "full") == "pairwise":
        combos = generate_pairwise_matrix(spec)
    else:
        combos = generate_full_matrix(spec)
    return expand_with_seeds(combos, spec.get("seeds", [1337]))
```

---

## 3. Run ID Derivation

Every ablation and every individual run within it receives a deterministic, unique, sortable, human-readable identifier.

### Ablation ID

```
ablation_id = "{name}_{timestamp}_{short_hash}"
```

- `name`: the `ablation.name` field, lowercased and sanitized (alphanumeric + underscores)
- `timestamp`: `YYYYMMDD_HHMMSS` in UTC
- `short_hash`: first 8 characters of `SHA256(canonical_json(spec))`

```python
def derive_ablation_id(spec: dict) -> str:
    name = sanitize(spec["name"])
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    short_hash = hashlib.sha256(canonical.encode()).hexdigest()[:8]
    return f"{name}_{timestamp}_{short_hash}"
```

### Per-Run ID

```
run_id = "{ablation_id}_run{N:03d}_{overrides_hash}_{seed}"
```

- `N`: zero-padded run index (000, 001, 002, ...)
- `overrides_hash`: first 6 characters of `SHA256(canonical_json(sorted_overrides))`
- `seed`: the seed value for this run

```python
def derive_run_id(ablation_id: str, index: int, overrides: dict, seed: int) -> str:
    override_only = {k: v for k, v in sorted(overrides.items()) if k != "__seed__"}
    canonical = json.dumps(override_only, sort_keys=True, separators=(",", ":"))
    overrides_hash = hashlib.sha256(canonical.encode()).hexdigest()[:6]
    return f"{ablation_id}_run{index:03d}_{overrides_hash}_{seed}"
```

### Guarantees

| Property | Mechanism |
|---|---|
| Unique | Timestamp + spec hash + per-run override hash + seed |
| Deterministic | Same spec and same overrides always produce the same hash components |
| Sortable | Timestamp prefix enables chronological ordering; run index enables within-ablation ordering |
| Human-readable | Name prefix and seed suffix are directly interpretable |

---

## 4. Execution Strategies

### Sequential Execution

Run each combination one after another on a single device. Simple and low-resource.

```python
def execute_sequential(matrix: list[dict], spec: dict) -> list[RunResult]:
    results = []
    for i, run_config in enumerate(matrix):
        run_id = derive_run_id(ablation_id, i, run_config, run_config["__seed__"])
        try:
            result = run_single(spec, run_config, run_id)
            results.append(result)
        except Exception as e:
            results.append(RunResult(run_id=run_id, status="failed", error=str(e)))
            logger.error(f"Run {run_id} failed: {e}")
            # Continue with remaining runs -- do NOT abort the matrix
    return results
```

### Parallel Execution

Launch up to `max_concurrent` runs simultaneously. Requires resource management to prevent GPU memory contention.

```python
from concurrent.futures import ProcessPoolExecutor, as_completed

def execute_parallel(matrix: list[dict], spec: dict, max_concurrent: int = 4) -> list[RunResult]:
    results = []
    gpu_ids = spec.get("resource_limits", {}).get("gpu_ids", [0])

    with ProcessPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {}
        for i, run_config in enumerate(matrix):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            run_id = derive_run_id(ablation_id, i, run_config, run_config["__seed__"])
            future = executor.submit(run_single, spec, run_config, run_id, gpu_id=gpu_id)
            futures[future] = run_id

        for future in as_completed(futures):
            run_id = futures[future]
            try:
                results.append(future.result())
            except Exception as e:
                results.append(RunResult(run_id=run_id, status="failed", error=str(e)))
    return results
```

### Distributed Execution

For cluster environments, dispatch each combination to a separate node via a job scheduler (SLURM, Kubernetes). The ablation runner generates a job manifest:

```python
def generate_job_manifest(matrix: list[dict], spec: dict) -> list[dict]:
    """Generate per-run job descriptors for cluster submission."""
    jobs = []
    for i, run_config in enumerate(matrix):
        run_id = derive_run_id(ablation_id, i, run_config, run_config["__seed__"])
        jobs.append({
            "run_id": run_id,
            "command": build_run_command(spec, run_config, run_id),
            "resources": {"gpus": 1, "timeout": spec.get("resource_limits", {}).get("timeout_per_run", 86400)},
            "overrides": run_config,
        })
    return jobs
```

Submit these jobs with your cluster's job scheduler. Each job writes its results to the shared `ablations.csv`.

### Shared Upstream Phases

When the ablation only varies toggles that affect Phase 4 and later, phases 1-3 are identical across all combinations. Avoid re-running them:

1. Set `baseline_run_id` in the spec to point to an existing run that completed Phase 3.
2. The ablation runner validates that the baseline's Phase 3 boundary artifact is compatible with the current spec's `base_config`.
3. Each ablation run copies or symlinks the Phase 3 boundary checkpoint and starts directly at the first ablation phase.

```python
def validate_baseline(spec: dict) -> str:
    """Validate and return the Phase 3 boundary path from baseline_run_id."""
    baseline_id = spec.get("baseline_run_id")
    if baseline_id is None:
        return None

    boundary_path = f"runs/{baseline_id}/checkpoints/phase3/phase_boundary.pt"
    if not os.path.exists(boundary_path):
        raise FileNotFoundError(f"Baseline boundary not found: {boundary_path}")

    # Validate compatibility: workspace_dim, vocab_size, enabled modules
    boundary = torch.load(boundary_path, map_location="cpu")
    base_config = load_config(spec["base_config"])
    validate_boundary_compatibility(boundary, base_config)

    return boundary_path
```

Cost savings are substantial. For a 7-phase pipeline where phases 1-3 take 80% of total compute, sharing upstream saves ~80% of wall-clock time per ablation run.

### Failure Handling

The ablation runner never aborts the entire matrix on a single run failure. Instead:

1. Record the failure status, error message, and duration in `ablations.csv`.
2. Log the error at WARNING level.
3. Continue with the next run in the matrix.
4. At the end, report a failure summary (count, run IDs, error categories).

Runs can have four statuses: `completed`, `failed`, `interrupted` (caught SIGINT/SIGTERM), or `skipped` (constraint-pruned or dependency unavailable).

---

## 5. ablations.csv Schema

Every ablation produces a single CSV file at `runs/{ablation_id}/ablations.csv` with one row per run.

### Column Layout

```
run_id,ablation_id,parent_run_id,seed,use_engram,engram_mode,use_learnable_delays,use_ltn,workspace_ignition,phase,best_val_loss,best_val_acc,final_train_loss,final_val_loss,status,duration_seconds,git_sha,dataset_fingerprint,error_message
```

### Column Specification

| Column | Type | Source | Description |
|---|---|---|---|
| `run_id` | string | derived | Unique run identifier (Section 3) |
| `ablation_id` | string | derived | Parent ablation identifier |
| `parent_run_id` | string | spec | `baseline_run_id` if using shared upstream, else empty |
| `seed` | int | spec | Seed used for this run |
| *(toggle columns)* | varies | spec | One column per toggle in `toggles`, dynamically generated |
| `phase` | string | spec | Comma-separated phases executed (e.g., `"4,5"`) |
| `best_val_loss` | float | metrics | Best validation loss across all phases |
| `best_val_acc` | float | metrics | Best validation accuracy across all phases |
| `final_train_loss` | float | metrics | Final training loss of last phase |
| `final_val_loss` | float | metrics | Final validation loss of last phase |
| `status` | string | execution | `"completed"`, `"failed"`, `"interrupted"`, `"skipped"` |
| `duration_seconds` | float | execution | Wall-clock duration of the run |
| `git_sha` | string | manifest | Git commit SHA at run time |
| `dataset_fingerprint` | string | manifest | Dataset fingerprint hash |
| `error_message` | string | execution | Error message if `status != "completed"`, else empty |

### Dynamic Toggle Columns

Toggle columns are generated from the keys in `spec.toggles`. The column order matches sorted toggle names. Values are written as-is (booleans as `True`/`False`, strings quoted, numbers as literals).

### Writing the CSV

```python
import csv

def write_ablations_csv(ablation_id: str, results: list[RunResult], spec: dict):
    toggle_keys = sorted(spec["toggles"].keys())
    fieldnames = [
        "run_id", "ablation_id", "parent_run_id", "seed",
        *toggle_keys,
        "phase", "best_val_loss", "best_val_acc",
        "final_train_loss", "final_val_loss",
        "status", "duration_seconds", "git_sha",
        "dataset_fingerprint", "error_message",
    ]

    csv_path = f"runs/{ablation_id}/ablations.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {
                "run_id": r.run_id,
                "ablation_id": ablation_id,
                "parent_run_id": r.parent_run_id or "",
                "seed": r.seed,
                "phase": ",".join(str(p) for p in spec["phases"]),
                "best_val_loss": r.metrics.get("best_val_loss", ""),
                "best_val_acc": r.metrics.get("best_val_acc", ""),
                "final_train_loss": r.metrics.get("final_train_loss", ""),
                "final_val_loss": r.metrics.get("final_val_loss", ""),
                "status": r.status,
                "duration_seconds": r.duration_seconds,
                "git_sha": r.git_sha,
                "dataset_fingerprint": r.dataset_fingerprint,
                "error_message": r.error or "",
            }
            for k in toggle_keys:
                row[k] = r.overrides.get(k, "")
            writer.writerow(row)
```

---

## 6. Ablation Report Generation

After all runs complete, generate a structured summary report.

### Summary JSON

Write `runs/{ablation_id}/reports/ablation_summary.json`:

```json
{
  "ablation_id": "workspace_engram_ablation_20260220_143022_a1b2c3d4",
  "description": "Test workspace and engram module interactions",
  "total_runs": 72,
  "completed": 70,
  "failed": 2,
  "mode": "full",
  "primary_metric": "best_val_loss",

  "per_toggle_effects": {
    "use_engram": {
      "values": [false, true],
      "mean_metric": {"false": 0.342, "true": 0.298},
      "effect_size": -0.044,
      "effect_direction": "lower_is_better",
      "p_value": 0.003,
      "test": "wilcoxon",
      "significant": true
    },
    "use_ltn": {
      "values": [false, true],
      "mean_metric": {"false": 0.325, "true": 0.312},
      "effect_size": -0.013,
      "effect_direction": "lower_is_better",
      "p_value": 0.087,
      "test": "wilcoxon",
      "significant": false
    }
  },

  "best_combination": {
    "overrides": {"use_engram": true, "engram_mode": "layer", "use_learnable_delays": true, "use_ltn": true, "workspace_ignition": true},
    "mean_metric": 0.271,
    "std_metric": 0.008,
    "seeds": [1337, 42, 7],
    "run_ids": ["...run042...", "...run043...", "...run044..."]
  },

  "worst_combination": {
    "overrides": {"use_engram": false, "engram_mode": "encoder", "use_learnable_delays": false, "use_ltn": false, "workspace_ignition": false},
    "mean_metric": 0.387,
    "std_metric": 0.012
  },

  "failure_summary": {
    "count": 2,
    "run_ids": ["...run015...", "...run058..."],
    "errors": ["CUDA OOM at phase 5 step 2340", "NaN loss at phase 4 step 891"]
  }
}
```

### Statistical Tests

When multiple seeds are used, compute per-toggle effects with statistical significance:

```python
from scipy import stats

def compute_toggle_effects(df, toggle: str, metric: str) -> dict:
    """Compute effect size and significance for a single toggle."""
    values = df[toggle].unique()
    if len(values) != 2:
        # For non-binary toggles, use Kruskal-Wallis instead
        groups = [df[df[toggle] == v][metric].dropna().values for v in values]
        stat, p_value = stats.kruskal(*groups)
        test_name = "kruskal"
    else:
        group_a = df[df[toggle] == values[0]][metric].dropna().values
        group_b = df[df[toggle] == values[1]][metric].dropna().values
        if len(group_a) >= 3 and len(group_b) >= 3:
            stat, p_value = stats.wilcoxon_or_mannwhitneyu(group_a, group_b)
            test_name = "wilcoxon"
        else:
            stat, p_value = stats.ttest_ind(group_a, group_b)
            test_name = "t_test"

    mean_per_value = {str(v): float(df[df[toggle] == v][metric].mean()) for v in values}
    effect_size = mean_per_value[str(values[1])] - mean_per_value[str(values[0])]

    return {
        "values": [v for v in values],
        "mean_metric": mean_per_value,
        "effect_size": effect_size,
        "effect_direction": "lower_is_better",
        "p_value": float(p_value),
        "test": test_name,
        "significant": p_value < 0.05,
    }
```

### Plot Generation (Optional)

Generate bar charts showing per-toggle metric effects, saved as PNG:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_toggle_effects(summary: dict, output_dir: str):
    """Generate bar chart of per-toggle effect sizes."""
    effects = summary["per_toggle_effects"]
    toggles = sorted(effects.keys())
    sizes = [effects[t]["effect_size"] for t in toggles]
    significant = [effects[t]["significant"] for t in toggles]
    colors = ["#2196F3" if s else "#BDBDBD" for s in significant]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(toggles, sizes, color=colors)
    ax.set_xlabel(f"Effect on {summary['primary_metric']}")
    ax.set_title(f"Ablation: {summary['ablation_id']}")
    ax.axvline(x=0, color="black", linewidth=0.5)

    # Annotate significance
    for bar, sig, p in zip(bars, significant, [effects[t]["p_value"] for t in toggles]):
        label = f"p={p:.3f}" + (" *" if sig else "")
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f"  {label}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "toggle_effects.png"), dpi=150)
    plt.close()
```

---

## 7. Pairwise Covering Array Algorithm

### Problem Statement

For K toggles with value counts V_1, V_2, ..., V_K, the full Cartesian product has `product(V_i)` combinations. A pairwise covering array ensures that for every pair of toggles `(t_a, t_b)`, every combination of `(t_a=val_x, t_b=val_y)` appears in at least one run.

### Reduction Factor

Typical reduction for binary toggles:

| K (toggles) | Full | Pairwise (approx) |
|---|---|---|
| 3 | 8 | 4 |
| 4 | 16 | 5-6 |
| 5 | 32 | 8-10 |
| 6 | 64 | 10-12 |
| 8 | 256 | 12-16 |
| 10 | 1024 | 15-20 |

For mixed-value toggles (some binary, some ternary), the covering array is slightly larger but still far smaller than the full product.

### Greedy Algorithm

```python
def generate_pairwise_matrix(spec: dict) -> list[dict]:
    """Generate a pairwise covering array using a greedy algorithm."""
    toggles = spec["toggles"]
    constraints = spec.get("constraints", [])
    keys = sorted(toggles.keys())
    value_lists = [toggles[k] for k in keys]

    # Build the set of all pairs that must be covered
    uncovered_pairs = set()
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            for vi in value_lists[i]:
                for vj in value_lists[j]:
                    pair = (i, vi, j, vj)
                    uncovered_pairs.add(pair)

    covering_array = []

    while uncovered_pairs:
        best_row = None
        best_count = -1

        # Try candidate rows: for each uncovered pair, build a row
        # that covers it and greedily maximize additional coverage
        candidates = generate_candidates(keys, value_lists, uncovered_pairs)

        for candidate in candidates:
            combo = dict(zip(keys, candidate))
            if not passes_constraints(combo, constraints):
                continue
            count = count_covered_pairs(candidate, uncovered_pairs, len(keys))
            if count > best_count:
                best_count = count
                best_row = candidate

        if best_row is None:
            break  # No valid row can cover remaining pairs (constraint conflict)

        covering_array.append(dict(zip(keys, best_row)))
        # Remove newly covered pairs
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                pair = (i, best_row[i], j, best_row[j])
                uncovered_pairs.discard(pair)

    # Validate: all pairs covered
    assert len(uncovered_pairs) == 0, f"{len(uncovered_pairs)} pairs left uncovered"
    return covering_array


def generate_candidates(keys, value_lists, uncovered_pairs):
    """Generate candidate rows that each target at least one uncovered pair."""
    candidates = []
    for pair in list(uncovered_pairs)[:50]:  # Sample up to 50 uncovered pairs
        i, vi, j, vj = pair
        # Build a row fixing positions i and j, randomizing the rest
        for _ in range(5):  # 5 random completions per pair
            row = [random.choice(vl) for vl in value_lists]
            row[i] = vi
            row[j] = vj
            candidates.append(tuple(row))
    return candidates


def count_covered_pairs(row, uncovered_pairs, k):
    """Count how many uncovered pairs this row would cover."""
    count = 0
    for i in range(k):
        for j in range(i + 1, k):
            if (i, row[i], j, row[j]) in uncovered_pairs:
                count += 1
    return count
```

### Coverage Validation

After generating the covering array, validate that every pair is covered:

```python
def validate_pairwise_coverage(array: list[dict], spec: dict) -> bool:
    keys = sorted(spec["toggles"].keys())
    value_lists = [spec["toggles"][k] for k in keys]
    required_pairs = set()
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            for vi in value_lists[i]:
                for vj in value_lists[j]:
                    required_pairs.add((keys[i], vi, keys[j], vj))

    covered_pairs = set()
    for combo in array:
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                covered_pairs.add((keys[i], combo[keys[i]], keys[j], combo[keys[j]]))

    missing = required_pairs - covered_pairs
    if missing:
        logger.error(f"Pairwise coverage incomplete: {len(missing)} pairs missing")
        return False
    return True
```

---

## 8. Reusing Upstream Phases

### When to Reuse

Reuse upstream phases when all of the following hold:

1. The ablation spec's `phases` list starts at Phase N where N > 1.
2. A `baseline_run_id` is specified with a completed Phase N-1 boundary.
3. The base config's upstream-affecting parameters (workspace_dim, vocab size, SNN architecture, encoder architecture) match between the baseline and the current spec.

### Compatibility Checks

Before starting ablation runs, validate the baseline boundary:

```python
UPSTREAM_KEYS = [
    "encoder.output_dim",
    "workspace.workspace_dim",
    "snn.hidden_sizes",
    "encoder.text_vocab_size",
    "engram.vocab_size",
]

def validate_boundary_compatibility(boundary: dict, config: BrainAIConfig) -> None:
    """Raise ValueError if the baseline boundary is incompatible."""
    boundary_config = boundary["config"]
    for key in UPSTREAM_KEYS:
        baseline_val = resolve_dotted_key(boundary_config, key)
        current_val = resolve_dotted_key(config, key)
        if baseline_val != current_val:
            raise ValueError(
                f"Boundary incompatible on '{key}': "
                f"baseline={baseline_val}, current={current_val}"
            )
```

### Workflow

1. Load the spec and resolve `baseline_run_id`.
2. Validate boundary compatibility.
3. For each ablation run, symlink or copy the baseline boundary into the run directory:
   ```
   runs/{run_id}/checkpoints/phase3/phase_boundary.pt -> runs/{baseline_run_id}/checkpoints/phase3/phase_boundary.pt
   ```
4. Start the phase runner at the first phase in `spec.phases`, which loads the boundary and begins training.

### Cost Model

For a 7-phase pipeline where phases 1-3 consume T_upstream hours and phases 4-7 consume T_downstream hours per run:

- Without reuse: `N_runs * (T_upstream + T_downstream)`
- With reuse: `T_upstream + N_runs * T_downstream`
- Savings: `(N_runs - 1) * T_upstream`

For 72 runs where T_upstream = 10 GPU-hours: savings = 710 GPU-hours.

---

## 9. Ablation Comparison Utilities

### compare_ablations

Load the CSV and compute per-toggle effects with ranking:

```python
import pandas as pd

def compare_ablations(ablation_id: str, metric: str = "best_val_loss") -> pd.DataFrame:
    """Load ablation CSV and compute per-toggle effect summary."""
    csv_path = f"runs/{ablation_id}/ablations.csv"
    df = pd.read_csv(csv_path)
    df = df[df["status"] == "completed"]

    # Detect toggle columns (everything that's not a fixed schema column)
    fixed_cols = {
        "run_id", "ablation_id", "parent_run_id", "seed", "phase",
        "best_val_loss", "best_val_acc", "final_train_loss", "final_val_loss",
        "status", "duration_seconds", "git_sha", "dataset_fingerprint", "error_message",
    }
    toggle_cols = [c for c in df.columns if c not in fixed_cols]

    effects = {}
    for col in toggle_cols:
        effects[col] = compute_toggle_effects(df, col, metric)

    # Rank by absolute effect size
    ranked = sorted(effects.items(), key=lambda x: abs(x[1]["effect_size"]), reverse=True)
    return pd.DataFrame([
        {"toggle": k, **v} for k, v in ranked
    ])
```

### plot_ablation_matrix

Generate a heatmap of toggle combinations versus metric values:

```python
def plot_ablation_matrix(ablation_id: str, metric: str = "best_val_loss"):
    """Generate heatmap of toggle values vs metric."""
    csv_path = f"runs/{ablation_id}/ablations.csv"
    df = pd.read_csv(csv_path)
    df = df[df["status"] == "completed"]

    fixed_cols = {
        "run_id", "ablation_id", "parent_run_id", "seed", "phase",
        "best_val_loss", "best_val_acc", "final_train_loss", "final_val_loss",
        "status", "duration_seconds", "git_sha", "dataset_fingerprint", "error_message",
    }
    toggle_cols = sorted([c for c in df.columns if c not in fixed_cols])

    # Group by toggle combination, average over seeds
    grouped = df.groupby(toggle_cols)[metric].mean().reset_index()

    # Build label for each combination
    labels = []
    for _, row in grouped.iterrows():
        label = " | ".join(f"{c}={row[c]}" for c in toggle_cols)
        labels.append(label)

    fig, ax = plt.subplots(figsize=(12, max(6, len(labels) * 0.4)))
    colors = plt.cm.RdYlGn_r(plt.Normalize()(grouped[metric].values))
    ax.barh(labels, grouped[metric].values, color=colors)
    ax.set_xlabel(metric)
    ax.set_title(f"Ablation Matrix: {ablation_id}")
    plt.tight_layout()
    plt.savefig(f"runs/{ablation_id}/reports/ablation_matrix.png", dpi=150)
    plt.close()
```

### best_config

Return the configuration of the best-performing combination:

```python
def best_config(ablation_id: str, metric: str = "best_val_loss", lower_is_better: bool = True) -> dict:
    """Return the config overrides of the best-performing ablation run."""
    csv_path = f"runs/{ablation_id}/ablations.csv"
    df = pd.read_csv(csv_path)
    df = df[df["status"] == "completed"]

    fixed_cols = {
        "run_id", "ablation_id", "parent_run_id", "seed", "phase",
        "best_val_loss", "best_val_acc", "final_train_loss", "final_val_loss",
        "status", "duration_seconds", "git_sha", "dataset_fingerprint", "error_message",
    }
    toggle_cols = sorted([c for c in df.columns if c not in fixed_cols])

    # Average metric over seeds for each combination
    grouped = df.groupby(toggle_cols)[metric].mean()

    if lower_is_better:
        best_idx = grouped.idxmin()
    else:
        best_idx = grouped.idxmax()

    # Convert multi-index to dict
    if isinstance(best_idx, tuple):
        best_overrides = dict(zip(toggle_cols, best_idx))
    else:
        best_overrides = {toggle_cols[0]: best_idx}

    best_metric = grouped[best_idx]
    best_runs = df
    for col, val in best_overrides.items():
        best_runs = best_runs[best_runs[col] == val]

    return {
        "overrides": best_overrides,
        "mean_metric": float(best_metric),
        "std_metric": float(best_runs[metric].std()),
        "run_ids": best_runs["run_id"].tolist(),
        "seeds": best_runs["seed"].tolist(),
    }
```

---

## CLI Usage

Run ablations from the command line via `scripts/run_ablation.py`:

```bash
# Execute an ablation spec
python scripts/run_ablation.py --spec ablation_specs/workspace_engram.yaml

# Execute in parallel with 4 concurrent runs
python scripts/run_ablation.py --spec ablation_specs/workspace_engram.yaml --parallel --max-concurrent 4

# Generate matrix only (dry run), print run count
python scripts/run_ablation.py --spec ablation_specs/workspace_engram.yaml --dry-run

# Generate report from completed ablation
python scripts/run_ablation.py --report --ablation-id workspace_engram_ablation_20260220_143022_a1b2c3d4

# Print best config
python scripts/run_ablation.py --best-config --ablation-id workspace_engram_ablation_20260220_143022_a1b2c3d4 --metric best_val_loss
```

---

## End-to-End Example

Define a small ablation spec for dev mode:

```yaml
# ablation_specs/snn_workspace_dev.yaml
ablation:
  name: "snn_workspace_dev"
  description: "Ablate SNN delays and workspace ignition in dev mode"
  base_config: "dev"
  phases: [1, 4]
  baseline_run_id: null

  toggles:
    snn.use_learnable_delays: [false, true]
    snn.use_heterogeneous_tau: [false, true]
    workspace.ignition_threshold: [0.2, 0.3]

  constraints: []
  mode: "full"
  seeds: [1337, 42]
```

This produces `2 * 2 * 2 = 8` combinations, times 2 seeds = 16 runs. Execute:

```bash
python scripts/run_ablation.py --spec ablation_specs/snn_workspace_dev.yaml --parallel --max-concurrent 2
```

After completion, inspect results:

```python
from brain_ai.training.ablation import compare_ablations, best_config, plot_ablation_matrix

# Rank toggles by effect size
effects = compare_ablations("snn_workspace_dev_20260220_150000_abcd1234")
print(effects)

# Get best config
best = best_config("snn_workspace_dev_20260220_150000_abcd1234", metric="best_val_loss")
print(f"Best overrides: {best['overrides']}")
print(f"Mean val loss: {best['mean_metric']:.4f} +/- {best['std_metric']:.4f}")

# Generate plots
plot_ablation_matrix("snn_workspace_dev_20260220_150000_abcd1234")
```
