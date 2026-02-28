#!/usr/bin/env python3
"""
ablation_template.py -- Comprehensive template for the automatic ablation system.

Provides AblationSpec, MatrixGenerator (full + pairwise), AblationExecutor,
AblationReport, CSV I/O, and deterministic run-ID derivation. Self-contained
with no brain_ai imports; uses only the Python standard library plus optional
PyYAML for spec parsing (falls back to JSON).

Classes:
    AblationRun          -- dataclass for a single ablation run descriptor
    AblationRunIDDeriver -- deterministic, sortable, unique run/ablation IDs
    AblationSpec         -- defines toggles, constraints, mode, seeds
    PairwiseCoveringArray-- greedy pairwise covering array generator
    MatrixGenerator      -- full/pairwise matrix with constraint filtering
    AblationRunResult    -- dataclass for completed run results
    AblationCSVWriter    -- dynamic-column CSV I/O for ablation results
    AblationReport       -- aggregation, best-run, toggle-effect analysis
    AblationExecutor     -- sequential/parallel execution with failure handling

Usage:
    python ablation_template.py          # run 50+ self-tests
"""

from __future__ import annotations

import csv
import hashlib
import io
import itertools
import json
import math
import os
import random
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

# ---------------------------------------------------------------------------
# Optional YAML support -- fall back to JSON if pyyaml is not installed
# ---------------------------------------------------------------------------
try:
    import yaml as _yaml

    _HAS_YAML = True
except ImportError:
    _yaml = None  # type: ignore[assignment]
    _HAS_YAML = False


# ===================================================================
# 1. AblationRun dataclass
# ===================================================================

@dataclass
class AblationRun:
    """Descriptor for a single run within an ablation matrix.

    Attributes:
        run_index:       Zero-based position in the matrix.
        run_id:          Deterministic identifier derived from overrides + seed.
        ablation_id:     Parent ablation identifier.
        overrides:       Toggle values for this particular run.
        seed:            RNG seed for this run.
        phases:          Training phases to execute.
        baseline_run_id: Optional upstream run whose checkpoints to reuse.
        status:          Lifecycle status of this run.
    """
    run_index: int
    run_id: str
    ablation_id: str
    overrides: Dict[str, Any]
    seed: int
    phases: List[int]
    baseline_run_id: Optional[str] = None
    status: str = "pending"  # pending | running | completed | failed | skipped

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ===================================================================
# 2. AblationRunIDDeriver
# ===================================================================

class AblationRunIDDeriver:
    """Deterministic, unique, sortable identifiers for ablations and runs.

    Guarantees
    ----------
    - **Unique**: timestamp + spec hash + per-run override hash + seed.
    - **Deterministic**: same inputs always yield the same hash components.
    - **Sortable**: timestamp prefix for chronological order; run index for
      within-ablation order.
    - **Human-readable**: name prefix and seed suffix are directly interpretable.
    """

    @staticmethod
    def _sanitize(name: str) -> str:
        """Lowercase, strip non-alphanumeric characters except underscores."""
        return "".join(c if c.isalnum() or c == "_" else "_" for c in name.lower()).strip("_")

    @staticmethod
    def _canonical_json(obj: Any) -> str:
        """Produce a deterministic JSON string (sorted keys, compact)."""
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)

    @classmethod
    def derive_ablation_id(cls, name: str, timestamp: Optional[str] = None,
                           spec_dict: Optional[Dict[str, Any]] = None) -> str:
        """Derive ablation-level ID: ``{name}_{timestamp}_{spec_hash[:8]}``.

        Parameters
        ----------
        name : str
            Human-readable ablation name.
        timestamp : str, optional
            UTC timestamp string (``YYYYMMDD_HHMMSS``).  Generated if *None*.
        spec_dict : dict, optional
            Full spec dictionary used for the hash component.
        """
        sanitized = cls._sanitize(name)
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        if spec_dict is not None:
            canonical = cls._canonical_json(spec_dict)
            spec_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]
        else:
            spec_hash = "00000000"
        return f"{sanitized}_{timestamp}_{spec_hash}"

    @classmethod
    def derive_run_id(cls, ablation_id: str, run_index: int,
                      overrides: Dict[str, Any], seed: int) -> str:
        """Derive per-run ID: ``{ablation_id}_run{N:03d}_{hash[:6]}_{seed}``.

        The hash is computed from the *sorted* override dict so that
        identical overrides always produce the same ID regardless of
        insertion order.
        """
        clean = {k: v for k, v in sorted(overrides.items())}
        canonical = cls._canonical_json(clean)
        overrides_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:6]
        return f"{ablation_id}_run{run_index:03d}_{overrides_hash}_{seed}"


# ===================================================================
# 3. AblationSpec
# ===================================================================

class AblationSpec:
    """Specification for an ablation experiment.

    Parameters
    ----------
    name : str
        Human-readable name.
    description : str
        Free-text description stored in reports.
    base_config : str
        Path to base config YAML or preset name.
    toggles : Dict[str, List[Any]]
        Toggle names mapped to lists of values to sweep.
    constraints : list, optional
        If-then rules for pruning invalid combinations.
    mode : str
        ``"full"`` (Cartesian product) or ``"pairwise"`` (covering array).
    phases : list of int, optional
        Training phases to execute per combination.
    seeds : list of int, optional
        Seeds for multi-run statistical analysis.
    baseline_run_id : str, optional
        Existing run whose upstream phase checkpoints to reuse.
    metrics : dict, optional
        ``{"primary": "best_val_loss", "secondary": [...]}``.
    resource_limits : dict, optional
        Concurrency, timeout, GPU constraints.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        base_config: str = "dev",
        toggles: Optional[Dict[str, List[Any]]] = None,
        constraints: Optional[List[Dict[str, Any]]] = None,
        mode: str = "full",
        phases: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        baseline_run_id: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        resource_limits: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.description = description
        self.base_config = base_config
        self.toggles: Dict[str, List[Any]] = toggles or {}
        self.constraints: List[Dict[str, Any]] = constraints or []
        self.mode = mode
        self.phases: List[int] = phases or [1]
        self.seeds: List[int] = seeds or [1337]
        self.baseline_run_id = baseline_run_id
        self.metrics: Dict[str, Any] = metrics or {"primary": "best_val_loss", "secondary": []}
        self.resource_limits: Dict[str, Any] = resource_limits or {}

    # ---- Serialization ------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain-dict representation suitable for JSON/YAML."""
        return {
            "ablation": {
                "name": self.name,
                "description": self.description,
                "base_config": self.base_config,
                "toggles": self.toggles,
                "constraints": self.constraints,
                "mode": self.mode,
                "phases": self.phases,
                "seeds": self.seeds,
                "baseline_run_id": self.baseline_run_id,
                "metrics": self.metrics,
                "resource_limits": self.resource_limits,
            }
        }

    def save(self, path: str) -> None:
        """Serialize to YAML (preferred) or JSON."""
        data = self.to_dict()
        with open(path, "w") as fh:
            if _HAS_YAML and path.endswith((".yaml", ".yml")):
                _yaml.dump(data, fh, default_flow_style=False, sort_keys=False)
            else:
                json.dump(data, fh, indent=2, sort_keys=False)

    @classmethod
    def load(cls, path: str) -> "AblationSpec":
        """Load from a YAML or JSON spec file."""
        with open(path, "r") as fh:
            if _HAS_YAML and path.endswith((".yaml", ".yml")):
                data = _yaml.safe_load(fh)
            else:
                data = json.load(fh)
        ab = data.get("ablation", data)
        return cls(
            name=ab["name"],
            description=ab.get("description", ""),
            base_config=ab.get("base_config", "dev"),
            toggles=ab.get("toggles", {}),
            constraints=ab.get("constraints", []),
            mode=ab.get("mode", "full"),
            phases=ab.get("phases", [1]),
            seeds=ab.get("seeds", [1337]),
            baseline_run_id=ab.get("baseline_run_id"),
            metrics=ab.get("metrics", {"primary": "best_val_loss", "secondary": []}),
            resource_limits=ab.get("resource_limits", {}),
        )

    # ---- Validation ---------------------------------------------------

    def validate(self, known_toggles: Optional[Set[str]] = None) -> List[str]:
        """Validate the spec and return a list of error messages (empty = valid).

        Parameters
        ----------
        known_toggles : set of str, optional
            If provided, toggles are checked against this set to ensure they
            correspond to real config fields.
        """
        errors: List[str] = []

        if not self.name:
            errors.append("Spec name must be non-empty.")
        if self.mode not in ("full", "pairwise"):
            errors.append(f"Invalid mode {self.mode!r}; must be 'full' or 'pairwise'.")
        if not self.toggles:
            errors.append("At least one toggle must be specified.")
        for tname, tvals in self.toggles.items():
            if not isinstance(tvals, list) or len(tvals) < 1:
                errors.append(f"Toggle {tname!r} must have at least one value.")
        if not self.seeds:
            errors.append("At least one seed must be specified.")
        if not self.phases:
            errors.append("At least one phase must be specified.")

        # Check toggles against known config fields
        if known_toggles is not None:
            for tname in self.toggles:
                if tname not in known_toggles:
                    errors.append(f"Unknown toggle {tname!r}; not in base config.")

        # Check constraints reference valid toggles
        toggle_names = set(self.toggles.keys())
        for i, rule in enumerate(self.constraints):
            if_clause = rule.get("if", {})
            then_clause = rule.get("then", {})
            for key in list(if_clause.keys()) + list(then_clause.keys()):
                if key not in toggle_names:
                    errors.append(
                        f"Constraint #{i} references unknown toggle {key!r}."
                    )

        return errors


# ===================================================================
# 4. PairwiseCoveringArray
# ===================================================================

class PairwiseCoveringArray:
    """Greedy algorithm for generating pairwise (2-way) covering arrays.

    A pairwise covering array ensures that for every pair of toggles
    ``(t_a, t_b)``, every combination ``(t_a=val_x, t_b=val_y)`` appears
    in at least one row of the resulting array.

    Typical reduction for binary toggles:
        5 toggles: 32 full -> ~8 pairwise
        6 toggles: 64 full -> ~10 pairwise
    """

    def __init__(self, rng_seed: int = 42) -> None:
        self._rng = random.Random(rng_seed)

    def generate(self, toggles: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate a pairwise covering array for the given toggles.

        Returns a list of override dicts such that every pair of
        ``(toggle_a=val_x, toggle_b=val_y)`` is present in at least one row.
        """
        keys = sorted(toggles.keys())
        value_lists = [toggles[k] for k in keys]
        k = len(keys)

        if k == 0:
            return []
        if k == 1:
            return [{keys[0]: v} for v in value_lists[0]]

        # Build the universe of pairs that must be covered
        uncovered: Set[Tuple[int, Any, int, Any]] = set()
        for i in range(k):
            for j in range(i + 1, k):
                for vi in value_lists[i]:
                    for vj in value_lists[j]:
                        uncovered.add((i, vi, j, vj))

        covering: List[Tuple[Any, ...]] = []

        while uncovered:
            best_row: Optional[Tuple[Any, ...]] = None
            best_count = -1

            # Sample uncovered pairs to seed candidate rows
            sample_size = min(len(uncovered), 50)
            sampled_pairs = self._rng.sample(sorted(uncovered), sample_size)

            candidates: List[Tuple[Any, ...]] = []
            for pair in sampled_pairs:
                pi, pvi, pj, pvj = pair
                for _ in range(5):  # 5 random completions per sampled pair
                    row = [self._rng.choice(vl) for vl in value_lists]
                    row[pi] = pvi
                    row[pj] = pvj
                    candidates.append(tuple(row))

            for candidate in candidates:
                count = self._count_covered(candidate, uncovered, k)
                if count > best_count:
                    best_count = count
                    best_row = candidate

            if best_row is None or best_count == 0:
                # Fallback: brute-force pick any row that covers at least 1 pair
                for pair in sorted(uncovered):
                    pi, pvi, pj, pvj = pair
                    row = [self._rng.choice(vl) for vl in value_lists]
                    row[pi] = pvi
                    row[pj] = pvj
                    best_row = tuple(row)
                    break
                if best_row is None:
                    break  # should never happen

            covering.append(best_row)
            # Remove all pairs covered by the chosen row
            for i in range(k):
                for j in range(i + 1, k):
                    uncovered.discard((i, best_row[i], j, best_row[j]))

        return [dict(zip(keys, row)) for row in covering]

    @staticmethod
    def _count_covered(row: Tuple[Any, ...], uncovered: Set[Tuple[int, Any, int, Any]],
                       k: int) -> int:
        """Count how many uncovered pairs this row would satisfy."""
        count = 0
        for i in range(k):
            for j in range(i + 1, k):
                if (i, row[i], j, row[j]) in uncovered:
                    count += 1
        return count

    @staticmethod
    def verify_coverage(combinations: List[Dict[str, Any]],
                        toggles: Dict[str, List[Any]]) -> bool:
        """Assert that all pairwise combinations are present.

        Returns True if coverage is complete, raises AssertionError otherwise.
        """
        keys = sorted(toggles.keys())
        value_lists = [toggles[k] for k in keys]
        k = len(keys)

        required: Set[Tuple[str, Any, str, Any]] = set()
        for i in range(k):
            for j in range(i + 1, k):
                for vi in value_lists[i]:
                    for vj in value_lists[j]:
                        required.add((keys[i], vi, keys[j], vj))

        covered: Set[Tuple[str, Any, str, Any]] = set()
        for combo in combinations:
            for i in range(k):
                for j in range(i + 1, k):
                    covered.add((keys[i], combo[keys[i]], keys[j], combo[keys[j]]))

        missing = required - covered
        assert len(missing) == 0, (
            f"Pairwise coverage incomplete: {len(missing)} pairs missing out of "
            f"{len(required)} required."
        )
        return True


# ===================================================================
# 5. MatrixGenerator
# ===================================================================

class MatrixGenerator:
    """Generate ablation run matrices from an AblationSpec.

    Supports full (Cartesian product) and pairwise (covering array)
    generation modes, with constraint filtering and seed expansion.
    """

    def __init__(self, pairwise_seed: int = 42) -> None:
        self._pairwise_gen = PairwiseCoveringArray(rng_seed=pairwise_seed)

    # ---- Full Cartesian Product --------------------------------------

    def generate_full(self, spec: AblationSpec) -> List[Dict[str, Any]]:
        """Cartesian product of all toggle values, filtered by constraints."""
        keys = sorted(spec.toggles.keys())
        value_lists = [spec.toggles[k] for k in keys]
        raw = [dict(zip(keys, vals)) for vals in itertools.product(*value_lists)]
        return self.apply_constraints(raw, spec.constraints, spec.toggles)

    # ---- Pairwise Covering Array -------------------------------------

    def generate_pairwise(self, spec: AblationSpec) -> List[Dict[str, Any]]:
        """Pairwise covering array filtered by constraints."""
        combos = self._pairwise_gen.generate(spec.toggles)
        return self.apply_constraints(combos, spec.constraints, spec.toggles)

    # ---- Constraint Application --------------------------------------

    @staticmethod
    def apply_constraints(
        combinations: List[Dict[str, Any]],
        constraints: List[Dict[str, Any]],
        toggles: Dict[str, List[Any]],
    ) -> List[Dict[str, Any]]:
        """Filter/modify combinations based on if-then rules.

        When a ``then`` value is *None*, the constrained toggle is pinned to
        its first listed value (removing duplicate rows where the toggle has
        other values under the same ``if`` condition).
        """
        if not constraints:
            return combinations

        valid: List[Dict[str, Any]] = []
        for combo in combinations:
            keep = True
            for rule in constraints:
                if_clause: Dict[str, Any] = rule.get("if", {})
                then_clause: Dict[str, Any] = rule.get("then", {})

                # Check whether the ``if`` condition matches
                if_matches = all(
                    combo.get(k) == v for k, v in if_clause.items()
                )
                if not if_matches:
                    continue

                # Apply ``then`` clause
                for tk, tv in then_clause.items():
                    if tv is None:
                        # Pin to first value; skip rows with any other value
                        first_val = toggles[tk][0] if tk in toggles else None
                        if combo.get(tk) != first_val:
                            keep = False
                            break
                    else:
                        if combo.get(tk) != tv:
                            keep = False
                            break
                if not keep:
                    break
            if keep:
                valid.append(combo)
        return valid

    # ---- Seed Expansion -----------------------------------------------

    @staticmethod
    def expand_seeds(combinations: List[Dict[str, Any]],
                     seeds: List[int]) -> List[Dict[str, Any]]:
        """Multiply each combination by each seed."""
        expanded: List[Dict[str, Any]] = []
        for combo in combinations:
            for seed in seeds:
                entry = dict(combo)
                entry["__seed__"] = seed
                expanded.append(entry)
        return expanded

    # ---- Estimation ---------------------------------------------------

    def estimate_total_runs(self, spec: AblationSpec) -> int:
        """Estimate total run count without fully generating the matrix."""
        if spec.mode == "pairwise":
            # Rough upper bound for pairwise: max(v_i) * max(v_j) for
            # the two largest value lists, but the greedy algorithm usually
            # produces fewer.  We generate for an exact count.
            combos = self.generate_pairwise(spec)
        else:
            combos = self.generate_full(spec)
        return len(combos) * len(spec.seeds)

    # ---- Unified Entry Point ------------------------------------------

    def generate(self, spec: AblationSpec) -> List[AblationRun]:
        """Generate the full list of AblationRun descriptors from a spec."""
        ablation_id = AblationRunIDDeriver.derive_ablation_id(
            name=spec.name,
            spec_dict=spec.to_dict(),
        )

        if spec.mode == "pairwise":
            combos = self.generate_pairwise(spec)
        else:
            combos = self.generate_full(spec)

        expanded = self.expand_seeds(combos, spec.seeds)

        runs: List[AblationRun] = []
        for idx, entry in enumerate(expanded):
            seed = entry.pop("__seed__")
            run_id = AblationRunIDDeriver.derive_run_id(
                ablation_id=ablation_id,
                run_index=idx,
                overrides=entry,
                seed=seed,
            )
            runs.append(AblationRun(
                run_index=idx,
                run_id=run_id,
                ablation_id=ablation_id,
                overrides=dict(entry),
                seed=seed,
                phases=list(spec.phases),
                baseline_run_id=spec.baseline_run_id,
                status="pending",
            ))
        return runs


# ===================================================================
# 6. AblationRunResult dataclass
# ===================================================================

@dataclass
class AblationRunResult:
    """Outcome of a single completed (or failed) ablation run."""
    run_id: str
    ablation_id: str
    overrides: Dict[str, Any]
    seed: int
    phases: List[int]
    status: str  # completed | failed | skipped
    metrics: Dict[str, float] = field(default_factory=dict)
    duration_seconds: float = 0.0
    error: Optional[str] = None
    baseline_run_id: Optional[str] = None
    git_sha: str = ""
    dataset_fingerprint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ===================================================================
# 7. AblationCSVWriter
# ===================================================================

class AblationCSVWriter:
    """Dynamic-column CSV writer/reader for ablation results.

    Column layout:
        run_id, ablation_id, parent_run_id, seed,
        [toggle columns ...],
        phase, best_val_loss, best_val_acc, final_train_loss, final_val_loss,
        status, duration_seconds, git_sha, dataset_fingerprint, error_message
    """

    FIXED_PREFIX = ["run_id", "ablation_id", "parent_run_id", "seed"]
    FIXED_SUFFIX = [
        "phase", "best_val_loss", "best_val_acc",
        "final_train_loss", "final_val_loss",
        "status", "duration_seconds", "git_sha",
        "dataset_fingerprint", "error_message",
    ]

    @classmethod
    def fieldnames(cls, toggle_keys: List[str]) -> List[str]:
        """Compute the full ordered list of CSV columns."""
        return cls.FIXED_PREFIX + sorted(toggle_keys) + cls.FIXED_SUFFIX

    @classmethod
    def write_csv(cls, path: str, results: List[AblationRunResult],
                  toggle_keys: List[str], phases: List[int]) -> None:
        """Write a complete CSV from a list of run results."""
        fields = cls.fieldnames(toggle_keys)
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for r in results:
                cls._write_row(writer, r, toggle_keys, phases)

    @classmethod
    def append_row(cls, path: str, result: AblationRunResult,
                   toggle_keys: List[str], phases: List[int]) -> None:
        """Atomically append a single row to an existing CSV."""
        fields = cls.fieldnames(toggle_keys)
        file_exists = os.path.isfile(path)
        with open(path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            if not file_exists:
                writer.writeheader()
            cls._write_row(writer, result, toggle_keys, phases)

    @classmethod
    def _write_row(cls, writer: csv.DictWriter, r: AblationRunResult,
                   toggle_keys: List[str], phases: List[int]) -> None:
        row: Dict[str, Any] = {
            "run_id": r.run_id,
            "ablation_id": r.ablation_id,
            "parent_run_id": r.baseline_run_id or "",
            "seed": r.seed,
            "phase": ",".join(str(p) for p in phases),
            "best_val_loss": r.metrics.get("best_val_loss", ""),
            "best_val_acc": r.metrics.get("best_val_acc", ""),
            "final_train_loss": r.metrics.get("final_train_loss", ""),
            "final_val_loss": r.metrics.get("final_val_loss", ""),
            "status": r.status,
            "duration_seconds": f"{r.duration_seconds:.2f}",
            "git_sha": r.git_sha,
            "dataset_fingerprint": r.dataset_fingerprint,
            "error_message": r.error or "",
        }
        for tk in sorted(toggle_keys):
            row[tk] = r.overrides.get(tk, "")
        writer.writerow(row)

    @classmethod
    def load_csv(cls, path: str) -> List[Dict[str, str]]:
        """Parse a CSV file back into a list of row dicts."""
        with open(path, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            return [dict(row) for row in reader]


# ===================================================================
# 8. AblationReport
# ===================================================================

class AblationReport:
    """Aggregates results from an ablation run matrix.

    Provides CSV/JSON output, best-run selection, and per-toggle
    effect-size computation.
    """

    def __init__(self, runs: List[AblationRunResult],
                 toggle_keys: Optional[List[str]] = None,
                 phases: Optional[List[int]] = None) -> None:
        self.runs = runs
        self.toggle_keys: List[str] = toggle_keys or self._infer_toggle_keys()
        self.phases: List[int] = phases or [1]

    def _infer_toggle_keys(self) -> List[str]:
        """Infer toggle column names from the override keys of the first run."""
        keys: Set[str] = set()
        for r in self.runs:
            keys.update(r.overrides.keys())
        return sorted(keys)

    # ---- Persistence --------------------------------------------------

    def save_csv(self, path: str) -> None:
        """Write ``ablations.csv``."""
        AblationCSVWriter.write_csv(path, self.runs, self.toggle_keys, self.phases)

    def save_json(self, path: str) -> None:
        """Write ``ablation_summary.json``."""
        ablation_id = self.runs[0].ablation_id if self.runs else "unknown"
        completed = [r for r in self.runs if r.status == "completed"]
        failed = [r for r in self.runs if r.status == "failed"]

        best = self.get_best_run()
        toggle_effects = self.compute_toggle_effects()

        summary: Dict[str, Any] = {
            "ablation_id": ablation_id,
            "total_runs": len(self.runs),
            "completed": len(completed),
            "failed": len(failed),
            "toggle_effects": toggle_effects,
        }
        if best is not None:
            summary["best_run"] = {
                "run_id": best.run_id,
                "overrides": best.overrides,
                "seed": best.seed,
                "metrics": best.metrics,
            }
        if failed:
            summary["failure_summary"] = {
                "count": len(failed),
                "run_ids": [r.run_id for r in failed],
                "errors": [r.error for r in failed if r.error],
            }

        with open(path, "w") as fh:
            json.dump(summary, fh, indent=2, default=str)

    # ---- Analysis -----------------------------------------------------

    def get_best_run(self, metric: str = "best_val_loss",
                     mode: str = "min") -> Optional[AblationRunResult]:
        """Return the best-performing run by the specified metric.

        Parameters
        ----------
        metric : str
            Key in the run's ``metrics`` dict.
        mode : str
            ``"min"`` for lower-is-better, ``"max"`` for higher-is-better.
        """
        completed = [r for r in self.runs if r.status == "completed"
                      and metric in r.metrics]
        if not completed:
            return None

        if mode == "min":
            return min(completed, key=lambda r: r.metrics[metric])
        else:
            return max(completed, key=lambda r: r.metrics[metric])

    def compute_toggle_effects(self, metric: str = "best_val_loss") -> Dict[str, Dict[str, Any]]:
        """Compute per-toggle mean effect size.

        For each toggle, partition completed runs by toggle value and compute
        the mean metric for each partition.  The effect size is the difference
        between the mean of the *last* value and the mean of the *first* value
        in sorted order.
        """
        completed = [r for r in self.runs if r.status == "completed"
                      and metric in r.metrics]
        if not completed:
            return {}

        effects: Dict[str, Dict[str, Any]] = {}
        for tk in self.toggle_keys:
            # Group metrics by toggle value
            groups: Dict[Any, List[float]] = {}
            for r in completed:
                val = r.overrides.get(tk)
                if val is not None:
                    groups.setdefault(val, []).append(r.metrics[metric])

            if len(groups) < 2:
                continue

            sorted_vals = sorted(groups.keys(), key=lambda x: str(x))
            means = {str(v): _safe_mean(groups[v]) for v in sorted_vals}
            effect_size = means[str(sorted_vals[-1])] - means[str(sorted_vals[0])]

            effects[tk] = {
                "values": sorted_vals,
                "mean_metric": means,
                "effect_size": round(effect_size, 6),
            }
        return effects

    # ---- Summary ------------------------------------------------------

    def summary(self) -> str:
        """Return a formatted summary string."""
        total = len(self.runs)
        completed = sum(1 for r in self.runs if r.status == "completed")
        failed = sum(1 for r in self.runs if r.status == "failed")
        skipped = sum(1 for r in self.runs if r.status == "skipped")

        lines = [
            f"Ablation Report",
            f"  Total runs:     {total}",
            f"  Completed:      {completed}",
            f"  Failed:         {failed}",
            f"  Skipped:        {skipped}",
        ]

        best = self.get_best_run()
        if best is not None:
            lines.append(f"  Best run:       {best.run_id}")
            lines.append(f"  Best val_loss:  {best.metrics.get('best_val_loss', 'N/A')}")
            lines.append(f"  Best overrides: {best.overrides}")

        effects = self.compute_toggle_effects()
        if effects:
            lines.append(f"  Toggle effects:")
            for tk, eff in effects.items():
                lines.append(f"    {tk}: effect_size={eff['effect_size']:.4f}")

        return "\n".join(lines)


# ===================================================================
# 9. AblationExecutor
# ===================================================================

class AblationExecutor:
    """Executes an ablation matrix, running each combination through a
    configurable training function.

    Parameters
    ----------
    spec : AblationSpec
        The ablation specification.
    base_dir : str
        Root directory for run outputs.
    train_fn : callable, optional
        Function with signature
        ``(overrides: dict, seed: int, phases: list, run_dir: str) -> dict``
        returning a metrics dict.  Defaults to a mock that generates random
        metrics (useful for template testing).
    """

    def __init__(self, spec: AblationSpec, base_dir: str = "runs/",
                 train_fn: Optional[Callable[..., Dict[str, float]]] = None) -> None:
        self.spec = spec
        self.base_dir = base_dir
        self.train_fn = train_fn or self._mock_train

    @staticmethod
    def _mock_train(overrides: Dict[str, Any], seed: int,
                    phases: List[int], run_dir: str) -> Dict[str, float]:
        """Mock training function for template testing."""
        rng = random.Random(seed)
        # Simulate a small amount of work
        time.sleep(0.001)
        return {
            "best_val_loss": rng.uniform(0.2, 0.5),
            "best_val_acc": rng.uniform(0.7, 0.95),
            "final_train_loss": rng.uniform(0.1, 0.3),
            "final_val_loss": rng.uniform(0.2, 0.5),
        }

    def _get_git_sha(self) -> str:
        """Best-effort git SHA retrieval."""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
            return result.stdout.strip()[:12] if result.returncode == 0 else ""
        except Exception:
            return ""

    def _execute_single(self, run: AblationRun) -> AblationRunResult:
        """Execute a single ablation run."""
        run_dir = os.path.join(self.base_dir, run.ablation_id, run.run_id)
        os.makedirs(run_dir, exist_ok=True)

        # Save run manifest
        manifest_path = os.path.join(run_dir, "run_manifest.json")
        with open(manifest_path, "w") as fh:
            json.dump(run.to_dict(), fh, indent=2, default=str)

        git_sha = self._get_git_sha()
        start_time = time.monotonic()

        try:
            metrics = self.train_fn(
                overrides=run.overrides,
                seed=run.seed,
                phases=run.phases,
                run_dir=run_dir,
            )
            duration = time.monotonic() - start_time
            return AblationRunResult(
                run_id=run.run_id,
                ablation_id=run.ablation_id,
                overrides=run.overrides,
                seed=run.seed,
                phases=run.phases,
                status="completed",
                metrics=metrics,
                duration_seconds=duration,
                baseline_run_id=run.baseline_run_id,
                git_sha=git_sha,
            )
        except Exception as exc:
            duration = time.monotonic() - start_time
            return AblationRunResult(
                run_id=run.run_id,
                ablation_id=run.ablation_id,
                overrides=run.overrides,
                seed=run.seed,
                phases=run.phases,
                status="failed",
                metrics={},
                duration_seconds=duration,
                error=str(exc),
                baseline_run_id=run.baseline_run_id,
                git_sha=git_sha,
            )

    # ---- Sequential Execution ----------------------------------------

    def execute_sequential(self, matrix: List[AblationRun]) -> AblationReport:
        """Run each combination sequentially; capture results.

        Failures are recorded but do not abort the remaining runs.
        """
        results: List[AblationRunResult] = []
        for run in matrix:
            run.status = "running"
            result = self._execute_single(run)
            results.append(result)
        return AblationReport(
            runs=results,
            toggle_keys=sorted(self.spec.toggles.keys()),
            phases=self.spec.phases,
        )

    # ---- Parallel Execution ------------------------------------------

    def execute_parallel(self, matrix: List[AblationRun],
                         max_workers: int = 4) -> AblationReport:
        """Run up to *max_workers* combinations concurrently.

        Uses ``ProcessPoolExecutor``.  Note: the provided ``train_fn`` must
        be picklable for cross-process dispatch. The default mock training
        function satisfies this requirement; real training functions that
        capture unpicklable state (e.g. CUDA contexts) should use the
        sequential executor or a subprocess-based approach instead.

        Limitations:
        - GPU memory contention is not managed; callers should set
          ``CUDA_VISIBLE_DEVICES`` per worker externally.
        - Progress reporting is limited to post-completion collection.
        """
        results: List[AblationRunResult] = []

        # Because ProcessPoolExecutor requires picklable callables, and
        # instance methods are not always reliably picklable, we fall back
        # to sequential execution when the train_fn is a bound method.
        # For production use, wrap training in a top-level function.
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_map = {}
                for run in matrix:
                    run.status = "running"
                    future = executor.submit(self._execute_single, run)
                    future_map[future] = run.run_id

                for future in as_completed(future_map):
                    run_id = future_map[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as exc:
                        # Construct a failure result if the future itself raises
                        results.append(AblationRunResult(
                            run_id=run_id,
                            ablation_id=matrix[0].ablation_id if matrix else "",
                            overrides={},
                            seed=0,
                            phases=self.spec.phases,
                            status="failed",
                            error=f"ProcessPoolExecutor error: {exc}",
                        ))
        except Exception:
            # Fall back to sequential if multiprocessing fails
            return self.execute_sequential(matrix)

        return AblationReport(
            runs=results,
            toggle_keys=sorted(self.spec.toggles.keys()),
            phases=self.spec.phases,
        )


# ===================================================================
# Utility helpers
# ===================================================================

def _safe_mean(values: Sequence[float]) -> float:
    """Compute mean with empty-list safety."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _safe_std(values: Sequence[float]) -> float:
    """Compute sample standard deviation with small-N safety."""
    if len(values) < 2:
        return 0.0
    mean = _safe_mean(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


# ===================================================================
# 10. Self-test block
# ===================================================================

def _run_self_tests() -> None:
    """Execute 50+ self-tests covering all major components."""

    passed = 0
    failed = 0
    errors: List[str] = []

    def _assert(cond: bool, msg: str) -> None:
        nonlocal passed, failed
        if cond:
            passed += 1
        else:
            failed += 1
            errors.append(f"FAIL: {msg}")
            print(f"  FAIL: {msg}")

    print("=" * 70)
    print("Ablation Template Self-Tests")
    print("=" * 70)

    # ------------------------------------------------------------------
    # AblationSpec creation and validation
    # ------------------------------------------------------------------
    print("\n--- AblationSpec ---")

    spec = AblationSpec(
        name="test_ablation",
        description="Unit test ablation",
        base_config="dev",
        toggles={
            "use_engram": [False, True],
            "engram_mode": ["encoder", "layer"],
            "use_snn": [False, True],
        },
        constraints=[
            {"if": {"use_engram": False}, "then": {"engram_mode": None}},
        ],
        mode="full",
        phases=[4, 5],
        seeds=[1337, 42, 7],
    )

    errs = spec.validate()
    _assert(len(errs) == 0, f"AblationSpec.validate() should pass: {errs}")

    # Test validation catches empty name
    bad_spec = AblationSpec(name="", toggles={"a": [1]})
    errs = bad_spec.validate()
    _assert(any("name" in e.lower() for e in errs), "Should catch empty name")

    # Test validation catches invalid mode
    bad_spec2 = AblationSpec(name="x", toggles={"a": [1]}, mode="invalid")
    errs = bad_spec2.validate()
    _assert(any("mode" in e.lower() for e in errs), "Should catch invalid mode")

    # Test validation catches empty toggles
    bad_spec3 = AblationSpec(name="x", toggles={})
    errs = bad_spec3.validate()
    _assert(any("toggle" in e.lower() for e in errs), "Should catch empty toggles")

    # Test validation against known config fields
    known = {"use_engram", "engram_mode", "use_snn"}
    errs = spec.validate(known_toggles=known)
    _assert(len(errs) == 0, f"Should pass with known toggles: {errs}")

    bad_spec4 = AblationSpec(name="x", toggles={"unknown_field": [1, 2]})
    errs = bad_spec4.validate(known_toggles=known)
    _assert(any("unknown" in e.lower() for e in errs), "Should catch unknown toggle field")

    # Test constraint referencing unknown toggle
    bad_spec5 = AblationSpec(
        name="x",
        toggles={"a": [1, 2]},
        constraints=[{"if": {"a": 1}, "then": {"nonexistent": None}}],
    )
    errs = bad_spec5.validate()
    _assert(any("nonexistent" in e for e in errs), "Should catch constraint referencing unknown toggle")

    # ------------------------------------------------------------------
    # YAML/JSON save/load round-trip
    # ------------------------------------------------------------------
    print("\n--- Spec Save/Load Round-Trip ---")

    with tempfile.TemporaryDirectory() as tmpdir:
        # JSON round-trip
        json_path = os.path.join(tmpdir, "spec.json")
        spec.save(json_path)
        loaded = AblationSpec.load(json_path)
        _assert(loaded.name == spec.name, "JSON round-trip: name preserved")
        _assert(loaded.toggles == spec.toggles, "JSON round-trip: toggles preserved")
        _assert(loaded.constraints == spec.constraints, "JSON round-trip: constraints preserved")
        _assert(loaded.mode == spec.mode, "JSON round-trip: mode preserved")
        _assert(loaded.seeds == spec.seeds, "JSON round-trip: seeds preserved")
        _assert(loaded.phases == spec.phases, "JSON round-trip: phases preserved")

        # YAML round-trip (if pyyaml available)
        if _HAS_YAML:
            yaml_path = os.path.join(tmpdir, "spec.yaml")
            spec.save(yaml_path)
            loaded_yaml = AblationSpec.load(yaml_path)
            _assert(loaded_yaml.name == spec.name, "YAML round-trip: name preserved")
            _assert(loaded_yaml.toggles == spec.toggles, "YAML round-trip: toggles preserved")
            _assert(loaded_yaml.seeds == spec.seeds, "YAML round-trip: seeds preserved")
        else:
            print("  (skipping YAML tests -- pyyaml not installed)")

    # ------------------------------------------------------------------
    # Full matrix generation (2x2x2 = 8 combos)
    # ------------------------------------------------------------------
    print("\n--- Full Matrix Generation ---")

    simple_spec = AblationSpec(
        name="simple_full",
        toggles={
            "use_a": [False, True],
            "use_b": [False, True],
            "use_c": [False, True],
        },
        seeds=[42],
    )
    gen = MatrixGenerator()
    combos = gen.generate_full(simple_spec)
    _assert(len(combos) == 8, f"2x2x2 should produce 8 combos, got {len(combos)}")

    # Verify all 8 combinations are unique
    combo_strs = [json.dumps(c, sort_keys=True) for c in combos]
    _assert(len(set(combo_strs)) == 8, "All 8 combos should be unique")

    # ------------------------------------------------------------------
    # Constraint filtering
    # ------------------------------------------------------------------
    print("\n--- Constraint Filtering ---")

    combos_constrained = gen.generate_full(spec)  # 3 toggles with engram constraint

    # Without constraint: 2 * 2 * 2 = 8
    # With constraint: when use_engram=False, engram_mode pinned to "encoder"
    # So: use_engram=False gives 1 * 2 = 2 combos (engram_mode forced to "encoder")
    #     use_engram=True  gives 2 * 2 = 4 combos
    # Total: 6
    _assert(
        len(combos_constrained) == 6,
        f"Constraint should reduce 8 to 6, got {len(combos_constrained)}"
    )

    # Verify no combo has use_engram=False with engram_mode="layer"
    bad_combos = [
        c for c in combos_constrained
        if c.get("use_engram") is False and c.get("engram_mode") == "layer"
    ]
    _assert(len(bad_combos) == 0, "No combo should have engram=False + mode=layer")

    # ------------------------------------------------------------------
    # Seed expansion
    # ------------------------------------------------------------------
    print("\n--- Seed Expansion ---")

    seeds = [1337, 42, 7]
    expanded = gen.expand_seeds(combos_constrained, seeds)
    _assert(
        len(expanded) == len(combos_constrained) * len(seeds),
        f"Seed expansion: {len(combos_constrained)} * {len(seeds)} = "
        f"{len(combos_constrained) * len(seeds)}, got {len(expanded)}"
    )

    # Verify each entry has __seed__
    _assert(all("__seed__" in e for e in expanded), "All expanded entries should have __seed__")

    # ------------------------------------------------------------------
    # Pairwise covering array
    # ------------------------------------------------------------------
    print("\n--- Pairwise Covering Array ---")

    pw_toggles = {
        "a": [False, True],
        "b": [False, True],
        "c": [False, True],
        "d": [False, True],
        "e": [False, True],
    }

    pca = PairwiseCoveringArray(rng_seed=42)
    pw_combos = pca.generate(pw_toggles)

    # Full would be 2^5 = 32; pairwise should be much smaller
    _assert(
        len(pw_combos) < 32,
        f"Pairwise should be < 32, got {len(pw_combos)}"
    )
    _assert(
        len(pw_combos) >= 4,
        f"Pairwise should be >= 4, got {len(pw_combos)}"
    )

    # Verify all pairs are covered
    try:
        PairwiseCoveringArray.verify_coverage(pw_combos, pw_toggles)
        _assert(True, "Pairwise coverage verified")
    except AssertionError as e:
        _assert(False, f"Pairwise coverage failed: {e}")

    # Pairwise with mixed value counts
    mixed_toggles = {
        "binary_a": [False, True],
        "binary_b": [False, True],
        "ternary": ["low", "mid", "high"],
    }
    pw_mixed = pca.generate(mixed_toggles)
    try:
        PairwiseCoveringArray.verify_coverage(pw_mixed, mixed_toggles)
        _assert(True, "Mixed-value pairwise coverage verified")
    except AssertionError as e:
        _assert(False, f"Mixed-value pairwise coverage failed: {e}")

    # Pairwise with single toggle
    single_toggle = {"only": [1, 2, 3]}
    pw_single = pca.generate(single_toggle)
    _assert(len(pw_single) == 3, f"Single toggle should produce 3, got {len(pw_single)}")

    # Pairwise via MatrixGenerator
    pw_spec = AblationSpec(
        name="pw_test",
        toggles=pw_toggles,
        mode="pairwise",
        seeds=[42],
    )
    pw_runs = gen.generate(pw_spec)
    _assert(len(pw_runs) > 0, "Pairwise via MatrixGenerator should produce runs")
    _assert(len(pw_runs) < 32, f"Pairwise via MatrixGenerator should be < 32, got {len(pw_runs)}")

    # ------------------------------------------------------------------
    # AblationRunIDDeriver determinism
    # ------------------------------------------------------------------
    print("\n--- Run ID Derivation ---")

    deriver = AblationRunIDDeriver

    # Ablation ID is deterministic for same inputs
    aid1 = deriver.derive_ablation_id("test", timestamp="20260220_120000",
                                       spec_dict={"a": 1})
    aid2 = deriver.derive_ablation_id("test", timestamp="20260220_120000",
                                       spec_dict={"a": 1})
    _assert(aid1 == aid2, "Ablation ID should be deterministic")

    # Different specs produce different hashes
    aid3 = deriver.derive_ablation_id("test", timestamp="20260220_120000",
                                       spec_dict={"a": 2})
    _assert(aid1 != aid3, "Different spec should produce different ablation ID")

    # Run ID is deterministic
    rid1 = deriver.derive_run_id("abl_001", 0, {"use_a": True, "use_b": False}, 42)
    rid2 = deriver.derive_run_id("abl_001", 0, {"use_a": True, "use_b": False}, 42)
    _assert(rid1 == rid2, "Run ID should be deterministic for same inputs")

    # Different overrides produce different IDs
    rid3 = deriver.derive_run_id("abl_001", 0, {"use_a": False, "use_b": True}, 42)
    _assert(rid1 != rid3, "Different overrides should produce different run ID")

    # Different seeds produce different IDs
    rid4 = deriver.derive_run_id("abl_001", 0, {"use_a": True, "use_b": False}, 99)
    _assert(rid1 != rid4, "Different seeds should produce different run ID")

    # Override insertion order does not affect ID
    rid5 = deriver.derive_run_id("abl_001", 0, {"use_b": False, "use_a": True}, 42)
    _assert(rid1 == rid5, "Override insertion order should not affect run ID")

    # ID contains run index
    _assert("run000" in rid1, f"Run ID should contain run index: {rid1}")

    # ID contains seed
    _assert("_42" in rid1, f"Run ID should contain seed: {rid1}")

    # Sanitization
    _assert(deriver._sanitize("My Test!@#123") == "my_test___123",
            "Sanitize should lowercase and replace special chars")

    # ------------------------------------------------------------------
    # AblationExecutor sequential with mock training
    # ------------------------------------------------------------------
    print("\n--- AblationExecutor Sequential ---")

    exec_spec = AblationSpec(
        name="exec_test",
        toggles={
            "use_a": [False, True],
            "use_b": [False, True],
        },
        seeds=[42, 99],
        phases=[4],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        executor = AblationExecutor(spec=exec_spec, base_dir=tmpdir)
        matrix = gen.generate(exec_spec)
        _assert(len(matrix) == 8, f"2x2 x 2 seeds = 8 runs, got {len(matrix)}")

        report = executor.execute_sequential(matrix)
        _assert(len(report.runs) == 8, f"Report should have 8 results, got {len(report.runs)}")

        completed = [r for r in report.runs if r.status == "completed"]
        _assert(len(completed) == 8, f"All 8 should complete, got {len(completed)}")

        # Verify run directories were created
        for r in report.runs:
            run_dir = os.path.join(tmpdir, r.ablation_id, r.run_id)
            _assert(os.path.isdir(run_dir), f"Run directory should exist: {r.run_id}")
            manifest = os.path.join(run_dir, "run_manifest.json")
            _assert(os.path.isfile(manifest), f"Manifest should exist: {r.run_id}")

        # Verify metrics are present
        for r in completed:
            _assert("best_val_loss" in r.metrics, f"Metrics should include best_val_loss: {r.run_id}")
            _assert("best_val_acc" in r.metrics, f"Metrics should include best_val_acc: {r.run_id}")

    # ------------------------------------------------------------------
    # Failure handling
    # ------------------------------------------------------------------
    print("\n--- Failure Handling ---")

    call_count = 0

    def _failing_train(overrides: Dict, seed: int, phases: List, run_dir: str) -> Dict:
        nonlocal call_count
        call_count += 1
        if call_count == 3:
            raise RuntimeError("Simulated training failure")
        rng = random.Random(seed)
        return {
            "best_val_loss": rng.uniform(0.2, 0.5),
            "best_val_acc": rng.uniform(0.7, 0.95),
            "final_train_loss": rng.uniform(0.1, 0.3),
            "final_val_loss": rng.uniform(0.2, 0.5),
        }

    fail_spec = AblationSpec(
        name="fail_test",
        toggles={"x": [1, 2], "y": [3, 4]},
        seeds=[10],
        phases=[1],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        call_count = 0
        executor = AblationExecutor(spec=fail_spec, base_dir=tmpdir,
                                     train_fn=_failing_train)
        matrix = gen.generate(fail_spec)
        report = executor.execute_sequential(matrix)

        total = len(report.runs)
        comp = sum(1 for r in report.runs if r.status == "completed")
        fail = sum(1 for r in report.runs if r.status == "failed")

        _assert(total == 4, f"Should have 4 total runs, got {total}")
        _assert(comp == 3, f"Should have 3 completed, got {comp}")
        _assert(fail == 1, f"Should have 1 failed, got {fail}")

        # Verify the failed run has an error message
        failed_runs = [r for r in report.runs if r.status == "failed"]
        _assert(
            len(failed_runs) == 1 and "Simulated" in (failed_runs[0].error or ""),
            "Failed run should contain error message"
        )

    # ------------------------------------------------------------------
    # AblationReport CSV output
    # ------------------------------------------------------------------
    print("\n--- CSV Output ---")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Use the successful exec_test results
        executor = AblationExecutor(spec=exec_spec, base_dir=tmpdir)
        matrix = gen.generate(exec_spec)
        report = executor.execute_sequential(matrix)

        csv_path = os.path.join(tmpdir, "ablations.csv")
        report.save_csv(csv_path)

        _assert(os.path.isfile(csv_path), "CSV file should be created")

        rows = AblationCSVWriter.load_csv(csv_path)
        _assert(len(rows) == 8, f"CSV should have 8 rows, got {len(rows)}")

        # Check columns
        expected_cols = AblationCSVWriter.fieldnames(["use_a", "use_b"])
        actual_cols = list(rows[0].keys())
        _assert(
            set(actual_cols) == set(expected_cols),
            f"CSV columns mismatch: expected {expected_cols}, got {actual_cols}"
        )

        # Check run_id is present in every row
        _assert(all(row.get("run_id") for row in rows), "Every row should have a run_id")

        # Check status column
        _assert(
            all(row.get("status") == "completed" for row in rows),
            "All rows should have status=completed"
        )

    # ------------------------------------------------------------------
    # CSV append mode
    # ------------------------------------------------------------------
    print("\n--- CSV Append ---")

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "append_test.csv")
        toggle_keys = ["use_a", "use_b"]
        phases = [4]

        # Write first row
        result1 = AblationRunResult(
            run_id="run_001", ablation_id="abl_001",
            overrides={"use_a": True, "use_b": False},
            seed=42, phases=[4], status="completed",
            metrics={"best_val_loss": 0.3, "best_val_acc": 0.85,
                     "final_train_loss": 0.2, "final_val_loss": 0.35},
        )
        AblationCSVWriter.append_row(csv_path, result1, toggle_keys, phases)
        rows = AblationCSVWriter.load_csv(csv_path)
        _assert(len(rows) == 1, f"After first append: 1 row expected, got {len(rows)}")

        # Append second row
        result2 = AblationRunResult(
            run_id="run_002", ablation_id="abl_001",
            overrides={"use_a": False, "use_b": True},
            seed=99, phases=[4], status="completed",
            metrics={"best_val_loss": 0.25, "best_val_acc": 0.90,
                     "final_train_loss": 0.15, "final_val_loss": 0.28},
        )
        AblationCSVWriter.append_row(csv_path, result2, toggle_keys, phases)
        rows = AblationCSVWriter.load_csv(csv_path)
        _assert(len(rows) == 2, f"After second append: 2 rows expected, got {len(rows)}")

    # ------------------------------------------------------------------
    # JSON output
    # ------------------------------------------------------------------
    print("\n--- JSON Output ---")

    with tempfile.TemporaryDirectory() as tmpdir:
        executor = AblationExecutor(spec=exec_spec, base_dir=tmpdir)
        matrix = gen.generate(exec_spec)
        report = executor.execute_sequential(matrix)

        json_path = os.path.join(tmpdir, "summary.json")
        report.save_json(json_path)

        _assert(os.path.isfile(json_path), "JSON file should be created")
        with open(json_path) as fh:
            summary = json.load(fh)

        _assert(summary["total_runs"] == 8, f"JSON total_runs should be 8, got {summary['total_runs']}")
        _assert(summary["completed"] == 8, f"JSON completed should be 8, got {summary['completed']}")
        _assert("best_run" in summary, "JSON should contain best_run")
        _assert("toggle_effects" in summary, "JSON should contain toggle_effects")

    # ------------------------------------------------------------------
    # Toggle effect computation
    # ------------------------------------------------------------------
    print("\n--- Toggle Effects ---")

    # Create synthetic results with clear effect direction
    synth_results: List[AblationRunResult] = []
    for i, (a, b) in enumerate(itertools.product([False, True], repeat=2)):
        base_loss = 0.4
        if a:
            base_loss -= 0.1  # use_a helps
        if b:
            base_loss -= 0.05  # use_b helps a little
        for seed in [42, 99]:
            synth_results.append(AblationRunResult(
                run_id=f"synth_{i}_{seed}",
                ablation_id="synth_abl",
                overrides={"use_a": a, "use_b": b},
                seed=seed,
                phases=[1],
                status="completed",
                metrics={"best_val_loss": base_loss + random.Random(seed).uniform(-0.01, 0.01)},
            ))

    synth_report = AblationReport(runs=synth_results, toggle_keys=["use_a", "use_b"])
    effects = synth_report.compute_toggle_effects(metric="best_val_loss")

    _assert("use_a" in effects, "Toggle effects should include use_a")
    _assert("use_b" in effects, "Toggle effects should include use_b")

    # use_a should have a negative effect (lower loss = better)
    _assert(
        effects["use_a"]["effect_size"] < 0,
        f"use_a effect should be negative (helps), got {effects['use_a']['effect_size']}"
    )

    # use_a effect should be larger than use_b
    _assert(
        abs(effects["use_a"]["effect_size"]) > abs(effects["use_b"]["effect_size"]),
        "use_a should have larger absolute effect than use_b"
    )

    # ------------------------------------------------------------------
    # Best run selection
    # ------------------------------------------------------------------
    print("\n--- Best Run Selection ---")

    best_min = synth_report.get_best_run(metric="best_val_loss", mode="min")
    _assert(best_min is not None, "Should find a best run (min)")
    _assert(
        best_min.overrides.get("use_a") is True,
        f"Best run (min loss) should have use_a=True, got {best_min.overrides}"
    )

    best_max = synth_report.get_best_run(metric="best_val_loss", mode="max")
    _assert(best_max is not None, "Should find a best run (max)")
    _assert(
        best_max.overrides.get("use_a") is False,
        f"Worst run (max loss) should have use_a=False, got {best_max.overrides}"
    )

    # ------------------------------------------------------------------
    # Report summary string
    # ------------------------------------------------------------------
    print("\n--- Report Summary ---")

    summary_str = synth_report.summary()
    _assert("Total runs" in summary_str, "Summary should mention total runs")
    _assert("Completed" in summary_str, "Summary should mention completed")
    _assert("use_a" in summary_str, "Summary should mention toggle use_a")

    # ------------------------------------------------------------------
    # MatrixGenerator.generate() end-to-end
    # ------------------------------------------------------------------
    print("\n--- End-to-End generate() ---")

    e2e_spec = AblationSpec(
        name="e2e_test",
        toggles={
            "use_engram": [False, True],
            "engram_mode": ["encoder", "layer"],
            "use_snn": [False, True],
        },
        constraints=[
            {"if": {"use_engram": False}, "then": {"engram_mode": None}},
        ],
        mode="full",
        phases=[4, 5],
        seeds=[1337, 42],
    )

    runs = gen.generate(e2e_spec)
    # 6 combos (after constraint) * 2 seeds = 12 runs
    _assert(len(runs) == 12, f"E2E should produce 12 runs, got {len(runs)}")

    # All runs should have unique IDs
    run_ids = [r.run_id for r in runs]
    _assert(len(set(run_ids)) == len(run_ids), "All run IDs should be unique")

    # All runs should have the correct ablation_id
    ablation_ids = set(r.ablation_id for r in runs)
    _assert(len(ablation_ids) == 1, "All runs should share the same ablation_id")

    # All runs should have phases [4, 5]
    _assert(all(r.phases == [4, 5] for r in runs), "All runs should have phases [4, 5]")

    # All runs should be pending
    _assert(all(r.status == "pending" for r in runs), "All runs should start as pending")

    # ------------------------------------------------------------------
    # estimate_total_runs
    # ------------------------------------------------------------------
    print("\n--- Estimate Total Runs ---")

    est = gen.estimate_total_runs(e2e_spec)
    _assert(est == 12, f"Estimated total should be 12, got {est}")

    est_pw = gen.estimate_total_runs(AblationSpec(
        name="est_pw",
        toggles=pw_toggles,
        mode="pairwise",
        seeds=[42],
    ))
    _assert(est_pw < 32, f"Pairwise estimate should be < 32, got {est_pw}")
    _assert(est_pw > 0, f"Pairwise estimate should be > 0, got {est_pw}")

    # ------------------------------------------------------------------
    # AblationRun dataclass
    # ------------------------------------------------------------------
    print("\n--- AblationRun Dataclass ---")

    run = AblationRun(
        run_index=0,
        run_id="test_run_000_abc123_42",
        ablation_id="test_abl_20260220_120000_abcd1234",
        overrides={"use_a": True, "use_b": False},
        seed=42,
        phases=[4, 5],
        baseline_run_id="baseline_001",
        status="pending",
    )
    d = run.to_dict()
    _assert(d["run_index"] == 0, "to_dict should preserve run_index")
    _assert(d["run_id"] == "test_run_000_abc123_42", "to_dict should preserve run_id")
    _assert(d["status"] == "pending", "to_dict should preserve status")
    _assert(d["baseline_run_id"] == "baseline_001", "to_dict should preserve baseline_run_id")

    # ------------------------------------------------------------------
    # Pairwise vs Full comparison
    # ------------------------------------------------------------------
    print("\n--- Pairwise vs Full Comparison ---")

    cmp_toggles = {
        "t1": [False, True],
        "t2": [False, True],
        "t3": [False, True],
        "t4": [False, True],
        "t5": [False, True],
        "t6": [False, True],
    }
    cmp_spec_full = AblationSpec(name="cmp", toggles=cmp_toggles, mode="full", seeds=[1])
    cmp_spec_pw = AblationSpec(name="cmp", toggles=cmp_toggles, mode="pairwise", seeds=[1])

    full_count = gen.estimate_total_runs(cmp_spec_full)
    pw_count = gen.estimate_total_runs(cmp_spec_pw)

    _assert(full_count == 64, f"Full 2^6 should be 64, got {full_count}")
    _assert(pw_count < full_count, f"Pairwise ({pw_count}) should be < full ({full_count})")
    _assert(pw_count <= 16, f"Pairwise for 6 binary should be <= 16, got {pw_count}")

    reduction = full_count / max(pw_count, 1)
    print(f"  Full: {full_count}, Pairwise: {pw_count}, Reduction: {reduction:.1f}x")

    # ------------------------------------------------------------------
    # AblationRunResult to_dict
    # ------------------------------------------------------------------
    print("\n--- AblationRunResult ---")

    rr = AblationRunResult(
        run_id="rr_001", ablation_id="abl_001",
        overrides={"x": 1}, seed=42, phases=[1],
        status="completed", metrics={"best_val_loss": 0.3},
        duration_seconds=120.5, error=None, git_sha="abc123",
    )
    rr_d = rr.to_dict()
    _assert(rr_d["run_id"] == "rr_001", "RunResult to_dict: run_id")
    _assert(rr_d["metrics"]["best_val_loss"] == 0.3, "RunResult to_dict: metrics")
    _assert(rr_d["duration_seconds"] == 120.5, "RunResult to_dict: duration")

    # ------------------------------------------------------------------
    # CSV with failure rows
    # ------------------------------------------------------------------
    print("\n--- CSV with Failures ---")

    with tempfile.TemporaryDirectory() as tmpdir:
        call_count = 0
        executor = AblationExecutor(spec=fail_spec, base_dir=tmpdir,
                                     train_fn=_failing_train)
        matrix = gen.generate(fail_spec)
        report = executor.execute_sequential(matrix)

        csv_path = os.path.join(tmpdir, "ablations.csv")
        report.save_csv(csv_path)

        rows = AblationCSVWriter.load_csv(csv_path)
        _assert(len(rows) == 4, f"CSV with failures should have 4 rows, got {len(rows)}")

        failed_rows = [r for r in rows if r["status"] == "failed"]
        _assert(len(failed_rows) == 1, f"Should have 1 failed row, got {len(failed_rows)}")
        _assert(
            failed_rows[0]["error_message"] != "",
            "Failed row should have error_message"
        )

    # ------------------------------------------------------------------
    # Empty spec edge cases
    # ------------------------------------------------------------------
    print("\n--- Edge Cases ---")

    # Single toggle, single value
    single_spec = AblationSpec(name="single", toggles={"only": [42]}, seeds=[1])
    single_combos = gen.generate_full(single_spec)
    _assert(len(single_combos) == 1, f"Single value toggle: 1 combo, got {len(single_combos)}")

    # Report with no completed runs
    empty_report = AblationReport(runs=[
        AblationRunResult(run_id="f1", ablation_id="a", overrides={}, seed=1,
                          phases=[1], status="failed", error="oops"),
    ])
    best_empty = empty_report.get_best_run()
    _assert(best_empty is None, "No best run when all failed")

    effects_empty = empty_report.compute_toggle_effects()
    _assert(len(effects_empty) == 0, "No effects when all failed")

    summary_empty = empty_report.summary()
    _assert("Failed" in summary_empty, "Summary should mention failures")

    # ------------------------------------------------------------------
    # Constraint with explicit then value (not None)
    # ------------------------------------------------------------------
    print("\n--- Explicit Then Constraint ---")

    explicit_spec = AblationSpec(
        name="explicit",
        toggles={
            "mode": ["fast", "slow"],
            "depth": [1, 2, 3],
        },
        constraints=[
            {"if": {"mode": "fast"}, "then": {"depth": 1}},
        ],
    )
    explicit_combos = gen.generate_full(explicit_spec)
    # mode=fast -> depth must be 1 (1 combo)
    # mode=slow -> depth can be 1,2,3 (3 combos)
    # Total: 4
    _assert(len(explicit_combos) == 4, f"Explicit constraint: 4 combos, got {len(explicit_combos)}")

    fast_combos = [c for c in explicit_combos if c["mode"] == "fast"]
    _assert(
        all(c["depth"] == 1 for c in fast_combos),
        "All fast combos should have depth=1"
    )

    # ------------------------------------------------------------------
    # MatrixGenerator pairwise with constraints
    # ------------------------------------------------------------------
    print("\n--- Pairwise with Constraints ---")

    pw_constrained_spec = AblationSpec(
        name="pw_constrained",
        toggles={
            "use_engram": [False, True],
            "engram_mode": ["encoder", "layer"],
            "use_snn": [False, True],
        },
        constraints=[
            {"if": {"use_engram": False}, "then": {"engram_mode": None}},
        ],
        mode="pairwise",
        seeds=[42],
    )
    pw_constrained = gen.generate(pw_constrained_spec)
    for r in pw_constrained:
        if r.overrides.get("use_engram") is False:
            _assert(
                r.overrides.get("engram_mode") == "encoder",
                f"Pairwise constraint violated: {r.overrides}"
            )
    _assert(len(pw_constrained) > 0, "Pairwise with constraints should produce runs")

    # ------------------------------------------------------------------
    # Report toggle effect with 3+ values
    # ------------------------------------------------------------------
    print("\n--- Multi-Value Toggle Effects ---")

    mv_results: List[AblationRunResult] = []
    for depth in [1, 2, 3]:
        loss = 0.5 - depth * 0.1
        for seed in [10, 20]:
            mv_results.append(AblationRunResult(
                run_id=f"mv_{depth}_{seed}", ablation_id="mv",
                overrides={"depth": depth}, seed=seed, phases=[1],
                status="completed",
                metrics={"best_val_loss": loss + random.Random(seed).uniform(-0.01, 0.01)},
            ))

    mv_report = AblationReport(runs=mv_results, toggle_keys=["depth"])
    mv_effects = mv_report.compute_toggle_effects()
    _assert("depth" in mv_effects, "Multi-value toggle should appear in effects")
    _assert(
        mv_effects["depth"]["effect_size"] < 0,
        "Deeper should have lower loss (negative effect size)"
    )

    # ------------------------------------------------------------------
    # Full generate pipeline determinism
    # ------------------------------------------------------------------
    print("\n--- Pipeline Determinism ---")

    det_spec = AblationSpec(
        name="det_test",
        toggles={"a": [1, 2], "b": [3, 4]},
        seeds=[42, 99],
        mode="full",
    )
    runs_a = gen.generate(det_spec)
    runs_b = gen.generate(det_spec)

    ids_a = [r.run_id for r in runs_a]
    ids_b = [r.run_id for r in runs_b]
    _assert(ids_a == ids_b, "generate() should be deterministic across calls")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"Self-tests complete: {passed} passed, {failed} failed")
    if errors:
        print("\nFailures:")
        for e in errors:
            print(f"  {e}")
    print("=" * 70)

    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    _run_self_tests()
