#!/usr/bin/env python3
"""
run_ablation.py -- CLI script for executing ablation specs from YAML files.

Reads a YAML (or JSON) ablation spec, generates the toggle matrix (full or
pairwise), executes each run with a configurable training function, and
produces ablations.csv + ablation_summary.json.

Self-contained: no brain_ai imports.  Uses inline mock/stub training for
self-test purposes.

Usage:
    python run_ablation.py --spec ablation_spec.yaml
    python run_ablation.py --spec ablation_spec.yaml --dry-run
    python run_ablation.py --spec ablation_spec.yaml --parallel --max-workers 8
    python run_ablation.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import os
import random
import sys
import tempfile
import textwrap
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict, dataclass, field
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
# Dataclasses
# ===================================================================

@dataclass
class AblationRun:
    """Descriptor for a single run within an ablation matrix."""
    run_index: int
    run_id: str
    ablation_id: str
    overrides: Dict[str, Any]
    seed: int
    phases: List[int]
    baseline_run_id: Optional[str] = None
    status: str = "pending"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AblationRunResult:
    """Outcome of a single completed (or failed) ablation run."""
    run_id: str
    ablation_id: str
    overrides: Dict[str, Any]
    seed: int
    phases: List[int]
    status: str
    metrics: Dict[str, float] = field(default_factory=dict)
    duration_seconds: float = 0.0
    error: Optional[str] = None
    baseline_run_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ===================================================================
# AblationSpecLoader (~80 lines)
# ===================================================================

class AblationSpecLoader:
    """Load, validate, and generate ablation spec files."""

    @dataclass
    class AblationSpec:
        """Parsed ablation specification."""
        name: str
        description: str = ""
        base_config: str = "dev"
        toggles: Dict[str, List[Any]] = field(default_factory=dict)
        constraints: List[Dict[str, Any]] = field(default_factory=list)
        mode: str = "full"
        phases: List[int] = field(default_factory=lambda: [1])
        seeds: List[int] = field(default_factory=lambda: [1337])
        baseline_run_id: Optional[str] = None
        metrics: Dict[str, Any] = field(default_factory=lambda: {
            "primary": "best_val_loss", "secondary": []
        })
        resource_limits: Dict[str, Any] = field(default_factory=dict)

        def to_dict(self) -> Dict[str, Any]:
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

    @staticmethod
    def load(yaml_path: str) -> "AblationSpecLoader.AblationSpec":
        """Parse a YAML or JSON spec file into an AblationSpec.

        Detects format by file extension: .yaml/.yml use PyYAML, everything
        else falls back to JSON.
        """
        with open(yaml_path, "r") as fh:
            if _HAS_YAML and yaml_path.endswith((".yaml", ".yml")):
                data = _yaml.safe_load(fh)
            else:
                data = json.load(fh)

        ab = data.get("ablation", data)
        spec = AblationSpecLoader.AblationSpec(
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

        errors = AblationSpecLoader.validate(spec)
        if errors:
            raise ValueError(
                f"Invalid ablation spec '{yaml_path}':\n  " + "\n  ".join(errors)
            )
        return spec

    @staticmethod
    def validate(spec: "AblationSpecLoader.AblationSpec") -> List[str]:
        """Validate spec: required fields, valid toggle values. Returns error list."""
        errors: List[str] = []
        if not spec.name:
            errors.append("Spec name must be non-empty.")
        if spec.mode not in ("full", "pairwise"):
            errors.append(f"Invalid mode {spec.mode!r}; must be 'full' or 'pairwise'.")
        if not spec.toggles:
            errors.append("At least one toggle must be specified.")
        for tname, tvals in spec.toggles.items():
            if not isinstance(tvals, list) or len(tvals) < 1:
                errors.append(f"Toggle {tname!r} must have a list of at least one value.")
        if not spec.seeds:
            errors.append("At least one seed must be specified.")
        if not spec.phases:
            errors.append("At least one phase must be specified.")
        # Check constraints reference valid toggles
        toggle_names = set(spec.toggles.keys())
        for i, rule in enumerate(spec.constraints):
            for key in list(rule.get("if", {}).keys()) + list(rule.get("then", {}).keys()):
                if key not in toggle_names:
                    errors.append(f"Constraint #{i} references unknown toggle {key!r}.")
        return errors

    @staticmethod
    def load_from_string(content: str, fmt: str = "yaml") -> "AblationSpecLoader.AblationSpec":
        """Parse a spec from a string (for testing). fmt is 'yaml' or 'json'."""
        if fmt == "yaml" and _HAS_YAML:
            data = _yaml.safe_load(content)
        else:
            data = json.loads(content)
        ab = data.get("ablation", data)
        spec = AblationSpecLoader.AblationSpec(
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
        errors = AblationSpecLoader.validate(spec)
        if errors:
            raise ValueError("Invalid spec:\n  " + "\n  ".join(errors))
        return spec

    @staticmethod
    def generate_example_spec(output_path: str) -> None:
        """Write an example ablation spec YAML file with comments."""
        example = textwrap.dedent("""\
        # Ablation Spec -- Example
        # This file defines an automatic ablation experiment.
        ablation:
          # Human-readable name for this ablation experiment
          name: example_ablation

          # Free-text description
          description: "Example ablation over engram and SNN toggles"

          # Base configuration preset: "dev" (fast, MNIST) or "production"
          base_config: dev

          # Toggles to sweep: each key maps to a list of values
          toggles:
            use_engram: [false, true]
            use_snn: [false, true]
            use_htm: [false, true]

          # Constraints: if-then rules to prune invalid combinations
          # 'then: null' pins the toggle to its first listed value
          constraints: []

          # Matrix mode: "full" (Cartesian product) or "pairwise" (covering array)
          mode: full

          # Training phases to execute per combination
          phases: [4]

          # Seeds for multi-seed statistical analysis
          seeds: [1337, 42]

          # Metrics to track
          metrics:
            primary: best_val_loss
            secondary: [best_val_acc, final_train_loss]

          # Resource limits (optional)
          resource_limits:
            max_workers: 4
            timeout_per_run: 3600
        """)
        with open(output_path, "w") as fh:
            fh.write(example)


# ===================================================================
# Matrix generation helpers
# ===================================================================

AblationSpec = AblationSpecLoader.AblationSpec


def _derive_ablation_id(name: str, spec_dict: Optional[Dict[str, Any]] = None,
                        timestamp: Optional[str] = None) -> str:
    """Deterministic ablation-level ID: {name}_{timestamp}_{hash[:8]}."""
    sanitized = "".join(
        c if c.isalnum() or c == "_" else "_" for c in name.lower()
    ).strip("_")
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if spec_dict is not None:
        canonical = json.dumps(spec_dict, sort_keys=True, separators=(",", ":"), default=str)
        spec_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]
    else:
        spec_hash = "00000000"
    return f"{sanitized}_{timestamp}_{spec_hash}"


def _derive_run_id(ablation_id: str, run_index: int,
                   overrides: Dict[str, Any], seed: int) -> str:
    """Deterministic per-run ID: {ablation_id}_run{N:03d}_{hash[:6]}_{seed}."""
    clean = {k: v for k, v in sorted(overrides.items())}
    canonical = json.dumps(clean, sort_keys=True, separators=(",", ":"), default=str)
    overrides_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:6]
    return f"{ablation_id}_run{run_index:03d}_{overrides_hash}_{seed}"


def _generate_pairwise(toggles: Dict[str, List[Any]],
                       rng_seed: int = 42) -> List[Dict[str, Any]]:
    """Greedy pairwise covering array generator."""
    rng = random.Random(rng_seed)
    keys = sorted(toggles.keys())
    value_lists = [toggles[k] for k in keys]
    k = len(keys)

    if k == 0:
        return []
    if k == 1:
        return [{keys[0]: v} for v in value_lists[0]]

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
        sample_size = min(len(uncovered), 50)
        sampled = rng.sample(sorted(uncovered), sample_size)
        candidates: List[Tuple[Any, ...]] = []
        for pair in sampled:
            pi, pvi, pj, pvj = pair
            for _ in range(5):
                row = [rng.choice(vl) for vl in value_lists]
                row[pi] = pvi
                row[pj] = pvj
                candidates.append(tuple(row))
        for cand in candidates:
            count = sum(
                1 for i in range(k) for j in range(i + 1, k)
                if (i, cand[i], j, cand[j]) in uncovered
            )
            if count > best_count:
                best_count = count
                best_row = cand
        if best_row is None or best_count == 0:
            for pair in sorted(uncovered):
                pi, pvi, pj, pvj = pair
                row = [rng.choice(vl) for vl in value_lists]
                row[pi] = pvi
                row[pj] = pvj
                best_row = tuple(row)
                break
            if best_row is None:
                break
        covering.append(best_row)
        for i in range(k):
            for j in range(i + 1, k):
                uncovered.discard((i, best_row[i], j, best_row[j]))

    return [dict(zip(keys, row)) for row in covering]


def _apply_constraints(combinations: List[Dict[str, Any]],
                       constraints: List[Dict[str, Any]],
                       toggles: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Filter combinations by if-then constraint rules."""
    if not constraints:
        return combinations
    valid: List[Dict[str, Any]] = []
    for combo in combinations:
        keep = True
        for rule in constraints:
            if_clause = rule.get("if", {})
            then_clause = rule.get("then", {})
            if not all(combo.get(k) == v for k, v in if_clause.items()):
                continue
            for tk, tv in then_clause.items():
                if tv is None:
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


def generate_matrix(spec: AblationSpec, force_pairwise: bool = False,
                    override_seeds: Optional[List[int]] = None) -> List[AblationRun]:
    """Generate the full list of AblationRun descriptors from a spec."""
    seeds = override_seeds if override_seeds is not None else spec.seeds
    use_pairwise = force_pairwise or spec.mode == "pairwise"

    ablation_id = _derive_ablation_id(
        name=spec.name, spec_dict=spec.to_dict()
    )

    if use_pairwise:
        combos = _generate_pairwise(spec.toggles)
    else:
        keys = sorted(spec.toggles.keys())
        value_lists = [spec.toggles[k] for k in keys]
        combos = [dict(zip(keys, vals)) for vals in itertools.product(*value_lists)]

    combos = _apply_constraints(combos, spec.constraints, spec.toggles)

    runs: List[AblationRun] = []
    idx = 0
    for combo in combos:
        for seed in seeds:
            run_id = _derive_run_id(ablation_id, idx, combo, seed)
            runs.append(AblationRun(
                run_index=idx,
                run_id=run_id,
                ablation_id=ablation_id,
                overrides=dict(combo),
                seed=seed,
                phases=list(spec.phases),
                baseline_run_id=spec.baseline_run_id,
                status="pending",
            ))
            idx += 1
    return runs


# ===================================================================
# AblationMatrixPrinter (~40 lines)
# ===================================================================

class AblationMatrixPrinter:
    """Formatted printing for ablation matrices."""

    @staticmethod
    def print_matrix(matrix: List[AblationRun]) -> None:
        """Print a formatted table showing all combinations."""
        if not matrix:
            print("  (empty matrix)")
            return
        toggle_keys = sorted(matrix[0].overrides.keys())
        headers = ["run_idx", "seed"] + toggle_keys
        col_widths = [max(len(h), 8) for h in headers]

        # Measure max widths from data
        for run in matrix:
            vals = [str(run.run_index), str(run.seed)]
            vals += [str(run.overrides.get(k, "")) for k in toggle_keys]
            for i, v in enumerate(vals):
                col_widths[i] = max(col_widths[i], len(v))

        fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
        print(fmt.format(*headers))
        print(fmt.format(*["-" * w for w in col_widths]))
        for run in matrix:
            vals = [str(run.run_index), str(run.seed)]
            vals += [str(run.overrides.get(k, "")) for k in toggle_keys]
            print(fmt.format(*vals))

    @staticmethod
    def print_summary(matrix: List[AblationRun], spec: AblationSpec) -> None:
        """Print summary: 'Total: N runs (K toggles x M seeds)'."""
        n_toggles = len(spec.toggles)
        n_seeds = len(spec.seeds)
        n_combos = len(matrix) // max(n_seeds, 1) if matrix else 0
        print(f"Total: {len(matrix)} runs ({n_combos} combinations x {n_seeds} seeds, "
              f"{n_toggles} toggles)")

    @staticmethod
    def print_comparison(spec: AblationSpec) -> None:
        """Print 'Full: N runs vs Pairwise: ~M runs'."""
        keys = sorted(spec.toggles.keys())
        full_count = 1
        for k in keys:
            full_count *= len(spec.toggles[k])
        full_total = full_count * len(spec.seeds)

        pw_combos = _generate_pairwise(spec.toggles)
        pw_combos = _apply_constraints(pw_combos, spec.constraints, spec.toggles)
        pw_total = len(pw_combos) * len(spec.seeds)

        print(f"Full: {full_total} runs vs Pairwise: ~{pw_total} runs "
              f"({full_total / max(pw_total, 1):.1f}x reduction)")


# ===================================================================
# AblationExecutionEngine (~120 lines)
# ===================================================================

def _mock_train_fn(overrides: Dict[str, Any], seed: int,
                   phases: List[int], run_dir: str) -> Dict[str, float]:
    """Stub training function returning mock metrics (for testing)."""
    rng = random.Random(seed)
    # Slight sensitivity to overrides so ablation effects are visible
    bonus = sum(0.02 for v in overrides.values() if v is True)
    return {
        "best_val_loss": round(rng.uniform(0.2, 0.5) - bonus, 4),
        "best_val_acc": round(rng.uniform(0.7, 0.95) + bonus * 0.5, 4),
        "final_train_loss": round(rng.uniform(0.1, 0.3) - bonus, 4),
        "final_val_loss": round(rng.uniform(0.2, 0.5) - bonus, 4),
    }


class AblationExecutionEngine:
    """Execute an ablation matrix through a training function."""

    def __init__(self, run_dir: str = "runs/", mode: str = "dev",
                 train_fn: Optional[Callable[..., Dict[str, float]]] = None,
                 verbose: bool = False) -> None:
        self.run_dir = run_dir
        self.mode = mode
        self.train_fn = train_fn or _mock_train_fn
        self.verbose = verbose

    def execute(self, matrix: List[AblationRun],
                parallel: bool = False,
                max_workers: int = 4) -> List[AblationRunResult]:
        """Run all ablation combos. Returns list of results."""
        if parallel:
            return self._execute_parallel(matrix, max_workers)
        return self._execute_sequential(matrix)

    def _execute_sequential(self, matrix: List[AblationRun]) -> List[AblationRunResult]:
        """Run one at a time, print progress."""
        results: List[AblationRunResult] = []
        total = len(matrix)
        for run in matrix:
            overrides_str = ", ".join(
                f"{k}={v}" for k, v in sorted(run.overrides.items())
            )
            print(f"  [{run.run_index + 1}/{total}] Running {overrides_str}, "
                  f"seed={run.seed}...")
            result = self._execute_single(run)
            results.append(result)
            status_marker = "OK" if result.status == "completed" else "FAIL"
            if self.verbose and result.status == "completed":
                loss = result.metrics.get("best_val_loss", "N/A")
                print(f"    -> {status_marker} (val_loss={loss}, "
                      f"{result.duration_seconds:.2f}s)")
            elif result.status == "failed":
                print(f"    -> FAIL: {result.error}")
        return results

    def _execute_parallel(self, matrix: List[AblationRun],
                          max_workers: int) -> List[AblationRunResult]:
        """Use ProcessPoolExecutor for parallel runs."""
        results: List[AblationRunResult] = []
        total = len(matrix)
        print(f"  Executing {total} runs in parallel (max_workers={max_workers})...")

        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_map = {}
                for run in matrix:
                    future = executor.submit(self._execute_single, run)
                    future_map[future] = run

                for future in as_completed(future_map):
                    run = future_map[future]
                    try:
                        result = future.result()
                        results.append(result)
                        status_marker = "OK" if result.status == "completed" else "FAIL"
                        print(f"  [{len(results)}/{total}] {run.run_id[:40]}... "
                              f"{status_marker}")
                    except Exception as exc:
                        results.append(AblationRunResult(
                            run_id=run.run_id,
                            ablation_id=run.ablation_id,
                            overrides=run.overrides,
                            seed=run.seed,
                            phases=run.phases,
                            status="failed",
                            error=f"ProcessPoolExecutor error: {exc}",
                        ))
        except Exception:
            # Fall back to sequential if multiprocessing fails
            print("  Warning: parallel execution failed, falling back to sequential.")
            return self._execute_sequential(matrix)

        return results

    def _execute_single(self, run: AblationRun) -> AblationRunResult:
        """Execute a single ablation run."""
        run_path = os.path.join(self.run_dir, run.ablation_id, run.run_id)
        os.makedirs(run_path, exist_ok=True)

        # Save run manifest
        manifest_path = os.path.join(run_path, "run_manifest.json")
        with open(manifest_path, "w") as fh:
            json.dump(run.to_dict(), fh, indent=2, default=str)

        start = time.monotonic()
        try:
            metrics = self.train_fn(
                overrides=run.overrides,
                seed=run.seed,
                phases=run.phases,
                run_dir=run_path,
            )
            duration = time.monotonic() - start
            return AblationRunResult(
                run_id=run.run_id,
                ablation_id=run.ablation_id,
                overrides=run.overrides,
                seed=run.seed,
                phases=run.phases,
                status="completed",
                metrics=metrics,
                duration_seconds=round(duration, 4),
                baseline_run_id=run.baseline_run_id,
            )
        except Exception as exc:
            duration = time.monotonic() - start
            return AblationRunResult(
                run_id=run.run_id,
                ablation_id=run.ablation_id,
                overrides=run.overrides,
                seed=run.seed,
                phases=run.phases,
                status="failed",
                metrics={},
                duration_seconds=round(duration, 4),
                error=str(exc),
                baseline_run_id=run.baseline_run_id,
            )


# ===================================================================
# ResultCollector (~60 lines)
# ===================================================================

class ResultCollector:
    """Collect results as runs complete, and produce CSV/JSON output."""

    def __init__(self, toggle_keys: List[str], phases: List[int]) -> None:
        self.toggle_keys = sorted(toggle_keys)
        self.phases = phases
        self.results: List[AblationRunResult] = []

    def add_result(self, run_id: str, overrides: Dict[str, Any], seed: int,
                   status: str, metrics: Dict[str, float],
                   duration: float, error: Optional[str] = None) -> None:
        """Add a single result entry."""
        self.results.append(AblationRunResult(
            run_id=run_id,
            ablation_id=self.results[0].ablation_id if self.results else "",
            overrides=overrides,
            seed=seed,
            phases=self.phases,
            status=status,
            metrics=metrics,
            duration_seconds=duration,
            error=error,
        ))

    def add_results(self, results: List[AblationRunResult]) -> None:
        """Bulk add results."""
        self.results.extend(results)

    def save_csv(self, path: str) -> None:
        """Write ablations.csv with dynamic toggle columns."""
        prefix = ["run_id", "ablation_id", "seed"]
        suffix = [
            "phase", "best_val_loss", "best_val_acc",
            "final_train_loss", "final_val_loss",
            "status", "duration_seconds", "error_message",
        ]
        fieldnames = prefix + self.toggle_keys + suffix

        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.results:
                row: Dict[str, Any] = {
                    "run_id": r.run_id,
                    "ablation_id": r.ablation_id,
                    "seed": r.seed,
                    "phase": ",".join(str(p) for p in r.phases),
                    "best_val_loss": r.metrics.get("best_val_loss", ""),
                    "best_val_acc": r.metrics.get("best_val_acc", ""),
                    "final_train_loss": r.metrics.get("final_train_loss", ""),
                    "final_val_loss": r.metrics.get("final_val_loss", ""),
                    "status": r.status,
                    "duration_seconds": f"{r.duration_seconds:.4f}",
                    "error_message": r.error or "",
                }
                for tk in self.toggle_keys:
                    row[tk] = r.overrides.get(tk, "")
                writer.writerow(row)

    def save_json(self, path: str) -> None:
        """Write ablation_summary.json."""
        completed = [r for r in self.results if r.status == "completed"]
        failed = [r for r in self.results if r.status == "failed"]
        best = self.get_best_run("best_val_loss", "min")

        summary: Dict[str, Any] = {
            "total_runs": len(self.results),
            "completed": len(completed),
            "failed": len(failed),
        }
        if self.results:
            summary["ablation_id"] = self.results[0].ablation_id
        if best is not None:
            summary["best_run"] = {
                "run_id": best.run_id,
                "overrides": best.overrides,
                "seed": best.seed,
                "metrics": best.metrics,
            }
        if failed:
            summary["failures"] = [
                {"run_id": r.run_id, "error": r.error} for r in failed
            ]
        with open(path, "w") as fh:
            json.dump(summary, fh, indent=2, default=str)

    def print_results(self) -> None:
        """Print a formatted summary table."""
        completed = [r for r in self.results if r.status == "completed"]
        failed = [r for r in self.results if r.status == "failed"]

        print(f"\nAblation Results Summary")
        print(f"  Total runs:  {len(self.results)}")
        print(f"  Completed:   {len(completed)}")
        print(f"  Failed:      {len(failed)}")

        best = self.get_best_run("best_val_loss", "min")
        if best is not None:
            overrides_str = ", ".join(
                f"{k}={v}" for k, v in sorted(best.overrides.items())
            )
            print(f"  Best run:    {best.run_id}")
            print(f"  Best loss:   {best.metrics.get('best_val_loss', 'N/A')}")
            print(f"  Config:      {overrides_str}")

        if failed:
            print(f"  Failed runs:")
            for r in failed:
                print(f"    - {r.run_id}: {r.error}")

    def get_best_run(self, metric: str = "best_val_loss",
                     mode: str = "min") -> Optional[AblationRunResult]:
        """Return best-performing configuration."""
        completed = [r for r in self.results
                     if r.status == "completed" and metric in r.metrics]
        if not completed:
            return None
        if mode == "min":
            return min(completed, key=lambda r: r.metrics[metric])
        return max(completed, key=lambda r: r.metrics[metric])


# ===================================================================
# ExampleSpecs (~60 lines)
# ===================================================================

MINIMAL_SPEC: Dict[str, Any] = {
    "ablation": {
        "name": "minimal_test",
        "description": "Minimal 2-toggle test ablation",
        "base_config": "dev",
        "toggles": {
            "use_engram": [False, True],
            "use_snn": [False, True],
        },
        "constraints": [],
        "mode": "full",
        "phases": [1],
        "seeds": [1337],
        "metrics": {"primary": "best_val_loss", "secondary": []},
    }
}

FULL_SPEC: Dict[str, Any] = {
    "ablation": {
        "name": "brain_ai_full_ablation",
        "description": "Full 5-toggle ablation for brain_ai production evaluation",
        "base_config": "production",
        "toggles": {
            "use_engram": [False, True],
            "use_snn": [False, True],
            "use_htm": [False, True],
            "use_workspace": [False, True],
            "use_ltn": [False, True],
        },
        "constraints": [],
        "mode": "full",
        "phases": [4, 5],
        "seeds": [1337, 42, 7],
        "metrics": {
            "primary": "best_val_loss",
            "secondary": ["best_val_acc", "final_train_loss"],
        },
    }
}

PAIRWISE_SPEC: Dict[str, Any] = {
    "ablation": {
        "name": "brain_ai_pairwise",
        "description": "Pairwise-reduced 5-toggle ablation",
        "base_config": "production",
        "toggles": {
            "use_engram": [False, True],
            "use_snn": [False, True],
            "use_htm": [False, True],
            "use_workspace": [False, True],
            "use_ltn": [False, True],
        },
        "constraints": [],
        "mode": "pairwise",
        "phases": [4],
        "seeds": [1337, 42, 7],
        "metrics": {"primary": "best_val_loss", "secondary": []},
    }
}


# ===================================================================
# CLI Interface (~60 lines)
# ===================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Run ablation experiments from a YAML/JSON spec file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples:
          python run_ablation.py --spec ablation.yaml
          python run_ablation.py --spec ablation.yaml --dry-run
          python run_ablation.py --spec ablation.yaml --parallel --max-workers 8
          python run_ablation.py --spec ablation.yaml --pairwise --seeds 42,99,7
          python run_ablation.py --self-test
          python run_ablation.py --generate-example example_spec.yaml
        """),
    )
    parser.add_argument(
        "--spec", type=str, default=None,
        help="Path to ablation YAML/JSON spec file (required unless --self-test "
             "or --generate-example)",
    )
    parser.add_argument(
        "--run-dir", type=str, default="runs/",
        help="Base directory for run outputs (default: runs/)",
    )
    parser.add_argument(
        "--mode", type=str, default=None, choices=["dev", "production"],
        help="Override mode from spec (dev/production)",
    )
    parser.add_argument(
        "--parallel", action="store_true", default=False,
        help="Enable parallel execution",
    )
    parser.add_argument(
        "--max-workers", type=int, default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Generate matrix and print without executing",
    )
    parser.add_argument(
        "--pairwise", action="store_true", default=False,
        help="Override to pairwise mode even if spec says full",
    )
    parser.add_argument(
        "--seeds", type=str, default=None,
        help="Override seed list (comma-separated, e.g. 42,99,7)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path for ablations.csv (default: runs/<ablation_id>/ablations.csv)",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False,
        help="Verbose output",
    )
    parser.add_argument(
        "--generate-example", type=str, default=None, metavar="PATH",
        help="Generate an example spec YAML file and exit",
    )
    parser.add_argument(
        "--self-test", action="store_true", default=False,
        help="Run built-in self-tests",
    )
    return parser


# ===================================================================
# Main function (~40 lines)
# ===================================================================

def main(argv: Optional[List[str]] = None) -> int:
    """Parse CLI args, load spec, generate matrix, execute, and report."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Handle --generate-example
    if args.generate_example:
        AblationSpecLoader.generate_example_spec(args.generate_example)
        print(f"Example spec written to: {args.generate_example}")
        return 0

    # Handle --self-test
    if args.self_test:
        return _run_self_tests()

    # Require --spec for normal operation
    if args.spec is None:
        parser.error("--spec is required (or use --self-test / --generate-example)")

    # Load spec
    print(f"Loading spec: {args.spec}")
    spec = AblationSpecLoader.load(args.spec)

    # Apply CLI overrides
    override_seeds = None
    if args.seeds:
        override_seeds = [int(s.strip()) for s in args.seeds.split(",")]

    mode = args.mode or spec.base_config

    # Generate matrix
    print(f"Generating matrix (mode={spec.mode}"
          f"{', forced pairwise' if args.pairwise else ''})...")
    matrix = generate_matrix(spec, force_pairwise=args.pairwise,
                             override_seeds=override_seeds)

    # Print matrix info
    AblationMatrixPrinter.print_summary(matrix, spec)
    AblationMatrixPrinter.print_comparison(spec)

    if args.dry_run:
        print("\n--- Dry Run: Matrix ---")
        AblationMatrixPrinter.print_matrix(matrix)
        print(f"\nDry run complete. {len(matrix)} runs would be executed.")
        return 0

    # Execute
    print(f"\nExecuting {len(matrix)} runs (parallel={args.parallel})...")
    engine = AblationExecutionEngine(
        run_dir=args.run_dir, mode=mode, verbose=args.verbose,
    )
    results = engine.execute(matrix, parallel=args.parallel,
                             max_workers=args.max_workers)

    # Collect results
    toggle_keys = sorted(spec.toggles.keys())
    collector = ResultCollector(toggle_keys=toggle_keys, phases=spec.phases)
    collector.add_results(results)

    # Determine output paths
    ablation_id = matrix[0].ablation_id if matrix else "unknown"
    output_dir = os.path.join(args.run_dir, ablation_id)
    os.makedirs(output_dir, exist_ok=True)

    csv_path = args.output or os.path.join(output_dir, "ablations.csv")
    json_path = os.path.join(os.path.dirname(csv_path), "ablation_summary.json")

    # Save outputs
    collector.save_csv(csv_path)
    collector.save_json(json_path)

    # Print summary
    collector.print_results()
    print(f"\nResults saved to:")
    print(f"  CSV:  {csv_path}")
    print(f"  JSON: {json_path}")

    return 0


# ===================================================================
# Self-test block (~150 lines, 30+ tests)
# ===================================================================

def _run_self_tests() -> int:
    """Execute 30+ self-tests covering all major components."""

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
    print("run_ablation.py Self-Tests")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Spec loading from YAML string
    # ------------------------------------------------------------------
    print("\n--- Test: Spec loading from YAML string ---")

    yaml_content = json.dumps(MINIMAL_SPEC)
    spec = AblationSpecLoader.load_from_string(yaml_content, fmt="json")
    _assert(spec.name == "minimal_test", "Spec name parsed correctly")
    _assert(len(spec.toggles) == 2, "Spec has 2 toggles")
    _assert(spec.mode == "full", "Spec mode is full")
    _assert(spec.seeds == [1337], "Spec seeds parsed")
    _assert(spec.phases == [1], "Spec phases parsed")

    # YAML format (if available)
    if _HAS_YAML:
        yaml_str = _yaml.dump(MINIMAL_SPEC)
        spec_yaml = AblationSpecLoader.load_from_string(yaml_str, fmt="yaml")
        _assert(spec_yaml.name == "minimal_test", "YAML string parse: name")
        _assert(spec_yaml.toggles == spec.toggles, "YAML string parse: toggles match JSON")

    # ------------------------------------------------------------------
    # 2. Spec validation
    # ------------------------------------------------------------------
    print("\n--- Test: Spec validation ---")

    errs = AblationSpecLoader.validate(spec)
    _assert(len(errs) == 0, f"Valid spec should pass validation: {errs}")

    bad = AblationSpecLoader.AblationSpec(name="", toggles={})
    errs = AblationSpecLoader.validate(bad)
    _assert(len(errs) >= 2, f"Empty name + empty toggles should produce >=2 errors, got {len(errs)}")

    bad2 = AblationSpecLoader.AblationSpec(name="x", toggles={"a": [1]}, mode="invalid")
    errs2 = AblationSpecLoader.validate(bad2)
    _assert(any("mode" in e.lower() for e in errs2), "Should catch invalid mode")

    # Constraint referencing unknown toggle
    bad3 = AblationSpecLoader.AblationSpec(
        name="x", toggles={"a": [1, 2]},
        constraints=[{"if": {"a": 1}, "then": {"unknown": None}}],
    )
    errs3 = AblationSpecLoader.validate(bad3)
    _assert(any("unknown" in e for e in errs3), "Should catch unknown toggle in constraint")

    # ------------------------------------------------------------------
    # 3. Full matrix generation
    # ------------------------------------------------------------------
    print("\n--- Test: Full matrix generation ---")

    full_spec = AblationSpecLoader.AblationSpec(
        name="full_test",
        toggles={"a": [False, True], "b": [False, True], "c": [False, True]},
        seeds=[42],
    )
    matrix = generate_matrix(full_spec)
    _assert(len(matrix) == 8, f"2x2x2 x 1 seed = 8 runs, got {len(matrix)}")

    # All unique run IDs
    ids = [r.run_id for r in matrix]
    _assert(len(set(ids)) == 8, "All run IDs should be unique")

    # ------------------------------------------------------------------
    # 4. Pairwise matrix generation
    # ------------------------------------------------------------------
    print("\n--- Test: Pairwise matrix generation ---")

    pw_spec = AblationSpecLoader.AblationSpec(
        name="pw_test",
        toggles={
            "t1": [False, True], "t2": [False, True], "t3": [False, True],
            "t4": [False, True], "t5": [False, True],
        },
        mode="pairwise",
        seeds=[42],
    )
    pw_matrix = generate_matrix(pw_spec)
    _assert(len(pw_matrix) < 32, f"Pairwise 5 binary < 32, got {len(pw_matrix)}")
    _assert(len(pw_matrix) >= 4, f"Pairwise 5 binary >= 4, got {len(pw_matrix)}")

    # Verify pairwise coverage
    keys = sorted(pw_spec.toggles.keys())
    k = len(keys)
    covered: Set[Tuple[str, Any, str, Any]] = set()
    for run in pw_matrix:
        for i in range(k):
            for j in range(i + 1, k):
                covered.add((keys[i], run.overrides[keys[i]],
                             keys[j], run.overrides[keys[j]]))
    required: Set[Tuple[str, Any, str, Any]] = set()
    for i in range(k):
        for j in range(i + 1, k):
            for vi in pw_spec.toggles[keys[i]]:
                for vj in pw_spec.toggles[keys[j]]:
                    required.add((keys[i], vi, keys[j], vj))
    _assert(required <= covered, "Pairwise coverage should be complete")

    # ------------------------------------------------------------------
    # 5. Constraint filtering
    # ------------------------------------------------------------------
    print("\n--- Test: Constraint filtering ---")

    constrained_spec = AblationSpecLoader.AblationSpec(
        name="constrained",
        toggles={
            "use_engram": [False, True],
            "engram_mode": ["encoder", "layer"],
        },
        constraints=[{"if": {"use_engram": False}, "then": {"engram_mode": None}}],
        seeds=[42],
    )
    c_matrix = generate_matrix(constrained_spec)
    # Without constraint: 2 * 2 = 4; with: engram=False pins mode=encoder -> 3
    _assert(len(c_matrix) == 3, f"Constrained 2x2 -> 3, got {len(c_matrix)}")
    bad_combos = [r for r in c_matrix
                  if r.overrides.get("use_engram") is False
                  and r.overrides.get("engram_mode") == "layer"]
    _assert(len(bad_combos) == 0, "No invalid constrained combo should exist")

    # Explicit then value
    explicit_spec = AblationSpecLoader.AblationSpec(
        name="explicit",
        toggles={"mode": ["fast", "slow"], "depth": [1, 2, 3]},
        constraints=[{"if": {"mode": "fast"}, "then": {"depth": 1}}],
        seeds=[42],
    )
    ex_matrix = generate_matrix(explicit_spec)
    _assert(len(ex_matrix) == 4, f"Explicit constraint: 4, got {len(ex_matrix)}")

    # ------------------------------------------------------------------
    # 6. Execution engine with mock train function
    # ------------------------------------------------------------------
    print("\n--- Test: Execution engine (sequential) ---")

    with tempfile.TemporaryDirectory() as tmpdir:
        exec_spec = AblationSpecLoader.AblationSpec(
            name="exec_test",
            toggles={"a": [False, True], "b": [False, True]},
            seeds=[42, 99],
            phases=[4],
        )
        matrix = generate_matrix(exec_spec)
        engine = AblationExecutionEngine(run_dir=tmpdir)
        results = engine.execute(matrix, parallel=False)

        _assert(len(results) == 8, f"Should have 8 results, got {len(results)}")
        completed = [r for r in results if r.status == "completed"]
        _assert(len(completed) == 8, f"All 8 should complete, got {len(completed)}")

        for r in completed:
            _assert("best_val_loss" in r.metrics, f"Metrics should have best_val_loss: {r.run_id}")
            _assert(r.duration_seconds >= 0, f"Duration should be non-negative: {r.run_id}")

        # Verify run directories
        for r in results:
            run_path = os.path.join(tmpdir, r.ablation_id, r.run_id)
            _assert(os.path.isdir(run_path), f"Run dir should exist: {r.run_id}")
            _assert(
                os.path.isfile(os.path.join(run_path, "run_manifest.json")),
                f"Manifest should exist: {r.run_id}",
            )

    # ------------------------------------------------------------------
    # 7. Result collection and CSV output
    # ------------------------------------------------------------------
    print("\n--- Test: Result collection and CSV ---")

    with tempfile.TemporaryDirectory() as tmpdir:
        collector = ResultCollector(toggle_keys=["a", "b"], phases=[4])
        collector.add_results(results)

        csv_path = os.path.join(tmpdir, "ablations.csv")
        collector.save_csv(csv_path)
        _assert(os.path.isfile(csv_path), "CSV file should be created")

        with open(csv_path, "r") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        _assert(len(rows) == 8, f"CSV should have 8 rows, got {len(rows)}")
        _assert("a" in rows[0], "CSV should have toggle column 'a'")
        _assert("b" in rows[0], "CSV should have toggle column 'b'")
        _assert("best_val_loss" in rows[0], "CSV should have best_val_loss column")
        _assert(all(r["status"] == "completed" for r in rows), "All rows should be completed")

        # JSON output
        json_path = os.path.join(tmpdir, "summary.json")
        collector.save_json(json_path)
        _assert(os.path.isfile(json_path), "JSON file should be created")
        with open(json_path) as fh:
            summary = json.load(fh)
        _assert(summary["total_runs"] == 8, f"JSON total_runs should be 8")
        _assert(summary["completed"] == 8, f"JSON completed should be 8")
        _assert("best_run" in summary, "JSON should have best_run")

    # ------------------------------------------------------------------
    # 8. Dry-run mode (no execution)
    # ------------------------------------------------------------------
    print("\n--- Test: Dry-run mode ---")

    with tempfile.TemporaryDirectory() as tmpdir:
        spec_path = os.path.join(tmpdir, "test_spec.json")
        with open(spec_path, "w") as fh:
            json.dump(MINIMAL_SPEC, fh)

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        ret = main(["--spec", spec_path, "--dry-run", "--run-dir", tmpdir])
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout

        _assert(ret == 0, "Dry-run should return 0")
        _assert("Dry run complete" in output, "Dry-run should print completion message")
        _assert("4 runs would be executed" in output,
                f"Dry-run should report 4 runs for 2x2x1seed")

    # ------------------------------------------------------------------
    # 9. Parallel mode with mock function
    # ------------------------------------------------------------------
    print("\n--- Test: Parallel execution ---")

    with tempfile.TemporaryDirectory() as tmpdir:
        par_spec = AblationSpecLoader.AblationSpec(
            name="par_test",
            toggles={"x": [1, 2], "y": [3, 4]},
            seeds=[10],
            phases=[1],
        )
        matrix = generate_matrix(par_spec)
        engine = AblationExecutionEngine(run_dir=tmpdir)
        results = engine.execute(matrix, parallel=True, max_workers=2)

        _assert(len(results) == 4, f"Parallel should produce 4 results, got {len(results)}")
        completed = [r for r in results if r.status == "completed"]
        _assert(len(completed) == 4, f"All 4 should complete in parallel, got {len(completed)}")

    # ------------------------------------------------------------------
    # 10. Error handling (one run fails)
    # ------------------------------------------------------------------
    print("\n--- Test: Error handling ---")

    call_counter = {"count": 0}

    def _failing_train(overrides: Dict, seed: int, phases: List, run_dir: str) -> Dict:
        call_counter["count"] += 1
        if call_counter["count"] == 2:
            raise RuntimeError("Simulated training failure")
        rng = random.Random(seed)
        return {"best_val_loss": rng.uniform(0.2, 0.5), "best_val_acc": rng.uniform(0.7, 0.95)}

    with tempfile.TemporaryDirectory() as tmpdir:
        call_counter["count"] = 0
        fail_spec = AblationSpecLoader.AblationSpec(
            name="fail_test",
            toggles={"x": [1, 2], "y": [3, 4]},
            seeds=[10],
            phases=[1],
        )
        matrix = generate_matrix(fail_spec)
        engine = AblationExecutionEngine(run_dir=tmpdir, train_fn=_failing_train)
        results = engine.execute(matrix, parallel=False)

        total = len(results)
        comp = sum(1 for r in results if r.status == "completed")
        fail = sum(1 for r in results if r.status == "failed")

        _assert(total == 4, f"Should have 4 total, got {total}")
        _assert(comp == 3, f"Should have 3 completed, got {comp}")
        _assert(fail == 1, f"Should have 1 failed, got {fail}")

        failed_runs = [r for r in results if r.status == "failed"]
        _assert(
            len(failed_runs) == 1 and "Simulated" in (failed_runs[0].error or ""),
            "Failed run should contain error message",
        )

        # CSV should include the failed run
        collector = ResultCollector(toggle_keys=["x", "y"], phases=[1])
        collector.add_results(results)
        csv_path = os.path.join(tmpdir, "ablations.csv")
        collector.save_csv(csv_path)
        with open(csv_path) as fh:
            rows = list(csv.DictReader(fh))
        failed_rows = [r for r in rows if r["status"] == "failed"]
        _assert(len(failed_rows) == 1, "CSV should have 1 failed row")
        _assert(failed_rows[0]["error_message"] != "", "Failed row should have error_message")

    # ------------------------------------------------------------------
    # 11. Example spec generation
    # ------------------------------------------------------------------
    print("\n--- Test: Example spec generation ---")

    with tempfile.TemporaryDirectory() as tmpdir:
        example_path = os.path.join(tmpdir, "example.yaml")
        AblationSpecLoader.generate_example_spec(example_path)
        _assert(os.path.isfile(example_path), "Example spec file should be created")

        with open(example_path) as fh:
            content = fh.read()
        _assert("toggles:" in content, "Example should contain 'toggles:'")
        _assert("seeds:" in content, "Example should contain 'seeds:'")
        _assert("mode:" in content, "Example should contain 'mode:'")

        # Should be loadable if YAML is available
        if _HAS_YAML:
            loaded = AblationSpecLoader.load(example_path)
            _assert(loaded.name == "example_ablation",
                    f"Example spec name should be 'example_ablation', got {loaded.name!r}")

    # ------------------------------------------------------------------
    # 12. CLI argument parsing
    # ------------------------------------------------------------------
    print("\n--- Test: CLI argument parsing ---")

    parser = build_parser()

    args = parser.parse_args(["--spec", "test.yaml", "--dry-run", "--verbose"])
    _assert(args.spec == "test.yaml", "CLI: --spec parsed")
    _assert(args.dry_run is True, "CLI: --dry-run parsed")
    _assert(args.verbose is True, "CLI: --verbose parsed")
    _assert(args.run_dir == "runs/", "CLI: default run_dir")
    _assert(args.max_workers == 4, "CLI: default max_workers")

    args2 = parser.parse_args([
        "--spec", "s.yaml", "--parallel", "--max-workers", "8",
        "--seeds", "1,2,3", "--pairwise", "--mode", "production",
        "--output", "out.csv",
    ])
    _assert(args2.parallel is True, "CLI: --parallel parsed")
    _assert(args2.max_workers == 8, "CLI: --max-workers parsed")
    _assert(args2.seeds == "1,2,3", "CLI: --seeds parsed")
    _assert(args2.pairwise is True, "CLI: --pairwise parsed")
    _assert(args2.mode == "production", "CLI: --mode parsed")
    _assert(args2.output == "out.csv", "CLI: --output parsed")

    # ------------------------------------------------------------------
    # 13. Seed override
    # ------------------------------------------------------------------
    print("\n--- Test: Seed override ---")

    seed_spec = AblationSpecLoader.AblationSpec(
        name="seed_override",
        toggles={"a": [1, 2]},
        seeds=[42],
    )
    matrix_default = generate_matrix(seed_spec)
    _assert(len(matrix_default) == 2, f"Default seeds: 2 runs, got {len(matrix_default)}")
    _assert(all(r.seed == 42 for r in matrix_default), "Default seed should be 42")

    matrix_overridden = generate_matrix(seed_spec, override_seeds=[10, 20, 30])
    _assert(len(matrix_overridden) == 6, f"3 override seeds: 6 runs, got {len(matrix_overridden)}")

    # ------------------------------------------------------------------
    # 14. Force pairwise override
    # ------------------------------------------------------------------
    print("\n--- Test: Force pairwise override ---")

    full_spec2 = AblationSpecLoader.AblationSpec(
        name="force_pw",
        toggles={
            "t1": [False, True], "t2": [False, True],
            "t3": [False, True], "t4": [False, True],
        },
        mode="full",
        seeds=[42],
    )
    full_matrix = generate_matrix(full_spec2, force_pairwise=False)
    pw_matrix2 = generate_matrix(full_spec2, force_pairwise=True)
    _assert(len(full_matrix) == 16, f"Full 2^4 = 16, got {len(full_matrix)}")
    _assert(len(pw_matrix2) < 16, f"Forced pairwise < 16, got {len(pw_matrix2)}")

    # ------------------------------------------------------------------
    # 15. AblationMatrixPrinter
    # ------------------------------------------------------------------
    print("\n--- Test: Matrix printer ---")

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    AblationMatrixPrinter.print_matrix(matrix_default)
    printer_output = sys.stdout.getvalue()
    sys.stdout = old_stdout
    _assert("run_idx" in printer_output, "Printer should show run_idx header")
    _assert("seed" in printer_output, "Printer should show seed header")

    sys.stdout = io.StringIO()
    AblationMatrixPrinter.print_summary(matrix_default, seed_spec)
    summary_output = sys.stdout.getvalue()
    sys.stdout = old_stdout
    _assert("Total:" in summary_output, "Summary should contain 'Total:'")

    sys.stdout = io.StringIO()
    AblationMatrixPrinter.print_comparison(seed_spec)
    cmp_output = sys.stdout.getvalue()
    sys.stdout = old_stdout
    _assert("Full:" in cmp_output, "Comparison should mention 'Full:'")
    _assert("Pairwise:" in cmp_output, "Comparison should mention 'Pairwise:'")

    # ------------------------------------------------------------------
    # 16. Best run selection
    # ------------------------------------------------------------------
    print("\n--- Test: Best run selection ---")

    collector = ResultCollector(toggle_keys=["a", "b"], phases=[4])
    collector.add_results([
        AblationRunResult("r1", "abl", {"a": True, "b": False}, 42, [4],
                          "completed", {"best_val_loss": 0.3}),
        AblationRunResult("r2", "abl", {"a": False, "b": True}, 42, [4],
                          "completed", {"best_val_loss": 0.1}),
        AblationRunResult("r3", "abl", {"a": True, "b": True}, 42, [4],
                          "failed", error="boom"),
    ])
    best = collector.get_best_run("best_val_loss", "min")
    _assert(best is not None, "Should find a best run")
    _assert(best.run_id == "r2", f"Best run should be r2 (loss=0.1), got {best.run_id}")

    best_max = collector.get_best_run("best_val_loss", "max")
    _assert(best_max is not None and best_max.run_id == "r1",
            "Max best run should be r1 (loss=0.3)")

    # No best when all failed
    fail_collector = ResultCollector(toggle_keys=["a"], phases=[1])
    fail_collector.add_results([
        AblationRunResult("f1", "abl", {"a": True}, 42, [1], "failed", error="err"),
    ])
    _assert(fail_collector.get_best_run() is None, "No best run when all failed")

    # ------------------------------------------------------------------
    # 17. ExampleSpecs are valid
    # ------------------------------------------------------------------
    print("\n--- Test: ExampleSpecs validation ---")

    for name, spec_dict in [("MINIMAL", MINIMAL_SPEC), ("FULL", FULL_SPEC),
                             ("PAIRWISE", PAIRWISE_SPEC)]:
        s = AblationSpecLoader.load_from_string(json.dumps(spec_dict), fmt="json")
        errs = AblationSpecLoader.validate(s)
        _assert(len(errs) == 0, f"{name}_SPEC should be valid: {errs}")

    # ------------------------------------------------------------------
    # 18. End-to-end main() with file spec
    # ------------------------------------------------------------------
    print("\n--- Test: End-to-end main() ---")

    with tempfile.TemporaryDirectory() as tmpdir:
        spec_path = os.path.join(tmpdir, "spec.json")
        with open(spec_path, "w") as fh:
            json.dump(MINIMAL_SPEC, fh)

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        ret = main(["--spec", spec_path, "--run-dir", tmpdir, "--verbose"])
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout

        _assert(ret == 0, f"main() should return 0, got {ret}")
        _assert("Results saved to:" in output, "Should print save paths")

        # Check CSV was created
        # Find ablation directory
        subdirs = [d for d in os.listdir(tmpdir)
                   if os.path.isdir(os.path.join(tmpdir, d)) and d.startswith("minimal")]
        _assert(len(subdirs) >= 1, f"Should create ablation subdir, found {subdirs}")
        if subdirs:
            csv_path = os.path.join(tmpdir, subdirs[0], "ablations.csv")
            _assert(os.path.isfile(csv_path), f"CSV should exist at {csv_path}")

    # ------------------------------------------------------------------
    # 19. Deterministic run ID generation
    # ------------------------------------------------------------------
    print("\n--- Test: Deterministic run IDs ---")

    det_spec = AblationSpecLoader.AblationSpec(
        name="det_test",
        toggles={"a": [1, 2], "b": [3, 4]},
        seeds=[42, 99],
    )
    runs_a = generate_matrix(det_spec)
    runs_b = generate_matrix(det_spec)
    ids_a = [r.run_id for r in runs_a]
    ids_b = [r.run_id for r in runs_b]
    _assert(ids_a == ids_b, "generate_matrix() should be deterministic")

    # ------------------------------------------------------------------
    # 20. Spec file round-trip (JSON)
    # ------------------------------------------------------------------
    print("\n--- Test: Spec file round-trip ---")

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "roundtrip.json")
        original = AblationSpecLoader.AblationSpec(
            name="roundtrip",
            toggles={"x": [1, 2, 3], "y": [False, True]},
            constraints=[{"if": {"y": False}, "then": {"x": 1}}],
            seeds=[10, 20],
            phases=[4, 5],
        )
        with open(json_path, "w") as fh:
            json.dump(original.to_dict(), fh)
        loaded = AblationSpecLoader.load(json_path)
        _assert(loaded.name == "roundtrip", "Round-trip: name")
        _assert(loaded.toggles == original.toggles, "Round-trip: toggles")
        _assert(loaded.constraints == original.constraints, "Round-trip: constraints")
        _assert(loaded.seeds == original.seeds, "Round-trip: seeds")
        _assert(loaded.phases == original.phases, "Round-trip: phases")

    # ------------------------------------------------------------------
    # 21. Print results output
    # ------------------------------------------------------------------
    print("\n--- Test: print_results ---")

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    collector.print_results()
    pr_output = sys.stdout.getvalue()
    sys.stdout = old_stdout
    _assert("Total runs:" in pr_output, "print_results should show total")
    _assert("Completed:" in pr_output, "print_results should show completed")
    _assert("Failed:" in pr_output, "print_results should show failed count")

    # ------------------------------------------------------------------
    # 22. Empty matrix edge case
    # ------------------------------------------------------------------
    print("\n--- Test: Empty matrix edge case ---")

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    AblationMatrixPrinter.print_matrix([])
    empty_output = sys.stdout.getvalue()
    sys.stdout = old_stdout
    _assert("empty" in empty_output.lower(), "Empty matrix should print indication")

    # ------------------------------------------------------------------
    # 23. generate_example via main()
    # ------------------------------------------------------------------
    print("\n--- Test: generate_example via main ---")

    with tempfile.TemporaryDirectory() as tmpdir:
        example_path = os.path.join(tmpdir, "gen_example.yaml")
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        ret = main(["--generate-example", example_path])
        sys.stdout = old_stdout
        _assert(ret == 0, "generate-example should return 0")
        _assert(os.path.isfile(example_path), "Example file should be created via main")

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

    return 1 if failed > 0 else 0


# ===================================================================
# Entry point
# ===================================================================

if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(_run_self_tests())
    else:
        sys.exit(main())
