"""
TrialManager — Track, checkpoint, and analyze hyperparameter search trials.

Provides logging, retrieval, CSV export, parameter importance analysis, and
optional plotting.  Trials are persisted to disk as JSON for durability.

No external dependencies beyond stdlib + math.  Plotting is optional (matplotlib).

Usage:
    from trial_manager_template import TrialManager, TrialRecord

    mgr = TrialManager(study_dir="/tmp/my_study", metric="val_loss", direction="minimize")
    mgr.log_trial(params={"lr": 0.01}, metrics={"val_loss": 0.5}, trial_id="t_001")
    best = mgr.get_best_trials(n=5)
    importance = mgr.get_importance()
    mgr.export_csv("/tmp/results.csv")
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# TrialRecord
# ---------------------------------------------------------------------------

@dataclass
class TrialRecord:
    """A single trial's full record."""
    trial_id: str
    params: Dict[str, Any]
    metrics: Dict[str, float]
    status: str = "complete"       # complete | failed | pruned
    timestamp: float = 0.0        # epoch time
    duration: float = 0.0
    tags: Dict[str, str] = field(default_factory=dict)

    @property
    def primary_metric(self) -> Optional[float]:
        """First metric value (convenience)."""
        if self.metrics:
            return next(iter(self.metrics.values()))
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "params": self.params,
            "metrics": self.metrics,
            "status": self.status,
            "timestamp": self.timestamp,
            "duration": self.duration,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrialRecord":
        return cls(
            trial_id=d["trial_id"],
            params=d.get("params", {}),
            metrics=d.get("metrics", {}),
            status=d.get("status", "complete"),
            timestamp=d.get("timestamp", 0.0),
            duration=d.get("duration", 0.0),
            tags=d.get("tags", {}),
        )


# ---------------------------------------------------------------------------
# TrialManager
# ---------------------------------------------------------------------------

class TrialManager:
    """Manage hyperparameter search trials.

    Parameters
    ----------
    study_dir : Directory to persist trial data (JSON).
    metric : Primary metric name for ranking (e.g. "val_loss").
    direction : "minimize" or "maximize".
    """

    def __init__(self, study_dir: str, metric: str = "val_loss",
                 direction: str = "minimize"):
        self.study_dir = study_dir
        self.metric = metric
        self.direction = direction
        self._trials: Dict[str, TrialRecord] = {}
        self._counter = 0
        os.makedirs(study_dir, exist_ok=True)
        # Load existing trials
        self._load()

    # -- Logging ------------------------------------------------------------

    def log_trial(self, params: Dict[str, Any], metrics: Dict[str, float],
                  trial_id: Optional[str] = None,
                  status: str = "complete",
                  duration: float = 0.0,
                  tags: Optional[Dict[str, str]] = None) -> TrialRecord:
        """Log a trial.

        Parameters
        ----------
        params : Hyperparameter dict.
        metrics : Metric dict (must include self.metric key).
        trial_id : If None, auto-generated.
        """
        if trial_id is None:
            trial_id = f"trial_{self._counter:06d}"
            self._counter += 1
        else:
            self._counter = max(self._counter, _parse_trial_num(trial_id) + 1)

        rec = TrialRecord(
            trial_id=trial_id,
            params=dict(params),
            metrics=dict(metrics),
            status=status,
            timestamp=time.time(),
            duration=duration,
            tags=tags or {},
        )
        self._trials[trial_id] = rec
        self._persist_trial(rec)
        return rec

    # -- Retrieval ----------------------------------------------------------

    def get_trial(self, trial_id: str) -> Optional[TrialRecord]:
        return self._trials.get(trial_id)

    def get_all_trials(self) -> List[TrialRecord]:
        return list(self._trials.values())

    def get_best_trials(self, n: int = 5,
                        metric: Optional[str] = None) -> List[TrialRecord]:
        """Return the top-n trials ranked by the given metric."""
        m = metric or self.metric
        completed = [t for t in self._trials.values()
                     if t.status == "complete" and m in t.metrics
                     and math.isfinite(t.metrics[m])]
        reverse = self.direction == "maximize"
        completed.sort(key=lambda t: t.metrics[m], reverse=reverse)
        return completed[:n]

    @property
    def n_trials(self) -> int:
        return len(self._trials)

    @property
    def n_complete(self) -> int:
        return sum(1 for t in self._trials.values() if t.status == "complete")

    @property
    def n_failed(self) -> int:
        return sum(1 for t in self._trials.values() if t.status == "failed")

    # -- Importance ---------------------------------------------------------

    def get_importance(self, metric: Optional[str] = None,
                       method: str = "correlation") -> Dict[str, float]:
        """Estimate per-parameter importance via correlation with the metric.

        Uses absolute Pearson correlation between each param and the objective.
        Only numeric params are included.

        Parameters
        ----------
        metric : Which metric (default: self.metric).
        method : "correlation" (only method implemented).

        Returns
        -------
        Dict mapping param names to importance scores in [0, 1], normalized.
        """
        m = metric or self.metric
        completed = [t for t in self._trials.values()
                     if t.status == "complete" and m in t.metrics
                     and math.isfinite(t.metrics[m])]
        if len(completed) < 3:
            return {}

        y_vals = [t.metrics[m] for t in completed]

        # Gather all numeric param names
        param_names = set()
        for t in completed:
            for k, v in t.params.items():
                if isinstance(v, (int, float)):
                    param_names.add(k)

        importance: Dict[str, float] = {}
        for pname in param_names:
            x_vals = []
            y_matched = []
            for t in completed:
                if pname in t.params and isinstance(t.params[pname], (int, float)):
                    x_vals.append(float(t.params[pname]))
                    y_matched.append(t.metrics[m])
            if len(x_vals) < 3:
                importance[pname] = 0.0
                continue
            corr = abs(_pearson_correlation(x_vals, y_matched))
            importance[pname] = corr

        # Normalize so importances sum to ~1
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}

        # Sort descending
        importance = dict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True))
        return importance

    # -- Export -------------------------------------------------------------

    def export_csv(self, path: str) -> None:
        """Export all trials to CSV.

        Columns: trial_id, status, duration, <all param names>, <all metric names>.
        """
        if not self._trials:
            with open(path, "w", newline="") as f:
                f.write("trial_id\n")
            return

        # Collect all unique param and metric names
        all_params: List[str] = []
        all_metrics: List[str] = []
        param_set: set = set()
        metric_set: set = set()
        for t in self._trials.values():
            for k in t.params:
                if k not in param_set:
                    param_set.add(k)
                    all_params.append(k)
            for k in t.metrics:
                if k not in metric_set:
                    metric_set.add(k)
                    all_metrics.append(k)

        header = ["trial_id", "status", "duration"] + all_params + all_metrics

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for t in self._trials.values():
                row = [t.trial_id, t.status, f"{t.duration:.4f}"]
                for p in all_params:
                    row.append(str(t.params.get(p, "")))
                for m in all_metrics:
                    val = t.metrics.get(m, "")
                    row.append(str(val) if val != "" else "")
                writer.writerow(row)

    # -- Plotting -----------------------------------------------------------

    def plot_optimization_history(self, metric: Optional[str] = None):
        """Plot cumulative best metric over trials.

        Returns matplotlib Figure if available, else data dict.
        """
        m = metric or self.metric
        completed = [t for t in self._trials.values()
                     if t.status == "complete" and m in t.metrics]
        completed.sort(key=lambda t: t.timestamp)
        if not completed:
            return {"trial_ids": [], "cumulative_best": []}

        cum_best = []
        best_so_far = None
        for t in completed:
            val = t.metrics[m]
            if best_so_far is None:
                best_so_far = val
            elif self.direction == "minimize" and val < best_so_far:
                best_so_far = val
            elif self.direction == "maximize" and val > best_so_far:
                best_so_far = val
            cum_best.append(best_so_far)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            xs = list(range(1, len(cum_best) + 1))
            ax.plot(xs, cum_best, marker="o", markersize=3)
            ax.set_xlabel("Trial #")
            ax.set_ylabel(f"Best {m}")
            ax.set_title("Optimization History")
            plt.tight_layout()
            return fig
        except ImportError:
            return {
                "trial_ids": [t.trial_id for t in completed],
                "cumulative_best": cum_best,
            }

    def plot_param_importances(self, metric: Optional[str] = None):
        """Bar chart of parameter importances."""
        imp = self.get_importance(metric=metric)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(8, max(4, len(imp) * 0.4)))
            names = list(imp.keys())
            vals = list(imp.values())
            ax.barh(names, vals)
            ax.set_xlabel("Importance")
            ax.set_title("Parameter Importances")
            plt.tight_layout()
            return fig
        except ImportError:
            return imp

    # -- Persistence --------------------------------------------------------

    def _persist_trial(self, rec: TrialRecord) -> None:
        """Write a single trial to a JSON file."""
        trial_file = os.path.join(self.study_dir, f"{rec.trial_id}.json")
        with open(trial_file, "w") as f:
            json.dump(rec.to_dict(), f, indent=2, default=str)

    def _load(self) -> None:
        """Load all trials from study_dir."""
        if not os.path.isdir(self.study_dir):
            return
        for fname in sorted(os.listdir(self.study_dir)):
            if fname.endswith(".json"):
                fpath = os.path.join(self.study_dir, fname)
                try:
                    with open(fpath, "r") as f:
                        data = json.load(f)
                    rec = TrialRecord.from_dict(data)
                    self._trials[rec.trial_id] = rec
                    self._counter = max(self._counter, _parse_trial_num(rec.trial_id) + 1)
                except (json.JSONDecodeError, KeyError):
                    pass

    def save_all(self) -> None:
        """Re-persist all trials."""
        for rec in self._trials.values():
            self._persist_trial(rec)

    def clear(self) -> None:
        """Remove all trials from memory and disk."""
        for fname in os.listdir(self.study_dir):
            if fname.endswith(".json"):
                os.remove(os.path.join(self.study_dir, fname))
        self._trials.clear()
        self._counter = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_trial_num(trial_id: str) -> int:
    """Extract numeric suffix from trial ID."""
    parts = trial_id.rsplit("_", 1)
    if len(parts) == 2:
        try:
            return int(parts[1])
        except ValueError:
            pass
    return 0


def _pearson_correlation(x: List[float], y: List[float]) -> float:
    """Pearson correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0.0
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    cov = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    std_x = math.sqrt(sum((xi - x_mean) ** 2 for xi in x))
    std_y = math.sqrt(sum((yi - y_mean) ** 2 for yi in y))
    if std_x < 1e-12 or std_y < 1e-12:
        return 0.0
    return cov / (std_x * std_y)


# ---------------------------------------------------------------------------
# Self-tests  (30+ tests)
# ---------------------------------------------------------------------------

def _run_tests():
    import tempfile
    import random

    passed = 0
    failed = 0

    def check(name: str, condition: bool, msg: str = ""):
        nonlocal passed, failed
        if condition:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL: {name} — {msg}")

    rng = random.Random(42)

    print("=== TM-01: Log single trial ===")
    with tempfile.TemporaryDirectory() as d:
        mgr = TrialManager(d, metric="val_loss", direction="minimize")
        rec = mgr.log_trial(params={"lr": 0.01, "layers": 4},
                            metrics={"val_loss": 0.5, "val_acc": 0.8},
                            trial_id="t_001")
        got = mgr.get_trial("t_001")
        check("TM-01 logged", got is not None)
        check("TM-01 params", got.params == {"lr": 0.01, "layers": 4})
        check("TM-01 metrics", got.metrics["val_loss"] == 0.5)

    print("\n=== TM-02: Log multiple trials ===")
    with tempfile.TemporaryDirectory() as d:
        mgr = TrialManager(d, metric="val_loss", direction="minimize")
        for i in range(50):
            mgr.log_trial(
                params={"lr": rng.uniform(1e-4, 1e-1), "n": rng.randint(2, 12)},
                metrics={"val_loss": rng.uniform(0.1, 1.0)},
            )
        check("TM-02 count", mgr.n_trials == 50)

    print("\n=== TM-03: Trial IDs unique ===")
    with tempfile.TemporaryDirectory() as d:
        mgr = TrialManager(d, metric="val_loss", direction="minimize")
        for i in range(100):
            mgr.log_trial(params={"x": i}, metrics={"val_loss": float(i)})
        ids = [t.trial_id for t in mgr.get_all_trials()]
        check("TM-03 unique IDs", len(ids) == len(set(ids)))

    print("\n=== TM-04: Custom trial ID ===")
    with tempfile.TemporaryDirectory() as d:
        mgr = TrialManager(d, metric="val_loss", direction="minimize")
        mgr.log_trial(params={"x": 1}, metrics={"val_loss": 0.5}, trial_id="custom_01")
        check("TM-04 custom ID", mgr.get_trial("custom_01") is not None)

    print("\n=== TM-05: Overwrite trial ===")
    with tempfile.TemporaryDirectory() as d:
        mgr = TrialManager(d, metric="val_loss", direction="minimize")
        mgr.log_trial(params={"x": 1}, metrics={"val_loss": 0.9}, trial_id="x")
        mgr.log_trial(params={"x": 2}, metrics={"val_loss": 0.1}, trial_id="x")
        check("TM-05 overwrite", mgr.get_trial("x").params["x"] == 2)

    print("\n=== TM-06: get_best_trials(n=1) minimize ===")
    with tempfile.TemporaryDirectory() as d:
        mgr = TrialManager(d, metric="val_loss", direction="minimize")
        for i in range(50):
            mgr.log_trial(params={"x": i}, metrics={"val_loss": float(i)})
        best = mgr.get_best_trials(n=1)
        check("TM-06 best is min", len(best) == 1 and best[0].metrics["val_loss"] == 0.0)

    print("\n=== TM-07: get_best_trials(n=5) ===")
    with tempfile.TemporaryDirectory() as d:
        mgr = TrialManager(d, metric="val_loss", direction="minimize")
        for i in range(50):
            mgr.log_trial(params={"x": i}, metrics={"val_loss": float(i)})
        best5 = mgr.get_best_trials(n=5)
        check("TM-07 returns 5", len(best5) == 5)
        vals = [t.metrics["val_loss"] for t in best5]
        check("TM-07 sorted", vals == sorted(vals))

    print("\n=== TM-08: get_best_trials maximize ===")
    with tempfile.TemporaryDirectory() as d:
        mgr = TrialManager(d, metric="val_acc", direction="maximize")
        for i in range(50):
            mgr.log_trial(params={"x": i}, metrics={"val_acc": float(i)})
        best = mgr.get_best_trials(n=1)
        check("TM-08 maximize best", best[0].metrics["val_acc"] == 49.0)

    print("\n=== TM-09: get_best_trials more than available ===")
    with tempfile.TemporaryDirectory() as d:
        mgr = TrialManager(d, metric="val_loss", direction="minimize")
        for i in range(50):
            mgr.log_trial(params={"x": i}, metrics={"val_loss": float(i)})
        best100 = mgr.get_best_trials(n=100)
        check("TM-09 capped at 50", len(best100) == 50)

    print("\n=== TM-10: Tied metric values ===")
    with tempfile.TemporaryDirectory() as d:
        mgr = TrialManager(d, metric="val_loss", direction="minimize")
        for i in range(10):
            mgr.log_trial(params={"x": i}, metrics={"val_loss": 1.0})
        best_tied = mgr.get_best_trials(n=5)
        check("TM-10 tied", len(best_tied) == 5)

    print("\n=== TM-11: export_csv basic ===")
    with tempfile.TemporaryDirectory() as d:
        mgr = TrialManager(d, metric="val_loss", direction="minimize")
        for i in range(10):
            mgr.log_trial(params={"lr": 0.01 * i, "n": i},
                          metrics={"val_loss": float(i), "val_acc": float(i) / 10})
        csv_path = os.path.join(d, "results.csv")
        mgr.export_csv(csv_path)
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
        check("TM-11 header", len(rows[0]) > 3)
        check("TM-11 data rows", len(rows) == 11)  # header + 10

    print("\n=== TM-12: CSV has all columns ===")
    with tempfile.TemporaryDirectory() as d:
        mgr = TrialManager(d, metric="val_loss", direction="minimize")
        mgr.log_trial(params={"a": 1, "b": 2}, metrics={"loss": 0.5, "acc": 0.9})
        csv_path = os.path.join(d, "out.csv")
        mgr.export_csv(csv_path)
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
        check("TM-12 columns",
              "a" in header and "b" in header and "loss" in header and "acc" in header)

    print("\n=== TM-13: CSV round-trip ===")
    with tempfile.TemporaryDirectory() as d:
        mgr = TrialManager(d, metric="val_loss", direction="minimize")
        mgr.log_trial(params={"lr": 0.01}, metrics={"val_loss": 0.42}, trial_id="t_000")
        csv_path = os.path.join(d, "rt.csv")
        mgr.export_csv(csv_path)
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        check("TM-13 round-trip", abs(float(rows[0]["val_loss"]) - 0.42) < 1e-6)

    print("\n=== TM-14: Export empty ===")
    with tempfile.TemporaryDirectory() as d:
        mgr = TrialManager(d, metric="val_loss", direction="minimize")
        csv_path = os.path.join(d, "empty.csv")
        mgr.export_csv(csv_path)
        check("TM-14 empty file", os.path.exists(csv_path))

    print("\n=== TM-15: Special chars in CSV ===")
    with tempfile.TemporaryDirectory() as d:
        mgr = TrialManager(d, metric="val_loss", direction="minimize")
        mgr.log_trial(params={"desc": 'hello, "world"'},
                      metrics={"val_loss": 0.5})
        csv_path = os.path.join(d, "special.csv")
        mgr.export_csv(csv_path)
        with open(csv_path, "r") as f:
            content = f.read()
        check("TM-15 special chars", "hello" in content)

    print("\n=== TM-16: Importance with correlated param ===")
    with tempfile.TemporaryDirectory() as d:
        mgr = TrialManager(d, metric="val_loss", direction="minimize")
        for i in range(100):
            x1 = rng.uniform(0, 10)
            x2 = rng.uniform(0, 10)
            # val_loss = x1 + noise (x1 is important, x2 is noise)
            val = x1 + rng.gauss(0, 0.5)
            mgr.log_trial(params={"x1": x1, "x2": x2},
                          metrics={"val_loss": val})
        imp = mgr.get_importance()
        check("TM-16 x1 > x2", imp.get("x1", 0) > imp.get("x2", 0),
              f"x1={imp.get('x1', 0):.3f}, x2={imp.get('x2', 0):.3f}")

    print("\n=== TM-17: Importance sums to ~1 ===")
    if imp:
        total = sum(imp.values())
        check("TM-17 sum ~1", abs(total - 1.0) < 0.2, f"sum={total:.3f}")

    print("\n=== TM-18: Importance with constant params ===")
    with tempfile.TemporaryDirectory() as d:
        mgr = TrialManager(d, metric="val_loss", direction="minimize")
        for i in range(30):
            mgr.log_trial(params={"x": 5.0}, metrics={"val_loss": rng.uniform(0, 1)})
        imp_const = mgr.get_importance()
        check("TM-18 constant param", imp_const.get("x", 0) < 0.01,
              f"x_imp={imp_const.get('x', 0):.3f}")

    print("\n=== TM-19: Importance with 50 trials ===")
    with tempfile.TemporaryDirectory() as d:
        mgr = TrialManager(d, metric="val_loss", direction="minimize")
        for i in range(50):
            x1 = rng.uniform(0, 10)
            x2 = rng.uniform(0, 10)
            mgr.log_trial(params={"x1": x1, "x2": x2},
                          metrics={"val_loss": x1 * 2 + rng.gauss(0, 1)})
        imp50 = mgr.get_importance()
        check("TM-19 non-trivial", len(imp50) >= 2 and any(v > 0.01 for v in imp50.values()))

    print("\n=== TM-20: Importance with 5 trials ===")
    with tempfile.TemporaryDirectory() as d:
        mgr = TrialManager(d, metric="val_loss", direction="minimize")
        for i in range(5):
            mgr.log_trial(params={"x": float(i)}, metrics={"val_loss": float(i)})
        imp5 = mgr.get_importance()
        check("TM-20 few trials", isinstance(imp5, dict))

    print("\n=== TM-21: Persist and reload ===")
    with tempfile.TemporaryDirectory() as d:
        mgr1 = TrialManager(d, metric="val_loss", direction="minimize")
        for i in range(10):
            mgr1.log_trial(params={"x": float(i)}, metrics={"val_loss": float(i)})
        # Create new manager pointing to same dir
        mgr2 = TrialManager(d, metric="val_loss", direction="minimize")
        check("TM-21 reload", mgr2.n_trials == 10)
        check("TM-21 data", mgr2.get_best_trials(1)[0].metrics["val_loss"] == 0.0)

    print("\n=== TM-22: Concurrent writes ===")
    with tempfile.TemporaryDirectory() as d:
        mgr_a = TrialManager(d, metric="val_loss", direction="minimize")
        mgr_b = TrialManager(d, metric="val_loss", direction="minimize")
        mgr_a.log_trial(params={"x": 1}, metrics={"val_loss": 0.1}, trial_id="a_001")
        mgr_b.log_trial(params={"x": 2}, metrics={"val_loss": 0.2}, trial_id="b_001")
        # Reload
        mgr_c = TrialManager(d, metric="val_loss", direction="minimize")
        check("TM-22 concurrent", mgr_c.n_trials == 2)

    print("\n=== TM-23: Large trial count ===")
    with tempfile.TemporaryDirectory() as d:
        mgr = TrialManager(d, metric="val_loss", direction="minimize")
        for i in range(1000):
            mgr.log_trial(params={"x": float(i)}, metrics={"val_loss": float(i)})
        check("TM-23 1000 trials", mgr.n_trials == 1000)
        best = mgr.get_best_trials(1)
        check("TM-23 best correct", best[0].metrics["val_loss"] == 0.0)

    print("\n=== Additional tests ===")

    # Status tracking
    with tempfile.TemporaryDirectory() as d:
        mgr = TrialManager(d, metric="val_loss", direction="minimize")
        mgr.log_trial(params={"x": 1}, metrics={"val_loss": 0.5}, status="complete")
        mgr.log_trial(params={"x": 2}, metrics={"val_loss": 999}, status="failed")
        mgr.log_trial(params={"x": 3}, metrics={"val_loss": 0.8}, status="pruned")
        check("n_complete", mgr.n_complete == 1)
        check("n_failed", mgr.n_failed == 1)
        # get_best_trials only returns complete
        best = mgr.get_best_trials(n=5)
        check("best excludes failed", len(best) == 1)

    # Clear
    with tempfile.TemporaryDirectory() as d:
        mgr = TrialManager(d, metric="val_loss", direction="minimize")
        mgr.log_trial(params={"x": 1}, metrics={"val_loss": 0.5})
        mgr.clear()
        check("clear", mgr.n_trials == 0)

    # TrialRecord serialization
    rec = TrialRecord(trial_id="t1", params={"lr": 0.01},
                      metrics={"val_loss": 0.5}, tags={"phase": "1"})
    d = rec.to_dict()
    rec2 = TrialRecord.from_dict(d)
    check("TrialRecord round-trip", rec2.trial_id == "t1" and rec2.tags["phase"] == "1")

    # Primary metric
    check("primary_metric", rec.primary_metric == 0.5)

    # Plot methods (just check they don't crash)
    with tempfile.TemporaryDirectory() as d:
        mgr = TrialManager(d, metric="val_loss", direction="minimize")
        for i in range(20):
            mgr.log_trial(params={"x": float(i)}, metrics={"val_loss": float(i)})
        hist = mgr.plot_optimization_history()
        check("plot_history", hist is not None)
        imp_plot = mgr.plot_param_importances()
        check("plot_importance", imp_plot is not None)

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    if failed == 0:
        print("ALL TESTS PASSED")
    return failed == 0


if __name__ == "__main__":
    import sys
    success = _run_tests()
    sys.exit(0 if success else 1)
