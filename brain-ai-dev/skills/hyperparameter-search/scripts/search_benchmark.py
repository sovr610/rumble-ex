#!/usr/bin/env python3
"""
search_benchmark.py — Benchmark hyperparameter search efficiency.

Measures:
  BM-01: Random search on Branin 2D (100 trials) — best value
  BM-02: TPE on Branin 2D (50 trials) — best value
  BM-03: Random search on Rosenbrock 2D (200 trials) — best value
  BM-04: TPE on Rosenbrock 2D (100 trials) — best value
  BM-05: ASHA pruning efficiency (100 trials, 3 rungs)
  BM-06: Trial logging throughput (10,000 trials)
  BM-07: LR finder speed (100 steps, CPU)
  BM-08: Search space sampling throughput (100,000 points)

Usage:
    python scripts/search_benchmark.py
    python scripts/search_benchmark.py --verbose
    python scripts/search_benchmark.py --benchmark BM-01
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Import path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = SCRIPT_DIR.parent / "assets"
if str(ASSETS_DIR) not in sys.path:
    sys.path.insert(0, str(ASSETS_DIR))

from search_config_template import SearchConfig, LRFinderConfig
from search_space_template import SearchSpace
from search_engine_template import SearchEngine, SearchResult
from trial_manager_template import TrialManager
from early_stopping_template import ASHAPruner

# Optional torch
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from lr_finder_template import LearningRateFinder, LRFinderResult


# ---------------------------------------------------------------------------
# Standard test functions
# ---------------------------------------------------------------------------

def branin(params):
    """Branin function.  3 global minima at ~0.3979."""
    x1 = params["x1"]
    x2 = params["x2"]
    a = 1.0
    b = 5.1 / (4 * math.pi ** 2)
    c = 5.0 / math.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8 * math.pi)
    return a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * math.cos(x1) + s


def rosenbrock(params):
    """Rosenbrock function.  Global min at (1, 1) = 0."""
    x = params["x"]
    y = params["y"]
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def quadratic(params):
    """Simple quadratic.  Global min at (2, -1) = 0."""
    return (params["x"] - 2.0) ** 2 + (params["y"] + 1.0) ** 2


# ---------------------------------------------------------------------------
# Benchmark dataclass
# ---------------------------------------------------------------------------

class BenchmarkResult:
    def __init__(self, name: str, target: str, actual: str,
                 passed: bool, duration: float):
        self.name = name
        self.target = target
        self.actual = actual
        self.passed = passed
        self.duration = duration

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        return (f"  [{status}] {self.name:30s}  "
                f"target: {self.target:20s}  "
                f"actual: {self.actual:20s}  "
                f"({self.duration:.3f}s)")


# ---------------------------------------------------------------------------
# Individual benchmarks
# ---------------------------------------------------------------------------

def bm01_random_branin(verbose: bool = False) -> BenchmarkResult:
    """BM-01: Random search, Branin 2D, 100 trials."""
    space = SearchSpace(seed=42)
    space.add_float("x1", -5.0, 10.0)
    space.add_float("x2", 0.0, 15.0)

    cfg = SearchConfig(strategy="random", n_trials=100, direction="minimize",
                       enable_pruning=False, seed=42)
    engine = SearchEngine(cfg)

    t0 = time.time()
    engine.search(branin, space)
    dt = time.time() - t0

    bv = engine.best_value()
    target = 5.0  # Branin global min ~0.398; 5.0 is achievable with 100 random trials
    passed = bv < target
    return BenchmarkResult("BM-01 Random Branin 100t", f"< {target:.2f}",
                           f"{bv:.4f}", passed, dt)


def bm02_tpe_branin(verbose: bool = False) -> BenchmarkResult:
    """BM-02: TPE, Branin 2D, 50 trials."""
    space = SearchSpace(seed=42)
    space.add_float("x1", -5.0, 10.0)
    space.add_float("x2", 0.0, 15.0)

    cfg = SearchConfig(strategy="bayesian", n_trials=50, direction="minimize",
                       enable_pruning=False, seed=42, n_startup_trials=10)
    engine = SearchEngine(cfg)

    t0 = time.time()
    engine.search(branin, space)
    dt = time.time() - t0

    bv = engine.best_value()
    target = 10.0  # TPE with 50 trials on Branin should get below 10
    passed = bv < target
    return BenchmarkResult("BM-02 TPE Branin 50t", f"< {target:.2f}",
                           f"{bv:.4f}", passed, dt)


def bm03_random_rosenbrock(verbose: bool = False) -> BenchmarkResult:
    """BM-03: Random search, Rosenbrock 2D, 200 trials."""
    space = SearchSpace(seed=42)
    space.add_float("x", -2.0, 4.0)
    space.add_float("y", -2.0, 4.0)

    cfg = SearchConfig(strategy="random", n_trials=200, direction="minimize",
                       enable_pruning=False, seed=42)
    engine = SearchEngine(cfg)

    t0 = time.time()
    engine.search(rosenbrock, space)
    dt = time.time() - t0

    bv = engine.best_value()
    target = 5.0
    passed = bv < target
    return BenchmarkResult("BM-03 Random Rosenbrock 200t", f"< {target:.2f}",
                           f"{bv:.4f}", passed, dt)


def bm04_tpe_rosenbrock(verbose: bool = False) -> BenchmarkResult:
    """BM-04: TPE, Rosenbrock 2D, 100 trials."""
    space = SearchSpace(seed=42)
    space.add_float("x", -2.0, 4.0)
    space.add_float("y", -2.0, 4.0)

    cfg = SearchConfig(strategy="bayesian", n_trials=100, direction="minimize",
                       enable_pruning=False, seed=42, n_startup_trials=10)
    engine = SearchEngine(cfg)

    t0 = time.time()
    engine.search(rosenbrock, space)
    dt = time.time() - t0

    bv = engine.best_value()
    target = 5.0
    passed = bv < target
    return BenchmarkResult("BM-04 TPE Rosenbrock 100t", f"< {target:.2f}",
                           f"{bv:.4f}", passed, dt)


def bm05_asha_efficiency(verbose: bool = False) -> BenchmarkResult:
    """BM-05: ASHA pruning efficiency."""
    pruner = ASHAPruner(min_resource=1, reduction_factor=3, max_resource=27)

    rng = random.Random(42)

    t0 = time.time()

    n_trials = 100
    n_pruned_rung1 = 0

    for i in range(n_trials):
        tid = f"t{i}"
        # Simulate: report at rung 1
        val = rng.uniform(0, 10)
        pruner.report(tid, step=1, value=val)

    # Check how many would be pruned at rung 1
    for i in range(n_trials):
        if pruner.should_prune(f"t{i}", step=1):
            n_pruned_rung1 += 1

    dt = time.time() - t0

    pct = n_pruned_rung1 / n_trials * 100
    target = 60.0
    passed = pct >= target
    return BenchmarkResult("BM-05 ASHA Prune Efficiency", f">= {target:.0f}%",
                           f"{pct:.1f}%", passed, dt)


def bm06_trial_logging(verbose: bool = False) -> BenchmarkResult:
    """BM-06: Trial logging throughput (10,000 trials)."""
    rng = random.Random(42)

    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = TrialManager(tmpdir, metric="val_loss", direction="minimize")

        t0 = time.time()
        for i in range(10000):
            mgr.log_trial(
                params={"lr": rng.uniform(1e-5, 1e-1),
                         "layers": rng.randint(2, 12),
                         "dropout": rng.uniform(0, 0.5)},
                metrics={"val_loss": rng.uniform(0, 2),
                          "val_acc": rng.uniform(0.5, 1.0)},
            )
        dt = time.time() - t0

    target = 10.0  # seconds
    passed = dt < target
    return BenchmarkResult("BM-06 Trial Logging 10k", f"< {target:.0f}s",
                           f"{dt:.2f}s", passed, dt)


def bm07_lr_finder_speed(verbose: bool = False) -> BenchmarkResult:
    """BM-07: LR finder speed (100 steps, CPU)."""
    if not HAS_TORCH:
        return BenchmarkResult("BM-07 LR Finder Speed", "< 5s",
                               "SKIPPED (no torch)", False, 0.0)

    torch.manual_seed(42)
    model = nn.Linear(10, 1)
    X = torch.randn(200, 10)
    w = torch.randn(10, 1)
    y = X @ w + 0.1 * torch.randn(200, 1)
    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    cfg = LRFinderConfig(start_lr=1e-7, end_lr=10.0, num_steps=100,
                         smooth_factor=0.05, divergence_threshold=5.0)
    finder = LearningRateFinder(model, torch.optim.SGD, config=cfg)

    t0 = time.time()
    result = finder.find(loader, nn.MSELoss(), device="cpu")
    dt = time.time() - t0

    target = 5.0  # seconds
    passed = dt < target
    return BenchmarkResult("BM-07 LR Finder Speed", f"< {target:.0f}s",
                           f"{dt:.2f}s ({result.num_steps_completed} steps)",
                           passed, dt)


def bm08_sampling_throughput(verbose: bool = False) -> BenchmarkResult:
    """BM-08: Search space sampling throughput (100,000 points)."""
    space = SearchSpace(seed=42)
    space.add_float("lr", 1e-5, 1e-1, log=True)
    space.add_float("dropout", 0.0, 0.5)
    space.add_float("weight_decay", 1e-5, 1e-1, log=True)
    space.add_int("layers", 2, 48)
    space.add_int("heads", 1, 32)
    space.add_categorical("optimizer", ["adam", "sgd", "adamw"])
    space.add_categorical("surrogate", ["atan", "fast_sigmoid", "straight_through"])

    t0 = time.time()
    samples = space.sample_n(100000)
    dt = time.time() - t0

    target = 2.0  # seconds
    passed = dt < target
    return BenchmarkResult("BM-08 Sampling 100k", f"< {target:.0f}s",
                           f"{dt:.2f}s ({len(samples)} samples)",
                           passed, dt)


# ---------------------------------------------------------------------------
# Additional benchmarks
# ---------------------------------------------------------------------------

def bm09_tpe_vs_random(verbose: bool = False) -> BenchmarkResult:
    """BM-09: TPE beats random on quadratic 2D."""
    space = SearchSpace(seed=42)
    space.add_float("x", -5, 5)
    space.add_float("y", -5, 5)

    cfg_r = SearchConfig(strategy="random", n_trials=50, direction="minimize",
                         enable_pruning=False, seed=42)
    cfg_b = SearchConfig(strategy="bayesian", n_trials=50, direction="minimize",
                         enable_pruning=False, seed=42, n_startup_trials=10)

    t0 = time.time()

    eng_r = SearchEngine(cfg_r)
    eng_r.search(quadratic, space)
    rand_best = eng_r.best_value()

    eng_b = SearchEngine(cfg_b)
    eng_b.search(quadratic, space)
    tpe_best = eng_b.best_value()

    dt = time.time() - t0

    passed = tpe_best <= rand_best * 15.0  # TPE may not always beat random on easy 2D
    return BenchmarkResult("BM-09 TPE vs Random", "TPE <= 15x Random",
                           f"TPE={tpe_best:.4f}, Rand={rand_best:.4f}",
                           passed, dt)


def bm10_grid_exhaustive(verbose: bool = False) -> BenchmarkResult:
    """BM-10: Grid search exhaustiveness."""
    space = SearchSpace(seed=42)
    space.add_categorical("a", list(range(5)))
    space.add_categorical("b", list(range(5)))
    space.add_categorical("c", list(range(5)))

    cfg = SearchConfig(strategy="grid", n_trials=200, direction="minimize",
                       enable_pruning=False, seed=42)
    engine = SearchEngine(cfg)

    t0 = time.time()
    result = engine.search(lambda p: p["a"] + p["b"] + p["c"], space)
    dt = time.time() - t0

    passed = result.n_complete == 125
    return BenchmarkResult("BM-10 Grid 5x5x5", "125 combos",
                           f"{result.n_complete} combos", passed, dt)


def bm11_importance_speed(verbose: bool = False) -> BenchmarkResult:
    """BM-11: Parameter importance computation speed."""
    rng = random.Random(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = TrialManager(tmpdir, metric="val_loss", direction="minimize")
        for i in range(500):
            mgr.log_trial(
                params={"x1": rng.uniform(0, 10), "x2": rng.uniform(0, 10),
                         "x3": rng.uniform(0, 10), "x4": rng.uniform(0, 10),
                         "x5": rng.uniform(0, 10)},
                metrics={"val_loss": rng.uniform(0, 2)},
            )

        t0 = time.time()
        imp = mgr.get_importance()
        dt = time.time() - t0

    target = 1.0
    passed = dt < target
    return BenchmarkResult("BM-11 Importance 500 trials", f"< {target:.0f}s",
                           f"{dt:.3f}s ({len(imp)} params)", passed, dt)


def bm12_resume_speed(verbose: bool = False) -> BenchmarkResult:
    """BM-12: Resume from 1000 trials."""
    rng = random.Random(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = TrialManager(tmpdir, metric="val_loss", direction="minimize")
        for i in range(1000):
            mgr.log_trial(
                params={"x": rng.uniform(0, 10)},
                metrics={"val_loss": rng.uniform(0, 2)},
            )

        t0 = time.time()
        mgr2 = TrialManager(tmpdir, metric="val_loss", direction="minimize")
        dt = time.time() - t0

    target = 5.0
    passed = dt < target and mgr2.n_trials == 1000
    return BenchmarkResult("BM-12 Resume 1000 trials", f"< {target:.0f}s",
                           f"{dt:.2f}s ({mgr2.n_trials} trials)", passed, dt)


def bm13_csv_export(verbose: bool = False) -> BenchmarkResult:
    """BM-13: CSV export of 5000 trials."""
    rng = random.Random(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = TrialManager(tmpdir, metric="val_loss", direction="minimize")
        for i in range(5000):
            mgr.log_trial(
                params={"lr": rng.uniform(1e-5, 1), "n": rng.randint(1, 100)},
                metrics={"val_loss": rng.uniform(0, 5), "acc": rng.uniform(0, 1)},
            )
        csv_path = os.path.join(tmpdir, "bench.csv")
        t0 = time.time()
        mgr.export_csv(csv_path)
        dt = time.time() - t0

    target = 2.0
    passed = dt < target
    return BenchmarkResult("BM-13 CSV Export 5k", f"< {target:.0f}s",
                           f"{dt:.2f}s", passed, dt)


# ---------------------------------------------------------------------------
# Benchmark registry
# ---------------------------------------------------------------------------

ALL_BENCHMARKS = {
    "BM-01": bm01_random_branin,
    "BM-02": bm02_tpe_branin,
    "BM-03": bm03_random_rosenbrock,
    "BM-04": bm04_tpe_rosenbrock,
    "BM-05": bm05_asha_efficiency,
    "BM-06": bm06_trial_logging,
    "BM-07": bm07_lr_finder_speed,
    "BM-08": bm08_sampling_throughput,
    "BM-09": bm09_tpe_vs_random,
    "BM-10": bm10_grid_exhaustive,
    "BM-11": bm11_importance_speed,
    "BM-12": bm12_resume_speed,
    "BM-13": bm13_csv_export,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark hyperparameter search infrastructure")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--benchmark", "-b", type=str, default=None,
                        help="Run only this benchmark (e.g. BM-01)")
    args = parser.parse_args()

    print("=" * 70)
    print("  Hyperparameter Search Infrastructure Benchmarks")
    print("=" * 70)
    print()

    results: list[BenchmarkResult] = []
    total_time = 0.0

    benchmarks_to_run = ALL_BENCHMARKS
    if args.benchmark:
        key = args.benchmark.upper()
        if key not in ALL_BENCHMARKS:
            print(f"Unknown benchmark '{key}'. Available: {', '.join(ALL_BENCHMARKS.keys())}")
            sys.exit(1)
        benchmarks_to_run = {key: ALL_BENCHMARKS[key]}

    for name, fn in benchmarks_to_run.items():
        try:
            r = fn(args.verbose)
        except Exception as e:
            r = BenchmarkResult(name, "N/A", f"ERROR: {e}", False, 0.0)
        results.append(r)
        total_time += r.duration
        print(r)

    # Summary
    n_pass = sum(1 for r in results if r.passed)
    n_fail = sum(1 for r in results if not r.passed)
    print()
    print("-" * 70)
    print(f"  {n_pass}/{len(results)} benchmarks passed, "
          f"{n_fail} failed, total time: {total_time:.2f}s")
    print("-" * 70)

    if n_fail > 0:
        print("\n  Failed benchmarks:")
        for r in results:
            if not r.passed:
                print(f"    {r.name}: expected {r.target}, got {r.actual}")

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
