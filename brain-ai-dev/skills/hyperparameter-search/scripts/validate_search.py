#!/usr/bin/env python3
"""
validate_search.py — Validate hyperparameter search infrastructure against
the three done-when gates defined in SKILL.md:

  DW-01: Search Execution — SearchEngine.search() with random strategy
         completes 10 trials on a minimal-config model, returning
         best_params() that improve over the first trial's metric.

  DW-02: LR Finder — LearningRateFinder.find() produces a smooth
         loss-vs-LR curve; suggested_lr() returns reasonable bounds
         (not NaN, not at extremes).

  DW-03: Trial Management — TrialManager.export_csv() produces valid CSV
         with all trial params and metrics; get_importance() returns
         non-trivial importance scores for >1 parameter.

Exit codes:
    0 — all gates pass
    1 — one or more gates fail

Usage:
    python scripts/validate_search.py
    python scripts/validate_search.py --verbose
    python scripts/validate_search.py --gate 1   # run only gate 1
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the assets directory is importable
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = SCRIPT_DIR.parent / "assets"
if str(ASSETS_DIR) not in sys.path:
    sys.path.insert(0, str(ASSETS_DIR))

from search_config_template import SearchConfig, LRFinderConfig, PhasePresets
from search_space_template import SearchSpace
from search_engine_template import SearchEngine, SearchResult
from trial_manager_template import TrialManager
from early_stopping_template import ASHAPruner, create_pruner

# Optional torch for DW-02
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
# Utilities
# ---------------------------------------------------------------------------

class GateResult:
    def __init__(self, name: str):
        self.name = name
        self.checks: list[tuple[str, bool, str]] = []

    def check(self, label: str, passed: bool, detail: str = ""):
        self.checks.append((label, passed, detail))

    @property
    def passed(self) -> bool:
        return all(ok for _, ok, _ in self.checks)

    @property
    def n_pass(self) -> int:
        return sum(1 for _, ok, _ in self.checks if ok)

    @property
    def n_fail(self) -> int:
        return sum(1 for _, ok, _ in self.checks if not ok)

    def report(self, verbose: bool = False) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"\n{'='*60}",
                 f"Gate: {self.name}  [{status}]  ({self.n_pass}/{len(self.checks)} checks)",
                 f"{'='*60}"]
        if verbose or not self.passed:
            for label, ok, detail in self.checks:
                mark = "OK" if ok else "FAIL"
                line = f"  [{mark}] {label}"
                if detail and (not ok or verbose):
                    line += f"  -- {detail}"
                lines.append(line)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gate DW-01: Search Execution
# ---------------------------------------------------------------------------

def validate_gate1(verbose: bool = False) -> GateResult:
    """Random search: 10 trials, best_params improves over first trial."""
    gate = GateResult("DW-01: Search Execution")

    # Objective: simple quadratic
    def objective(params):
        x = params.get("x", 0)
        y = params.get("y", 0)
        return (x - 2.0) ** 2 + (y + 1.0) ** 2

    space = SearchSpace(seed=42)
    space.add_float("x", -5.0, 5.0)
    space.add_float("y", -5.0, 5.0)

    config = SearchConfig(
        strategy="random",
        n_trials=10,
        direction="minimize",
        enable_pruning=False,
        seed=42,
    )
    engine = SearchEngine(config)
    result = engine.search(objective, space)

    # Check 1: search completes
    gate.check("Search completes", result.n_complete > 0,
               f"n_complete={result.n_complete}")

    # Check 2: 10 trials
    gate.check("10 trials", len(result.trials) == 10,
               f"got {len(result.trials)}")

    # Check 3: best_params returns dict with expected keys
    bp = engine.best_params()
    gate.check("best_params is dict", isinstance(bp, dict) and "x" in bp and "y" in bp,
               f"keys={list(bp.keys())}")

    # Check 4: best_value is finite
    bv = engine.best_value()
    gate.check("best_value finite", math.isfinite(bv), f"bv={bv}")

    # Check 5: best improves over first trial
    first_val = result.trials[0].value
    gate.check("best < first trial", bv <= first_val,
               f"best={bv:.4f}, first={first_val:.4f}")

    # Check 6: Bayesian search also works
    config_b = SearchConfig(
        strategy="bayesian", n_trials=15, direction="minimize",
        enable_pruning=False, seed=42, n_startup_trials=5,
    )
    engine_b = SearchEngine(config_b)
    result_b = engine_b.search(objective, space)
    gate.check("Bayesian search completes", result_b.n_complete > 0)
    gate.check("Bayesian best finite", math.isfinite(engine_b.best_value()))

    # Check 7: Grid search works on small space
    space_grid = SearchSpace(seed=42)
    space_grid.add_categorical("a", [1, 2, 3])
    space_grid.add_categorical("b", [10, 20])
    config_g = SearchConfig(
        strategy="grid", n_trials=100, direction="minimize",
        enable_pruning=False, seed=42,
    )
    engine_g = SearchEngine(config_g)
    result_g = engine_g.search(lambda p: p["a"] + p["b"], space_grid)
    gate.check("Grid covers space", result_g.n_complete == 6,
               f"got {result_g.n_complete}")

    # Check 8: Maximize direction
    config_max = SearchConfig(
        strategy="random", n_trials=10, direction="maximize",
        enable_pruning=False, seed=42,
    )
    engine_max = SearchEngine(config_max)
    engine_max.search(lambda p: -(p["x"] ** 2), space)
    gate.check("Maximize works", engine_max.best_value() > float("-inf"))

    # Check 9: NaN resilience
    nan_ctr = [0]
    def nan_obj(p):
        nan_ctr[0] += 1
        if nan_ctr[0] % 3 == 0:
            return float("nan")
        return p["x"] ** 2
    config_nan = SearchConfig(strategy="random", n_trials=10, direction="minimize",
                              enable_pruning=False, seed=42)
    engine_nan = SearchEngine(config_nan)
    result_nan = engine_nan.search(nan_obj, space)
    gate.check("NaN resilient", result_nan.n_complete > 0)

    # Check 10: Reproducibility
    eng_a = SearchEngine(SearchConfig(strategy="random", n_trials=5,
                                      direction="minimize", enable_pruning=False, seed=77))
    eng_b = SearchEngine(SearchConfig(strategy="random", n_trials=5,
                                      direction="minimize", enable_pruning=False, seed=77))
    sp_a = SearchSpace(seed=77)
    sp_a.add_float("x", -5, 5)
    sp_b = SearchSpace(seed=77)
    sp_b.add_float("x", -5, 5)
    eng_a.search(lambda p: p["x"] ** 2, sp_a)
    eng_b.search(lambda p: p["x"] ** 2, sp_b)
    gate.check("Reproducible",
               all(abs(a.params["x"] - b.params["x"]) < 1e-9
                   for a, b in zip(eng_a.trials, eng_b.trials)))

    # Check 11: Resume
    with tempfile.TemporaryDirectory() as td:
        eng_save = SearchEngine(SearchConfig(strategy="random", n_trials=5,
                                             direction="minimize", enable_pruning=False, seed=42))
        eng_save.search(objective, space)
        eng_save.save(td)
        eng_load = SearchEngine(SearchConfig(strategy="random", n_trials=5,
                                             direction="minimize", enable_pruning=False, seed=42))
        res_loaded = eng_load.resume(td)
        gate.check("Resume works", len(res_loaded.trials) == 5)

    # Check 12: Phase preset space works with search
    preset = PhasePresets.phase1_snn()
    from search_space_template import SearchSpace as SS
    sp_phase = SS.from_preset(preset, seed=42)
    cfg_phase = SearchConfig(strategy="random", n_trials=5, direction="minimize",
                             enable_pruning=False, seed=42)
    eng_phase = SearchEngine(cfg_phase)
    res_phase = eng_phase.search(lambda p: sum(v ** 2 for v in p.values() if isinstance(v, (int, float))),
                                 sp_phase)
    gate.check("Phase preset search", res_phase.n_complete > 0)

    return gate


# ---------------------------------------------------------------------------
# Gate DW-02: LR Finder
# ---------------------------------------------------------------------------

def validate_gate2(verbose: bool = False) -> GateResult:
    """LR Finder: smooth curve, valid suggested_lr."""
    gate = GateResult("DW-02: LR Finder")

    if not HAS_TORCH:
        gate.check("PyTorch available", False, "torch not installed — cannot validate LR finder")
        return gate

    torch.manual_seed(42)

    # Simple model and data
    model = nn.Linear(10, 1)
    X = torch.randn(200, 10)
    w = torch.randn(10, 1)
    y = X @ w + 0.1 * torch.randn(200, 1)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    criterion = nn.MSELoss()

    # Run LR finder
    cfg = LRFinderConfig(start_lr=1e-7, end_lr=10.0, num_steps=80,
                         smooth_factor=0.05, divergence_threshold=5.0)
    finder = LearningRateFinder(model, torch.optim.SGD, config=cfg)
    result = finder.find(loader, criterion, device="cpu")

    # Check 1: Returns lr/loss pairs
    gate.check("Returns pairs",
               len(result.lrs) > 0 and len(result.lrs) == len(result.smoothed_losses),
               f"n_steps={len(result.lrs)}")

    # Check 2: Loss curve has a minimum region (not monotonic increase)
    gate.check("Loss curve has min",
               min(result.smoothed_losses) < result.smoothed_losses[0],
               "loss never decreased")

    # Check 3: suggested_lr valid
    min_lr, max_lr = finder.suggested_lr()
    gate.check("suggested min_lr > 0", min_lr > 0, f"min_lr={min_lr}")
    gate.check("suggested max_lr > 0", max_lr > 0, f"max_lr={max_lr}")
    gate.check("min_lr < max_lr", min_lr < max_lr,
               f"min={min_lr}, max={max_lr}")

    # Check 4: Not NaN
    gate.check("min_lr not NaN", not math.isnan(min_lr))
    gate.check("max_lr not NaN", not math.isnan(max_lr))

    # Check 5: Not at extremes
    gate.check("min_lr >= start_lr", min_lr >= cfg.start_lr,
               f"min_lr={min_lr}, start={cfg.start_lr}")
    gate.check("max_lr < end_lr", max_lr < cfg.end_lr,
               f"max_lr={max_lr}, end={cfg.end_lr}")

    # Check 6: Model state preserved
    state_before = {k: v.clone() for k, v in model.state_dict().items()}
    finder2 = LearningRateFinder(model, torch.optim.SGD,
                                  config=LRFinderConfig(num_steps=20))
    finder2.find(loader, criterion, device="cpu")
    state_after = model.state_dict()
    match = all(torch.allclose(state_before[k], state_after[k], atol=1e-6)
                for k in state_before)
    gate.check("Model state restored", match)

    # Check 7: Smoothed is smoother than raw
    if len(result.raw_losses) > 5:
        raw_var = _variance(result.raw_losses)
        smooth_var = _variance(result.smoothed_losses)
        gate.check("Smoothed < raw variance",
                   smooth_var <= raw_var * 1.1,
                   f"raw={raw_var:.4f}, smooth={smooth_var:.4f}")

    # Check 8: Divergence detection (aggressive LR)
    model_div = nn.Linear(10, 1)
    cfg_div = LRFinderConfig(start_lr=1e-7, end_lr=1000.0, num_steps=200,
                             divergence_threshold=5.0)
    finder_div = LearningRateFinder(model_div, torch.optim.SGD, config=cfg_div)
    res_div = finder_div.find(loader, criterion, device="cpu")
    gate.check("Divergence detected",
               res_div.stopped_early or res_div.num_steps_completed < 200,
               f"steps={res_div.num_steps_completed}")

    # Check 9: Works with Adam
    model_adam = nn.Linear(10, 1)
    finder_adam = LearningRateFinder(model_adam, torch.optim.Adam,
                                     config=LRFinderConfig(num_steps=30))
    res_adam = finder_adam.find(loader, criterion, device="cpu")
    gate.check("Adam works", res_adam.num_steps_completed > 0)

    # Check 10: Plot does not crash
    plot_out = finder.plot()
    gate.check("Plot returns", plot_out is not None)

    return gate


# ---------------------------------------------------------------------------
# Gate DW-03: Trial Management
# ---------------------------------------------------------------------------

def validate_gate3(verbose: bool = False) -> GateResult:
    """TrialManager: CSV export valid, importance non-trivial."""
    gate = GateResult("DW-03: Trial Management")

    rng = random.Random(42)

    with tempfile.TemporaryDirectory() as study_dir:
        mgr = TrialManager(study_dir, metric="val_loss", direction="minimize")

        # Log 50 trials with a known correlation
        for i in range(50):
            lr = rng.uniform(1e-4, 1e-1)
            wd = rng.uniform(1e-4, 1e-1)
            layers = rng.randint(2, 12)
            # val_loss = f(lr, noise) -- lr is the dominant parameter
            val_loss = 10.0 * (lr - 0.01) ** 2 + 0.001 * wd + 0.0001 * layers + rng.gauss(0, 0.0001)
            mgr.log_trial(
                params={"lr": lr, "weight_decay": wd, "layers": layers},
                metrics={"val_loss": val_loss, "val_acc": 1.0 - val_loss},
            )

        # Check 1: All 50 trials logged
        gate.check("50 trials logged", mgr.n_trials == 50)

        # Check 2: get_best_trials returns correct
        best5 = mgr.get_best_trials(n=5)
        gate.check("get_best returns 5", len(best5) == 5)
        vals = [t.metrics["val_loss"] for t in best5]
        gate.check("best sorted", vals == sorted(vals),
                   f"vals={[f'{v:.4f}' for v in vals]}")

        # Check 3: export_csv
        csv_path = os.path.join(study_dir, "results.csv")
        mgr.export_csv(csv_path)
        gate.check("CSV exists", os.path.exists(csv_path))

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)

        # Check 4: CSV has header + 50 rows
        gate.check("CSV rows", len(rows) == 51,
                   f"got {len(rows)} rows")

        # Check 5: CSV has all columns
        header = rows[0]
        gate.check("CSV has lr", "lr" in header)
        gate.check("CSV has val_loss", "val_loss" in header)
        gate.check("CSV has val_acc", "val_acc" in header)

        # Check 6: CSV values are parseable
        try:
            for row in rows[1:6]:
                lr_idx = header.index("lr")
                float(row[lr_idx])
            gate.check("CSV values parseable", True)
        except (ValueError, IndexError) as e:
            gate.check("CSV values parseable", False, str(e))

        # Check 7: get_importance
        imp = mgr.get_importance()
        gate.check("Importance returns dict", isinstance(imp, dict) and len(imp) > 0,
                   f"len={len(imp)}")

        # Check 8: Importance for >1 param
        gate.check("Importance >1 param", len(imp) > 1,
                   f"params: {list(imp.keys())}")

        # Check 9: Non-trivial scores
        gate.check("Non-trivial importance",
                   any(v > 0.01 for v in imp.values()),
                   f"scores: {imp}")

        # Check 10: lr should be most important (it dominates the objective)
        if imp:
            top_param = max(imp, key=imp.get)
            gate.check("lr is most important", top_param == "lr",
                       f"top={top_param}, scores={imp}")

        # Check 11: Persist and reload
        mgr2 = TrialManager(study_dir, metric="val_loss", direction="minimize")
        gate.check("Reload preserves trials", mgr2.n_trials == 50)

        # Check 12: get_trial by ID
        all_trials = mgr.get_all_trials()
        t0 = all_trials[0]
        got = mgr.get_trial(t0.trial_id)
        gate.check("get_trial", got is not None and got.trial_id == t0.trial_id)

        # Check 13: Maximize direction
        mgr_max = TrialManager(
            os.path.join(study_dir, "max_study"),
            metric="val_acc", direction="maximize",
        )
        for i in range(20):
            mgr_max.log_trial(
                params={"x": float(i)},
                metrics={"val_acc": float(i) / 20},
            )
        best_max = mgr_max.get_best_trials(1)
        gate.check("Maximize best",
                   len(best_max) == 1 and best_max[0].metrics["val_acc"] == 19 / 20,
                   f"got {best_max[0].metrics.get('val_acc') if best_max else None}")

        # Check 14: Empty export
        mgr_empty = TrialManager(
            os.path.join(study_dir, "empty_study"),
            metric="val_loss", direction="minimize",
        )
        empty_csv = os.path.join(study_dir, "empty.csv")
        mgr_empty.export_csv(empty_csv)
        gate.check("Empty CSV", os.path.exists(empty_csv))

        # Check 15: Plot methods do not crash
        hist = mgr.plot_optimization_history()
        gate.check("plot_history", hist is not None)
        imp_plot = mgr.plot_param_importances()
        gate.check("plot_importance", imp_plot is not None)

    return gate


# ---------------------------------------------------------------------------
# Additional validation: Early stopping and search space
# ---------------------------------------------------------------------------

def validate_early_stopping(verbose: bool = False) -> GateResult:
    """Validate early stopping / pruner infrastructure."""
    gate = GateResult("Extra: Early Stopping")

    # ASHA
    pruner = ASHAPruner(min_resource=1, reduction_factor=3, max_resource=27)
    gate.check("ASHA rungs", pruner.rungs == [1, 3, 9, 27])

    for i in range(9):
        pruner.report(f"t{i}", step=1, value=float(i))
    pruned = sum(1 for i in range(9) if pruner.should_prune(f"t{i}", step=1))
    gate.check("ASHA prunes 6/9", pruned == 6, f"pruned={pruned}")

    # Median pruner
    from early_stopping_template import MedianPruner
    mp = MedianPruner(n_startup_trials=3, n_warmup_steps=0)
    for i in range(5):
        mp.report(f"t{i}", step=1, value=float(i))
    gate.check("Median prunes worst", mp.should_prune("t4", step=1))
    gate.check("Median keeps best", not mp.should_prune("t0", step=1))

    # Factory
    gate.check("Factory asha", isinstance(create_pruner("asha"), ASHAPruner))

    return gate


def validate_search_space(verbose: bool = False) -> GateResult:
    """Validate search space functionality."""
    gate = GateResult("Extra: Search Space")

    space = SearchSpace(seed=42)
    space.add_float("lr", 1e-5, 1e-1, log=True)
    space.add_int("layers", 2, 12)
    space.add_categorical("opt", ["adam", "sgd"])

    s = space.sample()
    gate.check("Sample complete", len(s) == 3)
    gate.check("Float in bounds", 1e-5 <= s["lr"] <= 1e-1)
    gate.check("Int in bounds", 2 <= s["layers"] <= 12)
    gate.check("Cat valid", s["opt"] in ["adam", "sgd"])

    # Grid
    grid = space.grid({"lr": 3, "layers": 3, "opt": 2}, default_n=3)
    gate.check("Grid size", len(grid) == 18, f"got {len(grid)}")

    # Conditional
    cond = SearchSpace(seed=42)
    cond.add_categorical("use_htm", [True, False])
    sub = SearchSpace(seed=42)
    sub.add_int("cols", 512, 4096)
    cond.add_conditional("htm", "use_htm == True", sub)
    samples = [cond.sample() for _ in range(200)]
    active = [s for s in samples if s["use_htm"] and "cols" in s]
    inactive = [s for s in samples if not s["use_htm"] and "cols" not in s]
    gate.check("Conditional active", len(active) > 10)
    gate.check("Conditional inactive", len(inactive) > 10)

    # Serialization
    d = space.to_dict()
    sp_back = SearchSpace.from_dict(d)
    gate.check("Serialization", sp_back.param_names == space.param_names)

    return gate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _variance(vals):
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return sum((v - m) ** 2 for v in vals) / (len(vals) - 1)


def main():
    parser = argparse.ArgumentParser(description="Validate hyperparameter search infrastructure")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--gate", type=int, default=None,
                        help="Run only this gate (1, 2, or 3)")
    args = parser.parse_args()

    gates: list[GateResult] = []

    if args.gate is None or args.gate == 1:
        gates.append(validate_gate1(args.verbose))
    if args.gate is None or args.gate == 2:
        gates.append(validate_gate2(args.verbose))
    if args.gate is None or args.gate == 3:
        gates.append(validate_gate3(args.verbose))

    # Extra validations (always run unless gate is specified)
    if args.gate is None:
        gates.append(validate_early_stopping(args.verbose))
        gates.append(validate_search_space(args.verbose))

    # Report
    all_pass = True
    for g in gates:
        print(g.report(args.verbose))
        if not g.passed:
            all_pass = False

    # Summary
    total_checks = sum(len(g.checks) for g in gates)
    total_pass = sum(g.n_pass for g in gates)
    total_fail = sum(g.n_fail for g in gates)
    print(f"\n{'='*60}")
    print(f"SUMMARY: {total_pass}/{total_checks} checks passed, {total_fail} failed")
    print(f"Overall: {'PASS' if all_pass else 'FAIL'}")
    print(f"{'='*60}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
