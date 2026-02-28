"""
Early stopping / pruning strategies for hyperparameter search.

Implements ASHA (Asynchronous Successive Halving), Hyperband, and
MedianPruner from scratch.  No Optuna dependency.

Each pruner inherits from ``Pruner`` (ABC) and exposes:
    pruner.report(trial_id, step, value)
    pruner.should_prune(trial_id, step) -> bool

Usage:
    from early_stopping_template import ASHAPruner, HyperbandPruner, MedianPruner

    pruner = ASHAPruner(min_resource=1, reduction_factor=3, max_resource=81)
    pruner.report("trial_0", step=1, value=0.9)
    if pruner.should_prune("trial_0", step=1):
        # stop this trial early
        pass
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Pruner abstract base class
# ---------------------------------------------------------------------------

class Pruner(ABC):
    """Abstract base for trial pruning strategies."""

    @abstractmethod
    def report(self, trial_id: str, step: int, value: float) -> None:
        """Report an intermediate metric value at a given step."""
        ...

    @abstractmethod
    def should_prune(self, trial_id: str, step: int) -> bool:
        """Return True if the trial should be pruned at the given step."""
        ...

    def reset(self) -> None:
        """Clear all stored state."""
        ...


# ---------------------------------------------------------------------------
# ASHA — Asynchronous Successive Halving Algorithm
# ---------------------------------------------------------------------------

class ASHAPruner(Pruner):
    """ASHA pruner: prune the bottom (1 - 1/eta) fraction at each rung.

    Rungs are at steps: min_resource, min_resource * eta, min_resource * eta^2, ...
    up to max_resource.  At each rung, the bottom (1 - 1/eta) trials are pruned.

    Parameters
    ----------
    min_resource : Minimum resource level (e.g. 1 epoch) before any pruning.
    reduction_factor : eta -- fraction of trials promoted is 1/eta.
    max_resource : Maximum resource level.
    """

    def __init__(self, min_resource: int = 1, reduction_factor: int = 3,
                 max_resource: int = 81):
        if min_resource < 1:
            raise ValueError(f"min_resource must be >= 1, got {min_resource}")
        if reduction_factor < 2:
            raise ValueError(f"reduction_factor must be >= 2, got {reduction_factor}")
        if max_resource < min_resource:
            raise ValueError(f"max_resource must be >= min_resource")
        self.min_resource = min_resource
        self.eta = reduction_factor
        self.max_resource = max_resource
        # trial_id -> {step: value}
        self._history: Dict[str, Dict[int, float]] = defaultdict(dict)
        self._rungs = self._compute_rungs()

    def _compute_rungs(self) -> List[int]:
        """Compute the rung steps: min, min*eta, min*eta^2, ..., <= max."""
        rungs = []
        r = self.min_resource
        while r <= self.max_resource:
            rungs.append(r)
            r *= self.eta
        return rungs

    @property
    def rungs(self) -> List[int]:
        return list(self._rungs)

    def report(self, trial_id: str, step: int, value: float) -> None:
        self._history[trial_id][step] = value

    def should_prune(self, trial_id: str, step: int) -> bool:
        """Prune if at a rung and trial is in the bottom (1 - 1/eta) fraction."""
        if step < self.min_resource:
            return False
        if step not in self._rungs:
            return False
        # Gather all trials that have reported at this rung
        rung_values: List[Tuple[str, float]] = []
        for tid, hist in self._history.items():
            if step in hist:
                rung_values.append((tid, hist[step]))
        if len(rung_values) <= 1:
            return False  # cannot prune with a single trial
        # Sort by value (lower is better = minimize)
        rung_values.sort(key=lambda x: x[1])
        n_promote = max(1, len(rung_values) // self.eta)
        promoted_ids = {tid for tid, _ in rung_values[:n_promote]}
        return trial_id not in promoted_ids

    def should_prune_maximize(self, trial_id: str, step: int) -> bool:
        """Like should_prune but for maximization (higher is better)."""
        if step < self.min_resource:
            return False
        if step not in self._rungs:
            return False
        rung_values: List[Tuple[str, float]] = []
        for tid, hist in self._history.items():
            if step in hist:
                rung_values.append((tid, hist[step]))
        if len(rung_values) <= 1:
            return False
        rung_values.sort(key=lambda x: x[1], reverse=True)
        n_promote = max(1, len(rung_values) // self.eta)
        promoted_ids = {tid for tid, _ in rung_values[:n_promote]}
        return trial_id not in promoted_ids

    def reset(self) -> None:
        self._history.clear()

    def get_trial_history(self, trial_id: str) -> Dict[int, float]:
        return dict(self._history.get(trial_id, {}))

    @property
    def n_trials(self) -> int:
        return len(self._history)


# ---------------------------------------------------------------------------
# Hyperband
# ---------------------------------------------------------------------------

@dataclass
class HyperbandBracket:
    """A single bracket in the Hyperband schedule."""
    bracket_id: int
    n_trials: int
    min_resource: int
    max_resource: int
    reduction_factor: int
    rungs: List[Tuple[int, int]]  # list of (n_trials_at_rung, resource)


class HyperbandPruner(Pruner):
    """Hyperband: runs multiple brackets of successive halving.

    Each bracket trades off between number of initial trials and minimum
    resource per trial.

    Parameters
    ----------
    min_resource : Minimum resource (e.g. 1 epoch).
    reduction_factor : eta (typically 3).
    max_resource : Maximum resource (e.g. 81 epochs).
    """

    def __init__(self, min_resource: int = 1, reduction_factor: int = 3,
                 max_resource: int = 81):
        if min_resource < 1:
            raise ValueError(f"min_resource must be >= 1")
        if reduction_factor < 2:
            raise ValueError(f"reduction_factor must be >= 2")
        if max_resource < min_resource:
            raise ValueError(f"max_resource must be >= min_resource")

        self.min_resource = min_resource
        self.eta = reduction_factor
        self.max_resource = max_resource
        self._history: Dict[str, Dict[int, float]] = defaultdict(dict)

        # Compute brackets
        self.s_max = int(math.floor(math.log(max_resource / min_resource) / math.log(self.eta)))
        self.brackets: List[HyperbandBracket] = self._compute_brackets()

        # Internal ASHA pruners per bracket
        self._bracket_pruners: Dict[int, ASHAPruner] = {}
        self._trial_bracket: Dict[str, int] = {}  # trial -> bracket_id
        self._bracket_counter: int = 0

    def _compute_brackets(self) -> List[HyperbandBracket]:
        brackets = []
        for s in range(self.s_max + 1):
            n = int(math.ceil(
                (self.s_max + 1) * (self.eta ** s) / (s + 1)
            ))
            r = self.max_resource // (self.eta ** s)
            r = max(r, self.min_resource)
            # Compute rungs for this bracket
            rungs = []
            curr_n = n
            curr_r = r
            while curr_n >= 1 and curr_r <= self.max_resource:
                rungs.append((curr_n, curr_r))
                curr_n = max(1, curr_n // self.eta)
                curr_r = min(curr_r * self.eta, self.max_resource + 1)
                if curr_n < 1:
                    break
            brackets.append(HyperbandBracket(
                bracket_id=s,
                n_trials=n,
                min_resource=r,
                max_resource=self.max_resource,
                reduction_factor=self.eta,
                rungs=rungs,
            ))
        return brackets

    def assign_bracket(self, trial_id: str, bracket_id: Optional[int] = None) -> int:
        """Assign a trial to a bracket (round-robin if not specified)."""
        if bracket_id is None:
            bracket_id = self._bracket_counter % len(self.brackets)
            self._bracket_counter += 1
        self._trial_bracket[trial_id] = bracket_id
        return bracket_id

    def get_bracket(self, trial_id: str) -> Optional[HyperbandBracket]:
        bid = self._trial_bracket.get(trial_id)
        if bid is not None:
            return self.brackets[bid]
        return None

    def report(self, trial_id: str, step: int, value: float) -> None:
        self._history[trial_id][step] = value

    def should_prune(self, trial_id: str, step: int) -> bool:
        """Prune using bracket-specific rungs via ASHA logic."""
        bracket = self.get_bracket(trial_id)
        if bracket is None:
            return False  # unassigned -> do not prune
        # Only prune at rung resource levels for this bracket
        rung_resources = [r for _, r in bracket.rungs]
        if step not in rung_resources:
            return False
        if step < bracket.min_resource:
            return False
        # Gather all trials in the same bracket that reported at this rung
        same_bracket_trials = [
            tid for tid, bid in self._trial_bracket.items() if bid == bracket.bracket_id
        ]
        rung_values: List[Tuple[str, float]] = []
        for tid in same_bracket_trials:
            if tid in self._history and step in self._history[tid]:
                rung_values.append((tid, self._history[tid][step]))
        if len(rung_values) <= 1:
            return False
        rung_values.sort(key=lambda x: x[1])  # minimize
        n_promote = max(1, len(rung_values) // self.eta)
        promoted_ids = {tid for tid, _ in rung_values[:n_promote]}
        return trial_id not in promoted_ids

    def reset(self) -> None:
        self._history.clear()
        self._trial_bracket.clear()
        self._bracket_counter = 0

    @property
    def num_brackets(self) -> int:
        return len(self.brackets)

    def total_budget(self) -> int:
        """Approximate total resource units across all brackets."""
        total = 0
        for bracket in self.brackets:
            for n_trials, resource in bracket.rungs:
                total += n_trials * resource
        return total


# ---------------------------------------------------------------------------
# MedianPruner
# ---------------------------------------------------------------------------

class MedianPruner(Pruner):
    """Prune trials whose intermediate value is below the median of all trials
    at the same step.

    Parameters
    ----------
    n_startup_trials : Do not prune until this many trials have reported.
    n_warmup_steps : Do not prune before this step.
    percentile : Prune if below this percentile (default 50 = median).
    """

    def __init__(self, n_startup_trials: int = 5, n_warmup_steps: int = 1,
                 percentile: float = 50.0):
        if percentile < 0 or percentile > 100:
            raise ValueError(f"percentile must be in [0, 100], got {percentile}")
        self.n_startup_trials = n_startup_trials
        self.n_warmup_steps = n_warmup_steps
        self.percentile = percentile
        self._history: Dict[str, Dict[int, float]] = defaultdict(dict)

    def report(self, trial_id: str, step: int, value: float) -> None:
        self._history[trial_id][step] = value

    def should_prune(self, trial_id: str, step: int) -> bool:
        if step < self.n_warmup_steps:
            return False
        # Collect all values at this step
        values_at_step = []
        for tid, hist in self._history.items():
            if step in hist:
                values_at_step.append(hist[step])
        if len(values_at_step) < self.n_startup_trials:
            return False
        if trial_id not in self._history or step not in self._history[trial_id]:
            return False
        trial_value = self._history[trial_id][step]
        # Compute the percentile threshold (lower is better = minimize)
        sorted_vals = sorted(values_at_step)
        idx = int(math.ceil(self.percentile / 100.0 * len(sorted_vals))) - 1
        idx = max(0, min(idx, len(sorted_vals) - 1))
        threshold = sorted_vals[idx]
        return trial_value > threshold

    def should_prune_maximize(self, trial_id: str, step: int) -> bool:
        """Like should_prune but for maximization."""
        if step < self.n_warmup_steps:
            return False
        values_at_step = []
        for tid, hist in self._history.items():
            if step in hist:
                values_at_step.append(hist[step])
        if len(values_at_step) < self.n_startup_trials:
            return False
        if trial_id not in self._history or step not in self._history[trial_id]:
            return False
        trial_value = self._history[trial_id][step]
        sorted_vals = sorted(values_at_step, reverse=True)
        idx = int(math.ceil(self.percentile / 100.0 * len(sorted_vals))) - 1
        idx = max(0, min(idx, len(sorted_vals) - 1))
        threshold = sorted_vals[idx]
        return trial_value < threshold

    def reset(self) -> None:
        self._history.clear()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_pruner(name: str, **kwargs) -> Pruner:
    """Create a pruner by name.

    Parameters
    ----------
    name : "asha", "hyperband", "median", or "none"
    """
    name = name.lower().strip()
    if name == "asha":
        return ASHAPruner(**kwargs)
    elif name == "hyperband":
        return HyperbandPruner(**kwargs)
    elif name == "median":
        return MedianPruner(**kwargs)
    elif name == "none":
        return _NoPruner()
    else:
        raise ValueError(f"Unknown pruner '{name}'. Choose from: asha, hyperband, median, none")


class _NoPruner(Pruner):
    """A no-op pruner that never prunes."""
    def report(self, trial_id: str, step: int, value: float) -> None:
        pass

    def should_prune(self, trial_id: str, step: int) -> bool:
        return False


# ---------------------------------------------------------------------------
# Self-tests  (25+ tests)
# ---------------------------------------------------------------------------

def _run_tests():
    passed = 0
    failed = 0

    def check(name: str, condition: bool, msg: str = ""):
        nonlocal passed, failed
        if condition:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL: {name} — {msg}")

    print("=== ASHA Pruner Tests ===")

    # ES-01: Bottom trials pruned (9 trials at rung 1, eta=3 -> 6 pruned, 3 promoted)
    asha = ASHAPruner(min_resource=1, reduction_factor=3, max_resource=81)
    for i in range(9):
        asha.report(f"t{i}", step=1, value=float(i))  # t0=best, t8=worst
    pruned = sum(1 for i in range(9) if asha.should_prune(f"t{i}", step=1))
    promoted = 9 - pruned
    check("ES-01 bottom pruned", pruned == 6 and promoted == 3,
          f"pruned={pruned}, promoted={promoted}")

    # ES-02: Best trial never pruned
    check("ES-02 best never pruned",
          not asha.should_prune("t0", step=1))

    # ES-03: Min resource respected
    asha2 = ASHAPruner(min_resource=3, reduction_factor=3, max_resource=81)
    for i in range(9):
        asha2.report(f"t{i}", step=1, value=float(i))
    check("ES-03 min resource",
          not any(asha2.should_prune(f"t{i}", step=1) for i in range(9)))

    # ES-04: Reduction factor effect (eta=2 keeps more than eta=3)
    asha_eta2 = ASHAPruner(min_resource=1, reduction_factor=2, max_resource=64)
    asha_eta3 = ASHAPruner(min_resource=1, reduction_factor=3, max_resource=81)
    for i in range(12):
        asha_eta2.report(f"t{i}", step=1, value=float(i))
        asha_eta3.report(f"t{i}", step=1, value=float(i))
    pruned_2 = sum(1 for i in range(12) if asha_eta2.should_prune(f"t{i}", step=1))
    pruned_3 = sum(1 for i in range(12) if asha_eta3.should_prune(f"t{i}", step=1))
    check("ES-04 eta=2 keeps more", pruned_2 < pruned_3,
          f"eta2 pruned {pruned_2}, eta3 pruned {pruned_3}")

    # ES-05: Multiple rungs
    asha3 = ASHAPruner(min_resource=1, reduction_factor=3, max_resource=27)
    check("ES-05 rung list", asha3.rungs == [1, 3, 9, 27])
    # 27 trials report at rung 1
    for i in range(27):
        asha3.report(f"t{i}", step=1, value=float(i))
    pruned_r1 = sum(1 for i in range(27) if asha3.should_prune(f"t{i}", step=1))
    survived_r1 = 27 - pruned_r1
    check("ES-05 rung 1 survivors", survived_r1 == 9,
          f"survived={survived_r1}")
    # 9 survivors report at rung 3
    for i in range(9):
        asha3.report(f"t{i}", step=3, value=float(i))
    pruned_r2 = sum(1 for i in range(9) if asha3.should_prune(f"t{i}", step=3))
    survived_r2 = 9 - pruned_r2
    check("ES-05 rung 2 survivors", survived_r2 == 3,
          f"survived={survived_r2}")
    # 3 survivors report at rung 9
    for i in range(3):
        asha3.report(f"t{i}", step=9, value=float(i))
    pruned_r3 = sum(1 for i in range(3) if asha3.should_prune(f"t{i}", step=9))
    survived_r3 = 3 - pruned_r3
    check("ES-05 rung 3 survivors", survived_r3 == 1,
          f"survived={survived_r3}")

    # ASHA reset
    asha.reset()
    check("ASHA reset", asha.n_trials == 0)

    print("\n=== Hyperband Tests ===")

    # ES-06: Multiple brackets
    hb = HyperbandPruner(min_resource=1, reduction_factor=3, max_resource=81)
    check("ES-06 brackets", hb.num_brackets == hb.s_max + 1)
    check("ES-06 s_max", hb.s_max == 4,
          f"s_max={hb.s_max}")

    # ES-07: Bracket 0 has most resources (highest min_resource)
    b0 = hb.brackets[0]
    check("ES-07 bracket 0 most resources",
          b0.min_resource >= hb.brackets[-1].min_resource,
          f"b0.min_resource={b0.min_resource}")

    # ES-08: Last bracket has most trials
    b_last = hb.brackets[-1]
    check("ES-08 last bracket most trials",
          b_last.n_trials >= hb.brackets[0].n_trials,
          f"last.n={b_last.n_trials}, first.n={hb.brackets[0].n_trials}")

    # ES-09: Total budget is bounded
    budget = hb.total_budget()
    check("ES-09 budget bounded", budget > 0,
          f"budget={budget}")

    # Hyperband assign_bracket and prune
    hb2 = HyperbandPruner(min_resource=1, reduction_factor=3, max_resource=9)
    for i in range(9):
        hb2.assign_bracket(f"t{i}", bracket_id=hb2.s_max)
    for i in range(9):
        hb2.report(f"t{i}", step=1, value=float(i))
    pruned_hb = sum(1 for i in range(9)
                    if hb2.should_prune(f"t{i}", step=1))
    check("Hyperband pruning works", pruned_hb > 0,
          f"pruned={pruned_hb}")

    # Hyperband reset
    hb2.reset()
    check("Hyperband reset", len(hb2._history) == 0)

    print("\n=== MedianPruner Tests ===")

    # ES-10: Below-median pruned
    mp = MedianPruner(n_startup_trials=3, n_warmup_steps=0)
    for i in range(5):
        mp.report(f"t{i}", step=1, value=float(i))
    # t0=0, t1=1, t2=2 (median), t3=3, t4=4
    check("ES-10 below-median",
          mp.should_prune("t4", step=1),
          "worst trial should be pruned")

    # ES-11: Above-median kept
    check("ES-11 above-median kept",
          not mp.should_prune("t0", step=1))

    # ES-12: Not enough data
    mp_few = MedianPruner(n_startup_trials=10, n_warmup_steps=0)
    for i in range(5):
        mp_few.report(f"t{i}", step=1, value=float(i))
    check("ES-12 not enough data",
          not mp_few.should_prune("t4", step=1))

    # ES-13: No data at step
    check("ES-13 no data at step",
          not mp.should_prune("t0", step=99))

    print("\n=== General Pruner Tests ===")

    # ES-14: should_prune interface returns bool
    for pruner in [ASHAPruner(), MedianPruner(), _NoPruner()]:
        pruner.report("x", 1, 0.5)
        result = pruner.should_prune("x", 1)
        check(f"ES-14 {type(pruner).__name__} returns bool",
              isinstance(result, bool))

    # ES-15: Pruner with single trial never prunes
    sp = ASHAPruner(min_resource=1, reduction_factor=3, max_resource=9)
    sp.report("lonely", step=1, value=99.0)
    check("ES-15 single trial",
          not sp.should_prune("lonely", step=1))

    # ES-16: Report intermediate value
    rp = ASHAPruner()
    rp.report("trial_x", step=5, value=0.42)
    check("ES-16 report stored",
          rp.get_trial_history("trial_x") == {5: 0.42})

    # ES-17: Monotonic improvement not pruned
    mono = ASHAPruner(min_resource=1, reduction_factor=3, max_resource=27)
    # 9 trials, t0 is best at every rung
    for i in range(9):
        mono.report(f"t{i}", step=1, value=10.0 - i * 0.1)
    check("ES-17 monotonic best",
          not mono.should_prune("t8", step=1))  # t8 has lowest value

    # ES-18: Non-pruner never prunes
    np_ = _NoPruner()
    np_.report("t0", 1, 99.0)
    check("ES-18 NoPruner", not np_.should_prune("t0", 1))

    # Maximize direction
    asha_max = ASHAPruner(min_resource=1, reduction_factor=3, max_resource=9)
    for i in range(9):
        asha_max.report(f"t{i}", step=1, value=float(i))  # t8 is best (highest)
    check("Maximize best kept",
          not asha_max.should_prune_maximize("t8", step=1))
    check("Maximize worst pruned",
          asha_max.should_prune_maximize("t0", step=1))

    print("\n=== Factory Tests ===")

    check("Factory asha", isinstance(create_pruner("asha"), ASHAPruner))
    check("Factory hyperband", isinstance(create_pruner("hyperband"), HyperbandPruner))
    check("Factory median", isinstance(create_pruner("median"), MedianPruner))
    check("Factory none", isinstance(create_pruner("none"), _NoPruner))
    try:
        create_pruner("invalid")
        check("Factory invalid", False, "should raise")
    except ValueError:
        check("Factory invalid", True)

    print("\n=== Validation Tests ===")

    try:
        ASHAPruner(min_resource=-1)
        check("ASHA min_resource < 1", False, "should raise")
    except ValueError:
        check("ASHA min_resource < 1", True)

    try:
        ASHAPruner(reduction_factor=1)
        check("ASHA eta < 2", False, "should raise")
    except ValueError:
        check("ASHA eta < 2", True)

    try:
        MedianPruner(percentile=150)
        check("Median bad percentile", False, "should raise")
    except ValueError:
        check("Median bad percentile", True)

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    if failed == 0:
        print("ALL TESTS PASSED")
    return failed == 0


if __name__ == "__main__":
    import sys
    success = _run_tests()
    sys.exit(0 if success else 1)
