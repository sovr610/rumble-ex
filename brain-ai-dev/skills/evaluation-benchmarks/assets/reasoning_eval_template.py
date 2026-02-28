"""
ReasoningEvaluator: Logical reasoning task assessment for brain_ai.

Supports exact-match scoring (bAbI-style), logical consistency checking,
and proof-accuracy measurement.  All assessment is deterministic and uses
only torch + numpy + standard library.

Dependencies: torch, numpy (standard for brain_ai)
"""

from __future__ import annotations

import math
import re
import string
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


# ======================================================================
# Data classes
# ======================================================================

@dataclass
class ReasoningResult:
    """Result of a reasoning assessment run."""

    exact_match: float
    logical_consistency: float
    proof_accuracy: float
    num_samples: int
    task_type: str = "babi"
    per_task_accuracy: Dict[str, float] = field(default_factory=dict)
    duration_seconds: float = 0.0
    details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "exact_match": self.exact_match,
            "logical_consistency": self.logical_consistency,
            "proof_accuracy": self.proof_accuracy,
            "num_samples": self.num_samples,
            "task_type": self.task_type,
            "per_task_accuracy": self.per_task_accuracy,
            "duration_seconds": self.duration_seconds,
        }
        for k, v in list(out.items()):
            if isinstance(v, float) and math.isnan(v):
                out[k] = None
        return out


# ======================================================================
# Text normalisation
# ======================================================================

_ARTICLES = {"a", "an", "the"}


def normalize_answer(text: str) -> str:
    """
    Normalise an answer string: lowercase, strip whitespace, remove
    articles and punctuation.
    """
    text = text.lower().strip()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove articles
    tokens = text.split()
    tokens = [t for t in tokens if t not in _ARTICLES]
    return " ".join(tokens).strip()


def exact_match_score(prediction: str, target: str) -> float:
    """Return 1.0 if normalised prediction matches normalised target, else 0.0."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(target) else 0.0


def exact_match_multi(prediction: str, targets: List[str]) -> float:
    """Return 1.0 if normalised prediction matches any of the targets."""
    pred_norm = normalize_answer(prediction)
    for t in targets:
        if pred_norm == normalize_answer(t):
            return 1.0
    return 0.0


def numeric_match(prediction: str, target: str, tol: float = 1e-6) -> bool:
    """Compare two strings as numbers with tolerance, if possible."""
    try:
        pv = float(prediction.strip())
        tv = float(target.strip())
        return abs(pv - tv) / max(abs(tv), 1e-12) < tol
    except ValueError:
        return False


# ======================================================================
# Logical consistency checker
# ======================================================================

class LogicalConsistencyChecker:
    """
    Check whether a set of conclusions is logically consistent with
    given premises and rules.

    The checker operates on simple propositional statements of the form:
        "X is Y"       (positive)
        "X is not Y"   (negative)

    A derivation is *consistent* if it does not contradict itself or
    the known facts.
    """

    def __init__(self) -> None:
        self.facts: Dict[str, str] = {}   # entity -> property
        self.rules: List[Tuple[str, str, str, str]] = []  # (if_ent, if_prop, then_ent, then_prop)

    def add_fact(self, entity: str, prop: str) -> None:
        self.facts[entity.lower()] = prop.lower()

    def add_rule(self, if_entity: str, if_prop: str, then_entity: str, then_prop: str) -> None:
        self.rules.append((if_entity.lower(), if_prop.lower(),
                           then_entity.lower(), then_prop.lower()))

    def derive(self) -> Dict[str, str]:
        """Apply rules until convergence, return all derived facts."""
        derived = dict(self.facts)
        changed = True
        max_iter = 100
        it = 0
        while changed and it < max_iter:
            changed = False
            it += 1
            for if_ent, if_prop, then_ent, then_prop in self.rules:
                if derived.get(if_ent) == if_prop:
                    if then_ent not in derived:
                        derived[then_ent] = then_prop
                        changed = True
        return derived

    def check_conclusion(self, entity: str, prop: str) -> Optional[bool]:
        """
        Check if (entity, prop) is consistent with known facts.
        Returns True if consistent, False if contradicts, None if unknown.
        """
        derived = self.derive()
        ent = entity.lower()
        p = prop.lower()
        if ent in derived:
            return derived[ent] == p
        return None

    def check_conclusions(
        self, conclusions: List[Tuple[str, str]]
    ) -> Tuple[int, int]:
        """
        Check a batch of conclusions.
        Returns (n_consistent, n_total).
        """
        n_consistent = 0
        n_total = len(conclusions)
        derived = self.derive()
        for entity, prop in conclusions:
            ent = entity.lower()
            p = prop.lower()
            if ent in derived:
                if derived[ent] == p:
                    n_consistent += 1
            # If unknown, do not count as consistent
        return n_consistent, n_total


# ======================================================================
# Proof accuracy scorer
# ======================================================================

def compute_proof_accuracy(
    predicted_steps: List[str],
    gold_steps: List[str],
) -> Tuple[float, int, int]:
    """
    Score a multi-step proof by checking each predicted step against
    the gold steps (order matters).

    Returns:
        (accuracy, correct_count, total_count)
    """
    correct = 0
    total = len(gold_steps)
    if total == 0:
        return (1.0 if len(predicted_steps) == 0 else 0.0), 0, 0

    for i in range(min(len(predicted_steps), total)):
        if normalize_answer(predicted_steps[i]) == normalize_answer(gold_steps[i]):
            correct += 1

    return correct / total, correct, total


# ======================================================================
# Synthetic task generators
# ======================================================================

def generate_babi_tasks(n_tasks: int = 100, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Generate simple bAbI-style location-tracking tasks.

    Each task has premises, a question, and the correct answer.
    """
    rng = np.random.RandomState(seed)
    names = ["Mary", "John", "Sandra", "Daniel", "Alice", "Bob"]
    locations = ["bathroom", "kitchen", "garden", "hallway", "office", "bedroom"]

    tasks = []
    for _ in range(n_tasks):
        person = rng.choice(names)
        n_moves = rng.randint(1, 4)
        premises = []
        final_loc = None
        for _ in range(n_moves):
            loc = rng.choice(locations)
            action = rng.choice(["went to", "moved to", "travelled to"])
            premises.append(f"{person} {action} the {loc}.")
            final_loc = loc

        # Add distractors
        other = rng.choice([n for n in names if n != person])
        dist_loc = rng.choice(locations)
        premises.insert(rng.randint(0, len(premises)), f"{other} went to the {dist_loc}.")

        question = f"Where is {person}?"
        tasks.append({
            "premises": " ".join(premises),
            "question": question,
            "answer": final_loc,
            "task_type": "location_tracking",
        })
    return tasks


def generate_logic_tasks(n_tasks: int = 50, seed: int = 42) -> List[Dict[str, Any]]:
    """Generate simple propositional logic tasks with known answers."""
    rng = np.random.RandomState(seed)
    entities = ["cat", "dog", "bird", "fish", "rabbit"]
    properties = ["fluffy", "fast", "small", "loud", "quiet"]

    tasks = []
    for _ in range(n_tasks):
        e1 = rng.choice(entities)
        p1 = rng.choice(properties)
        e2 = rng.choice([e for e in entities if e != e1])
        p2 = rng.choice([p for p in properties if p != p1])

        fact = f"{e1} is {p1}"
        rule = f"If {e1} is {p1} then {e2} is {p2}"
        question = f"Is {e2} {p2}?"
        answer = "yes"

        tasks.append({
            "premises": f"{fact}. {rule}.",
            "question": question,
            "answer": answer,
            "conclusions": [(e2, p2)],
            "task_type": "propositional",
        })
    return tasks


# ======================================================================
# ReasoningEvaluator
# ======================================================================

class ReasoningEvaluator:
    """
    Assess a model on reasoning tasks.

    The assessor accepts tasks as dicts with keys: premises, question, answer.
    The model is expected to accept a dict of tensors and return text or logits.

    Args:
        model: torch.nn.Module or callable that generates answers.
        task_type: One of 'babi', 'propositional', 'proofwriter'.
        answer_fn: Optional callable (model, premises, question) -> str.
    """

    def __init__(
        self,
        model: nn.Module,
        task_type: str = "babi",
        answer_fn: Optional[Callable] = None,
    ) -> None:
        self.model = model
        self.task_type = task_type
        self.answer_fn = answer_fn or self._default_answer_fn

    def run_assessment(
        self,
        tasks: List[Dict[str, Any]],
    ) -> ReasoningResult:
        """
        Assess on a list of tasks.

        Each task dict must have: 'premises', 'question', 'answer'.
        Optionally: 'conclusions' for consistency, 'proof_steps' for proof accuracy.
        """
        t0 = time.time()
        em_scores: List[float] = []
        consistency_scores: List[float] = []
        proof_scores: List[float] = []
        details: List[Dict[str, Any]] = []
        per_task_type: Dict[str, List[float]] = {}

        for task in tasks:
            prediction = self.answer_fn(
                self.model, task["premises"], task["question"]
            )
            answer = task["answer"]

            # Exact match
            if isinstance(answer, list):
                em = exact_match_multi(prediction, answer)
            else:
                em = exact_match_score(prediction, str(answer))
            em_scores.append(em)

            # Per task type
            tt = task.get("task_type", self.task_type)
            per_task_type.setdefault(tt, []).append(em)

            # Logical consistency
            conclusions = task.get("conclusions")
            if conclusions:
                checker = LogicalConsistencyChecker()
                # Parse simple facts from premises
                self._parse_premises(checker, task["premises"])
                n_con, n_tot = checker.check_conclusions(conclusions)
                cons = n_con / max(n_tot, 1)
                consistency_scores.append(cons)

            # Proof accuracy
            proof_steps = task.get("proof_steps")
            pred_steps = task.get("predicted_proof_steps")
            if proof_steps and pred_steps:
                pa, _, _ = compute_proof_accuracy(pred_steps, proof_steps)
                proof_scores.append(pa)

            details.append({
                "prediction": prediction,
                "answer": answer,
                "exact_match": em,
            })

        duration = time.time() - t0

        return ReasoningResult(
            exact_match=float(np.mean(em_scores)) if em_scores else 0.0,
            logical_consistency=(
                float(np.mean(consistency_scores)) if consistency_scores else float("nan")
            ),
            proof_accuracy=(
                float(np.mean(proof_scores)) if proof_scores else float("nan")
            ),
            num_samples=len(tasks),
            task_type=self.task_type,
            per_task_accuracy={
                k: float(np.mean(v)) for k, v in per_task_type.items()
            },
            duration_seconds=duration,
            details=details,
        )

    # Alias for backward compat with SKILL.md contract
    evaluate = run_assessment

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _default_answer_fn(
        self, model: nn.Module, premises: str, question: str
    ) -> str:
        """
        Default: encode text as tensor, run model, decode argmax to string.
        For mock testing the model can store a lookup.
        """
        if hasattr(model, "answer"):
            return model.answer(premises, question)
        # Fallback: return empty string
        return ""

    @staticmethod
    def _parse_premises(checker: LogicalConsistencyChecker, premises: str) -> None:
        """Parse simple 'X is Y' facts and 'If X is Y then Z is W' rules."""
        sentences = premises.replace(".", " .").split(".")
        for s in sentences:
            s = s.strip().lower()
            # Rule pattern
            rule_match = re.match(
                r"if\s+(\w+)\s+is\s+(\w+)\s+then\s+(\w+)\s+is\s+(\w+)", s
            )
            if rule_match:
                checker.add_rule(
                    rule_match.group(1), rule_match.group(2),
                    rule_match.group(3), rule_match.group(4),
                )
                continue
            # Fact pattern
            fact_match = re.match(r"(\w+)\s+is\s+(\w+)", s)
            if fact_match:
                checker.add_fact(fact_match.group(1), fact_match.group(2))


# ======================================================================
# Mock models for testing
# ======================================================================

class _MockReasoningModel(nn.Module):
    """Answers with the last location mentioned in premises."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self._locations = ["bathroom", "kitchen", "garden", "hallway", "office", "bedroom"]

    def answer(self, premises: str, question: str) -> str:
        words = premises.lower().split()
        for loc in reversed(self._locations):
            if loc in words:
                return loc
        return "unknown"

    def forward(self, inputs, **kw):
        return torch.zeros(1)


class _PerfectReasoningModel(nn.Module):
    """Cheats: stores the correct answers."""

    def __init__(self, tasks: List[Dict[str, Any]]) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self._answers: Dict[str, str] = {}
        for t in tasks:
            key = normalize_answer(t["premises"] + " " + t["question"])
            ans = t["answer"] if isinstance(t["answer"], str) else t["answer"][0]
            self._answers[key] = ans

    def answer(self, premises: str, question: str) -> str:
        key = normalize_answer(premises + " " + question)
        return self._answers.get(key, "unknown")

    def forward(self, inputs, **kw):
        return torch.zeros(1)


class _RandomReasoningModel(nn.Module):
    """Returns a random location."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self._rng = np.random.RandomState(seed)
        self._options = ["bathroom", "kitchen", "garden", "hallway", "office", "bedroom"]

    def answer(self, premises: str, question: str) -> str:
        return self._rng.choice(self._options)

    def forward(self, inputs, **kw):
        return torch.zeros(1)


# ======================================================================
# Self-tests (20+)
# ======================================================================

def _run_self_tests() -> None:
    passed = 0
    failed = 0

    def check(cond: bool, name: str) -> None:
        nonlocal passed, failed
        if cond:
            passed += 1
            print(f"  PASS: {name}")
        else:
            failed += 1
            print(f"  FAIL: {name}")

    def approx(a: float, b: float, tol: float = 0.05) -> bool:
        if math.isnan(a) and math.isnan(b):
            return True
        if math.isnan(a) or math.isnan(b):
            return False
        return abs(a - b) < tol

    print("=" * 60)
    print("ReasoningEvaluator Self-Tests")
    print("=" * 60)

    # ---- normalize_answer ----
    check(normalize_answer("  The Bathroom  ") == "bathroom", "T01 normalize basic")
    check(normalize_answer("A cat.") == "cat", "T02 normalize article + punct")
    check(normalize_answer("AN apple!") == "apple", "T03 normalize AN + punct")
    check(normalize_answer("") == "", "T04 normalize empty")

    # ---- exact_match_score ----
    check(exact_match_score("bathroom", "Bathroom") == 1.0, "T05 exact_match case insensitive")
    check(exact_match_score("kitchen", "bathroom") == 0.0, "T06 exact_match mismatch")
    check(exact_match_score("The Garden", "garden") == 1.0, "T07 exact_match strip article")

    # ---- exact_match_multi ----
    check(exact_match_multi("bathroom", ["kitchen", "bathroom"]) == 1.0, "T08 multi match")
    check(exact_match_multi("bedroom", ["kitchen", "bathroom"]) == 0.0, "T09 multi no match")

    # ---- numeric_match ----
    check(numeric_match("3.14", "3.14") is True, "T10 numeric match equal")
    check(numeric_match("3.140001", "3.14") is True, "T11 numeric match within tol")
    check(numeric_match("abc", "3.14") is False, "T12 numeric match non-numeric")

    # ---- LogicalConsistencyChecker ----
    lcc = LogicalConsistencyChecker()
    lcc.add_fact("cat", "fluffy")
    lcc.add_rule("cat", "fluffy", "dog", "happy")
    derived = lcc.derive()
    check(derived.get("dog") == "happy", "T13 logic derivation")
    check(lcc.check_conclusion("dog", "happy") is True, "T14 conclusion consistent")
    check(lcc.check_conclusion("dog", "sad") is False, "T15 conclusion inconsistent")
    check(lcc.check_conclusion("bird", "fast") is None, "T16 conclusion unknown")

    # ---- compute_proof_accuracy ----
    pa, c, t = compute_proof_accuracy(["step A", "step B"], ["step A", "step B"])
    check(approx(pa, 1.0), "T17 perfect proof accuracy")
    pa2, _, _ = compute_proof_accuracy(["step A", "wrong"], ["step A", "step B"])
    check(approx(pa2, 0.5), "T18 half proof accuracy")
    pa3, _, _ = compute_proof_accuracy([], [])
    check(approx(pa3, 1.0), "T19 empty proof accuracy")

    # ---- generate_babi_tasks ----
    tasks = generate_babi_tasks(n_tasks=50, seed=42)
    check(len(tasks) == 50, "T20 generated 50 bAbI tasks")
    check("premises" in tasks[0], "T21 task has premises")
    check("question" in tasks[0], "T22 task has question")
    check("answer" in tasks[0], "T23 task has answer")

    # ---- generate_logic_tasks ----
    ltasks = generate_logic_tasks(n_tasks=30, seed=42)
    check(len(ltasks) == 30, "T24 generated 30 logic tasks")
    check("conclusions" in ltasks[0], "T25 logic task has conclusions")

    # ---- ReasoningEvaluator with perfect model ----
    perfect = _PerfectReasoningModel(tasks)
    assessor = ReasoningEvaluator(perfect, task_type="babi")
    result = assessor.run_assessment(tasks)
    check(isinstance(result, ReasoningResult), "T26 returns ReasoningResult")
    check(approx(result.exact_match, 1.0), "T27 perfect model exact_match=1.0")
    check(result.num_samples == 50, "T28 num_samples correct")

    # ---- ReasoningEvaluator with random model ----
    rand_model = _RandomReasoningModel(seed=0)
    rand_assessor = ReasoningEvaluator(rand_model, task_type="babi")
    res_rand = rand_assessor.run_assessment(tasks)
    check(res_rand.exact_match < 0.8, "T29 random model EM < 0.8")

    # ---- to_dict ----
    d = result.to_dict()
    check("exact_match" in d, "T30 to_dict has exact_match")
    check("logical_consistency" in d, "T31 to_dict has logical_consistency")
    check("per_task_accuracy" in d, "T32 to_dict has per_task_accuracy")

    # ---- per_task_accuracy populated ----
    check(len(result.per_task_accuracy) > 0, "T33 per_task_accuracy non-empty")

    # ---- duration ----
    check(result.duration_seconds >= 0.0, "T34 duration >= 0")

    # ---- Logic tasks with assessor ----
    logic_model = _PerfectReasoningModel(ltasks)
    logic_assessor = ReasoningEvaluator(logic_model, task_type="propositional")
    res_logic = logic_assessor.run_assessment(ltasks)
    check(approx(res_logic.exact_match, 1.0), "T35 perfect logic EM=1.0")

    # ---- ReasoningResult NaN handling ----
    rr = ReasoningResult(
        exact_match=0.5, logical_consistency=float("nan"),
        proof_accuracy=float("nan"), num_samples=10,
    )
    d_nan = rr.to_dict()
    check(d_nan["logical_consistency"] is None, "T36 NaN -> None")

    # ---- Mock reasoning model (heuristic) ----
    mock = _MockReasoningModel()
    mock_assessor = ReasoningEvaluator(mock, task_type="babi")
    res_mock = mock_assessor.run_assessment(tasks[:20])
    check(0.0 <= res_mock.exact_match <= 1.0, "T37 mock EM in [0,1]")

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")
    print("=" * 60)
    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    _run_self_tests()
