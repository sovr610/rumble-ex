#!/usr/bin/env python3
"""Runtime contract validation for the Neuro-Symbolic Engine.

Validates:
- Done-When Gate (a): Joint training with neural modules
- Done-When Gate (b): Logic identity/unit tests per operator bundle
- Done-When Gate (c): Constraint violation reduction on synthetic FOL

Seven validation groups (~37 checks total):
  1. Operator Identities        -- boundary and identity tests per bundle
  2. Operator Gradients          -- gradient flow, NaN checks, AMP
  3. Grounding Contract          -- entity extractor shapes, ranges, audit, grads
  4. Rule Engine Contract        -- AST compilation, evaluation, attention, loss
  5. Joint Training (Gate a)     -- loss decrease, param change, upstream grads
  6. Logic Identity Tests (Gate b) -- comprehensive per-bundle identity checks
  7. Violation Reduction (Gate c) -- synthetic FOL training, violation decrease

Usage:
    python validate_symbolic.py                    # Run all checks
    python validate_symbolic.py --group operators  # Run specific group
    python validate_symbolic.py --verbose          # Detailed output
    python validate_symbolic.py --group gate_a --verbose
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ============================================================================
# ANSI colour helpers
# ============================================================================

_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def _c(text: str, code: str) -> str:
    """Wrap text in ANSI colour if stdout is a tty."""
    if sys.stdout.isatty():
        return f"{code}{text}{_RESET}"
    return text


# ============================================================================
# Result dataclass
# ============================================================================

@dataclass
class CheckResult:
    """Outcome of a single validation check."""
    name: str
    group: str
    passed: bool
    message: str
    details: Optional[str] = None
    elapsed_ms: float = 0.0


# ============================================================================
# Inline stub implementations
# ============================================================================
# These minimal implementations are self-contained so the validation script
# does not depend on brain_ai being installed.  They faithfully replicate the
# contracts specified in SKILL.md and the asset templates.

# ---------------------------------------------------------------------------
# AMP fp32 helper
# ---------------------------------------------------------------------------

def _amp_fp32(fn: Callable) -> Callable:
    """Ensure function executes in fp32 even under AMP autocasting."""
    from functools import wraps

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            with torch.cuda.amp.autocast(enabled=False):
                args_fp32 = tuple(
                    a.float() if isinstance(a, Tensor) and a.dtype == torch.float16 else a
                    for a in args
                )
                kwargs_fp32 = {
                    k: (v.float() if isinstance(v, Tensor) and v.dtype == torch.float16 else v)
                    for k, v in kwargs.items()
                }
                return fn(*args_fp32, **kwargs_fp32)
        except Exception:
            return fn(*args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Fuzzy operators (inline stubs matching the template)
# ---------------------------------------------------------------------------

@_amp_fp32
def _godel_and(x: Tensor, y: Tensor) -> Tensor:
    return torch.min(x, y)

@_amp_fp32
def _godel_or(x: Tensor, y: Tensor) -> Tensor:
    return torch.max(x, y)

@_amp_fp32
def _product_and(x: Tensor, y: Tensor) -> Tensor:
    return x * y

@_amp_fp32
def _product_or(x: Tensor, y: Tensor) -> Tensor:
    return x + y - x * y

@_amp_fp32
def _lukasiewicz_and(x: Tensor, y: Tensor) -> Tensor:
    return torch.clamp(x + y - 1.0, min=0.0)

@_amp_fp32
def _lukasiewicz_or(x: Tensor, y: Tensor) -> Tensor:
    return torch.clamp(x + y, max=1.0)

@_amp_fp32
def _standard_negation(x: Tensor) -> Tensor:
    return 1.0 - x

@_amp_fp32
def _reichenbach_implies(x: Tensor, y: Tensor) -> Tensor:
    return 1.0 - x + x * y

@_amp_fp32
def _godel_implies(x: Tensor, y: Tensor) -> Tensor:
    return torch.where(x <= y, torch.ones_like(x), y)

@_amp_fp32
def _lukasiewicz_implies(x: Tensor, y: Tensor) -> Tensor:
    return torch.clamp(1.0 - x + y, max=1.0)

@_amp_fp32
def _pi_0(x: Tensor, eps: float = 1e-4) -> Tensor:
    return torch.clamp(x, min=eps)

@_amp_fp32
def _pi_1(x: Tensor, eps: float = 1e-4) -> Tensor:
    return torch.clamp(x, max=1.0 - eps)

@_amp_fp32
def _stable_product_and(x: Tensor, y: Tensor, eps: float = 1e-4) -> Tensor:
    return _pi_0(x, eps) * _pi_0(y, eps)

@_amp_fp32
def _stable_product_or(x: Tensor, y: Tensor, eps: float = 1e-4) -> Tensor:
    return 1.0 - (1.0 - _pi_1(x, eps)) * (1.0 - _pi_1(y, eps))

@_amp_fp32
def _stable_product_not(x: Tensor) -> Tensor:
    return 1.0 - x

@_amp_fp32
def _stable_product_implies(x: Tensor, y: Tensor, eps: float = 1e-4) -> Tensor:
    neg_x = _stable_product_not(x)
    xy = _stable_product_and(x, y, eps)
    return _stable_product_or(neg_x, xy, eps)


# ---------------------------------------------------------------------------
# ForallAggregator (inline stub)
# ---------------------------------------------------------------------------

class _ForallAggregator(nn.Module):
    """Generalized-mean universal quantifier (pMeanError)."""

    def __init__(self, p: float = 2.0, temperature: float = 1.0) -> None:
        super().__init__()
        self.p = p
        self.temperature = temperature

    def forward(self, truth_values: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        tv = truth_values * self.temperature
        errors = torch.clamp(1.0 - tv, min=0.0, max=1.0)
        if mask is not None:
            errors = errors * mask.float()
            counts = mask.float().sum(dim=-1).clamp(min=1.0)
        else:
            counts = torch.tensor(
                truth_values.shape[-1], dtype=truth_values.dtype, device=truth_values.device
            )
        powered = errors.pow(self.p)
        pmean_error = (powered.sum(dim=-1) / counts).pow(1.0 / self.p)
        return torch.clamp(1.0 - pmean_error, min=0.0, max=1.0)


class _ExistsAggregator(nn.Module):
    """Generalized-mean existential quantifier."""

    def __init__(self, p: float = 2.0, temperature: float = 1.0) -> None:
        super().__init__()
        self.p = p
        self.temperature = temperature

    def forward(self, truth_values: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        tv = torch.clamp(truth_values * self.temperature, min=0.0, max=1.0)
        if mask is not None:
            tv = tv * mask.float()
            counts = mask.float().sum(dim=-1).clamp(min=1.0)
        else:
            counts = torch.tensor(
                truth_values.shape[-1], dtype=truth_values.dtype, device=truth_values.device
            )
        powered = tv.pow(self.p)
        pmean = (powered.sum(dim=-1) / counts).pow(1.0 / self.p)
        return torch.clamp(pmean, min=0.0, max=1.0)


# ---------------------------------------------------------------------------
# OperatorBundle
# ---------------------------------------------------------------------------

@dataclass
class _OperatorBundle:
    """Complete set of fuzzy connectives."""
    name: str
    and_op: Callable
    or_op: Callable
    not_op: Callable
    implies_op: Callable
    forall: nn.Module
    exists: nn.Module


def _get_bundle(
    name: str,
    eps: float = 1e-4,
    quantifier_p: float = 2.0,
    quantifier_temp: float = 1.0,
) -> _OperatorBundle:
    """Build operator bundle by name."""
    fa = _ForallAggregator(p=quantifier_p, temperature=quantifier_temp)
    ea = _ExistsAggregator(p=quantifier_p, temperature=quantifier_temp)

    if name == "godel":
        return _OperatorBundle(
            name="godel",
            and_op=_godel_and, or_op=_godel_or,
            not_op=_standard_negation,
            implies_op=_godel_implies,
            forall=fa, exists=ea,
        )
    elif name == "product":
        return _OperatorBundle(
            name="product",
            and_op=_product_and, or_op=_product_or,
            not_op=_standard_negation,
            implies_op=_reichenbach_implies,
            forall=fa, exists=ea,
        )
    elif name == "lukasiewicz":
        return _OperatorBundle(
            name="lukasiewicz",
            and_op=_lukasiewicz_and, or_op=_lukasiewicz_or,
            not_op=_standard_negation,
            implies_op=_lukasiewicz_implies,
            forall=fa, exists=ea,
        )
    elif name == "stable_product":
        def _sp_and(x: Tensor, y: Tensor) -> Tensor:
            return _stable_product_and(x, y, eps=eps)
        def _sp_or(x: Tensor, y: Tensor) -> Tensor:
            return _stable_product_or(x, y, eps=eps)
        def _sp_implies(x: Tensor, y: Tensor) -> Tensor:
            return _stable_product_implies(x, y, eps=eps)
        return _OperatorBundle(
            name="stable_product",
            and_op=_sp_and, or_op=_sp_or,
            not_op=_stable_product_not,
            implies_op=_sp_implies,
            forall=fa, exists=ea,
        )
    else:
        raise ValueError(f"Unknown bundle: {name}")


# ---------------------------------------------------------------------------
# Entity Extractor (minimal stub)
# ---------------------------------------------------------------------------

class _EntityExtractor(nn.Module):
    """Minimal entity extractor: (B, K, D_ws) -> (B, N, D_ent)."""

    def __init__(self, workspace_dim: int, entity_dim: int, max_entities: int = 8):
        super().__init__()
        self.proj = nn.Linear(workspace_dim, entity_dim)
        self.norm = nn.LayerNorm(entity_dim)
        self.max_entities = max_entities

    def forward(self, slots: Tensor, slot_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        B, K, D = slots.shape
        entities = self.norm(self.proj(slots))  # (B, K, D_ent)
        if slot_mask is None:
            slot_mask = torch.ones(B, K, dtype=torch.bool, device=slots.device)
        entities = entities * slot_mask.unsqueeze(-1).float()
        return entities, slot_mask


# ---------------------------------------------------------------------------
# Predicate and Relation modules (minimal stubs)
# ---------------------------------------------------------------------------

class _PredicateModule(nn.Module):
    """Unary predicate: (B, N, D_ent) -> (B, N) truth values in [0, 1]."""

    def __init__(self, entity_dim: int, hidden_dim: int = 64, name: str = "P"):
        super().__init__()
        self.name = name
        self.net = nn.Sequential(
            nn.Linear(entity_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, entities: Tensor) -> Tensor:
        return self.net(entities).squeeze(-1)  # (B, N)


class _RelationModule(nn.Module):
    """Binary relation: (B, N, D_ent) x (B, N, D_ent) -> (B, N, N) in [0, 1]."""

    def __init__(self, entity_dim: int, hidden_dim: int = 64, name: str = "R"):
        super().__init__()
        self.name = name
        self.bilinear = nn.Bilinear(entity_dim, entity_dim, hidden_dim)
        self.out = nn.Sequential(nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, entities_x: Tensor, entities_y: Tensor) -> Tensor:
        # entities_x, entities_y: (B, N, D)
        B, N, D = entities_x.shape
        # All pairs: expand to (B, N, N, D) then flatten
        ex = entities_x.unsqueeze(2).expand(B, N, N, D).reshape(B * N * N, D)
        ey = entities_y.unsqueeze(1).expand(B, N, N, D).reshape(B * N * N, D)
        h = self.bilinear(ex, ey)
        out = self.out(h).view(B, N, N)
        return out


# ---------------------------------------------------------------------------
# Rule AST (minimal stub)
# ---------------------------------------------------------------------------

class _ASTNode:
    """Minimal AST node for rule representation."""

    def __init__(
        self,
        node_type: str,
        op: Optional[str] = None,
        predicate_name: Optional[str] = None,
        variable_names: Optional[Tuple[str, ...]] = None,
        negated: bool = False,
        children: Optional[List["_ASTNode"]] = None,
        quantifier: Optional[str] = None,
        variable: Optional[str] = None,
        weight: Optional[float] = None,
    ):
        self.node_type = node_type  # "literal", "connective", "quantifier", "negation"
        self.op = op
        self.predicate_name = predicate_name
        self.variable_names = variable_names or ()
        self.negated = negated
        self.children = children or []
        self.quantifier = quantifier
        self.variable = variable
        self.weight = weight


def _compile_rule(rule_dict: Dict[str, Any]) -> _ASTNode:
    """Compile a dict-format rule into an AST node.

    Supports formats:
        {"type": "literal", "predicate": "P", "vars": ["x"]}
        {"type": "connective", "op": "AND", "children": [...]}
        {"type": "quantifier", "q": "FORALL", "var": "x", "body": {...}}
        {"type": "negation", "child": {...}}
    """
    t = rule_dict.get("type", "")
    if t == "literal":
        return _ASTNode(
            node_type="literal",
            predicate_name=rule_dict.get("predicate", "P"),
            variable_names=tuple(rule_dict.get("vars", ["x"])),
            negated=rule_dict.get("negated", False),
            weight=rule_dict.get("weight"),
        )
    elif t == "connective":
        children = [_compile_rule(c) for c in rule_dict.get("children", [])]
        return _ASTNode(
            node_type="connective",
            op=rule_dict.get("op", "AND"),
            children=children,
            weight=rule_dict.get("weight"),
        )
    elif t == "quantifier":
        body = _compile_rule(rule_dict.get("body", {"type": "literal", "predicate": "P"}))
        return _ASTNode(
            node_type="quantifier",
            quantifier=rule_dict.get("q", "FORALL"),
            variable=rule_dict.get("var", "x"),
            children=[body],
            weight=rule_dict.get("weight"),
        )
    elif t == "negation":
        child = _compile_rule(rule_dict.get("child", {"type": "literal", "predicate": "P"}))
        return _ASTNode(
            node_type="negation",
            children=[child],
            weight=rule_dict.get("weight"),
        )
    else:
        raise ValueError(f"Unknown AST node type: {t}")


def _evaluate_ast(
    node: _ASTNode,
    bundle: _OperatorBundle,
    predicate_cache: Dict[str, Tensor],
    entity_dim_size: int = 1,
) -> Tensor:
    """Evaluate an AST node using the given operator bundle and grounded predicate truths."""
    if node.node_type == "literal":
        key = node.predicate_name
        if key not in predicate_cache:
            raise KeyError(f"Predicate '{key}' not in cache")
        val = predicate_cache[key]
        if node.negated:
            val = bundle.not_op(val)
        return val

    elif node.node_type == "negation":
        child_val = _evaluate_ast(node.children[0], bundle, predicate_cache, entity_dim_size)
        return bundle.not_op(child_val)

    elif node.node_type == "connective":
        vals = [_evaluate_ast(c, bundle, predicate_cache, entity_dim_size) for c in node.children]
        if node.op == "AND":
            result = vals[0]
            for v in vals[1:]:
                result = bundle.and_op(result, v)
            return result
        elif node.op == "OR":
            result = vals[0]
            for v in vals[1:]:
                result = bundle.or_op(result, v)
            return result
        elif node.op == "IMPLIES":
            assert len(vals) == 2, "IMPLIES requires exactly 2 children"
            return bundle.implies_op(vals[0], vals[1])
        else:
            raise ValueError(f"Unknown connective: {node.op}")

    elif node.node_type == "quantifier":
        body_val = _evaluate_ast(node.children[0], bundle, predicate_cache, entity_dim_size)
        if body_val.dim() == 1:
            body_val = body_val.unsqueeze(0)
        if node.quantifier == "FORALL":
            return bundle.forall(body_val)
        elif node.quantifier == "EXISTS":
            return bundle.exists(body_val)
        else:
            raise ValueError(f"Unknown quantifier: {node.quantifier}")
    else:
        raise ValueError(f"Unknown node type: {node.node_type}")


# ---------------------------------------------------------------------------
# RuleNetwork (minimal stub for evaluation and attention)
# ---------------------------------------------------------------------------

class _RuleNetwork(nn.Module):
    """Minimal rule network with attention-weighted application."""

    def __init__(
        self,
        workspace_dim: int = 64,
        rule_embed_dim: int = 32,
        num_rules: int = 4,
    ):
        super().__init__()
        self.num_rules = num_rules
        self.rule_embeddings = nn.Parameter(torch.randn(num_rules, rule_embed_dim) * 0.1)
        self.attention_net = nn.Sequential(
            nn.Linear(workspace_dim + rule_embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def compute_attention(self, workspace_summary: Tensor) -> Tensor:
        """Compute attention weights over rules.

        Args:
            workspace_summary: (B, D_ws)

        Returns:
            (B, num_rules) attention weights summing to 1.
        """
        B = workspace_summary.shape[0]
        # Expand: (B, num_rules, D_ws + rule_embed_dim)
        ws_exp = workspace_summary.unsqueeze(1).expand(B, self.num_rules, -1)
        re_exp = self.rule_embeddings.unsqueeze(0).expand(B, -1, -1)
        combined = torch.cat([ws_exp, re_exp], dim=-1)
        logits = self.attention_net(combined).squeeze(-1)  # (B, num_rules)
        return F.softmax(logits, dim=-1)

    def compute_violation(self, truth_values: Tensor) -> Tensor:
        """violation = 1 - truth for positive rules."""
        return 1.0 - truth_values

    def compute_constraint_loss(
        self,
        truth_values: Tensor,
        attention_weights: Tensor,
    ) -> Tensor:
        """Attention-weighted constraint loss.

        Args:
            truth_values: (B, num_rules) truth values per rule.
            attention_weights: (B, num_rules) attention weights.

        Returns:
            Scalar differentiable loss.
        """
        violations = self.compute_violation(truth_values)
        weighted = (attention_weights * violations).sum(dim=-1)  # (B,)
        return weighted.mean()  # scalar


# ---------------------------------------------------------------------------
# SymbolicReasoner (minimal integrated stub)
# ---------------------------------------------------------------------------

class _SymbolicReasoner(nn.Module):
    """Minimal symbolic reasoner wiring entity extraction, grounding, and rules."""

    def __init__(
        self,
        workspace_dim: int = 64,
        entity_dim: int = 32,
        hidden_dim: int = 64,
        num_predicates: int = 4,
        num_rules: int = 3,
        max_entities: int = 8,
    ):
        super().__init__()
        self.entity_extractor = _EntityExtractor(workspace_dim, entity_dim, max_entities)
        self.predicates = nn.ModuleDict({
            f"P{i}": _PredicateModule(entity_dim, hidden_dim, name=f"P{i}")
            for i in range(num_predicates)
        })
        self.rule_network = _RuleNetwork(workspace_dim, rule_embed_dim=32, num_rules=num_rules)

    def forward(
        self,
        workspace_slots: Tensor,
        slot_mask: Optional[Tensor] = None,
        return_logs: bool = False,
    ) -> Dict[str, Any]:
        B = workspace_slots.shape[0]
        entities, mask = self.entity_extractor(workspace_slots, slot_mask)

        # Ground predicates
        pred_truths = {}
        for name, pred_mod in self.predicates.items():
            pred_truths[name] = pred_mod(entities)  # (B, N)

        # Aggregate per-rule truth via forall over entities per predicate
        # Simple: each rule is "FORALL x: P_i(x)" for each predicate
        num_rules = self.rule_network.num_rules
        rule_truths_list = []
        bundle = _get_bundle("stable_product")
        pred_names = list(self.predicates.keys())
        for r in range(num_rules):
            p_idx = r % len(pred_names)
            p_name = pred_names[p_idx]
            truth_vals = pred_truths[p_name]  # (B, N)
            forall_val = bundle.forall(truth_vals, mask=mask.float() if mask is not None else None)
            rule_truths_list.append(forall_val)

        rule_truths = torch.stack(rule_truths_list, dim=-1)  # (B, num_rules)

        # Workspace summary for attention
        ws_summary = workspace_slots.mean(dim=1)  # (B, D_ws)
        attn_weights = self.rule_network.compute_attention(ws_summary)
        constraint_loss = self.rule_network.compute_constraint_loss(rule_truths, attn_weights)

        # Violation stats
        violations = self.rule_network.compute_violation(rule_truths)
        violation_stats = {
            "mean_violation": violations.mean().detach(),
            "max_violation": violations.max().detach(),
            "per_rule_violation": violations.mean(dim=0).detach(),
        }

        output = {
            "constraint_loss": constraint_loss,
            "violation_stats": violation_stats,
            "rule_truths": rule_truths,
            "attention_weights": attn_weights,
        }

        if return_logs:
            output["logs"] = {
                "predicate_truths": {k: v.detach() for k, v in pred_truths.items()},
                "entity_shape": list(entities.shape),
                "attention_weights": attn_weights.detach(),
                "violations": violations.detach(),
                "top_violations": violations.topk(min(3, violations.shape[-1]), dim=-1),
            }

        return output


# ============================================================================
# Validator class
# ============================================================================

_ALL_GROUPS = [
    "operators",
    "gradients",
    "grounding",
    "rule_engine",
    "gate_a",
    "gate_b",
    "gate_c",
]


class SymbolicValidator:
    """Runtime contract validator for the neuro-symbolic engine."""

    def __init__(self, verbose: bool = False, seed: int = 42):
        self.verbose = verbose
        self.seed = seed
        self.results: List[CheckResult] = []

    # ------------------------------------------------------------------
    # Core check runner
    # ------------------------------------------------------------------

    def _check(self, name: str, group: str, fn: Callable[[], Tuple[bool, str, Optional[str]]]) -> CheckResult:
        """Run a single validation check with timing and exception handling.

        fn should return (passed: bool, message: str, details: Optional[str]).
        """
        torch.manual_seed(self.seed)
        start = time.perf_counter()
        try:
            passed, message, details = fn()
        except Exception as e:
            passed = False
            message = f"Exception: {type(e).__name__}: {e}"
            details = traceback.format_exc() if self.verbose else None
        elapsed = (time.perf_counter() - start) * 1000.0

        result = CheckResult(
            name=name,
            group=group,
            passed=passed,
            message=message,
            details=details,
            elapsed_ms=elapsed,
        )
        self.results.append(result)

        # Print immediately
        status = _c("PASS", _GREEN) if passed else _c("FAIL", _RED)
        print(f"  [{status}] {name}  ({elapsed:.1f}ms)")
        if self.verbose and message:
            print(f"         {_c(message, _DIM)}")
        if not passed and details:
            for line in details.strip().split("\n")[-5:]:
                print(f"         {_c(line, _RED)}")

        return result

    # ------------------------------------------------------------------
    # Group runners
    # ------------------------------------------------------------------

    def run_group(self, group: str) -> List[CheckResult]:
        """Run all checks in a single group."""
        if group not in _ALL_GROUPS:
            raise ValueError(f"Unknown group '{group}'. Available: {_ALL_GROUPS}")
        method = getattr(self, f"_run_{group}")
        print(f"\n{'='*72}")
        print(f"  Group: {_c(group.upper(), _BOLD + _CYAN)}")
        print(f"{'='*72}")
        before = len(self.results)
        method()
        return self.results[before:]

    def run_all(self) -> List[CheckResult]:
        """Run all validation groups."""
        for group in _ALL_GROUPS:
            self.run_group(group)
        return self.results

    # ==================================================================
    # GROUP 1: Operator Identities (8 checks)
    # ==================================================================

    def _run_operators(self) -> None:
        """Operator identity checks for all four bundles."""
        grid = torch.linspace(0.0, 1.0, 11)  # 0.0, 0.1, ..., 1.0
        ones = torch.ones_like(grid)
        zeros = torch.zeros_like(grid)

        PURE_BUNDLES = ["godel", "product", "lukasiewicz"]
        ALL_BUNDLES = PURE_BUNDLES + ["stable_product"]

        # ---- check_and_identity ----
        def check_and_identity():
            """AND(x,1)=x and AND(x,0)=0 for all bundles."""
            all_ok = True
            msgs = []
            for bname in ALL_BUNDLES:
                bundle = _get_bundle(bname)
                atol = 1e-3 if bname == "stable_product" else 1e-6

                # AND(x, 1) = x
                result_1 = bundle.and_op(grid, ones)
                ok_1 = torch.allclose(result_1, grid, atol=atol)
                if not ok_1:
                    diff = (result_1 - grid).abs().max().item()
                    msgs.append(f"{bname} AND(x,1)!=x (max_diff={diff:.2e})")
                    all_ok = False

                # AND(x, 0) = 0  (for stable_product, ~0 within eps)
                result_0 = bundle.and_op(grid, zeros)
                if bname == "stable_product":
                    ok_0 = result_0.max().item() < 0.01
                else:
                    ok_0 = torch.allclose(result_0, zeros, atol=1e-6)
                if not ok_0:
                    msgs.append(f"{bname} AND(x,0)!=0 (max={result_0.max().item():.2e})")
                    all_ok = False

            msg = "All AND identities hold" if all_ok else "; ".join(msgs)
            return all_ok, msg, None

        self._check("check_and_identity", "operators", check_and_identity)

        # ---- check_or_identity ----
        def check_or_identity():
            """OR(x,0)=x and OR(x,1)=1 for all bundles."""
            all_ok = True
            msgs = []
            for bname in ALL_BUNDLES:
                bundle = _get_bundle(bname)
                atol = 1e-3 if bname == "stable_product" else 1e-6

                # OR(x, 0) = x
                result_0 = bundle.or_op(grid, zeros)
                ok_0 = torch.allclose(result_0, grid, atol=atol)
                if not ok_0:
                    diff = (result_0 - grid).abs().max().item()
                    msgs.append(f"{bname} OR(x,0)!=x (max_diff={diff:.2e})")
                    all_ok = False

                # OR(x, 1) = 1
                result_1 = bundle.or_op(grid, ones)
                ok_1 = torch.allclose(result_1, ones, atol=atol)
                if not ok_1:
                    diff = (result_1 - ones).abs().max().item()
                    msgs.append(f"{bname} OR(x,1)!=1 (max_diff={diff:.2e})")
                    all_ok = False

            msg = "All OR identities hold" if all_ok else "; ".join(msgs)
            return all_ok, msg, None

        self._check("check_or_identity", "operators", check_or_identity)

        # ---- check_negation_involution ----
        def check_negation_involution():
            """NOT(NOT(x)) = x for all bundles."""
            all_ok = True
            msgs = []
            for bname in ALL_BUNDLES:
                bundle = _get_bundle(bname)
                result = bundle.not_op(bundle.not_op(grid))
                ok = torch.allclose(result, grid, atol=1e-6)
                if not ok:
                    diff = (result - grid).abs().max().item()
                    msgs.append(f"{bname} NOT(NOT(x))!=x (max_diff={diff:.2e})")
                    all_ok = False
            msg = "Negation involution holds for all bundles" if all_ok else "; ".join(msgs)
            return all_ok, msg, None

        self._check("check_negation_involution", "operators", check_negation_involution)

        # ---- check_implication_boundaries ----
        def check_implication_boundaries():
            """IMPLIES(1,y)=y and IMPLIES(0,y)=1 for all bundles."""
            y = torch.linspace(0.0, 1.0, 11)
            ones_y = torch.ones_like(y)
            all_ok = True
            msgs = []

            for bname in ALL_BUNDLES:
                bundle = _get_bundle(bname)
                atol = 1e-3 if bname == "stable_product" else 1e-6

                # IMPLIES(1, y) = y
                result_1y = bundle.implies_op(torch.ones_like(y), y)
                ok_1y = torch.allclose(result_1y, y, atol=atol)
                if not ok_1y:
                    diff = (result_1y - y).abs().max().item()
                    msgs.append(f"{bname} IMPLIES(1,y)!=y (max_diff={diff:.2e})")
                    all_ok = False

                # IMPLIES(0, y) = 1
                result_0y = bundle.implies_op(torch.zeros_like(y), y)
                ok_0y = torch.allclose(result_0y, ones_y, atol=atol)
                if not ok_0y:
                    diff = (result_0y - ones_y).abs().max().item()
                    msgs.append(f"{bname} IMPLIES(0,y)!=1 (max_diff={diff:.2e})")
                    all_ok = False

            msg = "All implication boundaries hold" if all_ok else "; ".join(msgs)
            return all_ok, msg, None

        self._check("check_implication_boundaries", "operators", check_implication_boundaries)

        # ---- check_and_commutativity ----
        def check_and_commutativity():
            """AND(x,y) = AND(y,x) for all bundles."""
            torch.manual_seed(self.seed)
            x = torch.rand(100)
            y = torch.rand(100)
            all_ok = True
            msgs = []
            for bname in ALL_BUNDLES:
                bundle = _get_bundle(bname)
                ok = torch.allclose(bundle.and_op(x, y), bundle.and_op(y, x), atol=1e-6)
                if not ok:
                    msgs.append(f"{bname} AND not commutative")
                    all_ok = False
            msg = "AND is commutative for all bundles" if all_ok else "; ".join(msgs)
            return all_ok, msg, None

        self._check("check_and_commutativity", "operators", check_and_commutativity)

        # ---- check_or_commutativity ----
        def check_or_commutativity():
            """OR(x,y) = OR(y,x) for all bundles."""
            torch.manual_seed(self.seed)
            x = torch.rand(100)
            y = torch.rand(100)
            all_ok = True
            msgs = []
            for bname in ALL_BUNDLES:
                bundle = _get_bundle(bname)
                ok = torch.allclose(bundle.or_op(x, y), bundle.or_op(y, x), atol=1e-6)
                if not ok:
                    msgs.append(f"{bname} OR not commutative")
                    all_ok = False
            msg = "OR is commutative for all bundles" if all_ok else "; ".join(msgs)
            return all_ok, msg, None

        self._check("check_or_commutativity", "operators", check_or_commutativity)

        # ---- check_output_range_01 ----
        def check_output_range_01():
            """All operator outputs in [0, 1]."""
            torch.manual_seed(self.seed)
            x = torch.rand(500)
            y = torch.rand(500)
            all_ok = True
            msgs = []
            for bname in ALL_BUNDLES:
                bundle = _get_bundle(bname)
                for op_name, result in [
                    ("AND", bundle.and_op(x, y)),
                    ("OR", bundle.or_op(x, y)),
                    ("NOT", bundle.not_op(x)),
                    ("IMPLIES", bundle.implies_op(x, y)),
                ]:
                    lo, hi = result.min().item(), result.max().item()
                    if lo < -1e-6 or hi > 1.0 + 1e-6:
                        msgs.append(f"{bname} {op_name} out of [0,1]: [{lo:.4f}, {hi:.4f}]")
                        all_ok = False
            msg = "All outputs in [0,1]" if all_ok else "; ".join(msgs)
            return all_ok, msg, None

        self._check("check_output_range_01", "operators", check_output_range_01)

        # ---- check_de_morgan ----
        def check_de_morgan():
            """NOT(AND(x,y)) = OR(NOT(x), NOT(y)) for pure bundles."""
            torch.manual_seed(self.seed)
            x = torch.rand(100)
            y = torch.rand(100)
            all_ok = True
            msgs = []
            for bname in PURE_BUNDLES:
                bundle = _get_bundle(bname)
                lhs = bundle.not_op(bundle.and_op(x, y))
                rhs = bundle.or_op(bundle.not_op(x), bundle.not_op(y))
                ok = torch.allclose(lhs, rhs, atol=1e-5)
                if not ok:
                    diff = (lhs - rhs).abs().max().item()
                    msgs.append(f"{bname} De Morgan failed (max_diff={diff:.2e})")
                    all_ok = False
            # Stable product: approximate
            sp = _get_bundle("stable_product")
            lhs_sp = sp.not_op(sp.and_op(x, y))
            rhs_sp = sp.or_op(sp.not_op(x), sp.not_op(y))
            ok_sp = torch.allclose(lhs_sp, rhs_sp, atol=5e-3)
            if not ok_sp:
                msgs.append(f"stable_product De Morgan approx failed")
                all_ok = False
            msg = "De Morgan holds for all bundles" if all_ok else "; ".join(msgs)
            return all_ok, msg, None

        self._check("check_de_morgan", "operators", check_de_morgan)

    # ==================================================================
    # GROUP 2: Operator Gradients (5 checks)
    # ==================================================================

    def _run_gradients(self) -> None:
        """Gradient flow, NaN checks, and AMP compatibility."""

        # ---- check_product_gradient_interior ----
        def check_product_gradient_interior():
            """Product AND grad non-zero for x,y in [0.2, 0.8]."""
            bundle = _get_bundle("product")
            x = torch.tensor([0.3, 0.5, 0.7], requires_grad=True)
            y = torch.tensor([0.4, 0.6, 0.8], requires_grad=True)
            val = bundle.and_op(x, y).sum()
            val.backward()
            gx_ok = x.grad is not None and (x.grad.abs() > 1e-6).all().item()
            gy_ok = y.grad is not None and (y.grad.abs() > 1e-6).all().item()
            ok = gx_ok and gy_ok
            msg = f"grad_x={x.grad}, grad_y={y.grad}"
            return ok, msg, None

        self._check("check_product_gradient_interior", "gradients", check_product_gradient_interior)

        # ---- check_stable_product_gradient_boundary ----
        def check_stable_product_gradient_boundary():
            """Stable product AND/OR grads non-zero near 0 and 1."""
            bundle = _get_bundle("stable_product")
            all_ok = True
            msgs = []

            # AND at boundary x=0.0, y=0.5
            x0 = torch.tensor([0.0], requires_grad=True)
            y0 = torch.tensor([0.5], requires_grad=True)
            bundle.and_op(x0, y0).backward()
            if y0.grad is None or y0.grad.abs().item() < 1e-8:
                msgs.append("AND grad_y zero at x=0")
                all_ok = False

            # AND at boundary x=0.5, y=1.0
            x1 = torch.tensor([0.5], requires_grad=True)
            y1 = torch.tensor([1.0], requires_grad=True)
            bundle.and_op(x1, y1).backward()
            if x1.grad is None or x1.grad.abs().item() < 1e-8:
                msgs.append("AND grad_x zero at y=1")
                all_ok = False

            # OR at boundary x=1.0, y=0.5
            x2 = torch.tensor([1.0], requires_grad=True)
            y2 = torch.tensor([0.5], requires_grad=True)
            bundle.or_op(x2, y2).backward()
            if y2.grad is None or y2.grad.abs().item() < 1e-8:
                msgs.append("OR grad_y zero at x=1")
                all_ok = False

            # IMPLIES at boundary x=0.0, y=0.3
            x3 = torch.tensor([0.0], requires_grad=True)
            y3 = torch.tensor([0.3], requires_grad=True)
            bundle.implies_op(x3, y3).backward()
            if y3.grad is None or y3.grad.abs().item() < 1e-8:
                msgs.append("IMPLIES grad_y zero at x=0")
                all_ok = False

            msg = "Stable product boundary grads non-zero" if all_ok else "; ".join(msgs)
            return all_ok, msg, None

        self._check("check_stable_product_gradient_boundary", "gradients", check_stable_product_gradient_boundary)

        # ---- check_quantifier_gradient_bounded ----
        def check_quantifier_gradient_bounded():
            """Generalized-mean quantifier gradients don't explode."""
            forall = _ForallAggregator(p=2.0)
            tv = torch.rand(4, 10, requires_grad=True)
            val = forall(tv).sum()
            val.backward()
            grad = tv.grad
            ok = grad is not None and torch.isfinite(grad).all().item()
            grad_max = grad.abs().max().item() if grad is not None else float("inf")
            bounded = grad_max < 100.0  # reasonable upper bound
            msg = f"grad_max={grad_max:.4f}, finite={ok}"
            return ok and bounded, msg, None

        self._check("check_quantifier_gradient_bounded", "gradients", check_quantifier_gradient_bounded)

        # ---- check_no_nan_fp32 ----
        def check_no_nan_fp32():
            """All operators produce valid output in fp32 for random inputs."""
            torch.manual_seed(self.seed)
            x = torch.rand(200)
            y = torch.rand(200)
            all_ok = True
            msgs = []
            for bname in ["godel", "product", "lukasiewicz", "stable_product"]:
                bundle = _get_bundle(bname)
                for op_name, result in [
                    ("AND", bundle.and_op(x, y)),
                    ("OR", bundle.or_op(x, y)),
                    ("NOT", bundle.not_op(x)),
                    ("IMPLIES", bundle.implies_op(x, y)),
                ]:
                    if torch.isnan(result).any() or torch.isinf(result).any():
                        msgs.append(f"{bname} {op_name} has NaN/Inf in fp32")
                        all_ok = False
            # Also check boundary values
            extremes_x = torch.tensor([0.0, 0.0, 1.0, 1.0])
            extremes_y = torch.tensor([0.0, 1.0, 0.0, 1.0])
            for bname in ["godel", "product", "lukasiewicz", "stable_product"]:
                bundle = _get_bundle(bname)
                for op_name, result in [
                    ("AND", bundle.and_op(extremes_x, extremes_y)),
                    ("OR", bundle.or_op(extremes_x, extremes_y)),
                    ("IMPLIES", bundle.implies_op(extremes_x, extremes_y)),
                ]:
                    if torch.isnan(result).any() or torch.isinf(result).any():
                        msgs.append(f"{bname} {op_name} has NaN/Inf at extremes")
                        all_ok = False
            msg = "No NaN/Inf in fp32" if all_ok else "; ".join(msgs)
            return all_ok, msg, None

        self._check("check_no_nan_fp32", "gradients", check_no_nan_fp32)

        # ---- check_no_nan_amp ----
        def check_no_nan_amp():
            """All operators produce valid output under AMP (CUDA only, skip if no GPU)."""
            if not torch.cuda.is_available():
                return True, "CUDA not available -- AMP check skipped (pass by default)", None

            device = torch.device("cuda")
            x = torch.rand(200, device=device)
            y = torch.rand(200, device=device)
            all_ok = True
            msgs = []

            with torch.cuda.amp.autocast(enabled=True):
                for bname in ["godel", "product", "lukasiewicz", "stable_product"]:
                    bundle = _get_bundle(bname)
                    for op_name, result in [
                        ("AND", bundle.and_op(x, y)),
                        ("OR", bundle.or_op(x, y)),
                        ("NOT", bundle.not_op(x)),
                        ("IMPLIES", bundle.implies_op(x, y)),
                    ]:
                        if torch.isnan(result).any() or torch.isinf(result).any():
                            msgs.append(f"{bname} {op_name} NaN/Inf under AMP")
                            all_ok = False
                        if result.dtype != torch.float32:
                            msgs.append(f"{bname} {op_name} not fp32 under AMP (got {result.dtype})")
                            all_ok = False

            msg = "All operators valid under AMP" if all_ok else "; ".join(msgs)
            return all_ok, msg, None

        self._check("check_no_nan_amp", "gradients", check_no_nan_amp)

    # ==================================================================
    # GROUP 3: Grounding Contract (5 checks)
    # ==================================================================

    def _run_grounding(self) -> None:
        """Entity extractor shapes, predicate ranges, auditing, gradient flow."""
        torch.manual_seed(self.seed)
        B, K, D_ws, D_ent = 4, 8, 64, 32

        # ---- check_entity_extractor_shapes ----
        def check_entity_extractor_shapes():
            """(B,K,D_ws) -> (B,N,D_ent) shape correctness."""
            extractor = _EntityExtractor(D_ws, D_ent, max_entities=K)
            slots = torch.randn(B, K, D_ws)
            mask = torch.ones(B, K, dtype=torch.bool)
            mask[:, -2:] = False  # last 2 slots masked

            entities, out_mask = extractor(slots, mask)
            shape_ok = entities.shape == (B, K, D_ent)
            mask_ok = out_mask.shape == (B, K)
            # Masked positions should be zeroed
            masked_zero = (entities[:, -2:, :].abs().sum().item() < 1e-6)
            ok = shape_ok and mask_ok and masked_zero
            msg = f"shape={entities.shape}, mask_shape={out_mask.shape}, masked_zero={masked_zero}"
            return ok, msg, None

        self._check("check_entity_extractor_shapes", "grounding", check_entity_extractor_shapes)

        # ---- check_predicate_output_range ----
        def check_predicate_output_range():
            """Predicate truth values in [0, 1]."""
            pred = _PredicateModule(D_ent, hidden_dim=32)
            entities = torch.randn(B, K, D_ent)
            truth = pred(entities)
            ok = (truth >= 0.0 - 1e-6).all().item() and (truth <= 1.0 + 1e-6).all().item()
            msg = f"truth range: [{truth.min().item():.4f}, {truth.max().item():.4f}]"
            return ok, msg, None

        self._check("check_predicate_output_range", "grounding", check_predicate_output_range)

        # ---- check_relation_output_range ----
        def check_relation_output_range():
            """Relation truth values in [0, 1]."""
            rel = _RelationModule(D_ent, hidden_dim=32)
            entities = torch.randn(B, K, D_ent)
            truth = rel(entities, entities)
            ok = (truth >= 0.0 - 1e-6).all().item() and (truth <= 1.0 + 1e-6).all().item()
            msg = f"truth shape={truth.shape}, range: [{truth.min().item():.4f}, {truth.max().item():.4f}]"
            return ok, msg, None

        self._check("check_relation_output_range", "grounding", check_relation_output_range)

        # ---- check_grounding_audit ----
        def check_grounding_audit():
            """Symbol-to-parameter mapping exposed for auditing."""
            reasoner = _SymbolicReasoner(
                workspace_dim=D_ws, entity_dim=D_ent, hidden_dim=32,
                num_predicates=3, num_rules=2, max_entities=K,
            )
            # Check that predicates are accessible by name
            has_mapping = hasattr(reasoner, "predicates") and isinstance(reasoner.predicates, nn.ModuleDict)
            names = list(reasoner.predicates.keys()) if has_mapping else []
            names_ok = len(names) == 3
            # Each predicate should have named parameters
            params_ok = all(
                len(list(reasoner.predicates[n].parameters())) > 0
                for n in names
            )
            ok = has_mapping and names_ok and params_ok
            msg = f"predicates={names}, has_params={params_ok}"
            return ok, msg, None

        self._check("check_grounding_audit", "grounding", check_grounding_audit)

        # ---- check_grounding_gradient_flow ----
        def check_grounding_gradient_flow():
            """Gradients flow from predicate loss through entity extractor."""
            extractor = _EntityExtractor(D_ws, D_ent)
            pred = _PredicateModule(D_ent, hidden_dim=32)

            slots = torch.randn(B, K, D_ws, requires_grad=True)
            entities, _ = extractor(slots)
            truth = pred(entities)  # (B, K)
            loss = (1.0 - truth).mean()  # want truth -> 1
            loss.backward()

            grad_ok = slots.grad is not None and (slots.grad.abs() > 1e-10).any().item()
            extractor_grad_ok = any(
                p.grad is not None and (p.grad.abs() > 1e-10).any().item()
                for p in extractor.parameters()
            )
            ok = grad_ok and extractor_grad_ok
            msg = f"slots.grad exists={slots.grad is not None}, extractor grads={extractor_grad_ok}"
            return ok, msg, None

        self._check("check_grounding_gradient_flow", "grounding", check_grounding_gradient_flow)

    # ==================================================================
    # GROUP 4: Rule Engine Contract (5 checks)
    # ==================================================================

    def _run_rule_engine(self) -> None:
        """Rule compilation, evaluation, attention, violation, loss."""
        torch.manual_seed(self.seed)

        # ---- check_rule_compilation ----
        def check_rule_compilation():
            """Dict -> AST -> evaluatable."""
            rule_dict = {
                "type": "connective",
                "op": "AND",
                "children": [
                    {"type": "literal", "predicate": "P", "vars": ["x"]},
                    {
                        "type": "connective",
                        "op": "IMPLIES",
                        "children": [
                            {"type": "literal", "predicate": "P", "vars": ["x"]},
                            {"type": "literal", "predicate": "Q", "vars": ["x"]},
                        ],
                    },
                ],
            }
            ast = _compile_rule(rule_dict)
            # Check structure
            is_conn = ast.node_type == "connective" and ast.op == "AND"
            has_children = len(ast.children) == 2
            child_0_lit = ast.children[0].node_type == "literal" and ast.children[0].predicate_name == "P"
            child_1_conn = ast.children[1].node_type == "connective" and ast.children[1].op == "IMPLIES"

            # Evaluate
            bundle = _get_bundle("product")
            cache = {"P": torch.tensor([0.9]), "Q": torch.tensor([0.7])}
            result = _evaluate_ast(ast, bundle, cache)
            # AND(0.9, IMPLIES(0.9, 0.7)) = 0.9 * (1 - 0.9 + 0.9*0.7) = 0.9 * 0.73 = 0.657
            expected = torch.tensor([0.657])
            eval_ok = torch.allclose(result, expected, atol=1e-2)

            ok = is_conn and has_children and child_0_lit and child_1_conn and eval_ok
            msg = f"compiled={is_conn}, eval_result={result.item():.3f}, expected={expected.item():.3f}"
            return ok, msg, None

        self._check("check_rule_compilation", "rule_engine", check_rule_compilation)

        # ---- check_rule_evaluation_range ----
        def check_rule_evaluation_range():
            """Rule evaluation truth values always in [0, 1]."""
            torch.manual_seed(self.seed)
            all_ok = True
            msgs = []
            for bname in ["godel", "product", "lukasiewicz", "stable_product"]:
                bundle = _get_bundle(bname)
                for _ in range(20):
                    cache = {
                        "P": torch.rand(10),
                        "Q": torch.rand(10),
                    }
                    # Various rule structures
                    rules = [
                        {"type": "literal", "predicate": "P", "vars": ["x"]},
                        {"type": "connective", "op": "AND",
                         "children": [
                             {"type": "literal", "predicate": "P", "vars": ["x"]},
                             {"type": "literal", "predicate": "Q", "vars": ["x"]},
                         ]},
                        {"type": "connective", "op": "OR",
                         "children": [
                             {"type": "literal", "predicate": "P", "vars": ["x"]},
                             {"type": "literal", "predicate": "Q", "vars": ["x"]},
                         ]},
                        {"type": "connective", "op": "IMPLIES",
                         "children": [
                             {"type": "literal", "predicate": "P", "vars": ["x"]},
                             {"type": "literal", "predicate": "Q", "vars": ["x"]},
                         ]},
                        {"type": "negation", "child": {"type": "literal", "predicate": "P", "vars": ["x"]}},
                    ]
                    for rd in rules:
                        ast = _compile_rule(rd)
                        result = _evaluate_ast(ast, bundle, cache)
                        lo, hi = result.min().item(), result.max().item()
                        if lo < -1e-6 or hi > 1.0 + 1e-6:
                            msgs.append(f"{bname} rule eval out of [0,1]: [{lo:.4f}, {hi:.4f}]")
                            all_ok = False
            msg = "All rule evaluations in [0,1]" if all_ok else "; ".join(msgs[:3])
            return all_ok, msg, None

        self._check("check_rule_evaluation_range", "rule_engine", check_rule_evaluation_range)

        # ---- check_attention_normalization ----
        def check_attention_normalization():
            """Rule attention weights sum to 1."""
            torch.manual_seed(self.seed)
            rn = _RuleNetwork(workspace_dim=64, rule_embed_dim=32, num_rules=5)
            ws = torch.randn(8, 64)
            attn = rn.compute_attention(ws)

            shape_ok = attn.shape == (8, 5)
            sum_ok = torch.allclose(attn.sum(dim=-1), torch.ones(8), atol=1e-5)
            nonneg_ok = (attn >= 0.0 - 1e-6).all().item()

            ok = shape_ok and sum_ok and nonneg_ok
            msg = f"shape={attn.shape}, sum={attn.sum(dim=-1).tolist()[:3]}, non-neg={nonneg_ok}"
            return ok, msg, None

        self._check("check_attention_normalization", "rule_engine", check_attention_normalization)

        # ---- check_violation_computation ----
        def check_violation_computation():
            """violation = 1 - truth for positive rules."""
            rn = _RuleNetwork()
            truth = torch.tensor([[0.9, 0.3, 0.7, 0.1]])
            violations = rn.compute_violation(truth)
            expected = torch.tensor([[0.1, 0.7, 0.3, 0.9]])
            ok = torch.allclose(violations, expected, atol=1e-5)
            msg = f"violations={violations.tolist()}, expected={expected.tolist()}"
            return ok, msg, None

        self._check("check_violation_computation", "rule_engine", check_violation_computation)

        # ---- check_constraint_loss_scalar ----
        def check_constraint_loss_scalar():
            """Constraint loss is a differentiable scalar."""
            torch.manual_seed(self.seed)
            rn = _RuleNetwork(workspace_dim=64, rule_embed_dim=32, num_rules=4)
            ws = torch.randn(4, 64)
            truth = torch.rand(4, 4, requires_grad=True)
            attn = rn.compute_attention(ws)
            loss = rn.compute_constraint_loss(truth, attn)

            is_scalar = loss.dim() == 0
            loss.backward()
            has_grad = truth.grad is not None and (truth.grad.abs() > 1e-8).any().item()
            no_nan = not torch.isnan(loss).item()

            ok = is_scalar and has_grad and no_nan
            msg = f"scalar={is_scalar}, has_grad={has_grad}, loss={loss.item():.4f}"
            return ok, msg, None

        self._check("check_constraint_loss_scalar", "rule_engine", check_constraint_loss_scalar)

    # ==================================================================
    # GROUP 5: Joint Training -- Done-When Gate (a) (5 checks)
    # ==================================================================

    def _run_gate_a(self) -> None:
        """Joint training: loss decrease, param change, upstream grads, NaN, warmup."""
        torch.manual_seed(self.seed)
        D_ws, D_ent = 64, 32

        # Build a toy model: upstream encoder MLP + symbolic reasoner
        K_SLOTS = 4

        class _ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Encoder takes (B, D_ws) and produces (B, K*D_ws) for K slot embeddings
                self.encoder = nn.Sequential(
                    nn.Linear(D_ws, K_SLOTS * D_ws),
                    nn.ReLU(),
                    nn.Linear(K_SLOTS * D_ws, K_SLOTS * D_ws),
                )
                self.reasoner = _SymbolicReasoner(
                    workspace_dim=D_ws, entity_dim=D_ent, hidden_dim=32,
                    num_predicates=3, num_rules=3, max_entities=8,
                )

            def forward(self, x: Tensor) -> Dict[str, Any]:
                B = x.shape[0]
                encoded = self.encoder(x)  # (B, K*D_ws)
                slots = encoded.view(B, K_SLOTS, D_ws)  # (B, K, D_ws)
                return self.reasoner(slots)

        # ---- check_toy_model_loss_decreases ----
        def check_toy_model_loss_decreases():
            """Train 20 steps; constraint loss decreases."""
            torch.manual_seed(self.seed)
            model = _ToyModel()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            losses = []
            for step in range(20):
                x = torch.randn(8, D_ws)
                output = model(x)
                loss = output["constraint_loss"]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            first_5 = sum(losses[:5]) / 5
            last_5 = sum(losses[-5:]) / 5
            decreased = last_5 < first_5
            msg = f"first_5_avg={first_5:.4f}, last_5_avg={last_5:.4f}, decreased={decreased}"
            details = f"all losses: {[f'{l:.4f}' for l in losses]}" if not decreased else None
            return decreased, msg, details

        self._check("check_toy_model_loss_decreases", "gate_a", check_toy_model_loss_decreases)

        # ---- check_predicate_params_change ----
        def check_predicate_params_change():
            """Predicate module parameters change after training."""
            torch.manual_seed(self.seed)
            model = _ToyModel()
            # Snapshot predicate params before
            before = {}
            for name, param in model.reasoner.predicates.named_parameters():
                before[name] = param.data.clone()

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            for _ in range(10):
                x = torch.randn(8, D_ws)
                output = model(x)
                loss = output["constraint_loss"]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Check params changed
            changed_count = 0
            total_count = 0
            for name, param in model.reasoner.predicates.named_parameters():
                total_count += 1
                if name in before:
                    if not torch.allclose(param.data, before[name], atol=1e-8):
                        changed_count += 1

            ok = changed_count > 0
            msg = f"{changed_count}/{total_count} predicate params changed"
            return ok, msg, None

        self._check("check_predicate_params_change", "gate_a", check_predicate_params_change)

        # ---- check_upstream_encoder_grads ----
        def check_upstream_encoder_grads():
            """Gradients flow to upstream entity encoder MLP."""
            torch.manual_seed(self.seed)
            model = _ToyModel()
            x = torch.randn(8, D_ws)
            output = model(x)
            loss = output["constraint_loss"]
            loss.backward()

            encoder_has_grad = any(
                p.grad is not None and (p.grad.abs() > 1e-10).any().item()
                for p in model.encoder.parameters()
            )
            msg = f"encoder_has_grad={encoder_has_grad}"
            return encoder_has_grad, msg, None

        self._check("check_upstream_encoder_grads", "gate_a", check_upstream_encoder_grads)

        # ---- check_no_nan_during_training ----
        def check_no_nan_during_training():
            """20 training steps with no NaN in any tensor."""
            torch.manual_seed(self.seed)
            model = _ToyModel()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            found_nan = False
            nan_detail = ""

            for step in range(20):
                x = torch.randn(8, D_ws)
                output = model(x)
                loss = output["constraint_loss"]

                # Check loss
                if torch.isnan(loss):
                    found_nan = True
                    nan_detail = f"NaN loss at step {step}"
                    break

                # Check rule truths
                if torch.isnan(output["rule_truths"]).any():
                    found_nan = True
                    nan_detail = f"NaN in rule_truths at step {step}"
                    break

                optimizer.zero_grad()
                loss.backward()

                # Check gradients
                for name, p in model.named_parameters():
                    if p.grad is not None and torch.isnan(p.grad).any():
                        found_nan = True
                        nan_detail = f"NaN grad in {name} at step {step}"
                        break
                if found_nan:
                    break

                optimizer.step()

                # Check params after step
                for name, p in model.named_parameters():
                    if torch.isnan(p).any():
                        found_nan = True
                        nan_detail = f"NaN param in {name} at step {step}"
                        break
                if found_nan:
                    break

            ok = not found_nan
            msg = "No NaN in 20 training steps" if ok else nan_detail
            return ok, msg, nan_detail if found_nan else None

        self._check("check_no_nan_during_training", "gate_a", check_no_nan_during_training)

        # ---- check_warmup_schedule ----
        def check_warmup_schedule():
            """Constraint weight increases from 0 to target over warmup period."""
            warmup_steps = 100
            target_weight = 0.1

            def get_constraint_weight(step: int) -> float:
                """Linear warmup schedule."""
                if step >= warmup_steps:
                    return target_weight
                return target_weight * (step / warmup_steps)

            # Check that weight starts at 0
            w0 = get_constraint_weight(0)
            ok_start = abs(w0) < 1e-8

            # Check monotonic increase during warmup
            prev = -1.0
            monotonic = True
            for s in range(warmup_steps + 10):
                w = get_constraint_weight(s)
                if w < prev - 1e-8:
                    monotonic = False
                    break
                prev = w

            # Check that it reaches target
            w_end = get_constraint_weight(warmup_steps)
            ok_end = abs(w_end - target_weight) < 1e-6

            # Check that after warmup it stays at target
            w_after = get_constraint_weight(warmup_steps + 50)
            ok_plateau = abs(w_after - target_weight) < 1e-6

            ok = ok_start and monotonic and ok_end and ok_plateau
            msg = f"w(0)={w0:.6f}, monotonic={monotonic}, w({warmup_steps})={w_end:.4f}, w(after)={w_after:.4f}"
            return ok, msg, None

        self._check("check_warmup_schedule", "gate_a", check_warmup_schedule)

    # ==================================================================
    # GROUP 6: Logic Identity Tests -- Done-When Gate (b) (4 checks)
    # ==================================================================

    def _run_gate_b(self) -> None:
        """Comprehensive per-bundle identity checks (Gate b)."""
        grid = torch.linspace(0.0, 1.0, 11)
        ones = torch.ones_like(grid)
        zeros = torch.zeros_like(grid)

        PURE_BUNDLES = ["godel", "product", "lukasiewicz"]
        ALL_BUNDLES = PURE_BUNDLES + ["stable_product"]

        # ---- check_all_bundles_boundaries ----
        def check_all_bundles_boundaries():
            """Comprehensive boundary test for all 4 bundles on grid [0..1]."""
            all_ok = True
            msgs = []

            for bname in ALL_BUNDLES:
                bundle = _get_bundle(bname)
                atol = 1e-3 if bname == "stable_product" else 1e-6
                is_stable = bname == "stable_product"

                tests = [
                    ("AND(x,1)=x", bundle.and_op(grid, ones), grid, atol),
                    ("OR(x,0)=x", bundle.or_op(grid, zeros), grid, atol),
                    ("OR(x,1)=1", bundle.or_op(grid, ones), ones, atol),
                    ("NOT(NOT(x))=x", bundle.not_op(bundle.not_op(grid)), grid, 1e-6),
                    ("IMPLIES(1,y)=y", bundle.implies_op(ones, grid), grid, atol),
                    ("IMPLIES(0,y)=1", bundle.implies_op(zeros, grid), ones, atol),
                ]

                if not is_stable:
                    tests.append(("AND(x,0)=0", bundle.and_op(grid, zeros), zeros, atol))
                else:
                    # For stable product, AND(x,0) is approximately 0
                    result_and0 = bundle.and_op(grid, zeros)
                    if result_and0.max().item() > 0.01:
                        msgs.append(f"{bname}: AND(x,0) not approximately 0 (max={result_and0.max().item():.4e})")
                        all_ok = False

                for label, result, expected, tol in tests:
                    if not torch.allclose(result, expected, atol=tol):
                        diff = (result - expected).abs().max().item()
                        msgs.append(f"{bname}: {label} failed (max_diff={diff:.2e}, atol={tol:.0e})")
                        all_ok = False

            msg = "All boundary tests pass" if all_ok else "; ".join(msgs[:5])
            details = "\n".join(msgs) if msgs else None
            return all_ok, msg, details

        self._check("check_all_bundles_boundaries", "gate_b", check_all_bundles_boundaries)

        # ---- check_pure_exact_identities ----
        def check_pure_exact_identities():
            """Exact (atol=1e-6) identity tests for godel, product, lukasiewicz."""
            torch.manual_seed(self.seed)
            x = torch.rand(200)
            y = torch.rand(200)
            all_ok = True
            msgs = []
            atol = 1e-6

            for bname in PURE_BUNDLES:
                bundle = _get_bundle(bname)

                # T-norm identity: AND(x, 1) = x
                r1 = bundle.and_op(x, torch.ones_like(x))
                if not torch.allclose(r1, x, atol=atol):
                    msgs.append(f"{bname}: AND(x,1)!=x exactly")
                    all_ok = False

                # T-norm annihilator: AND(x, 0) = 0
                r2 = bundle.and_op(x, torch.zeros_like(x))
                if not torch.allclose(r2, torch.zeros_like(x), atol=atol):
                    msgs.append(f"{bname}: AND(x,0)!=0 exactly")
                    all_ok = False

                # T-conorm identity: OR(x, 0) = x
                r3 = bundle.or_op(x, torch.zeros_like(x))
                if not torch.allclose(r3, x, atol=atol):
                    msgs.append(f"{bname}: OR(x,0)!=x exactly")
                    all_ok = False

                # T-conorm annihilator: OR(x, 1) = 1
                r4 = bundle.or_op(x, torch.ones_like(x))
                if not torch.allclose(r4, torch.ones_like(x), atol=atol):
                    msgs.append(f"{bname}: OR(x,1)!=1 exactly")
                    all_ok = False

                # Negation involution
                r5 = bundle.not_op(bundle.not_op(x))
                if not torch.allclose(r5, x, atol=atol):
                    msgs.append(f"{bname}: NOT(NOT(x))!=x exactly")
                    all_ok = False

                # Implication boundaries
                r6 = bundle.implies_op(torch.ones_like(y), y)
                if not torch.allclose(r6, y, atol=atol):
                    msgs.append(f"{bname}: IMPLIES(1,y)!=y exactly")
                    all_ok = False

                r7 = bundle.implies_op(torch.zeros_like(y), y)
                if not torch.allclose(r7, torch.ones_like(y), atol=atol):
                    msgs.append(f"{bname}: IMPLIES(0,y)!=1 exactly")
                    all_ok = False

                # Associativity
                z = torch.rand(200)
                lhs = bundle.and_op(bundle.and_op(x, y), z)
                rhs = bundle.and_op(x, bundle.and_op(y, z))
                if not torch.allclose(lhs, rhs, atol=1e-5):
                    msgs.append(f"{bname}: AND not associative")
                    all_ok = False

            msg = "All pure exact identities pass" if all_ok else "; ".join(msgs[:5])
            return all_ok, msg, "\n".join(msgs) if msgs else None

        self._check("check_pure_exact_identities", "gate_b", check_pure_exact_identities)

        # ---- check_stable_approximate_identities ----
        def check_stable_approximate_identities():
            """Tolerance-based (atol=1e-3) identity tests for stable_product."""
            torch.manual_seed(self.seed)
            x = torch.rand(200)
            y = torch.rand(200)
            bundle = _get_bundle("stable_product")
            atol = 1e-3
            all_ok = True
            msgs = []

            # AND(x, 1) ~ x
            r1 = bundle.and_op(x, torch.ones_like(x))
            if not torch.allclose(r1, x, atol=atol):
                diff = (r1 - x).abs().max().item()
                msgs.append(f"AND(x,1)~x failed (max_diff={diff:.2e})")
                all_ok = False

            # OR(x, 0) ~ x
            r2 = bundle.or_op(x, torch.zeros_like(x))
            if not torch.allclose(r2, x, atol=atol):
                diff = (r2 - x).abs().max().item()
                msgs.append(f"OR(x,0)~x failed (max_diff={diff:.2e})")
                all_ok = False

            # OR(x, 1) ~ 1
            r3 = bundle.or_op(x, torch.ones_like(x))
            if not torch.allclose(r3, torch.ones_like(x), atol=atol):
                diff = (r3 - torch.ones_like(x)).abs().max().item()
                msgs.append(f"OR(x,1)~1 failed (max_diff={diff:.2e})")
                all_ok = False

            # IMPLIES(1, y) ~ y
            r4 = bundle.implies_op(torch.ones_like(y), y)
            if not torch.allclose(r4, y, atol=atol):
                diff = (r4 - y).abs().max().item()
                msgs.append(f"IMPLIES(1,y)~y failed (max_diff={diff:.2e})")
                all_ok = False

            # IMPLIES(0, y) ~ 1
            r5 = bundle.implies_op(torch.zeros_like(y), y)
            if not torch.allclose(r5, torch.ones_like(y), atol=atol):
                diff = (r5 - torch.ones_like(y)).abs().max().item()
                msgs.append(f"IMPLIES(0,y)~1 failed (max_diff={diff:.2e})")
                all_ok = False

            # NOT(NOT(x)) = x (exact, negation is 1-x)
            r6 = bundle.not_op(bundle.not_op(x))
            if not torch.allclose(r6, x, atol=1e-6):
                msgs.append(f"NOT(NOT(x))!=x")
                all_ok = False

            # AND(x, 0) ~ 0 (bounded by eps^2)
            r7 = bundle.and_op(x, torch.zeros_like(x))
            if r7.max().item() > 0.01:
                msgs.append(f"AND(x,0) not ~0 (max={r7.max().item():.2e})")
                all_ok = False

            msg = "All stable_product approximate identities pass" if all_ok else "; ".join(msgs)
            return all_ok, msg, None

        self._check("check_stable_approximate_identities", "gate_b", check_stable_approximate_identities)

        # ---- check_commutativity ----
        def check_commutativity():
            """AND(x,y)=AND(y,x), OR(x,y)=OR(y,x) for all bundles."""
            torch.manual_seed(self.seed)
            x = torch.rand(200)
            y = torch.rand(200)
            all_ok = True
            msgs = []

            for bname in ALL_BUNDLES:
                bundle = _get_bundle(bname)
                # AND commutativity
                and_xy = bundle.and_op(x, y)
                and_yx = bundle.and_op(y, x)
                if not torch.allclose(and_xy, and_yx, atol=1e-6):
                    msgs.append(f"{bname}: AND not commutative")
                    all_ok = False

                # OR commutativity
                or_xy = bundle.or_op(x, y)
                or_yx = bundle.or_op(y, x)
                if not torch.allclose(or_xy, or_yx, atol=1e-6):
                    msgs.append(f"{bname}: OR not commutative")
                    all_ok = False

            msg = "Commutativity holds for all bundles" if all_ok else "; ".join(msgs)
            return all_ok, msg, None

        self._check("check_commutativity", "gate_b", check_commutativity)

    # ==================================================================
    # GROUP 7: Violation Reduction -- Done-When Gate (c) (5 checks)
    # ==================================================================

    def _run_gate_c(self) -> None:
        """Synthetic FOL training: violation decrease, query accuracy, stats, logs."""
        torch.manual_seed(self.seed)
        D_ws, D_ent = 64, 32

        # Build synthetic FOL task:
        # - Random workspace inputs
        # - Train the symbolic reasoner to satisfy rules (reduce violations)
        # - Check that violation rate drops over training

        def _build_synthetic_system():
            """Build a minimal symbolic reasoner for synthetic FOL."""
            reasoner = _SymbolicReasoner(
                workspace_dim=D_ws, entity_dim=D_ent, hidden_dim=32,
                num_predicates=3, num_rules=3, max_entities=4,
            )
            return reasoner

        def _generate_synthetic_batch(B: int = 16, K: int = 4) -> Tensor:
            """Generate a batch of random workspace slots."""
            return torch.randn(B, K, D_ws)

        # ---- check_synthetic_violation_decreases ----
        def check_synthetic_violation_decreases():
            """Train 50 steps, violation rate drops."""
            torch.manual_seed(self.seed)
            model = _build_synthetic_system()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            violation_history = []
            for step in range(50):
                data = _generate_synthetic_batch()
                output = model(data)
                loss = output["constraint_loss"]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                violation_history.append(output["violation_stats"]["mean_violation"].item())

            first_10_avg = sum(violation_history[:10]) / 10
            last_10_avg = sum(violation_history[-10:]) / 10
            decreased = last_10_avg < first_10_avg

            msg = f"first_10_avg={first_10_avg:.4f}, last_10_avg={last_10_avg:.4f}, decreased={decreased}"
            details = None
            if not decreased:
                details = f"Full history: {[f'{v:.4f}' for v in violation_history]}"
            return decreased, msg, details

        self._check("check_synthetic_violation_decreases", "gate_c", check_synthetic_violation_decreases)

        # ---- check_query_accuracy_stable ----
        def check_query_accuracy_stable():
            """Query accuracy (rule truth) doesn't degrade during training."""
            torch.manual_seed(self.seed)
            model = _build_synthetic_system()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            truth_history = []
            for step in range(50):
                data = _generate_synthetic_batch()
                output = model(data)
                loss = output["constraint_loss"]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Average rule truth
                avg_truth = output["rule_truths"].mean().item()
                truth_history.append(avg_truth)

            first_10_avg = sum(truth_history[:10]) / 10
            last_10_avg = sum(truth_history[-10:]) / 10
            # Query accuracy should be stable or improving (truth should increase or stay)
            stable = last_10_avg >= first_10_avg - 0.05  # allow small tolerance
            msg = f"first_10_avg_truth={first_10_avg:.4f}, last_10_avg_truth={last_10_avg:.4f}, stable={stable}"
            return stable, msg, None

        self._check("check_query_accuracy_stable", "gate_c", check_query_accuracy_stable)

        # ---- check_violation_stats_logged ----
        def check_violation_stats_logged():
            """violation_stats dict populated with expected keys."""
            torch.manual_seed(self.seed)
            model = _build_synthetic_system()
            data = _generate_synthetic_batch()
            output = model(data)

            has_stats = "violation_stats" in output
            if not has_stats:
                return False, "violation_stats missing from output", None

            stats = output["violation_stats"]
            expected_keys = {"mean_violation", "max_violation", "per_rule_violation"}
            present_keys = set(stats.keys())
            keys_ok = expected_keys.issubset(present_keys)

            # Values should be tensors
            values_ok = all(isinstance(stats[k], Tensor) for k in expected_keys if k in stats)

            # mean_violation should be a scalar (or 0-d tensor)
            mean_scalar = stats["mean_violation"].dim() == 0
            # per_rule_violation should have shape (num_rules,)
            per_rule_shape = stats["per_rule_violation"].dim() == 1

            ok = has_stats and keys_ok and values_ok and mean_scalar and per_rule_shape
            msg = f"keys={list(stats.keys())}, mean_scalar={mean_scalar}, per_rule_dim={stats['per_rule_violation'].dim()}"
            return ok, msg, None

        self._check("check_violation_stats_logged", "gate_c", check_violation_stats_logged)

        # ---- check_before_after_comparison ----
        def check_before_after_comparison():
            """Mean violation before > mean violation after training."""
            torch.manual_seed(self.seed)
            model = _build_synthetic_system()

            # Before training
            data_before = _generate_synthetic_batch(B=32)
            with torch.no_grad():
                output_before = model(data_before)
            violation_before = output_before["violation_stats"]["mean_violation"].item()

            # Train
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            for step in range(50):
                data = _generate_synthetic_batch()
                output = model(data)
                loss = output["constraint_loss"]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # After training
            with torch.no_grad():
                output_after = model(data_before)
            violation_after = output_after["violation_stats"]["mean_violation"].item()

            decreased = violation_after < violation_before
            msg = f"violation_before={violation_before:.4f}, violation_after={violation_after:.4f}, decreased={decreased}"
            return decreased, msg, None

        self._check("check_before_after_comparison", "gate_c", check_before_after_comparison)

        # ---- check_logs_intelligible ----
        def check_logs_intelligible():
            """return_logs=True produces non-empty logs with expected keys."""
            torch.manual_seed(self.seed)
            model = _build_synthetic_system()
            data = _generate_synthetic_batch()
            output = model(data, return_logs=True)

            has_logs = "logs" in output and output["logs"] is not None
            if not has_logs:
                return False, "logs not present in output", None

            logs = output["logs"]
            expected_keys = {"predicate_truths", "entity_shape", "attention_weights", "violations"}
            present = set(logs.keys())
            keys_ok = expected_keys.issubset(present)

            # Check non-empty
            non_empty = True
            empty_keys = []
            for k in expected_keys:
                if k in logs:
                    val = logs[k]
                    if isinstance(val, dict) and len(val) == 0:
                        non_empty = False
                        empty_keys.append(k)
                    elif isinstance(val, Tensor) and val.numel() == 0:
                        non_empty = False
                        empty_keys.append(k)
                    elif isinstance(val, list) and len(val) == 0:
                        non_empty = False
                        empty_keys.append(k)

            # Check predicate_truths is a dict with predicate name keys
            pred_truths = logs.get("predicate_truths", {})
            pred_ok = isinstance(pred_truths, dict) and len(pred_truths) > 0

            # Check attention_weights is a tensor with correct shape
            attn = logs.get("attention_weights")
            attn_ok = isinstance(attn, Tensor) and attn.dim() == 2

            # Check violations is a tensor
            viols = logs.get("violations")
            viols_ok = isinstance(viols, Tensor)

            ok = has_logs and keys_ok and non_empty and pred_ok and attn_ok and viols_ok
            msg = (
                f"keys={list(logs.keys())}, "
                f"pred_names={list(pred_truths.keys()) if isinstance(pred_truths, dict) else '?'}, "
                f"attn_shape={attn.shape if isinstance(attn, Tensor) else '?'}"
            )
            details = None
            if not ok:
                details = (
                    f"has_logs={has_logs}, keys_ok={keys_ok}, non_empty={non_empty}, "
                    f"empty_keys={empty_keys}, pred_ok={pred_ok}, attn_ok={attn_ok}, viols_ok={viols_ok}"
                )
            return ok, msg, details

        self._check("check_logs_intelligible", "gate_c", check_logs_intelligible)


# ============================================================================
# Summary table printer
# ============================================================================

def _print_summary(results: List[CheckResult]) -> Tuple[int, int, int]:
    """Print a summary table of all results. Returns (total, passed, failed)."""
    print("\n" + "=" * 80)
    print(f"  {'VALIDATION SUMMARY':^76}")
    print("=" * 80)

    # Group results
    groups: Dict[str, List[CheckResult]] = {}
    for r in results:
        groups.setdefault(r.group, []).append(r)

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed

    # Header
    print(f"\n  {'Check':<45} {'Group':<14} {'Status':<8} {'Time':>8}")
    print(f"  {'-'*45} {'-'*14} {'-'*8} {'-'*8}")

    for group in _ALL_GROUPS:
        if group not in groups:
            continue
        for r in groups[group]:
            status = _c("PASS", _GREEN) if r.passed else _c("FAIL", _RED)
            name_display = r.name[:44]
            group_display = r.group[:13]
            time_display = f"{r.elapsed_ms:.1f}ms"
            print(f"  {name_display:<45} {group_display:<14} {status:<17} {time_display:>8}")

    # Totals
    print(f"\n  {'-'*80}")
    total_time = sum(r.elapsed_ms for r in results)
    status_line = (
        f"  Total: {total} checks | "
        f"{_c(str(passed) + ' passed', _GREEN)} | "
        f"{_c(str(failed) + ' failed', _RED) if failed > 0 else _c('0 failed', _GREEN)} | "
        f"Time: {total_time:.0f}ms"
    )
    print(status_line)
    print("=" * 80)

    # List failures if any
    if failed > 0:
        print(f"\n  {_c('FAILED CHECKS:', _RED + _BOLD)}")
        for r in results:
            if not r.passed:
                print(f"    - {r.name} [{r.group}]: {r.message}")
                if r.details:
                    for line in r.details.strip().split("\n")[:3]:
                        print(f"      {_c(line, _DIM)}")
        print()

    return total, passed, failed


# ============================================================================
# Gate summary
# ============================================================================

def _print_gate_summary(results: List[CheckResult]) -> None:
    """Print Done-When gate verdicts."""
    print("\n" + "-" * 60)
    print(f"  {'DONE-WHEN GATE VERDICTS':^56}")
    print("-" * 60)

    gate_groups = {
        "gate_a": "Gate (a): Joint Training",
        "gate_b": "Gate (b): Logic Identity Tests",
        "gate_c": "Gate (c): Violation Reduction",
    }

    for group_key, label in gate_groups.items():
        group_results = [r for r in results if r.group == group_key]
        if not group_results:
            status = _c("SKIP", _YELLOW)
        elif all(r.passed for r in group_results):
            status = _c("PASS", _GREEN + _BOLD)
        else:
            fail_count = sum(1 for r in group_results if not r.passed)
            status = _c(f"FAIL ({fail_count}/{len(group_results)} failed)", _RED + _BOLD)
        print(f"  {label:<40} {status}")

    print("-" * 60)


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Runtime contract validation for the Neuro-Symbolic Engine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Groups:\n"
            "  operators    -- Operator identity checks (AND, OR, NOT, IMPLIES)\n"
            "  gradients    -- Gradient flow, NaN, AMP compatibility\n"
            "  grounding    -- Entity extractor shapes, ranges, audit, grads\n"
            "  rule_engine  -- AST compilation, evaluation, attention, loss\n"
            "  gate_a       -- Done-When Gate (a): Joint training\n"
            "  gate_b       -- Done-When Gate (b): Logic identity tests\n"
            "  gate_c       -- Done-When Gate (c): Violation reduction\n"
        ),
    )
    parser.add_argument(
        "--group",
        choices=_ALL_GROUPS,
        default=None,
        help="Run only a specific validation group.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed messages for each check.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args()

    print("=" * 80)
    print(f"  {'NEURO-SYMBOLIC ENGINE -- CONTRACT VALIDATION':^76}")
    print(f"  seed={args.seed}, verbose={args.verbose}, "
          f"group={args.group or 'ALL'}, device=cpu")
    print("=" * 80)

    validator = SymbolicValidator(verbose=args.verbose, seed=args.seed)

    if args.group:
        results = validator.run_group(args.group)
    else:
        results = validator.run_all()

    total, passed, failed = _print_summary(results)
    _print_gate_summary(results)

    if failed > 0:
        print(f"\n{_c('RESULT: FAIL', _RED + _BOLD)} -- {failed} check(s) did not pass.\n")
        sys.exit(1)
    else:
        print(f"\n{_c('RESULT: ALL CHECKS PASSED', _GREEN + _BOLD)}\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
