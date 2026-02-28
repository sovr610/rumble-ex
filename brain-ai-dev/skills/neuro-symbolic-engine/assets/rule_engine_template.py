"""
Rule Engine Template for Neuro-Symbolic Reasoning
==================================================

Implements:
  - Rule AST representation (LiteralNode, ConnectiveNode, QuantifierNode, NegationNode)
  - Rule compilation from dict representation
  - AST-based recursive rule evaluation with fuzzy logic operators
  - Attention-weighted rule application (RuleNetwork)
  - Violation metrics and aggregation (mean, soft_min, p_mean)
  - Interpretable logging (top-K violated rules, attention weights, per-rule truths)
  - Theory class for managing rule collections
  - Warmup schedule for constraint weight ramping

Design notes:
  - All fuzzy truth values are in [0, 1] and support gradient flow.
  - RuleEvaluator delegates logical connectives to a pluggable OperatorBundle
    (product, godel, lukasiewicz, stable_product) so the same AST works with
    any fuzzy logic family.
  - RuleNetwork computes L_symbolic = sum(alpha_r * violation(r)) where alpha_r
    is attention-weighted per-rule importance.  Total loss is
    L_total = L_task + lambda(step) * L_symbolic  with warmup schedule.
  - All tensor outputs are detached in logs to prevent accidental gradient leaks.

Usage:
  This file is a *template* -- copy to brain_ai/reasoning/rule_engine.py and
  integrate with the OperatorBundle from fuzzy_operators.py and the grounding
  modules from grounding.py.

Dependencies:
  - torch (PyTorch)
  - dataclasses (stdlib)
  - typing (stdlib)

No other external packages required.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# 1. RuleConfig
# =============================================================================


@dataclass
class RuleConfig:
    """Configuration for the rule engine and RuleNetwork.

    Attributes:
        max_rules: Maximum number of rules the RuleNetwork can hold embeddings for.
        use_attention: Whether to use attention-weighted rule application.
        attention_dim: Dimension of workspace summary / query embedding vectors.
        rule_embed_dim: Dimension of learned per-rule embedding vectors.
        violation_agg: Aggregation mode when attention is disabled.
            One of ``"mean"``, ``"soft_min"``, ``"p_mean"``.
        p_mean_p: Exponent for p-mean aggregation (only used when
            ``violation_agg == "p_mean"``).
        constraint_weight: Lambda multiplier -- ``L_total = L_task + lambda * L_symbolic``.
        warmup_steps: Number of training steps to linearly ramp constraint weight
            from 0 to ``constraint_weight``.
        top_k_logs: How many top-violated rules to include in interpretable logs.
        full_tensor_logs: If True, include full-size tensors in logs (opt-in for
            debugging; can be expensive for large batch/rule counts).
    """

    max_rules: int = 64
    use_attention: bool = True
    attention_dim: int = 256
    rule_embed_dim: int = 128
    violation_agg: str = "mean"  # "mean", "soft_min", "p_mean"
    p_mean_p: float = 2.0
    constraint_weight: float = 0.1
    warmup_steps: int = 1000
    top_k_logs: int = 5
    full_tensor_logs: bool = False


# =============================================================================
# 2. AST Node Classes
# =============================================================================


class ASTNode:
    """Base class for all rule AST nodes.

    Every concrete node type represents one syntactic element of a first-order
    fuzzy logic formula.  The tree is evaluated recursively by
    :class:`RuleEvaluator`.
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class LiteralNode(ASTNode):
    """Leaf node representing a predicate or relation application.

    Examples:
        - Unary predicate: ``is_bird(x)``  ->  ``LiteralNode("is_bird", ["x"])``
        - Binary relation: ``parent(x, y)`` ->  ``LiteralNode("parent", ["x", "y"])``
        - Negated literal: ``NOT is_bird(x)`` -> ``LiteralNode("is_bird", ["x"], negated=True)``

    Attributes:
        predicate_name: Name key used to look up truth values in the predicate /
            relation truth dictionaries.
        variables: Variable names bound to the predicate arguments.  Length 1
            for unary predicates, length 2 for binary relations.
        negated: If ``True``, the truth value is complemented (``1 - t``).
    """

    def __init__(
        self,
        predicate_name: str,
        variables: List[str],
        negated: bool = False,
    ) -> None:
        super().__init__()
        self.predicate_name = predicate_name
        self.variables = list(variables)
        self.negated = negated

    def __repr__(self) -> str:
        neg = "NOT " if self.negated else ""
        return f"LiteralNode({neg}{self.predicate_name}({', '.join(self.variables)}))"


class ConnectiveNode(ASTNode):
    """Internal node representing a binary or n-ary logical connective.

    Supported connectives: ``"and"``, ``"or"``, ``"implies"``, ``"iff"``.

    For ``"implies"`` exactly two children are expected (antecedent, consequent).
    For ``"iff"`` exactly two children are expected.
    ``"and"`` and ``"or"`` accept two or more children (evaluated left-to-right).

    Attributes:
        connective: One of ``"and"``, ``"or"``, ``"implies"``, ``"iff"``.
        children: Ordered list of child AST nodes.
    """

    VALID_CONNECTIVES = {"and", "or", "implies", "iff"}

    def __init__(self, connective: str, children: List[ASTNode]) -> None:
        super().__init__()
        if connective not in self.VALID_CONNECTIVES:
            raise ValueError(
                f"Unknown connective '{connective}'. "
                f"Must be one of {self.VALID_CONNECTIVES}."
            )
        if connective in ("implies", "iff") and len(children) != 2:
            raise ValueError(
                f"'{connective}' requires exactly 2 children, got {len(children)}."
            )
        if len(children) < 2:
            raise ValueError(
                f"Connective '{connective}' requires at least 2 children."
            )
        self.connective = connective
        self.children = list(children)

    def __repr__(self) -> str:
        return (
            f"ConnectiveNode({self.connective}, "
            f"[{', '.join(repr(c) for c in self.children)}])"
        )


class QuantifierNode(ASTNode):
    """Quantifier node scoping a single variable over a sub-formula body.

    Supported quantifiers: ``"forall"``, ``"exists"``.

    At evaluation time the body is evaluated for each entity binding of the
    scoped variable, then aggregated with the appropriate fuzzy quantifier.

    Attributes:
        quantifier: ``"forall"`` or ``"exists"``.
        variable: Name of the variable being quantified.
        body: The sub-formula (AST subtree) over which the quantifier ranges.
    """

    VALID_QUANTIFIERS = {"forall", "exists"}

    def __init__(self, quantifier: str, variable: str, body: ASTNode) -> None:
        super().__init__()
        if quantifier not in self.VALID_QUANTIFIERS:
            raise ValueError(
                f"Unknown quantifier '{quantifier}'. "
                f"Must be one of {self.VALID_QUANTIFIERS}."
            )
        self.quantifier = quantifier
        self.variable = variable
        self.body = body

    def __repr__(self) -> str:
        return (
            f"QuantifierNode({self.quantifier} {self.variable}: {self.body!r})"
        )


class NegationNode(ASTNode):
    """Negation node applying logical NOT to a sub-formula.

    Attributes:
        child: The sub-formula being negated.
    """

    def __init__(self, child: ASTNode) -> None:
        super().__init__()
        self.child = child

    def __repr__(self) -> str:
        return f"NegationNode({self.child!r})"


# =============================================================================
# 3. Rule (wrapper with metadata)
# =============================================================================


@dataclass
class Rule:
    """A named, weighted symbolic rule wrapping an AST.

    Attributes:
        ast: Root node of the rule's AST.
        name: Human-readable rule name (for logging / identification).
        weight: Soft constraint weight.  Can be fixed or made a learnable
            ``nn.Parameter`` externally.
        intended_true: If ``True``, violation is ``1 - truth(rule)``.
            If ``False`` (negative constraint), violation is ``truth(rule)``.
        metadata: Optional arbitrary metadata (e.g. source dataset, author).
    """

    ast: ASTNode
    name: str = ""
    weight: float = 1.0
    intended_true: bool = True
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"Rule(name={self.name!r}, weight={self.weight}, "
            f"intended_true={self.intended_true}, ast={self.ast!r})"
        )


# =============================================================================
# 4. Rule Compilation (dict -> AST)
# =============================================================================


def _compile_node(d: Dict) -> ASTNode:
    """Recursively compile a dict subtree into an AST node.

    The dict schema mirrors the AST class hierarchy:

    Literal::

        {"type": "literal", "predicate": "is_bird", "vars": ["x"],
         "negated": false}

    Connective::

        {"type": "and"|"or"|"implies"|"iff",
         "children": [<node>, <node>, ...]}

        For ``"implies"`` a shorthand with ``"lhs"``/``"rhs"`` keys is also
        accepted instead of ``"children"``.

    Quantifier::

        {"type": "forall"|"exists", "var": "x", "body": <node>}

    Negation::

        {"type": "not", "child": <node>}
    """
    node_type = d.get("type", "")

    # -- Literal --
    if node_type == "literal":
        return LiteralNode(
            predicate_name=d["predicate"],
            variables=d.get("vars", d.get("variables", [])),
            negated=d.get("negated", False),
        )

    # -- Connective --
    if node_type in ConnectiveNode.VALID_CONNECTIVES:
        if "children" in d:
            children = [_compile_node(c) for c in d["children"]]
        elif node_type == "implies" and "lhs" in d and "rhs" in d:
            children = [_compile_node(d["lhs"]), _compile_node(d["rhs"])]
        elif node_type == "iff" and "lhs" in d and "rhs" in d:
            children = [_compile_node(d["lhs"]), _compile_node(d["rhs"])]
        elif node_type in ("and", "or") and "lhs" in d and "rhs" in d:
            children = [_compile_node(d["lhs"]), _compile_node(d["rhs"])]
        else:
            raise ValueError(
                f"Connective '{node_type}' requires 'children' list or "
                f"'lhs'/'rhs' keys."
            )
        return ConnectiveNode(connective=node_type, children=children)

    # -- Quantifier --
    if node_type in QuantifierNode.VALID_QUANTIFIERS:
        return QuantifierNode(
            quantifier=node_type,
            variable=d["var"],
            body=_compile_node(d["body"]),
        )

    # -- Negation --
    if node_type == "not":
        return NegationNode(child=_compile_node(d["child"]))

    raise ValueError(f"Unknown node type: {node_type!r}")


def compile_rule_from_dict(rule_dict: Dict) -> Rule:
    """Compile a single rule from a dict representation to a :class:`Rule`.

    Top-level keys:
        - ``name`` (str, optional): Rule name.
        - ``weight`` (float, optional): Constraint weight, default 1.0.
        - ``intended_true`` (bool, optional): Default ``True``.
        - ``metadata`` (dict, optional): Arbitrary metadata.
        - Plus all keys required by the root AST node (``type``, etc.).

    Example::

        {
            "name": "birds_fly",
            "type": "forall", "var": "x",
            "body": {
                "type": "implies",
                "lhs": {"type": "literal", "predicate": "is_bird", "vars": ["x"]},
                "rhs": {"type": "literal", "predicate": "can_fly", "vars": ["x"]}
            }
        }
    """
    name = rule_dict.get("name", "")
    weight = rule_dict.get("weight", 1.0)
    intended_true = rule_dict.get("intended_true", True)
    metadata = rule_dict.get("metadata", {})
    ast = _compile_node(rule_dict)
    return Rule(
        ast=ast,
        name=name,
        weight=weight,
        intended_true=intended_true,
        metadata=metadata,
    )


def compile_rules_from_list(rule_dicts: List[Dict]) -> List[Rule]:
    """Compile a list of rule dicts into a list of :class:`Rule` objects."""
    return [compile_rule_from_dict(rd) for rd in rule_dicts]


# =============================================================================
# 5. OperatorBundle Protocol
# =============================================================================


class OperatorBundle:
    """Minimal fuzzy-logic operator bundle interface.

    An OperatorBundle provides the basic connectives and quantifiers needed by
    :class:`RuleEvaluator`.  This base class implements the *product* t-norm
    family.  Subclass or replace with the full ``OperatorBundle`` from
    ``fuzzy_operators.py`` for other families (godel, lukasiewicz,
    stable_product).
    """

    # -- Connectives --

    def fuzzy_and(self, a: Tensor, b: Tensor) -> Tensor:
        """Product t-norm: a * b."""
        return a * b

    def fuzzy_or(self, a: Tensor, b: Tensor) -> Tensor:
        """Probabilistic sum: a + b - a*b."""
        return a + b - a * b

    def fuzzy_not(self, a: Tensor) -> Tensor:
        """Standard negation: 1 - a."""
        return 1.0 - a

    def fuzzy_implies(self, a: Tensor, b: Tensor) -> Tensor:
        """Reichenbach implication: 1 - a + a*b."""
        return 1.0 - a + a * b

    def fuzzy_iff(self, a: Tensor, b: Tensor) -> Tensor:
        """Equivalence as conjunction of both implications."""
        return self.fuzzy_and(
            self.fuzzy_implies(a, b),
            self.fuzzy_implies(b, a),
        )

    # -- Quantifiers --

    def fuzzy_forall(self, x: Tensor, dim: int = -1) -> Tensor:
        """Product universal quantifier: product along *dim*."""
        return x.prod(dim=dim)

    def fuzzy_exists(self, x: Tensor, dim: int = -1) -> Tensor:
        """Product existential quantifier: 1 - prod(1 - x) along *dim*."""
        return 1.0 - (1.0 - x).prod(dim=dim)


class StableProductBundle(OperatorBundle):
    """Product bundle with epsilon projections for training stability.

    Truth values are clamped to ``[eps, 1 - eps]`` before operations that
    can produce vanishing gradients near boundaries.
    """

    def __init__(self, eps: float = 1e-4) -> None:
        self.eps = eps

    def _stabilise(self, x: Tensor) -> Tensor:
        return x.clamp(min=self.eps, max=1.0 - self.eps)

    def fuzzy_and(self, a: Tensor, b: Tensor) -> Tensor:
        return (self._stabilise(a) * self._stabilise(b)).clamp(0.0, 1.0)

    def fuzzy_or(self, a: Tensor, b: Tensor) -> Tensor:
        a_s, b_s = self._stabilise(a), self._stabilise(b)
        return (a_s + b_s - a_s * b_s).clamp(0.0, 1.0)

    def fuzzy_implies(self, a: Tensor, b: Tensor) -> Tensor:
        a_s = self._stabilise(a)
        return (1.0 - a_s + a_s * b).clamp(0.0, 1.0)

    def fuzzy_forall(self, x: Tensor, dim: int = -1) -> Tensor:
        return self._stabilise(x).prod(dim=dim).clamp(0.0, 1.0)

    def fuzzy_exists(self, x: Tensor, dim: int = -1) -> Tensor:
        return (1.0 - (1.0 - self._stabilise(x)).prod(dim=dim)).clamp(0.0, 1.0)


class GodelBundle(OperatorBundle):
    """Godel (min/max) t-norm bundle."""

    def fuzzy_and(self, a: Tensor, b: Tensor) -> Tensor:
        return torch.min(a, b)

    def fuzzy_or(self, a: Tensor, b: Tensor) -> Tensor:
        return torch.max(a, b)

    def fuzzy_implies(self, a: Tensor, b: Tensor) -> Tensor:
        return torch.where(a <= b, torch.ones_like(a), b)

    def fuzzy_forall(self, x: Tensor, dim: int = -1) -> Tensor:
        return x.min(dim=dim)[0]

    def fuzzy_exists(self, x: Tensor, dim: int = -1) -> Tensor:
        return x.max(dim=dim)[0]


class LukasiewiczBundle(OperatorBundle):
    """Lukasiewicz t-norm bundle."""

    def fuzzy_and(self, a: Tensor, b: Tensor) -> Tensor:
        return (a + b - 1.0).clamp(min=0.0)

    def fuzzy_or(self, a: Tensor, b: Tensor) -> Tensor:
        return (a + b).clamp(max=1.0)

    def fuzzy_implies(self, a: Tensor, b: Tensor) -> Tensor:
        return (1.0 - a + b).clamp(max=1.0)

    def fuzzy_forall(self, x: Tensor, dim: int = -1) -> Tensor:
        return (x.sum(dim=dim) - x.shape[dim] + 1).clamp(min=0.0)

    def fuzzy_exists(self, x: Tensor, dim: int = -1) -> Tensor:
        return x.sum(dim=dim).clamp(max=1.0)


def get_operator_bundle(name: str = "product", **kwargs: Any) -> OperatorBundle:
    """Factory for operator bundles by name.

    Args:
        name: One of ``"product"``, ``"stable_product"``, ``"godel"``,
            ``"lukasiewicz"``.
        **kwargs: Forwarded to the bundle constructor (e.g. ``eps`` for
            stable_product).

    Returns:
        An :class:`OperatorBundle` instance.
    """
    bundles: Dict[str, type] = {
        "product": OperatorBundle,
        "stable_product": StableProductBundle,
        "godel": GodelBundle,
        "lukasiewicz": LukasiewiczBundle,
    }
    cls = bundles.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown operator bundle '{name}'. Choose from {list(bundles.keys())}."
        )
    # Only pass kwargs for classes that accept them
    if cls is OperatorBundle:
        return cls()
    return cls(**kwargs)


# =============================================================================
# 6. Rule Evaluator
# =============================================================================


class RuleEvaluator:
    """Evaluate rule ASTs given entity embeddings, predicate/relation truth
    tensors, and a fuzzy-logic operator bundle.

    The evaluator recursively walks the AST and computes a batched truth value
    ``(B,)`` for each rule.

    Predicate truths
        ``predicate_truths`` is ``Dict[str, Tensor]`` mapping predicate names
        to tensors of shape ``(B, N)`` where ``N`` is the number of entities.
        Entry ``[b, i]`` is the truth of ``P(entity_i)`` in batch element ``b``.

    Relation truths
        ``relation_truths`` is ``Dict[str, Tensor]`` mapping relation names
        to tensors of shape ``(B, N, N)``.  Entry ``[b, i, j]`` is the truth
        of ``R(entity_i, entity_j)`` in batch element ``b``.

    Entity mask
        Optional ``(B, N)`` mask where 1 means the entity exists and 0 means
        padding.  Used to mask out padded entities in quantifier aggregation.

    Variable bindings
        During recursive evaluation, quantifiers bind variables to entity
        indices.  Bindings are tracked in a ``Dict[str, int]`` mapping variable
        names to entity indices.  When a :class:`LiteralNode` is reached with
        all variables bound, the truth value is looked up directly.  When a
        quantifier is reached, the body is evaluated once per entity and the
        results are aggregated.
    """

    def __init__(self, operator_bundle: Optional[OperatorBundle] = None) -> None:
        self.ops = operator_bundle or OperatorBundle()

    # ---- Public API ---------------------------------------------------------

    def evaluate(
        self,
        rule: Rule,
        predicate_truths: Dict[str, Tensor],
        relation_truths: Dict[str, Tensor],
        entity_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Evaluate a single rule.

        Returns:
            Tensor of shape ``(B,)`` with truth value per batch element.
        """
        return self._eval_node(
            node=rule.ast,
            pred_truths=predicate_truths,
            rel_truths=relation_truths,
            mask=entity_mask,
            bindings={},
        )

    def evaluate_batch(
        self,
        rules: List[Rule],
        predicate_truths: Dict[str, Tensor],
        relation_truths: Dict[str, Tensor],
        entity_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Evaluate all rules.

        Returns:
            Tensor of shape ``(B, R)`` with truth values per batch element
            and per rule.
        """
        truths = [
            self.evaluate(r, predicate_truths, relation_truths, entity_mask)
            for r in rules
        ]
        return torch.stack(truths, dim=-1)  # (B, R)

    # ---- Recursive evaluation -----------------------------------------------

    def _eval_node(
        self,
        node: ASTNode,
        pred_truths: Dict[str, Tensor],
        rel_truths: Dict[str, Tensor],
        mask: Optional[Tensor],
        bindings: Dict[str, int],
    ) -> Tensor:
        """Recursively evaluate an AST node.

        Returns:
            Tensor of shape ``(B,)`` (or ``(B, N)`` intermediately when
            variables are unbound -- collapsed by quantifiers).
        """
        if isinstance(node, LiteralNode):
            return self._eval_literal(node, pred_truths, rel_truths, mask, bindings)
        elif isinstance(node, ConnectiveNode):
            return self._eval_connective(node, pred_truths, rel_truths, mask, bindings)
        elif isinstance(node, QuantifierNode):
            return self._eval_quantifier(node, pred_truths, rel_truths, mask, bindings)
        elif isinstance(node, NegationNode):
            return self._eval_negation(node, pred_truths, rel_truths, mask, bindings)
        else:
            raise TypeError(f"Unknown AST node type: {type(node)}")

    def _eval_literal(
        self,
        node: LiteralNode,
        pred_truths: Dict[str, Tensor],
        rel_truths: Dict[str, Tensor],
        mask: Optional[Tensor],
        bindings: Dict[str, int],
    ) -> Tensor:
        """Evaluate a literal node.

        If all variables are bound (via enclosing quantifiers), returns a
        point-wise truth value ``(B,)``.  If some variables are free, returns
        a higher-rank tensor that will be aggregated by an enclosing quantifier.
        """
        name = node.predicate_name
        variables = node.variables

        if len(variables) == 1:
            # Unary predicate: P(x)
            var = variables[0]
            if name not in pred_truths:
                raise KeyError(
                    f"Predicate '{name}' not found in predicate_truths. "
                    f"Available: {list(pred_truths.keys())}"
                )
            truth_table = pred_truths[name]  # (B, N)

            if var in bindings:
                # Variable is bound -> index into entity dimension
                idx = bindings[var]
                truth = truth_table[:, idx]  # (B,)
            else:
                # Variable is free -> return full entity dimension for quantifier
                truth = truth_table  # (B, N)

        elif len(variables) == 2:
            # Binary relation: R(x, y)
            var_x, var_y = variables
            if name not in rel_truths:
                raise KeyError(
                    f"Relation '{name}' not found in relation_truths. "
                    f"Available: {list(rel_truths.keys())}"
                )
            truth_table = rel_truths[name]  # (B, N, N)

            x_bound = var_x in bindings
            y_bound = var_y in bindings

            if x_bound and y_bound:
                truth = truth_table[:, bindings[var_x], bindings[var_y]]  # (B,)
            elif x_bound and not y_bound:
                truth = truth_table[:, bindings[var_x], :]  # (B, N)
            elif not x_bound and y_bound:
                truth = truth_table[:, :, bindings[var_y]]  # (B, N)
            else:
                # Both free: return (B, N, N) -- quantifiers will reduce
                truth = truth_table

        else:
            raise ValueError(
                f"Predicates/relations with arity > 2 not supported. "
                f"Got {len(variables)} variables for '{name}'."
            )

        # Apply literal-level negation
        if node.negated:
            truth = self.ops.fuzzy_not(truth)

        return truth

    def _eval_connective(
        self,
        node: ConnectiveNode,
        pred_truths: Dict[str, Tensor],
        rel_truths: Dict[str, Tensor],
        mask: Optional[Tensor],
        bindings: Dict[str, int],
    ) -> Tensor:
        """Evaluate a connective node by recursively evaluating children."""
        children_vals = [
            self._eval_node(child, pred_truths, rel_truths, mask, bindings)
            for child in node.children
        ]

        if node.connective == "and":
            result = children_vals[0]
            for cv in children_vals[1:]:
                result = self.ops.fuzzy_and(result, cv)
            return result

        elif node.connective == "or":
            result = children_vals[0]
            for cv in children_vals[1:]:
                result = self.ops.fuzzy_or(result, cv)
            return result

        elif node.connective == "implies":
            return self.ops.fuzzy_implies(children_vals[0], children_vals[1])

        elif node.connective == "iff":
            return self.ops.fuzzy_iff(children_vals[0], children_vals[1])

        else:
            raise ValueError(f"Unhandled connective: {node.connective}")

    def _eval_quantifier(
        self,
        node: QuantifierNode,
        pred_truths: Dict[str, Tensor],
        rel_truths: Dict[str, Tensor],
        mask: Optional[Tensor],
        bindings: Dict[str, int],
    ) -> Tensor:
        """Evaluate a quantifier node.

        Strategy: evaluate the body once per entity binding, stack results,
        and aggregate with the appropriate fuzzy quantifier.

        If the body already returns a tensor with a free entity dimension
        (because the body's literal has the quantified variable free), we
        aggregate along that dimension directly.  Otherwise we loop.
        """
        var = node.variable

        # Determine number of entities from any available truth table
        N = self._get_num_entities(pred_truths, rel_truths)

        # Evaluate body for each binding of the quantified variable
        body_truths: List[Tensor] = []
        for entity_idx in range(N):
            new_bindings = dict(bindings)
            new_bindings[var] = entity_idx
            body_val = self._eval_node(
                node.body, pred_truths, rel_truths, mask, new_bindings
            )
            body_truths.append(body_val)

        # Stack along a new entity dimension: (B, N) or (B, N, ...)
        stacked = torch.stack(body_truths, dim=-1)  # (..., N)

        # Apply entity mask if available (set masked entries to neutral value)
        if mask is not None:
            # Broadcast mask to match stacked shape
            # mask is (B, N), stacked is (..., N) where ... starts with B
            if stacked.dim() == 2:
                # (B, N)
                mask_expanded = mask[:, :stacked.shape[-1]]
            else:
                mask_expanded = mask[:, :stacked.shape[-1]]
                for _ in range(stacked.dim() - 2):
                    mask_expanded = mask_expanded.unsqueeze(1)

            if node.quantifier == "forall":
                # Masked entities should not reduce forall: set to 1
                stacked = stacked * mask_expanded + (1.0 - mask_expanded) * 1.0
            else:
                # Masked entities should not contribute to exists: set to 0
                stacked = stacked * mask_expanded

        # Aggregate
        if node.quantifier == "forall":
            return self.ops.fuzzy_forall(stacked, dim=-1)
        else:
            return self.ops.fuzzy_exists(stacked, dim=-1)

    def _eval_negation(
        self,
        node: NegationNode,
        pred_truths: Dict[str, Tensor],
        rel_truths: Dict[str, Tensor],
        mask: Optional[Tensor],
        bindings: Dict[str, int],
    ) -> Tensor:
        """Evaluate a negation node."""
        child_val = self._eval_node(
            node.child, pred_truths, rel_truths, mask, bindings
        )
        return self.ops.fuzzy_not(child_val)

    # ---- Helpers ------------------------------------------------------------

    @staticmethod
    def _get_num_entities(
        pred_truths: Dict[str, Tensor],
        rel_truths: Dict[str, Tensor],
    ) -> int:
        """Infer number of entities from available truth tables."""
        for t in pred_truths.values():
            return t.shape[1]  # (B, N)
        for t in rel_truths.values():
            return t.shape[1]  # (B, N, N)
        raise ValueError(
            "Cannot determine number of entities: both predicate_truths "
            "and relation_truths are empty."
        )


# =============================================================================
# 7. RuleNetworkOutput
# =============================================================================


@dataclass
class RuleNetworkOutput:
    """Output from :class:`RuleNetwork`.

    Attributes:
        constraint_loss: Scalar loss ready for backpropagation.
            Incorporates warmup weight: ``warmup_weight * raw_loss``.
        violations: ``(B, R)`` per-rule violation values.
        attention_weights: ``(B, R)`` attention weights when attention is
            enabled, ``None`` otherwise.
        warmup_weight: Current warmup multiplier (0 to constraint_weight).
        logs: Interpretable logs when ``return_logs=True``, ``None`` otherwise.
    """

    constraint_loss: Tensor
    violations: Tensor
    attention_weights: Optional[Tensor]
    warmup_weight: float
    logs: Optional[Dict[str, Any]]


# =============================================================================
# 8. Violation Aggregation Functions
# =============================================================================


def mean_aggregation(violations: Tensor, dim: int = -1) -> Tensor:
    """Simple mean aggregation of violations along *dim*.

    Args:
        violations: Tensor of violation values.
        dim: Dimension to aggregate over.

    Returns:
        Mean violations.
    """
    return violations.mean(dim=dim)


def soft_min_aggregation(
    violations: Tensor,
    temperature: float = 0.1,
    dim: int = -1,
) -> Tensor:
    """Soft-min aggregation using negative-temperature softmax weighting.

    Approximates min as temperature -> 0.  At higher temperatures, falls back
    toward a weighted mean.  This encourages the network to fix the *most*
    violated rule preferentially.

    Args:
        violations: Tensor of violation values.
        temperature: Softmax temperature (lower = closer to hard min).
        dim: Dimension to aggregate over.

    Returns:
        Soft-min aggregated violations.
    """
    # Weights: softmax(-violations / temperature) emphasises smallest values
    weights = F.softmax(-violations / max(temperature, 1e-8), dim=dim)
    return (weights * violations).sum(dim=dim)


def p_mean_aggregation(
    violations: Tensor,
    p: float = 2.0,
    dim: int = -1,
) -> Tensor:
    """Generalized mean (p-mean) aggregation.

    ``M_p(x) = (mean(x^p))^(1/p)``

    For ``p > 1`` the aggregation emphasises larger violations (closer to max).
    For ``p = 1`` this is equivalent to the arithmetic mean.
    For ``p < 1`` it is closer to min.

    Args:
        violations: Tensor of violation values (should be non-negative).
        p: Exponent.  Must be non-zero.
        dim: Dimension to aggregate over.

    Returns:
        P-mean aggregated violations.
    """
    if abs(p) < 1e-8:
        raise ValueError("p must be non-zero for p-mean aggregation.")
    # Clamp to avoid 0^p gradient issues when p < 1
    v = violations.clamp(min=1e-8)
    return v.pow(p).mean(dim=dim).pow(1.0 / p)


_AGGREGATION_FNS: Dict[str, Callable[..., Tensor]] = {
    "mean": mean_aggregation,
    "soft_min": soft_min_aggregation,
    "p_mean": p_mean_aggregation,
}


# =============================================================================
# 9. RuleNetwork (nn.Module)
# =============================================================================


class RuleNetwork(nn.Module):
    """Attention-weighted rule application with constraint loss computation.

    The RuleNetwork does *not* evaluate rules (that is the job of
    :class:`RuleEvaluator`).  Instead it takes pre-computed rule truth values,
    computes per-rule violations, optionally applies learned attention weights,
    aggregates into a scalar constraint loss, and produces interpretable logs.

    Architecture:
        - ``rule_embeddings``: learned ``(max_rules, rule_embed_dim)`` vectors.
        - ``attention_mlp``: ``[workspace_summary; query_embedding; rule_embedding]
          -> scalar`` via a two-layer MLP with Tanh activation.
        - ``_step``: internal training step counter for warmup scheduling.

    Forward pass:
        1. Compute violations from rule truths.
        2. (Optional) Compute attention weights from workspace summary, query,
           and rule embeddings.
        3. Aggregate violations into scalar loss.
        4. Apply warmup weight.
        5. (Optional) Extract interpretable logs.

    Args:
        config: :class:`RuleConfig` instance.
    """

    def __init__(self, config: Optional[RuleConfig] = None) -> None:
        super().__init__()
        self.config = config or RuleConfig()

        # Per-rule learned embeddings
        self.rule_embeddings = nn.Embedding(
            self.config.max_rules, self.config.rule_embed_dim
        )

        # Attention MLP: concat(workspace, query, rule_embed) -> scalar
        if self.config.use_attention:
            attn_input_dim = (
                self.config.attention_dim  # workspace_summary
                + self.config.attention_dim  # query_embedding
                + self.config.rule_embed_dim  # rule_embedding
            )
            self.attention_mlp = nn.Sequential(
                nn.Linear(attn_input_dim, self.config.attention_dim),
                nn.Tanh(),
                nn.Linear(self.config.attention_dim, 1),
            )
        else:
            self.attention_mlp = None

        # Internal step counter (not a parameter, but tracked in state_dict)
        self.register_buffer("_step_buffer", torch.tensor(0, dtype=torch.long))

    @property
    def _step(self) -> int:
        return self._step_buffer.item()

    @_step.setter
    def _step(self, value: int) -> None:
        self._step_buffer.fill_(value)

    # ---- Attention -----------------------------------------------------------

    def compute_attention(
        self,
        workspace_summary: Tensor,
        query_embedding: Optional[Tensor],
        num_rules: int,
    ) -> Tensor:
        """Compute per-rule attention weights.

        Args:
            workspace_summary: ``(B, attention_dim)`` global workspace summary.
            query_embedding: ``(B, attention_dim)`` optional query context.
                If ``None``, a zero vector is used.
            num_rules: Number of active rules ``R <= max_rules``.

        Returns:
            ``(B, R)`` attention weights summing to 1 along the rule dimension.
        """
        B = workspace_summary.shape[0]
        device = workspace_summary.device

        # Rule embeddings for active rules: (R, rule_embed_dim)
        rule_indices = torch.arange(num_rules, device=device)
        rule_embeds = self.rule_embeddings(rule_indices)  # (R, E_r)

        # Default query to zeros if not provided
        if query_embedding is None:
            query_embedding = torch.zeros(
                B, self.config.attention_dim, device=device
            )

        # Expand for broadcasting: workspace (B,1,D), query (B,1,D), rules (1,R,E_r)
        ws_exp = workspace_summary.unsqueeze(1).expand(B, num_rules, -1)
        qe_exp = query_embedding.unsqueeze(1).expand(B, num_rules, -1)
        re_exp = rule_embeds.unsqueeze(0).expand(B, num_rules, -1)

        # Concatenate: (B, R, D + D + E_r)
        attn_input = torch.cat([ws_exp, qe_exp, re_exp], dim=-1)

        # MLP -> (B, R, 1) -> squeeze -> (B, R)
        logits = self.attention_mlp(attn_input).squeeze(-1)

        # Softmax over rules
        alpha = F.softmax(logits, dim=-1)  # (B, R)
        return alpha

    # ---- Violations ----------------------------------------------------------

    def compute_violation(
        self,
        rule_truths: Tensor,
        rules: List[Rule],
    ) -> Tensor:
        """Compute per-rule violation from truth values.

        For rules where ``intended_true=True``:
            ``violation = (1 - truth) * weight``

        For rules where ``intended_true=False`` (negative constraints):
            ``violation = truth * weight``

        Args:
            rule_truths: ``(B, R)`` truth values in [0, 1].
            rules: List of :class:`Rule` objects (length R).

        Returns:
            ``(B, R)`` violation values.
        """
        B, R = rule_truths.shape
        device = rule_truths.device

        # Build masks and weights from rule metadata
        intended_true_mask = torch.tensor(
            [r.intended_true for r in rules], device=device, dtype=rule_truths.dtype
        )  # (R,)
        weights = torch.tensor(
            [r.weight for r in rules], device=device, dtype=rule_truths.dtype
        )  # (R,)

        # Violation: intended_true -> 1 - truth, intended_false -> truth
        # v = mask * (1 - truth) + (1 - mask) * truth
        violations = (
            intended_true_mask.unsqueeze(0) * (1.0 - rule_truths)
            + (1.0 - intended_true_mask.unsqueeze(0)) * rule_truths
        )

        # Apply per-rule weights
        violations = violations * weights.unsqueeze(0)

        return violations

    # ---- Constraint Loss -----------------------------------------------------

    def compute_constraint_loss(
        self,
        violations: Tensor,
        attention_weights: Optional[Tensor] = None,
    ) -> Tensor:
        """Aggregate violations into a scalar constraint loss.

        If attention is enabled and weights are provided:
            ``loss = mean_over_batch(sum_over_rules(alpha * violation))``

        Otherwise:
            ``loss = mean_over_batch(agg_over_rules(violation))``

        where ``agg`` is one of ``mean``, ``soft_min``, or ``p_mean`` as
        configured in :class:`RuleConfig`.

        Args:
            violations: ``(B, R)`` per-rule violations.
            attention_weights: ``(B, R)`` attention weights, or ``None``.

        Returns:
            Scalar loss tensor.
        """
        if attention_weights is not None:
            # Attention-weighted sum per batch, then mean over batch
            per_batch = (attention_weights * violations).sum(dim=-1)  # (B,)
            return per_batch.mean()

        # Non-attention aggregation
        agg_name = self.config.violation_agg
        agg_fn = _AGGREGATION_FNS.get(agg_name)
        if agg_fn is None:
            raise ValueError(
                f"Unknown violation_agg '{agg_name}'. "
                f"Choose from {list(_AGGREGATION_FNS.keys())}."
            )

        # Build kwargs for the specific aggregation function
        if agg_name == "soft_min":
            per_batch = agg_fn(violations, dim=-1)
        elif agg_name == "p_mean":
            per_batch = agg_fn(violations, p=self.config.p_mean_p, dim=-1)
        else:
            per_batch = agg_fn(violations, dim=-1)

        return per_batch.mean()  # scalar

    # ---- Warmup --------------------------------------------------------------

    def get_warmup_weight(self) -> float:
        """Return current warmup multiplier for the constraint weight.

        Linearly ramps from 0 to ``config.constraint_weight`` over
        ``config.warmup_steps`` steps.

        Returns:
            Float multiplier to apply to the raw constraint loss.
        """
        if self.config.warmup_steps <= 0:
            return self.config.constraint_weight
        if self._step >= self.config.warmup_steps:
            return self.config.constraint_weight
        return self.config.constraint_weight * (self._step / self.config.warmup_steps)

    # ---- Log Extraction ------------------------------------------------------

    def _extract_logs(
        self,
        violations: Tensor,
        rule_truths: Tensor,
        attention_weights: Optional[Tensor],
        rules: List[Rule],
    ) -> Dict[str, Any]:
        """Extract interpretable logs.

        All tensors in the logs are detached (no gradient tracking).

        Returns a dict with:
            - ``attention_weights``: ``(B, R)`` detached (or None).
            - ``top_violated_rules``: List of dicts with rule indices,
              names, and mean violation values (top-K).
            - ``per_rule_truths``: ``(B, R)`` detached.
            - ``violation_summary``: dict with mean, max, min violation stats.
        """
        B, R = violations.shape
        logs: Dict[str, Any] = {}

        # Attention weights (detached)
        if attention_weights is not None:
            logs["attention_weights"] = attention_weights.detach()
        else:
            logs["attention_weights"] = None

        # Per-rule truths (detached)
        logs["per_rule_truths"] = rule_truths.detach()

        # Violation summary stats
        viol_detached = violations.detach()
        logs["violation_summary"] = {
            "mean": viol_detached.mean().item(),
            "max": viol_detached.max().item(),
            "min": viol_detached.min().item(),
        }

        # Top-K violated rules (by mean violation across batch)
        mean_violations = viol_detached.mean(dim=0)  # (R,)
        k = min(self.config.top_k_logs, R)
        topk_vals, topk_indices = torch.topk(mean_violations, k)

        top_violated: List[Dict[str, Any]] = []
        for i in range(k):
            idx = topk_indices[i].item()
            rule_name = rules[idx].name if idx < len(rules) else f"rule_{idx}"
            top_violated.append(
                {
                    "index": idx,
                    "name": rule_name,
                    "mean_violation": topk_vals[i].item(),
                }
            )
        logs["top_violated_rules"] = top_violated

        # Optionally include full tensors for debugging
        if self.config.full_tensor_logs:
            logs["violations_full"] = viol_detached
            logs["rule_truths_full"] = rule_truths.detach()
            if attention_weights is not None:
                logs["attention_weights_full"] = attention_weights.detach()

        return logs

    # ---- Forward -------------------------------------------------------------

    def forward(
        self,
        rules: List[Rule],
        rule_truths: Tensor,
        workspace_summary: Optional[Tensor] = None,
        query_embedding: Optional[Tensor] = None,
        return_logs: bool = False,
    ) -> RuleNetworkOutput:
        """Compute constraint loss with optional attention and logging.

        Args:
            rules: List of :class:`Rule` objects (length R).
            rule_truths: ``(B, R)`` truth values from :class:`RuleEvaluator`.
            workspace_summary: ``(B, D)`` from global workspace (needed for
                attention).
            query_embedding: ``(B, D)`` optional query context for attention.
            return_logs: Whether to produce interpretable logs.

        Returns:
            :class:`RuleNetworkOutput` with constraint loss, violations,
            optional attention weights, warmup weight, and optional logs.
        """
        B, R = rule_truths.shape

        # Step 1: Compute violations
        violations = self.compute_violation(rule_truths, rules)  # (B, R)

        # Step 2: Compute attention weights (if enabled and workspace available)
        attention_weights: Optional[Tensor] = None
        if self.config.use_attention and workspace_summary is not None:
            attention_weights = self.compute_attention(
                workspace_summary, query_embedding, R
            )

        # Step 3: Aggregate into scalar loss
        raw_loss = self.compute_constraint_loss(violations, attention_weights)

        # Step 4: Apply warmup weight
        warmup_w = self.get_warmup_weight()
        constraint_loss = warmup_w * raw_loss

        # Step 5: Increment step counter
        self._step = self._step + 1

        # Step 6: Extract logs if requested
        logs: Optional[Dict[str, Any]] = None
        if return_logs:
            logs = self._extract_logs(
                violations, rule_truths, attention_weights, rules
            )

        return RuleNetworkOutput(
            constraint_loss=constraint_loss,
            violations=violations,
            attention_weights=attention_weights,
            warmup_weight=warmup_w,
            logs=logs,
        )


# =============================================================================
# 10. Theory
# =============================================================================


class Theory:
    """A collection of rules forming a knowledge base / theory.

    Provides add/remove operations and JSON-serializable dict conversion
    for persistence and transfer.

    Attributes:
        rules: List of :class:`Rule` objects.
        name: Human-readable theory name.
    """

    def __init__(
        self,
        rules: Optional[List[Rule]] = None,
        name: str = "",
    ) -> None:
        self.rules: List[Rule] = rules or []
        self.name = name

    def add_rule(self, rule: Rule) -> None:
        """Add a rule to the theory."""
        self.rules.append(rule)

    def remove_rule(self, name: str) -> None:
        """Remove the first rule with the given name.

        Raises:
            ValueError: If no rule with the given name exists.
        """
        for i, r in enumerate(self.rules):
            if r.name == name:
                self.rules.pop(i)
                return
        raise ValueError(f"Rule '{name}' not found in theory '{self.name}'.")

    def get_rule(self, name: str) -> Rule:
        """Get a rule by name.

        Raises:
            KeyError: If no rule with the given name exists.
        """
        for r in self.rules:
            if r.name == name:
                return r
        raise KeyError(f"Rule '{name}' not found in theory '{self.name}'.")

    def __len__(self) -> int:
        return len(self.rules)

    def __iter__(self):
        return iter(self.rules)

    def __repr__(self) -> str:
        return (
            f"Theory(name={self.name!r}, "
            f"rules=[{', '.join(r.name or repr(r) for r in self.rules)}])"
        )

    # ---- Serialization -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize theory to a JSON-compatible dict.

        Returns:
            Dict with ``name``, ``rules`` (list of rule dicts).
        """
        return {
            "name": self.name,
            "rules": [self._rule_to_dict(r) for r in self.rules],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Theory:
        """Deserialize theory from a dict.

        Args:
            d: Dict with ``name`` and ``rules`` keys.

        Returns:
            A new :class:`Theory` instance.
        """
        rules = compile_rules_from_list(d.get("rules", []))
        return cls(rules=rules, name=d.get("name", ""))

    @staticmethod
    def _node_to_dict(node: ASTNode) -> Dict[str, Any]:
        """Recursively serialize an AST node to a dict."""
        if isinstance(node, LiteralNode):
            d: Dict[str, Any] = {
                "type": "literal",
                "predicate": node.predicate_name,
                "vars": node.variables,
            }
            if node.negated:
                d["negated"] = True
            return d

        elif isinstance(node, ConnectiveNode):
            return {
                "type": node.connective,
                "children": [Theory._node_to_dict(c) for c in node.children],
            }

        elif isinstance(node, QuantifierNode):
            return {
                "type": node.quantifier,
                "var": node.variable,
                "body": Theory._node_to_dict(node.body),
            }

        elif isinstance(node, NegationNode):
            return {
                "type": "not",
                "child": Theory._node_to_dict(node.child),
            }

        else:
            raise TypeError(f"Cannot serialize AST node type: {type(node)}")

    @staticmethod
    def _rule_to_dict(rule: Rule) -> Dict[str, Any]:
        """Serialize a single Rule to a dict."""
        d = Theory._node_to_dict(rule.ast)
        d["name"] = rule.name
        d["weight"] = rule.weight
        d["intended_true"] = rule.intended_true
        if rule.metadata:
            d["metadata"] = rule.metadata
        return d


# =============================================================================
# 11. Utility: Total Loss Combiner
# =============================================================================


def compute_total_loss(
    task_loss: Tensor,
    rule_network_output: RuleNetworkOutput,
) -> Tensor:
    """Compute total training loss combining task and symbolic constraint losses.

    ``L_total = L_task + warmup_weight * L_symbolic``

    Note: The warmup weight is already applied inside
    ``RuleNetworkOutput.constraint_loss``, so this is simply an addition.

    Args:
        task_loss: Scalar task loss.
        rule_network_output: Output from :meth:`RuleNetwork.forward`.

    Returns:
        Scalar total loss.
    """
    return task_loss + rule_network_output.constraint_loss


# =============================================================================
# 12. Self-Tests
# =============================================================================


def _run_self_tests() -> None:
    """Run self-tests for the rule engine module.

    Each test prints PASS or FAIL.  Exits with code 1 on any failure.
    """
    num_passed = 0
    num_failed = 0
    test_results: List[Tuple[str, bool, str]] = []

    def _test(name: str, condition: bool, detail: str = "") -> None:
        nonlocal num_passed, num_failed
        if condition:
            num_passed += 1
            print(f"  PASS: {name}")
        else:
            num_failed += 1
            msg = f"  FAIL: {name}"
            if detail:
                msg += f" -- {detail}"
            print(msg)
        test_results.append((name, condition, detail))

    torch.manual_seed(42)
    B, N, R_count = 4, 5, 3  # batch, entities, rules
    device = torch.device("cpu")

    print("=" * 70)
    print("Rule Engine Template -- Self-Tests")
    print("=" * 70)

    # ---- Test 1: LiteralNode creation and attributes ----
    print("\n--- Test 1: LiteralNode creation and attributes ---")
    lit = LiteralNode("is_bird", ["x"])
    _test(
        "LiteralNode predicate_name",
        lit.predicate_name == "is_bird",
        f"got {lit.predicate_name!r}",
    )
    _test(
        "LiteralNode variables",
        lit.variables == ["x"],
        f"got {lit.variables!r}",
    )
    _test(
        "LiteralNode negated default",
        lit.negated is False,
        f"got {lit.negated!r}",
    )
    lit_neg = LiteralNode("is_bird", ["x"], negated=True)
    _test(
        "LiteralNode negated=True",
        lit_neg.negated is True,
        f"got {lit_neg.negated!r}",
    )

    # ---- Test 2: ConnectiveNode with AND/OR children ----
    print("\n--- Test 2: ConnectiveNode with AND/OR children ---")
    child_a = LiteralNode("P", ["x"])
    child_b = LiteralNode("Q", ["x"])
    and_node = ConnectiveNode("and", [child_a, child_b])
    _test(
        "ConnectiveNode AND connective",
        and_node.connective == "and",
        f"got {and_node.connective!r}",
    )
    _test(
        "ConnectiveNode AND children count",
        len(and_node.children) == 2,
        f"got {len(and_node.children)}",
    )
    or_node = ConnectiveNode("or", [child_a, child_b])
    _test(
        "ConnectiveNode OR connective",
        or_node.connective == "or",
        f"got {or_node.connective!r}",
    )

    # ---- Test 3: QuantifierNode with FORALL scoping ----
    print("\n--- Test 3: QuantifierNode with FORALL scoping ---")
    body = LiteralNode("is_bird", ["x"])
    forall_node = QuantifierNode("forall", "x", body)
    _test(
        "QuantifierNode quantifier",
        forall_node.quantifier == "forall",
        f"got {forall_node.quantifier!r}",
    )
    _test(
        "QuantifierNode variable",
        forall_node.variable == "x",
        f"got {forall_node.variable!r}",
    )
    _test(
        "QuantifierNode body type",
        isinstance(forall_node.body, LiteralNode),
        f"got {type(forall_node.body).__name__}",
    )

    # ---- Test 4: Rule compilation from dict ----
    print("\n--- Test 4: Rule compilation from dict ---")
    rule_dict = {
        "name": "birds_fly",
        "type": "forall",
        "var": "x",
        "body": {
            "type": "implies",
            "lhs": {"type": "literal", "predicate": "is_bird", "vars": ["x"]},
            "rhs": {"type": "literal", "predicate": "can_fly", "vars": ["x"]},
        },
    }
    rule = compile_rule_from_dict(rule_dict)
    _test("Rule name", rule.name == "birds_fly", f"got {rule.name!r}")
    _test(
        "Rule AST root is QuantifierNode",
        isinstance(rule.ast, QuantifierNode),
        f"got {type(rule.ast).__name__}",
    )
    _test(
        "Rule AST body is ConnectiveNode(implies)",
        isinstance(rule.ast.body, ConnectiveNode) and rule.ast.body.connective == "implies",
        f"got {type(rule.ast.body).__name__}",
    )

    # ---- Test 5: Rule evaluation -- simple literal ----
    print("\n--- Test 5: Rule evaluation -- simple literal ---")
    ops = OperatorBundle()
    evaluator = RuleEvaluator(ops)

    # Simple rule: P(entity_0) for batch of 4 items with 5 entities
    simple_rule = Rule(
        ast=LiteralNode("P", ["x"]),
        name="simple_P",
    )
    # Wrap in a quantifier so we get (B,) output
    simple_forall = Rule(
        ast=QuantifierNode("forall", "x", LiteralNode("P", ["x"])),
        name="forall_P",
    )
    pred_truths = {"P": torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.5]] * B)}
    truth = evaluator.evaluate(simple_forall, pred_truths, {})
    _test(
        "Simple literal FORALL P truth shape",
        truth.shape == (B,),
        f"got shape {truth.shape}",
    )
    # Product FORALL = product of all truth values = 0.9*0.8*0.7*0.6*0.5
    expected_product = 0.9 * 0.8 * 0.7 * 0.6 * 0.5
    _test(
        "Simple literal FORALL P truth value",
        abs(truth[0].item() - expected_product) < 1e-4,
        f"expected {expected_product:.4f}, got {truth[0].item():.4f}",
    )

    # ---- Test 6: Rule evaluation -- conjunction P(x) AND Q(x) ----
    print("\n--- Test 6: Rule evaluation -- conjunction P(x) AND Q(x) ---")
    conj_rule = Rule(
        ast=QuantifierNode(
            "forall",
            "x",
            ConnectiveNode("and", [
                LiteralNode("P", ["x"]),
                LiteralNode("Q", ["x"]),
            ]),
        ),
        name="forall_P_and_Q",
    )
    pred_truths_pq = {
        "P": torch.ones(B, N) * 0.8,
        "Q": torch.ones(B, N) * 0.9,
    }
    truth_conj = evaluator.evaluate(conj_rule, pred_truths_pq, {})
    _test(
        "Conjunction truth shape",
        truth_conj.shape == (B,),
        f"got shape {truth_conj.shape}",
    )
    # AND = 0.8 * 0.9 = 0.72 per entity, FORALL = 0.72^5
    expected_conj = (0.8 * 0.9) ** N
    _test(
        "Conjunction truth value",
        abs(truth_conj[0].item() - expected_conj) < 1e-3,
        f"expected {expected_conj:.4f}, got {truth_conj[0].item():.4f}",
    )

    # ---- Test 7: Rule evaluation -- implication P(x) -> Q(x) ----
    print("\n--- Test 7: Rule evaluation -- implication P(x) -> Q(x) ---")
    impl_rule = Rule(
        ast=QuantifierNode(
            "forall",
            "x",
            ConnectiveNode("implies", [
                LiteralNode("P", ["x"]),
                LiteralNode("Q", ["x"]),
            ]),
        ),
        name="forall_P_implies_Q",
    )
    truth_impl = evaluator.evaluate(impl_rule, pred_truths_pq, {})
    _test(
        "Implication truth shape",
        truth_impl.shape == (B,),
        f"got shape {truth_impl.shape}",
    )
    # Reichenbach implication: 1 - 0.8 + 0.8*0.9 = 0.2 + 0.72 = 0.92 per entity
    # FORALL = 0.92^5
    expected_impl = (1.0 - 0.8 + 0.8 * 0.9) ** N
    _test(
        "Implication truth value",
        abs(truth_impl[0].item() - expected_impl) < 1e-3,
        f"expected {expected_impl:.4f}, got {truth_impl[0].item():.4f}",
    )

    # ---- Test 8: Rule evaluation -- FORALL x: P(x) aggregation ----
    print("\n--- Test 8: Rule evaluation -- FORALL x: P(x) aggregation ---")
    # Varying per-entity truths
    varying_truths = {"P": torch.tensor([[1.0, 0.5, 0.8, 0.3, 0.9]] * B)}
    forall_rule = Rule(
        ast=QuantifierNode("forall", "x", LiteralNode("P", ["x"])),
        name="forall_varying",
    )
    truth_forall = evaluator.evaluate(forall_rule, varying_truths, {})
    expected_forall = 1.0 * 0.5 * 0.8 * 0.3 * 0.9
    _test(
        "FORALL aggregation value",
        abs(truth_forall[0].item() - expected_forall) < 1e-4,
        f"expected {expected_forall:.4f}, got {truth_forall[0].item():.4f}",
    )

    # ---- Test 9: Rule evaluation -- nested FORALL x EXISTS y R(x,y) ----
    print("\n--- Test 9: Rule evaluation -- nested FORALL x EXISTS y R(x,y) ---")
    nested_rule = Rule(
        ast=QuantifierNode(
            "forall",
            "x",
            QuantifierNode(
                "exists",
                "y",
                LiteralNode("likes", ["x", "y"]),
            ),
        ),
        name="forall_x_exists_y_likes",
    )
    # R(x,y) truth table: (B, N, N) -- make it so each row has at least one high value
    rel_data = torch.zeros(B, N, N)
    rel_data[:, 0, 1] = 0.9  # entity 0 likes entity 1
    rel_data[:, 1, 2] = 0.8  # entity 1 likes entity 2
    rel_data[:, 2, 0] = 0.7  # entity 2 likes entity 0
    rel_data[:, 3, 3] = 0.6  # entity 3 likes entity 3
    rel_data[:, 4, 0] = 0.5  # entity 4 likes entity 0
    rel_truths = {"likes": rel_data}
    truth_nested = evaluator.evaluate(nested_rule, {}, rel_truths)
    _test(
        "Nested FORALL EXISTS shape",
        truth_nested.shape == (B,),
        f"got shape {truth_nested.shape}",
    )
    # EXISTS y R(x,y) for product: 1 - prod(1 - R(x,:)) per x
    # Then FORALL x: product of all EXISTS results
    _test(
        "Nested FORALL EXISTS value in (0, 1)",
        0.0 < truth_nested[0].item() < 1.0,
        f"got {truth_nested[0].item():.4f}",
    )

    # ---- Test 10: Attention weights sum to 1 ----
    print("\n--- Test 10: Attention weights sum to 1 ---")
    config = RuleConfig(
        max_rules=16,
        use_attention=True,
        attention_dim=32,
        rule_embed_dim=16,
    )
    net = RuleNetwork(config)
    ws = torch.randn(B, 32)
    qe = torch.randn(B, 32)
    alpha = net.compute_attention(ws, qe, num_rules=R_count)
    _test(
        "Attention weights shape",
        alpha.shape == (B, R_count),
        f"got shape {alpha.shape}",
    )
    alpha_sum = alpha.sum(dim=-1)
    _test(
        "Attention weights sum to 1",
        torch.allclose(alpha_sum, torch.ones(B), atol=1e-5),
        f"sums = {alpha_sum.tolist()}",
    )

    # ---- Test 11: Violation computation -- intended_true vs intended_false ----
    print("\n--- Test 11: Violation computation -- intended_true vs intended_false ---")
    rules_mixed = [
        Rule(ast=LiteralNode("P", ["x"]), name="r1", intended_true=True, weight=1.0),
        Rule(ast=LiteralNode("Q", ["x"]), name="r2", intended_true=False, weight=1.0),
        Rule(ast=LiteralNode("S", ["x"]), name="r3", intended_true=True, weight=2.0),
    ]
    truths_mixed = torch.tensor([[0.8, 0.3, 0.6]] * B)  # (B, 3)
    violations_mixed = net.compute_violation(truths_mixed, rules_mixed)
    _test(
        "Violation shape",
        violations_mixed.shape == (B, 3),
        f"got shape {violations_mixed.shape}",
    )
    # r1 (intended_true, weight=1.0): violation = 1 - 0.8 = 0.2
    _test(
        "Violation r1 (intended_true)",
        abs(violations_mixed[0, 0].item() - 0.2) < 1e-5,
        f"expected 0.2, got {violations_mixed[0, 0].item():.4f}",
    )
    # r2 (intended_false, weight=1.0): violation = 0.3
    _test(
        "Violation r2 (intended_false)",
        abs(violations_mixed[0, 1].item() - 0.3) < 1e-5,
        f"expected 0.3, got {violations_mixed[0, 1].item():.4f}",
    )
    # r3 (intended_true, weight=2.0): violation = (1 - 0.6) * 2 = 0.8
    _test(
        "Violation r3 (weighted)",
        abs(violations_mixed[0, 2].item() - 0.8) < 1e-5,
        f"expected 0.8, got {violations_mixed[0, 2].item():.4f}",
    )

    # ---- Test 12: Constraint loss with mean aggregation ----
    print("\n--- Test 12: Constraint loss with mean aggregation ---")
    config_no_attn = RuleConfig(use_attention=False, violation_agg="mean")
    net_no_attn = RuleNetwork(config_no_attn)
    viols = torch.tensor([[0.1, 0.3, 0.5]] * B)
    loss_mean = net_no_attn.compute_constraint_loss(viols, attention_weights=None)
    expected_mean = viols.mean().item()
    _test(
        "Mean constraint loss",
        abs(loss_mean.item() - expected_mean) < 1e-5,
        f"expected {expected_mean:.4f}, got {loss_mean.item():.4f}",
    )

    # ---- Test 13: Constraint loss with attention weighting ----
    print("\n--- Test 13: Constraint loss with attention weighting ---")
    alpha_fixed = torch.tensor([[0.5, 0.3, 0.2]] * B)
    loss_attn = net.compute_constraint_loss(viols, attention_weights=alpha_fixed)
    expected_attn = (0.5 * 0.1 + 0.3 * 0.3 + 0.2 * 0.5)
    _test(
        "Attention-weighted constraint loss",
        abs(loss_attn.item() - expected_attn) < 1e-5,
        f"expected {expected_attn:.4f}, got {loss_attn.item():.4f}",
    )

    # ---- Test 14: Warmup schedule correctness ----
    print("\n--- Test 14: Warmup schedule correctness ---")
    config_warmup = RuleConfig(
        constraint_weight=0.5,
        warmup_steps=100,
        use_attention=False,
        violation_agg="mean",
    )
    net_warmup = RuleNetwork(config_warmup)
    # At step 0, warmup weight = 0
    _test(
        "Warmup at step 0",
        abs(net_warmup.get_warmup_weight() - 0.0) < 1e-8,
        f"got {net_warmup.get_warmup_weight():.6f}",
    )
    # Simulate 50 steps
    net_warmup._step = 50
    expected_w50 = 0.5 * (50 / 100)
    _test(
        "Warmup at step 50",
        abs(net_warmup.get_warmup_weight() - expected_w50) < 1e-6,
        f"expected {expected_w50:.4f}, got {net_warmup.get_warmup_weight():.4f}",
    )
    # At step 100, should be full weight
    net_warmup._step = 100
    _test(
        "Warmup at step 100 (full)",
        abs(net_warmup.get_warmup_weight() - 0.5) < 1e-8,
        f"got {net_warmup.get_warmup_weight():.4f}",
    )
    # Beyond warmup steps
    net_warmup._step = 200
    _test(
        "Warmup beyond warmup_steps",
        abs(net_warmup.get_warmup_weight() - 0.5) < 1e-8,
        f"got {net_warmup.get_warmup_weight():.4f}",
    )

    # ---- Test 15: Log extraction with top-K ----
    print("\n--- Test 15: Log extraction with top-K ---")
    config_logs = RuleConfig(
        max_rules=16,
        use_attention=True,
        attention_dim=32,
        rule_embed_dim=16,
        top_k_logs=2,
        constraint_weight=0.1,
        warmup_steps=0,  # no warmup so loss is nonzero
    )
    net_logs = RuleNetwork(config_logs)
    rules_log = [
        Rule(ast=LiteralNode("A", ["x"]), name="rule_a", intended_true=True),
        Rule(ast=LiteralNode("B", ["x"]), name="rule_b", intended_true=True),
        Rule(ast=LiteralNode("C", ["x"]), name="rule_c", intended_true=True),
    ]
    truths_log = torch.tensor([[0.9, 0.3, 0.7]] * B)  # rule_b has highest violation
    ws_log = torch.randn(B, 32)
    output = net_logs.forward(
        rules=rules_log,
        rule_truths=truths_log,
        workspace_summary=ws_log,
        return_logs=True,
    )
    _test(
        "Output has logs",
        output.logs is not None,
        "logs is None",
    )
    _test(
        "Logs has top_violated_rules",
        "top_violated_rules" in output.logs,
        f"keys: {list(output.logs.keys()) if output.logs else 'None'}",
    )
    top_viol = output.logs["top_violated_rules"]
    _test(
        "Top-K count is 2",
        len(top_viol) == 2,
        f"got {len(top_viol)}",
    )
    # rule_b (index 1) has truth 0.3 -> violation 0.7, should be in top-2
    top_indices = [tv["index"] for tv in top_viol]
    _test(
        "Most violated rule is rule_b (index 1)",
        top_indices[0] == 1,
        f"top indices = {top_indices}",
    )
    _test(
        "Logs has violation_summary",
        "violation_summary" in output.logs,
        f"keys: {list(output.logs.keys()) if output.logs else 'None'}",
    )
    _test(
        "violation_summary has mean/max/min",
        all(k in output.logs["violation_summary"] for k in ("mean", "max", "min")),
        f"keys: {list(output.logs['violation_summary'].keys())}",
    )
    _test(
        "Logs attention_weights are detached",
        output.logs["attention_weights"] is not None
        and not output.logs["attention_weights"].requires_grad,
        "attention_weights has grad or is None",
    )
    _test(
        "Logs per_rule_truths shape",
        output.logs["per_rule_truths"].shape == (B, 3),
        f"got shape {output.logs['per_rule_truths'].shape}",
    )

    # ---- Test 16: Theory serialization round-trip ----
    print("\n--- Test 16: Theory serialization round-trip ---")
    theory = Theory(name="test_theory")
    theory.add_rule(compile_rule_from_dict(rule_dict))
    theory.add_rule(
        Rule(
            ast=ConnectiveNode("and", [
                LiteralNode("P", ["x"]),
                LiteralNode("Q", ["x"]),
            ]),
            name="P_and_Q",
        )
    )
    d = theory.to_dict()
    _test(
        "Theory to_dict name",
        d["name"] == "test_theory",
        f"got {d['name']!r}",
    )
    _test(
        "Theory to_dict rule count",
        len(d["rules"]) == 2,
        f"got {len(d['rules'])}",
    )
    theory2 = Theory.from_dict(d)
    _test(
        "Theory round-trip name",
        theory2.name == "test_theory",
        f"got {theory2.name!r}",
    )
    _test(
        "Theory round-trip rule count",
        len(theory2) == 2,
        f"got {len(theory2)}",
    )
    _test(
        "Theory round-trip rule names",
        [r.name for r in theory2.rules] == ["birds_fly", "P_and_Q"],
        f"got {[r.name for r in theory2.rules]}",
    )

    # ---- Test 17: p-mean aggregation ----
    print("\n--- Test 17: p-mean aggregation ---")
    viols_pm = torch.tensor([0.1, 0.4, 0.9])
    pm2 = p_mean_aggregation(viols_pm, p=2.0, dim=0)
    expected_pm2 = ((0.1**2 + 0.4**2 + 0.9**2) / 3) ** 0.5
    _test(
        "p-mean (p=2) value",
        abs(pm2.item() - expected_pm2) < 1e-4,
        f"expected {expected_pm2:.4f}, got {pm2.item():.4f}",
    )

    # ---- Test 18: soft_min aggregation ----
    print("\n--- Test 18: soft_min aggregation ---")
    viols_sm = torch.tensor([0.1, 0.5, 0.9])
    sm = soft_min_aggregation(viols_sm, temperature=0.01, dim=0)
    # At very low temperature, soft_min approaches the actual min
    _test(
        "soft_min at low temp close to min",
        abs(sm.item() - 0.1) < 0.05,
        f"expected ~0.1, got {sm.item():.4f}",
    )

    # ---- Test 19: RuleNetwork forward integrates everything ----
    print("\n--- Test 19: RuleNetwork forward full integration ---")
    config_full = RuleConfig(
        max_rules=16,
        use_attention=True,
        attention_dim=32,
        rule_embed_dim=16,
        constraint_weight=0.1,
        warmup_steps=0,
    )
    net_full = RuleNetwork(config_full)
    rules_full = [
        Rule(ast=LiteralNode("P", ["x"]), name="r1"),
        Rule(ast=LiteralNode("Q", ["x"]), name="r2"),
    ]
    truths_full = torch.tensor([[0.9, 0.3]] * B)
    ws_full = torch.randn(B, 32)
    out_full = net_full.forward(
        rules=rules_full,
        rule_truths=truths_full,
        workspace_summary=ws_full,
        return_logs=True,
    )
    _test(
        "Full forward constraint_loss is scalar",
        out_full.constraint_loss.dim() == 0,
        f"got dim {out_full.constraint_loss.dim()}",
    )
    _test(
        "Full forward violations shape",
        out_full.violations.shape == (B, 2),
        f"got {out_full.violations.shape}",
    )
    _test(
        "Full forward attention_weights shape",
        out_full.attention_weights is not None
        and out_full.attention_weights.shape == (B, 2),
        f"got {out_full.attention_weights.shape if out_full.attention_weights is not None else 'None'}",
    )
    _test(
        "Full forward warmup_weight",
        abs(out_full.warmup_weight - 0.1) < 1e-8,
        f"got {out_full.warmup_weight}",
    )
    # Gradient flows through constraint_loss
    out_full.constraint_loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in net_full.parameters())
    _test(
        "Gradient flows through constraint_loss",
        has_grad,
        "no parameter received gradients",
    )

    # ---- Test 20: evaluate_batch returns (B, R) ----
    print("\n--- Test 20: evaluate_batch returns correct shape ---")
    rules_batch = [
        Rule(ast=QuantifierNode("forall", "x", LiteralNode("P", ["x"])), name="r1"),
        Rule(ast=QuantifierNode("forall", "x", LiteralNode("Q", ["x"])), name="r2"),
        Rule(
            ast=QuantifierNode(
                "forall",
                "x",
                ConnectiveNode("implies", [
                    LiteralNode("P", ["x"]),
                    LiteralNode("Q", ["x"]),
                ]),
            ),
            name="r3",
        ),
    ]
    pred_batch = {
        "P": torch.rand(B, N),
        "Q": torch.rand(B, N),
    }
    batch_truths = evaluator.evaluate_batch(rules_batch, pred_batch, {})
    _test(
        "evaluate_batch shape",
        batch_truths.shape == (B, 3),
        f"got {batch_truths.shape}",
    )
    _test(
        "evaluate_batch values in [0, 1]",
        (batch_truths >= 0).all().item() and (batch_truths <= 1).all().item(),
        f"min={batch_truths.min().item():.4f}, max={batch_truths.max().item():.4f}",
    )

    # ---- Test 21: Godel bundle evaluation ----
    print("\n--- Test 21: Godel bundle evaluation ---")
    godel_eval = RuleEvaluator(GodelBundle())
    forall_godel = Rule(
        ast=QuantifierNode("forall", "x", LiteralNode("P", ["x"])),
        name="godel_forall",
    )
    godel_truths = {"P": torch.tensor([[0.9, 0.3, 0.7, 0.5, 0.8]] * B)}
    truth_godel = godel_eval.evaluate(forall_godel, godel_truths, {})
    _test(
        "Godel FORALL = min",
        abs(truth_godel[0].item() - 0.3) < 1e-5,
        f"expected 0.3, got {truth_godel[0].item():.4f}",
    )

    # ---- Summary ----
    print("\n" + "=" * 70)
    print(f"TOTAL: {num_passed} passed, {num_failed} failed out of "
          f"{num_passed + num_failed} checks")
    print("=" * 70)

    if num_failed > 0:
        print("\nFailed tests:")
        for name, passed, detail in test_results:
            if not passed:
                print(f"  - {name}: {detail}")
        sys.exit(1)
    else:
        print("\nAll tests passed.")


if __name__ == "__main__":
    _run_self_tests()
