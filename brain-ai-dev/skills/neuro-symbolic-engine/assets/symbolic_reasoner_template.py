"""
brain_ai/reasoning/symbolic.py -- SymbolicReasoner: Main Neuro-Symbolic Integration Module

Wires together the fuzzy operator bundles, grounding pipeline (entity extraction +
predicate/relation computation), and rule engine into a single differentiable module
that can serve as (a) a standalone constraint-loss regularizer and (b) a symbolic
feature provider for System 2 metacognitive routing.

Pipeline:
    workspace_slots  ->  EntityExtractor  ->  entities (B, N, D_ent)
                                                  |
                         PredicateRegistry  <-----+--->  RelationRegistry
                              |                              |
                         predicate_truths              relation_truths
                              |                              |
                              +------> RuleAssessor <--------+
                                            |
                                       (B, R) rule truths
                                            |
                                       RuleNetwork (attention + aggregation)
                                            |
                                     constraint_loss + violation_stats
                                            |
                         QueryAssessor  <---+--- optional queries
                              |
                        (B, Q) truth values

The SymbolicReasoner integrates with the broader BrainAI cognitive pipeline:
  - Upstream: receives slot representations from the Global Workspace module
  - Downstream: provides constraint_loss for training, truth features for System 2

Brain analog: this module corresponds to the prefrontal/parietal integration of
symbolic rule assessment over perceptual groundings -- the "logical consistency
checker" that grounds abstract rules in sensory evidence.

Hard invariants:
  - All truth values in [0, 1].
  - constraint_loss is differentiable: gradients flow through predicate/relation
    groundings into upstream entity extractors and workspace encoders.
  - Operator computations use fp32 regardless of AMP autocast state.
  - No NaN or Inf in any output tensor for well-formed inputs.

References:
    Badreddine et al. (2022) "Logic Tensor Networks", AI 303
    van Krieken et al. (2022) "Analyzing Differentiable Fuzzy Logic Operators", AI 302
    SKILL.md and references/fuzzy-operators.md for project conventions

Usage:
    >>> from brain_ai.reasoning.symbolic import (
    ...     SymbolicReasoner, SymbolicOutput, SymbolicLoss,
    ...     SymbolicFeatureExtractor, create_symbolic_reasoner,
    ... )
    >>> config = SymbolicFullConfig.minimal()
    >>> reasoner = create_symbolic_reasoner(config)
    >>> output = reasoner.forward_from_workspace(workspace_slots, rules)
    >>> print(output.constraint_loss)

Copy this template to brain_ai/reasoning/symbolic.py when integrating.
"""

from __future__ import annotations

import logging
import math
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ============================================================================
# SECTION 0: Import stubs for sibling template modules
# ============================================================================
# These imports refer to modules defined by the other neuro-symbolic-engine
# templates (fuzzy_operators_template.py, grounding_template.py,
# rule_engine_template.py, symbolic_config_template.py).  When running
# standalone (e.g., self-tests), we provide minimal stub implementations.
# ============================================================================

_SIBLING_MODULES_AVAILABLE = False

try:
    from .fuzzy_operators import (
        OperatorBundle,
        get_operator_bundle,
        forall_pMeanError,
        exists_pMean,
    )
    from .grounding import (
        EntityExtractor,
        PredicateRegistry,
        RelationRegistry,
        create_grounding_layer,
        GroundingLayer,
    )
    from .rule_engine import (
        ASTNode,
        ASTNodeType,
        Rule,
        RuleAssessor,
        RuleNetwork,
    )
    from .symbolic_config import (
        SymbolicFullConfig,
        OperatorConfig,
        GroundingConfig,
        RuleConfig,
        SymbolicConfig,
    )

    _SIBLING_MODULES_AVAILABLE = True
except ImportError:
    logger.debug(
        "Sibling neuro-symbolic modules not available; using built-in stubs."
    )


# ---------------------------------------------------------------------------
# Minimal stubs -- used only when sibling templates are not importable.
# These provide enough interface to run the self-tests at the bottom.
# ---------------------------------------------------------------------------

if not _SIBLING_MODULES_AVAILABLE:

    # -- ASTNode stubs -------------------------------------------------------

    class ASTNodeType(Enum):
        """Minimal AST node types for rule / query formulas."""
        PREDICATE = "predicate"
        RELATION = "relation"
        AND = "and"
        OR = "or"
        NOT = "not"
        IMPLIES = "implies"
        FORALL = "forall"
        EXISTS = "exists"
        LITERAL = "literal"

    @dataclass
    class ASTNode:
        """Minimal AST node stub for rule formulas."""
        node_type: ASTNodeType
        name: Optional[str] = None
        children: Optional[List["ASTNode"]] = None
        variable: Optional[str] = None

        def __repr__(self) -> str:
            return f"ASTNode({self.node_type.value}, name={self.name})"

    @dataclass
    class Rule:
        """Compiled rule: an AST root node with metadata."""
        name: str
        ast: ASTNode
        weight: float = 1.0

    # -- OperatorBundle stub -------------------------------------------------

    class OperatorBundle(NamedTuple):
        """Complete set of fuzzy operators."""
        AND: Callable[[Tensor, Tensor], Tensor]
        OR: Callable[[Tensor, Tensor], Tensor]
        NOT: Callable[[Tensor], Tensor]
        IMPLIES: Callable[[Tensor, Tensor], Tensor]
        FORALL: Callable[..., Tensor]
        EXISTS: Callable[..., Tensor]
        name: str

    def _stable_and(x: Tensor, y: Tensor, eps: float = 1e-4) -> Tensor:
        return x.clamp(min=eps, max=1.0) * y.clamp(min=eps, max=1.0)

    def _stable_or(x: Tensor, y: Tensor, eps: float = 1e-4) -> Tensor:
        nx = 1.0 - x.clamp(min=0.0, max=1.0 - eps)
        ny = 1.0 - y.clamp(min=0.0, max=1.0 - eps)
        return 1.0 - nx * ny

    def _stable_not(x: Tensor) -> Tensor:
        return 1.0 - x

    def _stable_implies(x: Tensor, y: Tensor, eps: float = 1e-4) -> Tensor:
        xp = x.clamp(min=eps, max=1.0)
        yp = y.clamp(min=0.0, max=1.0 - eps)
        return 1.0 - xp + xp * yp

    def forall_pMeanError(
        truth_values: Tensor,
        p: float = 2.0,
        dim: int = -1,
        eps: float = 1e-8,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Universal quantifier via pMeanError aggregation."""
        errors = 1.0 - truth_values
        if mask is not None:
            errors = errors.masked_fill(~mask, 0.0)
            n = mask.float().sum(dim=dim, keepdim=True).clamp(min=1.0)
        else:
            n = truth_values.shape[dim]
        mean_error_p = errors.pow(p).sum(dim=dim, keepdim=True) / n
        mean_error = mean_error_p.pow(1.0 / p).squeeze(dim)
        return (1.0 - mean_error).clamp(min=0.0, max=1.0)

    def exists_pMean(
        truth_values: Tensor,
        p: float = 6.0,
        dim: int = -1,
        eps: float = 1e-8,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Existential quantifier via generalized p-mean."""
        vals = truth_values.clamp(min=eps)
        if mask is not None:
            vals = vals.masked_fill(~mask, 0.0)
            n = mask.float().sum(dim=dim, keepdim=True).clamp(min=1.0)
        else:
            n = truth_values.shape[dim]
        mean_p = (vals.pow(p).sum(dim=dim, keepdim=True) / n).pow(1.0 / p)
        return mean_p.squeeze(dim).clamp(0.0, 1.0)

    def _stub_forall(v: Tensor, dim: int = -1) -> Tensor:
        return forall_pMeanError(v, p=2.0, dim=dim)

    def _stub_exists(v: Tensor, dim: int = -1) -> Tensor:
        return exists_pMean(v, p=6.0, dim=dim)

    _STABLE_PRODUCT_BUNDLE = OperatorBundle(
        AND=_stable_and,
        OR=_stable_or,
        NOT=_stable_not,
        IMPLIES=_stable_implies,
        FORALL=_stub_forall,
        EXISTS=_stub_exists,
        name="stable_product",
    )

    def get_operator_bundle(config: Any) -> "OperatorBundle":
        """Stub: always returns stable_product bundle."""
        return _STABLE_PRODUCT_BUNDLE

    # -- Config stubs --------------------------------------------------------

    @dataclass
    class OperatorConfig:
        bundle: str = "stable_product"
        eps: float = 1e-4
        quantifier_p: float = 2.0
        quantifier_temp: float = 1.0
        implication_type: str = "reichenbach"

    @dataclass
    class GroundingConfig:
        extractor_type: str = "slot_identity"
        max_entities: int = 32
        predicate_type: str = "mlp"
        relation_type: str = "bilinear"

    @dataclass
    class RuleConfig:
        max_rules: int = 64
        use_attention: bool = True
        violation_agg: str = "mean"
        constraint_weight: float = 0.1
        warmup_steps: int = 1000

    @dataclass
    class SymbolicConfig:
        entity_dim: int = 256
        num_predicates: int = 32
        num_relations: int = 16
        use_ltn: bool = False
        hidden_dim: int = 512

    @dataclass
    class SymbolicFullConfig:
        """Aggregated configuration for the full neuro-symbolic engine."""
        symbolic: SymbolicConfig = field(default_factory=SymbolicConfig)
        operator: OperatorConfig = field(default_factory=OperatorConfig)
        grounding: GroundingConfig = field(default_factory=GroundingConfig)
        rule: RuleConfig = field(default_factory=RuleConfig)
        workspace_dim: int = 512

        @classmethod
        def minimal(cls) -> "SymbolicFullConfig":
            """Minimal configuration for unit tests (~small param count)."""
            return cls(
                symbolic=SymbolicConfig(
                    entity_dim=64, num_predicates=4, num_relations=2,
                    hidden_dim=128,
                ),
                operator=OperatorConfig(),
                grounding=GroundingConfig(max_entities=8),
                rule=RuleConfig(max_rules=8),
                workspace_dim=128,
            )

        @classmethod
        def dev(cls) -> "SymbolicFullConfig":
            return cls()

        @classmethod
        def production_1b(cls) -> "SymbolicFullConfig":
            return cls(workspace_dim=2048)

        @classmethod
        def production_3b(cls) -> "SymbolicFullConfig":
            return cls(workspace_dim=4096)

        @classmethod
        def production_7b(cls) -> "SymbolicFullConfig":
            return cls(workspace_dim=4096)

    # -- PredicateRegistry / RelationRegistry stubs --------------------------

    class _MLPPredicate(nn.Module):
        """Stub MLP predicate: entity -> truth in [0,1]."""
        def __init__(self, entity_dim: int, hidden_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(entity_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )

        def forward(self, entities: Tensor) -> Tensor:
            return self.net(entities).squeeze(-1)

    class PredicateRegistry(nn.ModuleDict):
        """Stub predicate registry backed by nn.ModuleDict."""
        def __init__(self, entity_dim: int, hidden_dim: int):
            super().__init__()
            self.entity_dim = entity_dim
            self.hidden_dim = hidden_dim

        def register_predicate(
            self, name: str, module: Optional[nn.Module] = None
        ) -> None:
            if module is None:
                module = _MLPPredicate(self.entity_dim, self.hidden_dim)
            self[name] = module

        def assess(
            self, name: str, entities: Tensor, mask: Tensor
        ) -> Tensor:
            """Compute truth values for a named predicate, masking invalid entities."""
            truth = self[name](entities)
            truth = truth * mask.float()
            return truth

        def assess_all(
            self, entities: Tensor, mask: Tensor
        ) -> Dict[str, Tensor]:
            """Compute truth values for all registered predicates."""
            return {name: self.assess(name, entities, mask) for name in self}

    class _BilinearRelation(nn.Module):
        """Stub bilinear relation: (e_x, e_y) -> truth in [0,1]."""
        def __init__(self, entity_dim: int):
            super().__init__()
            self.W = nn.Parameter(torch.randn(entity_dim, entity_dim) * 0.02)
            self.bias = nn.Parameter(torch.zeros(1))

        def forward(self, e_x: Tensor, e_y: Tensor) -> Tensor:
            score = torch.einsum("...i,ij,...j->...", e_x, self.W, e_y) + self.bias
            return torch.sigmoid(score)

    class RelationRegistry(nn.ModuleDict):
        """Stub relation registry backed by nn.ModuleDict."""
        def __init__(
            self,
            entity_dim: int,
            hidden_dim: int,
            default_type: str = "bilinear",
        ):
            super().__init__()
            self.entity_dim = entity_dim
            self.hidden_dim = hidden_dim
            self.default_type = default_type

        def register_relation(
            self,
            name: str,
            module: Optional[nn.Module] = None,
            symmetric: bool = False,
        ) -> None:
            if module is None:
                module = _BilinearRelation(self.entity_dim)
            self[name] = module

        def assess(self, name: str, e_x: Tensor, e_y: Tensor) -> Tensor:
            """Compute truth value for a named relation on a pair."""
            return self[name](e_x, e_y)

        def assess_pairwise(
            self, name: str, entities: Tensor, mask: Tensor
        ) -> Tensor:
            """Compute truth matrix for all entity pairs. Returns (B, N, N)."""
            B, N, D = entities.shape
            e_x = entities.unsqueeze(2).expand(B, N, N, D)
            e_y = entities.unsqueeze(1).expand(B, N, N, D)
            truth_matrix = self[name](e_x, e_y)
            pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
            return truth_matrix * pair_mask.float()

    # -- EntityExtractor stub ------------------------------------------------

    class EntityExtractor(nn.Module):
        """Stub: project workspace slots to entity dim."""
        def __init__(self, workspace_dim: int, entity_dim: int):
            super().__init__()
            self.proj = nn.Linear(workspace_dim, entity_dim)
            self.norm = nn.LayerNorm(entity_dim)

        def forward(
            self, slots: Tensor, slot_mask: Optional[Tensor] = None
        ) -> Tuple[Tensor, Tensor]:
            entities = self.norm(self.proj(slots))
            if slot_mask is None:
                slot_mask = torch.ones(
                    slots.shape[0], slots.shape[1],
                    dtype=torch.bool, device=slots.device,
                )
            entities = entities * slot_mask.unsqueeze(-1).float()
            return entities, slot_mask

    # -- GroundingLayer stub -------------------------------------------------

    class GroundingLayer(nn.Module):
        """Stub grounding layer wrapping extractor + registries."""
        def __init__(
            self,
            workspace_dim: int,
            entity_dim: int,
            hidden_dim: int,
            num_predicates: int = 4,
            num_relations: int = 2,
        ):
            super().__init__()
            self.extractor = EntityExtractor(workspace_dim, entity_dim)
            self.predicates = PredicateRegistry(entity_dim, hidden_dim)
            self.relations = RelationRegistry(entity_dim, hidden_dim)
            for i in range(num_predicates):
                self.predicates.register_predicate(f"pred_{i}")
            for i in range(num_relations):
                self.relations.register_relation(f"rel_{i}")

        def extract_entities(
            self, slots: Tensor, slot_mask: Optional[Tensor] = None
        ) -> Tuple[Tensor, Tensor]:
            return self.extractor(slots, slot_mask)

    def create_grounding_layer(config: Any) -> "GroundingLayer":
        """Stub factory for grounding layer."""
        ws_dim = getattr(config, "workspace_dim", 128)
        ent_dim = getattr(config, "entity_dim", 64)
        hid_dim = getattr(config, "hidden_dim", 128)
        n_pred = getattr(config, "num_predicates", 4)
        n_rel = getattr(config, "num_relations", 2)
        return GroundingLayer(ws_dim, ent_dim, hid_dim, n_pred, n_rel)

    # -- RuleAssessor stub ---------------------------------------------------

    class RuleAssessor:
        """Stub: assess rule ASTs against predicate/relation truth dicts.

        Recursively walks AST nodes and applies the configured fuzzy operators
        to compute a single scalar truth value per rule per batch element.
        """
        def __init__(self, ops: OperatorBundle):
            self.ops = ops

        def assess_ast(
            self,
            node: ASTNode,
            predicate_truths: Dict[str, Tensor],
            relation_truths: Dict[str, Tensor],
            entity_mask: Optional[Tensor] = None,
        ) -> Tensor:
            """Recursively assess an AST node. Returns (B,) or (B, N) truth."""
            if node.node_type == ASTNodeType.PREDICATE:
                name = node.name or "pred_0"
                if name in predicate_truths:
                    t = predicate_truths[name]
                    if entity_mask is not None and t.dim() > 1:
                        return self.ops.FORALL(t * entity_mask.float(), dim=-1)
                    elif t.dim() > 1:
                        return self.ops.FORALL(t, dim=-1)
                    return t
                return torch.ones(1)

            elif node.node_type == ASTNodeType.RELATION:
                name = node.name or "rel_0"
                if name in relation_truths:
                    t = relation_truths[name]
                    while t.dim() > 1:
                        t = self.ops.FORALL(t, dim=-1)
                    return t
                return torch.ones(1)

            elif node.node_type == ASTNodeType.AND:
                children = node.children or []
                if len(children) < 2:
                    return torch.ones(1)
                left = self.assess_ast(
                    children[0], predicate_truths, relation_truths, entity_mask
                )
                right = self.assess_ast(
                    children[1], predicate_truths, relation_truths, entity_mask
                )
                return self.ops.AND(left, right)

            elif node.node_type == ASTNodeType.OR:
                children = node.children or []
                if len(children) < 2:
                    return torch.ones(1)
                left = self.assess_ast(
                    children[0], predicate_truths, relation_truths, entity_mask
                )
                right = self.assess_ast(
                    children[1], predicate_truths, relation_truths, entity_mask
                )
                return self.ops.OR(left, right)

            elif node.node_type == ASTNodeType.NOT:
                children = node.children or []
                if len(children) < 1:
                    return torch.zeros(1)
                child = self.assess_ast(
                    children[0], predicate_truths, relation_truths, entity_mask
                )
                return self.ops.NOT(child)

            elif node.node_type == ASTNodeType.IMPLIES:
                children = node.children or []
                if len(children) < 2:
                    return torch.ones(1)
                ante = self.assess_ast(
                    children[0], predicate_truths, relation_truths, entity_mask
                )
                cons = self.assess_ast(
                    children[1], predicate_truths, relation_truths, entity_mask
                )
                return self.ops.IMPLIES(ante, cons)

            elif node.node_type == ASTNodeType.FORALL:
                children = node.children or []
                if len(children) < 1:
                    return torch.ones(1)
                child = self.assess_ast(
                    children[0], predicate_truths, relation_truths, entity_mask
                )
                if child.dim() > 1:
                    return self.ops.FORALL(child, dim=-1)
                return child

            elif node.node_type == ASTNodeType.EXISTS:
                children = node.children or []
                if len(children) < 1:
                    return torch.zeros(1)
                child = self.assess_ast(
                    children[0], predicate_truths, relation_truths, entity_mask
                )
                if child.dim() > 1:
                    return self.ops.EXISTS(child, dim=-1)
                return child

            elif node.node_type == ASTNodeType.LITERAL:
                val = 1.0 if node.name != "false" else 0.0
                return torch.tensor(val)

            else:
                return torch.ones(1)

        def assess_rules(
            self,
            rules: List[Rule],
            predicate_truths: Dict[str, Tensor],
            relation_truths: Dict[str, Tensor],
            entity_mask: Optional[Tensor] = None,
        ) -> Tensor:
            """Assess all rules. Returns (B, R) truth values."""
            if not rules:
                return torch.ones(1, 1)
            truths = []
            for rule in rules:
                t = self.assess_ast(
                    rule.ast, predicate_truths, relation_truths, entity_mask
                )
                if t.dim() == 0:
                    t = t.unsqueeze(0)
                truths.append(t)
            # Stack and ensure (B, R) shape
            stacked = torch.stack(truths, dim=-1)
            if stacked.dim() == 1:
                stacked = stacked.unsqueeze(0)
            return stacked

    # -- RuleNetwork stub ----------------------------------------------------

    class RuleNetwork(nn.Module):
        """Stub rule network: attention + constraint loss from rule truths."""
        def __init__(self, config: RuleConfig):
            super().__init__()
            self.config = config
            self._step: int = 0
            self.violation_agg = config.violation_agg
            self.constraint_weight = config.constraint_weight
            self.warmup_steps = config.warmup_steps
            self.use_attention = config.use_attention
            if self.use_attention:
                self.rule_attn_proj = nn.Linear(128, config.max_rules)

        @property
        def warmup_factor(self) -> float:
            """Current warmup multiplier in [0, 1]."""
            if self.warmup_steps <= 0:
                return 1.0
            return min(1.0, self._step / self.warmup_steps)

        def compute_constraint_loss(
            self,
            rule_truths: Tensor,
            rule_weights: Optional[Tensor] = None,
            workspace_summary: Optional[Tensor] = None,
        ) -> Tuple[Tensor, Dict[str, Tensor]]:
            """Compute constraint loss from rule truth values.

            Args:
                rule_truths: (B, R) truth values for each rule.
                rule_weights: (R,) per-rule static weights (from Rule.weight).
                workspace_summary: (B, D) optional context for attention.

            Returns:
                constraint_loss: scalar differentiable tensor.
                violation_stats: per-rule violation rates.
            """
            # Violations = 1 - truth
            violations = 1.0 - rule_truths  # (B, R)
            B, R = violations.shape

            # Optional attention weighting
            if self.use_attention and workspace_summary is not None:
                if hasattr(self, "rule_attn_proj"):
                    logits = self.rule_attn_proj(workspace_summary)  # (B, max_rules)
                    logits = logits[:, :R]
                    alpha = F.softmax(logits, dim=-1)  # (B, R)
                    violations = violations * alpha

            # Apply static rule weights
            if rule_weights is not None:
                violations = violations * rule_weights.unsqueeze(0)

            # Aggregate violations
            if self.violation_agg == "mean":
                per_sample = violations.mean(dim=-1)  # (B,)
            elif self.violation_agg == "soft_min":
                # soft-min: focus on worst violation via max
                per_sample = violations.max(dim=-1)[0]
            elif self.violation_agg == "p_mean":
                p = 2.0
                per_sample = (violations.pow(p).mean(dim=-1)).pow(1.0 / p)
            else:
                per_sample = violations.mean(dim=-1)

            constraint_loss = per_sample.mean()

            # Violation stats
            violation_stats: Dict[str, Tensor] = {
                "mean_violation": violations.mean(dim=0).detach(),
                "max_violation": violations.max(dim=0)[0].detach(),
                "per_sample_loss": per_sample.detach(),
            }

            return constraint_loss, violation_stats


# ============================================================================
# SECTION 1: SymbolicOutput dataclass
# ============================================================================


@dataclass
class SymbolicOutput:
    """Output produced by the SymbolicReasoner.

    Attributes:
        truth: Query truth values of shape ``(B, Q)`` in [0, 1]. None if no
            query was provided.
        constraint_loss: Scalar (or ``(R,)``) differentiable loss computed
            from rule violations.  Always requires_grad when at least one
            predicate/relation module has trainable parameters.
        violation_stats: Per-rule / per-clause violation statistics. Keys
            include ``"mean_violation"`` (shape ``(R,)``), ``"max_violation"``
            (shape ``(R,)``), and ``"per_sample_loss"`` (shape ``(B,)``).
        logs: Interpretable diagnostic information (rule attention weights,
            top violations, proof trace). None when ``return_logs=False``.
    """

    truth: Optional[Tensor]
    constraint_loss: Tensor
    violation_stats: Dict[str, Tensor]
    logs: Optional[Dict[str, Any]]

    def __post_init__(self) -> None:
        """Validate output invariants."""
        if self.truth is not None:
            assert self.truth.dim() >= 1, (
                f"truth must be at least 1-D, got {self.truth.dim()}-D"
            )
        assert self.constraint_loss.dim() <= 1, (
            f"constraint_loss must be scalar or 1-D, got {self.constraint_loss.dim()}-D"
        )


# ============================================================================
# SECTION 2: QueryAssessor
# ============================================================================


class QueryAssessor:
    """Assess query formulas given predicate/relation truth dictionaries.

    The query assessor traverses query ASTs and computes truth values using
    the configured operator bundle.  It does not own any trainable parameters;
    all parameters reside in the predicate/relation modules that produced the
    truth dictionaries.

    Args:
        operator_bundle: Complete set of fuzzy operators for connective
            computation (AND, OR, NOT, IMPLIES, FORALL, EXISTS).
    """

    def __init__(self, operator_bundle: OperatorBundle) -> None:
        self.ops = operator_bundle
        self._rule_assessor = RuleAssessor(operator_bundle)

    def run(
        self,
        queries: List[ASTNode],
        predicate_truths: Dict[str, Tensor],
        relation_truths: Dict[str, Tensor],
        entity_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Run query ASTs against grounded truth dictionaries.

        Args:
            queries: List of Q query formula ASTs.
            predicate_truths: ``{pred_name: (B, N) truth}`` from predicate
                computation.
            relation_truths: ``{rel_name: (B, N, N) truth}`` from pairwise
                relation computation.
            entity_mask: ``(B, N)`` boolean mask for valid entities.

        Returns:
            Tensor of shape ``(B, Q)`` with truth values in [0, 1].
        """
        if not queries:
            raise ValueError("QueryAssessor.run requires at least one query.")

        results: List[Tensor] = []
        for q_ast in queries:
            t = self._rule_assessor.assess_ast(
                q_ast, predicate_truths, relation_truths, entity_mask
            )
            if t.dim() == 0:
                t = t.unsqueeze(0)
            results.append(t)

        # Stack along query dimension -> (B, Q) or (Q,) then broadcast
        stacked = torch.stack(results, dim=-1)
        if stacked.dim() == 1:
            stacked = stacked.unsqueeze(0)
        return stacked.clamp(0.0, 1.0)


# ============================================================================
# SECTION 3: SymbolicReasoner (main module)
# ============================================================================


class SymbolicReasoner(nn.Module):
    """Main neuro-symbolic engine: assesses predicates/rules/queries with
    fuzzy semantics.

    Usable as:
      (a) **Standalone loss module**: ``constraint_loss`` for regularization
          alongside task-specific losses.
      (b) **Feature module**: ``truth`` values feed System 2 as a symbolic
          consistency score for metacognitive routing.

    The reasoner does not maintain internal state across forward calls;
    it is a pure function of (entities, predicates, rules, query).

    Args:
        config: Full neuro-symbolic engine configuration aggregating
            symbolic, operator, grounding, and rule sub-configs.
    """

    def __init__(self, config: SymbolicFullConfig) -> None:
        super().__init__()
        self.config = config

        # 1. Operator bundle from config
        self.ops: OperatorBundle = get_operator_bundle(config.operator)

        # 2. Grounding layer (entity extraction + predicate/relation modules)
        grounding_cfg = _build_grounding_cfg(config)
        self.grounding: GroundingLayer = create_grounding_layer(grounding_cfg)

        # 3. Rule assessor (uses operator bundle, no trainable params)
        self.rule_assessor = RuleAssessor(self.ops)

        # 4. Rule network (attention + constraint loss computation)
        self.rule_network = RuleNetwork(config.rule)

        # 5. Query assessor (for entailment tasks)
        self.query_assessor = QueryAssessor(self.ops)

        logger.info(
            "SymbolicReasoner initialized: bundle=%s, entity_dim=%d, "
            "max_rules=%d, warmup=%d",
            self.ops.name,
            config.symbolic.entity_dim,
            config.rule.max_rules,
            config.rule.warmup_steps,
        )

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def forward(
        self,
        entities: Tensor,
        predicates: PredicateRegistry,
        rules: List[Rule],
        *,
        relations: Optional[RelationRegistry] = None,
        entity_mask: Optional[Tensor] = None,
        query: Optional[List[ASTNode]] = None,
        return_logs: bool = False,
        workspace_summary: Optional[Tensor] = None,
    ) -> SymbolicOutput:
        """Full forward pass: compute predicates, rules, and optional queries.

        All operator computations are performed in fp32 regardless of the
        global AMP autocast state to preserve truth value precision.

        Args:
            entities: ``(B, N, D_ent)`` entity embeddings.
            predicates: Registry of named predicate modules.
            rules: Compiled rule ASTs for truth assessment.
            relations: Optional registry of named relation modules. If None,
                uses ``self.grounding.relations``.
            entity_mask: ``(B, N)`` boolean mask; True for valid entities.
            query: Optional list of query formula ASTs for truth estimation.
            return_logs: Whether to collect interpretable diagnostic logs.
            workspace_summary: ``(B, D)`` summary vector from global workspace,
                used for attention-weighted rule application.

        Returns:
            SymbolicOutput with truth, constraint_loss, violation_stats, logs.
        """
        if relations is None:
            relations = self.grounding.relations

        B, N, D_ent = entities.shape
        device = entities.device

        # Build entity mask if not provided
        if entity_mask is None:
            entity_mask = torch.ones(B, N, dtype=torch.bool, device=device)

        # Enforce fp32 for all fuzzy operator computations
        with torch.cuda.amp.autocast(enabled=False):
            entities_fp32 = entities.float()

            # Step 1: Compute all predicate truths on entities
            predicate_truths: Dict[str, Tensor] = predicates.assess_all(
                entities_fp32, entity_mask
            )

            # Step 2: Compute all relation truths on entity pairs
            relation_truths: Dict[str, Tensor] = {}
            for rel_name in relations:
                relation_truths[rel_name] = relations.assess_pairwise(
                    rel_name, entities_fp32, entity_mask
                )

            # Step 3: Assess each rule AST -> (B, R) truth values
            rule_truths = self.rule_assessor.assess_rules(
                rules, predicate_truths, relation_truths, entity_mask
            )
            # Ensure (B, R) shape
            if rule_truths.dim() == 1:
                rule_truths = rule_truths.unsqueeze(0).expand(B, -1)
            elif rule_truths.shape[0] != B and rule_truths.shape[0] == 1:
                rule_truths = rule_truths.expand(B, -1)

            # Step 4: Compute constraint loss via rule_network
            rule_weights = torch.tensor(
                [r.weight for r in rules], device=device, dtype=torch.float32
            ) if rules else None

            ws_summary_fp32 = (
                workspace_summary.float() if workspace_summary is not None else None
            )

            constraint_loss, violation_stats = self.rule_network.compute_constraint_loss(
                rule_truths,
                rule_weights=rule_weights,
                workspace_summary=ws_summary_fp32,
            )

            # Step 5: If query provided, run query ASTs -> (B, Q)
            truth: Optional[Tensor] = None
            if query is not None and len(query) > 0:
                truth = self.query_assessor.run(
                    query, predicate_truths, relation_truths, entity_mask
                )
                # Expand to batch if needed
                if truth.shape[0] != B and truth.shape[0] == 1:
                    truth = truth.expand(B, -1)

        # Step 6: violation_stats is already populated above

        # Step 7: Collect interpretable logs if requested
        logs: Optional[Dict[str, Any]] = None
        if return_logs:
            logs = self._build_logs(
                rule_truths=rule_truths,
                violation_stats=violation_stats,
                predicate_truths=predicate_truths,
                relation_truths=relation_truths,
                rules=rules,
                workspace_summary=ws_summary_fp32,
            )

        return SymbolicOutput(
            truth=truth,
            constraint_loss=constraint_loss,
            violation_stats=violation_stats,
            logs=logs,
        )

    # ------------------------------------------------------------------
    # Convenience: forward from workspace slots
    # ------------------------------------------------------------------

    def forward_from_workspace(
        self,
        workspace_slots: Tensor,
        rules: List[Rule],
        *,
        query: Optional[List[ASTNode]] = None,
        return_logs: bool = False,
        slot_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
    ) -> SymbolicOutput:
        """Extract entities from workspace, then run forward.

        This is the primary entry point when operating within the full
        BrainAI pipeline, where workspace slot representations are the
        upstream input.

        Args:
            workspace_slots: ``(B, K, D_ws)`` raw workspace representation.
            rules: Compiled rule ASTs.
            query: Optional list of query formula ASTs.
            return_logs: Toggle interpretable diagnostic logs.
            slot_mask: ``(B, K)`` boolean mask for valid workspace slots.
            context: ``(B, D)`` optional context vector (workspace summary).

        Returns:
            SymbolicOutput with truth, constraint_loss, violation_stats, logs.
        """
        # Extract entities via grounding layer
        entities, entity_mask = self.grounding.extract_entities(
            workspace_slots, slot_mask
        )

        # Use grounding's internal registries
        predicates = self.grounding.predicates
        relations = self.grounding.relations

        # Compute workspace summary if not provided
        workspace_summary = context
        if workspace_summary is None and workspace_slots is not None:
            # Mean-pool valid slots as a simple summary
            if slot_mask is not None:
                ws_mask = slot_mask.unsqueeze(-1).float()
                workspace_summary = (
                    (workspace_slots * ws_mask).sum(dim=1)
                    / ws_mask.sum(dim=1).clamp(min=1.0)
                )
            else:
                workspace_summary = workspace_slots.mean(dim=1)

        return self.forward(
            entities=entities,
            predicates=predicates,
            rules=rules,
            relations=relations,
            entity_mask=entity_mask,
            query=query,
            return_logs=return_logs,
            workspace_summary=workspace_summary,
        )

    # ------------------------------------------------------------------
    # Convenience: loss-only and feature-only interfaces
    # ------------------------------------------------------------------

    def as_loss(
        self,
        workspace_slots: Tensor,
        rules: List[Rule],
        slot_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Return only the scalar constraint_loss for regularization.

        This is the cheapest interface: no query assessment, no logs.

        Args:
            workspace_slots: ``(B, K, D_ws)`` workspace representation.
            rules: Compiled rule ASTs.
            slot_mask: ``(B, K)`` boolean mask for valid slots.

        Returns:
            Scalar differentiable constraint loss tensor.
        """
        output = self.forward_from_workspace(
            workspace_slots, rules, slot_mask=slot_mask
        )
        return output.constraint_loss

    def as_feature(
        self,
        workspace_slots: Tensor,
        rules: List[Rule],
        query: List[ASTNode],
        slot_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Return query truth values as features for System 2.

        Args:
            workspace_slots: ``(B, K, D_ws)`` workspace representation.
            rules: Compiled rule ASTs.
            query: List of query formula ASTs.
            slot_mask: ``(B, K)`` boolean mask for valid slots.

        Returns:
            ``(B, Q)`` truth value tensor in [0, 1].
        """
        output = self.forward_from_workspace(
            workspace_slots, rules, query=query, slot_mask=slot_mask
        )
        if output.truth is None:
            raise RuntimeError(
                "as_feature() requires non-empty query list, but truth is None."
            )
        return output.truth

    # ------------------------------------------------------------------
    # Step counter for warmup scheduling
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Increment warmup step counter in rule network.

        Call once per training step (not per batch element) to advance the
        warmup schedule for the constraint loss weight.
        """
        self.rule_network._step += 1

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_logs(
        self,
        rule_truths: Tensor,
        violation_stats: Dict[str, Tensor],
        predicate_truths: Dict[str, Tensor],
        relation_truths: Dict[str, Tensor],
        rules: List[Rule],
        workspace_summary: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        """Collect interpretable diagnostic information.

        Returns a dictionary with:
          - ``attention_weights``: rule attention weights (B, R) if available.
          - ``violations``: per-rule violation info.
          - ``rule_truths``: raw (B, R) rule truth values.
          - ``top_violations``: indices of top-3 violated rules per sample.
          - ``predicate_summary``: mean truth per predicate.
          - ``relation_summary``: mean truth per relation.
        """
        logs: Dict[str, Any] = {}

        # Rule truths
        logs["rule_truths"] = rule_truths.detach()
        logs["violations"] = violation_stats

        # Attention weights
        if (
            self.rule_network.use_attention
            and workspace_summary is not None
            and hasattr(self.rule_network, "rule_attn_proj")
        ):
            R = rule_truths.shape[-1]
            logits = self.rule_network.rule_attn_proj(workspace_summary)
            logits = logits[:, :R]
            attention_weights = F.softmax(logits, dim=-1)
            logs["attention_weights"] = attention_weights.detach()
        else:
            logs["attention_weights"] = None

        # Top violations
        violations = 1.0 - rule_truths
        top_k = min(3, violations.shape[-1])
        top_vals, top_idx = violations.topk(top_k, dim=-1)
        logs["top_violations"] = {
            "indices": top_idx.detach(),
            "values": top_vals.detach(),
        }

        # Predicate summary
        pred_summary: Dict[str, float] = {}
        for name, t in predicate_truths.items():
            pred_summary[name] = t.mean().item()
        logs["predicate_summary"] = pred_summary

        # Relation summary
        rel_summary: Dict[str, float] = {}
        for name, t in relation_truths.items():
            rel_summary[name] = t.mean().item()
        logs["relation_summary"] = rel_summary

        return logs


# ============================================================================
# SECTION 4: SymbolicLoss wrapper
# ============================================================================


class SymbolicLoss(nn.Module):
    """Wrapper for using SymbolicReasoner as a loss term in training.

    Handles:
      - **Warmup scheduling**: linearly ramps constraint weight from 0 to
        ``weight`` over ``warmup_steps`` training steps.
      - **Loss weighting**: scales constraint loss by ``weight`` (lambda).
      - **Gradient detach options**: for ablation studies, can optionally
        detach constraint_loss gradients.

    The total loss is::

        L_total = task_loss + warmup_weight(step) * weight * constraint_loss

    Args:
        reasoner: SymbolicReasoner instance.
        weight: Maximum weight (lambda) for the constraint loss.
        warmup_steps: Number of training steps for linear warmup.
        detach_constraint: If True, detach constraint_loss from the graph
            (useful for ablation: measure violation without back-propagating
            through symbolic modules).
    """

    def __init__(
        self,
        reasoner: SymbolicReasoner,
        weight: float = 0.1,
        warmup_steps: int = 1000,
        detach_constraint: bool = False,
    ) -> None:
        super().__init__()
        self.reasoner = reasoner
        self.weight = weight
        self.warmup_steps = warmup_steps
        self.detach_constraint = detach_constraint
        self._step: int = 0

    @property
    def warmup_factor(self) -> float:
        """Current warmup multiplier in [0, 1]."""
        if self.warmup_steps <= 0:
            return 1.0
        return min(1.0, self._step / self.warmup_steps)

    @property
    def effective_weight(self) -> float:
        """Current effective constraint weight after warmup scaling."""
        return self.warmup_factor * self.weight

    def forward(
        self,
        workspace_slots: Tensor,
        rules: List[Rule],
        task_loss: Tensor,
        slot_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute total loss: task_loss + warmup_weight * constraint_loss.

        Args:
            workspace_slots: ``(B, K, D_ws)`` workspace representation.
            rules: Compiled rule ASTs.
            task_loss: Scalar task-specific loss (e.g., cross-entropy).
            slot_mask: ``(B, K)`` boolean mask for valid workspace slots.

        Returns:
            Scalar total loss tensor.
        """
        constraint_loss = self.reasoner.as_loss(
            workspace_slots, rules, slot_mask=slot_mask
        )

        if self.detach_constraint:
            constraint_loss = constraint_loss.detach()

        w = self.effective_weight
        total_loss = task_loss + w * constraint_loss

        return total_loss

    def step(self) -> None:
        """Increment training step counter for warmup scheduling.

        Also increments the internal reasoner step counter.
        """
        self._step += 1
        self.reasoner.step()


# ============================================================================
# SECTION 5: SymbolicFeatureExtractor wrapper
# ============================================================================


class SymbolicFeatureExtractor(nn.Module):
    """Wrapper for using SymbolicReasoner as a feature extractor for System 2.

    Produces a symbolic consistency vector that can be concatenated with other
    features for metacognitive routing or System 2 iterative refinement.

    The output features are:
      - ``truth``: query truth values ``(B, Q)``
      - ``violation_summary``: per-sample aggregated violation ``(B, 1)``
      - Concatenated: ``(B, Q + 1)`` total feature dimension

    Args:
        reasoner: SymbolicReasoner instance.
        project_dim: If > 0, project the concatenated features through a
            learned linear layer to this dimension.
    """

    def __init__(
        self,
        reasoner: SymbolicReasoner,
        project_dim: int = 0,
    ) -> None:
        super().__init__()
        self.reasoner = reasoner
        self.project_dim = project_dim
        self._proj: Optional[nn.Linear] = None
        # Projection will be lazily initialized on first forward

    def forward(
        self,
        workspace_slots: Tensor,
        rules: List[Rule],
        query: List[ASTNode],
        slot_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Extract symbolic features from workspace.

        Args:
            workspace_slots: ``(B, K, D_ws)`` workspace representation.
            rules: Compiled rule ASTs.
            query: List of query formula ASTs (determines Q dimension).
            slot_mask: ``(B, K)`` boolean mask for valid workspace slots.

        Returns:
            ``(B, feature_dim)`` symbolic feature tensor, where
            ``feature_dim = Q + 1`` (or ``project_dim`` if projection enabled).
        """
        output = self.reasoner.forward_from_workspace(
            workspace_slots, rules, query=query, slot_mask=slot_mask
        )

        if output.truth is None:
            raise RuntimeError(
                "SymbolicFeatureExtractor requires non-empty query list."
            )

        # Truth values: (B, Q)
        truth = output.truth

        # Violation summary: (B, 1)
        violation_summary = output.violation_stats["per_sample_loss"].unsqueeze(-1)

        # Concatenate: (B, Q + 1)
        features = torch.cat([truth, violation_summary], dim=-1)

        # Optional projection
        if self.project_dim > 0:
            if self._proj is None or self._proj.in_features != features.shape[-1]:
                self._proj = nn.Linear(
                    features.shape[-1], self.project_dim
                ).to(features.device)
            features = self._proj(features)

        return features


# ============================================================================
# SECTION 6: Factory functions
# ============================================================================


def _build_grounding_cfg(config: SymbolicFullConfig) -> Any:
    """Build a grounding-layer-compatible config object from SymbolicFullConfig.

    The grounding layer factory expects a config with workspace_dim, entity_dim,
    hidden_dim, num_predicates, and num_relations.  We construct a simple
    namespace for this purpose.
    """

    class _GroundingCfg:
        pass

    cfg = _GroundingCfg()
    cfg.workspace_dim = config.workspace_dim  # type: ignore[attr-defined]
    cfg.entity_dim = config.symbolic.entity_dim  # type: ignore[attr-defined]
    cfg.hidden_dim = config.symbolic.hidden_dim  # type: ignore[attr-defined]
    cfg.num_predicates = config.symbolic.num_predicates  # type: ignore[attr-defined]
    cfg.num_relations = config.symbolic.num_relations  # type: ignore[attr-defined]
    cfg.extractor_type = config.grounding.extractor_type  # type: ignore[attr-defined]
    cfg.max_entities = config.grounding.max_entities  # type: ignore[attr-defined]
    cfg.predicate_type = config.grounding.predicate_type  # type: ignore[attr-defined]
    cfg.relation_type = config.grounding.relation_type  # type: ignore[attr-defined]
    return cfg


def create_symbolic_reasoner(config: SymbolicFullConfig) -> SymbolicReasoner:
    """Factory: build SymbolicReasoner from config.

    Args:
        config: Full neuro-symbolic engine configuration.

    Returns:
        Initialized SymbolicReasoner module.

    Example::

        config = SymbolicFullConfig.minimal()
        reasoner = create_symbolic_reasoner(config)
    """
    return SymbolicReasoner(config)


def create_symbolic_loss(
    config: SymbolicFullConfig,
    weight: float = 0.1,
    warmup_steps: int = 1000,
) -> SymbolicLoss:
    """Factory: build SymbolicLoss wrapper from config.

    Args:
        config: Full neuro-symbolic engine configuration.
        weight: Maximum constraint loss weight (lambda).
        warmup_steps: Linear warmup steps.

    Returns:
        SymbolicLoss module wrapping a SymbolicReasoner.
    """
    reasoner = create_symbolic_reasoner(config)
    return SymbolicLoss(reasoner, weight=weight, warmup_steps=warmup_steps)


def create_symbolic_feature_extractor(
    config: SymbolicFullConfig,
    project_dim: int = 0,
) -> SymbolicFeatureExtractor:
    """Factory: build SymbolicFeatureExtractor wrapper from config.

    Args:
        config: Full neuro-symbolic engine configuration.
        project_dim: If > 0, project concatenated features to this dimension.

    Returns:
        SymbolicFeatureExtractor module wrapping a SymbolicReasoner.
    """
    reasoner = create_symbolic_reasoner(config)
    return SymbolicFeatureExtractor(reasoner, project_dim=project_dim)


# ============================================================================
# SECTION 7: Self-tests
# ============================================================================


def _make_test_config() -> SymbolicFullConfig:
    """Build a minimal config for self-tests."""
    return SymbolicFullConfig.minimal()


def _make_test_rules() -> List[Rule]:
    """Build a small set of test rules."""
    # Rule 1: pred_0(x) AND pred_1(x) -- conjunction of two predicates
    rule1 = Rule(
        name="rule_1",
        ast=ASTNode(
            node_type=ASTNodeType.AND,
            children=[
                ASTNode(node_type=ASTNodeType.PREDICATE, name="pred_0"),
                ASTNode(node_type=ASTNodeType.PREDICATE, name="pred_1"),
            ],
        ),
        weight=1.0,
    )
    # Rule 2: pred_0(x) IMPLIES pred_2(x)
    rule2 = Rule(
        name="rule_2",
        ast=ASTNode(
            node_type=ASTNodeType.IMPLIES,
            children=[
                ASTNode(node_type=ASTNodeType.PREDICATE, name="pred_0"),
                ASTNode(node_type=ASTNodeType.PREDICATE, name="pred_2"),
            ],
        ),
        weight=1.0,
    )
    # Rule 3: NOT pred_3(x) OR pred_1(x)
    rule3 = Rule(
        name="rule_3",
        ast=ASTNode(
            node_type=ASTNodeType.OR,
            children=[
                ASTNode(
                    node_type=ASTNodeType.NOT,
                    children=[
                        ASTNode(node_type=ASTNodeType.PREDICATE, name="pred_3"),
                    ],
                ),
                ASTNode(node_type=ASTNodeType.PREDICATE, name="pred_1"),
            ],
        ),
        weight=0.5,
    )
    return [rule1, rule2, rule3]


def _make_test_queries() -> List[ASTNode]:
    """Build test query ASTs."""
    q1 = ASTNode(node_type=ASTNodeType.PREDICATE, name="pred_0")
    q2 = ASTNode(
        node_type=ASTNodeType.AND,
        children=[
            ASTNode(node_type=ASTNodeType.PREDICATE, name="pred_1"),
            ASTNode(node_type=ASTNodeType.PREDICATE, name="pred_2"),
        ],
    )
    return [q1, q2]


def _run_self_tests() -> None:
    """Run comprehensive self-tests. Prints PASS/FAIL, exits 1 on failure."""
    import sys
    import traceback

    passed = 0
    failed = 0

    def check(test_name: str, condition: bool, detail: str = "") -> None:
        nonlocal passed, failed
        if condition:
            print(f"  PASS: {test_name}")
            passed += 1
        else:
            msg = f"  FAIL: {test_name}"
            if detail:
                msg += f" -- {detail}"
            print(msg)
            failed += 1

    print("=" * 72)
    print("SymbolicReasoner Self-Tests")
    print("=" * 72)

    torch.manual_seed(42)
    device = torch.device("cpu")
    config = _make_test_config()
    rules = _make_test_rules()
    queries = _make_test_queries()

    B, K = 4, 8
    D_ws = config.workspace_dim
    workspace_slots = torch.randn(B, K, D_ws, device=device)
    slot_mask = torch.ones(B, K, dtype=torch.bool, device=device)

    try:
        reasoner = create_symbolic_reasoner(config)
        reasoner.to(device)
    except Exception as e:
        print(f"  FATAL: Failed to create SymbolicReasoner: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ---- Test 1: SymbolicOutput fields present and correct types -----------
    print("\nTest 1: SymbolicOutput fields present and correct types")
    try:
        output = reasoner.forward_from_workspace(workspace_slots, rules)
        has_truth = hasattr(output, "truth")
        has_cl = hasattr(output, "constraint_loss")
        has_vs = hasattr(output, "violation_stats")
        has_logs = hasattr(output, "logs")
        types_ok = (
            isinstance(output.constraint_loss, Tensor)
            and isinstance(output.violation_stats, dict)
        )
        check(
            "SymbolicOutput fields and types",
            has_truth and has_cl and has_vs and has_logs and types_ok,
            f"truth={has_truth}, cl={has_cl}, vs={has_vs}, logs={has_logs}, types={types_ok}",
        )
    except Exception as e:
        check("SymbolicOutput fields and types", False, str(e))

    # ---- Test 2: Forward pass with entities + predicates + rules (no query)
    print("\nTest 2: Forward pass with entities + rules (no query)")
    try:
        output = reasoner.forward_from_workspace(workspace_slots, rules)
        cl_ok = output.constraint_loss.dim() == 0  # scalar
        truth_none = output.truth is None
        vs_has_keys = "mean_violation" in output.violation_stats
        check(
            "Forward no-query",
            cl_ok and truth_none and vs_has_keys,
            f"cl_dim={output.constraint_loss.dim()}, truth={output.truth}, "
            f"vs_keys={list(output.violation_stats.keys())}",
        )
    except Exception as e:
        check("Forward no-query", False, str(e))

    # ---- Test 3: Forward pass with query -> truth values in [0,1] ----------
    print("\nTest 3: Forward pass with query -> truth in [0,1]")
    try:
        output = reasoner.forward_from_workspace(
            workspace_slots, rules, query=queries
        )
        truth = output.truth
        truth_ok = (
            truth is not None
            and truth.dim() == 2
            and truth.shape == (B, len(queries))
        )
        range_ok = truth_ok and (truth >= 0.0).all() and (truth <= 1.0).all()
        check(
            "Forward with query truth in [0,1]",
            truth_ok and range_ok,
            f"shape={truth.shape if truth is not None else None}, "
            f"range=[{truth.min().item():.4f}, {truth.max().item():.4f}]"
            if truth is not None else "truth is None",
        )
    except Exception as e:
        check("Forward with query truth in [0,1]", False, str(e))

    # ---- Test 4: constraint_loss is differentiable -------------------------
    print("\nTest 4: constraint_loss is differentiable")
    try:
        ws = workspace_slots.clone().requires_grad_(True)
        output = reasoner.forward_from_workspace(ws, rules)
        cl = output.constraint_loss
        grad_ok = cl.requires_grad
        check(
            "constraint_loss requires_grad",
            grad_ok,
            f"requires_grad={cl.requires_grad}",
        )
    except Exception as e:
        check("constraint_loss requires_grad", False, str(e))

    # ---- Test 5: Gradient flow: constraint_loss -> predicate params --------
    print("\nTest 5: Gradient flow through predicate params")
    try:
        reasoner.zero_grad()
        ws = workspace_slots.clone().requires_grad_(True)
        output = reasoner.forward_from_workspace(ws, rules)
        output.constraint_loss.backward()
        pred_grads = []
        for name, param in reasoner.grounding.predicates.named_parameters():
            if param.grad is not None:
                pred_grads.append(param.grad.abs().sum().item())
        has_nonzero_grad = any(g > 0 for g in pred_grads)
        check(
            "Gradients flow to predicate params",
            has_nonzero_grad,
            f"num_graded_params={len(pred_grads)}, "
            f"nonzero={sum(1 for g in pred_grads if g > 0)}",
        )
    except Exception as e:
        check("Gradients flow to predicate params", False, str(e))

    # ---- Test 6: Gradient flow: constraint_loss -> entity extractor --------
    print("\nTest 6: Gradient flow through entity extractor (forward_from_workspace)")
    try:
        reasoner.zero_grad()
        ws = workspace_slots.clone().requires_grad_(True)
        output = reasoner.forward_from_workspace(ws, rules)
        output.constraint_loss.backward()
        ext_grads = []
        for name, param in reasoner.grounding.extractor.named_parameters():
            if param.grad is not None:
                ext_grads.append(param.grad.abs().sum().item())
        has_ext_grad = any(g > 0 for g in ext_grads)
        check(
            "Gradients flow to entity extractor",
            has_ext_grad,
            f"num_graded_params={len(ext_grads)}, "
            f"nonzero={sum(1 for g in ext_grads if g > 0)}",
        )
    except Exception as e:
        check("Gradients flow to entity extractor", False, str(e))

    # ---- Test 7: return_logs=False -> logs is None -------------------------
    print("\nTest 7: return_logs=False -> logs is None")
    try:
        output = reasoner.forward_from_workspace(
            workspace_slots, rules, return_logs=False
        )
        check("logs is None when return_logs=False", output.logs is None)
    except Exception as e:
        check("logs is None when return_logs=False", False, str(e))

    # ---- Test 8: return_logs=True -> logs contain expected keys -----------
    print("\nTest 8: return_logs=True -> logs contain expected keys")
    try:
        output = reasoner.forward_from_workspace(
            workspace_slots, rules, return_logs=True
        )
        logs = output.logs
        has_keys = (
            logs is not None
            and "attention_weights" in logs
            and "violations" in logs
            and "rule_truths" in logs
            and "top_violations" in logs
        )
        check(
            "logs contain attention_weights, violations, rule_truths",
            has_keys,
            f"keys={list(logs.keys()) if logs else 'None'}",
        )
    except Exception as e:
        check(
            "logs contain attention_weights, violations, rule_truths",
            False,
            str(e),
        )

    # ---- Test 9: as_loss returns scalar tensor ----------------------------
    print("\nTest 9: as_loss returns scalar tensor")
    try:
        loss = reasoner.as_loss(workspace_slots, rules)
        check(
            "as_loss returns scalar",
            isinstance(loss, Tensor) and loss.dim() == 0,
            f"type={type(loss)}, dim={loss.dim() if isinstance(loss, Tensor) else 'N/A'}",
        )
    except Exception as e:
        check("as_loss returns scalar", False, str(e))

    # ---- Test 10: as_feature returns (B, Q) tensor ------------------------
    print("\nTest 10: as_feature returns (B, Q) tensor")
    try:
        feat = reasoner.as_feature(workspace_slots, rules, queries)
        expected_shape = (B, len(queries))
        check(
            "as_feature shape (B, Q)",
            isinstance(feat, Tensor) and feat.shape == expected_shape,
            f"shape={feat.shape}, expected={expected_shape}",
        )
    except Exception as e:
        check("as_feature shape (B, Q)", False, str(e))

    # ---- Test 11: SymbolicLoss warmup: weight ramps from 0 ----------------
    print("\nTest 11: SymbolicLoss warmup schedule")
    try:
        sym_loss = SymbolicLoss(
            reasoner, weight=0.5, warmup_steps=100
        )
        # Initially at step 0, warmup_factor = 0
        w0 = sym_loss.effective_weight
        at_zero = abs(w0) < 1e-8
        # After 50 steps, warmup_factor = 0.5
        for _ in range(50):
            sym_loss.step()
        w50 = sym_loss.effective_weight
        at_half = abs(w50 - 0.25) < 1e-6  # 0.5 * 0.5 = 0.25
        # After 100 steps total, warmup_factor = 1.0
        for _ in range(50):
            sym_loss.step()
        w100 = sym_loss.effective_weight
        at_full = abs(w100 - 0.5) < 1e-6
        check(
            "Warmup ramps 0 -> 0.25 -> 0.5",
            at_zero and at_half and at_full,
            f"w0={w0:.6f}, w50={w50:.6f}, w100={w100:.6f}",
        )
    except Exception as e:
        check("Warmup ramps 0 -> 0.25 -> 0.5", False, str(e))

    # ---- Test 12: SymbolicLoss total = task + weighted constraint ----------
    print("\nTest 12: SymbolicLoss total = task + weighted constraint")
    try:
        sym_loss2 = SymbolicLoss(
            reasoner, weight=0.1, warmup_steps=0  # no warmup
        )
        task_loss = torch.tensor(1.5, requires_grad=True)
        total = sym_loss2(workspace_slots, rules, task_loss)
        constraint = reasoner.as_loss(workspace_slots, rules)
        expected = task_loss + 0.1 * constraint
        close = (total - expected).abs().item() < 1e-4
        check(
            "total_loss = task + weighted constraint",
            close,
            f"total={total.item():.6f}, expected={expected.item():.6f}",
        )
    except Exception as e:
        check("total_loss = task + weighted constraint", False, str(e))

    # ---- Test 13: Deterministic: same input -> same output -----------------
    print("\nTest 13: Deterministic output")
    try:
        reasoner_state = reasoner.training
        reasoner.train(False)
        with torch.no_grad():
            out1 = reasoner.forward_from_workspace(
                workspace_slots, rules, query=queries
            )
            out2 = reasoner.forward_from_workspace(
                workspace_slots, rules, query=queries
            )
        cl_same = (out1.constraint_loss - out2.constraint_loss).abs().item() < 1e-7
        truth_same = True
        if out1.truth is not None and out2.truth is not None:
            truth_same = (out1.truth - out2.truth).abs().max().item() < 1e-7
        check(
            "Deterministic: same input -> same output",
            cl_same and truth_same,
            f"cl_diff={(out1.constraint_loss - out2.constraint_loss).abs().item():.2e}, "
            f"truth_diff={(out1.truth - out2.truth).abs().max().item():.2e}"
            if out1.truth is not None
            else f"cl_diff={(out1.constraint_loss - out2.constraint_loss).abs().item():.2e}",
        )
        reasoner.train(reasoner_state)
    except Exception as e:
        check("Deterministic: same input -> same output", False, str(e))
        reasoner.train(True)

    # ---- Test 14: No NaN in any output tensor ------------------------------
    print("\nTest 14: No NaN in any output tensor")
    try:
        output = reasoner.forward_from_workspace(
            workspace_slots, rules, query=queries, return_logs=True
        )
        nan_found = False
        nan_location = ""
        # Check constraint_loss
        if torch.isnan(output.constraint_loss).any():
            nan_found = True
            nan_location = "constraint_loss"
        # Check truth
        if output.truth is not None and torch.isnan(output.truth).any():
            nan_found = True
            nan_location = "truth"
        # Check violation_stats tensors
        for k, v in output.violation_stats.items():
            if isinstance(v, Tensor) and torch.isnan(v).any():
                nan_found = True
                nan_location = f"violation_stats[{k}]"
                break
        # Check logs tensors
        if output.logs is not None:
            for k, v in output.logs.items():
                if isinstance(v, Tensor) and torch.isnan(v).any():
                    nan_found = True
                    nan_location = f"logs[{k}]"
                    break
        check(
            "No NaN in outputs",
            not nan_found,
            f"NaN found in: {nan_location}" if nan_found else "",
        )
    except Exception as e:
        check("No NaN in outputs", False, str(e))

    # ---- Summary -----------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"Results: {passed}/{passed + failed} passed, {failed} failed")
    print("=" * 72)

    if failed > 0:
        sys.exit(1)
    else:
        print("\nAll self-tests passed.")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    _run_self_tests()
