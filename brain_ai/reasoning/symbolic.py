"""
Neuro-Symbolic Reasoning Module

Implements fuzzy logic operations for verified multi-step reasoning.
Combines neural representations with symbolic logic operations.

Uses differentiable fuzzy logic that allows backpropagation through
logical operations while maintaining interpretable reasoning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Callable
from dataclasses import dataclass
from enum import Enum


class FuzzyLogicType(Enum):
    """Types of fuzzy logic t-norms."""
    PRODUCT = "product"  # Product t-norm
    GODEL = "godel"  # Godel (min/max) t-norm
    LUKASIEWICZ = "lukasiewicz"  # Lukasiewicz t-norm


@dataclass
class SymbolicConfig:
    """Configuration for symbolic reasoning."""
    hidden_dim: int = 512
    num_entities: int = 100  # Maximum entities in knowledge base
    num_predicates: int = 50  # Number of predicate types
    embedding_dim: int = 64
    logic_type: str = "product"  # product, godel, lukasiewicz
    regularization: float = 0.01  # Encourage crisp values


class FuzzyLogic:
    """
    Differentiable fuzzy logic operations.

    Provides basic logical connectives that work with continuous
    truth values in [0, 1] and support gradient flow.
    """

    def __init__(self, logic_type: str = "product"):
        self.logic_type = FuzzyLogicType(logic_type)

    def AND(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Fuzzy conjunction (AND)."""
        if self.logic_type == FuzzyLogicType.PRODUCT:
            return a * b
        elif self.logic_type == FuzzyLogicType.GODEL:
            return torch.min(a, b)
        elif self.logic_type == FuzzyLogicType.LUKASIEWICZ:
            return torch.clamp(a + b - 1, min=0)

    def OR(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Fuzzy disjunction (OR)."""
        if self.logic_type == FuzzyLogicType.PRODUCT:
            return a + b - a * b
        elif self.logic_type == FuzzyLogicType.GODEL:
            return torch.max(a, b)
        elif self.logic_type == FuzzyLogicType.LUKASIEWICZ:
            return torch.clamp(a + b, max=1)

    def NOT(self, a: torch.Tensor) -> torch.Tensor:
        """Fuzzy negation (NOT)."""
        return 1 - a

    def IMPLIES(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Fuzzy implication (a -> b)."""
        if self.logic_type == FuzzyLogicType.PRODUCT:
            # Reichenbach implication
            return 1 - a + a * b
        elif self.logic_type == FuzzyLogicType.GODEL:
            # Godel implication
            return torch.where(a <= b, torch.ones_like(a), b)
        elif self.logic_type == FuzzyLogicType.LUKASIEWICZ:
            return torch.clamp(1 - a + b, max=1)

    def EQUIV(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Fuzzy equivalence (a <-> b)."""
        return self.AND(self.IMPLIES(a, b), self.IMPLIES(b, a))

    def FORALL(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Universal quantifier (FOR ALL)."""
        if self.logic_type == FuzzyLogicType.PRODUCT:
            return x.prod(dim=dim)
        elif self.logic_type == FuzzyLogicType.GODEL:
            return x.min(dim=dim)[0]
        elif self.logic_type == FuzzyLogicType.LUKASIEWICZ:
            return torch.clamp(x.sum(dim=dim) - x.shape[dim] + 1, min=0)

    def EXISTS(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Existential quantifier (EXISTS)."""
        if self.logic_type == FuzzyLogicType.PRODUCT:
            return 1 - (1 - x).prod(dim=dim)
        elif self.logic_type == FuzzyLogicType.GODEL:
            return x.max(dim=dim)[0]
        elif self.logic_type == FuzzyLogicType.LUKASIEWICZ:
            return torch.clamp(x.sum(dim=dim), max=1)


class PredicateEncoder(nn.Module):
    """
    Encodes entities and predicates into neural representations.

    Maps symbolic entities (objects, concepts) to vectors that can
    be used for fuzzy logic operations.
    """

    def __init__(
        self,
        num_entities: int,
        num_predicates: int,
        embedding_dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.num_entities = num_entities
        self.num_predicates = num_predicates
        self.embedding_dim = embedding_dim

        # Entity embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)

        # Predicate networks: compute truth value for predicates
        # Unary predicates: P(x)
        self.unary_predicates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
            for _ in range(num_predicates // 2)
        ])

        # Binary predicates: R(x, y)
        self.binary_predicates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
            for _ in range(num_predicates - num_predicates // 2)
        ])

    def get_entity(self, entity_idx: torch.Tensor) -> torch.Tensor:
        """Get entity embedding."""
        return self.entity_embeddings(entity_idx)

    def unary_predicate(
        self,
        pred_idx: int,
        entity: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate unary predicate P(entity)."""
        if pred_idx >= len(self.unary_predicates):
            raise ValueError(f"Invalid unary predicate index: {pred_idx}")
        return self.unary_predicates[pred_idx](entity).squeeze(-1)

    def binary_predicate(
        self,
        pred_idx: int,
        entity1: torch.Tensor,
        entity2: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate binary predicate R(entity1, entity2)."""
        if pred_idx >= len(self.binary_predicates):
            raise ValueError(f"Invalid binary predicate index: {pred_idx}")
        combined = torch.cat([entity1, entity2], dim=-1)
        return self.binary_predicates[pred_idx](combined).squeeze(-1)


class RuleNetwork(nn.Module):
    """
    Neural network for learning logical rules.

    Learns rules of the form: IF antecedent THEN consequent
    """

    def __init__(
        self,
        hidden_dim: int,
        num_rules: int = 10,
    ):
        super().__init__()

        self.num_rules = num_rules

        # Rule attention: which rules apply to input
        self.rule_attention = nn.Linear(hidden_dim, num_rules)

        # Rule heads: what each rule concludes
        self.rule_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_rules)
        ])

        # Rule confidence
        self.rule_confidence = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
            for _ in range(num_rules)
        ])

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rules to input.

        Args:
            x: Input representation (batch, hidden_dim)

        Returns:
            output: Rule-processed output
            confidences: Confidence for each rule
        """
        # Get rule attention weights
        attention = F.softmax(self.rule_attention(x), dim=-1)

        # Apply each rule
        outputs = []
        confidences = []

        for i in range(self.num_rules):
            rule_out = self.rule_heads[i](x)
            rule_conf = self.rule_confidence[i](x)
            outputs.append(rule_out)
            confidences.append(rule_conf)

        outputs = torch.stack(outputs, dim=1)  # (batch, num_rules, hidden)
        confidences = torch.stack(confidences, dim=1).squeeze(-1)  # (batch, num_rules)

        # Weighted combination
        combined_conf = attention * confidences
        combined_conf = combined_conf / (combined_conf.sum(dim=-1, keepdim=True) + 1e-8)

        output = (outputs * combined_conf.unsqueeze(-1)).sum(dim=1)

        return output, confidences


class SymbolicReasoner(nn.Module):
    """
    Neuro-Symbolic Reasoner combining neural networks with fuzzy logic.

    Provides:
    1. Entity and predicate encoding
    2. Fuzzy logic operations
    3. Rule-based inference
    4. Multi-step reasoning chains
    """

    def __init__(
        self,
        config: Optional[SymbolicConfig] = None,
        **kwargs,
    ):
        super().__init__()

        self.config = config or SymbolicConfig(**kwargs)

        # Fuzzy logic operations
        self.logic = FuzzyLogic(self.config.logic_type)

        # Predicate encoder
        self.predicates = PredicateEncoder(
            num_entities=self.config.num_entities,
            num_predicates=self.config.num_predicates,
            embedding_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
        )

        # Rule network
        self.rules = RuleNetwork(
            hidden_dim=self.config.hidden_dim,
            num_rules=10,
        )

        # Input encoder
        self.input_encoder = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
        )

        # Output projection
        self.output_proj = nn.Linear(self.config.hidden_dim, self.config.hidden_dim)

    def reason_step(
        self,
        state: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single reasoning step.

        Args:
            state: Current reasoning state
            context: Optional context from observations

        Returns:
            new_state: Updated state after reasoning
            confidence: Reasoning confidence
        """
        # Apply rules
        rule_out, confidences = self.rules(state)

        # Combine with context if available
        if context is not None:
            rule_out = rule_out + 0.1 * context

        # Compute overall confidence
        confidence = confidences.max(dim=-1)[0]

        return rule_out, confidence

    def forward(
        self,
        x: torch.Tensor,
        num_steps: int = 3,
        return_trace: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Multi-step symbolic reasoning.

        Args:
            x: Input representation (batch, hidden_dim)
            num_steps: Number of reasoning steps
            return_trace: Whether to return intermediate states

        Returns:
            Dict with output, confidence, and optionally trace
        """
        # Encode input
        state = self.input_encoder(x)
        context = x

        trace = [state] if return_trace else None
        confidences = []

        # Iterative reasoning
        for _ in range(num_steps):
            state, conf = self.reason_step(state, context)
            confidences.append(conf)

            if return_trace:
                trace.append(state)

        # Project output
        output = self.output_proj(state)
        final_confidence = torch.stack(confidences, dim=-1).mean(dim=-1)

        result = {
            'output': output,
            'confidence': final_confidence,
            'state': state,
        }

        if return_trace:
            result['trace'] = torch.stack(trace, dim=1)

        return result

    def transitive_reasoning(
        self,
        relation_matrix: torch.Tensor,
        num_hops: int = 2,
    ) -> torch.Tensor:
        """
        Perform transitive reasoning over relations.

        Given R(a,b) and R(b,c), infer R(a,c).

        Args:
            relation_matrix: (batch, n, n) relation truth values
            num_hops: Number of transitive hops

        Returns:
            Extended relation matrix with transitive closure
        """
        result = relation_matrix

        for _ in range(num_hops):
            # Matrix multiplication in fuzzy logic
            # R(a,c) = EXISTS_b(AND(R(a,b), R(b,c)))
            extended = torch.zeros_like(result)

            for i in range(result.shape[-2]):
                for j in range(result.shape[-1]):
                    # Path through all intermediate nodes
                    paths = self.logic.AND(
                        result[..., i, :],
                        result[..., :, j]
                    )
                    extended[..., i, j] = self.logic.EXISTS(paths, dim=-1)

            # Combine with original
            result = self.logic.OR(result, extended)

        return result


class KnowledgeBase(nn.Module):
    """
    Neural Knowledge Base for storing and querying facts.

    Stores facts as continuous embeddings with truth values.
    """

    def __init__(
        self,
        hidden_dim: int,
        max_facts: int = 1000,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_facts = max_facts

        # Fact storage
        self.register_buffer('fact_keys', torch.zeros(max_facts, hidden_dim))
        self.register_buffer('fact_values', torch.zeros(max_facts, hidden_dim))
        self.register_buffer('fact_truths', torch.zeros(max_facts))
        self.register_buffer('num_facts', torch.tensor(0))

        # Query network
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)

    def add_fact(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        truth: float = 1.0,
    ):
        """Add fact to knowledge base."""
        idx = self.num_facts.item()
        if idx >= self.max_facts:
            # Overwrite oldest fact
            idx = idx % self.max_facts

        self.fact_keys[idx] = key.detach()
        self.fact_values[idx] = value.detach()
        self.fact_truths[idx] = truth
        self.num_facts = torch.tensor(min(self.num_facts.item() + 1, self.max_facts))

    def query(
        self,
        query: torch.Tensor,
        top_k: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Query knowledge base.

        Args:
            query: Query vector (batch, hidden_dim)
            top_k: Number of facts to retrieve

        Returns:
            values: Retrieved fact values
            truths: Truth values of retrieved facts
        """
        num_facts = self.num_facts.item()
        if num_facts == 0:
            return torch.zeros_like(query), torch.zeros(query.shape[0])

        # Project query
        q = self.query_proj(query)
        k = self.key_proj(self.fact_keys[:num_facts])

        # Attention scores
        scores = torch.matmul(q, k.T) / (self.hidden_dim ** 0.5)
        attention = F.softmax(scores, dim=-1)

        # Retrieve
        retrieved = torch.matmul(attention, self.fact_values[:num_facts])
        truth_weighted = torch.matmul(attention, self.fact_truths[:num_facts])

        return retrieved, truth_weighted


# Factory function
def create_symbolic_reasoner(
    hidden_dim: int = 512,
    num_entities: int = 100,
    num_predicates: int = 50,
    logic_type: str = "product",
    **kwargs,
) -> SymbolicReasoner:
    """Create symbolic reasoner with specified configuration."""
    config = SymbolicConfig(
        hidden_dim=hidden_dim,
        num_entities=num_entities,
        num_predicates=num_predicates,
        logic_type=logic_type,
        **kwargs,
    )
    return SymbolicReasoner(config)
