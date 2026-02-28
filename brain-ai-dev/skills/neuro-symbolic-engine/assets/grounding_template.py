#!/usr/bin/env python3
"""Grounding Pipeline for Neuro-Symbolic Reasoning.

This module implements the entity extraction, predicate scoring, relation
scoring, and Real Logic grounding layers that form the bridge between
continuous neural representations (workspace slots) and discrete symbolic
reasoning (fuzzy logic rules, LTN formulas).

Architecture overview::

    Workspace Slots (B, K, D_ws)
            |
    [ EntityExtractor ]  -- SlotIdentity or ProposalHead
            |
    Entities (B, N, D_ent) + mask (B, N)
            |
    +-------+-------+
    |               |
    [ PredicateReg ]   [ RelationReg ]
    P_i(x) -> [0,1]   R_j(x,y) -> [0,1]
    per entity         per entity pair
    |               |
    +-------+-------+
            |
    GroundingOutput {entities, entity_mask, predicate_truths, relation_truths}

Typical import (once integrated into brain_ai)::

    from brain_ai.reasoning.grounding import (
        GroundingConfig,
        GroundingLayer,
        GroundingOutput,
        EntityExtractor,
        SlotIdentityExtractor,
        ProposalHeadExtractor,
        PredicateModule,
        MLPRelation,
        BilinearRelation,
        NTNRelation,
        PredicateRegistry,
        RelationRegistry,
        RealLogicGrounding,
        create_grounding_layer,
    )

Copy this file to ``brain_ai/reasoning/grounding.py`` when integrating
into the main package.

Dependencies: torch (no external dependencies).
"""

from __future__ import annotations

import math
import sys
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPS: float = 1e-8
_DEDUP_COS_THRESHOLD: float = 0.95
_DEFAULT_ENTITY_DIM: int = 256
_DEFAULT_WORKSPACE_DIM: int = 4096
_DEFAULT_HIDDEN_DIM: int = 512


# ===================================================================== #
#                        Configuration                                  #
# ===================================================================== #


@dataclass
class GroundingConfig:
    """Configuration for the grounding pipeline.

    Attributes:
        entity_dim: Dimensionality of entity embeddings produced by the
            extractor.  This is the representation consumed by predicates
            and relations.
        workspace_dim: Dimensionality of upstream workspace slot vectors.
            Must match the workspace output (typically 4096).
        hidden_dim: Width of hidden layers in predicate / relation MLPs.
        extractor_type: Entity extraction strategy.  ``"slot_identity"``
            treats each workspace slot as an entity (N=K).
            ``"proposal_head"`` uses learnable cross-attention queries so
            N can differ from K.
        max_entities: Upper bound on the number of entities (N) per batch
            item.  For ``slot_identity`` this caps K; for
            ``proposal_head`` this determines the number of query vectors.
        predicate_type: Scoring architecture for unary predicates.  One of
            ``"mlp"`` (default), ``"bilinear"``, ``"ntn"``.
        relation_type: Scoring architecture for binary relations.  One of
            ``"mlp"``, ``"bilinear"`` (default), ``"ntn"``.
        num_predicates: Number of predicate slots available in the
            registry when auto-populating.
        num_relations: Number of relation slots available in the registry
            when auto-populating.
        use_ltn: When True, construct a ``RealLogicGrounding`` module
            instead of the standard ``GroundingLayer``.
        ntn_slices: Number of tensor slices (k) for the Neural Tensor
            Network relation scorer.
        dropout: Dropout probability applied inside MLP bodies.
        num_attention_heads: Number of attention heads for the
            ProposalHeadExtractor cross-attention block.
        dedup_threshold: Cosine-similarity threshold for optional entity
            deduplication in the ProposalHeadExtractor.
        num_constants: Number of learned constant embeddings for the LTN
            Real Logic grounding mode.
        num_functions: Number of neural function modules in the LTN Real
            Logic grounding mode.
    """

    entity_dim: int = 256
    workspace_dim: int = 4096
    hidden_dim: int = 512
    extractor_type: str = "slot_identity"  # "slot_identity" or "proposal_head"
    max_entities: int = 32
    predicate_type: str = "mlp"  # "mlp", "bilinear", "ntn"
    relation_type: str = "bilinear"  # "mlp", "bilinear", "ntn"
    num_predicates: int = 32
    num_relations: int = 16
    use_ltn: bool = False
    ntn_slices: int = 4  # for NTN relation scoring
    dropout: float = 0.1
    num_attention_heads: int = 4
    dedup_threshold: float = 0.95
    num_constants: int = 64
    num_functions: int = 8

    def __post_init__(self) -> None:
        assert self.extractor_type in ("slot_identity", "proposal_head"), (
            f"extractor_type must be 'slot_identity' or 'proposal_head', "
            f"got '{self.extractor_type}'"
        )
        assert self.predicate_type in ("mlp", "bilinear", "ntn"), (
            f"predicate_type must be 'mlp', 'bilinear', or 'ntn', "
            f"got '{self.predicate_type}'"
        )
        assert self.relation_type in ("mlp", "bilinear", "ntn"), (
            f"relation_type must be 'mlp', 'bilinear', or 'ntn', "
            f"got '{self.relation_type}'"
        )
        assert self.entity_dim > 0, "entity_dim must be positive"
        assert self.workspace_dim > 0, "workspace_dim must be positive"
        assert self.max_entities > 0, "max_entities must be positive"
        assert self.ntn_slices > 0, "ntn_slices must be positive"
        assert 0.0 <= self.dropout < 1.0, "dropout must be in [0, 1)"


# ===================================================================== #
#                      Grounding Output                                 #
# ===================================================================== #


@dataclass
class GroundingOutput:
    """Container for all outputs of a grounding forward pass.

    Attributes:
        entities: Entity embeddings extracted from workspace slots.
            Shape ``(B, N, D_ent)``.
        entity_mask: Boolean mask indicating valid (non-padding) entities.
            Shape ``(B, N)``.  True means valid.
        predicate_truths: Dictionary mapping predicate names to per-entity
            truth tensors.  Each value has shape ``(B, N)`` with values in
            [0, 1].
        relation_truths: Dictionary mapping relation names to pairwise
            truth matrices.  Each value has shape ``(B, N, N)`` with
            values in [0, 1].
        confidence: Per-entity confidence scores from the proposal head
            extractor.  Shape ``(B, N)`` with values in [0, 1].
            ``None`` when the ``slot_identity`` extractor is used.
    """

    entities: Tensor                            # (B, N, D_ent)
    entity_mask: Tensor                         # (B, N)
    predicate_truths: Dict[str, Tensor]         # {name: (B, N)}
    relation_truths: Dict[str, Tensor]          # {name: (B, N, N)}
    confidence: Optional[Tensor] = None         # (B, N) for proposal_head


# ===================================================================== #
#                     Entity Extractors                                 #
# ===================================================================== #


class EntityExtractor(nn.Module, ABC):
    """Abstract base class for entity extraction from workspace slots.

    Subclasses must implement ``forward`` which accepts workspace slot
    representations and returns entity embeddings with an accompanying
    validity mask.
    """

    @abstractmethod
    def forward(
        self,
        workspace_slots: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Extract entities from workspace slots.

        Args:
            workspace_slots: Workspace slot representations of shape
                ``(B, K, D_ws)`` where K is the number of slots and D_ws
                is the workspace dimensionality.
            mask: Optional boolean mask of shape ``(B, K)`` where True
                indicates valid slots.

        Returns:
            Tuple of:
                entities: Entity embeddings ``(B, N, D_ent)``
                entity_mask: Boolean mask ``(B, N)``
        """
        raise NotImplementedError


class SlotIdentityExtractor(EntityExtractor):
    """Treat each workspace slot as an entity.  N = K.

    Applies a linear projection from workspace dimensionality to entity
    dimensionality followed by LayerNorm.  The slot mask is propagated
    as-is to the entity mask.

    Architecture::

        workspace_slots (B, K, D_ws)
              |
        [ Linear(D_ws, D_ent) ]
              |
        [ LayerNorm(D_ent) ]
              |
        entities (B, K, D_ent)
    """

    def __init__(self, workspace_dim: int, entity_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(workspace_dim, entity_dim)
        self.norm = nn.LayerNorm(entity_dim)

    def forward(
        self,
        workspace_slots: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Project each workspace slot to an entity embedding.

        Args:
            workspace_slots: ``(B, K, D_ws)``
            mask: ``(B, K)`` boolean.  Defaults to all-True.

        Returns:
            entities: ``(B, K, D_ent)``
            entity_mask: ``(B, K)`` boolean
        """
        B, K, _ = workspace_slots.shape
        entities = self.norm(self.proj(workspace_slots))  # (B, K, D_ent)
        if mask is None:
            entity_mask = torch.ones(B, K, dtype=torch.bool, device=entities.device)
        else:
            entity_mask = mask
        return entities, entity_mask


class ProposalHeadExtractor(EntityExtractor):
    """Cross-attention entity proposal head.  N learnable queries attend to
    K workspace slots, producing N entity embeddings plus confidence scores.

    The number of entities N is controlled by ``max_entities`` in
    ``GroundingConfig`` and may differ from the number of workspace slots K.

    Architecture::

        learnable queries (N, D_ent)     workspace_slots (B, K, D_ws)
              |                                    |
              |                          [ key/value projection ]
              |                                    |
              +-----> [ MultiHeadCrossAttention ] <+
                               |
                          (B, N, D_ent)
                               |
                    +----------+----------+
                    |                     |
              [ LayerNorm ]         [ ConfidenceHead ]
                    |                     |
              entities (B, N, D_ent)   confidence (B, N)

    Optionally, a cosine-similarity deduplication step can merge near-
    duplicate proposals.
    """

    def __init__(
        self,
        workspace_dim: int,
        entity_dim: int,
        max_entities: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        dedup_threshold: float = 0.95,
    ) -> None:
        super().__init__()
        self.max_entities = max_entities
        self.entity_dim = entity_dim
        self.dedup_threshold = dedup_threshold

        # Learnable query vectors
        self.queries = nn.Parameter(
            torch.randn(max_entities, entity_dim) * 0.02
        )

        # Project workspace slots to key/value space
        self.kv_proj = nn.Linear(workspace_dim, entity_dim * 2)

        # Multi-head cross-attention
        assert entity_dim % num_heads == 0, (
            f"entity_dim ({entity_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.num_heads = num_heads
        self.head_dim = entity_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(entity_dim, entity_dim)
        self.out_proj = nn.Linear(entity_dim, entity_dim)
        self.norm = nn.LayerNorm(entity_dim)
        self.dropout = nn.Dropout(dropout)

        # Confidence scoring head: per-entity [0, 1]
        self.confidence_head = nn.Sequential(
            nn.Linear(entity_dim, entity_dim // 2),
            nn.ReLU(),
            nn.Linear(entity_dim // 2, 1),
            nn.Sigmoid(),
        )

    def _cross_attention(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        key_mask: Optional[Tensor],
    ) -> Tensor:
        """Multi-head cross-attention.

        Args:
            queries: ``(B, N, D_ent)``
            keys: ``(B, K, D_ent)``
            values: ``(B, K, D_ent)``
            key_mask: ``(B, K)`` boolean, True = valid

        Returns:
            ``(B, N, D_ent)``
        """
        B, N, D = queries.shape
        K = keys.shape[1]
        H = self.num_heads
        d = self.head_dim

        q = self.q_proj(queries).view(B, N, H, d).transpose(1, 2)  # (B, H, N, d)
        k = keys.view(B, K, H, d).transpose(1, 2)                  # (B, H, K, d)
        v = values.view(B, K, H, d).transpose(1, 2)                # (B, H, K, d)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale   # (B, H, N, K)

        if key_mask is not None:
            # Expand mask: (B, 1, 1, K) -- broadcast over heads and queries
            mask_expanded = key_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(~mask_expanded, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)                                 # (B, H, N, d)
        out = out.transpose(1, 2).contiguous().view(B, N, D)        # (B, N, D)
        out = self.out_proj(out)
        return out

    def _deduplicate(
        self,
        entities: Tensor,
        entity_mask: Tensor,
        confidence: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Optionally merge near-duplicate entity proposals.

        Entities whose cosine similarity exceeds ``dedup_threshold`` are
        merged by keeping the one with higher confidence and masking the
        other.

        Args:
            entities: ``(B, N, D_ent)``
            entity_mask: ``(B, N)``
            confidence: ``(B, N)``

        Returns:
            Tuple of (entities, entity_mask, confidence) with duplicates
            masked out.
        """
        if self.dedup_threshold >= 1.0:
            return entities, entity_mask, confidence

        B, N, D = entities.shape
        # Normalize for cosine similarity
        normed = F.normalize(entities, p=2, dim=-1)        # (B, N, D)
        sim = torch.bmm(normed, normed.transpose(1, 2))    # (B, N, N)

        # Upper-triangular pairs above threshold (exclude diagonal)
        upper_mask = torch.triu(torch.ones(N, N, device=entities.device, dtype=torch.bool), diagonal=1)
        is_dup = (sim > self.dedup_threshold) & upper_mask.unsqueeze(0)  # (B, N, N)

        # For each duplicate pair, mask the lower-confidence entity
        for b in range(B):
            dup_pairs = is_dup[b].nonzero(as_tuple=False)  # (num_dups, 2)
            for pair in dup_pairs:
                i, j = pair[0].item(), pair[1].item()
                if confidence[b, i] >= confidence[b, j]:
                    entity_mask[b, j] = False
                else:
                    entity_mask[b, i] = False

        return entities, entity_mask, confidence

    def forward(
        self,
        workspace_slots: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Extract entities via cross-attention proposals.

        Args:
            workspace_slots: ``(B, K, D_ws)``
            mask: ``(B, K)`` boolean.  Defaults to all-True.

        Returns:
            entities: ``(B, N, D_ent)``
            entity_mask: ``(B, N)`` boolean

        Note:
            ``self.last_confidence`` is set as a side effect so that
            ``GroundingLayer`` can include it in ``GroundingOutput``.
        """
        B, K, D_ws = workspace_slots.shape
        N = self.max_entities

        # Project workspace slots to key / value
        kv = self.kv_proj(workspace_slots)                  # (B, K, 2*D_ent)
        keys, values = kv.chunk(2, dim=-1)                  # each (B, K, D_ent)

        # Expand learnable queries to batch
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)  # (B, N, D_ent)

        # Cross-attention
        attended = self._cross_attention(queries, keys, values, key_mask=mask)
        entities = self.norm(attended + queries)             # (B, N, D_ent)

        # Confidence per entity
        confidence = self.confidence_head(entities).squeeze(-1)  # (B, N)

        # Entity mask: all proposals start as valid
        entity_mask = torch.ones(B, N, dtype=torch.bool, device=entities.device)

        # Optional deduplication
        entities, entity_mask, confidence = self._deduplicate(
            entities, entity_mask, confidence
        )

        # Store confidence for GroundingOutput construction
        self.last_confidence = confidence
        return entities, entity_mask


# ===================================================================== #
#                      Predicate Modules                                #
# ===================================================================== #


class PredicateModule(nn.Module):
    """Unary predicate P(x) -> truth in [0, 1].

    Implements a lightweight MLP scorer that maps each entity embedding
    to a scalar truth value:

        Linear(D_ent, hidden) -> ReLU -> Dropout -> Linear(hidden, 1) -> Sigmoid

    This is the core building block for grounding unary predicates such
    as ``IsAnimal(x)``, ``IsLarge(x)``, etc.

    Args:
        entity_dim: Dimensionality of input entity embeddings.
        hidden_dim: Width of the hidden layer.
        name: Human-readable predicate name for debugging and audit.
        dropout: Dropout probability in the hidden layer.
    """

    def __init__(
        self,
        entity_dim: int,
        hidden_dim: int,
        name: str = "",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.name = name
        self.entity_dim = entity_dim
        self.mlp = nn.Sequential(
            nn.Linear(entity_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, entities: Tensor) -> Tensor:
        """Evaluate predicate on entity embeddings.

        Args:
            entities: Entity embeddings ``(B, N, D_ent)`` or ``(B, D_ent)``.

        Returns:
            Truth values in [0, 1].  Shape ``(B, N)`` for 3-D input or
            ``(B,)`` for 2-D input.
        """
        return self.mlp(entities).squeeze(-1)

    def extra_repr(self) -> str:
        return f"name='{self.name}', entity_dim={self.entity_dim}"


# ===================================================================== #
#                      Relation Modules                                 #
# ===================================================================== #


class MLPRelation(nn.Module):
    """Binary relation R(x, y) via MLP over concatenated entity pairs.

    For every pair (i, j) in the entity set, concatenates e_i and e_j
    and passes through:

        Linear(2*D_ent, hidden) -> ReLU -> Dropout -> Linear(hidden, 1) -> Sigmoid

    Output is a ``(B, N, N)`` truth matrix.

    Args:
        entity_dim: Dimensionality of entity embeddings.
        hidden_dim: Width of the hidden layer.
        name: Human-readable relation name for debugging.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        entity_dim: int,
        hidden_dim: int,
        name: str = "",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.name = name
        self.entity_dim = entity_dim
        self.mlp = nn.Sequential(
            nn.Linear(entity_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, entities: Tensor) -> Tensor:
        """Compute pairwise relation truth matrix.

        Args:
            entities: ``(B, N, D_ent)``

        Returns:
            Truth matrix ``(B, N, N)`` with values in [0, 1].
        """
        B, N, D = entities.shape

        # Build all pairs: expand entities along two axes
        # e_i: (B, N, 1, D) -> (B, N, N, D)
        # e_j: (B, 1, N, D) -> (B, N, N, D)
        e_i = entities.unsqueeze(2).expand(B, N, N, D)
        e_j = entities.unsqueeze(1).expand(B, N, N, D)
        pairs = torch.cat([e_i, e_j], dim=-1)  # (B, N, N, 2*D)

        return self.mlp(pairs).squeeze(-1)  # (B, N, N)

    def extra_repr(self) -> str:
        return f"name='{self.name}', entity_dim={self.entity_dim}"


class BilinearRelation(nn.Module):
    """Binary relation R(x, y) via bilinear scoring.

    Computes ``sigmoid(e_x^T W e_y + bias)`` for each entity pair.

    When ``symmetric=True``, W is replaced with ``(W + W^T) / 2`` at
    forward time so that R(x, y) = R(y, x).

    Args:
        entity_dim: Dimensionality of entity embeddings.
        name: Human-readable relation name.
        symmetric: If True, enforce symmetric scoring.
    """

    def __init__(
        self,
        entity_dim: int,
        name: str = "",
        symmetric: bool = False,
    ) -> None:
        super().__init__()
        self.name = name
        self.entity_dim = entity_dim
        self.symmetric = symmetric

        self.weight = nn.Parameter(torch.empty(entity_dim, entity_dim))
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.weight)

    def _get_weight(self) -> Tensor:
        """Return the (possibly symmetrized) weight matrix."""
        if self.symmetric:
            return (self.weight + self.weight.t()) * 0.5
        return self.weight

    def forward(self, entities: Tensor) -> Tensor:
        """Compute pairwise bilinear relation truth matrix.

        Args:
            entities: ``(B, N, D_ent)``

        Returns:
            Truth matrix ``(B, N, N)`` with values in [0, 1].
        """
        W = self._get_weight()  # (D, D)
        # e_x^T W -> (B, N, D) @ (D, D) = (B, N, D)
        Wx = torch.matmul(entities, W)  # (B, N, D)
        # (B, N, D) @ (B, D, N) -> (B, N, N)
        scores = torch.bmm(Wx, entities.transpose(1, 2)) + self.bias
        return torch.sigmoid(scores)  # (B, N, N)

    def extra_repr(self) -> str:
        return (
            f"name='{self.name}', entity_dim={self.entity_dim}, "
            f"symmetric={self.symmetric}"
        )


class NTNRelation(nn.Module):
    """Neural Tensor Network relation scoring.

    Implements the NTN scoring function::

        h_k = e_x^T W[k] e_y          for k = 1..num_slices
        v_k = V[k] @ [e_x; e_y]       linear part
        score = U^T tanh(h + v + b)    -> sigmoid -> [0,1]

    where ``W`` has shape ``(k, D, D)``, ``V`` has shape ``(k, 2*D)``,
    ``b`` has shape ``(k,)``, and ``U`` has shape ``(k,)``.

    Args:
        entity_dim: Dimensionality of entity embeddings.
        num_slices: Number of tensor slices k.
        name: Human-readable relation name.
    """

    def __init__(
        self,
        entity_dim: int,
        num_slices: int = 4,
        name: str = "",
    ) -> None:
        super().__init__()
        self.name = name
        self.entity_dim = entity_dim
        self.num_slices = num_slices

        # Tensor slices: W[k] each (D, D)
        self.W = nn.Parameter(torch.empty(num_slices, entity_dim, entity_dim))
        # Linear part: V[k] operates on [e_x; e_y]
        self.V = nn.Parameter(torch.empty(num_slices, 2 * entity_dim))
        # Bias per slice
        self.b = nn.Parameter(torch.zeros(num_slices))
        # Final scoring vector
        self.U = nn.Parameter(torch.empty(1, num_slices))

        # Initialize
        nn.init.xavier_uniform_(self.W.view(num_slices, entity_dim, entity_dim))
        nn.init.xavier_uniform_(self.V.unsqueeze(0))
        nn.init.xavier_uniform_(self.U)

    def forward(self, entities: Tensor) -> Tensor:
        """Compute pairwise NTN relation truth matrix.

        Args:
            entities: ``(B, N, D_ent)``

        Returns:
            Truth matrix ``(B, N, N)`` with values in [0, 1].
        """
        B, N, D = entities.shape
        k = self.num_slices

        # --- Bilinear term: h_k = e_i^T W[k] e_j for all (i,j) ---
        # h[b, k, i, j] = sum_m sum_n entities[b,i,m] * W[k,m,n] * entities[b,j,n]
        h = torch.einsum("bim, kmn, bjn -> bkij", entities, self.W, entities)
        # h: (B, k, N, N)

        # --- Linear term: v_k = V[k] @ [e_i; e_j] for all (i,j) ---
        # Build pair features for the linear term
        e_i = entities.unsqueeze(2).expand(B, N, N, D)    # (B, N, N, D)
        e_j = entities.unsqueeze(1).expand(B, N, N, D)    # (B, N, N, D)
        pairs = torch.cat([e_i, e_j], dim=-1)              # (B, N, N, 2*D)

        # V: (k, 2*D) @ pairs^T -> (B, k, N, N)
        v = torch.einsum("kd, bijd -> bkij", self.V, pairs)
        # v: (B, k, N, N)

        # --- Combine: tanh(h + v + b) ---
        combined = torch.tanh(h + v + self.b.view(1, k, 1, 1))  # (B, k, N, N)

        # --- Final score: U^T @ combined -> sigmoid ---
        # U: (1, k) -> dot product over k dimension
        scores = torch.einsum("ck, bkij -> bcij", self.U, combined)  # (B, 1, N, N)
        scores = scores.squeeze(1)  # (B, N, N)
        return torch.sigmoid(scores)

    def extra_repr(self) -> str:
        return (
            f"name='{self.name}', entity_dim={self.entity_dim}, "
            f"num_slices={self.num_slices}"
        )


# ===================================================================== #
#                       Registries                                      #
# ===================================================================== #


class PredicateRegistry(nn.Module):
    """Named registry of predicate modules.

    Manages a set of ``PredicateModule`` instances keyed by string names.
    All parameters are properly tracked by ``nn.ModuleDict`` so they
    participate in ``model.parameters()`` and checkpointing.

    Typical usage::

        reg = PredicateRegistry()
        reg.register("IsAnimal", PredicateModule(256, 512, name="IsAnimal"))
        reg.register("IsLarge", PredicateModule(256, 512, name="IsLarge"))

        truths = reg.evaluate_all(entities)  # {"IsAnimal": (B,N), "IsLarge": (B,N)}
    """

    def __init__(self) -> None:
        super().__init__()
        self.predicates: nn.ModuleDict = nn.ModuleDict()

    def register(self, name: str, predicate: PredicateModule) -> None:
        """Register a predicate module under the given name.

        Args:
            name: Unique predicate name.
            predicate: ``PredicateModule`` instance.
        """
        self.predicates[name] = predicate

    def __getitem__(self, name: str) -> PredicateModule:
        """Retrieve a predicate by name.

        Args:
            name: Predicate name.

        Returns:
            The registered ``PredicateModule``.

        Raises:
            KeyError: If the name is not registered.
        """
        if name not in self.predicates:
            raise KeyError(f"Predicate '{name}' not registered. "
                           f"Available: {list(self.predicates.keys())}")
        return self.predicates[name]  # type: ignore[return-value]

    def __contains__(self, name: str) -> bool:
        return name in self.predicates

    def __len__(self) -> int:
        return len(self.predicates)

    def names(self) -> List[str]:
        """Return list of registered predicate names."""
        return list(self.predicates.keys())

    def evaluate_all(self, entities: Tensor) -> Dict[str, Tensor]:
        """Evaluate all registered predicates on the given entities.

        Args:
            entities: ``(B, N, D_ent)``

        Returns:
            Dictionary mapping predicate names to truth tensors ``(B, N)``.
        """
        return {
            name: module(entities)
            for name, module in self.predicates.items()
        }

    def audit(self) -> Dict[str, Dict[str, torch.Size]]:
        """Expose symbol_name -> parameter shapes for debugging.

        Returns:
            Nested dict: ``{pred_name: {param_name: shape}}``.
        """
        result: Dict[str, Dict[str, torch.Size]] = {}
        for name, module in self.predicates.items():
            result[name] = {
                pname: p.shape for pname, p in module.named_parameters()
            }
        return result

    def forward(self, entities: Tensor) -> Dict[str, Tensor]:
        """Forward pass: evaluate all predicates.

        Alias for ``evaluate_all``.
        """
        return self.evaluate_all(entities)


class RelationRegistry(nn.Module):
    """Named registry of relation modules.

    Manages a set of relation modules (``MLPRelation``, ``BilinearRelation``,
    ``NTNRelation``) keyed by string names.  Same interface pattern as
    ``PredicateRegistry``.

    Typical usage::

        reg = RelationRegistry()
        reg.register("Eats", BilinearRelation(256, name="Eats"))
        reg.register("PartOf", NTNRelation(256, num_slices=4, name="PartOf"))

        truths = reg.evaluate_all(entities)  # {"Eats": (B,N,N), "PartOf": (B,N,N)}
    """

    def __init__(self) -> None:
        super().__init__()
        self.relations: nn.ModuleDict = nn.ModuleDict()

    def register(self, name: str, relation: nn.Module) -> None:
        """Register a relation module under the given name.

        Args:
            name: Unique relation name.
            relation: Relation module instance.
        """
        self.relations[name] = relation

    def __getitem__(self, name: str) -> nn.Module:
        """Retrieve a relation by name."""
        if name not in self.relations:
            raise KeyError(f"Relation '{name}' not registered. "
                           f"Available: {list(self.relations.keys())}")
        return self.relations[name]

    def __contains__(self, name: str) -> bool:
        return name in self.relations

    def __len__(self) -> int:
        return len(self.relations)

    def names(self) -> List[str]:
        """Return list of registered relation names."""
        return list(self.relations.keys())

    def evaluate_all(self, entities: Tensor) -> Dict[str, Tensor]:
        """Evaluate all registered relations on the given entities.

        Args:
            entities: ``(B, N, D_ent)``

        Returns:
            Dictionary mapping relation names to truth matrices ``(B, N, N)``.
        """
        return {
            name: module(entities)
            for name, module in self.relations.items()
        }

    def audit(self) -> Dict[str, Dict[str, torch.Size]]:
        """Expose symbol_name -> parameter shapes for debugging.

        Returns:
            Nested dict: ``{relation_name: {param_name: shape}}``.
        """
        result: Dict[str, Dict[str, torch.Size]] = {}
        for name, module in self.relations.items():
            result[name] = {
                pname: p.shape for pname, p in module.named_parameters()
            }
        return result

    def forward(self, entities: Tensor) -> Dict[str, Tensor]:
        """Forward pass: evaluate all relations.

        Alias for ``evaluate_all``.
        """
        return self.evaluate_all(entities)


# ===================================================================== #
#                     Factory Helpers                                    #
# ===================================================================== #


def _make_extractor(config: GroundingConfig) -> EntityExtractor:
    """Construct the entity extractor specified by config.

    Args:
        config: ``GroundingConfig`` with extractor settings.

    Returns:
        An ``EntityExtractor`` subclass instance.
    """
    if config.extractor_type == "slot_identity":
        return SlotIdentityExtractor(config.workspace_dim, config.entity_dim)
    elif config.extractor_type == "proposal_head":
        return ProposalHeadExtractor(
            workspace_dim=config.workspace_dim,
            entity_dim=config.entity_dim,
            max_entities=config.max_entities,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
            dedup_threshold=config.dedup_threshold,
        )
    else:
        raise ValueError(f"Unknown extractor_type: {config.extractor_type}")


def _make_predicate(config: GroundingConfig, name: str) -> PredicateModule:
    """Construct a predicate module for the given config.

    Currently all predicate types use the same MLP architecture since
    unary predicates only operate on single entities.

    Args:
        config: ``GroundingConfig`` with predicate settings.
        name: Human-readable predicate name.

    Returns:
        A ``PredicateModule`` instance.
    """
    return PredicateModule(
        entity_dim=config.entity_dim,
        hidden_dim=config.hidden_dim,
        name=name,
        dropout=config.dropout,
    )


def _make_relation(config: GroundingConfig, name: str) -> nn.Module:
    """Construct a relation module for the given config.

    Args:
        config: ``GroundingConfig`` with relation settings.
        name: Human-readable relation name.

    Returns:
        A relation module (``MLPRelation``, ``BilinearRelation``, or
        ``NTNRelation``).
    """
    if config.relation_type == "mlp":
        return MLPRelation(
            entity_dim=config.entity_dim,
            hidden_dim=config.hidden_dim,
            name=name,
            dropout=config.dropout,
        )
    elif config.relation_type == "bilinear":
        return BilinearRelation(
            entity_dim=config.entity_dim,
            name=name,
        )
    elif config.relation_type == "ntn":
        return NTNRelation(
            entity_dim=config.entity_dim,
            num_slices=config.ntn_slices,
            name=name,
        )
    else:
        raise ValueError(f"Unknown relation_type: {config.relation_type}")


# ===================================================================== #
#                     Grounding Layer                                   #
# ===================================================================== #


class GroundingLayer(nn.Module):
    """Full grounding pipeline: extract entities, evaluate predicates and
    relations, and return a structured ``GroundingOutput``.

    This module integrates:

    1. **EntityExtractor** -- converts workspace slots ``(B, K, D_ws)``
       into entity embeddings ``(B, N, D_ent)`` + mask.
    2. **PredicateRegistry** -- evaluates all registered unary predicates
       ``P_i(x)`` producing per-entity truth values.
    3. **RelationRegistry** -- evaluates all registered binary relations
       ``R_j(x, y)`` producing pairwise truth matrices.

    An optional ``context`` tensor (e.g. a workspace summary) can be used
    by downstream components but is stored for reference only.

    The module exposes an ``audit`` method returning the full
    ``symbol_name -> parameter_shapes`` mapping for checkpointing
    and debugging.

    Args:
        config: ``GroundingConfig`` controlling all sub-modules.
    """

    def __init__(self, config: GroundingConfig) -> None:
        super().__init__()
        self.config = config
        self.extractor: EntityExtractor = _make_extractor(config)
        self.predicates = PredicateRegistry()
        self.relations = RelationRegistry()

        # Optional context projection (if workspace summary is provided)
        self._context_proj: Optional[nn.Linear] = None

    def add_predicate(self, name: str, predicate: Optional[PredicateModule] = None) -> PredicateModule:
        """Register a predicate.  If no module is given, create one from config.

        Args:
            name: Predicate name.
            predicate: Optional pre-built ``PredicateModule``.

        Returns:
            The registered ``PredicateModule``.
        """
        if predicate is None:
            predicate = _make_predicate(self.config, name)
        self.predicates.register(name, predicate)
        return predicate

    def add_relation(self, name: str, relation: Optional[nn.Module] = None) -> nn.Module:
        """Register a relation.  If no module is given, create one from config.

        Args:
            name: Relation name.
            relation: Optional pre-built relation module.

        Returns:
            The registered relation module.
        """
        if relation is None:
            relation = _make_relation(self.config, name)
        self.relations.register(name, relation)
        return relation

    def forward(
        self,
        workspace_slots: Tensor,
        slot_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
    ) -> GroundingOutput:
        """Run the full grounding pipeline.

        Args:
            workspace_slots: Upstream workspace slots ``(B, K, D_ws)``.
            slot_mask: Boolean mask ``(B, K)`` for valid slots.
            context: Optional workspace summary ``(B, D_ctx)``.  Stored
                for downstream use but does not affect entity/predicate/
                relation computation in this base class.

        Returns:
            ``GroundingOutput`` with entities, masks, predicate truths,
            and relation truths.
        """
        # fp32 precision for truth values
        orig_dtype = workspace_slots.dtype
        workspace_slots = workspace_slots.float()

        # 1. Entity extraction
        entities, entity_mask = self.extractor(workspace_slots, mask=slot_mask)
        # entities: (B, N, D_ent), entity_mask: (B, N)

        # 2. Evaluate predicates
        predicate_truths: Dict[str, Tensor] = {}
        if len(self.predicates) > 0:
            predicate_truths = self.predicates.evaluate_all(entities)

        # 3. Evaluate relations
        relation_truths: Dict[str, Tensor] = {}
        if len(self.relations) > 0:
            relation_truths = self.relations.evaluate_all(entities)

        # 4. Extract confidence from proposal head (if applicable)
        confidence: Optional[Tensor] = None
        if isinstance(self.extractor, ProposalHeadExtractor):
            confidence = self.extractor.last_confidence

        return GroundingOutput(
            entities=entities,
            entity_mask=entity_mask,
            predicate_truths=predicate_truths,
            relation_truths=relation_truths,
            confidence=confidence,
        )

    def audit(self) -> Dict[str, Dict[str, torch.Size]]:
        """Expose symbol_name -> parameter shapes for debugging.

        Returns:
            Dictionary with keys ``"predicates"`` and ``"relations"``,
            each mapping symbol names to their parameter shape dicts.
        """
        return {
            "predicates": self.predicates.audit(),
            "relations": self.relations.audit(),
        }


# ===================================================================== #
#                   Real Logic Grounding (LTN Mode)                     #
# ===================================================================== #


class NeuralFunction(nn.Module):
    """A neural function mapping entity embeddings to new entity embeddings.

    Used in Real Logic grounding mode to represent LTN functions such as
    ``Father(x)`` which returns the embedding of x's father.

    Architecture::

        [e_1; e_2; ...; e_arity] -> Linear -> ReLU -> Linear -> entity_dim

    Args:
        entity_dim: Dimensionality of entity embeddings.
        hidden_dim: Hidden layer width.
        arity: Number of input entities (typically 1).
        name: Human-readable function name.
    """

    def __init__(
        self,
        entity_dim: int,
        hidden_dim: int,
        arity: int = 1,
        name: str = "",
    ) -> None:
        super().__init__()
        self.name = name
        self.arity = arity
        self.entity_dim = entity_dim
        self.mlp = nn.Sequential(
            nn.Linear(entity_dim * arity, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, entity_dim),
        )

    def forward(self, *args: Tensor) -> Tensor:
        """Apply neural function to input entity embeddings.

        Args:
            *args: One or more entity tensors, each ``(B, D_ent)`` or
                ``(B, N, D_ent)``.

        Returns:
            New entity embedding with the same shape as the first input.
        """
        assert len(args) == self.arity, (
            f"Function '{self.name}' expects {self.arity} args, got {len(args)}"
        )
        combined = torch.cat(args, dim=-1)
        return self.mlp(combined)

    def extra_repr(self) -> str:
        return f"name='{self.name}', arity={self.arity}, entity_dim={self.entity_dim}"


class RealLogicGrounding(nn.Module):
    """LTN-style grounding: constants as vectors, functions as neural maps,
    predicates as neural functions mapping to [0, 1].

    This module provides the full Real Logic grounding interface:

    - **Constants** are learned embedding vectors retrieved by integer ID.
    - **Functions** are neural networks mapping entity embeddings to new
      entity embeddings (e.g. ``Father(x)``).
    - **Predicates** are scored via ``PredicateRegistry`` (unary P(x)).
    - **Relations** are scored via ``RelationRegistry`` (binary R(x,y)).

    It extends ``GroundingLayer`` functionality with the additional ability
    to ground individual constants and apply function modules, which is
    needed for LTN-style formula evaluation.

    Args:
        config: ``GroundingConfig`` with ``use_ltn=True``.
    """

    def __init__(self, config: GroundingConfig) -> None:
        super().__init__()
        self.config = config

        # Entity extractor (same as GroundingLayer)
        self.extractor: EntityExtractor = _make_extractor(config)

        # Learned constant embeddings: symbol_id -> vector
        self.constant_embeddings = nn.Embedding(
            config.num_constants, config.entity_dim,
        )
        nn.init.normal_(self.constant_embeddings.weight, std=0.02)

        # Function modules: name -> NeuralFunction
        self.function_modules: nn.ModuleDict = nn.ModuleDict()

        # Predicate and relation registries
        self.predicate_modules = PredicateRegistry()
        self.relation_modules = RelationRegistry()

    def add_function(
        self,
        name: str,
        arity: int = 1,
        hidden_dim: Optional[int] = None,
    ) -> NeuralFunction:
        """Register a neural function.

        Args:
            name: Function name (e.g. ``"Father"``).
            arity: Number of input entities.
            hidden_dim: Hidden layer width.  Defaults to ``config.hidden_dim``.

        Returns:
            The registered ``NeuralFunction``.
        """
        func = NeuralFunction(
            entity_dim=self.config.entity_dim,
            hidden_dim=hidden_dim or self.config.hidden_dim,
            arity=arity,
            name=name,
        )
        self.function_modules[name] = func
        return func

    def add_predicate(self, name: str, predicate: Optional[PredicateModule] = None) -> PredicateModule:
        """Register a predicate module.

        Args:
            name: Predicate name.
            predicate: Optional pre-built module.

        Returns:
            The registered ``PredicateModule``.
        """
        if predicate is None:
            predicate = _make_predicate(self.config, name)
        self.predicate_modules.register(name, predicate)
        return predicate

    def add_relation(self, name: str, relation: Optional[nn.Module] = None) -> nn.Module:
        """Register a relation module.

        Args:
            name: Relation name.
            relation: Optional pre-built module.

        Returns:
            The registered relation module.
        """
        if relation is None:
            relation = _make_relation(self.config, name)
        self.relation_modules.register(name, relation)
        return relation

    def ground_constant(self, symbol_id: Union[int, Tensor]) -> Tensor:
        """Retrieve the grounding vector for a constant symbol.

        Args:
            symbol_id: Integer constant ID or ``LongTensor`` of IDs.

        Returns:
            Entity embedding(s).  Shape ``(D_ent,)`` for a single ID or
            ``(batch, D_ent)`` for a tensor of IDs.
        """
        if isinstance(symbol_id, int):
            symbol_id = torch.tensor(symbol_id, dtype=torch.long,
                                     device=self.constant_embeddings.weight.device)
        return self.constant_embeddings(symbol_id)

    def ground_function(self, func_name: str, *args: Tensor) -> Tensor:
        """Apply a named neural function to entity embeddings.

        Args:
            func_name: Registered function name.
            *args: Entity tensors to pass to the function.

        Returns:
            Resulting entity embedding.

        Raises:
            KeyError: If the function name is not registered.
        """
        if func_name not in self.function_modules:
            raise KeyError(
                f"Function '{func_name}' not registered. "
                f"Available: {list(self.function_modules.keys())}"
            )
        return self.function_modules[func_name](*args)

    def ground_predicate(self, pred_name: str, *entities: Tensor) -> Tensor:
        """Evaluate a named predicate on entity embeddings.

        For unary predicates, pass a single entity tensor.  For binary
        predicates (relations), pass two entity tensors.

        Args:
            pred_name: Registered predicate or relation name.
            *entities: One entity tensor for predicates, two for relations.

        Returns:
            Truth value(s) in [0, 1].
        """
        if len(entities) == 1:
            # Unary predicate
            if pred_name not in self.predicate_modules:
                raise KeyError(
                    f"Predicate '{pred_name}' not registered. "
                    f"Available: {self.predicate_modules.names()}"
                )
            return self.predicate_modules[pred_name](entities[0])
        elif len(entities) == 2:
            # Binary relation -- pack into a single tensor for the relation module
            if pred_name not in self.relation_modules:
                raise KeyError(
                    f"Relation '{pred_name}' not registered. "
                    f"Available: {self.relation_modules.names()}"
                )
            # Relations expect (B, N, D) and return (B, N, N).
            # For a single pair, stack along N dimension.
            stacked = torch.stack([entities[0], entities[1]], dim=-2)
            if stacked.dim() == 2:
                stacked = stacked.unsqueeze(0)  # add batch dim
            truth_matrix = self.relation_modules[pred_name](stacked)
            # Return the (0, 1) element for the pair
            return truth_matrix[..., 0, 1]
        else:
            raise ValueError(
                f"ground_predicate expects 1 or 2 entity tensors, got {len(entities)}"
            )

    def forward(
        self,
        workspace_slots: Tensor,
        slot_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
    ) -> GroundingOutput:
        """Run the Real Logic grounding pipeline.

        Identical interface to ``GroundingLayer.forward``.

        Args:
            workspace_slots: ``(B, K, D_ws)``
            slot_mask: ``(B, K)`` boolean
            context: ``(B, D_ctx)`` optional workspace summary

        Returns:
            ``GroundingOutput`` with entities, masks, predicate truths,
            and relation truths.
        """
        # fp32 for truth value precision
        workspace_slots = workspace_slots.float()

        # 1. Entity extraction
        entities, entity_mask = self.extractor(workspace_slots, mask=slot_mask)

        # 2. Predicates
        predicate_truths: Dict[str, Tensor] = {}
        if len(self.predicate_modules) > 0:
            predicate_truths = self.predicate_modules.evaluate_all(entities)

        # 3. Relations
        relation_truths: Dict[str, Tensor] = {}
        if len(self.relation_modules) > 0:
            relation_truths = self.relation_modules.evaluate_all(entities)

        # 4. Confidence (for proposal head)
        confidence: Optional[Tensor] = None
        if isinstance(self.extractor, ProposalHeadExtractor):
            confidence = self.extractor.last_confidence

        return GroundingOutput(
            entities=entities,
            entity_mask=entity_mask,
            predicate_truths=predicate_truths,
            relation_truths=relation_truths,
            confidence=confidence,
        )

    def audit(self) -> Dict[str, Any]:
        """Expose full symbol-to-parameter mapping for debugging.

        Returns:
            Dictionary with keys ``"constants"``, ``"functions"``,
            ``"predicates"``, and ``"relations"``.
        """
        return {
            "constants": {
                "embedding": self.constant_embeddings.weight.shape,
            },
            "functions": {
                name: {pn: p.shape for pn, p in mod.named_parameters()}
                for name, mod in self.function_modules.items()
            },
            "predicates": self.predicate_modules.audit(),
            "relations": self.relation_modules.audit(),
        }


# ===================================================================== #
#                     Factory Function                                  #
# ===================================================================== #


def create_grounding_layer(config: GroundingConfig) -> Union[GroundingLayer, RealLogicGrounding]:
    """Factory: create a grounding layer from config.

    When ``config.use_ltn`` is True, returns a ``RealLogicGrounding``
    module with the full LTN interface.  Otherwise returns a standard
    ``GroundingLayer``.

    Args:
        config: ``GroundingConfig`` instance.

    Returns:
        ``GroundingLayer`` or ``RealLogicGrounding`` depending on
        ``config.use_ltn``.
    """
    if config.use_ltn:
        return RealLogicGrounding(config)
    return GroundingLayer(config)


# ===================================================================== #
#                 Convenience: Auto-Populate Registries                 #
# ===================================================================== #


def auto_populate_predicates(
    layer: Union[GroundingLayer, RealLogicGrounding],
    names: Optional[List[str]] = None,
) -> None:
    """Automatically register a set of predicate modules.

    If ``names`` is not provided, creates generic predicates named
    ``"pred_0"`` through ``"pred_{N-1}"`` where N is
    ``config.num_predicates``.

    Args:
        layer: The grounding layer (or RealLogicGrounding) to populate.
        names: Optional list of predicate names.
    """
    config = layer.config
    if names is None:
        names = [f"pred_{i}" for i in range(config.num_predicates)]
    for name in names:
        layer.add_predicate(name)


def auto_populate_relations(
    layer: Union[GroundingLayer, RealLogicGrounding],
    names: Optional[List[str]] = None,
) -> None:
    """Automatically register a set of relation modules.

    If ``names`` is not provided, creates generic relations named
    ``"rel_0"`` through ``"rel_{M-1}"`` where M is
    ``config.num_relations``.

    Args:
        layer: The grounding layer (or RealLogicGrounding) to populate.
        names: Optional list of relation names.
    """
    config = layer.config
    if names is None:
        names = [f"rel_{i}" for i in range(config.num_relations)]
    for name in names:
        layer.add_relation(name)


# ===================================================================== #
#                     Self-Tests                                        #
# ===================================================================== #


def _run_tests() -> None:
    """Run self-tests for the grounding pipeline.

    Each test prints PASS or FAIL.  Exits with code 1 on any failure.
    """

    passed: int = 0
    failed: int = 0
    total: int = 0

    def check(test_name: str, condition: bool, detail: str = "") -> None:
        nonlocal passed, failed, total
        total += 1
        if condition:
            passed += 1
            print(f"  [{total:2d}] PASS  {test_name}")
        else:
            failed += 1
            msg = f"  [{total:2d}] FAIL  {test_name}"
            if detail:
                msg += f"  -- {detail}"
            print(msg)

    print("=" * 72)
    print("Grounding Pipeline Self-Tests")
    print("=" * 72)

    device = torch.device("cpu")
    B, K, D_ws, D_ent, H = 4, 8, 128, 64, 128
    # Use smaller dims for fast testing

    # ---------------------------------------------------------------
    # Test 1: SlotIdentityExtractor shape correctness
    # ---------------------------------------------------------------
    try:
        extractor = SlotIdentityExtractor(workspace_dim=D_ws, entity_dim=D_ent)
        slots = torch.randn(B, K, D_ws, device=device)
        entities, mask = extractor(slots)
        ok = (entities.shape == (B, K, D_ent)) and (mask.shape == (B, K))
        check(
            "SlotIdentityExtractor shape correctness",
            ok,
            f"entities={entities.shape}, mask={mask.shape}",
        )
    except Exception as e:
        check("SlotIdentityExtractor shape correctness", False, str(e))

    # ---------------------------------------------------------------
    # Test 2: SlotIdentityExtractor mask propagation
    # ---------------------------------------------------------------
    try:
        slot_mask = torch.ones(B, K, dtype=torch.bool, device=device)
        slot_mask[:, -2:] = False  # mask out last two slots
        entities, out_mask = extractor(slots, mask=slot_mask)
        # out_mask should be identical to slot_mask
        ok = torch.equal(out_mask, slot_mask)
        check(
            "SlotIdentityExtractor mask propagation",
            ok,
            f"mask match: {ok}",
        )
    except Exception as e:
        check("SlotIdentityExtractor mask propagation", False, str(e))

    # ---------------------------------------------------------------
    # Test 3: ProposalHeadExtractor shape correctness (N < K)
    # ---------------------------------------------------------------
    try:
        N_proposals = 5  # N < K=8
        prop_extractor = ProposalHeadExtractor(
            workspace_dim=D_ws,
            entity_dim=D_ent,
            max_entities=N_proposals,
            num_heads=2,
            dropout=0.0,
            dedup_threshold=1.0,  # disable dedup for shape test
        )
        entities_p, mask_p = prop_extractor(slots)
        ok = (entities_p.shape == (B, N_proposals, D_ent)) and (mask_p.shape == (B, N_proposals))
        check(
            "ProposalHeadExtractor shape correctness (N < K)",
            ok,
            f"entities={entities_p.shape}, mask={mask_p.shape}",
        )
    except Exception as e:
        check("ProposalHeadExtractor shape correctness (N < K)", False, str(e))

    # ---------------------------------------------------------------
    # Test 4: ProposalHeadExtractor confidence in [0, 1]
    # ---------------------------------------------------------------
    try:
        conf = prop_extractor.last_confidence  # set during forward
        ok = (conf.min() >= 0.0) and (conf.max() <= 1.0)
        check(
            "ProposalHeadExtractor confidence in [0,1]",
            ok,
            f"min={conf.min().item():.6f}, max={conf.max().item():.6f}",
        )
    except Exception as e:
        check("ProposalHeadExtractor confidence in [0,1]", False, str(e))

    # ---------------------------------------------------------------
    # Test 5: PredicateModule output range [0, 1]
    # ---------------------------------------------------------------
    try:
        pred = PredicateModule(entity_dim=D_ent, hidden_dim=H, name="TestPred", dropout=0.0)
        test_entities = torch.randn(B, K, D_ent, device=device)
        truth = pred(test_entities)
        ok = (truth.min() >= 0.0) and (truth.max() <= 1.0)
        check(
            "PredicateModule output range [0,1]",
            ok,
            f"min={truth.min().item():.6f}, max={truth.max().item():.6f}",
        )
    except Exception as e:
        check("PredicateModule output range [0,1]", False, str(e))

    # ---------------------------------------------------------------
    # Test 6: PredicateModule batch shape (B, N)
    # ---------------------------------------------------------------
    try:
        ok = truth.shape == (B, K)
        check(
            "PredicateModule batch shape (B, N)",
            ok,
            f"expected ({B}, {K}), got {truth.shape}",
        )
    except Exception as e:
        check("PredicateModule batch shape (B, N)", False, str(e))

    # ---------------------------------------------------------------
    # Test 7: BilinearRelation output shape (B, N, N)
    # ---------------------------------------------------------------
    try:
        rel = BilinearRelation(entity_dim=D_ent, name="TestRel")
        rel_truth = rel(test_entities)
        ok = rel_truth.shape == (B, K, K)
        check(
            "BilinearRelation output shape (B, N, N)",
            ok,
            f"expected ({B}, {K}, {K}), got {rel_truth.shape}",
        )
    except Exception as e:
        check("BilinearRelation output shape (B, N, N)", False, str(e))

    # ---------------------------------------------------------------
    # Test 8: BilinearRelation output range [0, 1]
    # ---------------------------------------------------------------
    try:
        ok = (rel_truth.min() >= 0.0) and (rel_truth.max() <= 1.0)
        check(
            "BilinearRelation output range [0,1]",
            ok,
            f"min={rel_truth.min().item():.6f}, max={rel_truth.max().item():.6f}",
        )
    except Exception as e:
        check("BilinearRelation output range [0,1]", False, str(e))

    # ---------------------------------------------------------------
    # Test 9: NTNRelation output shape and range
    # ---------------------------------------------------------------
    try:
        ntn = NTNRelation(entity_dim=D_ent, num_slices=4, name="TestNTN")
        ntn_truth = ntn(test_entities)
        shape_ok = ntn_truth.shape == (B, K, K)
        range_ok = (ntn_truth.min() >= 0.0) and (ntn_truth.max() <= 1.0)
        ok = shape_ok and range_ok
        check(
            "NTNRelation output shape and range",
            ok,
            f"shape={ntn_truth.shape}, min={ntn_truth.min().item():.6f}, "
            f"max={ntn_truth.max().item():.6f}",
        )
    except Exception as e:
        check("NTNRelation output shape and range", False, str(e))

    # ---------------------------------------------------------------
    # Test 10: PredicateRegistry register and lookup
    # ---------------------------------------------------------------
    try:
        preg = PredicateRegistry()
        p1 = PredicateModule(D_ent, H, name="IsAnimal")
        p2 = PredicateModule(D_ent, H, name="IsLarge")
        preg.register("IsAnimal", p1)
        preg.register("IsLarge", p2)

        lookup_ok = preg["IsAnimal"] is p1 and preg["IsLarge"] is p2
        len_ok = len(preg) == 2
        names_ok = set(preg.names()) == {"IsAnimal", "IsLarge"}

        all_truths = preg.evaluate_all(test_entities)
        eval_ok = (
            "IsAnimal" in all_truths
            and "IsLarge" in all_truths
            and all_truths["IsAnimal"].shape == (B, K)
        )

        ok = lookup_ok and len_ok and names_ok and eval_ok
        check("PredicateRegistry register and lookup", ok)
    except Exception as e:
        check("PredicateRegistry register and lookup", False, str(e))

    # ---------------------------------------------------------------
    # Test 11: RelationRegistry register and lookup
    # ---------------------------------------------------------------
    try:
        rreg = RelationRegistry()
        r1 = BilinearRelation(D_ent, name="Eats")
        r2 = NTNRelation(D_ent, num_slices=2, name="PartOf")
        rreg.register("Eats", r1)
        rreg.register("PartOf", r2)

        lookup_ok = rreg["Eats"] is r1 and rreg["PartOf"] is r2
        len_ok = len(rreg) == 2
        names_ok = set(rreg.names()) == {"Eats", "PartOf"}

        all_rels = rreg.evaluate_all(test_entities)
        eval_ok = (
            "Eats" in all_rels
            and "PartOf" in all_rels
            and all_rels["Eats"].shape == (B, K, K)
        )

        ok = lookup_ok and len_ok and names_ok and eval_ok
        check("RelationRegistry register and lookup", ok)
    except Exception as e:
        check("RelationRegistry register and lookup", False, str(e))

    # ---------------------------------------------------------------
    # Test 12: GroundingLayer full forward pass
    # ---------------------------------------------------------------
    try:
        config = GroundingConfig(
            entity_dim=D_ent,
            workspace_dim=D_ws,
            hidden_dim=H,
            extractor_type="slot_identity",
            max_entities=K,
            predicate_type="mlp",
            relation_type="bilinear",
            num_predicates=3,
            num_relations=2,
            use_ltn=False,
            dropout=0.0,
        )
        layer = GroundingLayer(config)
        layer.add_predicate("color")
        layer.add_predicate("shape")
        layer.add_relation("near")
        layer.add_relation("above")

        out = layer(slots)
        ent_ok = out.entities.shape == (B, K, D_ent)
        mask_ok = out.entity_mask.shape == (B, K)
        pred_ok = (
            "color" in out.predicate_truths
            and "shape" in out.predicate_truths
            and out.predicate_truths["color"].shape == (B, K)
        )
        rel_ok = (
            "near" in out.relation_truths
            and "above" in out.relation_truths
            and out.relation_truths["near"].shape == (B, K, K)
        )
        conf_ok = out.confidence is None  # slot_identity has no confidence

        ok = ent_ok and mask_ok and pred_ok and rel_ok and conf_ok
        check(
            "GroundingLayer full forward pass",
            ok,
            f"entities={out.entities.shape}, preds={list(out.predicate_truths.keys())}, "
            f"rels={list(out.relation_truths.keys())}",
        )
    except Exception as e:
        check("GroundingLayer full forward pass", False, str(e))

    # ---------------------------------------------------------------
    # Test 13: GroundingLayer audit returns correct structure
    # ---------------------------------------------------------------
    try:
        audit = layer.audit()
        has_preds = "predicates" in audit and "color" in audit["predicates"]
        has_rels = "relations" in audit and "near" in audit["relations"]
        # Each predicate/relation should have parameter shapes
        pred_params = audit["predicates"]["color"]
        has_shapes = all(isinstance(v, torch.Size) for v in pred_params.values())
        ok = has_preds and has_rels and has_shapes and len(pred_params) > 0
        check(
            "GroundingLayer audit returns correct structure",
            ok,
            f"keys={list(audit.keys())}, pred_params={list(pred_params.keys())}",
        )
    except Exception as e:
        check("GroundingLayer audit returns correct structure", False, str(e))

    # ---------------------------------------------------------------
    # Test 14: Gradient flow: grounding loss -> entity extractor -> upstream
    # ---------------------------------------------------------------
    try:
        config_grad = GroundingConfig(
            entity_dim=D_ent,
            workspace_dim=D_ws,
            hidden_dim=H,
            extractor_type="slot_identity",
            dropout=0.0,
        )
        layer_grad = GroundingLayer(config_grad)
        layer_grad.add_predicate("test_pred")
        layer_grad.add_relation("test_rel")

        # Upstream "encoder" whose gradients we want to verify
        upstream = nn.Linear(D_ws, D_ws)
        x = torch.randn(B, K, D_ws, requires_grad=True)
        encoded = upstream(x)

        out = layer_grad(encoded)

        # Compute a loss from predicate and relation truths
        pred_loss = out.predicate_truths["test_pred"].mean()
        rel_loss = out.relation_truths["test_rel"].mean()
        loss = pred_loss + rel_loss
        loss.backward()

        # Check that gradients flow back to the upstream encoder
        upstream_grad_ok = upstream.weight.grad is not None and upstream.weight.grad.abs().sum() > 0
        # Check that the extractor projection also received gradients
        extractor_grad_ok = (
            layer_grad.extractor.proj.weight.grad is not None
            and layer_grad.extractor.proj.weight.grad.abs().sum() > 0
        )
        # Check that the input x also received gradients
        input_grad_ok = x.grad is not None and x.grad.abs().sum() > 0

        ok = upstream_grad_ok and extractor_grad_ok and input_grad_ok
        check(
            "Gradient flow: grounding loss -> extractor -> upstream",
            ok,
            f"upstream_grad={upstream_grad_ok}, extractor_grad={extractor_grad_ok}, "
            f"input_grad={input_grad_ok}",
        )
    except Exception as e:
        check(
            "Gradient flow: grounding loss -> extractor -> upstream",
            False,
            f"{e}\n{traceback.format_exc()}",
        )

    # ---------------------------------------------------------------
    # Test 15: MLPRelation output shape and range
    # ---------------------------------------------------------------
    try:
        mlp_rel = MLPRelation(entity_dim=D_ent, hidden_dim=H, name="TestMLPRel", dropout=0.0)
        mlp_truth = mlp_rel(test_entities)
        shape_ok = mlp_truth.shape == (B, K, K)
        range_ok = (mlp_truth.min() >= 0.0) and (mlp_truth.max() <= 1.0)
        ok = shape_ok and range_ok
        check(
            "MLPRelation output shape and range",
            ok,
            f"shape={mlp_truth.shape}, min={mlp_truth.min().item():.6f}, "
            f"max={mlp_truth.max().item():.6f}",
        )
    except Exception as e:
        check("MLPRelation output shape and range", False, str(e))

    # ---------------------------------------------------------------
    # Test 16: RealLogicGrounding forward + ground_constant
    # ---------------------------------------------------------------
    try:
        ltn_config = GroundingConfig(
            entity_dim=D_ent,
            workspace_dim=D_ws,
            hidden_dim=H,
            extractor_type="slot_identity",
            use_ltn=True,
            num_constants=16,
            num_functions=4,
            dropout=0.0,
        )
        ltn_layer = RealLogicGrounding(ltn_config)
        ltn_layer.add_predicate("IsRed")
        ltn_layer.add_relation("SameColor")
        ltn_layer.add_function("Complement", arity=1)

        # Test forward pass
        out_ltn = ltn_layer(slots)
        ent_ok = out_ltn.entities.shape == (B, K, D_ent)
        pred_ok = "IsRed" in out_ltn.predicate_truths

        # Test ground_constant
        c_vec = ltn_layer.ground_constant(0)
        const_ok = c_vec.shape == (D_ent,)

        # Test ground_function
        f_out = ltn_layer.ground_function("Complement", c_vec)
        func_ok = f_out.shape == (D_ent,)

        ok = ent_ok and pred_ok and const_ok and func_ok
        check(
            "RealLogicGrounding forward + ground_constant/function",
            ok,
            f"ent={out_ltn.entities.shape}, const={c_vec.shape}, func={f_out.shape}",
        )
    except Exception as e:
        check(
            "RealLogicGrounding forward + ground_constant/function",
            False,
            f"{e}\n{traceback.format_exc()}",
        )

    # ---------------------------------------------------------------
    # Test 17: RealLogicGrounding audit structure
    # ---------------------------------------------------------------
    try:
        audit_ltn = ltn_layer.audit()
        has_const = "constants" in audit_ltn
        has_func = "functions" in audit_ltn and "Complement" in audit_ltn["functions"]
        has_pred = "predicates" in audit_ltn and "IsRed" in audit_ltn["predicates"]
        has_rel = "relations" in audit_ltn and "SameColor" in audit_ltn["relations"]
        ok = has_const and has_func and has_pred and has_rel
        check(
            "RealLogicGrounding audit structure",
            ok,
            f"keys={list(audit_ltn.keys())}",
        )
    except Exception as e:
        check("RealLogicGrounding audit structure", False, str(e))

    # ---------------------------------------------------------------
    # Test 18: create_grounding_layer factory with use_ltn=False
    # ---------------------------------------------------------------
    try:
        cfg_std = GroundingConfig(use_ltn=False, entity_dim=D_ent, workspace_dim=D_ws)
        layer_std = create_grounding_layer(cfg_std)
        ok = isinstance(layer_std, GroundingLayer) and not isinstance(layer_std, RealLogicGrounding)
        check(
            "create_grounding_layer factory (use_ltn=False)",
            ok,
            f"type={type(layer_std).__name__}",
        )
    except Exception as e:
        check("create_grounding_layer factory (use_ltn=False)", False, str(e))

    # ---------------------------------------------------------------
    # Test 19: create_grounding_layer factory with use_ltn=True
    # ---------------------------------------------------------------
    try:
        cfg_ltn = GroundingConfig(use_ltn=True, entity_dim=D_ent, workspace_dim=D_ws)
        layer_ltn = create_grounding_layer(cfg_ltn)
        ok = isinstance(layer_ltn, RealLogicGrounding)
        check(
            "create_grounding_layer factory (use_ltn=True)",
            ok,
            f"type={type(layer_ltn).__name__}",
        )
    except Exception as e:
        check("create_grounding_layer factory (use_ltn=True)", False, str(e))

    # ---------------------------------------------------------------
    # Test 20: ProposalHead grounding layer end-to-end
    # ---------------------------------------------------------------
    try:
        config_ph = GroundingConfig(
            entity_dim=D_ent,
            workspace_dim=D_ws,
            hidden_dim=H,
            extractor_type="proposal_head",
            max_entities=6,
            num_attention_heads=2,
            dropout=0.0,
            dedup_threshold=1.0,
        )
        layer_ph = GroundingLayer(config_ph)
        layer_ph.add_predicate("shape")
        layer_ph.add_relation("touches")

        out_ph = layer_ph(slots)
        ent_ok = out_ph.entities.shape == (B, 6, D_ent)
        conf_ok = out_ph.confidence is not None and out_ph.confidence.shape == (B, 6)
        pred_ok = out_ph.predicate_truths["shape"].shape == (B, 6)
        rel_ok = out_ph.relation_truths["touches"].shape == (B, 6, 6)

        ok = ent_ok and conf_ok and pred_ok and rel_ok
        check(
            "ProposalHead grounding layer end-to-end",
            ok,
            f"entities={out_ph.entities.shape}, conf={out_ph.confidence.shape if out_ph.confidence is not None else None}",
        )
    except Exception as e:
        check("ProposalHead grounding layer end-to-end", False, str(e))

    # ---------------------------------------------------------------
    # Test 21: BilinearRelation symmetric mode
    # ---------------------------------------------------------------
    try:
        sym_rel = BilinearRelation(entity_dim=D_ent, name="Similar", symmetric=True)
        sym_truth = sym_rel(test_entities)
        # For a symmetric relation, R(i,j) should equal R(j,i)
        diff = (sym_truth - sym_truth.transpose(1, 2)).abs().max().item()
        ok = diff < 1e-5
        check(
            "BilinearRelation symmetric mode",
            ok,
            f"max asymmetry={diff:.8f}",
        )
    except Exception as e:
        check("BilinearRelation symmetric mode", False, str(e))

    # ---------------------------------------------------------------
    # Test 22: auto_populate helpers
    # ---------------------------------------------------------------
    try:
        cfg_auto = GroundingConfig(
            entity_dim=D_ent,
            workspace_dim=D_ws,
            hidden_dim=H,
            num_predicates=3,
            num_relations=2,
            dropout=0.0,
        )
        layer_auto = GroundingLayer(cfg_auto)
        auto_populate_predicates(layer_auto)
        auto_populate_relations(layer_auto)

        pred_count = len(layer_auto.predicates)
        rel_count = len(layer_auto.relations)
        ok = pred_count == 3 and rel_count == 2
        check(
            "auto_populate helpers",
            ok,
            f"predicates={pred_count}, relations={rel_count}",
        )
    except Exception as e:
        check("auto_populate helpers", False, str(e))

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("=" * 72)
    print(f"Results: {passed}/{total} passed, {failed}/{total} failed")
    print("=" * 72)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    _run_tests()
