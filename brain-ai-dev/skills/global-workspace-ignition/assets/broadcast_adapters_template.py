"""
brain_ai/workspace/broadcast.py — Broadcast Adapter Subsystem of the Global Workspace.

This module implements the broadcast phase of Global Workspace Theory (GWT).
After specialist modules compete for workspace access and winners are selected,
the winning content must be **broadcast** to all downstream consumers in
format-appropriate representations.

The key insight is that different consumers need different shapes:
  - HTM / Temporal Memory  →  (B, T', D)   temporal token stream
  - Active Inference        →  (B, D_dec)   compact belief state vector
  - Symbolic Reasoning      →  (B, N_pred, D_pred)  predicate embeddings
  - General downstream      →  (B, D)       single pooled vector

Rather than hardcoding per-modality Linear projections (the old InformationBroadcast),
we introduce a typed adapter pattern with a registry, enabling:
  1. Deterministic, shape-documented broadcast outputs
  2. Mask-aligned output (consumers know which positions are valid)
  3. Feedback collection for iterative refinement
  4. Custom adapter registration for user extensions

Usage:
    from brain_ai.workspace.broadcast import (
        BroadcastConfig, BroadcastPacket, BroadcastAdapter,
        BroadcastToTemporal, BroadcastToPooled, BroadcastToSymbolic,
        BroadcastToDecision, BroadcastAdapterRegistry, FeedbackCollector,
        BroadcastModule, create_broadcast_adapters,
    )

    # Build default adapter registry
    registry = create_broadcast_adapters(
        workspace_dim=512,
        adapter_configs={
            'temporal':  {'output_dim': 512},
            'pooled':    {'output_dim': 512},
            'symbolic':  {'n_predicates': 16, 'pred_dim': 64},
            'decision':  {'decision_dim': 128},
        },
    )

    # Broadcast all
    packets = registry.broadcast_all(slots, slot_mask, gain)
    temporal_stream = packets['temporal'].content   # (B, K, D)
    belief_state    = packets['decision'].content   # (B, D_decision)

Canonical data-flow:
    workspace slots (B, K, D_ws) + slot_mask (B, K) + gain (B,)
        → BroadcastAdapter.forward()
        → BroadcastPacket(content, mask, source_slots, metadata)

NOTE: This is a TEMPLATE file.  Integration points where the actual module
internals must be wired in are marked with ``# TODO:`` comments.

References:
    Baars (1988)    "A Cognitive Theory of Consciousness"
    Dehaene (2014)  "Consciousness and the Brain"
    Mashour et al. (2020) "Conscious Processing and the Global Neuronal Workspace"
"""

from __future__ import annotations

import math
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ===========================================================================
# SECTION 1: Configuration
# ===========================================================================

@dataclass
class BroadcastConfig:
    """Configuration for the broadcast adapter subsystem.

    Attributes:
        broadcast_iterations: Number of broadcast-feedback-refine cycles.
            More iterations allow the workspace to incorporate specialist
            feedback before the final broadcast.  Default 2 balances latency
            and quality.
        broadcast_decay: Exponential decay factor applied between iterations.
            Blends the refined workspace back toward the original content so
            that feedback cannot completely overwrite the competitive result.
        use_feedback: Whether to enable feedback collection from specialist
            modules.  When False, the BroadcastModule runs a single forward
            pass with no refinement loop.
        feedback_gate_dim: Dimension of the gating network for feedback
            integration.  Defaults to workspace_dim when None.
        adapter_dropout: Dropout probability applied inside each adapter
            projection.  Regularizes the broadcast pathways during training.
    """

    broadcast_iterations: int = 2
    broadcast_decay: float = 0.9
    use_feedback: bool = True
    feedback_gate_dim: Optional[int] = None  # defaults to workspace_dim
    adapter_dropout: float = 0.1


# ===========================================================================
# SECTION 2: BroadcastPacket — typed output contract
# ===========================================================================

@dataclass
class BroadcastPacket:
    """Typed broadcast output from an adapter.

    Every adapter's ``forward`` returns one of these.  Downstream consumers
    read ``content`` and ``mask`` without knowing which adapter produced it.

    Shape rules:
        content      — (B, ..., D)     float, shape varies by adapter
        mask         — (B, ...)         bool, True = valid, aligned with content
        source_slots — (B, K, D_ws)    optional reference to original workspace slots
        metadata     — dict             adapter name, output shape description, etc.
    """

    content: Tensor
    mask: Optional[Tensor] = None
    source_slots: Optional[Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ===========================================================================
# SECTION 3: BroadcastAdapter — abstract base class
# ===========================================================================

class BroadcastAdapter(nn.Module, ABC):
    """Abstract base class for broadcast adapters.

    Each adapter converts workspace slots (B, K, D_ws) into a format suitable
    for a specific downstream consumer.  Adapters must:

    1. Produce deterministic outputs given the same inputs (no internal random
       state beyond standard dropout, which respects training/eval mode).
    2. Emit masks that are aligned with the content tensor (i.e., every
       valid position in content corresponds to True in mask).
    3. Apply gain scaling so that the neuromodulatory system can modulate
       broadcast strength.

    Subclasses must implement:
        forward(slots, slot_mask, gain) -> BroadcastPacket
        output_shape_description (property) -> str
    """

    @abstractmethod
    def forward(
        self,
        slots: Tensor,
        slot_mask: Tensor,
        gain: Tensor,
    ) -> BroadcastPacket:
        """Transform workspace slots into adapted broadcast content.

        Args:
            slots:     (B, K, D_ws)  float — workspace slot contents after
                       competition.  K = number of slots (capacity_limit),
                       D_ws = workspace_dim.
            slot_mask: (B, K)        bool  — True for valid (occupied) slots,
                       False for empty/padding slots.
            gain:      (B,)          float — per-sample broadcast gain from
                       the neuromodulatory system.  Values typically in [0, 2].

        Returns:
            BroadcastPacket with content, mask, source_slots, and metadata.
        """
        ...

    @property
    @abstractmethod
    def output_shape_description(self) -> str:
        """Human-readable description of the output shape.

        Used for logging, debugging, and documentation generation.

        Examples:
            "temporal: (B, K, D)"
            "pooled: (B, D)"
            "symbolic: (B, N_predicates, D_pred)"
        """
        ...


# ===========================================================================
# SECTION 4: BroadcastToTemporal — slots as temporal token stream
# ===========================================================================

class BroadcastToTemporal(BroadcastAdapter):
    """Convert workspace slots into a temporal token stream for HTM.

    The HTM / Temporal Memory layer expects input of shape (B, T', D) where
    T' is a sequence length.  This adapter treats each workspace slot as a
    temporal token: T' = K.

    Optionally adds sinusoidal positional encoding so that the temporal module
    can distinguish slot order even when slot contents are similar.

    Output shape:  (B, K, D_out)
    Mask shape:    (B, K)  — directly from slot_mask
    """

    def __init__(
        self,
        workspace_dim: int,
        output_dim: int,
        use_positional_encoding: bool = True,
        max_slots: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.workspace_dim = workspace_dim
        self.output_dim = output_dim
        self.use_positional_encoding = use_positional_encoding
        self.max_slots = max_slots

        # Projection: workspace_dim -> output_dim
        self.projection = nn.Sequential(
            nn.Linear(workspace_dim, workspace_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(workspace_dim, output_dim),
        )

        # Layer norm on output
        self.norm = nn.LayerNorm(output_dim)

        # Positional encoding buffer
        if use_positional_encoding:
            pe = self._build_sinusoidal_pe(max_slots, output_dim)
            self.register_buffer("positional_encoding", pe)  # (1, max_slots, D)

    @staticmethod
    def _build_sinusoidal_pe(max_len: int, dim: int) -> Tensor:
        """Build sinusoidal positional encoding table."""
        position = torch.arange(max_len).unsqueeze(1).float()  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        pe = torch.zeros(1, max_len, dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term[: dim // 2 + dim % 2])
        return pe

    def forward(
        self,
        slots: Tensor,
        slot_mask: Tensor,
        gain: Tensor,
    ) -> BroadcastPacket:
        """
        Args:
            slots:     (B, K, D_ws)
            slot_mask: (B, K)        bool
            gain:      (B,)          float

        Returns:
            BroadcastPacket with content (B, K, D_out), mask (B, K)
        """
        B, K, D = slots.shape

        # Project to output dimension
        output = self.projection(slots)  # (B, K, D_out)

        # Add positional encoding
        if self.use_positional_encoding:
            output = output + self.positional_encoding[:, :K, :]

        # Normalize
        output = self.norm(output)

        # Apply gain scaling: gain is (B,) -> (B, 1, 1)
        output = output * gain.unsqueeze(-1).unsqueeze(-1)

        # Zero out masked positions for safety
        mask_expanded = slot_mask.unsqueeze(-1).float()  # (B, K, 1)
        output = output * mask_expanded

        return BroadcastPacket(
            content=output,
            mask=slot_mask,
            source_slots=slots,
            metadata={
                "adapter": "temporal",
                "output_shape": f"(B={B}, K={K}, D={self.output_dim})",
            },
        )

    @property
    def output_shape_description(self) -> str:
        return f"temporal: (B, K, {self.output_dim})"


# ===========================================================================
# SECTION 5: BroadcastToPooled — attention-pooled single vector
# ===========================================================================

class BroadcastToPooled(BroadcastAdapter):
    """Convert workspace slots into a single pooled vector via learned attention.

    Uses a learnable query vector that attends over the workspace slots,
    producing a weighted sum.  Masked slots receive -inf attention so they
    do not contribute.

    This is the general-purpose adapter for any downstream module that
    expects a flat (B, D) representation.

    Output shape:  (B, D_out)
    Mask shape:    None (always fully valid since output is a single vector)
    """

    def __init__(
        self,
        workspace_dim: int,
        output_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.workspace_dim = workspace_dim
        self.output_dim = output_dim
        self.num_heads = num_heads

        # Learnable query vector for attention pooling
        self.query = nn.Parameter(torch.randn(1, 1, workspace_dim) * 0.02)

        # Multi-head attention: query attends over slots (keys/values)
        self.attention = nn.MultiheadAttention(
            embed_dim=workspace_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(workspace_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        slots: Tensor,
        slot_mask: Tensor,
        gain: Tensor,
    ) -> BroadcastPacket:
        """
        Args:
            slots:     (B, K, D_ws)
            slot_mask: (B, K)        bool — True = valid
            gain:      (B,)          float

        Returns:
            BroadcastPacket with content (B, D_out), mask None
        """
        B, K, D = slots.shape

        # Expand query to batch size
        query = self.query.expand(B, -1, -1)  # (B, 1, D_ws)

        # Build key_padding_mask: True = IGNORE (PyTorch convention)
        # slot_mask: True = valid, so we invert
        key_padding_mask = ~slot_mask  # (B, K), True = pad/ignore

        # Attention pooling
        pooled, attn_weights = self.attention(
            query, slots, slots,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )  # pooled: (B, 1, D_ws)

        pooled = pooled.squeeze(1)  # (B, D_ws)

        # Project to output dim
        output = self.output_proj(pooled)  # (B, D_out)

        # Apply gain scaling: gain is (B,) -> (B, 1)
        output = output * gain.unsqueeze(-1)

        return BroadcastPacket(
            content=output,
            mask=None,
            source_slots=slots,
            metadata={
                "adapter": "pooled",
                "output_shape": f"(B={B}, D={self.output_dim})",
            },
        )

    @property
    def output_shape_description(self) -> str:
        return f"pooled: (B, {self.output_dim})"


# ===========================================================================
# SECTION 6: BroadcastToSymbolic — predicate embeddings for reasoning
# ===========================================================================

class BroadcastToSymbolic(BroadcastAdapter):
    """Map workspace slots to predicate embeddings for the reasoning module.

    The symbolic / neuro-symbolic reasoning subsystem operates on predicate
    embeddings of shape (B, N_predicates, D_pred).  This adapter uses
    cross-attention where learnable predicate queries attend over workspace
    slots to extract symbolic structure.

    Each predicate query is a learnable vector that specializes in detecting
    a particular type of relation or attribute in the workspace content.

    Output shape:  (B, N_predicates, D_pred)
    Mask shape:    (B, N_predicates) — all True (predicates are always valid)
    """

    def __init__(
        self,
        workspace_dim: int,
        n_predicates: int = 16,
        pred_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.workspace_dim = workspace_dim
        self.n_predicates = n_predicates
        self.pred_dim = pred_dim
        self.num_heads = num_heads

        # Learnable predicate queries
        self.predicate_queries = nn.Parameter(
            torch.randn(1, n_predicates, workspace_dim) * 0.02
        )

        # Cross-attention: predicate queries attend over slots
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=workspace_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Project from workspace_dim to predicate dimension
        self.pred_projection = nn.Sequential(
            nn.Linear(workspace_dim, workspace_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(workspace_dim // 2, pred_dim),
        )

        # Layer norm on output
        self.norm = nn.LayerNorm(pred_dim)

    def forward(
        self,
        slots: Tensor,
        slot_mask: Tensor,
        gain: Tensor,
    ) -> BroadcastPacket:
        """
        Args:
            slots:     (B, K, D_ws)
            slot_mask: (B, K)        bool
            gain:      (B,)          float

        Returns:
            BroadcastPacket with content (B, N_predicates, D_pred),
            mask (B, N_predicates)
        """
        B, K, D = slots.shape

        # Expand predicate queries to batch size
        queries = self.predicate_queries.expand(B, -1, -1)  # (B, N_pred, D_ws)

        # Key padding mask for slots: True = IGNORE
        key_padding_mask = ~slot_mask  # (B, K)

        # Cross-attention: predicates attend over workspace slots
        attended, _ = self.cross_attention(
            queries, slots, slots,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )  # (B, N_pred, D_ws)

        # Project to predicate dimension
        pred_embeddings = self.pred_projection(attended)  # (B, N_pred, D_pred)

        # Normalize
        pred_embeddings = self.norm(pred_embeddings)

        # Apply gain scaling: gain (B,) -> (B, 1, 1)
        pred_embeddings = pred_embeddings * gain.unsqueeze(-1).unsqueeze(-1)

        # All predicates are valid (not masked)
        pred_mask = torch.ones(B, self.n_predicates, dtype=torch.bool, device=slots.device)

        return BroadcastPacket(
            content=pred_embeddings,
            mask=pred_mask,
            source_slots=slots,
            metadata={
                "adapter": "symbolic",
                "output_shape": f"(B={B}, N_pred={self.n_predicates}, D_pred={self.pred_dim})",
                "n_predicates": self.n_predicates,
                "pred_dim": self.pred_dim,
            },
        )

    @property
    def output_shape_description(self) -> str:
        return f"symbolic: (B, {self.n_predicates}, {self.pred_dim})"


# ===========================================================================
# SECTION 7: BroadcastToDecision — compact belief state for active inference
# ===========================================================================

class BroadcastToDecision(BroadcastAdapter):
    """Map workspace slots to a compact belief state for active inference.

    The active inference / decision module needs a compact state vector of
    shape (B, D_decision).  This adapter uses a two-stage process:

    Stage 1:  Attention-pool slots to a single vector (B, D_ws)
    Stage 2:  Project to decision dimension (B, D_decision)

    The two-stage design allows the intermediate representation to carry
    full workspace information before compression to the decision space.

    Output shape:  (B, D_decision)
    Mask shape:    None (single vector, always valid)
    """

    def __init__(
        self,
        workspace_dim: int,
        decision_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.workspace_dim = workspace_dim
        self.decision_dim = decision_dim
        self.num_heads = num_heads

        # Stage 1: Attention pooling over slots
        # Learnable query for pooling
        self.pool_query = nn.Parameter(
            torch.randn(1, 1, workspace_dim) * 0.02
        )

        self.pool_attention = nn.MultiheadAttention(
            embed_dim=workspace_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Stage 2: Project to decision space
        self.decision_projection = nn.Sequential(
            nn.Linear(workspace_dim, workspace_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(workspace_dim // 2, decision_dim),
            nn.LayerNorm(decision_dim),
        )

    def forward(
        self,
        slots: Tensor,
        slot_mask: Tensor,
        gain: Tensor,
    ) -> BroadcastPacket:
        """
        Args:
            slots:     (B, K, D_ws)
            slot_mask: (B, K)        bool
            gain:      (B,)          float

        Returns:
            BroadcastPacket with content (B, D_decision), mask None
        """
        B, K, D = slots.shape

        # Stage 1: Attention pool
        query = self.pool_query.expand(B, -1, -1)  # (B, 1, D_ws)
        key_padding_mask = ~slot_mask  # (B, K), True = ignore

        pooled, _ = self.pool_attention(
            query, slots, slots,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )  # (B, 1, D_ws)
        pooled = pooled.squeeze(1)  # (B, D_ws)

        # Stage 2: Project to decision dimension
        belief_state = self.decision_projection(pooled)  # (B, D_decision)

        # Apply gain scaling: gain (B,) -> (B, 1)
        belief_state = belief_state * gain.unsqueeze(-1)

        return BroadcastPacket(
            content=belief_state,
            mask=None,
            source_slots=slots,
            metadata={
                "adapter": "decision",
                "output_shape": f"(B={B}, D_decision={self.decision_dim})",
            },
        )

    @property
    def output_shape_description(self) -> str:
        return f"decision: (B, {self.decision_dim})"


# ===========================================================================
# SECTION 8: BroadcastAdapterRegistry
# ===========================================================================

class BroadcastAdapterRegistry(nn.Module):
    """Registry of named broadcast adapters.

    Manages a collection of BroadcastAdapter instances and provides batch
    broadcast to all registered adapters.  Supports dynamic registration
    of custom adapters.

    The registry is itself an nn.Module so that all adapter parameters are
    properly registered for optimizer discovery and device transfer.

    Usage:
        registry = BroadcastAdapterRegistry()
        registry.register('temporal', BroadcastToTemporal(512, 512))
        registry.register('pooled', BroadcastToPooled(512, 512))

        packets = registry.broadcast_all(slots, slot_mask, gain)
        # packets['temporal'].content -> (B, K, 512)
        # packets['pooled'].content   -> (B, 512)
    """

    def __init__(self):
        super().__init__()
        self._adapters = nn.ModuleDict()

    def register(self, name: str, adapter: BroadcastAdapter) -> None:
        """Register a named broadcast adapter.

        Args:
            name:    Unique string key for this adapter.
            adapter: BroadcastAdapter instance.

        Raises:
            TypeError:  If adapter is not a BroadcastAdapter subclass.
            ValueError: If name is already registered.
        """
        if not isinstance(adapter, BroadcastAdapter):
            raise TypeError(
                f"Expected BroadcastAdapter, got {type(adapter).__name__}. "
                f"All adapters must subclass BroadcastAdapter."
            )
        if name in self._adapters:
            logger.warning(
                f"Overwriting existing adapter '{name}' in registry."
            )
        self._adapters[name] = adapter

    def get(self, name: str) -> BroadcastAdapter:
        """Retrieve a registered adapter by name.

        Args:
            name: The adapter key.

        Returns:
            The BroadcastAdapter instance.

        Raises:
            KeyError: If name is not registered.
        """
        if name not in self._adapters:
            raise KeyError(
                f"Adapter '{name}' not found. "
                f"Available: {self.list_adapters()}"
            )
        return self._adapters[name]

    def list_adapters(self) -> List[str]:
        """Return sorted list of registered adapter names."""
        return sorted(self._adapters.keys())

    def broadcast_all(
        self,
        slots: Tensor,
        slot_mask: Tensor,
        gain: Tensor,
    ) -> Dict[str, BroadcastPacket]:
        """Broadcast workspace slots through all registered adapters.

        Args:
            slots:     (B, K, D_ws)  float
            slot_mask: (B, K)        bool
            gain:      (B,)          float

        Returns:
            Dict mapping adapter name to BroadcastPacket.
        """
        packets: Dict[str, BroadcastPacket] = {}
        for name, adapter in self._adapters.items():
            packets[name] = adapter(slots, slot_mask, gain)
        return packets

    def __len__(self) -> int:
        return len(self._adapters)

    def __contains__(self, name: str) -> bool:
        return name in self._adapters

    def __repr__(self) -> str:
        lines = [f"BroadcastAdapterRegistry(adapters={len(self._adapters)}):"]
        for name, adapter in self._adapters.items():
            lines.append(f"  {name}: {adapter.output_shape_description}")
        return "\n".join(lines)


# ===========================================================================
# SECTION 9: FeedbackCollector — gather specialist feedback
# ===========================================================================

class FeedbackCollector(nn.Module):
    """Collect and integrate feedback from specialist modules after broadcast.

    After the workspace broadcasts to specialists, each specialist may
    produce a response (feedback) indicating how relevant or useful the
    broadcast was.  The FeedbackCollector:

    1. Projects each specialist's feedback to workspace dimension.
    2. Aggregates all feedback via gated integration.
    3. Returns refined workspace content.

    This enables the bidirectional communication loop that is central to
    GWT: the workspace broadcasts, specialists respond, and the workspace
    refines its content based on those responses.

    Args:
        workspace_dim:    Dimension of the workspace representation.
        specialist_dims:  Dict mapping specialist name to its feedback
                          tensor dimension.
        gate_dim:         Dimension of the internal gating network.
                          Defaults to workspace_dim.
        dropout:          Dropout probability for feedback projections.
    """

    def __init__(
        self,
        workspace_dim: int,
        specialist_dims: Dict[str, int],
        gate_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.workspace_dim = workspace_dim
        self.specialist_dims = specialist_dims
        gate_dim = gate_dim or workspace_dim

        # Per-specialist projection: specialist_dim -> workspace_dim
        self.feedback_projections = nn.ModuleDict()
        for name, dim in specialist_dims.items():
            self.feedback_projections[name] = nn.Sequential(
                nn.Linear(dim, gate_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(gate_dim, workspace_dim),
            )

        # Gated integration of aggregated feedback with workspace content
        # Input: [workspace_content || aggregated_feedback] -> gate values
        self.feedback_gate = nn.Sequential(
            nn.Linear(workspace_dim * 2, gate_dim),
            nn.GELU(),
            nn.Linear(gate_dim, workspace_dim),
            nn.Sigmoid(),
        )

        # Post-integration normalization
        self.norm = nn.LayerNorm(workspace_dim)

    def forward(
        self,
        workspace_content: Tensor,
        specialist_states: Dict[str, Tensor],
    ) -> Tensor:
        """Integrate specialist feedback into workspace content.

        Args:
            workspace_content: (B, D_ws) current workspace representation.
            specialist_states: Dict mapping specialist name to feedback
                               tensor (B, specialist_dim).

        Returns:
            refined_content: (B, D_ws) workspace content refined by feedback.
        """
        B = workspace_content.shape[0]

        # Project each specialist's feedback and accumulate
        feedback_sum = torch.zeros_like(workspace_content)  # (B, D_ws)
        num_feedbacks = 0

        for name, state in specialist_states.items():
            if name in self.feedback_projections:
                fb = self.feedback_projections[name](state)  # (B, D_ws)
                feedback_sum = feedback_sum + fb
                num_feedbacks += 1

        if num_feedbacks == 0:
            # No feedback available, return content unchanged
            return workspace_content

        # Average feedback
        feedback_avg = feedback_sum / num_feedbacks

        # Gated integration
        combined = torch.cat([workspace_content, feedback_avg], dim=-1)  # (B, 2*D_ws)
        gate = self.feedback_gate(combined)  # (B, D_ws) in [0, 1]

        # Blend: gate=1 keeps workspace, gate=0 replaces with feedback
        refined = gate * workspace_content + (1.0 - gate) * feedback_avg

        # Normalize
        refined = self.norm(refined)

        return refined


# ===========================================================================
# SECTION 10: BroadcastModule — full broadcast orchestrator
# ===========================================================================

class BroadcastModule(nn.Module):
    """Orchestrates the complete broadcast cycle with iterative refinement.

    The BroadcastModule manages the broadcast-feedback loop:

        for each iteration:
            1. Broadcast workspace slots through all adapters
            2. (Optional) Collect feedback from specialists
            3. Refine workspace content using feedback
            4. Apply broadcast decay toward original content
            5. Re-broadcast with refined content

    The final output includes:
        - Broadcast packets from the last iteration (Dict[str, BroadcastPacket])
        - Refined workspace content after all iterations (B, D_ws)

    Components:
        adapter_registry:   BroadcastAdapterRegistry with all adapters
        feedback_collector: FeedbackCollector for specialist feedback
        config:             BroadcastConfig controlling iteration count, decay, etc.

    Args:
        workspace_dim:      Dimension of workspace representation.
        adapter_registry:   Pre-configured BroadcastAdapterRegistry.
        feedback_collector: Optional FeedbackCollector.  Required if
                            config.use_feedback is True.
        config:             BroadcastConfig instance.
    """

    def __init__(
        self,
        workspace_dim: int,
        adapter_registry: BroadcastAdapterRegistry,
        feedback_collector: Optional[FeedbackCollector] = None,
        config: Optional[BroadcastConfig] = None,
    ):
        super().__init__()

        self.workspace_dim = workspace_dim
        self.config = config or BroadcastConfig()
        self.adapter_registry = adapter_registry

        if self.config.use_feedback and feedback_collector is None:
            logger.warning(
                "BroadcastModule: use_feedback=True but no FeedbackCollector "
                "provided.  Feedback will be skipped."
            )

        self.feedback_collector = feedback_collector

        # Slot refinement layer: updates slot content based on refined workspace
        # Takes workspace refinement signal and projects it back into per-slot updates
        self.slot_refinement = nn.Sequential(
            nn.Linear(workspace_dim, workspace_dim),
            nn.GELU(),
            nn.Linear(workspace_dim, workspace_dim),
        )

        # Refinement gate: controls how much each slot changes
        self.refinement_gate = nn.Sequential(
            nn.Linear(workspace_dim * 2, workspace_dim),
            nn.Sigmoid(),
        )

        # Workspace content aggregation layer (slots -> single vector for feedback)
        self.aggregate = nn.Sequential(
            nn.Linear(workspace_dim, workspace_dim),
            nn.GELU(),
        )

    def _aggregate_slots(self, slots: Tensor, slot_mask: Tensor) -> Tensor:
        """Aggregate slots into a single workspace vector for feedback.

        Uses masked mean pooling.

        Args:
            slots:     (B, K, D_ws)
            slot_mask: (B, K)        bool

        Returns:
            workspace_content: (B, D_ws)
        """
        mask_float = slot_mask.unsqueeze(-1).float()  # (B, K, 1)
        masked_slots = slots * mask_float
        slot_sum = masked_slots.sum(dim=1)  # (B, D_ws)
        slot_count = mask_float.sum(dim=1).clamp(min=1.0)  # (B, 1)
        pooled = slot_sum / slot_count  # (B, D_ws)
        return self.aggregate(pooled)

    def _refine_slots(
        self,
        slots: Tensor,
        refined_workspace: Tensor,
        slot_mask: Tensor,
    ) -> Tensor:
        """Refine slot contents using the refined workspace signal.

        Args:
            slots:              (B, K, D_ws)  current slot contents
            refined_workspace:  (B, D_ws)     refined workspace from feedback
            slot_mask:          (B, K)         bool

        Returns:
            updated_slots:      (B, K, D_ws)
        """
        B, K, D = slots.shape

        # Broadcast workspace refinement to each slot position
        refinement_signal = self.slot_refinement(refined_workspace)  # (B, D_ws)
        refinement_expanded = refinement_signal.unsqueeze(1).expand(-1, K, -1)  # (B, K, D)

        # Gate per slot
        combined = torch.cat([slots, refinement_expanded], dim=-1)  # (B, K, 2*D)
        gate = self.refinement_gate(combined)  # (B, K, D)

        # Apply gated refinement
        updated = gate * slots + (1.0 - gate) * refinement_expanded

        # Mask out invalid slots
        mask_expanded = slot_mask.unsqueeze(-1).float()  # (B, K, 1)
        updated = updated * mask_expanded

        return updated

    def forward(
        self,
        slots: Tensor,
        slot_mask: Tensor,
        gain: Tensor,
        specialist_states: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Dict[str, BroadcastPacket], Tensor]:
        """Run the full broadcast-feedback-refine cycle.

        Args:
            slots:              (B, K, D_ws) workspace slot contents.
            slot_mask:          (B, K)       bool, True = valid slot.
            gain:               (B,)         broadcast gain.
            specialist_states:  Optional dict of specialist feedback tensors.

        Returns:
            packets:            Dict[str, BroadcastPacket] from final iteration.
            refined_workspace:  (B, D_ws) workspace content after all feedback.
        """
        original_slots = slots  # Keep reference for decay blending
        current_slots = slots
        iterations = self.config.broadcast_iterations

        # If no feedback is configured or no specialist states, single pass
        if not self.config.use_feedback or self.feedback_collector is None:
            iterations = 1

        final_packets: Dict[str, BroadcastPacket] = {}

        for iteration in range(iterations):
            # Step 1: Broadcast through all adapters
            packets = self.adapter_registry.broadcast_all(
                current_slots, slot_mask, gain
            )

            # Step 2: Collect feedback (if enabled and states provided)
            if (
                self.config.use_feedback
                and self.feedback_collector is not None
                and specialist_states is not None
                and len(specialist_states) > 0
                and iteration < iterations - 1  # Skip feedback on last iteration
            ):
                # Aggregate current slots to workspace vector
                workspace_content = self._aggregate_slots(current_slots, slot_mask)

                # Integrate specialist feedback
                refined_workspace = self.feedback_collector(
                    workspace_content, specialist_states
                )

                # Refine slot contents
                current_slots = self._refine_slots(
                    current_slots, refined_workspace, slot_mask
                )

                # Apply broadcast decay: blend refined toward original
                decay = self.config.broadcast_decay
                current_slots = (
                    decay * current_slots
                    + (1.0 - decay) * original_slots
                )

            # Keep last iteration's packets
            final_packets = packets

        # Final workspace content (aggregated from final slots)
        refined_workspace = self._aggregate_slots(current_slots, slot_mask)

        return final_packets, refined_workspace


# ===========================================================================
# SECTION 11: Factory function
# ===========================================================================

# Default adapter type mapping
_ADAPTER_BUILDERS: Dict[str, Type[BroadcastAdapter]] = {
    "temporal": BroadcastToTemporal,
    "pooled": BroadcastToPooled,
    "symbolic": BroadcastToSymbolic,
    "decision": BroadcastToDecision,
}


def create_broadcast_adapters(
    workspace_dim: int,
    adapter_configs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> BroadcastAdapterRegistry:
    """Factory function to create a BroadcastAdapterRegistry with default adapters.

    Builds the four standard adapters (temporal, pooled, symbolic, decision)
    with configurable dimensions.  Custom adapters can be added to the returned
    registry via ``registry.register()``.

    Args:
        workspace_dim:   Dimension of the workspace representation (D_ws).
        adapter_configs: Optional dict mapping adapter name to its kwargs.
                         Each entry overrides defaults for that adapter.
                         Unrecognized names are silently skipped (use
                         registry.register() for custom adapters).

    Returns:
        BroadcastAdapterRegistry with all configured adapters.

    Example:
        registry = create_broadcast_adapters(
            workspace_dim=512,
            adapter_configs={
                'temporal':  {'output_dim': 512},
                'pooled':    {'output_dim': 512},
                'symbolic':  {'n_predicates': 16, 'pred_dim': 64},
                'decision':  {'decision_dim': 128},
            },
        )
    """
    adapter_configs = adapter_configs or {}
    registry = BroadcastAdapterRegistry()

    # --- Temporal adapter ---
    temporal_cfg = adapter_configs.get("temporal", {})
    temporal_adapter = BroadcastToTemporal(
        workspace_dim=workspace_dim,
        output_dim=temporal_cfg.get("output_dim", workspace_dim),
        use_positional_encoding=temporal_cfg.get("use_positional_encoding", True),
        max_slots=temporal_cfg.get("max_slots", 64),
        dropout=temporal_cfg.get("dropout", 0.1),
    )
    registry.register("temporal", temporal_adapter)

    # --- Pooled adapter ---
    pooled_cfg = adapter_configs.get("pooled", {})
    pooled_adapter = BroadcastToPooled(
        workspace_dim=workspace_dim,
        output_dim=pooled_cfg.get("output_dim", workspace_dim),
        num_heads=pooled_cfg.get("num_heads", 4),
        dropout=pooled_cfg.get("dropout", 0.1),
    )
    registry.register("pooled", pooled_adapter)

    # --- Symbolic adapter ---
    symbolic_cfg = adapter_configs.get("symbolic", {})
    symbolic_adapter = BroadcastToSymbolic(
        workspace_dim=workspace_dim,
        n_predicates=symbolic_cfg.get("n_predicates", 16),
        pred_dim=symbolic_cfg.get("pred_dim", 64),
        num_heads=symbolic_cfg.get("num_heads", 4),
        dropout=symbolic_cfg.get("dropout", 0.1),
    )
    registry.register("symbolic", symbolic_adapter)

    # --- Decision adapter ---
    decision_cfg = adapter_configs.get("decision", {})
    decision_adapter = BroadcastToDecision(
        workspace_dim=workspace_dim,
        decision_dim=decision_cfg.get("decision_dim", 128),
        num_heads=decision_cfg.get("num_heads", 4),
        dropout=decision_cfg.get("dropout", 0.1),
    )
    registry.register("decision", decision_adapter)

    logger.info(
        f"Created BroadcastAdapterRegistry with {len(registry)} adapters: "
        f"{registry.list_adapters()}"
    )

    return registry


def create_broadcast_module(
    workspace_dim: int,
    adapter_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    specialist_dims: Optional[Dict[str, int]] = None,
    config: Optional[BroadcastConfig] = None,
) -> BroadcastModule:
    """Factory function to create a complete BroadcastModule.

    Builds the adapter registry, feedback collector, and wraps them in a
    BroadcastModule that manages the iterative broadcast-feedback cycle.

    Args:
        workspace_dim:    Dimension of workspace representation.
        adapter_configs:  Adapter-specific configuration (see
                          create_broadcast_adapters).
        specialist_dims:  Dict mapping specialist name to feedback tensor
                          dimension.  Required for feedback collection.
        config:           BroadcastConfig instance.  Uses defaults if None.

    Returns:
        Fully configured BroadcastModule.
    """
    config = config or BroadcastConfig()

    # Build adapter registry
    registry = create_broadcast_adapters(workspace_dim, adapter_configs)

    # Build feedback collector if feedback is enabled and specialist dims given
    feedback_collector = None
    if config.use_feedback and specialist_dims:
        feedback_collector = FeedbackCollector(
            workspace_dim=workspace_dim,
            specialist_dims=specialist_dims,
            gate_dim=config.feedback_gate_dim,
            dropout=config.adapter_dropout,
        )

    return BroadcastModule(
        workspace_dim=workspace_dim,
        adapter_registry=registry,
        feedback_collector=feedback_collector,
        config=config,
    )


# ===========================================================================
# SECTION 12: __all__ and public API
# ===========================================================================

__all__ = [
    # Config
    "BroadcastConfig",
    # Data contracts
    "BroadcastPacket",
    # Abstract base
    "BroadcastAdapter",
    # Concrete adapters
    "BroadcastToTemporal",
    "BroadcastToPooled",
    "BroadcastToSymbolic",
    "BroadcastToDecision",
    # Registry
    "BroadcastAdapterRegistry",
    # Feedback
    "FeedbackCollector",
    # Orchestrator
    "BroadcastModule",
    # Factories
    "create_broadcast_adapters",
    "create_broadcast_module",
]


# ===========================================================================
# SECTION 13: Self-tests
# ===========================================================================

def _run_self_tests() -> None:
    """Run ~25 self-tests validating shapes, masks, gains, and registries.

    Invoke via:
        python -m brain_ai.workspace.broadcast
    or:
        python broadcast_adapters_template.py
    """
    import traceback
    import sys

    passed = 0
    failed = 0
    errors: List[str] = []

    def _test(name: str, fn: Callable[[], None]) -> None:
        nonlocal passed, failed
        try:
            fn()
            passed += 1
            print(f"  PASS: {name}")
        except Exception as e:
            failed += 1
            tb = traceback.format_exc()
            errors.append(f"  FAIL: {name}\n{tb}")
            print(f"  FAIL: {name} -- {e}")

    print("=" * 70)
    print("BroadcastAdapters Self-Tests")
    print("=" * 70)

    # ---- Shared test fixtures ----
    torch.manual_seed(42)
    B, K, D_ws = 4, 7, 128
    D_out = 128
    D_pred = 64
    N_pred = 16
    D_decision = 32

    slots = torch.randn(B, K, D_ws)
    slot_mask = torch.ones(B, K, dtype=torch.bool)
    slot_mask[0, -2:] = False  # Mask last 2 slots in first sample
    slot_mask[2, -1:] = False  # Mask last slot in third sample
    gain = torch.ones(B)
    gain[1] = 0.5  # Half gain for second sample

    # ================================================================
    # Test 1: BroadcastToTemporal output shape
    # ================================================================
    def test_temporal_shape():
        adapter = BroadcastToTemporal(D_ws, D_out)
        packet = adapter(slots, slot_mask, gain)
        assert packet.content.shape == (B, K, D_out), \
            f"Expected (B, K, D_out)={(B, K, D_out)}, got {packet.content.shape}"

    _test("BroadcastToTemporal output shape (B,K,D)", test_temporal_shape)

    # ================================================================
    # Test 2: BroadcastToTemporal mask alignment
    # ================================================================
    def test_temporal_mask():
        adapter = BroadcastToTemporal(D_ws, D_out)
        packet = adapter(slots, slot_mask, gain)
        assert packet.mask is not None, "Temporal mask should not be None"
        assert packet.mask.shape == (B, K), \
            f"Expected mask shape (B, K)={(B, K)}, got {packet.mask.shape}"
        assert torch.equal(packet.mask, slot_mask), \
            "Temporal mask should equal slot_mask"

    _test("BroadcastToTemporal mask alignment", test_temporal_mask)

    # ================================================================
    # Test 3: BroadcastToTemporal masked positions are zero
    # ================================================================
    def test_temporal_masked_zero():
        adapter = BroadcastToTemporal(D_ws, D_out)
        packet = adapter(slots, slot_mask, gain)
        # First sample, last 2 slots should be zero
        assert torch.allclose(
            packet.content[0, -2:, :],
            torch.zeros(2, D_out),
            atol=1e-6,
        ), "Masked temporal positions should be zero"

    _test("BroadcastToTemporal masked positions zeroed", test_temporal_masked_zero)

    # ================================================================
    # Test 4: BroadcastToTemporal positional encoding applied
    # ================================================================
    def test_temporal_positional_encoding():
        adapter_pe = BroadcastToTemporal(D_ws, D_out, use_positional_encoding=True)
        adapter_no_pe = BroadcastToTemporal(D_ws, D_out, use_positional_encoding=False)
        # Different adapters should give different results (non-trivially)
        # Just check the PE flag works without crash
        p1 = adapter_pe(slots, slot_mask, gain)
        p2 = adapter_no_pe(slots, slot_mask, gain)
        assert p1.content.shape == p2.content.shape

    _test("BroadcastToTemporal positional encoding toggle", test_temporal_positional_encoding)

    # ================================================================
    # Test 5: BroadcastToPooled output shape
    # ================================================================
    def test_pooled_shape():
        adapter = BroadcastToPooled(D_ws, D_out)
        packet = adapter(slots, slot_mask, gain)
        assert packet.content.shape == (B, D_out), \
            f"Expected (B, D_out)={(B, D_out)}, got {packet.content.shape}"

    _test("BroadcastToPooled output shape (B,D)", test_pooled_shape)

    # ================================================================
    # Test 6: BroadcastToPooled mask is None
    # ================================================================
    def test_pooled_mask_none():
        adapter = BroadcastToPooled(D_ws, D_out)
        packet = adapter(slots, slot_mask, gain)
        assert packet.mask is None, "Pooled mask should be None (always valid)"

    _test("BroadcastToPooled mask is None", test_pooled_mask_none)

    # ================================================================
    # Test 7: BroadcastToPooled mask-aware attention
    # ================================================================
    def test_pooled_mask_aware():
        torch.manual_seed(42)
        adapter = BroadcastToPooled(D_ws, D_out)
        adapter.train(False)
        # All slots valid
        full_mask = torch.ones(B, K, dtype=torch.bool)
        p_full = adapter(slots, full_mask, torch.ones(B))
        # Only first slot valid
        sparse_mask = torch.zeros(B, K, dtype=torch.bool)
        sparse_mask[:, 0] = True
        p_sparse = adapter(slots, sparse_mask, torch.ones(B))
        # Outputs should differ when different slots are visible
        assert not torch.allclose(p_full.content, p_sparse.content, atol=1e-4), \
            "Pooled output should differ when mask changes"

    _test("BroadcastToPooled mask-aware attention", test_pooled_mask_aware)

    # ================================================================
    # Test 8: BroadcastToSymbolic output shape
    # ================================================================
    def test_symbolic_shape():
        adapter = BroadcastToSymbolic(D_ws, n_predicates=N_pred, pred_dim=D_pred)
        packet = adapter(slots, slot_mask, gain)
        assert packet.content.shape == (B, N_pred, D_pred), \
            f"Expected (B, N_pred, D_pred)={(B, N_pred, D_pred)}, got {packet.content.shape}"

    _test("BroadcastToSymbolic output shape (B,N_pred,D_pred)", test_symbolic_shape)

    # ================================================================
    # Test 9: BroadcastToSymbolic mask shape
    # ================================================================
    def test_symbolic_mask():
        adapter = BroadcastToSymbolic(D_ws, n_predicates=N_pred, pred_dim=D_pred)
        packet = adapter(slots, slot_mask, gain)
        assert packet.mask is not None
        assert packet.mask.shape == (B, N_pred), \
            f"Expected mask (B, N_pred)={(B, N_pred)}, got {packet.mask.shape}"
        assert packet.mask.all(), "All predicate positions should be valid"

    _test("BroadcastToSymbolic mask shape and validity", test_symbolic_mask)

    # ================================================================
    # Test 10: BroadcastToDecision output shape
    # ================================================================
    def test_decision_shape():
        adapter = BroadcastToDecision(D_ws, decision_dim=D_decision)
        packet = adapter(slots, slot_mask, gain)
        assert packet.content.shape == (B, D_decision), \
            f"Expected (B, D_decision)={(B, D_decision)}, got {packet.content.shape}"

    _test("BroadcastToDecision output shape (B,D_decision)", test_decision_shape)

    # ================================================================
    # Test 11: BroadcastToDecision mask is None
    # ================================================================
    def test_decision_mask_none():
        adapter = BroadcastToDecision(D_ws, decision_dim=D_decision)
        packet = adapter(slots, slot_mask, gain)
        assert packet.mask is None, "Decision mask should be None"

    _test("BroadcastToDecision mask is None", test_decision_mask_none)

    # ================================================================
    # Test 12: Gain scaling -- temporal
    # ================================================================
    def test_gain_scaling_temporal():
        torch.manual_seed(42)
        adapter = BroadcastToTemporal(D_ws, D_out)
        adapter.train(False)
        unit_gain = torch.ones(B)
        half_gain = torch.ones(B) * 0.5
        p_unit = adapter(slots, slot_mask, unit_gain)
        p_half = adapter(slots, slot_mask, half_gain)
        # Half gain should produce half the magnitude
        ratio = p_half.content / (p_unit.content + 1e-10)
        valid = slot_mask.unsqueeze(-1).expand_as(ratio)
        valid_ratios = ratio[valid]
        # Filter out near-zero values to avoid division artifacts
        significant = valid_ratios[p_unit.content[valid].abs() > 1e-4]
        if len(significant) > 0:
            assert torch.allclose(significant, torch.full_like(significant, 0.5), atol=1e-3), \
                "Gain=0.5 should halve the output"

    _test("Gain scaling applied correctly (temporal)", test_gain_scaling_temporal)

    # ================================================================
    # Test 13: Gain scaling -- pooled
    # ================================================================
    def test_gain_scaling_pooled():
        torch.manual_seed(42)
        adapter = BroadcastToPooled(D_ws, D_out)
        adapter.train(False)
        unit_gain = torch.ones(B)
        double_gain = torch.ones(B) * 2.0
        p_unit = adapter(slots, slot_mask, unit_gain)
        p_double = adapter(slots, slot_mask, double_gain)
        ratio = p_double.content / (p_unit.content + 1e-10)
        significant = ratio[p_unit.content.abs() > 1e-4]
        if len(significant) > 0:
            assert torch.allclose(significant, torch.full_like(significant, 2.0), atol=1e-3), \
                "Gain=2.0 should double the output"

    _test("Gain scaling applied correctly (pooled)", test_gain_scaling_pooled)

    # ================================================================
    # Test 14: Gain scaling -- symbolic
    # ================================================================
    def test_gain_scaling_symbolic():
        torch.manual_seed(42)
        adapter = BroadcastToSymbolic(D_ws, n_predicates=N_pred, pred_dim=D_pred)
        adapter.train(False)
        unit_gain = torch.ones(B)
        half_gain = torch.ones(B) * 0.5
        p_unit = adapter(slots, slot_mask, unit_gain)
        p_half = adapter(slots, slot_mask, half_gain)
        ratio = p_half.content / (p_unit.content + 1e-10)
        significant = ratio[p_unit.content.abs() > 1e-4]
        if len(significant) > 0:
            assert torch.allclose(significant, torch.full_like(significant, 0.5), atol=1e-3), \
                "Gain=0.5 should halve symbolic output"

    _test("Gain scaling applied correctly (symbolic)", test_gain_scaling_symbolic)

    # ================================================================
    # Test 15: Gain scaling -- decision
    # ================================================================
    def test_gain_scaling_decision():
        torch.manual_seed(42)
        adapter = BroadcastToDecision(D_ws, decision_dim=D_decision)
        adapter.train(False)
        unit_gain = torch.ones(B)
        half_gain = torch.ones(B) * 0.5
        p_unit = adapter(slots, slot_mask, unit_gain)
        p_half = adapter(slots, slot_mask, half_gain)
        ratio = p_half.content / (p_unit.content + 1e-10)
        significant = ratio[p_unit.content.abs() > 1e-4]
        if len(significant) > 0:
            assert torch.allclose(significant, torch.full_like(significant, 0.5), atol=1e-3), \
                "Gain=0.5 should halve decision output"

    _test("Gain scaling applied correctly (decision)", test_gain_scaling_decision)

    # ================================================================
    # Test 16: Registry register / get / list
    # ================================================================
    def test_registry_register_get_list():
        reg = BroadcastAdapterRegistry()
        a1 = BroadcastToTemporal(D_ws, D_out)
        a2 = BroadcastToPooled(D_ws, D_out)
        reg.register("temporal", a1)
        reg.register("pooled", a2)
        assert reg.get("temporal") is a1
        assert reg.get("pooled") is a2
        assert reg.list_adapters() == ["pooled", "temporal"]  # sorted
        assert len(reg) == 2
        assert "temporal" in reg
        assert "symbolic" not in reg

    _test("Registry register / get / list", test_registry_register_get_list)

    # ================================================================
    # Test 17: Registry get missing key raises KeyError
    # ================================================================
    def test_registry_missing_key():
        reg = BroadcastAdapterRegistry()
        try:
            reg.get("nonexistent")
            assert False, "Should have raised KeyError"
        except KeyError:
            pass

    _test("Registry get missing key raises KeyError", test_registry_missing_key)

    # ================================================================
    # Test 18: Registry register type check
    # ================================================================
    def test_registry_type_check():
        reg = BroadcastAdapterRegistry()
        try:
            reg.register("bad", nn.Linear(10, 10))  # type: ignore
            assert False, "Should have raised TypeError"
        except TypeError:
            pass

    _test("Registry register type check", test_registry_type_check)

    # ================================================================
    # Test 19: Registry broadcast_all
    # ================================================================
    def test_registry_broadcast_all():
        reg = create_broadcast_adapters(
            D_ws,
            adapter_configs={
                "temporal": {"output_dim": D_out},
                "pooled": {"output_dim": D_out},
                "symbolic": {"n_predicates": N_pred, "pred_dim": D_pred},
                "decision": {"decision_dim": D_decision},
            },
        )
        packets = reg.broadcast_all(slots, slot_mask, gain)
        assert set(packets.keys()) == {"temporal", "pooled", "symbolic", "decision"}
        assert packets["temporal"].content.shape == (B, K, D_out)
        assert packets["pooled"].content.shape == (B, D_out)
        assert packets["symbolic"].content.shape == (B, N_pred, D_pred)
        assert packets["decision"].content.shape == (B, D_decision)

    _test("Registry broadcast_all", test_registry_broadcast_all)

    # ================================================================
    # Test 20: Custom adapter registration
    # ================================================================
    def test_custom_adapter():
        class CustomAdapter(BroadcastAdapter):
            def __init__(self, ws_dim: int, out_dim: int):
                super().__init__()
                self.proj = nn.Linear(ws_dim, out_dim)
                self.out_dim = out_dim

            def forward(self, slots, slot_mask, gain):
                pooled = slots.mean(dim=1)
                output = self.proj(pooled) * gain.unsqueeze(-1)
                return BroadcastPacket(
                    content=output,
                    mask=None,
                    metadata={"adapter": "custom"},
                )

            @property
            def output_shape_description(self):
                return f"custom: (B, {self.out_dim})"

        reg = BroadcastAdapterRegistry()
        custom = CustomAdapter(D_ws, 42)
        reg.register("my_custom", custom)
        packet = reg.get("my_custom")(slots, slot_mask, gain)
        assert packet.content.shape == (B, 42)
        assert packet.metadata["adapter"] == "custom"

    _test("Custom adapter registration", test_custom_adapter)

    # ================================================================
    # Test 21: FeedbackCollector with multiple specialists
    # ================================================================
    def test_feedback_collector():
        specialist_dims = {"htm": 64, "reasoning": 128, "decision": 32}
        fc = FeedbackCollector(D_ws, specialist_dims)
        workspace_content = torch.randn(B, D_ws)
        specialist_states = {
            "htm": torch.randn(B, 64),
            "reasoning": torch.randn(B, 128),
            "decision": torch.randn(B, 32),
        }
        refined = fc(workspace_content, specialist_states)
        assert refined.shape == (B, D_ws), \
            f"Expected (B, D_ws)={(B, D_ws)}, got {refined.shape}"

    _test("FeedbackCollector with multiple specialists", test_feedback_collector)

    # ================================================================
    # Test 22: FeedbackCollector with no matching specialists
    # ================================================================
    def test_feedback_no_match():
        specialist_dims = {"htm": 64}
        fc = FeedbackCollector(D_ws, specialist_dims)
        workspace_content = torch.randn(B, D_ws)
        # Provide feedback from unknown specialist
        refined = fc(workspace_content, {"unknown_module": torch.randn(B, 32)})
        assert torch.equal(refined, workspace_content), \
            "No matching feedback should return workspace unchanged"

    _test("FeedbackCollector with no matching specialists", test_feedback_no_match)

    # ================================================================
    # Test 23: BroadcastModule iterative refinement
    # ================================================================
    def test_broadcast_module_iterative():
        config = BroadcastConfig(
            broadcast_iterations=3,
            broadcast_decay=0.9,
            use_feedback=True,
        )
        specialist_dims = {"htm": 64, "reasoning": 128}
        module = create_broadcast_module(
            workspace_dim=D_ws,
            adapter_configs={
                "temporal": {"output_dim": D_out},
                "pooled": {"output_dim": D_out},
                "symbolic": {"n_predicates": N_pred, "pred_dim": D_pred},
                "decision": {"decision_dim": D_decision},
            },
            specialist_dims=specialist_dims,
            config=config,
        )
        specialist_states = {
            "htm": torch.randn(B, 64),
            "reasoning": torch.randn(B, 128),
        }
        packets, refined_ws = module(slots, slot_mask, gain, specialist_states)
        assert isinstance(packets, dict)
        assert refined_ws.shape == (B, D_ws)
        assert "temporal" in packets
        assert "pooled" in packets

    _test("BroadcastModule iterative refinement", test_broadcast_module_iterative)

    # ================================================================
    # Test 24: BroadcastModule without feedback
    # ================================================================
    def test_broadcast_module_no_feedback():
        config = BroadcastConfig(
            broadcast_iterations=2,
            use_feedback=False,
        )
        module = create_broadcast_module(
            workspace_dim=D_ws,
            config=config,
        )
        packets, refined_ws = module(slots, slot_mask, gain)
        assert isinstance(packets, dict)
        assert refined_ws.shape == (B, D_ws)

    _test("BroadcastModule without feedback", test_broadcast_module_no_feedback)

    # ================================================================
    # Test 25: Deterministic outputs when not training
    # ================================================================
    def test_deterministic_not_training():
        torch.manual_seed(42)
        adapter = BroadcastToTemporal(D_ws, D_out)
        adapter.train(False)
        p1 = adapter(slots, slot_mask, gain)
        p2 = adapter(slots, slot_mask, gain)
        assert torch.allclose(p1.content, p2.content, atol=1e-6), \
            "Non-training mode should produce deterministic outputs"

    _test("Deterministic outputs when not training", test_deterministic_not_training)

    # ================================================================
    # Test 26: BroadcastPacket has all fields
    # ================================================================
    def test_packet_fields():
        adapter = BroadcastToTemporal(D_ws, D_out)
        packet = adapter(slots, slot_mask, gain)
        assert hasattr(packet, "content")
        assert hasattr(packet, "mask")
        assert hasattr(packet, "source_slots")
        assert hasattr(packet, "metadata")
        assert packet.source_slots is not None
        assert torch.equal(packet.source_slots, slots)
        assert isinstance(packet.metadata, dict)
        assert "adapter" in packet.metadata

    _test("BroadcastPacket has all fields", test_packet_fields)

    # ================================================================
    # Test 27: All adapters produce aligned masks
    # ================================================================
    def test_aligned_masks():
        registry = create_broadcast_adapters(
            D_ws,
            adapter_configs={
                "temporal": {"output_dim": D_out},
                "pooled": {"output_dim": D_out},
                "symbolic": {"n_predicates": N_pred, "pred_dim": D_pred},
                "decision": {"decision_dim": D_decision},
            },
        )
        packets = registry.broadcast_all(slots, slot_mask, gain)

        # Temporal: mask aligned with content dim 1
        t_pkt = packets["temporal"]
        assert t_pkt.mask.shape[0] == t_pkt.content.shape[0]
        assert t_pkt.mask.shape[1] == t_pkt.content.shape[1]

        # Pooled: mask is None (valid)
        assert packets["pooled"].mask is None

        # Symbolic: mask aligned with content dim 1
        s_pkt = packets["symbolic"]
        assert s_pkt.mask.shape[0] == s_pkt.content.shape[0]
        assert s_pkt.mask.shape[1] == s_pkt.content.shape[1]

        # Decision: mask is None
        assert packets["decision"].mask is None

    _test("All adapters produce aligned masks", test_aligned_masks)

    # ================================================================
    # Test 28: output_shape_description property
    # ================================================================
    def test_output_shape_descriptions():
        t = BroadcastToTemporal(D_ws, D_out)
        assert "temporal" in t.output_shape_description
        p = BroadcastToPooled(D_ws, D_out)
        assert "pooled" in p.output_shape_description
        s = BroadcastToSymbolic(D_ws, n_predicates=N_pred, pred_dim=D_pred)
        assert "symbolic" in s.output_shape_description
        d = BroadcastToDecision(D_ws, decision_dim=D_decision)
        assert "decision" in d.output_shape_description

    _test("output_shape_description property", test_output_shape_descriptions)

    # ================================================================
    # Test 29: BroadcastConfig defaults
    # ================================================================
    def test_broadcast_config_defaults():
        cfg = BroadcastConfig()
        assert cfg.broadcast_iterations == 2
        assert cfg.broadcast_decay == 0.9
        assert cfg.use_feedback is True
        assert cfg.feedback_gate_dim is None
        assert cfg.adapter_dropout == 0.1

    _test("BroadcastConfig defaults", test_broadcast_config_defaults)

    # ================================================================
    # Test 30: create_broadcast_adapters factory default configs
    # ================================================================
    def test_factory_defaults():
        registry = create_broadcast_adapters(D_ws)
        adapters = registry.list_adapters()
        assert "temporal" in adapters
        assert "pooled" in adapters
        assert "symbolic" in adapters
        assert "decision" in adapters
        assert len(adapters) == 4

    _test("create_broadcast_adapters factory defaults", test_factory_defaults)

    # ================================================================
    # Summary
    # ================================================================
    print()
    print("=" * 70)
    total = passed + failed
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if errors:
        print("\nFailed tests:")
        for e in errors:
            print(e)
    print("=" * 70)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    _run_self_tests()
